"""Chat orchestration worker for LLM + MCP tool execution."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PySide6.QtCore import QObject, Signal, Slot

from chat_models import ChatMessage, ChatSessionState
from llm_adapter import LiteLLMClient
from mcp_bridge import MCPBridge

logger = logging.getLogger(__name__)


def _coerce_json_strings(args: Dict[str, Any]) -> Dict[str, Any]:
    """Parse any string values that look like JSON arrays or objects.

    Weak LLMs occasionally double-encode array arguments as strings
    (e.g. region_ids='[0,1]' instead of region_ids=[0,1]).
    """
    result = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith(("[", "{")):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        else:
            result[k] = v
    return result


@dataclass
class ChatAgentConfig:
    """Configuration for chat worker behavior."""

    model: str
    api_key: str
    llm_timeout_s: float = 60.0
    tool_timeout_s: float = 30.0
    context_messages: int = 20


class ChatAgentWorker(QObject):
    """QObject worker that runs in a dedicated QThread."""

    response_started = Signal(str)
    response_delta = Signal(str, str)
    tool_started = Signal(str, str, str)
    tool_finished = Signal(str, str, object, object)
    response_finished = Signal(str, str, object, int)
    error_emitted = Signal(str, str, str, str)
    state_changed = Signal(str)

    def __init__(self, config: ChatAgentConfig) -> None:
        super().__init__()
        self._config = config
        self._session = ChatSessionState()
        self._cancel_requested = False
        self._allowed_tools: Set[str] = {
            "list_data",
            "inspect_h5",
            "describe_slide",
            "compare_models",
            "rank_models_by_separability",
            "compute_elbow_analysis",
            "list_labeled_regions",
            "compute_region_stats",
            "find_similar_patches",
            "compute_cluster_stats",
            "get_boundary_patches",
            "get_pca_info",
            "compare_selected_clusters",
            "atlas_cluster_representation",
            "rank_models_by_selected_cluster_separability",
            "rank_models_by_labeled_region_separability",
        }
        self._cached_tools: Optional[List[Dict[str, Any]]] = None
        self._mcp = MCPBridge()
        self._llm = LiteLLMClient(
            model=config.model,
            api_key=config.api_key,
            timeout_s=config.llm_timeout_s,
        )

    @Slot(str, object)
    def submit_user_message(self, text: str, app_context: Dict[str, Any]) -> None:
        """Handle one full request lifecycle."""
        if self._session.running:
            self.error_emitted.emit("", "busy", "Agent is already running", "")
            return

        self._cancel_requested = False
        self._session.running = True
        request_id = str(uuid.uuid4())
        self._session.current_request_id = request_id
        self.state_changed.emit("running")

        started_at = time.time()
        final_text = ""
        usage: Dict[str, Any] = {}

        try:
            root_dir = (app_context or {}).get("root_dir") or ""
            self._session.messages.append(
                ChatMessage(role="user", content=text, message_id=str(uuid.uuid4()))
            )

            tools = self._mcp_tools_to_openai_tools()
            messages = self._build_prompt_messages(root_dir=root_dir)

            initial_response = self._llm.complete_with_tools(
                messages=messages,
                tools=tools,
                timeout_s=self._config.llm_timeout_s,
            )
            tool_calls = self._llm.normalize_tool_calls(initial_response)

            if self._cancel_requested:
                return

            if tool_calls:
                # OpenAI/LiteLLM requires that any `role: tool` messages are a response
                # to a preceding assistant message that contains `tool_calls`.
                messages.append(self._build_assistant_tool_calls_message(tool_calls))

                tool_messages = self._execute_tool_calls(
                    tool_calls=tool_calls,
                    root_dir=root_dir,
                )
                messages.extend(tool_messages)
                final_text, usage = self._stream_final_response(
                    request_id=request_id,
                    messages=messages,
                )
            else:
                final_text = self._extract_message_content(initial_response)
                usage = self._llm.normalize_usage(initial_response)
                self.response_started.emit(request_id)
                for character in final_text:
                    if self._cancel_requested:
                        break
                    self.response_delta.emit(request_id, character)

            if self._cancel_requested:
                return

            self._session.messages.append(
                ChatMessage(
                    role="assistant",
                    content=final_text,
                    message_id=str(uuid.uuid4()),
                )
            )
            latency_ms = int((time.time() - started_at) * 1000)
            self.response_finished.emit(request_id, final_text, usage, latency_ms)
        except Exception as exc:
            self.error_emitted.emit(
                request_id,
                "runtime_error",
                "Chat request failed",
                str(exc),
            )
        finally:
            self._session.running = False
            self._session.current_request_id = None
            self.state_changed.emit("idle")

    @Slot()
    def cancel_current(self) -> None:
        """Set cancellation flag checked between phases."""
        if not self._session.running:
            return
        self._cancel_requested = True
        self.state_changed.emit("cancelling")

    @Slot()
    def shutdown(self) -> None:
        """Release external resources."""
        self._cancel_requested = True
        self._cached_tools = None
        try:
            self._mcp.stop()
        except Exception:
            pass
        self.state_changed.emit("idle")

    def _build_prompt_messages(self, root_dir: str) -> List[Dict[str, Any]]:
        """Build model prompt with bounded historical context."""
        system_content = (
            "You are a pathology-aware assistant inside FoundationDetector, "
            "a tool for visualizing patch-level features from whole-slide images (WSI). "
            "Use tools proactively to answer questions about available data. "
            "Allowed data boundary: read-only within selected root_dir only. "
            f"Current root_dir: {root_dir or 'not set'}.\n\n"
            "Available tools:\n"
            "Tier 1 — filesystem tools (require root_dir, read H5 files directly):\n"
            "- list_data: enumerate all slides/models/magnifications/patch_sizes in root_dir\n"
            "- inspect_h5: raw HDF5 dataset shapes and attributes for a specific file path\n"
            "- describe_slide: enriched summary (patch count, feature dim, coord range, patch size) "
            "for a specific slide/model/mag/patch_size — use list_data first to find valid values\n"
            "- compare_models: compare K-means clustering quality between two FOUNDATION MODELS "
            "(silhouette, Davies-Bouldin, ARI). model_a and model_b are foundation model names "
            "(e.g. 'UNI', 'CONCH') — not magnifications or patch sizes.\n"
            "- rank_models_by_separability: rank all available foundation models for a slide by "
            "cluster separability score — re-clusters each model's features from scratch\n"
            "- compute_elbow_analysis: suggest optimal cluster count k using the elbow method; "
            "returns inertia curve and recommended k\n\n"
            "Tier 2 — live GUI state tools (no arguments needed beyond optional ids):\n"
            "IMPORTANT — 'model' always means the foundation model encoder name (e.g. UNI, CONCH, "
            "ViT-S/16). It never refers to magnification, patch size, or K-means. Use the right "
            "ranking tool for the user's context:\n"
            "  • labeled regions exist → rank_models_by_labeled_region_separability\n"
            "  • K-means clusters are highlighted → rank_models_by_selected_cluster_separability\n"
            "  • no current labels/selections → rank_models_by_separability (re-clusters fresh)\n\n"
            "- list_labeled_regions: list all user-created annotations with name, color, patch "
            "count, source mode, kmeans_cluster, and bounding box; call this first when asked "
            "about annotations\n"
            "- compute_region_stats(region_id, metrics): compute only the requested stats for a "
            "labeled region. metrics list can contain any of: spread, area_covered, area_bbox, "
            "area_hull, homogeneity, global_distance, pca_extent, top_dims, area_mm2, width_mm\n"
            "- find_similar_patches(region_id, top_k, metric): find patches most similar to a "
            "region's centroid by cosine or euclidean distance\n"
            "- compute_cluster_stats(cluster_id, metrics): compute only the requested stats for a "
            "K-means cluster. metrics list can contain: spread, area_covered, area_bbox, "
            "area_hull, separation, discriminating_dims, patch_count, area_mm2, width_mm\n"
            "- get_boundary_patches(top_k): patches sitting closest to cluster decision boundaries "
            "(smallest gap between assigned and nearest-other centroid distance)\n"
            "- get_pca_info: PCA embedding metadata — explained variance ratio, cumulative "
            "variance, and a human-readable note\n"
            "- compare_selected_clusters(metrics): run stats for ALL currently selected clusters "
            "in one call and get a ranked comparison. metrics: patch_count, area_mm2, width_mm, "
            "area_bbox, area_hull, spread, separation, discriminating_dims, homogeneity, "
            "pca_extent. Use for questions like: 'which is largest/most unique/widest among my "
            "selected clusters?', 'which selected cluster is most compact?'\n"
            "- atlas_cluster_representation(): cross-slide atlas distribution — which clusters "
            "appear in most slides vs. only one slide. Optional params: normalize (add fraction "
            "of each slide's tissue), include_entropy (Shannon entropy), selected_only (filter to "
            "selected clusters). Requires atlas to be built in GUI.\n"
            "- rank_models_by_selected_cluster_separability(mag, patch_size): rank all foundation "
            "models by how well their features separate the currently highlighted K-means clusters "
            "(uses existing cluster labels as ground truth, not re-clustering). mag and patch_size "
            "are optional — defaults to currently loaded values. Optional: cluster_ids to override "
            "selection, metric ('euclidean' or 'cosine').\n"
            "- rank_models_by_labeled_region_separability(region_ids): rank all foundation models "
            "by how well their features separate specific labeled regions. region_ids is a list of "
            "at least 2 region IDs (use list_labeled_regions to find them). mag and patch_size "
            "default to currently loaded values. Optional: metric ('euclidean' or 'cosine').\n\n"
            "When the user asks about a specific slide or model, call list_data first if you do "
            "not already know the available options. For questions about the currently loaded "
            "slide, annotations, clusters, or embedding, prefer Tier 2 tools."
        )
        history = self._session.messages[-self._config.context_messages :]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        for message in history:
            messages.append({"role": message.role, "content": message.content})
        return messages

    def _mcp_tools_to_openai_tools(self) -> List[Dict[str, Any]]:
        """Fetch MCP tool schemas and translate to OpenAI tool format (cached per session)."""
        if self._cached_tools is None:
            tools: List[Dict[str, Any]] = []
            for tool in self._mcp.list_tools():
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {"type": "object"}),
                        },
                    }
                )
            self._cached_tools = tools
        return self._cached_tools

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        root_dir: str,
    ) -> List[Dict[str, Any]]:
        """Run allowed tools and return tool result messages for model follow-up."""
        result_messages: List[Dict[str, Any]] = []

        root_path = Path(root_dir).resolve() if root_dir else None

        for call in tool_calls:
            if self._cancel_requested:
                break

            call_id = call.get("call_id") or str(uuid.uuid4())
            name = call.get("name") or ""
            args = call.get("arguments") or {}

            if name not in self._allowed_tools:
                raise ValueError(f"Tool not allowed: {name}")

            if root_path is not None:
                args.setdefault("root_dir", str(root_path))
                if "h5_path" in args:
                    self._ensure_path_within_root(root_path, str(args["h5_path"]))

            args = _coerce_json_strings(args)
            self.tool_started.emit(call_id, name, json.dumps(args, default=str))
            raw = self._mcp.call_tool(name, args, timeout_s=self._config.tool_timeout_s)
            logger.debug("Tool %s → %s", name, json.dumps(raw, default=str)[:500])
            summary = self._summarize_tool_result(raw)
            self.tool_finished.emit(call_id, name, summary, raw)

            result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(summary, default=str),
                }
            )

        return result_messages

    def _stream_final_response(
        self,
        request_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        """Stream final model response and aggregate full text."""
        self.response_started.emit(request_id)
        full_text: List[str] = []

        for chunk in self._llm.stream_with_tools(
            messages=messages,
            tools=[],
            timeout_s=self._config.llm_timeout_s,
        ):
            if self._cancel_requested:
                break
            delta = self._llm.extract_text_delta(chunk)
            if not delta:
                continue
            full_text.append(delta)
            self.response_delta.emit(request_id, delta)

        return "".join(full_text), {}

    @staticmethod
    def _build_assistant_tool_calls_message(
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construct an assistant message containing tool_calls.\n\n        This ensures any subsequent tool result messages are valid.\n        """
        formatted_calls: List[Dict[str, Any]] = []
        for call in tool_calls:
            call_id = call.get("call_id") or str(uuid.uuid4())
            name = call.get("name") or ""
            arguments = call.get("arguments") or {}
            formatted_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, default=str)},
                }
            )
        return {"role": "assistant", "content": "", "tool_calls": formatted_calls}

    @staticmethod
    def _extract_message_content(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for entry in content:
                if isinstance(entry, dict) and entry.get("type") == "text":
                    parts.append(entry.get("text") or "")
            return "".join(parts)
        return ""

    @staticmethod
    def _ensure_path_within_root(root_dir: Path, relative_path: str) -> None:
        resolved = (root_dir / relative_path).resolve()
        if not str(resolved).startswith(str(root_dir)):
            raise ValueError("Path outside root_dir")

    @staticmethod
    def _summarize_tool_result(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Build concise summary to send back to model context."""
        # Error passthrough — always surface errors fully
        if "error" in raw:
            return {"error": raw["error"]}

        # list_data
        if "count_summary" in raw:
            return {
                "slides": raw.get("slides", []),
                "models": raw.get("models", []),
                "magnifications": raw.get("magnifications", []),
                "patch_sizes": raw.get("patch_sizes", []),
                "count_summary": raw.get("count_summary", {}),
            }

        # inspect_h5
        if "datasets" in raw:
            return {
                "dataset_count": len(raw.get("datasets") or []),
                "datasets": raw.get("datasets") or [],
                "warnings": raw.get("warnings") or [],
            }

        # describe_slide
        if "patch_count" in raw and "feature_dim" in raw:
            return {
                "patch_count": raw.get("patch_count"),
                "feature_dim": raw.get("feature_dim"),
                "patch_size_px": raw.get("patch_size_px"),
                "coord_range": raw.get("coord_range"),
                "warnings": raw.get("warnings") or [],
            }

        # compare_models
        if "model_a" in raw and "model_b" in raw:
            ma = raw.get("model_a") or {}
            mb = raw.get("model_b") or {}
            return {
                "slide_name": raw.get("slide_name"),
                "model_a": {
                    "name": ma.get("name"),
                    "silhouette_score": ma.get("silhouette_score"),
                    "davies_bouldin_index": ma.get("davies_bouldin_index"),
                },
                "model_b": {
                    "name": mb.get("name"),
                    "silhouette_score": mb.get("silhouette_score"),
                    "davies_bouldin_index": mb.get("davies_bouldin_index"),
                },
                "overlap": raw.get("overlap"),
                "warnings": raw.get("warnings") or [],
            }

        # rank_models_by_separability / rank_models_by_selected_cluster_separability
        # / rank_models_by_labeled_region_separability
        if "ranking" in raw:
            return {
                "slide_name": raw.get("slide_name"),
                "region_ids": raw.get("region_ids"),
                "region_names": raw.get("region_names"),
                "target_cluster_ids": raw.get("target_cluster_ids"),
                "n_models_evaluated": raw.get("n_models_evaluated"),
                "n_clusters_used": raw.get("n_clusters_used"),
                "metric": raw.get("metric"),
                "ranking": [
                    {
                        "rank": r.get("rank"),
                        "model": r.get("model"),
                        "silhouette_score": r.get("silhouette_score"),
                        "davies_bouldin_index": r.get("davies_bouldin_index"),
                    }
                    for r in (raw.get("ranking") or [])
                ],
                "warnings": raw.get("warnings") or [],
            }

        # compute_elbow_analysis
        if "elbow_curve" in raw:
            return {
                "slide_name": raw.get("slide_name"),
                "model": raw.get("model"),
                "recommended_k": raw.get("recommended_k"),
                "method": raw.get("method"),
                "n_k_values_tested": len(raw.get("elbow_curve") or []),
                "warnings": raw.get("warnings") or [],
            }

        # list_labeled_regions
        if "labeled_regions" in raw:
            regions = raw.get("labeled_regions") or []
            return {
                "total_count": raw.get("total_count", len(regions)),
                "regions": [
                    {
                        "region_id": r.get("region_id"),
                        "name": r.get("name"),
                        "color_hex": r.get("color_hex"),
                        "patch_count": r.get("patch_count"),
                        "source_mode": r.get("source_mode"),
                        "kmeans_cluster": r.get("kmeans_cluster"),
                    }
                    for r in regions
                ],
            }

        # compute_region_stats / compute_cluster_stats
        if "computed" in raw:
            return {
                "region_id": raw.get("region_id"),
                "cluster_id": raw.get("cluster_id"),
                "name": raw.get("name"),
                "computed": raw["computed"],
                "unknown_metrics": raw.get("unknown_metrics", []),
            }

        # find_similar_patches
        if "similar_patches" in raw:
            patches = raw.get("similar_patches") or []
            return {
                "region_id": raw.get("region_id"),
                "region_name": raw.get("region_name"),
                "metric": raw.get("metric"),
                "top_k": raw.get("top_k"),
                "top_5_shown": [
                    {
                        "rank": p.get("rank"),
                        "patch_index": p.get("patch_index"),
                        "distance": p.get("distance"),
                        "cluster_id": p.get("cluster_id"),
                    }
                    for p in patches[:5]
                ],
            }

        # get_boundary_patches
        if "boundary_patches" in raw:
            patches = raw.get("boundary_patches") or []
            return {
                "top_k": raw.get("top_k"),
                "top_5_shown": [
                    {
                        "patch_index": p.get("patch_index"),
                        "assigned_cluster": p.get("assigned_cluster"),
                        "nearest_other_cluster": p.get("nearest_other_cluster"),
                        "gap": p.get("gap"),
                    }
                    for p in patches[:5]
                ],
                "note": raw.get("note"),
            }

        # get_pca_info
        if "pca_explained_variance_ratio" in raw:
            return {
                "n_components": raw.get("n_components"),
                "feature_dim": raw.get("feature_dim"),
                "pca_explained_variance_ratio": raw.get("pca_explained_variance_ratio"),
                "cumulative_explained_variance": raw.get("cumulative_explained_variance"),
                "note": raw.get("note"),
            }

        # compare_selected_clusters
        if "selected_cluster_ids" in raw:
            return {
                "selected_cluster_ids": raw["selected_cluster_ids"],
                "cluster_count": raw["cluster_count"],
                "results": raw["results"],
            }

        # atlas_cluster_representation
        if "single_slide_only" in raw:
            return {
                "n_clusters": raw["n_clusters"],
                "slide_names": raw["slide_names"],
                "most_represented": raw.get("most_represented", []),
                "single_slide_only": raw.get("single_slide_only", []),
                "clusters": raw.get("clusters", []),
            }

        return {"keys": sorted(list(raw.keys()))[:20]}

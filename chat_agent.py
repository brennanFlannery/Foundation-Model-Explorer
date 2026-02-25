"""Chat orchestration worker for LLM + MCP tool execution."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

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


RegionMetricName = Literal[
    "spread",
    "area_covered",
    "area_bbox",
    "area_hull",
    "homogeneity",
    "global_distance",
    "pca_extent",
    "top_dims",
    "area_mm2",
    "width_mm",
]

ClusterMetricName = Literal[
    "spread",
    "area_covered",
    "area_bbox",
    "area_hull",
    "homogeneity",
    "pca_extent",
    "patch_count",
    "area_mm2",
    "width_mm",
    "separation",
    "discriminating_dims",
]

REGION_METRICS: Set[RegionMetricName] = {
    "spread",
    "area_covered",
    "area_bbox",
    "area_hull",
    "homogeneity",
    "global_distance",
    "pca_extent",
    "top_dims",
    "area_mm2",
    "width_mm",
}

CLUSTER_METRICS: Set[ClusterMetricName] = {
    "spread",
    "area_covered",
    "area_bbox",
    "area_hull",
    "homogeneity",
    "pca_extent",
    "patch_count",
    "area_mm2",
    "width_mm",
    "separation",
    "discriminating_dims",
}

TOOL_HINTS: Dict[str, str] = {
    "list_data": "discover slides/models/magnifications/patch sizes from root_dir",
    "inspect_h5": "inspect raw HDF5 datasets/attributes",
    "describe_slide": "summarize one slide-model-mag-patch configuration",
    "compare_models": "compare two foundation models on same slide",
    "rank_models_by_separability": "rank models by fresh re-cluster separability",
    "compute_elbow_analysis": "choose candidate k with elbow method",
    "list_labeled_regions": "list current GUI annotations/regions",
    "compute_region_stats": "compute requested metrics for one region",
    "compute_region_geometry_stats": "geometry-focused metrics for one region",
    "compute_region_feature_stats": "feature-focused metrics for one region",
    "find_similar_patches": "find nearest patches to a region centroid",
    "compute_cluster_stats": "compute requested metrics for one cluster",
    "compute_cluster_geometry_stats": "geometry-focused metrics for one cluster",
    "compute_cluster_feature_stats": "feature-focused metrics for one cluster",
    "get_boundary_patches": "find patches near cluster decision boundaries",
    "get_pca_info": "inspect PCA explained variance and notes",
    "compare_selected_clusters": "compare all currently selected clusters at once",
    "atlas_cluster_representation": "cross-slide atlas distribution by cluster",
    "rank_models_by_selected_cluster_separability": "rank models for selected clusters",
    "rank_models_by_labeled_region_separability": "rank models for chosen labeled regions",
    # GUI action tools
    "find_most_similar_cluster": "find cluster most similar to a region by centroid distance",
    "label_cluster": "label an entire K-means cluster as a new region",
    "label_similar_cluster": "find + label the cluster most similar to a region",
    "label_similar_patches_as_region": "find similar patches and create a region from them",
    "create_region_from_patches": "create a labeled region from explicit patch indices",
    "delete_region": "remove a labeled region from the GUI",
    "rename_region": "rename a labeled region",
    "navigate_to_region": "pan and zoom slide view to show a region",
    "select_cluster": "highlight a K-means cluster in slide and scatter views",
    "clear_all_regions": "remove all labeled regions",
    "deselect_all_clusters": "clear cluster selection highlighting",
    "expand_region": "expand a labeled region outward by N grid rings of adjacent patches",
    "find_most_different_cluster": "find cluster most dissimilar to combined centroid of one or more regions",
    "find_most_distinct_cluster": "rank clusters by distinctness without requiring pre-labeled regions",
    "switch_to_atlas_view": "switch GUI sidebar to the Atlas tab",
    "highlight_atlas_cluster": "highlight a cluster across all atlas thumbnails and scatter view",
    "set_cluster_count": "change K-means cluster count and re-cluster",
    "load_slide": "switch the active slide in the GUI",
    "lookup_patch_by_coords": "find patch nearest to a given level-0 coordinate",
    "export_regions_geojson": "export all labeled regions to a GeoJSON file",
    "open_patch_exemplar_popup": "open a horizontal popup with sampled exemplar patch images",
    "export_current_exemplar_popup": "export images currently shown in exemplar popup",
    "close_exemplar_popup": "close the active exemplar popup",
    "generate_slide_qc_report": "create a markdown QC report plus JSON summary for a slide/model setup",
    "generate_cross_slide_qc_report": "create a cross-slide QC report with atlas-style OOD ranking",
    "detect_ood_patches": "score out-of-distribution patches using embedding-space k-NN distance",
    "label_ood_patches_as_region": "create labeled region annotations from OOD patch indices",
}


@dataclass
class ChatAgentConfig:
    """Configuration for chat worker behavior."""

    model: str
    api_key: str
    llm_timeout_s: float = 60.0
    tool_timeout_s: float = 30.0
    context_messages: int = 20
    max_tool_rounds: int = 4


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
            "compute_region_geometry_stats",
            "compute_region_feature_stats",
            "find_similar_patches",
            "compute_cluster_stats",
            "compute_cluster_geometry_stats",
            "compute_cluster_feature_stats",
            "get_boundary_patches",
            "get_pca_info",
            "compare_selected_clusters",
            "atlas_cluster_representation",
            "rank_models_by_selected_cluster_separability",
            "rank_models_by_labeled_region_separability",
            # GUI action tools
            "find_most_similar_cluster",
            "label_cluster",
            "label_similar_cluster",
            "label_similar_patches_as_region",
            "create_region_from_patches",
            "delete_region",
            "rename_region",
            "navigate_to_region",
            "select_cluster",
            "clear_all_regions",
            "deselect_all_clusters",
            "expand_region",
            "find_most_different_cluster",
            "find_most_distinct_cluster",
            "switch_to_atlas_view",
            "highlight_atlas_cluster",
            "set_cluster_count",
            "load_slide",
            "lookup_patch_by_coords",
            "export_regions_geojson",
            "open_patch_exemplar_popup",
            "export_current_exemplar_popup",
            "close_exemplar_popup",
            "generate_slide_qc_report",
            "generate_cross_slide_qc_report",
            "detect_ood_patches",
            "label_ood_patches_as_region",
        }
        self._cached_tools: Optional[List[Dict[str, Any]]] = None
        self._tool_stats: Dict[str, Dict[str, float]] = {}
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
            app_context = app_context or {}
            root_dir = app_context.get("root_dir") or ""
            self._session.messages.append(
                ChatMessage(role="user", content=text, message_id=str(uuid.uuid4()))
            )

            slash_command = app_context.get("slash_command")
            if isinstance(slash_command, dict) and slash_command.get("kind") == "help":
                help_text = self._slash_help_text()
                self.response_started.emit(request_id)
                for character in help_text:
                    if self._cancel_requested:
                        break
                    self.response_delta.emit(request_id, character)
                final_text = help_text
                usage = {"total_tokens": 0, "tool_telemetry": self._tool_telemetry_summary()}
            else:
                forced_call = self._forced_tool_call_from_context(slash_command)
                selected_tools = self._select_tool_subset(app_context, forced_call)
                tools = self._mcp_tools_to_openai_tools(allowed_names=selected_tools)
                messages = self._build_prompt_messages(
                    root_dir=root_dir,
                    selected_tools=selected_tools,
                    app_context=app_context,
                )

                any_tools_used = False
                if forced_call is not None:
                    any_tools_used = True
                    messages.append(self._build_assistant_tool_calls_message([forced_call]))
                    forced_results = self._execute_tool_calls(
                        tool_calls=[forced_call],
                        root_dir=root_dir,
                    )
                    messages.extend(forced_results)

                last_response: Optional[Dict[str, Any]] = None
                max_rounds = max(1, int(self._config.max_tool_rounds))

                for _ in range(max_rounds):
                    if self._cancel_requested:
                        return

                    response = self._llm.complete_with_tools(
                        messages=messages,
                        tools=tools,
                        timeout_s=self._config.llm_timeout_s,
                    )
                    last_response = response
                    tool_calls = self._llm.normalize_tool_calls(response)

                    if not tool_calls:
                        break

                    any_tools_used = True
                    messages.append(self._build_assistant_tool_calls_message(tool_calls))
                    tool_messages = self._execute_tool_calls(
                        tool_calls=tool_calls,
                        root_dir=root_dir,
                    )
                    messages.extend(tool_messages)
                else:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Stop using tools now and provide the best concise answer from "
                                "already collected tool outputs."
                            ),
                        }
                    )

                if self._cancel_requested:
                    return

                if any_tools_used:
                    final_text, usage = self._stream_final_response(
                        request_id=request_id,
                        messages=messages,
                    )
                else:
                    final_text = self._extract_message_content(last_response or {})
                    usage = self._llm.normalize_usage(last_response or {})
                    self.response_started.emit(request_id)
                    for character in final_text:
                        if self._cancel_requested:
                            break
                        self.response_delta.emit(request_id, character)

            if self._cancel_requested:
                return

            usage["tool_telemetry"] = self._tool_telemetry_summary()
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

    def _build_prompt_messages(
        self,
        root_dir: str,
        selected_tools: Set[str],
        app_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build model prompt with bounded history and dynamic tool context."""
        has_slide_loaded = bool(app_context.get("has_slide_loaded"))
        has_labeled_regions = bool(app_context.get("has_labeled_regions"))
        has_selected_clusters = bool(app_context.get("has_selected_clusters"))
        atlas_ready = bool(app_context.get("atlas_ready"))

        ordered_tools = sorted(selected_tools)
        tool_lines = []
        for name in ordered_tools:
            hint = TOOL_HINTS.get(name, "inspect tool schema for details")
            tool_lines.append(f"- {name}: {hint}")

        recommended: List[str] = []
        if not has_slide_loaded:
            recommended.append("Start with list_data.")
        if has_labeled_regions:
            recommended.append("When asked about annotations, call list_labeled_regions first.")
        if has_selected_clusters:
            recommended.append(
                "For model ranking on highlighted clusters, prefer rank_models_by_selected_cluster_separability."
            )
        elif has_labeled_regions:
            recommended.append(
                "For model ranking on annotated regions, prefer rank_models_by_labeled_region_separability."
            )
        else:
            recommended.append(
                "For model ranking without selections, use rank_models_by_separability."
            )
        if not atlas_ready and "atlas_cluster_representation" in selected_tools:
            recommended.append("atlas_cluster_representation needs a built atlas in GUI.")

        system_content = (
            "You are a pathology-aware assistant inside FoundationDetector. "
            "Use tools proactively and only answer from tool outputs when details are requested. "
            "Allowed data boundary: read-only within selected root_dir only.\n"
            f"Current root_dir: {root_dir or 'not set'}\n"
            f"GUI state: has_slide_loaded={has_slide_loaded}, "
            f"has_labeled_regions={has_labeled_regions}, "
            f"has_selected_clusters={has_selected_clusters}, atlas_ready={atlas_ready}\n\n"
            "Available tools for this turn:\n"
            + "\n".join(tool_lines)
            + "\n\nRouting hints:\n"
            + "\n".join(f"- {line}" for line in recommended)
        )
        history = self._session.messages[-self._config.context_messages :]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        for message in history:
            messages.append({"role": message.role, "content": message.content})
        return messages

    def _mcp_tools_to_openai_tools(self, allowed_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Fetch MCP tool schemas and translate to OpenAI tool format (cached per session)."""
        if self._cached_tools is None:
            tools: List[Dict[str, Any]] = []
            for tool in self._mcp.list_tools():
                name = tool.get("name", "")
                if name not in self._allowed_tools:
                    continue
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {"type": "object"}),
                        },
                    }
                )
            self._cached_tools = tools
        if not allowed_names:
            return self._cached_tools
        filtered = [
            t for t in self._cached_tools
            if (t.get("function") or {}).get("name") in allowed_names
        ]
        return filtered

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
            args = self._sanitize_tool_arguments(name=name, args=args)
            self.tool_started.emit(call_id, name, json.dumps(args, default=str))
            call_started = time.time()
            raw = self._mcp.call_tool(name, args, timeout_s=self._config.tool_timeout_s)
            latency_ms = int((time.time() - call_started) * 1000)
            logger.debug("Tool %s → %s", name, json.dumps(raw, default=str)[:500])
            summary = self._summarize_tool_result(raw)
            self.tool_finished.emit(call_id, name, summary, raw)
            self._record_tool_metric(
                tool_name=name,
                success="error" not in raw,
                latency_ms=latency_ms,
            )

            result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(summary, default=str),
                }
            )

        return result_messages

    def _forced_tool_call_from_context(
        self,
        slash_command: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(slash_command, dict):
            return None
        if slash_command.get("kind") != "tool":
            return None
        tool_name = str(slash_command.get("tool_name") or "").strip()
        args = slash_command.get("arguments") or {}
        if not tool_name:
            return None
        if not isinstance(args, dict):
            raise ValueError("Slash command arguments must be an object")
        return {
            "call_id": f"slash-{uuid.uuid4()}",
            "name": tool_name,
            "arguments": args,
        }

    def _select_tool_subset(
        self,
        app_context: Dict[str, Any],
        forced_call: Optional[Dict[str, Any]],
    ) -> Set[str]:
        if forced_call is not None:
            name = forced_call.get("name") or ""
            return {name} if name in self._allowed_tools else set()

        has_slide_loaded = bool(app_context.get("has_slide_loaded"))
        has_labeled_regions = bool(app_context.get("has_labeled_regions"))
        has_selected_clusters = bool(app_context.get("has_selected_clusters"))
        atlas_ready = bool(app_context.get("atlas_ready"))

        selected: Set[str] = {
            "list_data",
            "inspect_h5",
            "describe_slide",
            "compare_models",
            "rank_models_by_separability",
            "compute_elbow_analysis",
            "generate_slide_qc_report",
            "generate_cross_slide_qc_report",
            "load_slide",
        }

        if has_slide_loaded:
            selected.update(
                {
                    "list_labeled_regions",
                    "compute_region_stats",
                    "compute_region_geometry_stats",
                    "compute_region_feature_stats",
                    "find_similar_patches",
                    "compute_cluster_stats",
                    "compute_cluster_geometry_stats",
                    "compute_cluster_feature_stats",
                    "get_boundary_patches",
                    "get_pca_info",
                    "compare_selected_clusters",
                    # GUI action tools — available whenever a slide is loaded
                    "label_cluster",
                    "select_cluster",
                    "clear_all_regions",
                    "deselect_all_clusters",
                    "create_region_from_patches",
                    "navigate_to_region",
                    "set_cluster_count",
                    "lookup_patch_by_coords",
                    "open_patch_exemplar_popup",
                    "close_exemplar_popup",
                    "detect_ood_patches",
                    "label_ood_patches_as_region",
                    "find_most_distinct_cluster",
                }
            )

        if has_labeled_regions:
            selected.update(
                {
                    "rank_models_by_labeled_region_separability",
                    # GUI action tools — require at least one region to exist
                    "find_most_similar_cluster",
                    "label_similar_cluster",
                    "label_similar_patches_as_region",
                    "delete_region",
                    "rename_region",
                    "expand_region",
                    "find_most_different_cluster",
                    "export_regions_geojson",
                    "export_current_exemplar_popup",
                }
            )
        if has_selected_clusters:
            selected.add("rank_models_by_selected_cluster_separability")
        if atlas_ready:
            selected.update(
                {
                    "atlas_cluster_representation",
                    "switch_to_atlas_view",
                    "highlight_atlas_cluster",
                }
            )

        return {name for name in selected if name in self._allowed_tools}

    def _sanitize_tool_arguments(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp common argument mistakes so weaker models still succeed."""
        sanitized = dict(args)

        if name in (
            "compute_region_stats",
            "compute_region_feature_stats",
            "compute_region_geometry_stats",
        ):
            metrics = sanitized.get("metrics")
            if isinstance(metrics, list):
                valid = [m for m in metrics if isinstance(m, str) and m in REGION_METRICS]
                if valid:
                    sanitized["metrics"] = valid
                else:
                    sanitized["metrics"] = ["spread", "area_bbox"]

        if name in (
            "compute_cluster_stats",
            "compute_cluster_feature_stats",
            "compute_cluster_geometry_stats",
            "compare_selected_clusters",
        ):
            metrics = sanitized.get("metrics")
            if isinstance(metrics, list):
                valid = [m for m in metrics if isinstance(m, str) and m in CLUSTER_METRICS]
                if valid:
                    sanitized["metrics"] = valid
                else:
                    sanitized["metrics"] = ["patch_count", "area_bbox"]

        if "top_k" in sanitized:
            try:
                sanitized["top_k"] = max(1, int(sanitized["top_k"]))
            except Exception:
                sanitized["top_k"] = 20

        metric = sanitized.get("metric")
        if metric is not None and metric not in ("cosine", "euclidean"):
            sanitized["metric"] = "euclidean"

        return sanitized

    def _record_tool_metric(self, tool_name: str, success: bool, latency_ms: int) -> None:
        stats = self._tool_stats.setdefault(
            tool_name,
            {"calls": 0.0, "successes": 0.0, "failures": 0.0, "latency_total_ms": 0.0},
        )
        stats["calls"] += 1.0
        if success:
            stats["successes"] += 1.0
        else:
            stats["failures"] += 1.0
        stats["latency_total_ms"] += float(latency_ms)

    def _tool_telemetry_summary(self) -> Dict[str, Any]:
        by_tool: Dict[str, Any] = {}
        for name, stats in sorted(self._tool_stats.items()):
            calls = int(stats.get("calls", 0))
            total_ms = float(stats.get("latency_total_ms", 0.0))
            avg = round(total_ms / calls, 1) if calls else 0.0
            by_tool[name] = {
                "calls": calls,
                "successes": int(stats.get("successes", 0)),
                "failures": int(stats.get("failures", 0)),
                "avg_latency_ms": avg,
            }
        return {"tools": by_tool}

    @staticmethod
    def _slash_help_text() -> str:
        return (
            "Slash commands:\n"
            "\n"
            "— Inspection —\n"
            "/data {json}     -> list_data\n"
            "/slide {json}    -> describe_slide\n"
            "/models {json}   -> rank_models_by_separability\n"
            "/regions         -> list_labeled_regions\n"
            "/clusters        -> compare_selected_clusters\n"
            "/atlas           -> atlas_cluster_representation\n"
            "\n"
            "— GUI actions —\n"
            "/label {\"cluster_id\": N}                        -> label_cluster\n"
            "/select {\"cluster_id\": N}                       -> select_cluster\n"
            "/similar {\"region_id\": N}                       -> label_similar_cluster\n"
            "/navigate {\"region_id\": N}                      -> navigate_to_region\n"
            "/rename {\"region_id\": N, \"new_name\": \"...\"}    -> rename_region\n"
            "/delete {\"region_id\": N}                        -> delete_region\n"
            "/clear                                          -> clear_all_regions\n"
            "/deselect                                       -> deselect_all_clusters\n"
            "/expand {\"region_id\": N, \"n_rings\": 1}          -> expand_region\n"
            "/different {\"region_ids\": [N]}                   -> find_most_different_cluster\n"
            "/atlas-cluster {\"cluster_id\": N}                 -> highlight_atlas_cluster\n"
            "\n"
            "— Escape hatch —\n"
            "/tool <name> {json} -> direct MCP tool call\n"
            "/tools              -> show this help"
        )

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
                "color_hex": raw.get("color_hex"),
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
            result = {
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
            if "cluster_pair" in raw:
                result["cluster_pair"] = raw["cluster_pair"]
            return result

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

        # detect_ood_patches
        if "top_outliers" in raw and "outlier_count" in raw:
            return {
                "mode": raw.get("mode"),
                "method": raw.get("method"),
                "outlier_count": raw.get("outlier_count"),
                "outlier_fraction": raw.get("outlier_fraction"),
                "threshold_used": raw.get("threshold_used"),
                "reference_slides": raw.get("reference_slides"),
                "top_5_shown": [
                    {
                        "patch_index": p.get("patch_index"),
                        "score": p.get("score"),
                        "cluster_id": p.get("cluster_id"),
                    }
                    for p in (raw.get("top_outliers") or [])[:5]
                ],
                "warnings": raw.get("warnings") or [],
            }

        # label_ood_patches_as_region
        if "created_regions" in raw and "grouping_mode" in raw:
            return {
                "grouping_mode": raw.get("grouping_mode"),
                "component_count": raw.get("component_count"),
                "dropped_components": raw.get("dropped_components"),
                "total_input_patches": raw.get("total_input_patches"),
                "total_labeled_patches": raw.get("total_labeled_patches"),
                "exact_match": raw.get("exact_match"),
                "created_regions": raw.get("created_regions"),
            }

        # find_most_distinct_cluster / find_most_different_cluster
        if "ranked_clusters" in raw and "most_distinct_cluster_id" in raw:
            return {
                "metric": raw.get("metric"),
                "most_distinct_cluster_id": raw.get("most_distinct_cluster_id"),
                "top_ranked": (raw.get("ranked_clusters") or [])[:5],
                "note": raw.get("note"),
            }

        # generate_cross_slide_qc_report
        if "most_ood_slide" in raw and "summary_json_path" in raw and "scores" in raw:
            return {
                "report_path": raw.get("report_path"),
                "summary_json_path": raw.get("summary_json_path"),
                "most_ood_slide": raw.get("most_ood_slide"),
                "scores": raw.get("scores"),
                "warnings": raw.get("warnings") or [],
            }

        # generate_slide_qc_report
        if "report_path" in raw and "scores" in raw:
            return {
                "report_path": raw.get("report_path"),
                "summary_json_path": raw.get("summary_json_path"),
                "scores": raw.get("scores"),
                "warnings": raw.get("warnings") or [],
            }

        return {"keys": sorted(list(raw.keys()))[:20]}

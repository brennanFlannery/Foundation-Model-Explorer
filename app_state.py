"""Thread-safe application state registry shared between GUI and MCP tools.

No Qt imports. MainWindow writes Qt-free plain-Python values here;
mcp_server.py tools read from here.
"""
from __future__ import annotations

import threading
import queue as _queue
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class AtlasState:
    """Qt-free summary of a built multi-slide atlas."""

    global_labels: np.ndarray    # shape (total_patches,) — global cluster per patch
    slide_indices: np.ndarray    # shape (total_patches,) — slide index per patch
    slide_names: List[str]       # ordered list of slide names
    n_clusters: int


@dataclass
class LabeledRegionData:
    """Qt-free representation of a labeled region annotation."""

    region_id: int
    name: str
    color_hex: str          # QColor.name() output, e.g. "#e63946"
    patch_indices: List[int]
    source_mode: str        # "kmeans" or "local"
    kmeans_cluster: int


@dataclass
class AppState:
    """Snapshot of GUI state accessible to MCP tools."""

    # Slide identity
    slide_name: Optional[str] = None
    root_dir: Optional[str] = None
    selected_models: Optional[List[str]] = None
    magnification: Optional[str] = None
    patch_size: Optional[str] = None

    # Feature arrays (from _load_current_data)
    features: Optional[np.ndarray] = None          # (n_patches, feature_dim)
    coords_lv0: Optional[np.ndarray] = None        # (n_patches, 2)
    coords_thumb: Optional[np.ndarray] = None      # (n_patches, 2)
    patch_size_lv0: Optional[float] = None

    # Clustering
    cluster_labels: Optional[np.ndarray] = None    # (n_patches,) int
    cluster_centroids: Optional[np.ndarray] = None  # (k, feature_dim)
    cluster_colours: Optional[List[str]] = None     # len k

    # PCA
    embedding_2d: Optional[np.ndarray] = None                    # (n_patches, 2)
    pca_explained_variance_ratio: Optional[List[float]] = None

    # Annotations (Qt-free)
    labeled_regions: Dict[int, LabeledRegionData] = field(default_factory=dict)
    selected_clusters: Set[int] = field(default_factory=set)

    # Physical scale
    mpp: Optional[float] = None           # Microns-per-pixel at level 0 (from OpenSlide)

    # Multi-slide atlas (Qt-free)
    atlas_state: Optional[AtlasState] = None


_lock = threading.RLock()
_state = AppState()

# ---------------------------------------------------------------------------
# GUI action queue — MCP tools post here; main-thread QTimer drains
# ---------------------------------------------------------------------------

_gui_action_queue: _queue.SimpleQueue = _queue.SimpleQueue()
_gui_action_results: Dict[str, Any] = {}
_gui_results_lock = threading.Lock()


def post_gui_action(action_type: str, params: Dict[str, Any]) -> str:
    """Enqueue a GUI action. Returns action_id for polling."""
    action_id = str(_uuid.uuid4())
    _gui_action_queue.put({"action_id": action_id, "action_type": action_type, "params": params})
    return action_id


def set_gui_action_result(action_id: str, result: Any) -> None:
    """Store result for a completed GUI action."""
    with _gui_results_lock:
        _gui_action_results[action_id] = result


def get_gui_action_result(action_id: str) -> Optional[Any]:
    """Return and remove the result, or None if not yet ready."""
    with _gui_results_lock:
        return _gui_action_results.pop(action_id, None)


def drain_gui_actions() -> List[Dict[str, Any]]:
    """Drain all pending actions. Called from main-thread QTimer."""
    actions = []
    while True:
        try:
            actions.append(_gui_action_queue.get_nowait())
        except _queue.Empty:
            break
    return actions


def update(**kwargs) -> None:
    """Update one or more fields in the global state atomically."""
    with _lock:
        for k, v in kwargs.items():
            setattr(_state, k, v)


def get() -> AppState:
    """Return the current state snapshot (under lock)."""
    with _lock:
        return _state

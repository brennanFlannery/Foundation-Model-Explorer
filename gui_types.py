"""
gui_types.py
============

Shared enums and dataclasses used across the FoundationDetector GUI modules.
Extracted from gui.py to avoid circular imports when split into widget/mixin files.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple

from PySide6.QtGui import QColor


class SelectionMode(Enum):
    """Selection mode for the application."""
    KMEANS = "kmeans"        # Default - click selects entire K-means cluster
    LOCAL_REGION = "local"   # Click selects K-means cluster patches within radius


class SourceMode(Enum):
    """Source mode for labeled regions."""
    KMEANS = "kmeans"
    LOCAL  = "local"


@dataclass
class LocalRegionCluster:
    """Represents a user-defined local region cluster (subset of K-means cluster)."""
    cluster_id: int
    patch_indices: Set[int]  # indices into the patch arrays
    center_point: Tuple[float, float]  # click center in scene coords
    radius: float  # radius at time of creation
    color: QColor
    name: str
    kmeans_cluster: int  # the K-means cluster this region belongs to


@dataclass
class LabeledRegion:
    """Unified annotation record from any analysis mode."""
    region_id: int
    name: str
    color: QColor
    patch_indices: Set[int]
    source_mode: SourceMode
    kmeans_cluster: int
    center_point: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    slide_name: Optional[str] = None

    @property
    def cluster_id(self) -> int:
        """Backward-compatible alias for region_id."""
        return self.region_id


@dataclass
class ModelSelection:
    """Represents a selected model configuration for feature loading."""
    models: List[str]
    magnification: str
    patch_size: str

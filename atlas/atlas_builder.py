"""
atlas_builder.py
================

Cross-slide cluster atlas builder for unified clustering across multiple slides.

This module provides the AtlasBuilder class which aggregates feature embeddings
from multiple whole-slide images and performs global K-means clustering to create
a unified cluster atlas. This enables cross-slide pattern discovery and cohort-level
analysis.

Classes
-------
SlideAtlasEntry
    Stores per-slide data within the atlas including features, coordinates,
    and global cluster assignments.

ClusterAtlas
    Complete atlas structure containing global features, labels, PCA embeddings,
    and per-slide mappings.

AtlasBuilder
    Builder class for constructing a ClusterAtlas from multiple slides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtGui import QColor

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from utils import generate_palette


def _hsl_to_qcolor(hsl_string: str) -> QColor:
    """Convert HSL string to QColor."""
    try:
        values = hsl_string.strip().lower().replace('hsl(', '').rstrip(')').split(',')
        h = float(values[0])
        s = float(values[1].strip(' %')) / 100.0
        l = float(values[2].strip(' %')) / 100.0
        c = QColor()
        c.setHslF(h / 360.0, s, l)
        return c
    except Exception:
        return QColor('gray')


@dataclass
class SlideAtlasEntry:
    """Stores per-slide data for the atlas.

    Attributes
    ----------
    slide_name : str
        Name of the slide (without extension).
    features : np.ndarray
        Feature vectors of shape (n_patches, feature_dim).
    coords : np.ndarray
        Patch coordinates of shape (n_patches, 2-3).
    global_labels : np.ndarray
        Global cluster assignments of shape (n_patches,).
    local_to_global : np.ndarray
        Mapping from local patch index to global atlas index.
    """
    slide_name: str
    features: np.ndarray
    coords: np.ndarray
    global_labels: np.ndarray
    local_to_global: np.ndarray


@dataclass
class ClusterAtlas:
    """Global cluster atlas across multiple slides.

    Attributes
    ----------
    entries : Dict[str, SlideAtlasEntry]
        Mapping from slide name to per-slide atlas entry.
    global_features : np.ndarray
        Concatenated features of shape (total_patches, feature_dim).
    global_labels : np.ndarray
        Global cluster assignments of shape (total_patches,).
    global_pca_coords : np.ndarray
        2D PCA embeddings of shape (total_patches, 2).
    n_clusters : int
        Number of clusters in the atlas.
    cluster_colors : List[QColor]
        Color palette for clusters.
    slide_indices : np.ndarray
        Mapping from each point to its source slide index.
    slide_names : List[str]
        Ordered list of slide names (index corresponds to slide_indices values).
    """
    entries: Dict[str, SlideAtlasEntry]
    global_features: np.ndarray
    global_labels: np.ndarray
    global_pca_coords: np.ndarray
    n_clusters: int
    cluster_colors: List[QColor]
    slide_indices: np.ndarray
    slide_names: List[str] = field(default_factory=list)

    def get_slide_index(self, slide_name: str) -> int:
        """Get the index of a slide in the atlas."""
        try:
            return self.slide_names.index(slide_name)
        except ValueError:
            return -1

    def get_cluster_count(self, cluster_id: int) -> int:
        """Get total patch count for a cluster."""
        return int(np.sum(self.global_labels == cluster_id))

    def get_slide_cluster_count(self, slide_name: str, cluster_id: int) -> int:
        """Get patch count for a cluster within a specific slide."""
        if slide_name not in self.entries:
            return 0
        entry = self.entries[slide_name]
        return int(np.sum(entry.global_labels == cluster_id))


class AtlasBuilder:
    """Builds a unified cluster atlas from multiple slides.

    The builder aggregates features from multiple slides, performs global
    K-means clustering, and creates 2D PCA embeddings for visualization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for K-means.
    max_patches_per_slide : int
        Maximum patches to include per slide (subsampling for memory).

    Example
    -------
    >>> builder = AtlasBuilder(n_clusters=10)
    >>> builder.add_slide("slide1", features1, coords1)
    >>> builder.add_slide("slide2", features2, coords2)
    >>> atlas = builder.build()
    """

    def __init__(self, n_clusters: int = 10, max_patches_per_slide: int = 50000):
        self.n_clusters = n_clusters
        self.max_patches_per_slide = max_patches_per_slide
        self._slide_features: Dict[str, np.ndarray] = {}
        self._slide_coords: Dict[str, np.ndarray] = {}
        self._subsample_indices: Dict[str, Optional[np.ndarray]] = {}

    def add_slide(self, name: str, features: np.ndarray, coords: np.ndarray) -> int:
        """Add a slide's features to the atlas.

        Parameters
        ----------
        name : str
            Slide name (identifier).
        features : np.ndarray
            Feature vectors of shape (n_patches, feature_dim).
        coords : np.ndarray
            Patch coordinates of shape (n_patches, 2-3).

        Returns
        -------
        int
            Number of patches added (may be subsampled).
        """
        subsample_indices = None

        # Subsample if too many patches
        if len(features) > self.max_patches_per_slide:
            subsample_indices = np.random.choice(
                len(features), self.max_patches_per_slide, replace=False
            )
            subsample_indices.sort()  # Keep spatial ordering
            features = features[subsample_indices]
            coords = coords[subsample_indices]

        self._slide_features[name] = features
        self._slide_coords[name] = coords
        self._subsample_indices[name] = subsample_indices

        return len(features)

    def remove_slide(self, name: str) -> None:
        """Remove a slide from the builder."""
        self._slide_features.pop(name, None)
        self._slide_coords.pop(name, None)
        self._subsample_indices.pop(name, None)

    def clear(self) -> None:
        """Remove all slides from the builder."""
        self._slide_features.clear()
        self._slide_coords.clear()
        self._subsample_indices.clear()

    @property
    def slide_names(self) -> List[str]:
        """Get list of slide names added to the builder."""
        return list(self._slide_features.keys())

    @property
    def total_patches(self) -> int:
        """Get total number of patches across all slides."""
        return sum(len(f) for f in self._slide_features.values())

    def build(self, progress_callback: Optional[callable] = None) -> ClusterAtlas:
        """Build the atlas from all added slides.

        Parameters
        ----------
        progress_callback : callable, optional
            Function called with progress percentage (0-100).

        Returns
        -------
        ClusterAtlas
            The constructed cluster atlas.

        Raises
        ------
        ValueError
            If fewer than 2 slides have been added.
        RuntimeError
            If sklearn is not available.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for atlas building")

        if len(self._slide_features) < 2:
            raise ValueError("At least 2 slides are required to build an atlas")

        if progress_callback:
            progress_callback(5)

        # Concatenate all features
        all_features = []
        slide_indices = []
        slide_names = list(self._slide_features.keys())

        for i, name in enumerate(slide_names):
            features = self._slide_features[name]
            all_features.append(features)
            slide_indices.extend([i] * len(features))

        global_features = np.vstack(all_features)
        slide_indices = np.array(slide_indices)

        if progress_callback:
            progress_callback(20)

        # Global clustering
        print(f"DEBUG: Atlas clustering {len(global_features)} patches into {self.n_clusters} clusters")
        if len(global_features) > 100000:
            clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=min(10000, len(global_features))
            )
        else:
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )

        global_labels = clusterer.fit_predict(global_features)

        if progress_callback:
            progress_callback(60)

        # Global PCA for visualization
        print("DEBUG: Computing PCA for atlas visualization")
        pca = PCA(n_components=2)
        global_pca_coords = pca.fit_transform(global_features)

        if progress_callback:
            progress_callback(80)

        # Generate cluster colors
        palette = generate_palette(self.n_clusters)
        cluster_colors = [_hsl_to_qcolor(c) for c in palette]

        # Build per-slide entries
        entries: Dict[str, SlideAtlasEntry] = {}
        offset = 0
        for name in slide_names:
            n_patches = len(self._slide_features[name])
            entries[name] = SlideAtlasEntry(
                slide_name=name,
                features=self._slide_features[name],
                coords=self._slide_coords[name],
                global_labels=global_labels[offset:offset + n_patches],
                local_to_global=np.arange(offset, offset + n_patches)
            )
            offset += n_patches

        if progress_callback:
            progress_callback(100)

        return ClusterAtlas(
            entries=entries,
            global_features=global_features,
            global_labels=global_labels,
            global_pca_coords=global_pca_coords,
            n_clusters=self.n_clusters,
            cluster_colors=cluster_colors,
            slide_indices=slide_indices,
            slide_names=slide_names
        )

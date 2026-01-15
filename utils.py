"""
utils.py
========

Shared helper functions for the FoundationDetector GUI application.
These utilities provide colour palette generation, feature clustering,
coordinate transformations and radial ordering computations.  They are
designed to be lightweight and not depend on the GUI toolkit so that
they can be unit tested independently.

Functions
---------

generate_palette(n)
    Generate a list of visually distinct colours for `n` clusters.

cluster_features(features, n_clusters)
    Perform K‑means clustering on the provided feature vectors.

infer_slide_dims(coords, patch_size_lv0)
    Infer the approximate slide dimensions (width, height) at level 0
    from the coordinate grid and patch size.

radial_sweep_order(coords, click_point)
    Compute the order in which patches should be highlighted in a
    radial sweep animation from a click point.

The module will attempt to use scikit‑learn for clustering.  If
scikit‑learn is not installed, it will fall back to assigning random
clusters (useful for testing in constrained environments).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    # If sklearn isn't installed, we'll fall back to random labels.
    _SKLEARN_AVAILABLE = False


def generate_palette(n: int) -> List[str]:
    """Generate a list of distinct colours for `n` clusters.

    Colours are returned as CSS HSL strings (e.g., "hsl(210,70%,50%)")
    spaced evenly around the hue circle.  Saturation and lightness are
    fixed at 70% and 50% respectively to ensure good contrast on most
    backgrounds.

    Parameters
    ----------
    n : int
        Number of colours to generate.

    Returns
    -------
    List[str]
        A list of HSL strings of length `n`.

    Examples
    --------
    >>> generate_palette(3)
    ['hsl(0,70%,50%)', 'hsl(120,70%,50%)', 'hsl(240,70%,50%)']
    """
    if n <= 0:
        return []
    hues = np.linspace(0, 360, n, endpoint=False)
    return [f"hsl({int(h) % 360},70%,50%)" for h in hues]


def cluster_features(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster feature vectors into `n_clusters` using K‑means.

    Parameters
    ----------
    features : np.ndarray
        Array of shape `(n_samples, n_features)` containing feature
        vectors for each patch.
    n_clusters : int
        Desired number of clusters.

    Returns
    -------
    np.ndarray
        Array of shape `(n_samples,)` with integer cluster labels in
        the range `[0, n_clusters - 1]`.

    Notes
    -----
    If scikit‑learn is not available, this function will assign
    clusters randomly.  This behaviour is meant for environments
    without external dependencies; for real use, install scikit‑learn.
    """
    n_samples = features.shape[0]
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive")
    if n_samples == 0:
        return np.array([], dtype=int)
    if n_clusters > n_samples:
        # Cannot have more clusters than samples; cap to n_samples
        n_clusters = n_samples
    if _SKLEARN_AVAILABLE:
        # Use scikit‑learn's KMeans.  Explicitly set n_init to an integer
        # because not all versions of scikit‑learn support 'auto'.
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(features)
    else:
        # Fall back to random assignment
        rng = np.random.default_rng(seed=42)
        labels = rng.integers(0, n_clusters, size=n_samples)
    return labels.astype(int)


def infer_slide_dims(coords: np.ndarray, patch_size_lv0: int) -> Tuple[int, int]:
    """Infer slide dimensions from patch coordinates and patch size.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape `(n_patches, 2)` containing top‑left patch
        coordinates at level 0.
    patch_size_lv0 : int
        Patch size at level 0 magnification.

    Returns
    -------
    Tuple[int, int]
        Approximate width and height of the slide at level 0.

    Notes
    -----
    Some WSI formats (e.g., SVS) embed slide dimensions, but when
    reading only patch coordinates we may not know the full image size.
    This function assumes the slide is covered by a regular grid of
    patches with no gaps between them.  It computes the bounding box
    around all patches and adds one patch size to each dimension to
    estimate the full size.
    """
    if coords.size == 0:
        return (0, 0)
    xs = coords[:, 0]
    ys = coords[:, 1]
    min_x = float(xs.min())
    min_y = float(ys.min())
    max_x = float(xs.max())
    max_y = float(ys.max())
    width = int(max_x - min_x + patch_size_lv0)
    height = int(max_y - min_y + patch_size_lv0)
    return (width, height)


def normalize_to_scene(coords: np.ndarray, width: int, height: int, padding: int) -> np.ndarray:
    """Normalize 2D coordinates to a fixed scene size.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape `(n_points, 2)` with source coordinates.
    width : int
        Target scene width in pixels.
    height : int
        Target scene height in pixels.
    padding : int
        Padding applied on all sides.

    Returns
    -------
    np.ndarray
        Normalized coordinates of shape `(n_points, 2)`.
    """
    if coords.size == 0:
        return np.zeros((0, 2))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_range = max(float(x_max - x_min), 1e-9)
    y_range = max(float(y_max - y_min), 1e-9)
    scale = min((width - 2 * padding) / x_range, (height - 2 * padding) / y_range)
    x = (coords[:, 0] - x_min) * scale + padding
    y = (coords[:, 1] - y_min) * scale + padding
    return np.column_stack((x, y))


def radial_sweep_order(coords: np.ndarray, click_point: Tuple[float, float]) -> np.ndarray:
    """Compute indices of patches sorted by distance from a click point.

    Parameters
    ----------
    coords : np.ndarray
        Scaled coordinates of patches (shape `(n_patches, 2)`).
    click_point : Tuple[float, float]
        The (x, y) coordinate of the click in the same coordinate
        system as `coords`.

    Returns
    -------
    np.ndarray
        Indices of patches sorted by ascending Euclidean distance from
        the click point.  The returned array has shape `(n_patches,)`.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [0, 1]])
    >>> radial_sweep_order(coords, (0, 0))
    array([0, 1, 2])  # distances: 0, 1, 1
    """
    if coords.size == 0:
        return np.array([], dtype=int)
    click = np.array(click_point).reshape(1, 2)
    dists = np.linalg.norm(coords - click, axis=1)
    order = np.argsort(dists)
    return order.astype(int)

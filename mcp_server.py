"""Local MCP server exposing read-only FoundationDetector inspection tools."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import h5py

import app_state
import data_loader

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception:  # pragma: no cover - import guard
    class FastMCP:  # type: ignore[override]
        """Minimal fallback to keep local tests importable without mcp installed."""

        def __init__(self, _name: str) -> None:
            pass

        def tool(self):
            def _decorator(func):
                return func

            return _decorator

        def run(self) -> None:
            raise RuntimeError("MCP SDK is required to run the MCP server.")


mcp = FastMCP("foundational-detector-tools")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_root(root_dir: str) -> Path:
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root_dir: {root_dir}")
    return root


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    candidate = (root / rel_path).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError("Path outside root_dir")
    return candidate


def _resolve_h5_path(
    root: Path, slide_name: str, model: str, mag: str, patch_size: str
) -> Path:
    """Locate the H5 file for a given slide/model/mag/patch_size using data_loader."""
    slides = data_loader.parse_root_directory(str(root))
    if slide_name not in slides:
        raise FileNotFoundError(f"Slide '{slide_name}' not found under {root}")
    info = slides[slide_name]
    if model not in info.models:
        raise FileNotFoundError(f"Model '{model}' not found for slide '{slide_name}'")
    if mag not in info.models[model]:
        raise FileNotFoundError(
            f"Magnification '{mag}' not found for model '{model}' on slide '{slide_name}'"
        )
    if patch_size not in info.models[model][mag]:
        raise FileNotFoundError(
            f"Patch size '{patch_size}' not found for {model}/{mag} on slide '{slide_name}'"
        )
    h5_path = Path(info.models[model][mag][patch_size]).resolve()
    if not str(h5_path).startswith(str(root)):
        raise ValueError("Resolved H5 path is outside root_dir")
    return h5_path


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def _load_features_and_coords(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Auto-detect and load features, coords, patch_size, and file attributes.

    Returns
    -------
    features : np.ndarray  shape (n_patches, feature_dim), float32
    coords   : np.ndarray  shape (n_patches, 2), float32
    patch_size_lv0 : int
    attrs    : dict of H5 file-level attributes
    """
    with h5py.File(h5_path, "r") as f:
        features_ds = None
        coords_ds = None
        for name, dset in f.items():
            if isinstance(dset, h5py.Dataset) and len(dset.shape) == 2 and dset.shape[0] > 0:
                if dset.shape[1] > 10 and features_ds is None:
                    features_ds = dset
                elif dset.shape[1] in (2, 3) and coords_ds is None:
                    coords_ds = dset

        if features_ds is None or coords_ds is None:
            raise RuntimeError(f"Could not detect features or coords datasets in {h5_path}")

        features = features_ds[...].astype(np.float32)
        coords = coords_ds[...].astype(np.float32)[:, :2]

        # Resolve patch size from attributes or grid inference
        patch_size_lv0: Optional[int] = None
        for key in ("patch_size_level0", "patch_size_lv0", "patch_size"):
            if key in coords_ds.attrs:
                patch_size_lv0 = int(coords_ds.attrs[key])
                break
            if key in f.attrs:
                patch_size_lv0 = int(f.attrs[key])
                break
        if patch_size_lv0 is None:
            xs = np.sort(np.unique(coords[:, 0]))
            dxs = np.diff(xs)
            dxs = dxs[dxs > 1e-3]
            patch_size_lv0 = int(np.median(dxs)) if dxs.size > 0 else 256

        # Collect file-level attributes
        attrs: Dict[str, Any] = {}
        for key, val in f.attrs.items():
            if isinstance(val, bytes):
                attrs[key] = val.decode("utf-8", errors="replace")
            elif hasattr(val, "tolist"):
                attrs[key] = val.tolist()
            else:
                attrs[key] = val

    return features, coords, patch_size_lv0, attrs


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _run_clustering(features: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
    """Cluster features with K-means. Returns (labels, inertia).

    Uses full KMeans for small arrays (< 10k patches) for accuracy,
    MiniBatchKMeans otherwise for speed.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore

    n_clusters = min(n_clusters, len(features))
    if len(features) < 10_000:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    model.fit(features)
    return model.labels_, float(model.inertia_)


def _clustering_scores(
    features: np.ndarray, labels: np.ndarray, max_sample: int = 5_000
) -> Dict[str, float]:
    """Compute silhouette score and Davies-Bouldin index.

    Silhouette is O(n²) so is subsampled when n > max_sample.
    Davies-Bouldin is O(k²) and always uses the full array.
    """
    from sklearn.metrics import davies_bouldin_score, silhouette_score  # type: ignore

    n = len(features)
    if n > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_sample, replace=False)
        sil = float(silhouette_score(features[idx], labels[idx]))
    else:
        sil = float(silhouette_score(features, labels))

    dbi = float(davies_bouldin_score(features, labels))
    return {"silhouette_score": sil, "davies_bouldin_index": dbi}


# ---------------------------------------------------------------------------
# Existing tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_data(root_dir: str) -> Dict[str, Any]:
    """Enumerate discovered slides/models/mags/patch sizes under root_dir."""
    root = _resolve_root(root_dir)
    slides = data_loader.parse_root_directory(str(root))

    model_names = set()
    mags = set()
    patch_sizes = set()
    feature_files = 0

    for info in slides.values():
        model_names.update(info.models.keys())
        for model_value in info.models.values():
            mags.update(model_value.keys())
            for mag_value in model_value.values():
                patch_sizes.update(mag_value.keys())
                feature_files += len(mag_value)

    return {
        "slides": sorted(slides.keys()),
        "models": sorted(model_names),
        "magnifications": sorted(mags),
        "patch_sizes": sorted(patch_sizes),
        "count_summary": {
            "slides": len(slides),
            "feature_files": feature_files,
        },
    }


@mcp.tool()
def inspect_h5(root_dir: str, h5_path: str) -> Dict[str, Any]:
    """Summarize dataset structure and key attributes for a target HDF5 file."""
    root = _resolve_root(root_dir)
    target = _resolve_under_root(root, h5_path)
    if not target.exists() or not target.is_file():
        raise ValueError(f"HDF5 file not found: {h5_path}")

    datasets: List[Dict[str, Any]] = []
    attributes: Dict[str, Any] = {}

    with h5py.File(target, "r") as handle:
        def _collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(
                    {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )

        handle.visititems(_collect)
        for key, value in handle.attrs.items():
            if isinstance(value, bytes):
                attributes[key] = value.decode("utf-8", errors="replace")
            elif hasattr(value, "tolist"):
                attributes[key] = value.tolist()
            else:
                attributes[key] = value

    warnings: List[str] = []
    if not datasets:
        warnings.append("No datasets found in file")

    return {
        "datasets": datasets,
        "attributes": attributes,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Tier 1 tools
# ---------------------------------------------------------------------------

@mcp.tool()
def describe_slide(
    root_dir: str,
    slide_name: str,
    model: str,
    mag: str,
    patch_size: str,
) -> Dict[str, Any]:
    """Return enriched metadata for a specific slide/model/mag/patch_size combination.

    Reports patch count, feature dimension, coordinate ranges, inferred patch
    size in pixels, and any H5 file-level attributes.  Use list_data first to
    discover valid slide_name/model/mag/patch_size values.
    """
    root = _resolve_root(root_dir)
    try:
        h5_path = _resolve_h5_path(root, slide_name, model, mag, patch_size)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    features, coords, patch_size_lv0, attrs = _load_features_and_coords(h5_path)

    warnings: List[str] = []
    if coords.shape[1] != 2:
        warnings.append(f"Coordinate array has {coords.shape[1]} columns; using first 2")

    return {
        "h5_path": str(h5_path.relative_to(root)),
        "patch_count": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "patch_size_px": patch_size_lv0,
        "coord_range": {
            "x_min": int(coords[:, 0].min()),
            "y_min": int(coords[:, 1].min()),
            "x_max": int(coords[:, 0].max()),
            "y_max": int(coords[:, 1].max()),
        },
        "h5_attributes": attrs,
        "warnings": warnings,
    }


@mcp.tool()
def compare_models(
    root_dir: str,
    slide_name: str,
    model_a: str,
    model_b: str,
    mag: str,
    patch_size: str,
    n_clusters: int = 8,
) -> Dict[str, Any]:
    """Compare K-means clustering quality between two foundation models on the same slide.

    Runs independent K-means clustering on each model's features and reports
    silhouette score, Davies-Bouldin index, and Adjusted Rand Index measuring
    how similarly the two models partition the tissue.
    """
    root = _resolve_root(root_dir)
    warnings: List[str] = []

    try:
        h5_a = _resolve_h5_path(root, slide_name, model_a, mag, patch_size)
        h5_b = _resolve_h5_path(root, slide_name, model_b, mag, patch_size)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    feat_a, _, _, _ = _load_features_and_coords(h5_a)
    feat_b, _, _, _ = _load_features_and_coords(h5_b)

    patch_count_match = len(feat_a) == len(feat_b)
    if not patch_count_match:
        warnings.append(
            f"Patch count mismatch: {model_a}={len(feat_a)}, {model_b}={len(feat_b)}; "
            "ARI is unreliable"
        )

    labels_a, inertia_a = _run_clustering(feat_a, n_clusters)
    labels_b, inertia_b = _run_clustering(feat_b, n_clusters)

    scores_a = _clustering_scores(feat_a, labels_a)
    scores_b = _clustering_scores(feat_b, labels_b)

    ari: Optional[float] = None
    ari_interpretation = "skipped (patch count mismatch)"
    if patch_count_match:
        from sklearn.metrics import adjusted_rand_score  # type: ignore

        ari = float(adjusted_rand_score(labels_a, labels_b))
        if ari < 0.2:
            ari_interpretation = "low agreement"
        elif ari < 0.5:
            ari_interpretation = "moderate agreement"
        elif ari < 0.8:
            ari_interpretation = "high agreement"
        else:
            ari_interpretation = "very high agreement"

    return {
        "slide_name": slide_name,
        "model_a": {
            "name": model_a,
            "feature_dim": int(feat_a.shape[1]),
            "patch_count": int(len(feat_a)),
            "n_clusters": n_clusters,
            "inertia": inertia_a,
            **scores_a,
        },
        "model_b": {
            "name": model_b,
            "feature_dim": int(feat_b.shape[1]),
            "patch_count": int(len(feat_b)),
            "n_clusters": n_clusters,
            "inertia": inertia_b,
            **scores_b,
        },
        "overlap": {
            "adjusted_rand_index": ari,
            "interpretation": ari_interpretation,
        },
        "patch_count_match": patch_count_match,
        "warnings": warnings,
    }


@mcp.tool()
def rank_models_by_separability(
    root_dir: str,
    slide_name: str,
    mag: str,
    patch_size: str,
    n_clusters: int = 8,
) -> Dict[str, Any]:
    """Rank all available foundation models for a slide by cluster separability.

    Iterates over every model available for the given slide/mag/patch_size,
    runs K-means clustering, and ranks models by silhouette score (higher is
    better).  Models with missing files or mismatched patch counts are skipped
    with a warning.
    """
    root = _resolve_root(root_dir)
    slides = data_loader.parse_root_directory(str(root))

    if slide_name not in slides:
        return {"error": f"Slide '{slide_name}' not found under {root}"}

    info = slides[slide_name]
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []
    reference_patch_count: Optional[int] = None

    for model in sorted(info.models.keys()):
        if mag not in info.models[model]:
            warnings.append(f"'{model}' skipped: magnification '{mag}' not available")
            continue
        if patch_size not in info.models[model][mag]:
            warnings.append(f"'{model}' skipped: patch_size '{patch_size}' not available")
            continue

        h5_path = Path(info.models[model][mag][patch_size])
        try:
            features, _, _, _ = _load_features_and_coords(h5_path)
        except Exception as exc:
            warnings.append(f"'{model}' skipped: could not load features ({exc})")
            continue

        if reference_patch_count is None:
            reference_patch_count = len(features)
        elif len(features) != reference_patch_count:
            warnings.append(
                f"'{model}' skipped: patch count {len(features)} != {reference_patch_count}"
            )
            continue

        labels, _ = _run_clustering(features, n_clusters)
        scores = _clustering_scores(features, labels)
        results.append(
            {
                "model": model,
                "feature_dim": int(features.shape[1]),
                **scores,
            }
        )

    results.sort(key=lambda x: x["silhouette_score"], reverse=True)
    for i, entry in enumerate(results):
        entry["rank"] = i + 1

    return {
        "slide_name": slide_name,
        "ranking": results,
        "n_models_evaluated": len(results),
        "n_clusters_used": n_clusters,
        "warnings": warnings,
    }


@mcp.tool()
def compute_elbow_analysis(
    root_dir: str,
    slide_name: str,
    model: str,
    mag: str,
    patch_size: str,
    max_k: int = 15,
) -> Dict[str, Any]:
    """Suggest an optimal cluster count using the elbow method.

    Runs MiniBatchKMeans for k=2..max_k (capped at 20) and identifies the
    sharpest bend in the inertia curve using a second-derivative heuristic.
    Returns the full curve and a recommended k value.
    """
    root = _resolve_root(root_dir)
    warnings: List[str] = []

    if max_k > 20:
        warnings.append(f"max_k capped at 20 (requested {max_k})")
        max_k = 20

    try:
        h5_path = _resolve_h5_path(root, slide_name, model, mag, patch_size)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    features, _, _, _ = _load_features_and_coords(h5_path)

    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    curve: List[Dict[str, Any]] = []
    for k in range(2, max_k + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        km.fit(features)
        curve.append({"k": k, "inertia": float(km.inertia_)})

    # Second-derivative knee-point detection
    inertias = [p["inertia"] for p in curve]
    recommended_k = 2  # fallback

    if len(inertias) >= 3:
        d1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        d2 = [d1[i] - d1[i + 1] for i in range(len(d1) - 1)]
        elbow_idx = d2.index(max(d2))
        recommended_k = curve[elbow_idx + 1]["k"]

        # Warn if the curve has no meaningful elbow
        d2_range = max(d2) - min(d2) if len(d2) > 1 else 0
        if d2_range < 0.01 * inertias[0]:
            warnings.append(
                "No clear elbow detected; the inertia curve is nearly linear. "
                "Consider trying a different model or magnification."
            )

    return {
        "slide_name": slide_name,
        "model": model,
        "elbow_curve": curve,
        "recommended_k": recommended_k,
        "method": "second_derivative",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Tier 2 tools — require live GUI state via app_state registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Private stat helpers — signature: (state: AppState, indices: np.ndarray) -> dict
# ---------------------------------------------------------------------------

def _stat_spread(state, indices):
    features = state.features[indices]
    centroid = features.mean(axis=0)
    dists = np.linalg.norm(features - centroid, axis=1)
    result = {
        "feature_spread": {
            "mean": float(dists.mean()), "std": float(dists.std()),
            "min": float(dists.min()), "max": float(dists.max()),
        }
    }
    if state.coords_lv0 is not None:
        coords = state.coords_lv0[indices]
        result["spatial_spread"] = {
            "std_x": float(coords[:, 0].std()), "std_y": float(coords[:, 1].std()),
            "mean_x": float(coords[:, 0].mean()), "mean_y": float(coords[:, 1].mean()),
        }
    return result


def _stat_area_covered(state, indices):
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 not available"}
    patch_area = float(state.patch_size_lv0) ** 2
    return {
        "patch_count": len(indices),
        "patch_size_px": float(state.patch_size_lv0),
        "area_px2": int(len(indices) * patch_area),
    }


def _stat_area_bbox(state, indices):
    if state.coords_lv0 is None:
        return {"error": "coords_lv0 not available"}
    coords = state.coords_lv0[indices]
    pad = float(state.patch_size_lv0 or 0)
    w = float(coords[:, 0].max() - coords[:, 0].min()) + pad
    h = float(coords[:, 1].max() - coords[:, 1].min()) + pad
    return {
        "width_px": int(w), "height_px": int(h), "area_px2": int(w * h),
        "bbox": {
            "x_min": int(coords[:, 0].min()), "y_min": int(coords[:, 1].min()),
            "x_max": int(coords[:, 0].max()), "y_max": int(coords[:, 1].max()),
        },
    }


def _stat_area_hull(state, indices):
    if state.coords_lv0 is None:
        return {"error": "coords_lv0 not available"}
    try:
        from shapely.geometry import MultiPoint  # type: ignore
        hull = MultiPoint(state.coords_lv0[indices].tolist()).convex_hull
        return {"area_px2": int(hull.area), "perimeter_px": int(hull.length)}
    except Exception as exc:
        return {"error": f"shapely error: {exc}"}


def _stat_homogeneity(state, indices):
    per_dim_std = state.features[indices].std(axis=0)
    return {"intra_variance": float(per_dim_std.mean()), "note": "lower = more homogeneous"}


def _stat_global_distance(state, indices):
    region_centroid = state.features[indices].mean(axis=0)
    global_centroid = state.features.mean(axis=0)
    return {"l2_distance": float(np.linalg.norm(region_centroid - global_centroid))}


def _stat_pca_extent(state, indices):
    if state.embedding_2d is None:
        return {"error": "PCA embedding not available"}
    emb = state.embedding_2d[indices]
    return {
        "pc1_min": float(emb[:, 0].min()), "pc1_max": float(emb[:, 0].max()),
        "pc2_min": float(emb[:, 1].min()), "pc2_max": float(emb[:, 1].max()),
    }


def _stat_top_dims(state, indices):
    per_dim_std = state.features[indices].std(axis=0)
    top = np.argsort(per_dim_std)[::-1][:5]
    return {"top_5": [{"dim": int(d), "std": float(per_dim_std[d])} for d in top]}


def _stat_area_mm2(state, indices):
    if state.mpp is None:
        return {"error": "MPP not available for this slide — cannot convert to mm²"}
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 not available"}
    area_px2 = len(indices) * float(state.patch_size_lv0) ** 2
    px_to_mm = state.mpp * 1e-3   # µm/px → mm/px
    return {
        "patch_count": len(indices),
        "area_mm2": round(area_px2 * px_to_mm ** 2, 6),
        "mpp_used": state.mpp,
    }


def _stat_width_mm(state, indices):
    if state.coords_lv0 is None:
        return {"error": "coords_lv0 not available"}
    if state.mpp is None:
        return {"error": "MPP not available"}
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 not available"}
    pad = float(state.patch_size_lv0)
    coords = state.coords_lv0[indices]
    width_px = float(coords[:, 0].max() - coords[:, 0].min()) + pad
    height_px = float(coords[:, 1].max() - coords[:, 1].min()) + pad
    px_to_mm = state.mpp * 1e-3
    return {
        "width_mm": round(width_px * px_to_mm, 4),
        "height_mm": round(height_px * px_to_mm, 4),
    }


def _compute_separation(state, cluster_id: int) -> Dict[str, Any]:
    """Pairwise L2 distances from this cluster's centroid to all others."""
    if state.cluster_centroids is None:
        return {"error": "cluster_centroids not available"}
    centroid = state.cluster_centroids[cluster_id]
    other_ids = [i for i in range(len(state.cluster_centroids)) if i != cluster_id]
    distances = [
        {"cluster_id": int(oid), "l2_distance": float(np.linalg.norm(centroid - state.cluster_centroids[oid]))}
        for oid in other_ids
    ]
    distances.sort(key=lambda x: x["l2_distance"])
    return {"distances_to_other_centroids": distances}


def _compute_discriminating_dims(state, cluster_id: int) -> Dict[str, Any]:
    """Top 10 dims where this centroid deviates most from the mean of all others."""
    if state.cluster_centroids is None:
        return {"error": "cluster_centroids not available"}
    centroid = state.cluster_centroids[cluster_id]
    other_ids = [i for i in range(len(state.cluster_centroids)) if i != cluster_id]
    if not other_ids:
        return {"error": "No other clusters to compare"}
    mean_other = state.cluster_centroids[other_ids].mean(axis=0)
    delta = np.abs(centroid - mean_other)
    top = np.argsort(delta)[::-1][:10]
    return {"top_10": [{"dim": int(d), "abs_delta": float(delta[d])} for d in top]}


# Dispatch registries
_REGION_STATS = {
    "spread": _stat_spread,
    "area_covered": _stat_area_covered,
    "area_bbox": _stat_area_bbox,
    "area_hull": _stat_area_hull,
    "homogeneity": _stat_homogeneity,
    "global_distance": _stat_global_distance,
    "pca_extent": _stat_pca_extent,
    "top_dims": _stat_top_dims,
    "area_mm2": _stat_area_mm2,
    "width_mm": _stat_width_mm,
}

_CLUSTER_STATS = {
    "spread": _stat_spread,
    "area_covered": _stat_area_covered,
    "area_bbox": _stat_area_bbox,
    "area_hull": _stat_area_hull,
    "homogeneity": _stat_homogeneity,
    "pca_extent": _stat_pca_extent,
    "patch_count": lambda state, idx: {"patch_count": len(idx)},
    "area_mm2": _stat_area_mm2,
    "width_mm": _stat_width_mm,
}

RegionMetric = Literal[
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

ClusterMetric = Literal[
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


@mcp.tool()
def list_labeled_regions() -> Dict[str, Any]:
    """List all user-created labeled region annotations from the loaded slide.

    Returns each region's name, color, patch count, source mode (kmeans or
    local), kmeans cluster id, and bounding box in level-0 pixel coordinates.
    Requires a slide to be loaded in the GUI.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}

    regions = state.labeled_regions
    if not regions:
        return {"labeled_regions": [], "total_count": 0}

    result: List[Dict[str, Any]] = []
    for rid, r in regions.items():
        bbox: Dict[str, Any] = {}
        if state.coords_lv0 is not None and r.patch_indices:
            pts = state.coords_lv0[r.patch_indices]
            bbox = {
                "x_min": int(pts[:, 0].min()),
                "y_min": int(pts[:, 1].min()),
                "x_max": int(pts[:, 0].max()),
                "y_max": int(pts[:, 1].max()),
            }
        result.append(
            {
                "region_id": rid,
                "name": r.name,
                "color_hex": r.color_hex,
                "patch_count": len(r.patch_indices),
                "source_mode": r.source_mode,
                "kmeans_cluster": r.kmeans_cluster,
                "bounding_box_lv0": bbox,
            }
        )

    return {"labeled_regions": result, "total_count": len(result)}


@mcp.tool()
def compute_region_stats(region_id: int, metrics: List[RegionMetric]) -> Dict[str, Any]:
    """Compute selected statistics for a labeled region.

    Pass only the metrics you need to keep LLM context concise.
    Available metrics: spread, area_covered, area_bbox, area_hull,
    homogeneity, global_distance, pca_extent, top_dims, area_mm2, width_mm.
    Unrecognised metric names are collected in unknown_metrics — no crash.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}

    region = state.labeled_regions[region_id]
    idx = np.array(region.patch_indices, dtype=int)
    if len(idx) == 0:
        return {"error": "Region has no patches"}

    computed: Dict[str, Any] = {}
    unknown: List[str] = []
    for m in metrics:
        if m in _REGION_STATS:
            computed[m] = _REGION_STATS[m](state, idx)
        else:
            unknown.append(m)

    return {
        "region_id": region_id,
        "name": region.name,
        "computed": computed,
        "unknown_metrics": unknown,
    }


@mcp.tool()
def compute_region_geometry_stats(region_id: int) -> Dict[str, Any]:
    """Compute a geometry-focused metric bundle for one labeled region."""
    return compute_region_stats(
        region_id=region_id,
        metrics=["area_covered", "area_bbox", "area_hull", "area_mm2", "width_mm"],
    )


@mcp.tool()
def compute_region_feature_stats(region_id: int) -> Dict[str, Any]:
    """Compute a feature-focused metric bundle for one labeled region."""
    return compute_region_stats(
        region_id=region_id,
        metrics=["spread", "homogeneity", "global_distance", "pca_extent", "top_dims"],
    )


@mcp.tool()
def find_similar_patches(
    region_id: int,
    top_k: int = 20,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """Find patches most similar to a labeled region's feature centroid.

    Computes the mean feature vector for the region, then ranks all patches
    NOT already in the region by cosine or euclidean distance.  Returns the
    top_k closest patches with patch index, distance, cluster id, and
    level-0 coordinates.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}

    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}

    if metric not in ("cosine", "euclidean"):
        return {"error": "metric must be 'cosine' or 'euclidean'"}

    region = state.labeled_regions[region_id]
    region_set = set(region.patch_indices)
    idx_in = np.array(region.patch_indices, dtype=int)
    if len(idx_in) == 0:
        return {"error": "Region has no patches"}

    region_centroid = state.features[idx_in].mean(axis=0, keepdims=True)

    # Candidate patches (not in region)
    all_idx = np.arange(len(state.features))
    candidate_mask = ~np.isin(all_idx, idx_in)
    candidate_idx = all_idx[candidate_mask]

    if len(candidate_idx) == 0:
        return {"similar_patches": [], "metric": metric}

    from sklearn.metrics import pairwise_distances  # type: ignore

    candidate_features = state.features[candidate_idx]
    distances = pairwise_distances(candidate_features, region_centroid, metric=metric).ravel()

    top_n = min(top_k, len(candidate_idx))
    order = np.argsort(distances)[:top_n]

    patches: List[Dict[str, Any]] = []
    for rank_i, ci in enumerate(order):
        gi = int(candidate_idx[ci])
        entry: Dict[str, Any] = {
            "rank": rank_i + 1,
            "patch_index": gi,
            "distance": float(distances[ci]),
            "cluster_id": int(state.cluster_labels[gi]) if state.cluster_labels is not None else None,
        }
        if state.coords_lv0 is not None:
            entry["coords_lv0"] = {
                "x": int(state.coords_lv0[gi, 0]),
                "y": int(state.coords_lv0[gi, 1]),
            }
        patches.append(entry)

    return {
        "region_id": region_id,
        "region_name": region.name,
        "metric": metric,
        "top_k": top_n,
        "similar_patches": patches,
    }


@mcp.tool()
def compute_cluster_stats(cluster_id: int, metrics: List[ClusterMetric]) -> Dict[str, Any]:
    """Compute selected statistics for a K-means cluster.

    Pass only the metrics you need to keep LLM context concise.
    Available metrics: spread, area_covered, area_bbox, area_hull, homogeneity,
    pca_extent, patch_count, area_mm2, width_mm, separation, discriminating_dims.
    Unrecognised metric names are collected in unknown_metrics — no crash.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_labels is None:
        return {"error": "Clustering not available"}

    mask = state.cluster_labels == cluster_id
    if not mask.any():
        return {"error": f"Cluster {cluster_id} does not exist"}
    idx = np.where(mask)[0]

    computed: Dict[str, Any] = {}
    unknown: List[str] = []
    for m in metrics:
        if m in _CLUSTER_STATS:
            computed[m] = _CLUSTER_STATS[m](state, idx)
        elif m == "separation":
            computed[m] = _compute_separation(state, cluster_id)
        elif m == "discriminating_dims":
            computed[m] = _compute_discriminating_dims(state, cluster_id)
        else:
            unknown.append(m)

    return {
        "cluster_id": cluster_id,
        "computed": computed,
        "unknown_metrics": unknown,
    }


@mcp.tool()
def compute_cluster_geometry_stats(cluster_id: int) -> Dict[str, Any]:
    """Compute a geometry-focused metric bundle for one K-means cluster."""
    return compute_cluster_stats(
        cluster_id=cluster_id,
        metrics=["patch_count", "area_covered", "area_bbox", "area_hull", "area_mm2", "width_mm"],
    )


@mcp.tool()
def compute_cluster_feature_stats(cluster_id: int) -> Dict[str, Any]:
    """Compute a feature-focused metric bundle for one K-means cluster."""
    return compute_cluster_stats(
        cluster_id=cluster_id,
        metrics=["spread", "homogeneity", "pca_extent", "separation", "discriminating_dims"],
    )


@mcp.tool()
def get_boundary_patches(top_k: int = 30) -> Dict[str, Any]:
    """Identify patches that sit near cluster decision boundaries.

    For each patch computes the distance to its assigned centroid and to the
    nearest OTHER centroid.  Patches with the smallest gap (assigned_dist -
    nearest_other_dist) are most uncertain / boundary-like.  Returns top_k
    such patches with patch index, assigned cluster, nearest other cluster,
    gap, and level-0 coordinates.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_labels is None or state.cluster_centroids is None:
        return {"error": "Clustering not available"}

    centroids = state.cluster_centroids
    labels = state.cluster_labels
    features = state.features
    n_patches = len(features)

    # Compute distances from every patch to every centroid
    from sklearn.metrics import pairwise_distances  # type: ignore

    all_dists = pairwise_distances(features, centroids, metric="euclidean")  # (n, k)

    assigned_dist = all_dists[np.arange(n_patches), labels].astype(float)

    # For nearest OTHER centroid: mask assigned, then take min
    temp = all_dists.copy()
    temp[np.arange(n_patches), labels] = np.inf
    nearest_other_idx = np.argmin(temp, axis=1)
    nearest_other_dist = temp[np.arange(n_patches), nearest_other_idx].astype(float)

    gap = nearest_other_dist - assigned_dist
    order = np.argsort(gap)[: min(top_k, n_patches)]

    patches: List[Dict[str, Any]] = []
    for pi in order:
        entry: Dict[str, Any] = {
            "patch_index": int(pi),
            "assigned_cluster": int(labels[pi]),
            "nearest_other_cluster": int(nearest_other_idx[pi]),
            "assigned_dist": float(assigned_dist[pi]),
            "nearest_other_dist": float(nearest_other_dist[pi]),
            "gap": float(gap[pi]),
        }
        if state.coords_lv0 is not None:
            entry["coords_lv0"] = {
                "x": int(state.coords_lv0[pi, 0]),
                "y": int(state.coords_lv0[pi, 1]),
            }
        patches.append(entry)

    return {
        "top_k": len(patches),
        "boundary_patches": patches,
        "note": "Sorted by gap (assigned_dist - nearest_other_dist); smaller gap = more uncertain",
    }


@mcp.tool()
def get_pca_info() -> Dict[str, Any]:
    """Return PCA embedding metadata for the currently loaded slide.

    Reports the number of components (always 2), original feature dimension,
    per-component explained variance ratio, cumulative explained variance,
    and a human-readable interpretation.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.embedding_2d is None:
        return {"error": "PCA embedding not available"}

    ratio = state.pca_explained_variance_ratio or []
    cumulative = float(sum(ratio)) if ratio else 0.0

    note = (
        f"The 2D PCA scatter captures {cumulative * 100:.1f}% of feature variance. "
        + (
            "The embedding is fairly representative."
            if cumulative > 0.5
            else "A large fraction of variance is not visible in the scatter view."
        )
    )

    return {
        "n_components": 2,
        "feature_dim": int(state.features.shape[1]),
        "pca_explained_variance_ratio": [float(v) for v in ratio],
        "cumulative_explained_variance": cumulative,
        "note": note,
    }


@mcp.tool()
def compare_selected_clusters(metrics: List[ClusterMetric]) -> Dict[str, Any]:
    """Run compute_cluster_stats for every currently selected cluster in one call.

    Returns a list of per-cluster results sorted by the first metric requested.
    Always includes patch_count. Use selected_cluster_ids to know which clusters
    were included. Useful metrics: patch_count, area_mm2, width_mm, area_bbox,
    area_hull, spread, separation, discriminating_dims, homogeneity, pca_extent.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_labels is None:
        return {"error": "Clustering not available"}
    if not state.selected_clusters:
        return {"error": "No clusters are currently selected"}

    results: List[Dict[str, Any]] = []
    for cluster_id in sorted(state.selected_clusters):
        mask = state.cluster_labels == cluster_id
        if not mask.any():
            continue
        idx = np.where(mask)[0]

        computed: Dict[str, Any] = {}
        # Always include patch_count
        computed["patch_count"] = len(idx)
        for m in metrics:
            if m == "patch_count":
                continue  # already included
            if m in _CLUSTER_STATS:
                computed[m] = _CLUSTER_STATS[m](state, idx)
            elif m == "separation":
                computed[m] = _compute_separation(state, cluster_id)
            elif m == "discriminating_dims":
                computed[m] = _compute_discriminating_dims(state, cluster_id)

        results.append({"cluster_id": int(cluster_id), "computed": computed})

    # Sort by first metric (descending) when value is numeric
    if metrics:
        first_metric = metrics[0]

        def _sort_key(entry: Dict[str, Any]) -> float:
            val = entry["computed"].get(first_metric)
            if isinstance(val, dict):
                # e.g. patch_count returns {"patch_count": N}
                val = val.get(first_metric) or val.get(next(iter(val), ""), 0)
            return float(val) if isinstance(val, (int, float)) else 0.0

        results.sort(key=_sort_key, reverse=True)

    return {
        "selected_cluster_ids": sorted(int(c) for c in state.selected_clusters),
        "cluster_count": len(results),
        "results": results,
    }


@mcp.tool()
def atlas_cluster_representation(
    normalize: bool = False,
    include_entropy: bool = False,
    selected_only: bool = False,
) -> Dict[str, Any]:
    """Report how each global atlas cluster is distributed across slides.

    Returns per-cluster: total patch count, which slides it appears in, and the
    patch count per slide. Also returns a top-represented list (most slides) and
    single_slide_only list (clusters that appear in exactly one slide).
    Requires an atlas to have been built in the GUI.

    Parameters
    ----------
    normalize : bool
        When True, adds slide_prevalence showing each cluster's fraction of that
        slide's total patches.
    include_entropy : bool
        When True, adds distribution_entropy (Shannon entropy over slide
        fractions) — high = evenly distributed, low = concentrated in one slide.
    selected_only : bool
        When True, restricts output to clusters in state.selected_clusters.
    """
    state = app_state.get()
    if state.atlas_state is None:
        return {"error": "No atlas has been built yet"}

    atlas = state.atlas_state
    n_slides = len(atlas.slide_names)

    # Total patches per slide (for normalization)
    total_per_slide = [int(np.sum(atlas.slide_indices == si)) for si in range(n_slides)]

    cluster_ids = range(atlas.n_clusters)
    if selected_only and state.selected_clusters:
        cluster_ids = [c for c in cluster_ids if c in state.selected_clusters]

    clusters: List[Dict[str, Any]] = []
    for cluster_id in cluster_ids:
        cluster_mask = atlas.global_labels == cluster_id
        total_patches = int(cluster_mask.sum())

        slide_dist: Dict[str, int] = {}
        slide_counts: List[int] = []
        for si, sname in enumerate(atlas.slide_names):
            cnt = int(np.sum(cluster_mask & (atlas.slide_indices == si)))
            if cnt > 0:
                slide_dist[sname] = cnt
            slide_counts.append(cnt)

        entry: Dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "total_patches": total_patches,
            "slide_count": len(slide_dist),
            "slide_distribution": slide_dist,
        }

        if normalize:
            prevalence: Dict[str, float] = {}
            for si, sname in enumerate(atlas.slide_names):
                if total_per_slide[si] > 0:
                    prevalence[sname] = round(slide_counts[si] / total_per_slide[si], 6)
            entry["slide_prevalence"] = prevalence

        if include_entropy:
            nonzero = [c for c in slide_counts if c > 0]
            if len(nonzero) > 1:
                total = sum(nonzero)
                probs = [c / total for c in nonzero]
                entropy = -sum(p * np.log(p) for p in probs)
            else:
                entropy = 0.0
            entry["distribution_entropy"] = round(float(entropy), 6)

        clusters.append(entry)

    clusters.sort(key=lambda x: (x["slide_count"], x["total_patches"]), reverse=True)

    most_represented = [
        {
            "cluster_id": c["cluster_id"],
            "slide_count": c["slide_count"],
            "total_patches": c["total_patches"],
        }
        for c in clusters[:3]
    ]

    single_slide_only = [
        {
            "cluster_id": c["cluster_id"],
            "only_in_slide": next(iter(c["slide_distribution"])),
            "patch_count": c["total_patches"],
        }
        for c in clusters
        if c["slide_count"] == 1
    ]

    return {
        "n_clusters": atlas.n_clusters,
        "slide_names": list(atlas.slide_names),
        "clusters": clusters,
        "most_represented": most_represented,
        "single_slide_only": single_slide_only,
    }


@mcp.tool()
def rank_models_by_selected_cluster_separability(
    mag: Optional[str] = None,
    patch_size: Optional[str] = None,
    cluster_ids: Optional[List[int]] = None,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Rank all available foundation models by how well their features separate
    the currently selected K-means clusters.

    Unlike rank_models_by_separability (which re-clusters), this tool uses the
    existing cluster labels as ground truth and scores each foundation model's
    features on how well they discriminate between the selected clusters specifically.
    Loads feature files from root_dir for the current slide_name.

    Parameters
    ----------
    mag : str, optional
        Magnification level to evaluate (e.g. "20x"). Defaults to currently
        loaded magnification.
    patch_size : str, optional
        Patch size to evaluate (e.g. "256px"). Defaults to currently loaded
        patch size.
    cluster_ids : list of int, optional
        Override selected_clusters with a specific subset (e.g. [3, 7] for a
        pairwise comparison).  Defaults to state.selected_clusters.
    metric : str
        Distance metric for silhouette score: "euclidean" (default) or "cosine".
    """
    from sklearn.metrics import silhouette_score  # type: ignore

    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_labels is None:
        return {"error": "Clustering not available"}
    if state.slide_name is None or state.root_dir is None:
        return {"error": "slide_name or root_dir not set in app state"}
    if metric not in ("euclidean", "cosine"):
        return {"error": "metric must be 'euclidean' or 'cosine'"}

    mag = mag or state.magnification
    patch_size = patch_size or state.patch_size
    if not mag or not patch_size:
        return {"error": "mag and patch_size are required (or load a slide first)"}

    # Determine which clusters to compare
    target_clusters: set
    if cluster_ids is not None:
        target_clusters = set(cluster_ids)
    else:
        target_clusters = state.selected_clusters

    if not target_clusters:
        return {"error": "No clusters selected or provided via cluster_ids"}
    if len(target_clusters) < 2:
        return {"error": "At least 2 clusters are required to compute separability"}

    # Validate cluster IDs exist
    existing = set(np.unique(state.cluster_labels).tolist())
    invalid = target_clusters - existing
    if invalid:
        return {"error": f"Cluster IDs not found in current labels: {sorted(invalid)}"}

    mask = np.isin(state.cluster_labels, list(target_clusters))
    sub_labels = state.cluster_labels[mask]

    root = _resolve_root(state.root_dir)
    slides = data_loader.parse_root_directory(str(root))

    if state.slide_name not in slides:
        return {"error": f"Slide '{state.slide_name}' not found under root_dir"}

    info = slides[state.slide_name]
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    for model in sorted(info.models.keys()):
        if mag not in info.models[model]:
            warnings.append(f"'{model}' skipped: magnification '{mag}' not available")
            continue
        if patch_size not in info.models[model][mag]:
            warnings.append(f"'{model}' skipped: patch_size '{patch_size}' not available")
            continue

        h5_path = Path(info.models[model][mag][patch_size])
        try:
            features, _, _, _ = _load_features_and_coords(h5_path)
        except Exception as exc:
            warnings.append(f"'{model}' skipped: could not load features ({exc})")
            continue

        if len(features) != len(state.cluster_labels):
            warnings.append(
                f"'{model}' skipped: patch count {len(features)} != "
                f"{len(state.cluster_labels)} (current slide)"
            )
            continue

        sub_features = features[mask]

        # Subsample for performance
        n = len(sub_features)
        if n > 5000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, 5000, replace=False)
            sf = sub_features[sample_idx]
            sl = sub_labels[sample_idx]
        else:
            sf, sl = sub_features, sub_labels

        try:
            sil = float(silhouette_score(sf, sl, metric=metric))
        except Exception as exc:
            warnings.append(f"'{model}' skipped: silhouette failed ({exc})")
            continue

        results.append({"model": model, "feature_dim": int(features.shape[1]), "silhouette_score": sil})

    results.sort(key=lambda x: x["silhouette_score"], reverse=True)
    for i, entry in enumerate(results):
        entry["rank"] = i + 1

    return {
        "slide_name": state.slide_name,
        "target_cluster_ids": sorted(int(c) for c in target_clusters),
        "metric": metric,
        "ranking": results,
        "n_models_evaluated": len(results),
        "warnings": warnings,
    }


@mcp.tool()
def rank_models_by_labeled_region_separability(
    region_ids: List[int],
    mag: Optional[str] = None,
    patch_size: Optional[str] = None,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Rank all available foundation models by how well their features separate
    specific labeled regions.

    Uses the patch membership of each labeled region as ground-truth class labels
    and computes silhouette score for each foundation model's features.
    Only patches belonging to the requested regions are scored.

    Parameters
    ----------
    region_ids : list of int
        IDs of at least 2 labeled regions to compare (use list_labeled_regions).
    mag : str, optional
        Magnification (e.g. "20x"). Defaults to currently loaded magnification.
    patch_size : str, optional
        Patch size (e.g. "256px"). Defaults to currently loaded patch size.
    metric : str
        "euclidean" (default) or "cosine".
    """
    from sklearn.metrics import silhouette_score  # type: ignore

    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.slide_name is None or state.root_dir is None:
        return {"error": "slide_name or root_dir not set in app state"}
    if metric not in ("euclidean", "cosine"):
        return {"error": "metric must be 'euclidean' or 'cosine'"}
    if len(region_ids) < 2:
        return {"error": "At least 2 region IDs are required"}

    # Validate all region IDs exist
    missing = [rid for rid in region_ids if rid not in state.labeled_regions]
    if missing:
        return {"error": f"Region IDs not found: {missing}. Use list_labeled_regions to see available regions."}

    mag = mag or state.magnification
    patch_size = patch_size or state.patch_size
    if not mag or not patch_size:
        return {"error": "mag and patch_size are required (or load a slide first)"}

    # Build ground-truth label array from region membership
    sub_patch_idx: List[int] = []
    sub_labels_list: List[int] = []
    for label_int, rid in enumerate(region_ids):
        for pidx in state.labeled_regions[rid].patch_indices:
            sub_patch_idx.append(pidx)
            sub_labels_list.append(label_int)

    sub_patch_idx_arr = np.array(sub_patch_idx, dtype=int)
    sub_labels = np.array(sub_labels_list, dtype=int)

    if len(sub_patch_idx_arr) == 0:
        return {"error": "Selected regions have no patches"}

    root = _resolve_root(state.root_dir)
    slides = data_loader.parse_root_directory(str(root))

    if state.slide_name not in slides:
        return {"error": f"Slide '{state.slide_name}' not found under root_dir"}

    info = slides[state.slide_name]
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    for model in sorted(info.models.keys()):
        if mag not in info.models[model]:
            warnings.append(f"'{model}' skipped: magnification '{mag}' not available")
            continue
        if patch_size not in info.models[model][mag]:
            warnings.append(f"'{model}' skipped: patch_size '{patch_size}' not available")
            continue

        h5_path = Path(info.models[model][mag][patch_size])
        try:
            features, _, _, _ = _load_features_and_coords(h5_path)
        except Exception as exc:
            warnings.append(f"'{model}' skipped: could not load features ({exc})")
            continue

        if sub_patch_idx_arr.max() >= len(features):
            warnings.append(
                f"'{model}' skipped: patch count {len(features)} too small for region patch indices"
            )
            continue

        sub_features = features[sub_patch_idx_arr]

        # Subsample for performance
        n = len(sub_features)
        if n > 5000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, 5000, replace=False)
            sf = sub_features[sample_idx]
            sl = sub_labels[sample_idx]
        else:
            sf, sl = sub_features, sub_labels

        try:
            sil = float(silhouette_score(sf, sl, metric=metric))
        except Exception as exc:
            warnings.append(f"'{model}' skipped: silhouette failed ({exc})")
            continue

        results.append({"model": model, "feature_dim": int(features.shape[1]), "silhouette_score": sil})

    results.sort(key=lambda x: x["silhouette_score"], reverse=True)
    for i, entry in enumerate(results):
        entry["rank"] = i + 1

    return {
        "slide_name": state.slide_name,
        "region_ids": region_ids,
        "region_names": [state.labeled_regions[rid].name for rid in region_ids],
        "metric": metric,
        "ranking": results,
        "n_models_evaluated": len(results),
        "warnings": warnings,
    }


if __name__ == "__main__":
    mcp.run()

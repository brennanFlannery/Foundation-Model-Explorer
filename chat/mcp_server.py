"""Local MCP server exposing FoundationDetector inspection and GUI-action tools."""
from __future__ import annotations

import json
import time as _time
from datetime import datetime
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
        spatial: Dict[str, Any] = {
            "std_x": float(coords[:, 0].std()), "std_y": float(coords[:, 1].std()),
            "mean_x": float(coords[:, 0].mean()), "mean_y": float(coords[:, 1].mean()),
        }
        if state.mpp is not None:
            px_to_mm = state.mpp * 1e-3
            spatial["std_x_mm"] = round(float(coords[:, 0].std()) * px_to_mm, 4)
            spatial["std_y_mm"] = round(float(coords[:, 1].std()) * px_to_mm, 4)
            spatial["mpp"] = state.mpp
        result["spatial_spread"] = spatial
    return result


def _stat_area_covered(state, indices):
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 not available"}
    patch_size = float(state.patch_size_lv0)
    patch_area_px2 = patch_size ** 2
    result: Dict[str, Any] = {
        "patch_count": len(indices),
        "patch_size_px": patch_size,
        "area_px2": int(len(indices) * patch_area_px2),
    }
    if state.mpp is not None:
        px_to_mm = state.mpp * 1e-3
        result["patch_size_um"] = round(patch_size * state.mpp, 2)
        result["area_mm2"] = round(len(indices) * patch_area_px2 * px_to_mm ** 2, 6)
        result["area_um2"] = round(len(indices) * patch_area_px2 * state.mpp ** 2, 2)
        result["mpp"] = state.mpp
    return result


def _stat_area_bbox(state, indices):
    if state.coords_lv0 is None:
        return {"error": "coords_lv0 not available"}
    coords = state.coords_lv0[indices]
    pad = float(state.patch_size_lv0 or 0)
    w = float(coords[:, 0].max() - coords[:, 0].min()) + pad
    h = float(coords[:, 1].max() - coords[:, 1].min()) + pad
    result: Dict[str, Any] = {
        "width_px": int(w), "height_px": int(h), "area_px2": int(w * h),
        "bbox": {
            "x_min": int(coords[:, 0].min()), "y_min": int(coords[:, 1].min()),
            "x_max": int(coords[:, 0].max()), "y_max": int(coords[:, 1].max()),
        },
    }
    if state.mpp is not None:
        px_to_mm = state.mpp * 1e-3
        result["width_mm"] = round(w * px_to_mm, 4)
        result["height_mm"] = round(h * px_to_mm, 4)
        result["area_mm2"] = round(w * h * px_to_mm ** 2, 6)
        result["width_um"] = round(w * state.mpp, 1)
        result["height_um"] = round(h * state.mpp, 1)
        result["mpp"] = state.mpp
    return result


def _stat_area_hull(state, indices):
    if state.coords_lv0 is None:
        return {"error": "coords_lv0 not available"}
    try:
        from shapely.geometry import MultiPoint  # type: ignore
        hull = MultiPoint(state.coords_lv0[indices].tolist()).convex_hull
        result: Dict[str, Any] = {
            "area_px2": int(hull.area),
            "perimeter_px": int(hull.length),
        }
        if state.mpp is not None:
            px_to_mm = state.mpp * 1e-3
            result["area_mm2"] = round(hull.area * px_to_mm ** 2, 6)
            result["perimeter_mm"] = round(hull.length * px_to_mm, 4)
            result["area_um2"] = round(hull.area * state.mpp ** 2, 2)
            result["mpp"] = state.mpp
        return result
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


# ---------------------------------------------------------------------------
# GUI action dispatch helper
# ---------------------------------------------------------------------------

def _dispatch_gui_action(
    action_type: str,
    params: Dict[str, Any],
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Post a GUI action and block until the main thread executes it.

    FastMCP runs sync tools in a thread-pool executor, so time.sleep here
    blocks only the executor thread — not the asyncio event loop.
    """
    action_id = app_state.post_gui_action(action_type, params)
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        result = app_state.get_gui_action_result(action_id)
        if result is not None:
            return result
        _time.sleep(0.05)
    return {"error": f"GUI action '{action_type}' timed out after {timeout:.0f}s"}


def _clip_score(value: float) -> float:
    """Clamp a numeric score to [0, 100]."""
    return max(0.0, min(100.0, float(value)))


def _sample_patch_indices(
    state: app_state.AppState,
    candidate_indices: np.ndarray,
    n_samples: int,
    strategy: str,
    include_boundary: bool,
) -> Tuple[List[int], Dict[int, float], List[str]]:
    """Sample patch indices from a candidate pool with optional boundary blend."""
    warnings: List[str] = []
    if len(candidate_indices) == 0:
        return [], {}, warnings
    if state.features is None:
        return [], {}, ["No feature matrix available"]

    n_samples = max(1, int(n_samples))
    n_samples = min(n_samples, len(candidate_indices))
    feats = state.features[candidate_indices]
    centroid = feats.mean(axis=0)
    centroid_d = np.linalg.norm(feats - centroid, axis=1)

    if strategy == "centroid":
        order = np.argsort(centroid_d)
        picked = candidate_indices[order[:n_samples]]
    elif strategy == "boundary":
        order = np.argsort(centroid_d)[::-1]
        picked = candidate_indices[order[:n_samples]]
    else:  # diverse
        pool = candidate_indices
        pool_feats = feats
        if len(candidate_indices) > 2000:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(candidate_indices), 2000, replace=False)
            pool = candidate_indices[keep]
            pool_feats = feats[keep]
            warnings.append("Diverse sampling pool capped to 2000 candidates for speed")

        center = pool_feats.mean(axis=0)
        center_d = np.linalg.norm(pool_feats - center, axis=1)
        seed = int(np.argmin(center_d))
        selected = [seed]
        if n_samples > 1:
            min_d = np.linalg.norm(pool_feats - pool_feats[seed], axis=1)
            while len(selected) < min(n_samples, len(pool)):
                nxt = int(np.argmax(min_d))
                if nxt in selected:
                    break
                selected.append(nxt)
                min_d = np.minimum(min_d, np.linalg.norm(pool_feats - pool_feats[nxt], axis=1))
        picked = pool[selected]
        if len(picked) < n_samples:
            rem = [i for i in candidate_indices.tolist() if i not in set(picked.tolist())]
            picked = np.array(picked.tolist() + rem[: (n_samples - len(picked))], dtype=int)

    if include_boundary and strategy != "boundary" and len(candidate_indices) > n_samples:
        boundary_n = max(1, n_samples // 4)
        boundary = candidate_indices[np.argsort(centroid_d)[::-1][:boundary_n]]
        merged: List[int] = []
        seen = set()
        for idx in np.concatenate([picked, boundary]):
            idx_int = int(idx)
            if idx_int in seen:
                continue
            seen.add(idx_int)
            merged.append(idx_int)
            if len(merged) >= n_samples:
                break
        picked = np.array(merged, dtype=int)

    score_map: Dict[int, float] = {}
    d_lookup = {int(i): float(d) for i, d in zip(candidate_indices.tolist(), centroid_d.tolist())}
    for idx in picked.tolist():
        score_map[int(idx)] = float(d_lookup.get(int(idx), 0.0))
    return [int(i) for i in picked.tolist()], score_map, warnings


_LAST_OOD_DETECTION: Dict[str, Any] = {}


def _compute_knn_scores(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
    k_neighbors: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Return mean k-NN distance score for each candidate."""
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    n_ref = int(len(reference_features))
    if n_ref < 2:
        raise ValueError("Reference feature set is too small for k-NN scoring")
    k = max(1, int(k_neighbors))
    n_query = k + 1 if exclude_self else k
    n_query = min(n_query, n_ref)

    nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
    nn.fit(reference_features)
    dists, _ = nn.kneighbors(candidate_features, return_distance=True)
    if exclude_self and dists.shape[1] > 1:
        d_use = dists[:, 1:]
    else:
        d_use = dists
    return d_use.mean(axis=1).astype(float)


def _mad_threshold(scores: np.ndarray, robust_z: float) -> float:
    """Compute robust z-score threshold using MAD."""
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad < 1e-12:
        return float(np.max(scores) + 1.0)
    scale = 1.4826 * mad
    return median + float(robust_z) * scale


def _stable_name_seed(name: str) -> int:
    """Build a deterministic integer seed from a string."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def _select_cohort_slides(
    all_slide_names: List[str],
    compatible_slide_names: List[str],
    slide_selection_mode: Literal["all", "explicit", "first_n", "random_n"],
    slide_names: Optional[List[str]],
    max_slides: Optional[int],
    random_seed: int,
) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """Select target cohort slides and describe excluded entries."""
    warnings: List[str] = []
    skipped: List[Dict[str, str]] = []
    compatible_set = set(compatible_slide_names)

    if slide_selection_mode == "explicit":
        requested = [str(s).strip() for s in (slide_names or []) if str(s).strip()]
        if not requested:
            raise ValueError("slide_names must be provided when slide_selection_mode='explicit'")
        selected: List[str] = []
        seen: set = set()
        for s in requested:
            if s in seen:
                continue
            seen.add(s)
            if s not in all_slide_names:
                skipped.append({"slide_name": s, "reason": "not_found_under_root"})
                continue
            if s not in compatible_set:
                skipped.append({"slide_name": s, "reason": "missing_requested_model_mag_patch"})
                continue
            selected.append(s)
        return selected, skipped, warnings

    selected = list(compatible_slide_names)
    if slide_selection_mode in {"first_n", "random_n"}:
        if max_slides is None:
            raise ValueError("max_slides is required for slide_selection_mode='first_n' or 'random_n'")
        n_target = max(2, int(max_slides))
        if len(selected) < n_target:
            warnings.append(
                f"Requested {n_target} slides, but only {len(selected)} compatible slides were available"
            )
        n_final = min(n_target, len(selected))
        if slide_selection_mode == "first_n":
            selected = selected[:n_final]
        else:
            rng = np.random.default_rng(int(random_seed))
            idx = np.arange(len(selected))
            rng.shuffle(idx)
            selected = [selected[int(i)] for i in idx[:n_final]]

    incompatible = sorted(set(all_slide_names) - compatible_set)
    for s in incompatible:
        skipped.append({"slide_name": s, "reason": "missing_requested_model_mag_patch"})
    skipped.sort(key=lambda x: x["slide_name"])
    return selected, skipped, warnings


def _compute_js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Compute Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _minmax_norm(values: List[float]) -> List[float]:
    """Min-max normalize list values to [0, 1]."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [(float(v) - vmin) / (vmax - vmin) for v in values]


def _collect_cross_slide_reference(
    state: app_state.AppState,
    model: Optional[str],
    mag: Optional[str],
    patch_size: Optional[str],
    build_atlas_if_missing: bool,
) -> Tuple[np.ndarray, List[str], List[str], bool]:
    """Collect reference features for cross-slide OOD scoring."""
    warnings: List[str] = []
    atlas_used = False
    if state.root_dir is None:
        raise ValueError("No root directory is loaded")
    root = _resolve_root(state.root_dir)
    slides = data_loader.parse_root_directory(str(root))

    selected_model = model or ((state.selected_models or [None])[0] if state.selected_models else None)
    selected_mag = mag or state.magnification
    selected_patch = patch_size or state.patch_size
    if not selected_model or not selected_mag or not selected_patch:
        raise ValueError("model, mag, and patch_size are required for cross-slide OOD scoring")

    if state.atlas_state is not None and state.atlas_state.slide_names:
        candidate_slides = list(state.atlas_state.slide_names)
        atlas_used = True
    elif build_atlas_if_missing:
        candidate_slides = sorted(slides.keys())
        warnings.append("Atlas was missing; generated cross-slide reference from all root slides")
    else:
        raise ValueError("No atlas has been built yet")

    target_slide = state.slide_name
    reference_features: List[np.ndarray] = []
    used_slides: List[str] = []
    for sname in candidate_slides:
        if sname == target_slide:
            continue
        if sname not in slides:
            warnings.append(f"Slide '{sname}' is in atlas list but missing under root_dir")
            continue
        info = slides[sname]
        if selected_model not in info.models:
            warnings.append(f"Slide '{sname}' skipped: model '{selected_model}' not available")
            continue
        if selected_mag not in info.models[selected_model]:
            warnings.append(f"Slide '{sname}' skipped: mag '{selected_mag}' not available")
            continue
        if selected_patch not in info.models[selected_model][selected_mag]:
            warnings.append(f"Slide '{sname}' skipped: patch_size '{selected_patch}' not available")
            continue
        h5_path = Path(info.models[selected_model][selected_mag][selected_patch]).resolve()
        feats, _, _, _ = _load_features_and_coords(h5_path)
        reference_features.append(feats)
        used_slides.append(sname)

    if not reference_features:
        raise ValueError("No usable reference slides found for cross-slide OOD scoring")

    ref = np.vstack(reference_features).astype(np.float32)
    return ref, used_slides, warnings, atlas_used


def _spatial_components_from_patch_indices(
    patch_indices: List[int],
    coords: np.ndarray,
    patch_size_lv0: float,
) -> List[List[int]]:
    """Split patch indices into 4-connected components on grid coordinates."""
    if not patch_indices:
        return []
    step = int(round(float(patch_size_lv0)))
    idx_set = set(int(i) for i in patch_indices)
    coord_map = {(int(coords[i, 0]), int(coords[i, 1])): int(i) for i in idx_set}
    visited: set = set()
    components: List[List[int]] = []

    for start in sorted(idx_set):
        if start in visited:
            continue
        comp: List[int] = []
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            x, y = int(coords[node, 0]), int(coords[node, 1])
            for nx, ny in ((x + step, y), (x - step, y), (x, y + step), (x, y - step)):
                nb = coord_map.get((nx, ny))
                if nb is not None and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    return components


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
        "color_hex": (
            state.cluster_colours[int(cluster_id)]
            if state.cluster_colours is not None and int(cluster_id) < len(state.cluster_colours)
            else None
        ),
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
def get_boundary_patches(
    top_k: int = 30,
    cluster_a: Optional[int] = None,
    cluster_b: Optional[int] = None,
) -> Dict[str, Any]:
    """Identify patches that sit near cluster decision boundaries.

    For each patch computes the distance to its assigned centroid and to the
    nearest OTHER centroid.  Patches with the smallest gap (assigned_dist -
    nearest_other_dist) are most uncertain / boundary-like.  Returns top_k
    such patches with patch index, assigned cluster, nearest other cluster,
    gap, and level-0 coordinates.

    Parameters
    ----------
    top_k : int
        Number of boundary patches to return.
    cluster_a : int, optional
        First cluster of a pair to filter to.  Must be used together with
        cluster_b.  When both are set, only patches whose assigned cluster
        and nearest other cluster are exactly {cluster_a, cluster_b} are
        returned.
    cluster_b : int, optional
        Second cluster of a pair to filter to.
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

    # Optional cluster-pair filter
    pair_filter = None
    if cluster_a is not None and cluster_b is not None:
        pair_set = {int(cluster_a), int(cluster_b)}
        pair_filter = np.array([
            {int(labels[i]), int(nearest_other_idx[i])} == pair_set
            for i in range(n_patches)
        ])
        if not pair_filter.any():
            return {
                "top_k": 0,
                "boundary_patches": [],
                "cluster_pair": sorted(pair_set),
                "note": f"No boundary patches found between clusters {cluster_a} and {cluster_b}",
            }

    if pair_filter is not None:
        candidates = np.where(pair_filter)[0]
        order = candidates[np.argsort(gap[candidates])[: min(top_k, len(candidates))]]
    else:
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

    result = {
        "top_k": len(patches),
        "boundary_patches": patches,
        "note": "Sorted by gap (assigned_dist - nearest_other_dist); smaller gap = more uncertain",
    }
    if cluster_a is not None and cluster_b is not None:
        result["cluster_pair"] = sorted([int(cluster_a), int(cluster_b)])
    return result


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


# ---------------------------------------------------------------------------
# GUI action tools — mutate live GUI state via queue + QTimer drain
# ---------------------------------------------------------------------------


@mcp.tool()
def expand_region(
    region_id: int,
    n_rings: int = 1,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Expand a labeled region outward by N rings of spatially adjacent patches.

    Uses 4-connected grid adjacency on the patch coordinate grid.  Each ring
    adds all patches that share a grid edge with the current frontier.
    Returns a new labeled region containing the original patches plus the
    ring patches (via the create_agent_region GUI action).
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}

    region = state.labeled_regions[region_id]
    if not region.patch_indices:
        return {"error": "Region has no patches"}

    coords = state.coords_lv0 if state.coords_lv0 is not None else state.coords_thumb
    if coords is None:
        return {"error": "No coordinate data available"}
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 not available"}

    n_rings = max(1, int(n_rings))
    step = int(round(state.patch_size_lv0))
    coord_to_idx = {(int(coords[i, 0]), int(coords[i, 1])): i for i in range(len(coords))}

    frontier: set = set(region.patch_indices)
    expanded: set = set(region.patch_indices)
    for _ in range(n_rings):
        next_ring: set = set()
        for idx in frontier:
            x, y = int(coords[idx, 0]), int(coords[idx, 1])
            for nx, ny in [(x + step, y), (x - step, y), (x, y + step), (x, y - step)]:
                candidate = coord_to_idx.get((nx, ny))
                if candidate is not None and candidate not in expanded:
                    next_ring.add(candidate)
        expanded |= next_ring
        frontier = next_ring
        if not frontier:
            break

    added_count = len(expanded) - len(region.patch_indices)
    result = _dispatch_gui_action(
        "create_agent_region",
        {"patch_indices": sorted(expanded), "name": name, "source_region_id": region_id},
    )
    result["source_region_id"] = region_id
    result["original_patch_count"] = len(region.patch_indices)
    result["added_patches"] = added_count
    result["n_rings"] = n_rings
    return result


@mcp.tool()
def find_most_different_cluster(
    region_ids: List[int],
    metric: str = "cosine",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Find K-means cluster(s) most dissimilar to the combined centroid of labeled regions.

    Computes the mean feature vector across all patches in the given regions,
    then ranks all clusters by distance in descending order (most different first).
    Accepts one or more region IDs to form a combined centroid.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_centroids is None:
        return {"error": "Cluster centroids not available"}
    if not region_ids:
        return {"error": "region_ids must be a non-empty list"}
    if metric not in ("cosine", "euclidean"):
        return {"error": "metric must be 'cosine' or 'euclidean'"}

    missing = [rid for rid in region_ids if rid not in state.labeled_regions]
    if missing:
        return {"error": f"Region IDs not found: {missing}. Use list_labeled_regions to see available."}

    all_idx = np.concatenate([
        np.array(list(state.labeled_regions[rid].patch_indices), dtype=int)
        for rid in region_ids
    ])
    if len(all_idx) == 0:
        return {"error": "Selected regions have no patches"}

    combined_vec = state.features[all_idx].mean(axis=0, keepdims=True)  # (1, d)
    centroids = state.cluster_centroids  # (k, d)

    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances  # type: ignore
        dists = cosine_distances(combined_vec, centroids)[0]
    else:
        dists = np.linalg.norm(centroids - combined_vec, axis=1)

    top_n = min(max(1, int(top_k)), len(centroids))
    order = np.argsort(dists)[::-1][:top_n]
    results = [
        {
            "cluster_id": int(c),
            "distance": round(float(dists[c]), 6),
            "patch_count": int((state.cluster_labels == c).sum()) if state.cluster_labels is not None else 0,
        }
        for c in order
    ]
    return {
        "region_ids": region_ids,
        "metric": metric,
        "ranked_clusters": results,
        "note": "Clusters ranked most-different first",
    }


@mcp.tool()
def find_most_distinct_cluster(
    metric: Literal["cosine", "euclidean"] = "euclidean",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Rank clusters by distinctness using mean centroid distance to all others."""
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_centroids is None:
        return {"error": "Cluster centroids not available"}
    if state.cluster_labels is None:
        return {"error": "Cluster labels are not available"}
    if metric not in ("cosine", "euclidean"):
        return {"error": "metric must be 'cosine' or 'euclidean'"}

    centroids = state.cluster_centroids.astype(np.float32)
    n_clusters = int(len(centroids))
    if n_clusters < 2:
        return {"error": "At least 2 clusters are required"}

    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances  # type: ignore

        pairwise = cosine_distances(centroids, centroids)
    else:
        diff = centroids[:, None, :] - centroids[None, :, :]
        pairwise = np.linalg.norm(diff, axis=2)

    ranked: List[Dict[str, Any]] = []
    for cid in range(n_clusters):
        others = [i for i in range(n_clusters) if i != cid]
        d = pairwise[cid, others]
        color_hex = (
            state.cluster_colours[int(cid)]
            if state.cluster_colours is not None and int(cid) < len(state.cluster_colours)
            else None
        )
        ranked.append(
            {
                "cluster_id": int(cid),
                "color_hex": color_hex,
                "mean_distance": round(float(np.mean(d)), 6),
                "nearest_distance": round(float(np.min(d)), 6),
                "patch_count": int((state.cluster_labels == cid).sum()),
            }
        )

    ranked.sort(key=lambda x: x["mean_distance"], reverse=True)
    top_n = min(max(1, int(top_k)), len(ranked))
    top_ranked = ranked[:top_n]
    return {
        "metric": metric,
        "most_distinct_cluster_id": int(top_ranked[0]["cluster_id"]),
        "ranked_clusters": top_ranked,
        "note": "Ranked by descending mean centroid distance to all other clusters",
    }


@mcp.tool()
def find_most_similar_cluster(
    region_id: int,
    metric: str = "cosine",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Find K-means cluster(s) whose centroid is most similar to a labeled region.

    Uses the region's mean feature vector vs. each cluster centroid.
    Returns ranked list of cluster_ids with similarity scores.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.cluster_centroids is None:
        return {"error": "Cluster centroids not available"}
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}

    region = state.labeled_regions[region_id]
    idx = np.array(region.patch_indices, dtype=int)
    region_vec = state.features[idx].mean(axis=0, keepdims=True)   # (1, d)
    centroids = state.cluster_centroids                             # (k, d)

    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances  # type: ignore
        dists = cosine_distances(region_vec, centroids)[0]
    elif metric == "euclidean":
        dists = np.linalg.norm(centroids - region_vec, axis=1)
    else:
        return {"error": "metric must be 'cosine' or 'euclidean'"}

    order = np.argsort(dists)[:top_k]
    results = [
        {
            "cluster_id": int(c),
            "distance": round(float(dists[c]), 6),
            "patch_count": int((state.cluster_labels == c).sum()),
        }
        for c in order
    ]
    return {"region_id": region_id, "metric": metric, "ranked_clusters": results}


@mcp.tool()
def label_cluster(
    cluster_id: int,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Tell the GUI to create a LabeledRegion for an entire K-means cluster.

    Triggers the same radial-sweep animation as a manual cluster click.
    Returns the new region_id or an error if the cluster is already labeled.
    """
    state = app_state.get()
    if state.cluster_labels is None:
        return {"error": "No slide loaded"}
    n_clusters = len(state.cluster_centroids) if state.cluster_centroids is not None else 0
    if cluster_id < 0 or cluster_id >= n_clusters:
        return {"error": f"cluster_id {cluster_id} out of range [0, {n_clusters})"}
    return _dispatch_gui_action("label_cluster", {"cluster_id": cluster_id, "name": name})


@mcp.tool()
def label_similar_cluster(
    region_id: int,
    name: Optional[str] = None,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """Find the K-means cluster most similar to a region and label it.

    Convenience compound of find_most_similar_cluster + label_cluster.
    """
    best = find_most_similar_cluster(region_id=region_id, metric=metric, top_k=1)
    if "error" in best:
        return best
    cluster_id = best["ranked_clusters"][0]["cluster_id"]
    result = _dispatch_gui_action("label_cluster", {"cluster_id": cluster_id, "name": name})
    result["source_region_id"] = region_id
    result["chosen_cluster_id"] = cluster_id
    result["distance"] = best["ranked_clusters"][0]["distance"]
    return result


@mcp.tool()
def label_similar_patches_as_region(
    region_id: int,
    top_k: int = 30,
    name: Optional[str] = None,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """Find the top-K patches most similar to a region and create a new labeled region.

    Chains find_similar_patches → create_agent_region GUI action.
    The new region uses the most common cluster label among the found patches.
    """
    similar = find_similar_patches(region_id=region_id, top_k=top_k, metric=metric)
    if "error" in similar:
        return similar
    patch_indices = [p["patch_index"] for p in similar.get("similar_patches", [])]
    if not patch_indices:
        return {"error": "No similar patches found"}
    return _dispatch_gui_action(
        "create_agent_region",
        {"patch_indices": patch_indices, "name": name, "source_region_id": region_id},
    )


@mcp.tool()
def delete_region(region_id: int) -> Dict[str, Any]:
    """Delete a labeled region from the GUI."""
    state = app_state.get()
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}
    return _dispatch_gui_action("delete_region", {"region_id": region_id})


@mcp.tool()
def rename_region(region_id: int, new_name: str) -> Dict[str, Any]:
    """Rename a labeled region."""
    state = app_state.get()
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}
    if not new_name or not new_name.strip():
        return {"error": "new_name must be a non-empty string"}
    return _dispatch_gui_action(
        "rename_region", {"region_id": region_id, "new_name": new_name.strip()}
    )


@mcp.tool()
def navigate_to_region(region_id: int, padding_fraction: float = 0.15) -> Dict[str, Any]:
    """Pan and zoom the slide view to show a labeled region.

    padding_fraction: fraction of bounding-box size to add as margin (default 0.15).
    """
    state = app_state.get()
    if region_id not in state.labeled_regions:
        return {"error": f"Region {region_id} not found"}
    if state.coords_lv0 is None and state.coords_thumb is None:
        return {"error": "No coordinate data available"}
    return _dispatch_gui_action(
        "navigate_to_region",
        {"region_id": region_id, "padding_fraction": padding_fraction},
    )


@mcp.tool()
def create_region_from_patches(
    patch_indices: List[int],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new labeled region from an explicit list of patch indices.

    Useful after find_similar_patches: pass the returned patch_index values here.
    The region will have source_mode='local' and use the most common cluster label
    among the given patches as its kmeans_cluster reference.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if not patch_indices:
        return {"error": "patch_indices must be a non-empty list"}
    n = len(state.features)
    bad = [i for i in patch_indices if i < 0 or i >= n]
    if bad:
        return {"error": f"Out-of-range patch indices: {bad[:5]}"}
    return _dispatch_gui_action(
        "create_agent_region",
        {"patch_indices": list(patch_indices), "name": name, "source_region_id": None},
    )


@mcp.tool()
def select_cluster(cluster_id: int) -> Dict[str, Any]:
    """Highlight a K-means cluster in both slide and scatter views.

    Creates a LabeledRegion with the radial-sweep animation, same as a
    manual cluster click. Unlike deselect_all_clusters, this adds a selection.
    """
    state = app_state.get()
    if state.cluster_labels is None:
        return {"error": "No slide loaded"}
    n_clusters = len(state.cluster_centroids) if state.cluster_centroids is not None else 0
    if cluster_id < 0 or cluster_id >= n_clusters:
        return {"error": f"cluster_id {cluster_id} out of range [0, {n_clusters})"}
    return _dispatch_gui_action("select_cluster", {"cluster_id": cluster_id})


@mcp.tool()
def clear_all_regions() -> Dict[str, Any]:
    """Remove all labeled regions from the GUI."""
    return _dispatch_gui_action("clear_all_regions", {})


@mcp.tool()
def deselect_all_clusters() -> Dict[str, Any]:
    """Clear the cluster selection highlight in slide and scatter views."""
    return _dispatch_gui_action("deselect_all_clusters", {})


@mcp.tool()
def set_cluster_count(k: int) -> Dict[str, Any]:
    """Change the K-means cluster count and trigger re-clustering.

    Sets the cluster spinbox to the given value and re-runs clustering on the
    currently loaded slide.  Valid range: 2–10.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if k < 2 or k > 10:
        return {"error": f"k must be between 2 and 10 (got {k})"}
    return _dispatch_gui_action("set_cluster_count", {"k": k})


@mcp.tool()
def load_slide(slide_name: str) -> Dict[str, Any]:
    """Switch the active slide in the GUI.

    Selects a different slide from the loaded directory.  The slide must
    already be discovered (use list_data to see available slides).  Triggers
    thumbnail loading and, if a model selection is active, feature loading
    and clustering.
    """
    state = app_state.get()
    if state.root_dir is None:
        return {"error": "No root directory loaded"}
    return _dispatch_gui_action("load_slide", {"slide_name": slide_name}, timeout=15.0)


@mcp.tool()
def lookup_patch_by_coords(x: int, y: int) -> Dict[str, Any]:
    """Find the patch nearest to a given level-0 coordinate.

    Returns the patch index, cluster assignment, level-0 coordinates, and
    region membership (if any) of the closest patch.  Useful when a
    pathologist references a spatial position from the slide.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.coords_lv0 is None:
        return {"error": "No level-0 coordinates available"}

    target = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(state.coords_lv0 - target, axis=1)
    nearest_idx = int(np.argmin(dists))
    nearest_dist = float(dists[nearest_idx])

    result: Dict[str, Any] = {
        "patch_index": nearest_idx,
        "coords_lv0": {
            "x": int(state.coords_lv0[nearest_idx, 0]),
            "y": int(state.coords_lv0[nearest_idx, 1]),
        },
        "distance_px": round(nearest_dist, 1),
    }

    if state.cluster_labels is not None:
        result["cluster_id"] = int(state.cluster_labels[nearest_idx])

    # Check region membership
    regions_containing: List[Dict[str, Any]] = []
    for rid, r in state.labeled_regions.items():
        if nearest_idx in r.patch_indices:
            regions_containing.append({"region_id": rid, "name": r.name})
    result["regions"] = regions_containing

    return result


@mcp.tool()
def export_regions_geojson(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Export all labeled regions to a GeoJSON file.

    If output_path is omitted, a default path is generated next to the
    slide image.  Returns the path of the written file.
    """
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if not state.labeled_regions:
        return {"error": "No labeled regions to export"}
    params: Dict[str, Any] = {}
    if output_path:
        params["output_path"] = output_path
    return _dispatch_gui_action("export_regions_geojson", params, timeout=10.0)


@mcp.tool()
def switch_to_atlas_view() -> Dict[str, Any]:
    """Switch the GUI sidebar to the Atlas view tab.

    Requires an atlas to have been built in the GUI first.
    Triggers the same tab-switch that sets atlas labels on the slide and scatter views.
    """
    state = app_state.get()
    if state.atlas_state is None:
        return {"error": "No atlas has been built yet"}
    return _dispatch_gui_action("switch_to_atlas_view", {})


@mcp.tool()
def highlight_atlas_cluster(cluster_id: int) -> Dict[str, Any]:
    """Highlight a cluster across all atlas slide thumbnails and the atlas scatter view.

    Requires an atlas to have been built in the GUI first.
    Useful after switch_to_atlas_view to draw the user's attention to a specific
    tissue type across all slides.
    """
    state = app_state.get()
    if state.atlas_state is None:
        return {"error": "No atlas has been built yet"}
    n_clusters = state.atlas_state.n_clusters
    if cluster_id < 0 or cluster_id >= n_clusters:
        return {"error": f"cluster_id {cluster_id} out of range [0, {n_clusters})"}
    return _dispatch_gui_action("highlight_atlas_cluster", {"cluster_id": cluster_id})


@mcp.tool()
def detect_ood_patches(
    mode: Literal["single_slide", "cross_slide"] = "single_slide",
    top_k: int = 200,
    k_neighbors: int = 25,
    threshold_mode: Literal["quantile", "robust_z", "absolute"] = "quantile",
    quantile: float = 0.995,
    robust_z: float = 3.5,
    absolute_threshold: Optional[float] = None,
    normalize_embeddings: bool = True,
    build_atlas_if_missing: bool = True,
    model: Optional[str] = None,
    mag: Optional[str] = None,
    patch_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect out-of-distribution patches using k-NN embedding distance."""
    global _LAST_OOD_DETECTION

    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}
    if state.coords_lv0 is None:
        return {"error": "No level-0 coordinates available"}

    warnings: List[str] = []
    candidate_features = state.features.astype(np.float32)
    if normalize_embeddings:
        norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        candidate_features = candidate_features / norms

    if mode == "single_slide":
        reference = candidate_features
        used_slides = [state.slide_name or "current_slide"]
        atlas_used = False
        exclude_self = True
    else:
        try:
            ref_raw, used_slides, ref_warn, atlas_used = _collect_cross_slide_reference(
                state=state,
                model=model,
                mag=mag,
                patch_size=patch_size,
                build_atlas_if_missing=build_atlas_if_missing,
            )
        except Exception as exc:
            return {"error": str(exc)}
        warnings.extend(ref_warn)
        reference = ref_raw
        if normalize_embeddings:
            rnorm = np.linalg.norm(reference, axis=1, keepdims=True)
            rnorm = np.clip(rnorm, 1e-12, None)
            reference = reference / rnorm
        exclude_self = False

    try:
        scores = _compute_knn_scores(
            candidate_features=candidate_features,
            reference_features=reference,
            k_neighbors=k_neighbors,
            exclude_self=exclude_self,
        )
    except Exception as exc:
        return {"error": f"k-NN scoring failed: {exc}"}

    if threshold_mode == "absolute":
        if absolute_threshold is None:
            return {"error": "absolute_threshold is required when threshold_mode='absolute'"}
        threshold = float(absolute_threshold)
    elif threshold_mode == "robust_z":
        threshold = _mad_threshold(scores, robust_z=robust_z)
    else:
        q = min(max(float(quantile), 0.5), 0.9999)
        threshold = float(np.quantile(scores, q))

    flagged_mask = scores >= threshold
    flagged_idx = np.where(flagged_mask)[0].astype(int)
    if len(flagged_idx) == 0:
        ranked_idx = np.argsort(scores)[::-1][: min(max(1, top_k), len(scores))]
    else:
        ranked_idx = flagged_idx[np.argsort(scores[flagged_idx])[::-1]]
        ranked_idx = ranked_idx[: min(max(1, top_k), len(ranked_idx))]

    top_outliers: List[Dict[str, Any]] = []
    for rank_i, idx in enumerate(ranked_idx.tolist(), start=1):
        entry: Dict[str, Any] = {
            "rank": rank_i,
            "patch_index": int(idx),
            "score": float(scores[idx]),
            "coords_lv0": {
                "x": int(state.coords_lv0[idx, 0]),
                "y": int(state.coords_lv0[idx, 1]),
            },
        }
        if state.cluster_labels is not None:
            entry["cluster_id"] = int(state.cluster_labels[idx])
        top_outliers.append(entry)

    outlier_fraction = float(flagged_mask.mean()) if len(flagged_mask) else 0.0
    if outlier_fraction > 0.05:
        warnings.append(
            f"High outlier prevalence ({outlier_fraction * 100:.2f}%) suggests reference mismatch or domain shift"
        )

    result = {
        "mode": mode,
        "method": "knn_distance",
        "slide_name": state.slide_name,
        "k_neighbors": int(k_neighbors),
        "threshold_mode": threshold_mode,
        "threshold_used": float(threshold),
        "outlier_count": int(flagged_mask.sum()),
        "outlier_fraction": outlier_fraction,
        "reference_patch_count": int(len(reference)),
        "candidate_patch_count": int(len(candidate_features)),
        "reference_slides": used_slides,
        "atlas_used": bool(atlas_used),
        "top_outliers": top_outliers,
        "proposed_patch_indices": [int(x["patch_index"]) for x in top_outliers],
        "score_summary": {
            "min": float(np.min(scores)),
            "median": float(np.median(scores)),
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
        },
        "warnings": warnings,
    }
    _LAST_OOD_DETECTION = {
        "slide_name": state.slide_name,
        "patch_indices": result["proposed_patch_indices"],
        "mode": mode,
    }
    return result


@mcp.tool()
def label_ood_patches_as_region(
    patch_indices: Optional[List[int]] = None,
    from_last_detection: bool = False,
    grouping_mode: Literal["single_region", "spatial_components"] = "spatial_components",
    min_component_size: int = 5,
    name_prefix: str = "OOD",
) -> Dict[str, Any]:
    """Create annotated region(s) from OOD patch indices."""
    global _LAST_OOD_DETECTION

    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}

    if from_last_detection:
        if not _LAST_OOD_DETECTION:
            return {"error": "No previous OOD detection result is available"}
        if _LAST_OOD_DETECTION.get("slide_name") != state.slide_name:
            return {"error": "Last OOD result belongs to a different slide"}
        patch_indices = [int(i) for i in _LAST_OOD_DETECTION.get("patch_indices", [])]

    if not patch_indices:
        return {"error": "patch_indices must be provided or from_last_detection must be true"}
    uniq = sorted(set(int(i) for i in patch_indices))
    bad = [i for i in uniq if i < 0 or i >= len(state.features)]
    if bad:
        return {"error": f"Out-of-range patch indices: {bad[:10]}"}

    if grouping_mode == "single_region":
        created = _dispatch_gui_action(
            "create_agent_region",
            {
                "patch_indices": uniq,
                "name": f"{name_prefix} 1",
                "source_region_id": None,
                "select_dominant_cluster": False,
            },
        )
        if "error" in created:
            return created
        return {
            "created_regions": [created],
            "grouping_mode": grouping_mode,
            "dropped_components": 0,
            "total_input_patches": len(uniq),
            "total_labeled_patches": int(created.get("patch_count", 0)),
            "exact_match": int(created.get("patch_count", 0)) == len(uniq),
        }

    if state.coords_lv0 is None:
        return {"error": "coords_lv0 is required for spatial_components grouping"}
    if state.patch_size_lv0 is None:
        return {"error": "patch_size_lv0 is required for spatial_components grouping"}

    components = _spatial_components_from_patch_indices(
        patch_indices=uniq,
        coords=state.coords_lv0,
        patch_size_lv0=state.patch_size_lv0,
    )
    min_size = max(1, int(min_component_size))
    kept = [c for c in components if len(c) >= min_size]
    dropped = len(components) - len(kept)
    if not kept:
        return {"error": "No connected OOD components met min_component_size", "dropped_components": dropped}

    created_regions: List[Dict[str, Any]] = []
    labeled_total = 0
    for i, comp in enumerate(kept, start=1):
        res = _dispatch_gui_action(
            "create_agent_region",
            {
                "patch_indices": comp,
                "name": f"{name_prefix} {i}",
                "source_region_id": None,
                "select_dominant_cluster": False,
            },
        )
        if "error" in res:
            return {"error": res["error"], "created_regions": created_regions}
        created_regions.append(res)
        labeled_total += int(res.get("patch_count", 0))

    return {
        "created_regions": created_regions,
        "grouping_mode": grouping_mode,
        "component_count": len(components),
        "dropped_components": dropped,
        "total_input_patches": len(uniq),
        "total_labeled_patches": labeled_total,
        "exact_match": labeled_total == len(uniq),
    }


@mcp.tool()
def open_patch_exemplar_popup(
    source_type: Literal["cluster", "region", "patch_list"],
    source_id: Optional[int] = None,
    patch_indices: Optional[List[int]] = None,
    n_samples: int = 24,
    strategy: Literal["centroid", "diverse", "boundary"] = "diverse",
    include_boundary: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Open a horizontally scrollable patch exemplar popup in the GUI."""
    state = app_state.get()
    if state.features is None:
        return {"error": "No slide loaded"}

    candidates: List[int] = []
    source_label = ""
    if source_type == "cluster":
        if source_id is None:
            return {"error": "source_id is required for source_type='cluster'"}
        if state.cluster_labels is None:
            return {"error": "Cluster labels are not available"}
        candidates = np.where(state.cluster_labels == int(source_id))[0].astype(int).tolist()
        source_label = f"Cluster {int(source_id)}"
    elif source_type == "region":
        if source_id is None:
            return {"error": "source_id is required for source_type='region'"}
        rid = int(source_id)
        if rid not in state.labeled_regions:
            return {"error": f"Region {rid} not found"}
        candidates = [int(i) for i in state.labeled_regions[rid].patch_indices]
        source_label = f"Region {rid}"
    else:
        if not patch_indices:
            return {"error": "patch_indices must be provided for source_type='patch_list'"}
        total = int(len(state.features))
        bad = [int(i) for i in patch_indices if int(i) < 0 or int(i) >= total]
        if bad:
            return {"error": f"Out-of-range patch indices: {bad[:10]}"}
        candidates = [int(i) for i in patch_indices]
        source_label = "Custom Patch List"

    if not candidates:
        return {"error": "No candidate patches found for requested source"}

    sampled, score_map, warnings = _sample_patch_indices(
        state=state,
        candidate_indices=np.array(candidates, dtype=int),
        n_samples=n_samples,
        strategy=strategy,
        include_boundary=include_boundary,
    )
    if not sampled:
        return {"error": "Failed to sample exemplar patches", "warnings": warnings}

    popup_id = f"exemplar-{int(_time.time() * 1000)}"
    result = _dispatch_gui_action(
        "open_patch_exemplar_popup",
        {
            "popup_id": popup_id,
            "source_type": source_type,
            "source_id": source_id,
            "source_label": source_label,
            "strategy": strategy,
            "include_metadata": bool(include_metadata),
            "patch_indices": sampled,
            "scores_by_patch": score_map,
        },
        timeout=15.0,
    )
    if isinstance(result, dict):
        result.setdefault("popup_id", popup_id)
        result.setdefault("sample_indices", sampled)
        if warnings:
            existing = result.get("warnings") or []
            result["warnings"] = list(existing) + warnings
    return result


@mcp.tool()
def export_current_exemplar_popup(
    popup_id: str,
    output_dir: Optional[str] = None,
    format: Literal["png"] = "png",
    include_manifest: bool = True,
) -> Dict[str, Any]:
    """Export currently rendered exemplar images from the in-app popup."""
    params: Dict[str, Any] = {
        "popup_id": popup_id,
        "format": format,
        "include_manifest": bool(include_manifest),
    }
    if output_dir:
        params["output_dir"] = output_dir
    return _dispatch_gui_action("export_current_exemplar_popup", params, timeout=15.0)


@mcp.tool()
def close_exemplar_popup(popup_id: str) -> Dict[str, Any]:
    """Close the active exemplar popup if it matches popup_id."""
    return _dispatch_gui_action("close_exemplar_popup", {"popup_id": popup_id})


@mcp.tool()
def generate_slide_qc_report(
    root_dir: str,
    slide_name: str,
    model: str,
    mag: str,
    patch_size: str,
    output_path: Optional[str] = None,
    include_recommendations: bool = True,
    cluster_k_candidates: Optional[List[int]] = None,
    include_ood_assessment: bool = False,
    ood_mode: Literal["single_slide", "cross_slide"] = "single_slide",
    ood_top_k: int = 200,
    ood_k_neighbors: int = 25,
) -> Dict[str, Any]:
    """Generate a deterministic QC report (Markdown + JSON summary) for one configuration."""
    root = _resolve_root(root_dir)
    describe = describe_slide(root_dir, slide_name, model, mag, patch_size)
    if "error" in describe:
        return describe

    warnings: List[str] = []
    warnings.extend(describe.get("warnings", []))

    try:
        h5_path = _resolve_h5_path(root, slide_name, model, mag, patch_size)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    features, _, _, attrs = _load_features_and_coords(h5_path)
    patch_count = int(features.shape[0])
    feature_dim = int(features.shape[1])

    raw_candidates = cluster_k_candidates or [4, 6, 8, 10]
    k_candidates = sorted({int(k) for k in raw_candidates if int(k) >= 2})
    k_candidates = [k for k in k_candidates if k < patch_count]
    if not k_candidates:
        k_candidates = [max(2, min(8, patch_count - 1))]
        warnings.append("k candidate list was invalid; using a fallback candidate")

    k_sweep: List[Dict[str, Any]] = []
    for k in k_candidates:
        labels, inertia = _run_clustering(features, k)
        scores = _clustering_scores(features, labels)
        k_sweep.append(
            {
                "k": int(k),
                "inertia": float(inertia),
                "silhouette_score": float(scores["silhouette_score"]),
                "davies_bouldin_index": float(scores["davies_bouldin_index"]),
            }
        )

    elbow = compute_elbow_analysis(
        root_dir=root_dir,
        slide_name=slide_name,
        model=model,
        mag=mag,
        patch_size=patch_size,
        max_k=min(20, max(max(k_candidates), 10)),
    )
    if "error" not in elbow:
        warnings.extend(elbow.get("warnings", []))
    recommended_k = int(elbow.get("recommended_k", k_candidates[0]))
    if recommended_k not in k_candidates:
        best_by_sil = sorted(k_sweep, key=lambda x: x["silhouette_score"], reverse=True)[0]["k"]
        warnings.append(
            f"Elbow suggested k={recommended_k}, outside evaluated set; using best evaluated k={best_by_sil}"
        )
        recommended_k = int(best_by_sil)

    model_rank = rank_models_by_separability(
        root_dir=root_dir,
        slide_name=slide_name,
        mag=mag,
        patch_size=patch_size,
        n_clusters=recommended_k,
    )
    ranking = model_rank.get("ranking", [])
    warnings.extend(model_rank.get("warnings", []))
    current_rank_entry = next((r for r in ranking if r.get("model") == model), None)

    state = app_state.get()
    pca_context_available = (
        state.slide_name == slide_name
        and state.embedding_2d is not None
        and state.pca_explained_variance_ratio is not None
    )
    if pca_context_available:
        pca_ratio = [float(x) for x in state.pca_explained_variance_ratio or []]
    else:
        pca_ratio = []
        warnings.append(
            "Live PCA context unavailable for this slide/model; embedding fidelity section is limited"
        )
    pca_cumulative = float(sum(pca_ratio)) if pca_ratio else 0.0

    # Score model
    integrity = 100.0
    if patch_count < 200:
        integrity -= 30.0
        warnings.append("Patch count is low (<200), reducing robustness of clustering metrics")
    if feature_dim < 32:
        integrity -= 20.0
    if not attrs:
        integrity -= 5.0
    integrity -= min(40.0, 10.0 * len(describe.get("warnings", [])))
    data_integrity_score = _clip_score(integrity)

    best_sil = max(float(x["silhouette_score"]) for x in k_sweep)
    best_dbi = min(float(x["davies_bouldin_index"]) for x in k_sweep)
    silhouette_norm = ((best_sil + 1.0) / 2.0) * 100.0
    dbi_norm = (1.0 / (1.0 + max(best_dbi, 1e-6))) * 100.0
    cluster_quality_score = _clip_score(0.7 * silhouette_norm + 0.3 * dbi_norm)

    if ranking and current_rank_entry:
        n_models = max(1, int(len(ranking)))
        rank_idx = int(current_rank_entry.get("rank", n_models))
        model_fitness_score = _clip_score((n_models - rank_idx + 1) / n_models * 100.0)
    else:
        model_fitness_score = 50.0
        warnings.append("Could not determine current model's separability rank")

    if pca_ratio:
        embedding_fidelity_score = _clip_score(pca_cumulative * 100.0)
    else:
        embedding_fidelity_score = 35.0

    overall_score = _clip_score(
        0.35 * data_integrity_score
        + 0.30 * cluster_quality_score
        + 0.20 * model_fitness_score
        + 0.15 * embedding_fidelity_score
    )
    if overall_score >= 85:
        grade = "A"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 55:
        grade = "C"
    else:
        grade = "D"

    recommendations: List[Dict[str, str]] = []
    if include_recommendations:
        if current_rank_entry and int(current_rank_entry.get("rank", 1)) > 1 and ranking:
            top = ranking[0]
            recommendations.append(
                {
                    "recommendation": f"Consider switching to model '{top['model']}' for stronger separability on this slide.",
                    "confidence": "high",
                    "rationale": "Model ranking by silhouette score",
                }
            )
        if recommended_k != 8:
            recommendations.append(
                {
                    "recommendation": f"Use k={recommended_k} as default for this slide/model instead of a fixed k.",
                    "confidence": "medium",
                    "rationale": "Elbow + evaluated k sweep",
                }
            )
        if pca_ratio and pca_cumulative < 0.35:
            recommendations.append(
                {
                    "recommendation": "Treat 2D scatter interpretation as qualitative only; substantial variance is outside PC1/PC2.",
                    "confidence": "high",
                    "rationale": "Low PCA cumulative explained variance",
                }
            )

    ood_summary: Optional[Dict[str, Any]] = None
    if include_ood_assessment:
        state = app_state.get()
        if state.slide_name != slide_name or state.features is None:
            warnings.append(
                "OOD assessment skipped: requested slide is not the currently loaded GUI slide"
            )
        else:
            ood_result = detect_ood_patches(
                mode=ood_mode,
                top_k=ood_top_k,
                k_neighbors=ood_k_neighbors,
                threshold_mode="quantile",
                quantile=0.995,
                normalize_embeddings=True,
                build_atlas_if_missing=True,
                model=model,
                mag=mag,
                patch_size=patch_size,
            )
            if "error" in ood_result:
                warnings.append(f"OOD assessment skipped: {ood_result['error']}")
            else:
                ood_summary = {
                    "enabled": True,
                    "mode": ood_result.get("mode"),
                    "method": ood_result.get("method"),
                    "outlier_count": int(ood_result.get("outlier_count", 0)),
                    "outlier_fraction": float(ood_result.get("outlier_fraction", 0.0)),
                    "threshold_used": float(ood_result.get("threshold_used", 0.0)),
                    "top_examples": (ood_result.get("top_outliers") or [])[:10],
                    "warnings": ood_result.get("warnings", []),
                }
                if include_recommendations and ood_summary["outlier_fraction"] > 0.02:
                    recommendations.append(
                        {
                            "recommendation": "Investigate OOD patches; prevalence is high enough to suggest potential artifact or domain shift.",
                            "confidence": "medium",
                            "rationale": "Embedding-space outlier prevalence above 2%",
                        }
                    )

    summary_json = {
        "slide_name": slide_name,
        "model": model,
        "mag": mag,
        "patch_size": patch_size,
        "scores": {
            "data_integrity_score": round(data_integrity_score, 2),
            "cluster_quality_score": round(cluster_quality_score, 2),
            "model_fitness_score": round(model_fitness_score, 2),
            "embedding_fidelity_score": round(embedding_fidelity_score, 2),
            "overall_score": round(overall_score, 2),
            "grade": grade,
        },
        "recommended_k": int(recommended_k),
        "k_sweep": k_sweep,
        "model_rank": current_rank_entry,
        "n_models_ranked": len(ranking),
        "pca_explained_variance_ratio": pca_ratio,
        "pca_cumulative": round(pca_cumulative, 4),
        "warnings": warnings,
        "recommendations": recommendations,
    }
    if ood_summary is not None:
        summary_json["ood"] = ood_summary

    report_dir = root / "Reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    default_name = (
        f"qc_{slide_name}_{model}_{mag}_{patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    final_path = Path(output_path).expanduser() if output_path else (report_dir / default_name)
    if not final_path.is_absolute():
        final_path = (Path.cwd() / final_path).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(v: float) -> str:
        return f"{v:.3f}"

    lines: List[str] = [
        f"# QC Report: {slide_name} | {model} | {mag} | {patch_size}",
        "",
        "## Run Metadata",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Root directory: `{str(root)}`",
        f"- H5 path: `{str(h5_path)}`",
        "",
        "## Data Integrity Checks",
        f"- Patch count: **{patch_count}**",
        f"- Feature dimension: **{feature_dim}**",
        f"- H5 attributes found: **{len(attrs)}**",
        f"- Integrity score: **{data_integrity_score:.1f}/100**",
        "",
        "## Cluster Quality Analysis",
        "| k | silhouette | davies_bouldin | inertia |",
        "|---:|---:|---:|---:|",
    ]
    for row in k_sweep:
        lines.append(
            f"| {row['k']} | {_fmt(row['silhouette_score'])} | {_fmt(row['davies_bouldin_index'])} | {_fmt(row['inertia'])} |"
        )
    lines.extend(
        [
            "",
            f"- Recommended k: **{recommended_k}**",
            f"- Cluster quality score: **{cluster_quality_score:.1f}/100**",
            "",
            "## Model Benchmarking",
            f"- Models evaluated: **{len(ranking)}**",
        ]
    )
    if current_rank_entry:
        lines.append(
            f"- Current model rank: **{current_rank_entry.get('rank')} / {len(ranking)}** "
            f"(silhouette={_fmt(float(current_rank_entry.get('silhouette_score', 0.0)))})"
        )
    else:
        lines.append("- Current model rank: **N/A**")
    lines.extend(
        [
            f"- Model fitness score: **{model_fitness_score:.1f}/100**",
            "",
            "## Embedding Fidelity",
            f"- PCA explained variance ratio: `{pca_ratio}`",
            f"- PCA cumulative (PC1+PC2): **{pca_cumulative:.3f}**",
            f"- Embedding fidelity score: **{embedding_fidelity_score:.1f}/100**",
            "",
            "## Overall QC",
            f"- Overall score: **{overall_score:.1f}/100**",
            f"- Grade: **{grade}**",
            "",
        ]
    )
    if recommendations:
        lines.append("## Recommendations")
        for rec in recommendations:
            lines.append(
                f"- {rec['recommendation']} ({rec['confidence']}; rationale: {rec['rationale']})"
            )
        lines.append("")
    if ood_summary is not None:
        lines.extend(
            [
                "## OOD Outlier Assessment",
                f"- Mode: **{ood_summary['mode']}**",
                f"- Method: **{ood_summary['method']}**",
                f"- Outlier count: **{ood_summary['outlier_count']}**",
                f"- Outlier fraction: **{ood_summary['outlier_fraction'] * 100:.2f}%**",
                f"- Threshold used: **{ood_summary['threshold_used']:.6f}**",
            ]
        )
        top_examples = ood_summary.get("top_examples", [])
        if top_examples:
            lines.append("- Top examples:")
            for entry in top_examples[:5]:
                coords = entry.get("coords_lv0", {})
                lines.append(
                    f"  - patch {entry.get('patch_index')} @ ({coords.get('x')}, {coords.get('y')}), score={float(entry.get('score', 0.0)):.6f}"
                )
        lines.append("")
    lines.append("## Warnings / Limitations")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    final_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = final_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    return {
        "report_path": str(final_path),
        "summary_json_path": str(json_path),
        "summary_json": summary_json,
        "scores": summary_json["scores"],
        "warnings": warnings,
    }


@mcp.tool()
def generate_cross_slide_qc_report(
    root_dir: str,
    model: str,
    mag: str,
    patch_size: str,
    slide_names: Optional[List[str]] = None,
    slide_selection_mode: Literal["all", "explicit", "first_n", "random_n"] = "all",
    max_slides: Optional[int] = None,
    random_seed: int = 42,
    atlas_n_clusters: int = 12,
    normalize_embeddings: bool = True,
    max_patches_per_slide: int = 50000,
    ood_k_neighbors: int = 25,
    ood_threshold_mode: Literal["global_quantile", "global_robust_z", "global_absolute"] = "global_quantile",
    ood_quantile: float = 0.995,
    ood_robust_z: float = 3.5,
    ood_absolute_threshold: Optional[float] = None,
    top_k_per_slide: int = 100,
    include_recommendations: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a cross-slide QC report with atlas composition and OOD ranking."""
    root = _resolve_root(root_dir)
    slides = data_loader.parse_root_directory(str(root))
    if not slides:
        return {"error": f"No slides found under {root}"}

    all_slide_names = sorted(slides.keys())
    compatible_slide_names: List[str] = []
    compatibility_warnings: List[str] = []
    for sname in all_slide_names:
        info = slides[sname]
        if model not in info.models:
            continue
        if mag not in info.models[model]:
            continue
        if patch_size not in info.models[model][mag]:
            continue
        compatible_slide_names.append(sname)

    if len(compatible_slide_names) < 2:
        return {
            "error": (
                f"Need at least 2 compatible slides for model={model}, mag={mag}, patch_size={patch_size}. "
                f"Found {len(compatible_slide_names)}."
            )
        }

    try:
        selected_slides, skipped_slides, selection_warnings = _select_cohort_slides(
            all_slide_names=all_slide_names,
            compatible_slide_names=compatible_slide_names,
            slide_selection_mode=slide_selection_mode,
            slide_names=slide_names,
            max_slides=max_slides,
            random_seed=random_seed,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if len(selected_slides) < 2:
        return {
            "error": (
                f"Slide selection produced {len(selected_slides)} compatible slide(s); at least 2 are required. "
                "Adjust slide selection mode or filters."
            )
        }

    warnings: List[str] = []
    warnings.extend(compatibility_warnings)
    warnings.extend(selection_warnings)

    per_slide_features: Dict[str, np.ndarray] = {}
    per_slide_coords: Dict[str, np.ndarray] = {}
    per_slide_local_to_original: Dict[str, np.ndarray] = {}
    per_slide_patch_count_raw: Dict[str, int] = {}
    per_slide_patch_count_used: Dict[str, int] = {}

    max_patches = max(100, int(max_patches_per_slide))
    for sname in selected_slides:
        try:
            h5_path = _resolve_h5_path(root, sname, model, mag, patch_size)
            feats, coords, _, _ = _load_features_and_coords(h5_path)
        except Exception as exc:
            warnings.append(f"Slide '{sname}' skipped during loading: {exc}")
            continue

        n_raw = int(len(feats))
        if n_raw < 2:
            warnings.append(f"Slide '{sname}' skipped: too few patches ({n_raw})")
            continue
        idx = np.arange(n_raw, dtype=int)
        if n_raw > max_patches:
            seed = int(random_seed) + _stable_name_seed(sname)
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n_raw, size=max_patches, replace=False).astype(int))
            warnings.append(
                f"Slide '{sname}' subsampled from {n_raw} to {len(idx)} patches for cross-slide QC"
            )
        feats_used = feats[idx].astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(feats_used, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            feats_used = feats_used / norms
        per_slide_features[sname] = feats_used
        per_slide_coords[sname] = coords[idx]
        per_slide_local_to_original[sname] = idx
        per_slide_patch_count_raw[sname] = n_raw
        per_slide_patch_count_used[sname] = int(len(idx))

    selected_slides = [s for s in selected_slides if s in per_slide_features]
    if len(selected_slides) < 2:
        return {
            "error": (
                f"Only {len(selected_slides)} slide(s) remained after loading/subsampling; at least 2 are required."
            ),
            "warnings": warnings,
        }

    # Build atlas composition using pooled features.
    pooled_features = np.vstack([per_slide_features[s] for s in selected_slides])
    slide_offsets: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for sname in selected_slides:
        n = len(per_slide_features[sname])
        slide_offsets[sname] = (offset, offset + n)
        offset += n

    n_total = int(len(pooled_features))
    k = max(2, int(atlas_n_clusters))
    if k > n_total:
        k = n_total
        warnings.append(f"atlas_n_clusters capped to {k} (total pooled patches)")

    atlas_labels, _ = _run_clustering(pooled_features, n_clusters=k)
    global_cluster_counts = np.bincount(atlas_labels, minlength=k).astype(np.float64)
    global_cluster_dist = global_cluster_counts / max(1.0, float(global_cluster_counts.sum()))

    per_slide_cluster_dist: Dict[str, List[float]] = {}
    for sname in selected_slides:
        s0, s1 = slide_offsets[sname]
        c = np.bincount(atlas_labels[s0:s1], minlength=k).astype(np.float64)
        c = c / max(1.0, float(c.sum()))
        per_slide_cluster_dist[sname] = [round(float(x), 6) for x in c.tolist()]

    # Leave-one-slide-out OOD scoring.
    per_slide_ood_scores: Dict[str, np.ndarray] = {}
    pooled_ood_scores: List[np.ndarray] = []
    k_neighbors = max(1, int(ood_k_neighbors))
    for sname in selected_slides:
        cand = per_slide_features[sname]
        refs = [per_slide_features[other] for other in selected_slides if other != sname]
        ref = np.vstack(refs)
        try:
            scores = _compute_knn_scores(
                candidate_features=cand,
                reference_features=ref,
                k_neighbors=k_neighbors,
                exclude_self=False,
            )
        except Exception as exc:
            return {"error": f"Cross-slide k-NN scoring failed for slide '{sname}': {exc}"}
        per_slide_ood_scores[sname] = scores
        pooled_ood_scores.append(scores)

    pooled_scores = np.concatenate(pooled_ood_scores).astype(float)
    if ood_threshold_mode == "global_absolute":
        if ood_absolute_threshold is None:
            return {"error": "ood_absolute_threshold is required when ood_threshold_mode='global_absolute'"}
        threshold_used = float(ood_absolute_threshold)
    elif ood_threshold_mode == "global_robust_z":
        threshold_used = _mad_threshold(pooled_scores, robust_z=float(ood_robust_z))
    else:
        q = min(max(float(ood_quantile), 0.5), 0.9999)
        threshold_used = float(np.quantile(pooled_scores, q))

    top_k = max(1, int(top_k_per_slide))
    per_slide_ood_rows: List[Dict[str, Any]] = []
    for sname in selected_slides:
        scores = per_slide_ood_scores[sname]
        outlier_mask = scores >= threshold_used
        ranked_idx = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        top_outliers: List[Dict[str, Any]] = []
        local_to_orig = per_slide_local_to_original[sname]
        coords = per_slide_coords[sname]
        for i_rank, local_idx in enumerate(ranked_idx.tolist(), start=1):
            orig_idx = int(local_to_orig[local_idx])
            top_outliers.append(
                {
                    "rank": i_rank,
                    "local_patch_index": int(local_idx),
                    "patch_index": orig_idx,
                    "score": float(scores[local_idx]),
                    "coords_lv0": {
                        "x": int(coords[local_idx, 0]),
                        "y": int(coords[local_idx, 1]),
                    },
                }
            )
        per_slide_ood_rows.append(
            {
                "slide_name": sname,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_fraction": float(outlier_mask.mean()),
                "score_summary": {
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "p95": float(np.quantile(scores, 0.95)),
                    "p99": float(np.quantile(scores, 0.99)),
                    "max": float(np.max(scores)),
                },
                "top_outliers": top_outliers,
            }
        )

    per_slide_ood_rows.sort(
        key=lambda x: (x["outlier_fraction"], x["outlier_count"]), reverse=True
    )
    most_ood_by_fraction = per_slide_ood_rows[0] if per_slide_ood_rows else None
    most_ood_by_count = sorted(
        per_slide_ood_rows, key=lambda x: x["outlier_count"], reverse=True
    )[0] if per_slide_ood_rows else None

    # Additional cross-slide metrics.
    cohort_centroid = pooled_features.mean(axis=0)
    per_slide_metrics: List[Dict[str, Any]] = []
    rare_clusters = np.where(global_cluster_dist <= 0.02)[0].tolist()
    for sname in selected_slides:
        feats = per_slide_features[sname]
        centroid = feats.mean(axis=0)
        embedding_shift = float(np.linalg.norm(centroid - cohort_centroid))
        slide_dist = np.array(per_slide_cluster_dist[sname], dtype=np.float64)
        jsd = float(_compute_js_divergence(slide_dist, global_cluster_dist))

        s0, s1 = slide_offsets[sname]
        local_labels = atlas_labels[s0:s1]
        rare_burden = 0.0
        if rare_clusters:
            rare_burden = float(np.mean(np.isin(local_labels, rare_clusters)))

        try:
            within_disp_scores = _compute_knn_scores(
                candidate_features=feats,
                reference_features=feats,
                k_neighbors=min(k_neighbors, max(1, len(feats) - 1)),
                exclude_self=True,
            )
            within_dispersion = float(np.median(within_disp_scores))
        except Exception:
            within_dispersion = float(np.nan)
            warnings.append(f"within-slide dispersion failed for slide '{sname}'")

        coverage = int(np.sum(np.bincount(local_labels, minlength=k) > 0))
        per_slide_metrics.append(
            {
                "slide_name": sname,
                "embedding_shift_distance": embedding_shift,
                "distribution_js_divergence": jsd,
                "rare_cluster_burden": rare_burden,
                "within_slide_dispersion": within_dispersion,
                "effective_cluster_coverage": coverage,
            }
        )

    # Composite risk score and grading.
    metric_by_slide = {row["slide_name"]: row for row in per_slide_metrics}
    ood_by_slide = {row["slide_name"]: row for row in per_slide_ood_rows}
    metric_order = selected_slides

    ood_values = [ood_by_slide[s]["outlier_fraction"] for s in metric_order]
    shift_values = [metric_by_slide[s]["embedding_shift_distance"] for s in metric_order]
    jsd_values = [metric_by_slide[s]["distribution_js_divergence"] for s in metric_order]
    rare_values = [metric_by_slide[s]["rare_cluster_burden"] for s in metric_order]
    disp_values = [metric_by_slide[s]["within_slide_dispersion"] for s in metric_order]
    disp_values_clean = []
    finite_disp = [float(v) for v in disp_values if np.isfinite(v)]
    fallback_disp = float(np.median(finite_disp)) if finite_disp else 0.0
    for v in disp_values:
        disp_values_clean.append(float(v) if np.isfinite(v) else fallback_disp)
    cov_values = [float(metric_by_slide[s]["effective_cluster_coverage"]) for s in metric_order]

    ood_norm = _minmax_norm(ood_values)
    shift_norm = _minmax_norm(shift_values)
    jsd_norm = _minmax_norm(jsd_values)
    rare_norm = _minmax_norm(rare_values)
    disp_norm = _minmax_norm(disp_values_clean)
    cov_norm = _minmax_norm(cov_values)

    per_slide_scores: List[Dict[str, Any]] = []
    for i, sname in enumerate(metric_order):
        risk_0_1 = (
            0.40 * ood_norm[i]
            + 0.20 * shift_norm[i]
            + 0.20 * jsd_norm[i]
            + 0.10 * rare_norm[i]
            + 0.10 * (0.5 * disp_norm[i] + 0.5 * (1.0 - cov_norm[i]))
        )
        qc_score = _clip_score((1.0 - risk_0_1) * 100.0)
        if qc_score >= 85:
            grade = "A"
        elif qc_score >= 70:
            grade = "B"
        elif qc_score >= 55:
            grade = "C"
        else:
            grade = "D"
        per_slide_scores.append(
            {
                "slide_name": sname,
                "cross_slide_qc_score": round(float(qc_score), 2),
                "grade": grade,
            }
        )
    per_slide_scores.sort(key=lambda x: x["cross_slide_qc_score"], reverse=True)
    for rank_i, row in enumerate(per_slide_scores, start=1):
        row["rank"] = rank_i

    recommendations: List[Dict[str, str]] = []
    if include_recommendations and most_ood_by_fraction is not None:
        if float(most_ood_by_fraction["outlier_fraction"]) > 0.03:
            recommendations.append(
                {
                    "recommendation": (
                        f"Prioritize review of slide '{most_ood_by_fraction['slide_name']}' "
                        "because cross-slide OOD fraction is elevated."
                    ),
                    "confidence": "high",
                    "rationale": "Highest leave-one-slide-out OOD fraction",
                }
            )
        if rare_clusters:
            recommendations.append(
                {
                    "recommendation": (
                        "Inspect rare-cluster burden and top OOD patches together to separate "
                        "true artifacts from biologically rare tissue patterns."
                    ),
                    "confidence": "medium",
                    "rationale": "Rare atlas clusters were detected in the cohort",
                }
            )

    summary_json: Dict[str, Any] = {
        "cohort": {
            "root_dir": str(root),
            "model": model,
            "mag": mag,
            "patch_size": patch_size,
            "slide_selection_mode": slide_selection_mode,
            "selected_slides": selected_slides,
            "n_selected_slides": len(selected_slides),
            "skipped_slides": skipped_slides,
            "per_slide_patch_count_raw": per_slide_patch_count_raw,
            "per_slide_patch_count_used": per_slide_patch_count_used,
        },
        "atlas_summary": {
            "n_clusters": int(k),
            "total_patches": int(n_total),
            "global_cluster_distribution": [round(float(x), 6) for x in global_cluster_dist.tolist()],
            "per_slide_cluster_distribution": per_slide_cluster_dist,
        },
        "ood_summary": {
            "method": "leave_one_slide_out_knn_distance",
            "k_neighbors": int(k_neighbors),
            "threshold_mode": ood_threshold_mode,
            "threshold_used": float(threshold_used),
            "slides_ranked_by_outlier_fraction": per_slide_ood_rows,
            "most_ood_slide_by_fraction": {
                "slide_name": most_ood_by_fraction["slide_name"],
                "outlier_fraction": most_ood_by_fraction["outlier_fraction"],
                "outlier_count": most_ood_by_fraction["outlier_count"],
            } if most_ood_by_fraction is not None else None,
            "most_ood_slide_by_count": {
                "slide_name": most_ood_by_count["slide_name"],
                "outlier_fraction": most_ood_by_count["outlier_fraction"],
                "outlier_count": most_ood_by_count["outlier_count"],
            } if most_ood_by_count is not None else None,
        },
        "cross_slide_metrics": per_slide_metrics,
        "scores": {
            "per_slide": per_slide_scores,
        },
        "warnings": warnings,
        "recommendations": recommendations,
    }

    report_dir = root / "Reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    default_name = (
        f"cross_slide_qc_{model}_{mag}_{patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    final_path = Path(output_path).expanduser() if output_path else (report_dir / default_name)
    if not final_path.is_absolute():
        final_path = (Path.cwd() / final_path).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        f"# Cross-Slide QC Report: {model} | {mag} | {patch_size}",
        "",
        "## Run Metadata",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Root directory: `{str(root)}`",
        f"- Slide selection mode: **{slide_selection_mode}**",
        f"- Included slides: **{len(selected_slides)}**",
        "",
        "## Cohort Slides",
    ]
    for sname in selected_slides:
        lines.append(
            f"- {sname}: raw={per_slide_patch_count_raw[sname]}, used={per_slide_patch_count_used[sname]}"
        )
    lines.extend(
        [
            "",
            "## Atlas Summary",
            f"- Atlas clusters: **{k}**",
            f"- Total pooled patches: **{n_total}**",
            "",
            "## OOD Ranking (Leave-One-Slide-Out)",
            "| slide | outlier_fraction | outlier_count | mean_score | p99_score |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in per_slide_ood_rows:
        ss = row["score_summary"]
        lines.append(
            f"| {row['slide_name']} | {row['outlier_fraction']:.4f} | {row['outlier_count']} | "
            f"{ss['mean']:.6f} | {ss['p99']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Additional Cross-Slide Metrics",
            "| slide | embedding_shift | js_divergence | rare_cluster_burden | within_dispersion | cluster_coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_slide_metrics:
        lines.append(
            f"| {row['slide_name']} | {row['embedding_shift_distance']:.6f} | "
            f"{row['distribution_js_divergence']:.6f} | {row['rare_cluster_burden']:.4f} | "
            f"{row['within_slide_dispersion']:.6f} | {row['effective_cluster_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Slide QC Scores",
            "| rank | slide | score | grade |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in per_slide_scores:
        lines.append(
            f"| {row['rank']} | {row['slide_name']} | {row['cross_slide_qc_score']:.2f} | {row['grade']} |"
        )
    if recommendations:
        lines.extend(["", "## Recommendations"])
        for rec in recommendations:
            lines.append(
                f"- {rec['recommendation']} ({rec['confidence']}; rationale: {rec['rationale']})"
            )
    lines.extend(["", "## Warnings / Limitations"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    final_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = final_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    return {
        "report_path": str(final_path),
        "summary_json_path": str(json_path),
        "summary_json": summary_json,
        "most_ood_slide": summary_json["ood_summary"]["most_ood_slide_by_fraction"],
        "scores": summary_json["scores"],
        "warnings": warnings,
    }


if __name__ == "__main__":
    mcp.run()

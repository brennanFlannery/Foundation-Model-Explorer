from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import app_state
import mcp_server
from app_state import LabeledRegionData


def _write_h5(path: Path, n_patches: int = 120, dim: int = 16) -> None:
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(n_patches, dim)).astype(np.float32)
    xs = np.arange(n_patches) % 20
    ys = np.arange(n_patches) // 20
    coords = np.column_stack([xs * 256, ys * 256]).astype(np.int32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=feats)
        handle.create_dataset("coords", data=coords)
        handle.attrs["patch_size_level0"] = 256


def _write_h5_with_shift(path: Path, shift: float, n_patches: int = 160, dim: int = 12) -> None:
    rng = np.random.default_rng(17)
    feats = rng.normal(loc=shift, scale=1.0, size=(n_patches, dim)).astype(np.float32)
    xs = np.arange(n_patches) % 20
    ys = np.arange(n_patches) // 20
    coords = np.column_stack([xs * 256, ys * 256]).astype(np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=feats)
        handle.create_dataset("coords", data=coords)
        handle.attrs["patch_size_level0"] = 256


def test_generate_slide_qc_report_writes_markdown_and_json(tmp_path: Path) -> None:
    root = tmp_path
    (root / "slideA.svs").write_text("dummy", encoding="utf-8")

    features_dir = root / "features" / "modelX" / "20x" / "256px"
    features_dir.mkdir(parents=True)
    _write_h5(features_dir / "slideA.h5")

    result = mcp_server.generate_slide_qc_report(
        root_dir=str(root),
        slide_name="slideA",
        model="modelX",
        mag="20x",
        patch_size="256px",
    )

    assert "error" not in result
    report_path = Path(result["report_path"])
    summary_path = Path(result["summary_json_path"])
    assert report_path.exists()
    assert summary_path.exists()
    assert report_path.suffix == ".md"
    assert result["scores"]["grade"] in {"A", "B", "C", "D"}
    assert "summary_json" in result


def test_open_patch_exemplar_popup_cluster_dispatches_sampled_indices(monkeypatch) -> None:
    features = np.random.default_rng(3).normal(size=(30, 12)).astype(np.float32)
    labels = np.array(([0] * 15) + ([1] * 15), dtype=int)
    coords = np.column_stack([np.arange(30) * 10, np.arange(30) * 10]).astype(np.float32)
    app_state.update(
        features=features,
        cluster_labels=labels,
        coords_lv0=coords,
        patch_size_lv0=64.0,
        labeled_regions={1: LabeledRegionData(1, "R1", "#ff0000", [0, 1, 2], "local", 0)},
    )

    captured = {}

    def _fake_dispatch(action_type, params, timeout=5.0):
        captured["action_type"] = action_type
        captured["params"] = params
        return {"success": True, "shown_count": len(params.get("patch_indices", []))}

    monkeypatch.setattr(mcp_server, "_dispatch_gui_action", _fake_dispatch)

    result = mcp_server.open_patch_exemplar_popup(
        source_type="cluster",
        source_id=0,
        n_samples=6,
        strategy="centroid",
        include_boundary=False,
    )

    assert result["success"] is True
    assert captured["action_type"] == "open_patch_exemplar_popup"
    sampled = captured["params"]["patch_indices"]
    assert len(sampled) == 6
    assert all(labels[idx] == 0 for idx in sampled)

    app_state.update(
        features=None,
        cluster_labels=None,
        coords_lv0=None,
        patch_size_lv0=None,
        labeled_regions={},
    )


def test_detect_ood_patches_single_slide_returns_ranked_candidates() -> None:
    rng = np.random.default_rng(11)
    normal = rng.normal(loc=0.0, scale=1.0, size=(80, 8)).astype(np.float32)
    outliers = rng.normal(loc=12.0, scale=0.2, size=(1, 8)).astype(np.float32)
    feats = np.vstack([normal, outliers])
    coords = np.column_stack([np.arange(len(feats)) * 32, np.arange(len(feats)) * 16]).astype(np.float32)
    labels = np.zeros((len(feats),), dtype=int)

    app_state.update(
        slide_name="slideA",
        features=feats,
        cluster_labels=labels,
        coords_lv0=coords,
        patch_size_lv0=32.0,
    )

    result = mcp_server.detect_ood_patches(
        mode="single_slide",
        top_k=10,
        k_neighbors=10,
        threshold_mode="quantile",
        quantile=0.95,
        normalize_embeddings=False,
    )
    assert "error" not in result
    assert result["outlier_count"] > 0
    assert len(result["top_outliers"]) > 0
    top_idx = [int(x["patch_index"]) for x in result["top_outliers"][:5]]
    assert 80 in top_idx


def test_label_ood_patches_as_region_uses_last_detection(monkeypatch) -> None:
    feats = np.random.default_rng(1).normal(size=(40, 6)).astype(np.float32)
    coords = np.column_stack([np.arange(40) * 64, np.zeros((40,), dtype=int)]).astype(np.float32)
    app_state.update(
        slide_name="slideA",
        features=feats,
        cluster_labels=np.zeros((40,), dtype=int),
        coords_lv0=coords,
        patch_size_lv0=64.0,
    )

    # Seed last detection cache
    detect = mcp_server.detect_ood_patches(mode="single_slide", top_k=8, k_neighbors=5)
    assert "error" not in detect

    calls = []

    def _fake_dispatch(action_type, params, timeout=5.0):
        calls.append((action_type, params))
        return {"success": True, "region_id": len(calls), "patch_count": len(params.get("patch_indices", []))}

    monkeypatch.setattr(mcp_server, "_dispatch_gui_action", _fake_dispatch)
    labeled = mcp_server.label_ood_patches_as_region(
        from_last_detection=True,
        grouping_mode="single_region",
        name_prefix="OOD",
    )
    assert "error" not in labeled
    assert calls
    assert calls[0][0] == "create_agent_region"
    assert calls[0][1].get("select_dominant_cluster") is False
    assert labeled.get("exact_match") is True
    assert labeled.get("total_input_patches") == labeled.get("total_labeled_patches")


def test_find_most_distinct_cluster_works_without_regions() -> None:
    centroids = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(([0] * 10) + ([1] * 10) + ([2] * 10), dtype=int)
    feats = np.random.default_rng(9).normal(size=(30, 2)).astype(np.float32)
    coords = np.column_stack([np.arange(30) * 8, np.arange(30) * 8]).astype(np.float32)
    app_state.update(
        features=feats,
        cluster_labels=labels,
        cluster_centroids=centroids,
        coords_lv0=coords,
        labeled_regions={},
    )

    result = mcp_server.find_most_distinct_cluster(metric="euclidean", top_k=2)
    assert "error" not in result
    assert result["most_distinct_cluster_id"] == 2
    assert len(result["ranked_clusters"]) == 2


def test_generate_cross_slide_qc_report_ranks_most_shifted_slide_as_most_ood(tmp_path: Path) -> None:
    root = tmp_path
    for slide in ("slideA", "slideB", "slideC"):
        (root / f"{slide}.svs").write_text("dummy", encoding="utf-8")

    features_dir = root / "features" / "modelX" / "20x" / "256px"
    features_dir.mkdir(parents=True)
    _write_h5_with_shift(features_dir / "slideA.h5", shift=0.0)
    _write_h5_with_shift(features_dir / "slideB.h5", shift=0.1)
    _write_h5_with_shift(features_dir / "slideC.h5", shift=5.5)

    result = mcp_server.generate_cross_slide_qc_report(
        root_dir=str(root),
        model="modelX",
        mag="20x",
        patch_size="256px",
        slide_selection_mode="all",
        atlas_n_clusters=6,
        normalize_embeddings=False,
        ood_k_neighbors=10,
        ood_threshold_mode="global_quantile",
        ood_quantile=0.98,
        top_k_per_slide=20,
    )

    assert "error" not in result
    report_path = Path(result["report_path"])
    summary_path = Path(result["summary_json_path"])
    assert report_path.exists()
    assert summary_path.exists()
    assert result["most_ood_slide"]["slide_name"] == "slideC"
    per_slide_scores = result["summary_json"]["scores"]["per_slide"]
    assert len(per_slide_scores) == 3

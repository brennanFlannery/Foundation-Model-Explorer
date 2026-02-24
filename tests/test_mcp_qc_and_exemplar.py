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

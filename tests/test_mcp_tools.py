from __future__ import annotations

from pathlib import Path

import h5py

from chat.mcp_server import inspect_h5, list_data


def test_inspect_h5_returns_dataset_summary(tmp_path: Path):
    root = tmp_path
    features_dir = root / "features" / "modelX" / "20x" / "256px"
    features_dir.mkdir(parents=True)
    h5_file = features_dir / "slideA.h5"

    with h5py.File(h5_file, "w") as handle:
        handle.create_dataset("features", data=[[1.0, 2.0], [3.0, 4.0]])
        handle.attrs["patch_size_level0"] = 256

    result = inspect_h5(str(root), "features/modelX/20x/256px/slideA.h5")

    assert result["datasets"]
    assert result["attributes"]["patch_size_level0"] == 256


def test_list_data_returns_summary(tmp_path: Path):
    root = tmp_path
    (root / "slideA.svs").write_text("dummy", encoding="utf-8")

    features_dir = root / "features" / "modelX" / "20x" / "256px"
    features_dir.mkdir(parents=True)
    h5_file = features_dir / "slideA.h5"

    with h5py.File(h5_file, "w") as handle:
        handle.create_dataset("features", data=[[1.0, 2.0], [3.0, 4.0]])
        handle.create_dataset("coords", data=[[0, 0], [10, 10]])
        handle.attrs["patch_size_level0"] = 256

    result = list_data(str(root))

    assert "slideA" in result["slides"]
    assert "modelX" in result["models"]
    assert result["count_summary"]["slides"] == 1

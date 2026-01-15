# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FoundationalDetector is a Python desktop application for interactive visualization of patch-level features from whole-slide images (WSI) in digital pathology. It provides dual synchronized views: a spatial slide view showing WSI thumbnails with cluster-colored patch overlays, and a scatter view displaying PCA-reduced feature embeddings.

## Commands

```bash
# Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Run the application
python main.py
```

No build, lint, or test commands are configured. Tests would use pytest if added.

## Architecture

**MVC Pattern with Qt Signal/Slot Communication**

```
main.py                    Entry point, Qt app init, OpenSlide DLL config (Windows)
    ↓
gui.py (MainWindow)        Controller - coordinates data loading, clustering, view sync, GeoJSON export
    ↓
┌───────────────┐          ┌─────────────────┐
│ Data Layer    │          │ View Layer      │
├───────────────┤          ├─────────────────┤
│ data_loader   │          │ slide_view      │  WSI thumbnail + patch overlays, radial sweep animation
│ utils         │          │ scatter_view    │  2D PCA scatter plot with interactive points
│               │          │ tile_manager    │  Multi-resolution tile loading (adaptive zoom)
└───────────────┘          └─────────────────┘
```

**Data Flow**: User selects directory → `data_loader.parse_root_directory()` discovers slides/features → user picks slide/model/magnification → `data_loader.load_features()` reads HDF5 → K-means clustering → PCA reduction → views render synchronized visualizations.

**View Synchronization**: Qt signals connect interactions across views (e.g., `SlideGraphicsView.cluster_selected` → `MainWindow._update_scatter_for_cluster`).

## Expected Data Directory Structure

```
root_dir/
  slideA.svs                    # WSI files at root
  features/
    modelX/                     # Model name
      20x/                      # Magnification
        256px/                  # Patch size
          slideA.h5             # HDF5 with features (n_patches, feature_dim) and coords (n_patches, 2-3)
```

HDF5 datasets are auto-detected by shape inspection, not fixed names.

## Key Implementation Details

- **Windows OpenSlide**: DLL path configured in `main.py`; falls back to openslide-bin package
- **Thumbnail rendering**: Default mode magnifies thumbnail pixels; adaptive mode uses TileManager for true multi-resolution zoom
- **Performance**: Individual QGraphicsItem per patch/point; can be slow with >100k items
- **Clustering**: Uses scikit-learn K-means with MiniBatchKMeans fallback for large datasets
- **GeoJSON export**: Outputs cluster annotations compatible with QuPath

## Dependencies

PySide6 (Qt 6 GUI), scikit-learn (K-means/PCA), h5py (HDF5), openslide-python (WSI reading), Pillow (images), shapely (GeoJSON geometry).

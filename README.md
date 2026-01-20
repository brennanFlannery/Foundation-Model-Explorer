# FoundationalDetector README

An interactive visualization tool for exploring whole-slide image (WSI) patch features through clustering and embedding visualization. FoundationDetector provides a dual-view interface combining spatial patch visualization with feature space exploration.

## Features

- **Interactive Dual-View Visualization**
  - **Slide View**: Display whole-slide image thumbnails with color-coded patch overlays representing cluster assignments
  - **Scatter Plot View**: 2D PCA embedding visualization where each point represents a patch, colored by cluster
  - **Cluster Legend**: Interactive widget showing cluster colors, counts, with rename/export/toggle capabilities

- **Bidirectional Highlighting**
  - Click or hover on slide patches to highlight corresponding scatter points
  - Click or hover on scatter points to highlight corresponding slide patches
  - Real-time synchronization between views
  - Hover popups show patch details (index, coords, cluster, distance to centroid)

- **Animated Cluster Selection**
  - Radial sweep animation when selecting clusters, highlighting patches in order of distance from click point
  - Visual cascade effects for better spatial understanding
  - Synchronized animations across both views

- **Dynamic Clustering**
  - Adjustable number of clusters (2-10) with real-time K-means recomputation
  - No need to reload data when changing cluster count
  - Cluster centroid computation for distance calculations

- **Multi-Model Support**
  - Load and compare features from different models, magnifications, and patch sizes
  - Hierarchical selection: Slide → Model → Magnification → Patch Size
  - Multi-select models for concatenated analysis (checkbox dropdown)

- **Interactive Navigation**
  - Zoom with mouse wheel (both views)
  - Pan with middle mouse button (both views)
  - Scatter view docking/undocking with position persistence
  - Toggle patch overlay visibility

- **Advanced Export Capabilities**
  - Export individual clusters as GeoJSON with polygon merging
  - Support for custom cluster names
  - Multi-select export (all clusters or selected set)

- **Adaptive Zoom Modes**
  - **Thumbnail Mode**: Traditional single-image display with fast loading
  - **Adaptive Mode**: Multi-resolution tiled rendering using OpenSlide pyramid for seamless zoom from overview to cellular resolution
  - Automatic LOD updates during zoom/pan

- **Preferences and Settings**
  - Persistent application settings via QSettings
  - Configurable cache sizes and behavior

- **Hover and Popup System**
  - Translucent popups near cursor showing patch information
  - Edge avoidance with intelligent positioning
  - Fade in/out animations
  - Separate popups for slide and scatter views

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS

### Dependencies

All dependencies are listed in `requirements.txt`:

- **PySide6** (≥6.6.0): GUI framework
- **Pillow** (≥9.5.0): Image processing
- **openslide-python** (≥1.2.0): OpenSlide Python bindings
- **openslide-bin** (≥4.0.0): OpenSlide binary libraries (required on Windows)
- **scikit-learn** (≥1.2.0): Machine learning (K-means clustering)
- **h5py** (≥3.8.0): HDF5 file reading
- **pandas** (≥1.5.0): Data handling
- **shapely** (≥2.0.0): Geometry operations for GeoJSON export
- **pytest** (≥7.3.0): Optional, for unit testing

## Installation

### 1. Clone or Download Repository

```bash
git clone <repository-url>
cd FoundationalDetector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Windows-Specific: OpenSlide Configuration

On Windows, if you have manually installed OpenSlide (not via pip), you may need to configure the DLL path in `main.py`. The default path is:

```python
openslide_dll_dir = r"C:\Users\Brennan\OpenSlide\win32-x86-64"
```

If your OpenSlide installation is in a different location, modify this path in `main.py`. If you installed `openslide-bin` via pip, this step is usually not necessary.

## Usage

### Running the Application

```bash
python main.py
```

### Expected Directory Structure

The application expects a specific directory structure containing whole-slide images and associated feature files:

```
root_dir/
    slideA.svs                    # WSI files at root
    features/                       # Feature files directory
        modelX/                      # Model name
            20x/                   # Magnification
                256px/               # Patch size
                    slideA.h5           # HDF5 with features and coords
```

### HDF5 File Format

Each `.h5` file should contain:

1. **Feature Dataset**: 2D array of shape `(num_patches, feature_dim)` where `feature_dim` > 10
2. **Coordinate Dataset**: 2D array of shape `(num_patches, 2)` containing patch top-left corner coordinates in level-0 space
3. **Patch Size Attribute**: Attribute named `patch_size_level0`, `patch_size_lv0`, or `patch_size`

The loader automatically detects these datasets by examining array shapes, so specific dataset names are not required.

### Using the Application

1. **Select Root Folder**: Click "Open Folder" and choose directory containing your slides and features
2. **Choose Configuration**: Use dropdown menus to select:
   - Slide name
   - Model name (multi-select available for concatenated analysis)
   - Magnification level
   - Patch size
3. **Adjust Clusters**: Use "Clusters" spinbox to set number of clusters (2-10)
4. **Interact with Views**:
   - **Click** on slide patches or scatter points to select clusters
     - Left-click: single select
     - Ctrl+click: multi-select (add to selection)
   - **Hover** over patches/points to see cross-highlighting and detailed popups
   - **Scroll** mouse wheel to zoom
   - **Middle-click + drag** to pan (both views)
   - **Toggle** overlays visibility, cluster names, and adaptive zoom mode
5. **Export Results**:
   - Right-click cluster legend for options
   - Export individual clusters or all clusters as GeoJSON
   - Custom cluster names are preserved in exports

## Project Structure

```
FoundationalDetector/
├── main.py                    # Application entry point
├── gui.py                     # Main window and controller logic
├── data_loader.py              # WSI and HDF5 file loading utilities
├── slide_view.py               # Whole-slide image view with patch overlays
├── scatter_view.py             # Scatter plot view for feature embeddings
├── tile_manager.py             # Multi-resolution tile loading and caching
├── utils.py                   # Utility functions (clustering, colors, coordinates)
├── preferences_dialog.py        # Application settings and preferences
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Module Descriptions

- **main.py**: Initializes Qt application, handles OpenSlide DLL configuration (Windows), and launches the main window
- **gui.py**: Main application controller coordinating data loading, clustering, and view synchronization. Manages all UI components and signal connections
- **data_loader.py**: Parses directory structures, loads HDF5 feature files, generates slide thumbnails, and scales coordinates between zoom levels
- **slide_view.py**: Displays whole-slide images with interactive patch overlays. Supports both thumbnail and adaptive multi-resolution tiled rendering with animated cluster selection
- **scatter_view.py**: Displays 2D PCA embeddings as interactive scatter plots with zoom, pan, and selection support
- **tile_manager.py**: Provides multi-resolution tile loading, caching, and background threading for adaptive zoom mode
- **utils.py**: Provides clustering (K-means), color palette generation, coordinate transformations, and radial ordering for animations
- **preferences_dialog.py**: Persistent settings dialog for application configuration

## Technical Details

### Architecture

The application follows a Model-View-Controller (MVC) pattern:

- **Model**: Data loaded via `data_loader` module, features stored in memory, clustering via `utils.cluster_features`
- **View**: Two main views (`SlideGraphicsView` and `ScatterGraphicsView`) for spatial and feature space visualization
- **Controller**: `MainWindow` class coordinates user interactions, manages data loading, performs clustering, and synchronizes highlighting

### Signal/Slot Communication

The application uses Qt's signal/slot mechanism for decoupled communication:

- `SlideGraphicsView.cluster_selected` → `MainWindow._update_scatter_for_cluster`
- `SlideGraphicsView.patch_hovered` → `MainWindow._on_slide_patch_hovered`
- `ScatterGraphicsView.cluster_selected` → `MainWindow._on_scatter_cluster_selected`
- `ScatterGraphicsView.point_hovered` → `MainWindow._on_scatter_point_hovered`
- `ClusterLegendWidget.cluster_clicked/toggled/export/rename` → MainWindow handlers

### Data Flow

1. User selects root directory → `parse_root_directory()` discovers slides and feature files
2. User selects slide/model/mag/patch → `load_features()` reads HDF5 file
3. Features are clustered using K-means → `cluster_features()` assigns labels
4. PCA reduces features to 2D → `PCA.fit_transform()` creates embedding
5. Coordinates scaled appropriately for each view → thumbnails vs scene coordinates
6. Views updated → Slide view shows patches, scatter view shows embedding points
7. User interactions trigger signals → MainWindow coordinates cross-highlighting and animations
8. Export functionality → GeoJSON generation with polygon merging via shapely

### Animation System

Both views implement synchronized animated highlighting:

- **Slide View**: Radial sweep animation from click point, revealing patches in distance order
- **Scatter View**: Cascade animation highlighting cluster points in radial order from scene center
- **Timing**: Configurable intervals and batch updates for smooth performance
- **Synchronization**: `patches_highlighted` signal coordinates opacity changes between views

### Tile Management (Adaptive Mode)

- **LRU Cache**: Memory-limited caching with configurable size limit
- **Background Loading**: Separate thread for tile processing to maintain UI responsiveness
- **Priority Ordering**: Tiles near viewport center loaded first
- **LOD Management**: Automatic level selection based on view scale
- **Prefetching**: Loads tiles just outside viewport for smoother panning

## User Interface Components

### Main Window Layout

- **Left Panel**: Controls for data selection, clustering, and cluster legend
  - Slide/model/magnification/patch dropdowns
  - Clusters spinbox with real-time recomputation
  - Interactive cluster legend with counts and custom names
  - Overlay and adaptive zoom toggles

- **Center Area**: Slide view with patch overlays and interactive highlighting

- **Floating Dock**: Scatter plot view (dockable/undockable)
  - Auto-positioning near main window
  - Independent zoom/pan controls

### Cluster Legend Features

- **Visual**: Color swatches with cluster IDs and patch counts
- **Interactive**: Left-click to select, right-click for context menu
- **Checkboxes**: Toggle cluster visibility in views
- **Context Menu**: Rename cluster, export single/all clusters
- **Custom Names**: User-defined cluster names persisted in session

### Hover Popup System

- **Content**: Patch index, coordinates, cluster assignment, distance to centroid
- **Behavior**: Fade in/out animations, edge avoidance
- **Dual Popups**: Separate for slide and scatter views
- **Positioning**: Intelligent placement near cursor without going off-screen

## Troubleshooting

### OpenSlide DLL Not Found (Windows)

**Error**: `OSError: [WinError 126] The specified module could not be found`

**Solution**: 
1. Ensure `openslide-bin` is installed: `pip install openslide-bin`
2. If using a manual OpenSlide installation, update the DLL path in `main.py`
3. Verify the OpenSlide DLL directory exists and contains the required `.dll` files

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

### HDF5 File Not Found

**Error**: `FileNotFoundError` or `Could not find features or coords dataset`

**Solution**:
1. Verify the HDF5 file exists at the expected path
2. Check that the file contains datasets with the expected shapes:
   - One 2D dataset with >10 columns (features)
   - One 2D dataset with 2 columns (coordinates)
3. Ensure the directory structure matches the expected layout

### Slide Image Cannot Be Loaded

**Error**: `Failed to load thumbnail` or OpenSlide errors

**Solution**:
1. Verify the slide file exists and is a supported format (.svs, .tif, .tiff, .ndpi)
2. Check that OpenSlide is properly installed: `python -c "import openslide; print(openslide.__file__)"`
3. For corrupted files, the application will fall back to PIL, which may be slower

### Clustering Takes Too Long

**Solution**: 
- Reduce the number of patches (use larger patch sizes)
- Reduce the number of clusters
- The application shows a progress bar during loading; clustering itself is typically fast

### Views Not Synchronizing

**Solution**:
- Ensure both views have data loaded (check that slide/model/mag/patch are all selected)
- Try reloading the data by changing the patch size or reselecting the slide
- Check the console for debug messages indicating signal connections

### Scatter View Panning Not Working

**Solution**:
- Ensure middle mouse button is held while dragging
- Try re-docking the scatter view (close and reopen from View menu)
- Check console for "middle mouse button pressed/released" debug messages

### Performance Issues with Large Datasets

**Solution**:
- Enable adaptive zoom mode for better performance at high zoom levels
- Adjust tile cache size in preferences if memory is limited
- Use larger patch sizes to reduce the number of points/patches

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_module.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.
```

### Code Style

Follow these guidelines when contributing:

- Use `from __future__ import annotations` at the top of all Python files
- Use type hints consistently for all function parameters and return values
- Group imports: standard library, then third-party, then local imports
- Follow Qt naming conventions for signals and slots
- Use descriptive docstrings with reStructuredText format
- Implement proper error handling with specific exceptions

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]

## Reports

Detailed API documentation is available in `Reports/API_Documentation.md` for developers seeking comprehensive function and class references.
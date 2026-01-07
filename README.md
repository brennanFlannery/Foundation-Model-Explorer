# FoundationDetector

An interactive visualization tool for exploring whole-slide image (WSI) patch features through clustering and embedding visualization. FoundationDetector provides a dual-view interface combining spatial patch visualization with feature space exploration.

## Features

- **Interactive Dual-View Visualization**
  - **Slide View**: Display whole-slide image thumbnails with color-coded patch overlays representing cluster assignments
  - **Scatter Plot View**: 2D PCA embedding visualization where each point represents a patch, colored by cluster

- **Bidirectional Highlighting**
  - Click or hover on slide patches to highlight corresponding scatter points
  - Click or hover on scatter points to highlight corresponding slide patches
  - Real-time synchronization between views

- **Animated Cluster Selection**
  - Radial sweep animation when selecting clusters, highlighting patches in order of distance from click point
  - Visual cascade effects for better spatial understanding

- **Dynamic Clustering**
  - Adjustable number of clusters (2-10) with real-time K-means recomputation
  - No need to reload data when changing cluster count

- **Multi-Model Support**
  - Load and compare features from different models, magnifications, and patch sizes
  - Hierarchical selection: Slide → Model → Magnification → Patch Size

- **Interactive Navigation**
  - Zoom with mouse wheel (both views)
  - Pan with middle mouse button (slide view)
  - Toggle patch overlay visibility

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
- **pytest** (≥7.3.0): Optional, for unit testing

## Installation

### 1. Clone or Download the Repository

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
    slideA/
        slideA.svs              # Whole-slide image (supports .svs, .tif, .tiff, .ndpi)
        features/                # Feature files directory
            model1/              # Model name (e.g., "resnet50")
                20x/             # Magnification level
                    256px/       # Patch size
                        slideA.h5
                    512px/
                        slideA.h5
                40x/
                    256px/
                        slideA.h5
            model2/
                ...
    slideB/
        slideB.svs
        features/
            ...
```

### HDF5 File Format

Each `.h5` file should contain:

1. **Feature Dataset**: 2D array of shape `(num_patches, feature_dim)` where `feature_dim` > 10
2. **Coordinate Dataset**: 2D array of shape `(num_patches, 2)` or `(num_patches, 3)` containing patch top-left corner coordinates in level-0 space
3. **Patch Size Attribute** (optional): Attribute named `patch_size_level0`, `patch_size_lv0`, or `patch_size` on either the coordinate dataset or root group

The loader automatically detects these datasets by examining array shapes, so specific dataset names are not required.

### Using the Application

1. **Select Root Folder**: Click "Select Folder" and choose the directory containing your slides and features
2. **Choose Configuration**: Use the dropdown menus to select:
   - Slide name
   - Model name
   - Magnification level
   - Patch size
3. **Adjust Clusters**: Use the "Clusters" spinbox to set the number of clusters (2-10)
4. **Interact with Views**:
   - **Click** on slide patches or scatter points to select clusters
   - **Hover** over patches/points to see cross-highlighting
   - **Scroll** mouse wheel to zoom
   - **Middle-click + drag** to pan (slide view only)
   - **Toggle** "Show overlays" checkbox to hide/show patch rectangles

## Project Structure

```
FoundationalDetector/
├── main.py              # Application entry point
├── gui.py               # Main window and controller logic
├── data_loader.py       # WSI and HDF5 file loading utilities
├── slide_view.py        # Whole-slide image view with patch overlays
├── scatter_view.py      # Scatter plot view for feature embeddings
├── utils.py             # Utility functions (clustering, colors, etc.)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Module Descriptions

- **main.py**: Initializes Qt application, handles OpenSlide DLL configuration (Windows), and launches the main window
- **gui.py**: Main application controller coordinating data loading, clustering, and view synchronization
- **data_loader.py**: Parses directory structures, loads HDF5 feature files, generates slide thumbnails, and scales coordinates
- **slide_view.py**: Displays whole-slide image thumbnails with interactive patch overlays and radial sweep animations
- **scatter_view.py**: Displays 2D PCA embeddings as interactive scatter plots with zoom and selection support
- **utils.py**: Provides clustering (K-means), color palette generation, coordinate transformations, and radial ordering

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

### Data Flow

1. User selects root directory → `parse_root_directory()` discovers slides and feature files
2. User selects slide/model/mag/patch → `load_features()` reads HDF5 file
3. Features are clustered using K-means → `cluster_features()` assigns labels
4. PCA reduces features to 2D → `PCA.fit_transform()` creates embedding
5. Coordinates scaled to thumbnail → `scale_coords_to_thumbnail()` transforms coordinates
6. Views updated → Slide view shows patches, scatter view shows embedding points
7. User interactions trigger signals → MainWindow coordinates cross-highlighting

### Animation System

Both views implement animated highlighting:

- **Slide View**: Radial sweep animation from click point, revealing patches in distance order
- **Scatter View**: Cascade animation highlighting cluster points in radial order from scene center

Animations use `QTimer` with configurable intervals and batch updates for smooth performance.

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
   - One 2D dataset with 2-3 columns (coordinates)
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

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]

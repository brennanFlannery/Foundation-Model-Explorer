# Auto-split from original gui.py on 2025-08-17T19:45:46
"""
gui.py
======

Main GUI module for the FoundationDetector application.

This module provides the MainWindow class, which serves as the central
controller for the application's user interface. It coordinates between
data loading, feature visualization, clustering, and interactive exploration
of whole-slide image patches.

Architecture
------------

The GUI follows a Model-View-Controller pattern:

- **Model**: Data is loaded via `data_loader` module, which parses directory
  structures and reads HDF5 feature files. Features are stored in memory
  and clustered using K-means (via `utils.cluster_features`).

- **View**: Two main views are provided:
  - `SlideGraphicsView`: Displays the whole-slide image thumbnail with
    color-coded patch overlays representing cluster assignments
  - `ScatterGraphicsView`: Displays a 2D PCA embedding of patch features
    as an interactive scatter plot

- **Controller**: The `MainWindow` class coordinates user interactions,
  manages data loading, performs clustering, and synchronizes highlighting
  between the two views.

Key Features
------------

1. **Interactive Visualization**: Click on slide patches or scatter points
   to highlight corresponding items in the other view.

2. **Bidirectional Highlighting**: Hovering over a slide patch highlights
   its corresponding scatter point, and vice versa.

3. **Animated Cluster Selection**: When a cluster is selected, patches
   are highlighted with a radial sweep animation from the click point.

4. **Dynamic Clustering**: Adjust the number of clusters using a spinbox,
   and the clustering is recomputed in real-time without reloading data.

5. **Multi-Model Support**: Load and compare features from different models,
   magnifications, and patch sizes for the same slide.

Signal/Slot Architecture
-------------------------

The application uses Qt's signal/slot mechanism for communication:

- `SlideGraphicsView.cluster_selected` → `MainWindow._update_scatter_for_cluster`
- `SlideGraphicsView.patch_hovered` → `MainWindow._on_slide_patch_hovered`
- `ScatterGraphicsView.cluster_selected` → `MainWindow._on_scatter_cluster_selected`
- `ScatterGraphicsView.point_hovered` → `MainWindow._on_scatter_point_hovered`

This decoupled design allows the views to remain independent while the
MainWindow coordinates their interactions.

Classes
-------

MainWindow
    The main application window that orchestrates all GUI components and
    user interactions.
"""
from __future__ import annotations
import json
import os
import uuid
from typing import Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from PySide6.QtCore import Qt, QTimer, QPointF, QObject
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGraphicsEllipseItem,
    QCheckBox,
    QProgressBar,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import QSettings, QAbstractListModel, QModelIndex
from PySide6.QtCore import Qt as QtCore
from sklearn.decomposition import PCA
from PySide6.QtGui import QPen, QBrush
from PySide6.QtCore import QRectF
from PIL import Image
import data_loader
from utils import generate_palette, cluster_features, infer_slide_dims, radial_sweep_order
from PySide6.QtCore import Signal

from scatter_view import ScatterGraphicsItem, ScatterGraphicsView
from slide_view import SlideGraphicsView


class CheckableComboBoxModel(QAbstractListModel):
    """Model for QComboBox that supports checkable items."""
    
    def __init__(self, items: List[str] = None, parent=None):
        super().__init__(parent)
        self._items = items or []
        self._checked = {item: False for item in self._items}
        self._enabled = {item: True for item in self._items}
        self._tooltips = {item: "" for item in self._items}
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._items)
    
    def data(self, index, role=QtCore.DisplayRole):
        if not index.isValid():
            return None
        
        item = self._items[index.row()]
        
        if role == QtCore.DisplayRole:
            return item
        elif role == QtCore.CheckStateRole:
            return QtCore.Checked if self._checked[item] else QtCore.Unchecked
        elif role == QtCore.ToolTipRole:
            return self._tooltips.get(item, "")
        
        return None
    
    def setData(self, index, value, role=QtCore.CheckStateRole):
        if not index.isValid():
            return False
        
        item = self._items[index.row()]
        
        if role == QtCore.CheckStateRole and self._enabled[item]:
            self._checked[item] = (value == QtCore.Checked)
            self.dataChanged.emit(index, index, [role])
            return True
        
        return False
    
    def flags(self, index):
        if not index.isValid():
            return QtCore.NoItemFlags
        
        item = self._items[index.row()]
        flags = QtCore.ItemIsUserCheckable
        
        if self._enabled[item]:
            flags |= QtCore.ItemIsEnabled
        
        return flags
    
    def setItems(self, items: List[str]):
        """Set the list of items."""
        self.beginResetModel()
        self._items = items
        self._checked = {item: False for item in items}
        self._enabled = {item: True for item in items}
        self._tooltips = {item: "" for item in items}
        self.endResetModel()
    
    def getCheckedItems(self) -> List[str]:
        """Return list of checked items."""
        return [item for item in self._items if self._checked[item]]
    
    def setItemEnabled(self, item: str, enabled: bool):
        """Enable or disable a specific item."""
        if item in self._enabled:
            self._enabled[item] = enabled
            if not enabled:
                self._checked[item] = False
            # Find index and emit dataChanged
            try:
                idx = self._items.index(item)
                model_index = self.index(idx)
                self.dataChanged.emit(model_index, model_index)
            except ValueError:
                pass
    
    def setItemToolTip(self, item: str, tooltip: str):
        """Set tooltip for a specific item."""
        if item in self._tooltips:
            self._tooltips[item] = tooltip


class ModelMultiSelector(QComboBox):
    """Dropdown combobox with checkable items for multi-model selection.
    
    This widget looks like a standard QComboBox but allows multiple
    selections via checkboxes in the dropdown list. Selected models
    are displayed as comma-separated text when the dropdown is closed.
    
    Signals
    -------
    selectionChanged : Signal
        Emitted when the set of selected models changes.
    """
    selectionChanged = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up model
        self._model = CheckableComboBoxModel(parent=self)
        self.setModel(self._model)
        
        # Prevent default selection behavior
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select models...")
        
        # Connect model changes to update display text
        self._model.dataChanged.connect(self._update_text)
        
        # Prevent closing dropdown on item click
        self.view().viewport().installEventFilter(self)
        
        # Track selection state for deferred updates
        self._selection_changed_while_open = False
        self._previous_selection = set()
    
    def showPopup(self):
        """Track selection state when popup opens."""
        self._previous_selection = set(self.getSelectedModels())
        self._selection_changed_while_open = False
        print(f"DEBUG: Dropdown opened with selection: {self._previous_selection}")
        super().showPopup()
    
    def hidePopup(self):
        """Emit selectionChanged signal only when popup closes if selection changed."""
        super().hidePopup()
        
        # Check if selection actually changed
        current_selection = set(self.getSelectedModels())
        if current_selection != self._previous_selection:
            print(f"DEBUG: Dropdown closed, selection changed from {self._previous_selection} to {current_selection}")
            self.selectionChanged.emit()
        else:
            print("DEBUG: Dropdown closed, no selection change")
    
    def eventFilter(self, obj, event):
        """Prevent dropdown from closing when clicking checkboxes."""
        if obj == self.view().viewport():
            if event.type() == event.Type.MouseButtonRelease:
                # Get the index at click position
                index = self.view().indexAt(event.pos())
                if index.isValid():
                    # Toggle check state
                    current_state = self._model.data(index, QtCore.CheckStateRole)
                    new_state = QtCore.Unchecked if current_state == QtCore.Checked else QtCore.Checked
                    self._model.setData(index, new_state, QtCore.CheckStateRole)
                    self._selection_changed_while_open = True  # Mark that change occurred
                return True  # Prevent dropdown from closing
        
        return super().eventFilter(obj, event)
    
    def _update_text(self):
        """Update the display text to show selected items.
        
        Note: Signal emission is deferred until dropdown closes (see hidePopup).
        """
        selected = self._model.getCheckedItems()
        if selected:
            self.lineEdit().setText(", ".join(selected))
        else:
            self.lineEdit().setText("")
        # Signal will be emitted only when dropdown closes
    
    def clear(self):
        """Clear all items."""
        self._model.setItems([])
        self.lineEdit().clear()
    
    def addItems(self, items: List[str]):
        """Add model names as checkable items."""
        self._model.setItems(items)
        self.lineEdit().clear()
    
    def getSelectedModels(self) -> List[str]:
        """Return list of selected model names."""
        return self._model.getCheckedItems()
    
    def getAllModelNames(self) -> List[str]:
        """Return list of all available model names (not just selected)."""
        return self._model._items.copy()
    
    def setModelEnabled(self, model_name: str, enabled: bool):
        """Enable or disable a specific model item."""
        self._model.setItemEnabled(model_name, enabled)
    
    def setModelToolTip(self, model_name: str, tooltip: str):
        """Set tooltip for a specific model item."""
        self._model.setItemToolTip(model_name, tooltip)

class MainWindow(QMainWindow):
    """Main application window for FoundationDetector.
    
    This class serves as the central controller for the GUI application,
    managing data loading, feature clustering, visualization, and user
    interactions. It coordinates two main views: a slide view showing
    whole-slide images with patch overlays, and a scatter plot view
    displaying 2D PCA embeddings of patch features.
    
    The window provides controls for:
    - Selecting a root directory containing slides and features
    - Choosing slides, models, magnifications, and patch sizes
    - Adjusting the number of clusters for K-means clustering
    - Toggling patch overlay visibility
    
    User interactions are synchronized between views:
    - Clicking a slide patch highlights the corresponding scatter point
    - Clicking a scatter point highlights all patches in that cluster
    - Hovering over either view highlights the corresponding item in the other
    
    Attributes
    ----------
    slides : Dict[str, data_loader.SlideInfo]
        Mapping from slide names to SlideInfo objects containing paths
        to image files and feature HDF5 files.
    _scatter_items : List[QGraphicsEllipseItem]
        Graphics items representing points in the scatter plot.
    _current_embedding : Optional[np.ndarray]
        Current 2D PCA embedding of features (shape: n_patches x 2).
    _current_features : np.ndarray
        Current feature vectors loaded from HDF5 (shape: n_patches x feature_dim).
    _current_coords_thumb : np.ndarray
        Patch coordinates scaled to thumbnail dimensions.
    _current_patch_size_thumb : float
        Patch size in thumbnail coordinate space.
    _current_coords_lv0 : Optional[np.ndarray]
        Patch coordinates at full resolution (level-0) for GeoJSON export.
    _current_patch_size_lv0 : Optional[float]
        Patch size at full resolution (level-0) for GeoJSON export.
    _coord_scale_factor : Optional[float]
        Scale factor from thumbnail to level-0 coordinates (slide_w / thumb_w).
    _active_cluster : Optional[int]
        Currently selected cluster number, used for persistent opacity styling.
    
    Methods
    -------
    _create_widgets()
        Initialize and layout all GUI widgets.
    _connect_signals()
        Connect Qt signals to slot methods for user interactions.
    _select_folder()
        Prompt user to select root directory and parse slide structure.
    _load_current_data(recluster_only=False)
        Load features, compute clusters, and update both views.
    _update_scatter_for_cluster(cluster, click_point)
        Prepare scatter plot for synchronized cascade animation.
    _on_scatter_cluster_selected(cluster)
        Handle scatter plot cluster selection.
    _on_slide_patch_hovered(idx, state)
        Handle hover events from slide view.
    _on_scatter_point_hovered(idx, state)
        Handle hover events from scatter view.
    _on_patches_highlighted(indices)
        Synchronize scatter point opacity with slide patch highlights.
    _on_animation_completed()
        Apply persistent styling when cascade animation finishes.
    _merge_cluster_patches(cluster)
        Merge adjacent patches into continuous polygons for GeoJSON export.
    _on_export_clicked()
        Handle export button click for GeoJSON annotation export.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FoundationDetector (Offline)")
        # Slide info mapping
        self.slides: Dict[str, data_loader.SlideInfo] = {}
        # Storage for scatter plot items
        self._scatter_items: List[QGraphicsEllipseItem] = []
        self._current_embedding: Optional[np.ndarray] = None
        # Load preferences
        self.settings = QSettings("FoundationDetector", "FoundationDetector")
        self.normalize_features = self.settings.value("normalize_features", True, type=bool)
        # UI components
        self._create_widgets()
        self._connect_signals()
        # Track hover state for slide-rect overlay
        self._hovered_slide_rect_idx = None
        self._hover_prev_opacity = 0.0
        self._hover_prev_brush = None
        # Track currently hovered scatter index when hovering slide patches
        self._hovered_scatter_idx: Optional[int] = None
        # Track active cluster for persistent opacity reduction
        self._active_cluster: Optional[int] = None
        # Store level-0 coordinates for GeoJSON export
        self._current_coords_lv0: Optional[np.ndarray] = None
        self._current_patch_size_lv0: Optional[float] = None
        self._coord_scale_factor: Optional[float] = None
        # Track cascade animation state to prevent hover interference
        self._animation_in_progress: bool = False

    def _create_widgets(self) -> None:
        """Create and lay out widgets for the main window."""
        # Create menu bar
        self._create_menu_bar()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        
        # Folder selection row (button removed - now in File menu)
        folder_row = QHBoxLayout()
        self.root_label = QLabel("No folder selected")
        folder_row.addWidget(self.root_label)
        main_vbox.addLayout(folder_row)
        
        # Drop‑down row
        dropdown_row = QHBoxLayout()
        self.slide_combo = QComboBox()
        self.slide_combo.setPlaceholderText("Select slide")
        dropdown_row.addWidget(QLabel("Slide:"))
        dropdown_row.addWidget(self.slide_combo)
        
        # Model multi-selector dropdown
        self.model_selector = ModelMultiSelector()
        dropdown_row.addWidget(QLabel("Models:"))
        dropdown_row.addWidget(self.model_selector)
        
        self.mag_combo = QComboBox()
        dropdown_row.addWidget(QLabel("Magnification:"))
        dropdown_row.addWidget(self.mag_combo)
        self.patch_combo = QComboBox()
        dropdown_row.addWidget(QLabel("Patch size:"))
        dropdown_row.addWidget(self.patch_combo)
        main_vbox.addLayout(dropdown_row)
        
        # Clusters control
        cluster_row = QHBoxLayout()
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 10)
        self.cluster_spin.setValue(5)
        cluster_row.addWidget(QLabel("Clusters:"))
        cluster_row.addWidget(self.cluster_spin)
        main_vbox.addLayout(cluster_row)
        
        # Progress bar for loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate by default
        self.progress_bar.setVisible(False)
        main_vbox.addWidget(self.progress_bar)
        
        # Status label showing selected models and feature dimensions
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        main_vbox.addWidget(self.status_label)
        # Create horizontal layout for the two panels
        panels_layout = QHBoxLayout()
        # Left panel: Slide view
        self.graphics_view = SlideGraphicsView()
        self.graphics_view.setMinimumWidth(400)
        # Connect signals: slide click selects cluster and updates scatter
        self.graphics_view.cluster_selected.connect(
            lambda cluster, pos: self._update_scatter_for_cluster(cluster, pos)
        )
        # Connect slide hover to scatter
        self.graphics_view.patch_hovered.connect(self._on_slide_patch_hovered)
        # Connect slide animation signals for synchronized scatter cascade
        self.graphics_view.patches_highlighted.connect(self._on_patches_highlighted)
        self.graphics_view.animation_completed.connect(self._on_animation_completed)
        panels_layout.addWidget(self.graphics_view, stretch=1)
        # Right panel: Scatter plot view
        self.scatter_view = ScatterGraphicsView()
        self.scatter_view.setMinimumWidth(300)
        # Connect scatter click to highlight slide
        self.scatter_view.cluster_selected.connect(self._on_scatter_cluster_selected)
        # Connect scatter hover to highlight slide and scatter
        self.scatter_view.point_hovered.connect(self._on_scatter_point_hovered)
        panels_layout.addWidget(self.scatter_view, stretch=1)
        # Add panels layout
        main_vbox.addLayout(panels_layout)
        # Set starting size
        self.resize(1200, 800)

    def _create_menu_bar(self) -> None:
        """Create menu bar with File, View, and Help menus."""
        # File Menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Open Folder (replaces select_folder_button)
        open_action = QAction("Open Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._select_folder)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Export GeoJSON (replaces export_button functionality)
        self.export_action = QAction("Export Cluster as GeoJSON...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.setEnabled(False)  # Enable when cluster selected
        self.export_action.triggered.connect(self._on_export_clicked)
        file_menu.addAction(self.export_action)
        
        file_menu.addSeparator()
        
        # Preferences
        prefs_action = QAction("Preferences...", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.triggered.connect(self._show_preferences)
        file_menu.addAction(prefs_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Show Patch Overlays (replaces overlay_checkbox)
        self.overlay_action = QAction("Show Patch Overlays", self)
        self.overlay_action.setShortcut("Ctrl+1")
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.triggered.connect(self._on_overlay_toggled)
        view_menu.addAction(self.overlay_action)
        
        # Help Menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About
        about_action = QAction("About FoundationDetector", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        """Connect signals and slots for interactive widgets."""
        self.slide_combo.currentTextChanged.connect(self._on_slide_changed)
        self.model_selector.selectionChanged.connect(self._on_model_selection_changed)
        self.mag_combo.currentTextChanged.connect(self._on_mag_changed)
        self.patch_combo.currentTextChanged.connect(self._on_patch_changed)
        self.cluster_spin.valueChanged.connect(self._on_cluster_changed)

    # ---- Handlers ----

    def _select_folder(self) -> None:
        """Prompt the user to select a root directory and parse its contents."""
        directory = QFileDialog.getExistingDirectory(self, "Select root folder", os.getcwd())
        if not directory:
            print("DEBUG: No directory selected")
            return
        print(f"DEBUG: Selected directory: {directory}")
        self.root_label.setText(directory)
        
        try:
            print("DEBUG: Attempting to parse directory...")
            self.slides = data_loader.parse_root_directory(directory)
            print(f"DEBUG: Found {len(self.slides)} slides: {list(self.slides.keys())}")
            
            # Populate slide combo
            self.slide_combo.clear()
            slide_names = sorted(self.slides.keys())
            print(f"DEBUG: Adding slides to combo box: {slide_names}")
            self.slide_combo.addItems(slide_names)
            
            # Clear subsequent selectors and view
            self.model_selector.clear()
            self.mag_combo.clear()
            self.patch_combo.clear()
            self.graphics_view.scene().clear()
            if hasattr(self.graphics_view, 'rect_items'):
                self.graphics_view.rect_items.clear()
            
        except Exception as e:
            print(f"DEBUG: Error parsing directory: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to parse folder:\n{str(e)}")
            return

    def _on_slide_changed(self, slide_name: str) -> None:
        """Update model selector and mag/patch combos when a new slide is selected."""
        print(f"DEBUG: Slide changed to: {slide_name}")
        if not slide_name:
            print("DEBUG: No slide name provided")
            return
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: No slide info found for {slide_name}")
            return
        
        print(f"DEBUG: Found slide info: {info}")
        # Populate model selector
        self.model_selector.clear()
        model_names = sorted(info.models.keys())
        print(f"DEBUG: Adding models to selector: {model_names}")
        self.model_selector.addItems(model_names)
        
        self.mag_combo.clear()
        self.patch_combo.clear()

    def _on_model_selection_changed(self) -> None:
        """Update magnification combo when model selection changes."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return
        info = self.slides.get(slide_name)
        if info is None:
            return
        
        selected_models = self.model_selector.getSelectedModels()
        print(f"DEBUG: Selected models changed: {selected_models}")
        
        if len(selected_models) == 0:
            # No models selected - clear everything
            self.mag_combo.clear()
            self.patch_combo.clear()
            return
        
        # Find intersection of available magnifications across all selected models
        mag_sets = []
        for model_name in selected_models:
            model_dict = info.models.get(model_name, {})
            mag_sets.append(set(model_dict.keys()))
        
        common_mags = set.intersection(*mag_sets) if mag_sets else set()
        print(f"DEBUG: Common magnifications across selected models: {common_mags}")
        
        self.mag_combo.blockSignals(True)
        self.patch_combo.blockSignals(True)
        
        self.mag_combo.clear()
        if common_mags:
            self.mag_combo.addItems(sorted(common_mags))
        self.patch_combo.clear()
        
        self.mag_combo.blockSignals(False)
        self.patch_combo.blockSignals(False)
        
        # Trigger update for new magnification
        if self.mag_combo.count() > 0:
            self.mag_combo.setCurrentIndex(0)
            # Manually trigger the magnification changed signal
            self._on_mag_changed(self.mag_combo.currentText())
            
            # Explicitly trigger data loading if mag and patch are valid
            # (in case they didn't change but model selection did)
            if self.mag_combo.currentText() and self.patch_combo.currentText():
                print("DEBUG: Model selection changed with valid mag/patch; triggering data load")
                self._on_patch_changed(self.patch_combo.currentText())

    def _on_mag_changed(self, mag: str) -> None:
        """Update patch combo when magnification changes."""
        slide_name = self.slide_combo.currentText()
        selected_models = self.model_selector.getSelectedModels()
        if not slide_name or not selected_models or not mag:
            return
        info = self.slides.get(slide_name)
        if info is None:
            return

        # Find intersection of available patch sizes across all selected models
        patch_sets = []
        for model_name in selected_models:
            patch_sizes = info.models.get(model_name, {}).get(mag, {})
            patch_sets.append(set(patch_sizes.keys()))
        
        common_patches = set.intersection(*patch_sets) if patch_sets else set()
        print(f"DEBUG: Common patch sizes for magnification {mag}: {common_patches}")
        
        self.patch_combo.blockSignals(True)
        self.patch_combo.clear()
        if common_patches:
            self.patch_combo.addItems(sorted(common_patches))
        self.patch_combo.blockSignals(False)
        
        # Trigger update for new patch size
        if self.patch_combo.count() > 0:
            self.patch_combo.setCurrentIndex(0)
            self._on_patch_changed(self.patch_combo.currentText())

    def _on_patch_changed(self, patch: str) -> None:
        """Load new data when patch size changes."""
        # Validate model compatibility before loading
        self._validate_model_compatibility()
        self._load_current_data()

    def _on_cluster_changed(self, k: int) -> None:
        """Recompute clusters when the cluster count slider changes."""
        print(f"DEBUG: Cluster spin changed to {k}; reclustering current features")
        self._load_current_data(recluster_only=True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        """Show or hide patch overlays based on menu action state."""
        print(f"DEBUG: Overlay toggled; visible={checked}")
        # Adjust visibility of patch rects
        for rect in getattr(self.graphics_view, 'rect_items', []):
            rect.setVisible(checked)

    def _on_slide_patch_hovered(self, idx: int, state: bool) -> None:
        """Highlight the scatter point corresponding to the hovered slide patch."""
        if not self.scatter_view:
            return
        # Skip scatter opacity changes if animation is in progress
        if self._animation_in_progress:
            return
        # When hover leaves or invalid index
        if idx == -1 or not state:
            # Clear hover state in scatter
            if self._hovered_scatter_idx is not None:
                print(f"DEBUG: Slide hover leave for scatter index {self._hovered_scatter_idx}")
            # If no active cluster, reset all to default opacity
            if self._active_cluster is None:
                if self.scatter_view.labels is None:
                    return
                print("DEBUG: No active cluster; resetting scatter opacities to default")
                for item in self.scatter_view._scatter_items:
                    item.setOpacity(0.6)
            else:
                # Apply persistent opacity reduction for active cluster
                print(f"DEBUG: Maintaining active cluster {self._active_cluster} opacities on hover leave")
                self._apply_active_cluster_styles()
            self._hovered_scatter_idx = None
        else:
            # Hover over a valid patch
            # FIRST: Restore the previous hovered scatter point if transitioning
            if self._hovered_scatter_idx is not None and self._hovered_scatter_idx != idx:
                prev_idx = self._hovered_scatter_idx
                # Determine the correct baseline opacity for the previous point
                if self._active_cluster is not None and self.scatter_view.labels is not None:
                    prev_label = int(self.scatter_view.labels[prev_idx])
                    if prev_label == self._active_cluster:
                        # Previous point is in selected cluster → high opacity
                        self.scatter_view._scatter_items[prev_idx].setOpacity(1.0)
                    else:
                        # Previous point is NOT in selected cluster → low opacity
                        self.scatter_view._scatter_items[prev_idx].setOpacity(0.2)
                else:
                    # No active selection → medium opacity
                    self.scatter_view._scatter_items[prev_idx].setOpacity(0.6)
            
            # THEN: Update to the new hovered scatter point
            self._hovered_scatter_idx = idx
            print(f"DEBUG: Slide hover over patch {idx}; setting scatter point opacity to high")
            self.scatter_view.set_point_opacity(idx, True)

    
    def _on_scatter_point_hovered(self, idx: int, state: bool) -> None:
        """Highlight the slide patch corresponding to hovered scatter point.
        Red hover overlay is temporary and restores on leave."""
        # Helper to clear previous hover
        def _clear_previous_hover():
            if self._hovered_slide_rect_idx is not None:
                prev_idx = self._hovered_slide_rect_idx
                if 0 <= prev_idx < len(self.graphics_view.rect_items):
                    rect = self.graphics_view.rect_items[prev_idx]
                    # Restore original brush and opacity
                    if self._hover_prev_brush is not None:
                        rect.setBrush(self._hover_prev_brush)
                    rect.setOpacity(self._hover_prev_opacity)
            self._hovered_slide_rect_idx = None

        if not state or idx < 0 or idx >= len(self.graphics_view.rect_items):
            _clear_previous_hover()
            # Skip scatter opacity changes if animation is in progress
            # to avoid interrupting the cascade effect
            if self._animation_in_progress:
                return
            # Restore scatter opacities based on whether a cluster is active
            if self._active_cluster is None:
                # No active selection: return all points to medium opacity
                if self.scatter_view and self.scatter_view._scatter_items:
                    for item in self.scatter_view._scatter_items:
                        item.setOpacity(0.6)
            else:
                # Active selection persists low/high styling
                self._apply_active_cluster_styles()
            return

        # If we are switching items, clear the previous one
        if self._hovered_slide_rect_idx is not None and self._hovered_slide_rect_idx != idx:
            _clear_previous_hover()

        # Determine cluster label for hovered scatter point
        label = None
        if self.graphics_view.labels is not None and 0 <= idx < len(self.graphics_view.labels):
            label = int(self.graphics_view.labels[idx])
        # Apply colour-specific hover overlay to the target rect
        rect = self.graphics_view.rect_items[idx]
        # Save previous state (opacity + brush) once per hover start
        if self._hovered_slide_rect_idx is None:
            self._hover_prev_opacity = rect.opacity()
            self._hover_prev_brush = rect.brush()
        # Compute hover colour: lighten the cluster colour or default to red
        if label is not None and label < len(self.graphics_view.cluster_colors):
            base_col = self.graphics_view.cluster_colors[label]
            hover_col = self._lighter_color(base_col, factor=150)
        else:
            hover_col = QColor('red')
        rect.setBrush(QBrush(hover_col))
        rect.setOpacity(0.9)
        self._hovered_slide_rect_idx = idx

    def _on_scatter_cluster_selected(self, cluster: int) -> None:
        """Respond to cluster selection on scatter plot: highlight slide patches."""
        print(f"DEBUG: Scatter cluster selected {cluster}")
        # Set active cluster to maintain opacity reduction
        self._active_cluster = cluster
        # Enable export action now that a cluster is selected
        self.export_action.setEnabled(True)
        # Mark animation as in progress to prevent hover interference
        self._animation_in_progress = True
        self.scatter_view.set_animation_active(True)
        # Find cluster indices
        if self.graphics_view.labels is None:
            return
        cluster_indices = np.where(self.graphics_view.labels == cluster)[0]
        if cluster_indices.size == 0:
            return
        # Initialize scatter points to low opacity before cascade
        # The patches_highlighted signal will raise opacity for matching points
        for item in self.scatter_view._scatter_items:
            item.setOpacity(0.2)
        # Use the first patch's center as click point approximate
        idx0 = cluster_indices[0]
        x, y = self.graphics_view.coords[idx0] + self.graphics_view.patch_size / 2.0
        # Compute radial order and start slide animation
        cluster_coords = self.graphics_view.coords[cluster_indices]
        order_local = radial_sweep_order(cluster_coords, (x, y))
        order_global = cluster_indices[order_local]
        self.graphics_view._start_animation(cluster, order_global.tolist())
        # Scatter animation is now synchronized via patches_highlighted signal

    def _populate_scatter_scene(self, embedding: np.ndarray, labels: np.ndarray, colours: List[str]) -> None:
        scene = self.scatter_view.scene()
        scene.clear()
        self._scatter_items.clear()

        SCENE_SIZE = 500
        POINT_RADIUS = 3

        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

        margin = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

        scale = SCENE_SIZE / max(x_max - x_min, y_max - y_min)

        x = (embedding[:, 0] - x_min) * scale
        y = (embedding[:, 1] - y_min) * scale

        for i in range(len(embedding)):
            label = int(labels[i])
            color = self._hsl_string_to_qcolor(colours[label]) if label < len(colours) else QColor('red')

            ellipse = QGraphicsEllipseItem(
                x[i] - POINT_RADIUS,
                y[i] - POINT_RADIUS,
                2 * POINT_RADIUS,
                2 * POINT_RADIUS
            )
            ellipse.setBrush(QBrush(color))
            ellipse.setPen(QPen(Qt.NoPen))
            ellipse.setData(0, i)  # Store the index for later reference
            ellipse.setAcceptHoverEvents(True)  # Enable hover events
            scene.addItem(ellipse)
            self._scatter_items.append(ellipse)

        scene.setSceneRect(scene.itemsBoundingRect())
        self.scatter_view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def _highlight_corresponding_items(self, index: int) -> None:
        """Highlight corresponding items in both views.

        Parameters
        ----------
        index : int
            Index of the item to highlight
        """
        if not hasattr(self, '_current_features') or self.graphics_view.labels is None:
            return

        # Get the cluster label for this index
        label = int(self.graphics_view.labels[index])

        # Update scatter plot points
        for item in self._scatter_items:
            idx = item.data(0)
            item_label = int(self.graphics_view.labels[idx])
            if item_label == label:
                item.setBrush(QBrush(Qt.black))
            else:
                original_color = self._hsl_string_to_qcolor(
                    generate_palette(int(self.graphics_view.labels.max()) + 1)[item_label]
                )
                item.setBrush(QBrush(original_color))

        # Update slide view patches
        self.graphics_view._highlight_cluster(label)

    def _update_scatter_for_cluster(self, cluster: int, click_point: Tuple[float, float]) -> None:
        """Prepare scatter plot for synchronized cascade animation.
        
        This method sets up the scatter plot's initial state before the slide
        animation begins. The actual cascade animation is handled through the
        patches_highlighted signal from SlideGraphicsView, which ensures both
        views animate in perfect synchronization.
        """
        if not self.scatter_view._scatter_items or not hasattr(self, '_current_features'):
            return
        print(f"DEBUG: Preparing scatter for synchronized cascade, cluster {cluster}")
        # Set active cluster to maintain opacity reduction across panels
        self._active_cluster = cluster
        # Enable export action now that a cluster is selected
        self.export_action.setEnabled(True)
        # Mark animation as in progress to prevent hover interference
        self._animation_in_progress = True
        self.scatter_view.set_animation_active(True)

        # Baseline: set all scatter points to low opacity before cascade
        # The patches_highlighted signal will raise opacity for matching points
        for item in self.scatter_view._scatter_items:
            item.setOpacity(0.2)

    def _on_patches_highlighted(self, indices: list) -> None:
        """Synchronize scatter point opacity with slide patch highlights.
        
        This handler is called by the patches_highlighted signal from
        SlideGraphicsView during the cascade animation. It ensures that
        scatter points are highlighted at the exact same time as their
        corresponding slide patches.
        
        Parameters
        ----------
        indices : list
            List of patch indices that were highlighted in this animation step.
        """
        print(f"DEBUG: Synchronizing scatter for highlighted patches: {indices}")
        for idx in indices:
            if 0 <= idx < len(self.scatter_view._scatter_items):
                self.scatter_view._scatter_items[idx].setOpacity(1.0)

    def _on_animation_completed(self) -> None:
        """Apply persistent styling when cascade animation finishes.
        
        This handler is called by the animation_completed signal from
        SlideGraphicsView when the cascade animation finishes. It applies
        the persistent opacity styling to maintain the low/high distinction
        between selected and non-selected clusters.
        """
        print("DEBUG: Animation completed; applying persistent cluster styles")
        # Mark animation as complete
        self._animation_in_progress = False
        self.scatter_view.set_animation_active(False)
        self._apply_active_cluster_styles()

    def _merge_cluster_patches(self, cluster: int) -> List[List]:
        """Merge adjacent patches into continuous polygons.
        
        Works in thumbnail space for efficiency, then scales to level-0
        coordinates for QuPath-compatible GeoJSON export.
        
        Parameters
        ----------
        cluster : int
            The cluster number to merge patches for.
            
        Returns
        -------
        List[List]
            List of GeoJSON-ready polygon coordinate arrays. Each element
            is a list of rings (outer ring + any holes), where each ring
            is a list of [x, y] coordinate pairs in level-0 pixel units.
        """
        # Get patch indices for this cluster
        if self.graphics_view.labels is None:
            return []
        indices = np.where(self.graphics_view.labels == cluster)[0]
        if len(indices) == 0:
            return []
        
        # Create Shapely boxes in thumbnail space for efficiency
        boxes = []
        for idx in indices:
            x, y = self._current_coords_thumb[idx]
            size = self._current_patch_size_thumb
            boxes.append(box(x, y, x + size, y + size))
        
        # Merge all touching/overlapping boxes using unary_union
        merged = unary_union(boxes)
        
        # Scale factor from thumbnail to level-0 coordinates
        scale = self._coord_scale_factor
        
        # Handle Polygon vs MultiPolygon result
        if merged.geom_type == 'Polygon':
            polygons = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polygons = list(merged.geoms)
        else:
            # Unexpected geometry type (GeometryCollection, etc.)
            print(f"DEBUG: Unexpected geometry type from unary_union: {merged.geom_type}")
            polygons = []
        
        # Convert to GeoJSON coordinate format, scaled to level-0
        result = []
        for poly in polygons:
            coords = []
            # Process exterior ring and any interior rings (holes)
            for ring in [poly.exterior] + list(poly.interiors):
                # Scale coordinates to level-0 and convert to nested lists
                # Round to integers since QuPath expects pixel coordinates
                scaled_ring = [[int(x * scale), int(y * scale)] for x, y in ring.coords]
                coords.append(scaled_ring)
            result.append(coords)
        
        print(f"DEBUG: Merged {len(indices)} patches into {len(result)} polygon(s)")
        return result

    def _on_export_clicked(self) -> None:
        """Handle export button click: prompt for name and save GeoJSON.
        
        This method shows dialogs to get the annotation name and save
        location, then generates a QuPath-compatible GeoJSON file with
        merged polygon regions for the currently selected cluster.
        """
        if self._active_cluster is None:
            QMessageBox.warning(self, "No Selection", 
                "Please select a cluster first by clicking on the slide or scatter plot.")
            return
        
        # Check if we have the required coordinate data
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", 
                "Please load a slide first.")
            return
        
        # Dialog 1: Get annotation name
        default_name = f"Cluster_{self._active_cluster}"
        name, ok = QInputDialog.getText(
            self, "Annotation Name",
            "Enter a name for this annotation:",
            text=default_name
        )
        if not ok or not name.strip():
            return  # User cancelled
        
        annotation_name = name.strip()
        
        # Dialog 2: Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            f"{annotation_name}.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return  # User cancelled
        
        # Ensure .geojson extension
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'
        
        # Generate merged polygons
        polygons = self._merge_cluster_patches(self._active_cluster)
        
        if not polygons:
            QMessageBox.warning(self, "No Regions",
                "No patches found for the selected cluster.")
            return
        
        # Get cluster color for QuPath export
        colours = generate_palette(int(self.graphics_view.labels.max()) + 1)
        cluster_color_hsl = colours[self._active_cluster]
        color_rgb = self._hsl_to_qupath_rgb(cluster_color_hsl)
        
        # Build GeoJSON FeatureCollection with QuPath-compatible properties
        features = []
        for coords in polygons:
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": annotation_name,
                        "colorRGB": color_rgb
                    },
                    "isLocked": False,
                    "measurements": []
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Write file
        try:
            with open(file_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            QMessageBox.information(self, "Export Complete",
                f"Exported {len(features)} region(s) to:\n{file_path}")
            print(f"DEBUG: Exported GeoJSON with {len(features)} features to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                f"Failed to write file:\n{e}")
            print(f"DEBUG: Export failed: {e}")

    def _update_scatter_colours(self, labels: np.ndarray, colours: List[str]) -> None:
        """Update colours of existing scatter points based on new labels.

        Parameters
        ----------
        labels : np.ndarray
            New cluster labels
        colours : List[str]
            New colour palette
        """
        # In _update_scatter_colours method:
        for i, item in enumerate(self._scatter_items):
            label = int(labels[i])
            color = self._hsl_string_to_qcolor(colours[label]) if label < len(colours) else QColor('red')
            item.setBrush(QBrush(color))

    # --- persistent cluster styling ---
    def _apply_active_cluster_styles(self) -> None:
        """Apply opacity styling to scatter points based on active cluster.

        If an active cluster is set, cluster points are shown with high
        opacity (1.0) and other points with low opacity (0.2).  If no
        active cluster is set, all points revert to a default opacity
        (0.6).  This helper does not alter brushes or colours.
        """
        # If no scatter points are present, nothing to do
        if not self.scatter_view or not self.scatter_view._scatter_items:
            return
        # If there is no active cluster, restore base opacity for all points
        if self._active_cluster is None:
            for item in self.scatter_view._scatter_items:
                item.setOpacity(0.6)
            return
        # Apply cluster-specific opacities
        # Ensure labels are available
        if self.scatter_view.labels is None:
            return
        for i, item in enumerate(self.scatter_view._scatter_items):
            label = int(self.scatter_view.labels[i])
            if label == self._active_cluster:
                item.setOpacity(1.0)
            else:
                item.setOpacity(0.2)

    def _load_current_data(self, recluster_only: bool = False) -> None:
        """Load or recompute data based on current selections.

        Parameters
        ----------
        recluster_only : bool, optional
            If True, only recompute clusters for existing data (e.g.,
            when the cluster count changes) without reloading images
            or coordinates.  Defaults to False.
        """
        # Reset active cluster and disable export action when loading new data
        self._active_cluster = None
        self.export_action.setEnabled(False)
        
        slide_name = self.slide_combo.currentText()
        selected_models = self._get_selected_models()
        mag = self.mag_combo.currentText()
        patch = self.patch_combo.currentText()
        
        if not all([slide_name, mag, patch]):
            print("DEBUG: Incomplete selection; cannot load data")
            return
        
        if len(selected_models) == 0:
            print("DEBUG: No models selected; cannot load data")
            QMessageBox.warning(self, "No Models Selected", 
                "Please select at least one model to visualize.")
            return
        
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: Slide info not found for {slide_name}")
            return
        
        # If not reclustering, load features and image afresh
        if not recluster_only or self.graphics_view.coords is None:
            # Show progress bar
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            
            # Load features from multiple models
            feature_list = []
            coords_lv0 = None
            patch_size_lv0 = None
            
            try:
                for model_name in selected_models:
                    h5_path = info.models.get(model_name, {}).get(mag, {}).get(patch)
                    if h5_path is None:
                        print(f"DEBUG: H5 path not found for {model_name}")
                        QMessageBox.critical(self, "Error", 
                            f"Features not found for model: {model_name}")
                        self.progress_bar.setVisible(False)
                        return
                    
                    print(f"DEBUG: Loading features from {model_name}: {h5_path}")
                    model_features, model_coords, model_patch_size = data_loader.load_features(h5_path)
                    print(f"DEBUG: Loaded {model_features.shape[0]} features of dimension {model_features.shape[1]} for {model_name}")
                    
                    # Validate coordinates match (runtime check)
                    if coords_lv0 is None:
                        coords_lv0 = model_coords
                        patch_size_lv0 = model_patch_size
                    else:
                        if not np.allclose(coords_lv0, model_coords, atol=1.0):
                            QMessageBox.critical(self, "Coordinate Mismatch",
                                f"Model {model_name} has incompatible patch coordinates. "
                                f"Please select only compatible models.")
                            self.progress_bar.setVisible(False)
                            return
                    
                    # Apply z-score normalization if preference enabled
                    if self.normalize_features:
                        print(f"DEBUG: Applying z-score normalization to {model_name} features")
                        model_features = self._normalize_features_zscore(model_features)
                    
                    feature_list.append(model_features)
                
                # Concatenate features along feature dimension (axis=1)
                if len(feature_list) == 1:
                    features = feature_list[0]
                else:
                    features = np.concatenate(feature_list, axis=1)
                    print(f"DEBUG: Concatenated features from {len(feature_list)} models")
                
                print(f"DEBUG: Final feature array shape: {features.shape[0]} patches × {features.shape[1]} dimensions")
                
                # Update status label
                model_str = ", ".join(selected_models)
                norm_str = " (normalized)" if self.normalize_features else ""
                self.status_label.setText(
                    f"Models: {model_str} | "
                    f"Features: {features.shape[1]} dims{norm_str} | "
                    f"Patches: {features.shape[0]}"
                )
                
            except Exception as e:
                print(f"DEBUG: Error loading features: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load features:\n{e}")
                self.progress_bar.setVisible(False)
                return
            # Compute PCA embedding
            pca = PCA(n_components=2)
            try:
                print("DEBUG: Computing PCA embedding of features")
                self._current_embedding = pca.fit_transform(features)
                print(f"DEBUG: PCA embedding shape: {self._current_embedding.shape}")
            except Exception as e:
                print(f"DEBUG: Error computing PCA: {e}")
                QMessageBox.critical(self, "Error", f"Failed to compute PCA:\n{e}")
                return
            # Determine slide dimensions
            slide_w = slide_h = None
            # Try using openslide if available to obtain accurate dimensions
            try:
                print(f"DEBUG: Attempting to load WSI dimensions using openslide for {info.image_path}")
                from openslide import OpenSlide  # type: ignore
                slide = OpenSlide(info.image_path)
                slide_w, slide_h = slide.dimensions
                print(f"DEBUG: OpenSlide returned dimensions {slide_w}x{slide_h}")
            except Exception:
                # Fallback: infer from coordinate grid
                print("DEBUG: Failed to use OpenSlide; inferring slide dimensions from patch coordinates")
                slide_w, slide_h = infer_slide_dims(coords_lv0, patch_size_lv0)
                print(f"DEBUG: Inferred slide dimensions {slide_w}x{slide_h}")
            # Load thumbnail
            print("DEBUG: Loading thumbnail image")
            thumb_image = data_loader.load_thumbnail(info.image_path)
            thumb_w, thumb_h = thumb_image.size
            # Scale coords to thumbnail
            coords_thumb = data_loader.scale_coords_to_thumbnail(coords_lv0, (slide_w, slide_h), (thumb_w, thumb_h))
            print(f"DEBUG: Scaled coordinates to thumbnail with shape {coords_thumb.shape}")
            # Store features and coords for reclustering
            self._current_features = features
            self._current_coords_thumb = coords_thumb
            self._current_patch_size_thumb = (patch_size_lv0 * (thumb_w / float(slide_w)))
            # Store level-0 coordinates for GeoJSON export
            self._current_coords_lv0 = coords_lv0
            self._current_patch_size_lv0 = patch_size_lv0
            self._coord_scale_factor = float(slide_w) / float(thumb_w)
            # Compute clusters
            k = self.cluster_spin.value()
            print(f"DEBUG: Performing K-means clustering with k={k}")
            labels = cluster_features(features, k)
            colours = generate_palette(int(labels.max()) + 1)
            print(f"DEBUG: Generated {len(colours)} cluster colours")
            # Update both views
            self.graphics_view.load_slide(thumb_image, coords_thumb, self._current_patch_size_thumb, labels, colours)
            # Populate scatter plot using new view
            self.scatter_view.populate(self._current_embedding, labels, colours)
            print("DEBUG: Data loading complete; views updated")
            # Hide progress bar when done
            self.progress_bar.setVisible(False)
        else:
            # Only recompute clusters using existing features
            if not hasattr(self, '_current_features'):
                return
            features = self._current_features
            k = self.cluster_spin.value()
            print(f"DEBUG: Reclustering existing features with k={k}")
            labels = cluster_features(features, k)
            colours = generate_palette(int(labels.max()) + 1)
            print(f"DEBUG: Generated {len(colours)} cluster colours for reclustering")

            # Update both views with new labels and colours
            self.graphics_view.update_labels_and_colours(labels, colours)
            self.scatter_view.populate(self._current_embedding, labels, colours)

    def _hsl_string_to_qcolor(self, hsl_string: str) -> QColor:
        """Convert HSL string from generate_palette() to QColor.

        Parameters
        ----------
        hsl_string : str
            Color in format "hsl(H,S%,L%)" where H is 0-360, S and L are 0-100

        Returns
        -------
        QColor
            Equivalent QColor object
        """
        # Parse "hsl(H,S%,L%)" format
        values = hsl_string.strip('hsl()').split(',')
        # Convert H value directly
        h = float(values[0])
        # Remove '%' and convert S,L to 0-1 range
        s = float(values[1].strip(' %')) / 100
        l = float(values[2].strip(' %')) / 100

        # Create QColor from HSL values
        color = QColor()
        color.setHslF(h/360, s, l)
        return color

    def _hsl_to_qupath_rgb(self, hsl_string: str) -> int:
        """Convert HSL color string to QuPath packed ARGB integer.
        
        QuPath uses Java's signed 32-bit integer format for colors:
        (alpha << 24) | (red << 16) | (green << 8) | blue
        With alpha=255, this produces negative values due to signed overflow.
        
        Parameters
        ----------
        hsl_string : str
            Color in format "hsl(H,S%,L%)" from generate_palette()
            
        Returns
        -------
        int
            Packed ARGB integer compatible with QuPath's colorRGB field
        """
        qcolor = self._hsl_string_to_qcolor(hsl_string)
        r, g, b = qcolor.red(), qcolor.green(), qcolor.blue()
        # Pack as ARGB with alpha=255
        packed = (255 << 24) | (r << 16) | (g << 8) | b
        # Convert to signed 32-bit integer (Java style)
        if packed >= 0x80000000:
            packed -= 0x100000000
        return int(packed)

    # --- colour utilities ---
    def _lighter_color(self, color: QColor, factor: int = 150) -> QColor:
        """Return a lighter variant of a QColor.

        Parameters
        ----------
        color : QColor
            Base colour to lighten.
        factor : int, optional
            Lightening factor (100 = no change).  Values >100 produce
            lighter colours.  Defaults to 150 (50% lighter).

        Returns
        -------
        QColor
            A new colour that is a lighter version of the input.
        """
        c = QColor(color)
        return c.lighter(factor)

    # --- helper methods ---
    def _get_selected_models(self) -> List[str]:
        """Get list of currently selected model names."""
        return self.model_selector.getSelectedModels()
    
    def _normalize_features_zscore(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features (mean=0, std=1) per dimension.
        
        This normalization ensures that each feature dimension contributes
        equally to subsequent analysis (PCA, clustering), regardless of
        the original scale or magnitude of the features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array of shape (n_patches, feature_dim)
        
        Returns
        -------
        np.ndarray
            Normalized features with same shape as input
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    
    def _validate_model_compatibility(self) -> None:
        """Validate coordinate compatibility and disable incompatible models.
        
        This method checks if all selected models have matching patch coordinates.
        If incompatibilities are found, incompatible models are disabled in the UI
        with tooltips explaining the issue.
        """
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return
        
        info = self.slides.get(slide_name)
        if info is None:
            return
        
        selected_models = self._get_selected_models()
        if len(selected_models) == 0:
            # No models selected - enable all
            for model_name in self.model_selector.getAllModelNames():
                self.model_selector.setModelEnabled(model_name, True)
                self.model_selector.setModelToolTip(model_name, "")
            return
        
        # Get current mag and patch to check compatibility
        mag = self.mag_combo.currentText()
        patch = self.patch_combo.currentText()
        if not mag or not patch:
            return
        
        # Load coordinates from first selected model
        try:
            first_model = selected_models[0]
            h5_path = info.models.get(first_model, {}).get(mag, {}).get(patch)
            if not h5_path:
                return
            
            _, ref_coords, _ = data_loader.load_features(h5_path)
            
            # Check all other models in the selector
            for model_name in self.model_selector.getAllModelNames():
                if model_name in selected_models:
                    continue  # Skip already selected models
                
                # Check if this model has features for current mag/patch
                model_h5_path = info.models.get(model_name, {}).get(mag, {}).get(patch)
                if not model_h5_path:
                    self.model_selector.setModelEnabled(model_name, False)
                    self.model_selector.setModelToolTip(
                        model_name,
                        f"Not available for {mag}/{patch}"
                    )
                    continue
                
                # Load and compare coordinates
                try:
                    _, model_coords, _ = data_loader.load_features(model_h5_path)
                    
                    # Check coordinate compatibility
                    if model_coords.shape != ref_coords.shape:
                        self.model_selector.setModelEnabled(model_name, False)
                        self.model_selector.setModelToolTip(
                            model_name,
                            f"Incompatible: Different number of patches "
                            f"({model_coords.shape[0]} vs {ref_coords.shape[0]})"
                        )
                    elif not np.allclose(model_coords, ref_coords, atol=1.0):
                        self.model_selector.setModelEnabled(model_name, False)
                        self.model_selector.setModelToolTip(
                            model_name,
                            "Incompatible: Patch coordinates do not match"
                        )
                    else:
                        # Compatible - enable it
                        self.model_selector.setModelEnabled(model_name, True)
                        self.model_selector.setModelToolTip(model_name, "")
                except Exception as e:
                    print(f"DEBUG: Error checking compatibility for {model_name}: {e}")
                    self.model_selector.setModelEnabled(model_name, False)
                    self.model_selector.setModelToolTip(
                        model_name,
                        f"Error loading features: {str(e)}"
                    )
        except Exception as e:
            print(f"DEBUG: Error in validation: {e}")
            # On error, enable all models
            for model_name in self.model_selector.getAllModelNames():
                self.model_selector.setModelEnabled(model_name, True)
                self.model_selector.setModelToolTip(model_name, "")
    
    # --- preferences and about dialogs ---
    def _show_preferences(self) -> None:
        """Display preferences dialog."""
        from preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self)
        if dialog.exec():
            # Reload preferences after dialog closes
            self.normalize_features = self.settings.value("normalize_features", True, type=bool)
    
    def _show_about(self) -> None:
        """Display About dialog with version and attribution."""
        about_text = """
        <h3>FoundationDetector</h3>
        <p><b>Version:</b> 1.0.0</p>
        <p>Interactive exploration of whole-slide image patches using 
        foundation model embeddings.</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Multi-model feature concatenation</li>
            <li>PCA dimensionality reduction</li>
            <li>K-means clustering with interactive visualization</li>
            <li>QuPath GeoJSON annotation export</li>
        </ul>
        <p><b>License:</b> MIT</p>
        <p><b>Authors:</b> FoundationDetector Contributors</p>
        """
        QMessageBox.about(self, "About FoundationDetector", about_text)
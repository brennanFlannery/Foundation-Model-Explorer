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
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
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
from sklearn.decomposition import PCA
from PySide6.QtGui import QPen, QBrush
from PySide6.QtCore import QRectF
from PIL import Image
import data_loader
from utils import generate_palette, cluster_features, infer_slide_dims, radial_sweep_order
from PySide6.QtCore import Signal

from scatter_view import ScatterGraphicsItem, ScatterGraphicsView
from slide_view import SlideGraphicsView

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
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FoundationDetector (Offline)")
        # Slide info mapping
        self.slides: Dict[str, data_loader.SlideInfo] = {}
        # Storage for scatter plot items
        self._scatter_items: List[QGraphicsEllipseItem] = []
        self._current_embedding: Optional[np.ndarray] = None
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

    def _create_widgets(self) -> None:
        """Create and lay out widgets for the main window."""
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        
        # Folder selection row
        folder_row = QHBoxLayout()
        self.select_folder_button = QPushButton("Select Folder")
        folder_row.addWidget(self.select_folder_button)
        self.root_label = QLabel("No folder selected")
        folder_row.addWidget(self.root_label)
        main_vbox.addLayout(folder_row)
        
        # Drop‑down row
        dropdown_row = QHBoxLayout()
        self.slide_combo = QComboBox()
        self.slide_combo.setPlaceholderText("Select slide")
        dropdown_row.addWidget(QLabel("Slide:"))
        dropdown_row.addWidget(self.slide_combo)
        self.model_combo = QComboBox()
        dropdown_row.addWidget(QLabel("Model:"))
        dropdown_row.addWidget(self.model_combo)
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
        # Overlay toggle
        self.overlay_checkbox = QCheckBox("Show overlays")
        self.overlay_checkbox.setChecked(True)
        main_vbox.addWidget(self.overlay_checkbox)
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

    def _connect_signals(self) -> None:
        """Connect signals and slots for interactive widgets."""
        self.select_folder_button.clicked.connect(self._select_folder)
        self.slide_combo.currentTextChanged.connect(self._on_slide_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.mag_combo.currentTextChanged.connect(self._on_mag_changed)
        self.patch_combo.currentTextChanged.connect(self._on_patch_changed)
        self.cluster_spin.valueChanged.connect(self._on_cluster_changed)

        # Overlay checkbox toggles patch rect visibility
        self.overlay_checkbox.stateChanged.connect(self._on_overlay_toggled)

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
            
            # Clear subsequent combos and view
            self.model_combo.clear()
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
        """Update model/mag/patch combos when a new slide is selected."""
        print(f"DEBUG: Slide changed to: {slide_name}")
        if not slide_name:
            print("DEBUG: No slide name provided")
            return
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: No slide info found for {slide_name}")
            return
        
        print(f"DEBUG: Found slide info: {info}")
        # Populate model combo
        self.model_combo.blockSignals(True)
        self.mag_combo.blockSignals(True)
        self.patch_combo.blockSignals(True)
        
        self.model_combo.clear()
        model_names = sorted(info.models.keys())
        print(f"DEBUG: Adding models to combo box: {model_names}")
        self.model_combo.addItems(model_names)
        
        self.mag_combo.clear()
        self.patch_combo.clear()
        
        self.model_combo.blockSignals(False)
        self.mag_combo.blockSignals(False)
        self.patch_combo.blockSignals(False)
        
        # Set initial model and manually trigger the change
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
            # Manually trigger the model changed signal
            self._on_model_changed(self.model_combo.currentText())

    def _on_model_changed(self, model_name: str) -> None:
        """Update magnification combo when model changes."""
        slide_name = self.slide_combo.currentText()
        if not slide_name or not model_name:
            return
        info = self.slides.get(slide_name)
        if info is None:
            return

        # Get available magnifications for this model
        model_dict = info.models.get(model_name, {})
        print(f"DEBUG: Available magnifications for model {model_name}: {list(model_dict.keys())}")
        
        self.mag_combo.blockSignals(True)
        self.patch_combo.blockSignals(True)
        
        self.mag_combo.clear()
        self.mag_combo.addItems(sorted(model_dict.keys()))
        self.patch_combo.clear()
        
        self.mag_combo.blockSignals(False)
        self.patch_combo.blockSignals(False)
        
        # Trigger update for new magnification
        if self.mag_combo.count() > 0:
            self.mag_combo.setCurrentIndex(0)
            # Manually trigger the magnification changed signal
            self._on_mag_changed(self.mag_combo.currentText())

    def _on_mag_changed(self, mag: str) -> None:
        """Update patch combo when magnification changes."""
        slide_name = self.slide_combo.currentText()
        model_name = self.model_combo.currentText()
        if not slide_name or not model_name or not mag:
            return
        info = self.slides.get(slide_name)
        if info is None:
            return

        # Get available patch sizes for this magnification
        patch_sizes = info.models.get(model_name, {}).get(mag, {})
        print(f"DEBUG: Available patch sizes for magnification {mag}: {list(patch_sizes.keys())}")
        
        self.patch_combo.blockSignals(True)
        self.patch_combo.clear()
        self.patch_combo.addItems(sorted(patch_sizes.keys()))
        self.patch_combo.blockSignals(False)
        
        # Trigger update for new patch size
        if self.patch_combo.count() > 0:
            self.patch_combo.setCurrentIndex(0)
            self._on_patch_changed(self.patch_combo.currentText())

    def _on_patch_changed(self, patch: str) -> None:
        """Load new data when patch size changes."""
        self._load_current_data()

    def _on_cluster_changed(self, k: int) -> None:
        """Recompute clusters when the cluster count slider changes."""
        print(f"DEBUG: Cluster spin changed to {k}; reclustering current features")
        self._load_current_data(recluster_only=True)

    def _on_overlay_toggled(self, state: int) -> None:
        """Show or hide patch overlays based on checkbox state."""
        visible = bool(state)
        print(f"DEBUG: Overlay toggled; visible={visible}")
        # Adjust visibility of patch rects
        for rect in getattr(self.graphics_view, 'rect_items', []):
            if visible:
                rect.setVisible(True)
            else:
                rect.setVisible(False)

    def _on_slide_patch_hovered(self, idx: int, state: bool) -> None:
        """Highlight the scatter point corresponding to the hovered slide patch."""
        if not self.scatter_view:
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
        self._apply_active_cluster_styles()

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
        slide_name = self.slide_combo.currentText()
        model_name = self.model_combo.currentText()
        mag = self.mag_combo.currentText()
        patch = self.patch_combo.currentText()
        if not all([slide_name, model_name, mag, patch]):
            print("DEBUG: Incomplete selection; cannot load data")
            return
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: Slide info not found for {slide_name}")
            return
        h5_path = info.models.get(model_name, {}).get(mag, {}).get(patch)
        if h5_path is None:
            print(f"DEBUG: H5 path not found for selected configuration: {slide_name}, {model_name}, {mag}, {patch}")
            return
        # If not reclustering, load features and image afresh
        if not recluster_only or self.graphics_view.coords is None:
            # Show progress bar
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            try:
                print(f"DEBUG: Loading features from {h5_path}")
                # Load features and coordinates
                features, coords_lv0, patch_size_lv0 = data_loader.load_features(h5_path)
                print(f"DEBUG: Loaded {features.shape[0]} feature vectors of dimension {features.shape[1]}")
            except Exception as e:
                print(f"DEBUG: Error loading features: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load features:\n{e}")
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
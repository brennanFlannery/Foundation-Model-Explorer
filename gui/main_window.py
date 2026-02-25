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
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from PySide6.QtCore import (
    Qt,
    QTimer,
    QPointF,
    QObject,
    QPoint,
    QPropertyAnimation,
    QEasingCurve,
    QSize,
    QThread,
    QThreadPool,
    QRunnable,
)
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
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
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGraphicsEllipseItem,
    QCheckBox,
    QProgressBar,
    QGraphicsDropShadowEffect,
    QMenu,
    QStackedLayout,
    QSizePolicy,
    QSplitter,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import QSettings, QAbstractListModel, QModelIndex
from PySide6.QtCore import Qt as QtCore
from sklearn.decomposition import PCA
from PySide6.QtGui import QPen, QBrush
from PySide6.QtCore import QRectF
from PIL import Image
from PIL.ImageQt import ImageQt
import data_loader
from utils import (
    generate_palette, cluster_features, infer_slide_dims, radial_sweep_order,
    compute_spatial_subclusters, compute_region_pca_embedding, RegionInfo,
)
from PySide6.QtCore import Signal

from views import (
    ScatterGraphicsItem, ScatterGraphicsView,
    RegionScatterView, _hsl_to_qcolor,
    SlideGraphicsView,
)
from atlas import (
    AtlasBuilder, ClusterAtlas, SlideAtlasEntry,
    AtlasScatterView,
    AtlasSlideListWidget, ThumbnailLoadTask, SlideThumbnailItem,
    SlideThumbnailListWidget, AtlasThumbnailView, AtlasThumbnailLoadTask,
    AtlasThumbnailPanel, PatchExemplarPopup,
)
import app_state
from app_state import LabeledRegionData
from chat import ChatAgentConfig, ChatAgentWorker, ChatDockWidget
from gui_types import (
    SelectionMode, SourceMode, LocalRegionCluster, LabeledRegion, ModelSelection,
)
from widgets import (
    CheckableComboBoxModel, ModelMultiSelector, ModelSelectionDialog,
    _qcolors_to_hsl_strings,
    PatchInfoPanel, PatchInfoPopup,
    ClusterLegendWidget, LocalRegionWidget, LabeledRegionsWidget,
)
from .data_mixin import DataMixin
from .cluster_mixin import ClusterMixin
from .view_mixin import ViewMixin
from .local_region_mixin import LocalRegionMixin
from .atlas_mixin import AtlasMixin
from .chat_mixin import ChatMixin
class MainWindow(
    DataMixin,
    ClusterMixin,
    ViewMixin,
    LocalRegionMixin,
    AtlasMixin,
    ChatMixin,
    QMainWindow,
):
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
    _selected_clusters : set[int]
        Currently selected clusters, used for persistent opacity styling.
    
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
    chat_submit_requested = Signal(str, object)
    chat_cancel_requested = Signal()
    chat_shutdown_requested = Signal()

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
        self._root_dir: str = ""
        self._active_exemplar_popup: Optional[PatchExemplarPopup] = None
        self._active_exemplar_popup_id: Optional[str] = None
        self._active_exemplar_source_label: str = ""
        self._chat_thread: Optional[QThread] = None
        self._chat_worker: Optional[ChatAgentWorker] = None
        # UI components
        self._create_widgets()
        self._connect_signals()
        self._setup_chat_agent()
        # Track hover state for slide-rect overlay
        self._hovered_slide_rect_idx = None
        self._hover_prev_opacity = 0.0
        self._hover_prev_brush = None
        # Track currently hovered scatter index when hovering slide patches
        self._hovered_scatter_idx: Optional[int] = None
        # Track selected clusters for persistent opacity reduction
        self._selected_clusters: set[int] = set()
        # Toast for preview-only interactions
        self._preview_toast: Optional[QLabel] = None
        # Store level-0 coordinates for GeoJSON export
        self._current_coords_lv0: Optional[np.ndarray] = None
        self._current_patch_size_lv0: Optional[float] = None
        self._coord_scale_factor: Optional[float] = None
        # Store cluster centroids for distance calculations
        self._cluster_centroids: Optional[np.ndarray] = None
        self._max_cluster_distances: Optional[np.ndarray] = None
        self._current_labels: Optional[np.ndarray] = None
        self._current_colours: Optional[List[str]] = None
        # Region embedding state
        self._current_regions: Optional[List[RegionInfo]] = None
        self._current_region_embedding: Optional[np.ndarray] = None
        self._patches_per_region: int = self.settings.value("patches_per_region", 15, type=int)
        # Saved patch opacities while a region X marker is being hovered
        self._region_hover_saved_opacities: Dict[int, float] = {}
        # Persisted model selection for session
        self._model_selection: Optional[ModelSelection] = None
        # Track cascade animation state to prevent hover interference
        self._animation_in_progress: bool = False
        # Local region selection mode state
        self._selection_mode: SelectionMode = SelectionMode.KMEANS
        self._labeled_regions: Dict[int, LabeledRegion] = {}
        self._next_region_id: int = 0
        self._last_labeled_slide: Optional[str] = None
        self._local_region_radius: float = 50.0
        self._erase_mode: bool = False
        self._update_model_selection_label()
        self._set_model_dependent_ui_enabled(False)

        # GUI action queue drain timer — MCP tools post actions here
        self._gui_action_timer = QTimer(self)
        self._gui_action_timer.setInterval(50)
        self._gui_action_timer.timeout.connect(self._drain_gui_action_queue)
        self._gui_action_timer.start()


    def _create_widgets(self) -> None:
        """Create and lay out widgets for the main window."""
        # Create menu bar
        self._create_menu_bar()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        
        # Slide thumbnail selection list
        self.slide_combo = QComboBox()
        self.slide_combo.setPlaceholderText("Select slide")
        self.slide_combo.setVisible(False)

        self.slide_thumbnail_list = SlideThumbnailListWidget()
        self.slide_thumbnail_list.slide_selected.connect(self._on_thumbnail_slide_selected)
        main_vbox.addWidget(self.slide_thumbnail_list)

        # Model selection controls (popup)
        self.model_selector = ModelMultiSelector()
        self.model_selector.setVisible(False)
        self.mag_combo = QComboBox()
        self.mag_combo.setVisible(False)
        self.patch_combo = QComboBox()
        self.patch_combo.setVisible(False)

        model_row = QHBoxLayout()
        self.model_select_btn = QPushButton("Select Model...")
        self.model_select_btn.setEnabled(False)
        self.model_selection_label = QLabel("No model selected")
        self.model_selection_label.setWordWrap(True)
        self.model_selection_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_row.addWidget(self.model_select_btn)
        model_row.addWidget(self.model_selection_label, stretch=1)
        main_vbox.addLayout(model_row)

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
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_vbox.addWidget(self.status_label)
        
        # Splitter for sidebar + slide view
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(160)
        self.sidebar_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Tab 0: Local Region Selection / Full Slide toggle
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        local_layout.setContentsMargins(0, 0, 0, 0)
        self.local_region_widget = LocalRegionWidget()
        local_layout.addWidget(self.local_region_widget)
        local_layout.addStretch()
        self.sidebar_tabs.addTab(local_tab, "Local Region")

        # Tab 1: Cluster Legend
        self.cluster_legend = ClusterLegendWidget()
        self.sidebar_tabs.addTab(self.cluster_legend, "Clusters")

        # Connect Local Region widget signals
        self.local_region_widget.radius_changed.connect(self._on_local_region_radius_changed)
        self.local_region_widget.region_clicked.connect(self._on_local_region_clicked)
        self.local_region_widget.region_deleted.connect(self._on_local_region_deleted)
        self.local_region_widget.clear_all_requested.connect(self._clear_local_region_clusters)
        self.local_region_widget.export_requested.connect(self._export_local_region_clusters)
        self.local_region_widget.full_slide_toggled.connect(self._on_full_slide_toggled)

        # Tab 3: Cross-Slide Atlas
        atlas_tab = QWidget()
        atlas_layout = QVBoxLayout(atlas_tab)
        atlas_layout.setContentsMargins(5, 5, 5, 5)
        atlas_layout.setSpacing(8)

        # Atlas slide selection list
        atlas_layout.addWidget(QLabel("Slides for Atlas:"))
        self.atlas_slide_list = AtlasSlideListWidget()
        self.atlas_slide_list.setMinimumHeight(120)
        atlas_layout.addWidget(self.atlas_slide_list)

        # Add current slide button
        self.atlas_add_current_btn = QPushButton("Add Current Slide")
        self.atlas_add_current_btn.clicked.connect(self._on_atlas_add_current)
        atlas_layout.addWidget(self.atlas_add_current_btn)

        # Cluster count control
        atlas_k_row = QHBoxLayout()
        atlas_k_row.addWidget(QLabel("Clusters:"))
        self.atlas_k_spin = QSpinBox()
        self.atlas_k_spin.setRange(2, 50)
        self.atlas_k_spin.setValue(10)
        atlas_k_row.addWidget(self.atlas_k_spin)
        atlas_k_row.addStretch()
        atlas_layout.addLayout(atlas_k_row)

        # Build atlas button
        self.build_atlas_btn = QPushButton("Build Atlas")
        self.build_atlas_btn.clicked.connect(self._build_atlas)
        self.build_atlas_btn.setEnabled(False)
        atlas_layout.addWidget(self.build_atlas_btn)

        # Clear atlas button
        self.clear_atlas_btn = QPushButton("Clear Atlas")
        self.clear_atlas_btn.clicked.connect(self._clear_atlas)
        self.clear_atlas_btn.setEnabled(False)
        atlas_layout.addWidget(self.clear_atlas_btn)

        # Atlas progress bar
        self.atlas_progress = QProgressBar()
        self.atlas_progress.setVisible(False)
        atlas_layout.addWidget(self.atlas_progress)

        # Atlas info label
        self.atlas_info_label = QLabel("Add at least 2 slides to build atlas")
        self.atlas_info_label.setWordWrap(True)
        self.atlas_info_label.setStyleSheet("color: gray; font-size: 10px;")
        atlas_layout.addWidget(self.atlas_info_label)

        atlas_layout.addStretch()
        self.sidebar_tabs.addTab(atlas_tab, "Atlas")

        # Connect tab change to mode switch
        self.sidebar_tabs.currentChanged.connect(self._on_sidebar_tab_changed)

        # Main slide view (takes remaining width)
        self.graphics_view = SlideGraphicsView()
        self.graphics_view.setMinimumWidth(400)
        # Connect signals: slide click selects cluster and updates scatter
        self.graphics_view.cluster_selected.connect(
            lambda cluster, pos, ctrl: self._update_scatter_for_cluster(cluster, pos, ctrl)
        )
        # Connect slide hover to scatter
        self.graphics_view.patch_hovered.connect(self._on_slide_patch_hovered)
        self.graphics_view.preview_action_attempted.connect(self._show_preview_blocked_toast)
        # Connect slide animation signals for synchronized scatter cascade
        self.graphics_view.patches_highlighted.connect(self._on_patches_highlighted)
        self.graphics_view.animation_completed.connect(self._on_animation_completed)
        # Connect local region selection signal
        self.graphics_view.local_region_selected.connect(self._on_local_region_selected)

        # Atlas thumbnail panel (hidden until atlas is built)
        self.atlas_thumbnail_panel = AtlasThumbnailPanel()
        self.atlas_thumbnail_panel.setVisible(False)
        self.atlas_thumbnail_panel.slide_clicked.connect(self._on_atlas_thumbnail_clicked)

        # Sidebar container: tab widget + always-visible labeled regions panel
        sidebar_container = QWidget()
        sidebar_vbox = QVBoxLayout(sidebar_container)
        sidebar_vbox.setContentsMargins(0, 0, 0, 0)
        sidebar_vbox.setSpacing(0)
        sidebar_vbox.addWidget(self.sidebar_tabs, stretch=1)
        sidebar_sep = QFrame()
        sidebar_sep.setFrameShape(QFrame.Shape.HLine)
        sidebar_vbox.addWidget(sidebar_sep)

        # Label / Erase mode toggle (always visible above labeled regions panel)
        _erase_toggle_row = QWidget()
        _erase_toggle_layout = QHBoxLayout(_erase_toggle_row)
        _erase_toggle_layout.setContentsMargins(6, 4, 6, 4)
        _erase_toggle_layout.setSpacing(0)

        self.label_mode_btn = QPushButton("Label")
        self.label_mode_btn.setCheckable(True)
        self.label_mode_btn.setChecked(True)
        self.label_mode_btn.setFixedHeight(24)
        self.label_mode_btn.setEnabled(False)
        self.label_mode_btn.setStyleSheet(
            "QPushButton { font-size: 9pt; border-top-left-radius: 3px;"
            " border-bottom-left-radius: 3px; border-top-right-radius: 0px;"
            " border-bottom-right-radius: 0px;"
            " border: 1px solid #888; padding: 0 8px; }"
            "QPushButton:checked { background: #3a7bd5; color: white; border-color: #3a7bd5; }"
        )

        self.erase_mode_btn = QPushButton("Erase")
        self.erase_mode_btn.setCheckable(True)
        self.erase_mode_btn.setChecked(False)
        self.erase_mode_btn.setFixedHeight(24)
        self.erase_mode_btn.setEnabled(False)
        self.erase_mode_btn.setStyleSheet(
            "QPushButton { font-size: 9pt; border-top-right-radius: 3px;"
            " border-bottom-right-radius: 3px; border-top-left-radius: 0px;"
            " border-bottom-left-radius: 0px;"
            " border: 1px solid #888; border-left: none; padding: 0 8px; }"
            "QPushButton:checked { background: #e08024; color: white; border-color: #e08024; }"
        )

        _erase_toggle_layout.addWidget(self.label_mode_btn)
        _erase_toggle_layout.addWidget(self.erase_mode_btn)
        _erase_toggle_layout.addStretch()
        sidebar_vbox.addWidget(_erase_toggle_row)

        self.labeled_regions_widget = LabeledRegionsWidget()
        self.labeled_regions_widget.region_deleted.connect(self._on_labeled_region_deleted_from_panel)
        self.labeled_regions_widget.clear_all_requested.connect(self._clear_all_labeled_regions)
        self.labeled_regions_widget.export_requested.connect(self._export_labeled_regions)
        sidebar_vbox.addWidget(self.labeled_regions_widget, stretch=0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_container)
        splitter.addWidget(self.atlas_thumbnail_panel)
        splitter.addWidget(self.graphics_view)
        splitter.setStretchFactor(0, 0)  # sidebar: no stretch
        splitter.setStretchFactor(1, 0)  # thumbnail panel: no stretch
        splitter.setStretchFactor(2, 1)  # main view: stretch
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)

        main_vbox.addWidget(splitter, stretch=1)
        
        # Scatter plot view in floating dock widget
        self.scatter_view = ScatterGraphicsView()
        self.scatter_view.setMinimumSize(250, 250)
        # Connect scatter click to highlight slide
        self.scatter_view.cluster_selected.connect(self._on_scatter_cluster_selected)
        # Connect scatter hover to highlight slide and scatter
        self.scatter_view.point_hovered.connect(self._on_scatter_point_hovered)
        # Connect local region selection signal
        self.scatter_view.local_region_selected.connect(self._on_scatter_local_region_selected)

        # Create dock widget for scatter view
        self.scatter_dock = QDockWidget("Embedding View", self)
        self.scatter_dock.setWidget(self.scatter_view)
        self.scatter_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.scatter_dock.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        # Add to main window and set floating
        self.addDockWidget(Qt.RightDockWidgetArea, self.scatter_dock)
        self.scatter_dock.setFloating(True)
        self.scatter_dock.resize(350, 350)
        
        # Add scatter dock toggle to View menu (after dock is created)
        self._view_menu.addSeparator()
        self.scatter_dock_action = self.scatter_dock.toggleViewAction()
        self.scatter_dock_action.setText("Show Embedding View")
        self.scatter_dock_action.setShortcut("Ctrl+3")
        self._view_menu.addAction(self.scatter_dock_action)
        
        # Track if scatter dock has been positioned
        self._scatter_positioned = False
        
        # Connect dock visibility changed signal for snapping
        self.scatter_dock.visibilityChanged.connect(self._on_scatter_dock_visibility_changed)

        # Atlas scatter view in separate dock widget
        self.atlas_scatter_view = AtlasScatterView()
        self.atlas_scatter_view.setMinimumSize(300, 300)
        # Connect atlas scatter signals
        self.atlas_scatter_view.cluster_selected.connect(self._on_atlas_cluster_selected)
        self.atlas_scatter_view.point_hovered.connect(self._on_atlas_point_hovered)

        # Create dock widget for atlas scatter view
        self.atlas_scatter_dock = QDockWidget("Atlas Embedding View", self)
        self.atlas_scatter_dock.setWidget(self.atlas_scatter_view)
        self.atlas_scatter_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.atlas_scatter_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        # Add to main window but start hidden
        self.addDockWidget(Qt.RightDockWidgetArea, self.atlas_scatter_dock)
        self.atlas_scatter_dock.setFloating(True)
        self.atlas_scatter_dock.resize(400, 400)
        self.atlas_scatter_dock.hide()  # Hidden until atlas is built

        # Add atlas dock toggle to View menu
        self.atlas_scatter_dock_action = self.atlas_scatter_dock.toggleViewAction()
        self.atlas_scatter_dock_action.setText("Show Atlas Embedding View")
        self.atlas_scatter_dock_action.setShortcut("Ctrl+4")
        self._view_menu.addAction(self.atlas_scatter_dock_action)

        # Region Embedding View dock
        self.region_scatter_view = RegionScatterView()
        self.region_scatter_view.setMinimumSize(250, 250)
        self.region_scatter_view.region_clicked.connect(self._on_region_scatter_clicked)
        self.region_scatter_view.region_hovered.connect(self._on_region_scatter_hovered)

        self.region_scatter_dock = QDockWidget("Region Embedding View", self)
        self.region_scatter_dock.setWidget(self.region_scatter_view)
        self.region_scatter_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.region_scatter_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.region_scatter_dock)
        self.region_scatter_dock.setFloating(True)
        self.region_scatter_dock.resize(350, 350)
        self.region_scatter_dock.hide()

        self.region_scatter_dock_action = self.region_scatter_dock.toggleViewAction()
        self.region_scatter_dock_action.setText("Show Region Embedding View")
        self.region_scatter_dock_action.setShortcut("Ctrl+6")
        self._view_menu.addAction(self.region_scatter_dock_action)

        # Agent chat view in separate dock widget
        self.chat_dock = ChatDockWidget(self)
        self.chat_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.chat_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)
        self.chat_dock.setFloating(False)
        self.chat_dock.resize(420, 420)
        self.chat_dock.show()

        self.chat_dock_action = self.chat_dock.toggleViewAction()
        self.chat_dock_action.setText("Show Agent Chat")
        self.chat_dock_action.setShortcut("Ctrl+5")
        self._view_menu.addAction(self.chat_dock_action)

        self.chat_dock.send_requested.connect(self._on_chat_send_requested)
        self.chat_dock.cancel_requested.connect(self._on_chat_cancel_requested)

        # Initialize atlas-related state
        self._cluster_atlas: Optional[ClusterAtlas] = None
        self._atlas_builder: Optional[AtlasBuilder] = None

        # Connect atlas slide list signals
        self.atlas_slide_list.slide_removed.connect(self._on_atlas_slide_removed)
        self.atlas_slide_list.selection_changed.connect(self._update_atlas_ui_state)

        # Hover popup for scatter patch info
        self.patch_info_popup = PatchInfoPopup()

        # Hover popup for slide view patch info (bottom-left corner)
        self.slide_info_popup = PatchInfoPopup()

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

        # Export all labeled regions (unified)
        self.export_labeled_action = QAction("Export Labeled Regions as GeoJSON...", self)
        self.export_labeled_action.setShortcut("Ctrl+Shift+E")
        self.export_labeled_action.setEnabled(False)
        self.export_labeled_action.triggered.connect(self._export_labeled_regions)
        file_menu.addAction(self.export_labeled_action)

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
        
        view_menu.addSeparator()
        
        # Adaptive Zoom toggle
        self.adaptive_zoom_action = QAction("Adaptive Zoom (Multi-Resolution)", self)
        self.adaptive_zoom_action.setShortcut("Ctrl+2")
        self.adaptive_zoom_action.setCheckable(True)
        self.adaptive_zoom_action.setChecked(True)  # Default on
        self.adaptive_zoom_action.setToolTip(
            "Enable multi-resolution tile loading for high-quality zoom. "
            "Disable for faster loading with fixed-resolution thumbnail."
        )
        self.adaptive_zoom_action.triggered.connect(self._on_adaptive_zoom_toggled)
        view_menu.addAction(self.adaptive_zoom_action)
        
        # Store view_menu reference for adding scatter dock toggle later
        self._view_menu = view_menu
        
        # Help Menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About
        about_action = QAction("About FoundationDetector", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)


    def _connect_signals(self) -> None:
        """Connect signals and slots for interactive widgets."""
        self.slide_combo.currentTextChanged.connect(self._on_slide_changed)
        self.model_select_btn.clicked.connect(self._open_model_selection_dialog)
        self.model_selector.selectionChanged.connect(self._on_model_selection_changed)
        self.mag_combo.currentTextChanged.connect(self._on_mag_changed)
        self.patch_combo.currentTextChanged.connect(self._on_patch_changed)
        self.cluster_spin.valueChanged.connect(self._on_cluster_changed)
        self.cluster_legend.cluster_rename.connect(self._on_cluster_rename_requested)
        self.label_mode_btn.clicked.connect(self._on_label_mode_clicked)
        self.erase_mode_btn.clicked.connect(self._on_erase_mode_clicked)

    # ---- Handlers ----


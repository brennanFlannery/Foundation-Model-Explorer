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
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
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
from utils import generate_palette, cluster_features, infer_slide_dims, radial_sweep_order
from PySide6.QtCore import Signal

from scatter_view import ScatterGraphicsItem, ScatterGraphicsView
from slide_view import SlideGraphicsView
from atlas_builder import AtlasBuilder, ClusterAtlas, SlideAtlasEntry
from atlas_scatter_view import AtlasScatterView


class SelectionMode(Enum):
    """Selection mode for the application."""
    KMEANS = "kmeans"        # Default - click selects entire K-means cluster
    LOCAL_REGION = "local"   # Click selects K-means cluster patches within radius


@dataclass
class LocalRegionCluster:
    """Represents a user-defined local region cluster (subset of K-means cluster)."""
    cluster_id: int
    patch_indices: Set[int]  # indices into the patch arrays
    center_point: Tuple[float, float]  # click center in scene coords
    radius: float  # radius at time of creation
    color: QColor
    name: str
    kmeans_cluster: int  # the K-means cluster this region belongs to


@dataclass
class ModelSelection:
    """Represents a selected model configuration for feature loading."""
    models: List[str]
    magnification: str
    patch_size: str


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

    def setCheckedItems(self, items: List[str]) -> None:
        """Set the checked state for multiple items."""
        target = set(items)
        for item in self._items:
            self._checked[item] = item in target and self._enabled[item]
        if self._items:
            top = self.index(0)
            bottom = self.index(len(self._items) - 1)
            self.dataChanged.emit(top, bottom, [QtCore.CheckStateRole])


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

    def setSelectedModels(self, models: List[str]) -> None:
        """Programmatically select a list of models."""
        self._model.setCheckedItems(models)
        self._update_text()


class PatchInfoPanel(QWidget):
    """Widget displaying information about hovered/selected patch.
    
    This panel shows details about the currently hovered patch, including
    its index, coordinates, cluster assignment, and distance to the
    cluster centroid in feature space.
    
    Attributes
    ----------
    index_label : QLabel
        Displays the patch index.
    coords_label : QLabel
        Displays the patch coordinates.
    cluster_label : QLabel
        Displays the cluster assignment.
    distance_label : QLabel
        Displays distance to cluster centroid.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("Patch Info")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(header)
        
        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)
        
        # Info labels
        self.index_label = QLabel("Patch: -")
        self.index_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.index_label)
        
        self.coords_label = QLabel("Position: -")
        self.coords_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.coords_label)
        
        self.cluster_label = QLabel("Cluster: -")
        self.cluster_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.cluster_label)
        
        self.distance_label = QLabel("Centroid dist: -")
        self.distance_label.setStyleSheet("font-size: 9pt; color: #666;")
        layout.addWidget(self.distance_label)
        
        layout.addStretch()
        
        # Set size constraints
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)
    
    def update_patch_info(self, index: int, coords: Optional[Tuple[float, float]],
                          cluster: Optional[int], distance: Optional[float]) -> None:
        """Update displayed patch information.
        
        Parameters
        ----------
        index : int
            Patch index, or -1 to clear.
        coords : Optional[Tuple[float, float]]
            Patch coordinates (x, y) or None.
        cluster : Optional[int]
            Cluster assignment or None.
        distance : Optional[float]
            Distance to cluster centroid or None.
        """
        if index < 0:
            self.index_label.setText("Patch: -")
            self.coords_label.setText("Position: -")
            self.cluster_label.setText("Cluster: -")
            self.distance_label.setText("Centroid dist: -")
        else:
            self.index_label.setText(f"Patch: {index}")
            if coords is not None:
                self.coords_label.setText(f"Position: ({coords[0]:.0f}, {coords[1]:.0f})")
            else:
                self.coords_label.setText("Position: -")
            if cluster is not None:
                self.cluster_label.setText(f"Cluster: {cluster}")
            else:
                self.cluster_label.setText("Cluster: -")
            if distance is not None:
                self.distance_label.setText(f"Centroid dist: {distance:.1f}%")
            else:
                self.distance_label.setText("Centroid dist: -")
    
    def clear(self) -> None:
        """Clear all displayed information."""
        self.update_patch_info(-1, None, None, None)


class PatchInfoPopup(QWidget):
    """Translucent popup widget displaying patch information on hover.

    This popup appears near the cursor when hovering over scatter points,
    showing patch index, coordinates, cluster assignment, and centroid distance.

    The widget uses:
    - Qt.WindowFlags for frameless, translucent window
    - QPropertyAnimation for smooth fade in/out
    - Dynamic positioning to avoid viewport edge clipping
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Window flags for translucent, frameless, always-on-top popup
        self.setWindowFlags(
            Qt.ToolTip |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Setup UI
        self._setup_ui()

        # Animation for fade in/out
        self._opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_animation.setDuration(150)
        self._opacity_animation.setEasingCurve(QEasingCurve.OutCubic)

        # Hover delay timer (prevents flicker on fast mouse movement)
        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.timeout.connect(self._do_show)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._do_hide)

        # Pending position for delayed show
        self._pending_pos: Optional[QPoint] = None

        # Initially hidden
        self.hide()

    def _setup_ui(self) -> None:
        """Create the popup layout and labels."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container frame for styling
        self._frame = QFrame(self)
        self._frame.setObjectName("popupFrame")
        self._frame.setStyleSheet("""
            #popupFrame {
                background-color: rgba(40, 40, 45, 230);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(12, 10, 12, 10)
        frame_layout.setSpacing(4)

        # Header
        self._header = QLabel("Patch Info")
        self._header.setStyleSheet("""
            font-weight: bold;
            font-size: 10pt;
            color: rgba(255, 255, 255, 230);
        """)
        frame_layout.addWidget(self._header)

        # Info labels
        label_style = "font-size: 9pt; color: rgba(255, 255, 255, 200);"

        self.index_label = QLabel("Patch: -")
        self.index_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.index_label)

        self.coords_label = QLabel("Position: -")
        self.coords_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.coords_label)

        self.cluster_label = QLabel("Cluster: -")
        self.cluster_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.cluster_label)

        self.distance_label = QLabel("Centroid dist: -")
        self.distance_label.setStyleSheet(
            "font-size: 9pt; color: rgba(200, 200, 200, 180);"
        )
        frame_layout.addWidget(self.distance_label)

        # Add frame to main layout
        layout.addWidget(self._frame)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(2, 2)
        self._frame.setGraphicsEffect(shadow)

        # Fixed width for consistent appearance
        self.setFixedWidth(180)

    def update_info(self, index: int, coords: Optional[Tuple[float, float]],
                    cluster: Optional[int], distance: Optional[float],
                    cluster_color: Optional[QColor] = None) -> None:
        """Update displayed patch information.

        Parameters
        ----------
        index : int
            Patch index.
        coords : Optional[Tuple[float, float]]
            Patch coordinates (x, y).
        cluster : Optional[int]
            Cluster assignment.
        distance : Optional[float]
            Distance to cluster centroid (percentage).
        cluster_color : Optional[QColor]
            Cluster color for visual indicator.
        """
        self.index_label.setText(f"Patch: {index}")

        if coords is not None:
            self.coords_label.setText(f"Position: ({coords[0]:.0f}, {coords[1]:.0f})")
        else:
            self.coords_label.setText("Position: -")

        if cluster is not None:
            if cluster_color is not None:
                rgb = f"rgb({cluster_color.red()}, {cluster_color.green()}, {cluster_color.blue()})"
                self.cluster_label.setText(f"Cluster: {cluster}")
                self.cluster_label.setStyleSheet(
                    f"font-size: 9pt; color: {rgb}; font-weight: bold;"
                )
            else:
                self.cluster_label.setText(f"Cluster: {cluster}")
                self.cluster_label.setStyleSheet("font-size: 9pt; color: rgba(255, 255, 255, 200);")
        else:
            self.cluster_label.setText("Cluster: -")
            self.cluster_label.setStyleSheet("font-size: 9pt; color: rgba(255, 255, 255, 200);")

        if distance is not None:
            self.distance_label.setText(f"Centroid dist: {distance:.1f}%")
        else:
            self.distance_label.setText("Centroid dist: -")

    def show_at_cursor(self, global_pos: QPoint, delay_ms: int = 100) -> None:
        """Show the popup near the given global position with optional delay.

        Parameters
        ----------
        global_pos : QPoint
            Global screen position (typically from QCursor.pos()).
        delay_ms : int
            Delay before showing (prevents flicker on fast movement).
        """
        # Cancel any pending hide
        self._hide_timer.stop()

        # Store position for delayed show
        self._pending_pos = global_pos

        if delay_ms > 0:
            self._show_timer.start(delay_ms)
        else:
            self._do_show()

    def show_at_position(self, global_pos: QPoint, delay_ms: int = 100) -> None:
        """Show the popup at a specific position without edge avoidance.

        Parameters
        ----------
        global_pos : QPoint
            Global screen position to place the popup.
        delay_ms : int
            Delay before showing (prevents flicker on fast movement).
        """
        # Cancel any pending hide
        self._hide_timer.stop()

        # Store position for delayed show (use directly, no edge calc)
        self._pending_pos = global_pos
        self._use_direct_position = True

        if delay_ms > 0:
            self._show_timer.start(delay_ms)
        else:
            self._do_show()

    def _do_show(self) -> None:
        """Internal method to actually show the popup."""
        if self._pending_pos is None:
            return

        # Calculate position with edge avoidance, or use direct position
        if getattr(self, '_use_direct_position', False):
            pos = self._pending_pos
            self._use_direct_position = False
        else:
            pos = self._calculate_position(self._pending_pos)
        self.move(pos)

        # Fade in
        self._opacity_animation.stop()
        self._opacity_animation.setStartValue(self.windowOpacity())
        self._opacity_animation.setEndValue(1.0)
        self._opacity_animation.start()

        self.show()
        self.raise_()

    def hide_popup(self, delay_ms: int = 50) -> None:
        """Hide the popup with optional delay and fade out.

        Parameters
        ----------
        delay_ms : int
            Delay before hiding (allows re-hover to cancel hide).
        """
        # Cancel any pending show
        self._show_timer.stop()

        if delay_ms > 0:
            self._hide_timer.start(delay_ms)
        else:
            self._do_hide()

    def _do_hide(self) -> None:
        """Internal method to actually hide the popup."""
        # Fade out then hide
        self._opacity_animation.stop()
        self._opacity_animation.setStartValue(self.windowOpacity())
        self._opacity_animation.setEndValue(0.0)

        # Disconnect any previous connection to avoid duplicates
        try:
            self._opacity_animation.finished.disconnect(self._on_fade_out_finished)
        except RuntimeError:
            pass

        self._opacity_animation.finished.connect(self._on_fade_out_finished)
        self._opacity_animation.start()

    def _on_fade_out_finished(self) -> None:
        """Called when fade-out animation completes."""
        try:
            self._opacity_animation.finished.disconnect(self._on_fade_out_finished)
        except RuntimeError:
            pass
        if self.windowOpacity() == 0.0:
            self.hide()

    def _calculate_position(self, cursor_pos: QPoint) -> QPoint:
        """Calculate popup position avoiding screen edges.

        Parameters
        ----------
        cursor_pos : QPoint
            Global cursor position.

        Returns
        -------
        QPoint
            Adjusted position for the popup.
        """
        # Offset from cursor
        offset_x = 15
        offset_y = 15

        # Get screen geometry
        screen = QApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()

        # Calculate popup dimensions
        popup_width = self.sizeHint().width()
        popup_height = self.sizeHint().height()

        # Default position: right and below cursor
        x = cursor_pos.x() + offset_x
        y = cursor_pos.y() + offset_y

        # Check right edge
        if x + popup_width > screen_rect.right():
            x = cursor_pos.x() - popup_width - offset_x

        # Check bottom edge
        if y + popup_height > screen_rect.bottom():
            y = cursor_pos.y() - popup_height - offset_y

        # Ensure not off left/top edges
        x = max(x, screen_rect.left())
        y = max(y, screen_rect.top())

        return QPoint(x, y)


class ClusterLegendWidget(QWidget):
    """Widget displaying cluster color legend with patch counts.

    This widget shows a vertical list of cluster entries, each containing
    a colored square, cluster ID, and patch count. The legend automatically
    updates when clustering changes.

    Attributes
    ----------
    _cluster_rows : List[QWidget]
        List of row widgets for each cluster entry.
    """

    # Signals for cluster interactions
    cluster_clicked = Signal(int, bool)       # Emits cluster ID and ctrl state on left-click
    cluster_toggled = Signal(int, bool)       # Emits cluster ID and checked state
    cluster_rename = Signal(int)              # Emits cluster ID for rename request
    cluster_export = Signal(int)              # Emits cluster ID for single export
    export_all_requested = Signal()           # Emits when "Export All" selected

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)

        # Header
        header = QLabel("Clusters")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self._layout.addWidget(header)

        # Scrollable container for cluster rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        self._layout.addWidget(self._scroll)

        self._cluster_rows: List[QWidget] = []
        self._cluster_names: Dict[int, str] = {}  # Custom names for clusters
        self._checkboxes: Dict[int, QCheckBox] = {}

        # Set minimum width
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)
    
    def update_clusters(self, labels: np.ndarray, colors: List[str]) -> None:
        """Update legend with current cluster data.
        
        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for all patches.
        colors : List[str]
            HSL color strings for each cluster.
        """
        # Clear existing rows
        for row in self._cluster_rows:
            row.deleteLater()
        self._cluster_rows.clear()
        self._checkboxes.clear()
        
        if labels is None or len(labels) == 0:
            return
        
        # Count patches per cluster
        unique_labels = np.unique(labels)
        counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
        
        # Create row for each cluster
        for lbl in sorted(unique_labels):
            lbl = int(lbl)
            count = counts[lbl]
            color_str = colors[lbl] if lbl < len(colors) else "hsl(0, 0%, 50%)"
            
            row = self._create_cluster_row(lbl, count, color_str)
            # Insert before the stretch
            self._content_layout.insertWidget(
                self._content_layout.count() - 1, row
            )
            self._cluster_rows.append(row)
    
    def _create_cluster_row(self, cluster_id: int, count: int,
                            color_str: str) -> QWidget:
        """Create a single cluster row widget.

        Parameters
        ----------
        cluster_id : int
            The cluster ID/number.
        count : int
            Number of patches in this cluster.
        color_str : str
            HSL color string for the cluster.

        Returns
        -------
        QWidget
            The row widget containing color square, label, and count.
        """
        row = QWidget()
        row.setProperty("cluster_id", cluster_id)  # Store cluster ID for event handling
        row.setCursor(Qt.PointingHandCursor)  # Indicate clickable
        row.installEventFilter(self)  # Handle mouse events

        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        # Color square
        color_square = QFrame()
        color_square.setFixedSize(14, 14)
        # Convert HSL to hex for CSS
        qcolor = self._hsl_to_qcolor(color_str)
        hex_color = qcolor.name()
        color_square.setStyleSheet(
            f"background-color: {hex_color}; "
            f"border: 1px solid #666; border-radius: 2px;"
        )
        layout.addWidget(color_square)

        # Checkbox for multi-select
        checkbox = QCheckBox()
        checkbox.setToolTip("Select cluster")
        checkbox.toggled.connect(lambda checked, cid=cluster_id: self.cluster_toggled.emit(cid, checked))
        layout.addWidget(checkbox)
        self._checkboxes[cluster_id] = checkbox

        # Cluster label - use custom name if available
        display_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        label = QLabel(display_name)
        label.setStyleSheet("font-size: 9pt;")
        label.setObjectName("cluster_label")  # For later reference when renaming
        layout.addWidget(label)

        # Spacer
        layout.addStretch()

        # Count
        count_label = QLabel(f"({count:,})")
        count_label.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(count_label)

        return row

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle mouse events on cluster rows."""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        if event.type() == QEvent.MouseButtonPress:
            cluster_id = obj.property("cluster_id")
            if cluster_id is not None:
                if event.button() == Qt.LeftButton:
                    ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
                    self.cluster_clicked.emit(cluster_id, ctrl_pressed)
                    return True
                elif event.button() == Qt.RightButton:
                    self._show_context_menu(cluster_id, event.globalPos())
                    return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, cluster_id: int, pos) -> None:
        """Show context menu for a cluster row."""
        menu = QMenu(self)

        # Rename action
        rename_action = menu.addAction("Rename Cluster...")
        rename_action.triggered.connect(lambda: self.cluster_rename.emit(cluster_id))

        # Export single cluster
        export_action = menu.addAction("Export as GeoJSON...")
        export_action.triggered.connect(lambda: self.cluster_export.emit(cluster_id))

        menu.addSeparator()

        # Export all clusters
        export_all_action = menu.addAction("Export All Clusters...")
        export_all_action.triggered.connect(self.export_all_requested.emit)

        menu.exec_(pos)

    def set_cluster_checked(self, cluster_id: int, checked: bool,
                            block_signals: bool = True) -> None:
        """Set checkbox state for a cluster row."""
        checkbox = self._checkboxes.get(cluster_id)
        if checkbox is None:
            return
        if block_signals:
            checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        if block_signals:
            checkbox.blockSignals(False)

    def set_cluster_name(self, cluster_id: int, name: str) -> None:
        """Set a custom name for a cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID to rename.
        name : str
            The new display name.
        """
        self._cluster_names[cluster_id] = name

    def get_cluster_name(self, cluster_id: int) -> str:
        """Get the display name for a cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID.

        Returns
        -------
        str
            Custom name if set, otherwise default "Cluster N" format.
        """
        return self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    def clear_cluster_names(self) -> None:
        """Clear all custom cluster names."""
        self._cluster_names.clear()
    
    @staticmethod
    def _hsl_to_qcolor(hsl_string: str) -> QColor:
        """Convert HSL string to QColor.
        
        Parameters
        ----------
        hsl_string : str
            Color in format 'hsl(H, S%, L%)'.
            
        Returns
        -------
        QColor
            Converted Qt color object.
        """
        try:
            values = hsl_string.strip().lower().replace('hsl(', '').rstrip(')').split(',')
            h = float(values[0])
            s = float(values[1].strip(' %')) / 100.0
            l = float(values[2].strip(' %')) / 100.0
            c = QColor()
            c.setHslF(h / 360.0, s, l)
            return c
        except Exception:
            return QColor('black')


class LocalRegionWidget(QWidget):
    """Widget for local region selection controls and cluster list.

    This widget provides controls for the local region selection mode,
    including a radius slider and a list of user-defined regions.
    """

    # Signals
    radius_changed = Signal(float)       # Emits new radius value
    region_clicked = Signal(int)         # Emits region ID for highlighting
    region_deleted = Signal(int)         # Emits region ID for deletion
    clear_all_requested = Signal()       # Request to clear all regions
    export_requested = Signal()          # Request to export regions

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._region_rows: Dict[int, QWidget] = {}

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QLabel("Local Region Selection")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Click to select patches within radius\n"
            "(limited to same K-means cluster)"
        )
        instructions.setStyleSheet("color: gray; font-size: 9pt;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Radius slider with label
        radius_header = QLabel("Selection Radius:")
        radius_header.setStyleSheet("font-size: 9pt; margin-top: 4px;")
        layout.addWidget(radius_header)

        radius_row = QHBoxLayout()
        radius_row.setSpacing(8)

        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(10, 500)  # Will be recalculated based on patch size
        self.radius_slider.setValue(50)
        self.radius_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radius_slider.setTickInterval(50)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        radius_row.addWidget(self.radius_slider, stretch=1)

        self.radius_label = QLabel("50")
        self.radius_label.setMinimumWidth(40)
        self.radius_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.radius_label.setStyleSheet("font-size: 9pt;")
        radius_row.addWidget(self.radius_label)

        layout.addLayout(radius_row)

        # Separator before regions list
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        # Regions header
        regions_header = QLabel("User-Defined Regions:")
        regions_header.setStyleSheet("font-size: 9pt; margin-top: 4px;")
        layout.addWidget(regions_header)

        # Scrollable region list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, stretch=1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("font-size: 9pt;")
        self.clear_btn.clicked.connect(self.clear_all_requested.emit)
        self.clear_btn.setEnabled(False)  # Disabled when no regions
        btn_row.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("font-size: 9pt;")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.export_btn.setEnabled(False)  # Disabled when no regions
        btn_row.addWidget(self.export_btn)

        layout.addLayout(btn_row)

        # Set size constraints
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)

    def _on_radius_changed(self, value: int) -> None:
        """Handle radius slider change."""
        self.radius_label.setText(str(value))
        self.radius_changed.emit(float(value))

    def set_radius_range(self, min_val: int, max_val: int, current: int) -> None:
        """Set radius slider range based on patch size.

        Parameters
        ----------
        min_val : int
            Minimum radius value.
        max_val : int
            Maximum radius value.
        current : int
            Current/default radius value.
        """
        self.radius_slider.blockSignals(True)
        self.radius_slider.setRange(min_val, max_val)
        self.radius_slider.setValue(current)
        self.radius_slider.setTickInterval(max(1, (max_val - min_val) // 10))
        self.radius_label.setText(str(current))
        self.radius_slider.blockSignals(False)

    def get_radius(self) -> float:
        """Get the current radius value."""
        return float(self.radius_slider.value())

    def add_region(self, region_id: int, patch_count: int,
                   color: QColor, name: str) -> None:
        """Add a new region row to the list.

        Parameters
        ----------
        region_id : int
            Unique identifier for the region.
        patch_count : int
            Number of patches in the region.
        color : QColor
            Color for the region display.
        name : str
            Display name for the region.
        """
        row = QWidget()
        row.setProperty("region_id", region_id)
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.installEventFilter(self)

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)
        row_layout.setSpacing(6)

        # Color square
        color_square = QFrame()
        color_square.setFixedSize(14, 14)
        hex_color = color.name()
        color_square.setStyleSheet(
            f"background-color: {hex_color}; "
            f"border: 1px solid #666; border-radius: 2px;"
        )
        row_layout.addWidget(color_square)

        # Region name label
        label = QLabel(name)
        label.setStyleSheet("font-size: 9pt;")
        label.setObjectName("region_label")
        row_layout.addWidget(label)

        # Spacer
        row_layout.addStretch()

        # Count
        count_label = QLabel(f"({patch_count:,})")
        count_label.setStyleSheet("color: #888; font-size: 9pt;")
        count_label.setObjectName("region_count")
        row_layout.addWidget(count_label)

        # Delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(18, 18)
        delete_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; color: #888; border: none; padding: 0; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        delete_btn.setToolTip("Delete region")
        delete_btn.clicked.connect(lambda: self.region_deleted.emit(region_id))
        row_layout.addWidget(delete_btn)

        # Insert before the stretch
        self._content_layout.insertWidget(
            self._content_layout.count() - 1, row
        )
        self._region_rows[region_id] = row

        # Enable buttons
        self.clear_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def update_region(self, region_id: int, patch_count: int, name: Optional[str] = None) -> None:
        """Update an existing region row."""
        row = self._region_rows.get(region_id)
        if row is None:
            return

        label = row.findChild(QLabel, "region_label")
        if label is not None and name is not None:
            label.setText(name)

        count_label = row.findChild(QLabel, "region_count")
        if count_label is not None:
            count_label.setText(f"({patch_count:,})")

    def remove_region(self, region_id: int) -> None:
        """Remove a region row from the list."""
        row = self._region_rows.pop(region_id, None)
        if row is not None:
            row.deleteLater()

        # Disable buttons if no regions
        if not self._region_rows:
            self.clear_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

    def clear_regions(self) -> None:
        """Remove all region rows."""
        for row in self._region_rows.values():
            row.deleteLater()
        self._region_rows.clear()
        self.clear_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle mouse events on region rows."""
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.MouseButtonPress:
            region_id = obj.property("region_id")
            if region_id is not None:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.region_clicked.emit(region_id)
                    return True
        return super().eventFilter(obj, event)


class AtlasSlideListWidget(QWidget):
    """Widget for displaying and managing slides in the atlas builder.

    This widget shows a list of slides that have been added to the atlas,
    with controls to remove individual slides.

    Signals
    -------
    slide_removed(str)
        Emitted when a slide is removed from the list.
    selection_changed()
        Emitted when the slide selection changes.
    """
    slide_removed = Signal(str)
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._slide_data: Dict[str, Dict] = {}  # slide_name -> {features, coords, h5_path}

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Scroll area for slide items
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        scroll.setWidget(self._content)
        layout.addWidget(scroll)

        self._slide_rows: Dict[str, QWidget] = {}

    def add_slide(self, slide_name: str, features: np.ndarray, coords: np.ndarray,
                  h5_path: str = "", color: QColor = None) -> None:
        """Add a slide to the list.

        Parameters
        ----------
        slide_name : str
            Name of the slide.
        features : np.ndarray
            Feature vectors.
        coords : np.ndarray
            Patch coordinates.
        h5_path : str
            Path to the HDF5 file.
        color : QColor, optional
            Display color for the slide.
        """
        if slide_name in self._slide_rows:
            return  # Already added

        # Store data
        self._slide_data[slide_name] = {
            'features': features,
            'coords': coords,
            'h5_path': h5_path
        }

        # Create row widget
        row = QFrame()
        row.setFrameShape(QFrame.StyledPanel)
        row.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border-radius: 3px; padding: 2px; }"
            "QFrame:hover { background-color: #e0e0e0; }"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(5, 3, 5, 3)
        row_layout.setSpacing(5)

        # Color indicator
        if color:
            color_label = QLabel()
            color_label.setFixedSize(12, 12)
            color_label.setStyleSheet(
                f"background-color: {color.name()}; border-radius: 6px;"
            )
            row_layout.addWidget(color_label)

        # Slide name
        name_label = QLabel(slide_name[:25] + "..." if len(slide_name) > 25 else slide_name)
        name_label.setToolTip(slide_name)
        row_layout.addWidget(name_label, stretch=1)

        # Patch count
        count_label = QLabel(f"({len(features):,})")
        count_label.setStyleSheet("color: gray; font-size: 9px;")
        row_layout.addWidget(count_label)

        # Remove button
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(18, 18)
        remove_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; color: #888; border: none; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        remove_btn.setToolTip("Remove from atlas")
        remove_btn.clicked.connect(lambda: self._remove_slide(slide_name))
        row_layout.addWidget(remove_btn)

        # Insert before the stretch
        self._content_layout.insertWidget(
            self._content_layout.count() - 1, row
        )
        self._slide_rows[slide_name] = row

        self.selection_changed.emit()

    def _remove_slide(self, slide_name: str) -> None:
        """Remove a slide from the list."""
        row = self._slide_rows.pop(slide_name, None)
        if row:
            row.deleteLater()
        self._slide_data.pop(slide_name, None)
        self.slide_removed.emit(slide_name)
        self.selection_changed.emit()

    def clear(self) -> None:
        """Remove all slides from the list."""
        for row in self._slide_rows.values():
            row.deleteLater()
        self._slide_rows.clear()
        self._slide_data.clear()
        self.selection_changed.emit()

    def get_slide_names(self) -> List[str]:
        """Get list of slide names in the atlas."""
        return list(self._slide_data.keys())

    def get_slide_data(self, slide_name: str) -> Optional[Dict]:
        """Get stored data for a slide."""
        return self._slide_data.get(slide_name)

    def get_all_slide_data(self) -> Dict[str, Dict]:
        """Get all stored slide data."""
        return self._slide_data.copy()

    def count(self) -> int:
        """Get number of slides in the list."""
        return len(self._slide_data)


class ThumbnailLoadTask(QObject, QRunnable):
    """Background task for loading slide thumbnails."""

    loaded = Signal(str, QImage)
    failed = Signal(str, str)

    def __init__(self, slide_name: str, image_path: str, max_size: int) -> None:
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._slide_name = slide_name
        self._image_path = image_path
        self._max_size = max_size

    def run(self) -> None:
        try:
            image = data_loader.load_thumbnail(self._image_path, max_size=self._max_size)
            qimage = QImage(ImageQt(image))
            self.loaded.emit(self._slide_name, qimage)
        except Exception as exc:
            self.failed.emit(self._slide_name, str(exc))


class SlideThumbnailItem(QFrame):
    """Clickable slide thumbnail widget with loading and error states."""

    clicked = Signal(str)

    def __init__(self, slide_name: str, thumb_size: QSize, parent=None) -> None:
        super().__init__(parent)
        self._slide_name = slide_name
        self._thumb_size = thumb_size
        self._clickable = True

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { border: 1px solid #cfcfcf; border-radius: 4px; }"
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(self._thumb_size.width() + 16)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._thumb_container = QWidget()
        self._thumb_container.setFixedSize(self._thumb_size)

        self._stack = QStackedLayout(self._thumb_container)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(self._thumb_size)
        self._loading_bar.setStyleSheet("QProgressBar { border: none; }")

        self._thumb_label = QLabel()
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setFixedSize(self._thumb_size)
        self._thumb_label.setStyleSheet("background-color: #202020;")

        self._error_label = QLabel("Read Error")
        self._error_label.setAlignment(Qt.AlignCenter)
        self._error_label.setWordWrap(True)
        self._error_label.setFixedSize(self._thumb_size)
        self._error_label.setStyleSheet("color: #aa0000; font-size: 9px;")

        self._stack.addWidget(self._loading_bar)
        self._stack.addWidget(self._thumb_label)
        self._stack.addWidget(self._error_label)
        self._stack.setCurrentWidget(self._loading_bar)

        layout.addWidget(self._thumb_container, alignment=Qt.AlignCenter)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._name_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._name_label)

        self._update_name_label()

    def mousePressEvent(self, event) -> None:
        if self._clickable and event.button() == Qt.LeftButton:
            self.clicked.emit(self._slide_name)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_name_label()

    def set_loading(self) -> None:
        """Show loading indicator."""
        self._stack.setCurrentWidget(self._loading_bar)
        self._clickable = True
        self.setCursor(Qt.PointingHandCursor)

    def set_thumbnail(self, image: QImage) -> None:
        """Set the thumbnail image for the slide."""
        pixmap = QPixmap.fromImage(image)
        pixmap = self._crop_pixmap(pixmap)
        self._thumb_label.setPixmap(pixmap)
        self._stack.setCurrentWidget(self._thumb_label)
        self._clickable = True
        self.setCursor(Qt.PointingHandCursor)

    def set_error(self, message: str) -> None:
        """Set error state when thumbnail fails to load."""
        self._error_label.setToolTip(message)
        self._stack.setCurrentWidget(self._error_label)
        self._clickable = False
        self.setCursor(Qt.ArrowCursor)

    def set_selected(self, selected: bool) -> None:
        """Highlight the selected slide."""
        if selected:
            self.setStyleSheet(
                "QFrame { border: 2px solid #4a90e2; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { border: 1px solid #cfcfcf; border-radius: 4px; }"
            )

    def _update_name_label(self) -> None:
        metrics = self._name_label.fontMetrics()
        available_width = max(self._name_label.width() - 6, 20)
        elided = metrics.elidedText(self._slide_name, Qt.ElideRight, available_width)
        self._name_label.setText(elided)
        self._name_label.setToolTip(self._slide_name)

    def _crop_pixmap(self, pixmap: QPixmap) -> QPixmap:
        target = self._thumb_size
        scaled = pixmap.scaled(target, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        x = max((scaled.width() - target.width()) // 2, 0)
        y = max((scaled.height() - target.height()) // 2, 0)
        return scaled.copy(x, y, target.width(), target.height())


class SlideThumbnailListWidget(QWidget):
    """Scrollable list of slide thumbnails with lazy loading."""

    slide_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thumb_size = QSize(120, 80)
        self._items: Dict[str, SlideThumbnailItem] = {}
        self._slide_paths: Dict[str, str] = {}
        self._tasks: Dict[str, ThumbnailLoadTask] = {}
        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(2)
        self._scroll_area: Optional[QScrollArea] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Slides")
        header.setStyleSheet("font-weight: bold; padding: 4px 0;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(self._thumb_size.height() + 40)
        self._scroll_area = scroll

        self._content = QWidget()
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._content_layout.addStretch()

        scroll.setWidget(self._content)
        layout.addWidget(scroll)

    def set_slides(self, slide_names: List[str], slide_infos: Dict[str, data_loader.SlideInfo]) -> None:
        """Populate the thumbnail list with slides."""
        self._clear_items()
        self._slide_paths.clear()

        for slide_name in slide_names:
            info = slide_infos.get(slide_name)
            image_path = info.image_path if info else ""
            self._slide_paths[slide_name] = image_path

            item = SlideThumbnailItem(slide_name, self._thumb_size)
            item.clicked.connect(self.slide_selected.emit)
            self._content_layout.insertWidget(self._content_layout.count() - 1, item)
            self._items[slide_name] = item

            if not image_path:
                item.set_error("No image path available")
                continue

            item.set_loading()
            self._start_thumbnail_load(slide_name, image_path)

    def set_current_slide(self, slide_name: str) -> None:
        """Highlight the currently selected slide."""
        for name, item in self._items.items():
            item.set_selected(name == slide_name)

    def _clear_items(self) -> None:
        for item in self._items.values():
            item.deleteLater()
        self._items.clear()
        self._tasks.clear()

    def _start_thumbnail_load(self, slide_name: str, image_path: str) -> None:
        max_size = max(self._thumb_size.width(), self._thumb_size.height()) * 2
        task = ThumbnailLoadTask(slide_name, image_path, max_size)
        task.loaded.connect(self._on_thumbnail_loaded)
        task.failed.connect(self._on_thumbnail_failed)
        self._tasks[slide_name] = task
        self._thread_pool.start(task)

    def _on_thumbnail_loaded(self, slide_name: str, image: QImage) -> None:
        item = self._items.get(slide_name)
        if item is None:
            return
        item.set_thumbnail(image)
        self._tasks.pop(slide_name, None)

    def _on_thumbnail_failed(self, slide_name: str, message: str) -> None:
        item = self._items.get(slide_name)
        if item is None:
            return
        item.set_error(message)
        self._tasks.pop(slide_name, None)


class ModelSelectionDialog(QDialog):
    """Dialog for selecting models, magnification, and patch size."""

    def __init__(
        self,
        slide_info: data_loader.SlideInfo,
        current_selection: Optional[ModelSelection] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Models")
        self._slide_info = slide_info
        self._current_selection = current_selection

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._model_selector = ModelMultiSelector()
        self._model_selector.addItems(sorted(slide_info.models.keys()))
        self._model_selector.selectionChanged.connect(self._on_model_selection_changed)

        self._mag_combo = QComboBox()
        self._mag_combo.currentTextChanged.connect(self._on_mag_changed)

        self._patch_combo = QComboBox()
        self._patch_combo.currentTextChanged.connect(self._update_ok_state)

        layout.addWidget(QLabel("Models:"))
        layout.addWidget(self._model_selector)
        layout.addWidget(QLabel("Magnification:"))
        layout.addWidget(self._mag_combo)
        layout.addWidget(QLabel("Patch size:"))
        layout.addWidget(self._patch_combo)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        if current_selection:
            available_models = set(slide_info.models.keys())
            preselect = [m for m in current_selection.models if m in available_models]
            self._model_selector.setSelectedModels(preselect)

        self._update_magnifications(prefer=current_selection.magnification if current_selection else None)
        self._update_patches(prefer=current_selection.patch_size if current_selection else None)
        self._update_ok_state()

    def selection(self) -> Optional[ModelSelection]:
        """Return the selected configuration if valid."""
        models = self._model_selector.getSelectedModels()
        mag = self._mag_combo.currentText()
        patch = self._patch_combo.currentText()
        if not models or not mag or not patch:
            return None
        return ModelSelection(models=models, magnification=mag, patch_size=patch)

    def _on_model_selection_changed(self) -> None:
        self._update_magnifications()
        self._update_patches()
        self._update_ok_state()

    def _on_mag_changed(self, mag: str) -> None:
        self._update_patches()
        self._update_ok_state()

    def _update_magnifications(self, prefer: Optional[str] = None) -> None:
        models = self._model_selector.getSelectedModels()
        mags = self._get_common_magnifications(models)
        self._mag_combo.blockSignals(True)
        self._mag_combo.clear()
        self._mag_combo.addItems(mags)
        if prefer and prefer in mags:
            self._mag_combo.setCurrentText(prefer)
        self._mag_combo.blockSignals(False)

    def _update_patches(self, prefer: Optional[str] = None) -> None:
        models = self._model_selector.getSelectedModels()
        mag = self._mag_combo.currentText()
        patches = self._get_common_patches(models, mag)
        self._patch_combo.blockSignals(True)
        self._patch_combo.clear()
        self._patch_combo.addItems(patches)
        if prefer and prefer in patches:
            self._patch_combo.setCurrentText(prefer)
        self._patch_combo.blockSignals(False)

    def _update_ok_state(self) -> None:
        selection = self.selection()
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(selection is not None)

    def _get_common_magnifications(self, models: List[str]) -> List[str]:
        mag_sets = []
        for model_name in models:
            model_dict = self._slide_info.models.get(model_name, {})
            mag_sets.append(set(model_dict.keys()))
        if not mag_sets:
            return []
        return sorted(set.intersection(*mag_sets))

    def _get_common_patches(self, models: List[str], mag: str) -> List[str]:
        if not models or not mag:
            return []
        patch_sets = []
        for model_name in models:
            patches = self._slide_info.models.get(model_name, {}).get(mag, {})
            patch_sets.append(set(patches.keys()))
        if not patch_sets:
            return []
        return sorted(set.intersection(*patch_sets))


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
        # Persisted model selection for session
        self._model_selection: Optional[ModelSelection] = None
        # Track cascade animation state to prevent hover interference
        self._animation_in_progress: bool = False
        # Local region selection mode state
        self._selection_mode: SelectionMode = SelectionMode.KMEANS
        self._local_region_clusters: Dict[int, LocalRegionCluster] = {}
        self._next_local_cluster_id: int = 0
        self._local_region_radius: float = 50.0
        self._update_model_selection_label()
        self._set_model_dependent_ui_enabled(False)

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
        self.root_label.setWordWrap(True)
        self.root_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        folder_row.addWidget(self.root_label)
        main_vbox.addLayout(folder_row)
        
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

        # Tab 1: K-means Clusters (existing ClusterLegendWidget)
        kmeans_tab = QWidget()
        kmeans_layout = QVBoxLayout(kmeans_tab)
        kmeans_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_legend = ClusterLegendWidget()
        kmeans_layout.addWidget(self.cluster_legend)
        kmeans_layout.addStretch()
        self.sidebar_tabs.addTab(kmeans_tab, "K-means")

        # Connect K-means legend signals
        self.cluster_legend.cluster_clicked.connect(self._on_legend_cluster_clicked)
        self.cluster_legend.cluster_toggled.connect(self._on_legend_cluster_toggled)
        self.cluster_legend.cluster_rename.connect(self._on_rename_cluster)
        self.cluster_legend.cluster_export.connect(self._on_export_single_cluster)
        self.cluster_legend.export_all_requested.connect(self._on_export_all_clusters)

        # Tab 2: Local Region Selection (new LocalRegionWidget)
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        local_layout.setContentsMargins(0, 0, 0, 0)
        self.local_region_widget = LocalRegionWidget()
        local_layout.addWidget(self.local_region_widget)
        local_layout.addStretch()
        self.sidebar_tabs.addTab(local_tab, "Local Region")

        # Connect Local Region widget signals
        self.local_region_widget.radius_changed.connect(self._on_local_region_radius_changed)
        self.local_region_widget.region_clicked.connect(self._on_local_region_clicked)
        self.local_region_widget.region_deleted.connect(self._on_local_region_deleted)
        self.local_region_widget.clear_all_requested.connect(self._clear_local_region_clusters)
        self.local_region_widget.export_requested.connect(self._export_local_region_clusters)

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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.sidebar_tabs)
        splitter.addWidget(self.graphics_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

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
            self.slide_thumbnail_list.set_slides(slide_names, self.slides)
            
            # Clear subsequent selectors and view
            self._model_selection = None
            self._update_model_selection_label()
            self._set_model_dependent_ui_enabled(False)
            self.model_select_btn.setEnabled(bool(slide_names))
            self.model_selector.clear()
            self.mag_combo.clear()
            self.patch_combo.clear()
            self.graphics_view.scene().clear()
            if hasattr(self.graphics_view, 'rect_items'):
                self.graphics_view.rect_items.clear()
            self._clear_scatter_view()
            
        except Exception as e:
            print(f"DEBUG: Error parsing directory: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to parse folder:\n{str(e)}")
            return

    def _on_thumbnail_slide_selected(self, slide_name: str) -> None:
        """Handle selection from the thumbnail list."""
        if slide_name and slide_name != self.slide_combo.currentText():
            self.slide_combo.setCurrentText(slide_name)

    def _open_model_selection_dialog(self) -> None:
        """Open the model selection dialog for the current slide."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            QMessageBox.warning(self, "Select Slide", "Please select a slide first.")
            return
        info = self.slides.get(slide_name)
        if info is None:
            QMessageBox.warning(self, "Select Slide", "Slide information is unavailable.")
            return

        dialog = ModelSelectionDialog(info, self._model_selection, self)
        if dialog.exec():
            selection = dialog.selection()
            if selection is None:
                QMessageBox.warning(self, "Model Selection", "Select models, magnification, and patch size.")
                return
            self._model_selection = selection
            if not self._apply_model_selection(selection):
                self._model_selection = None
                self._update_model_selection_label()
                self._set_model_dependent_ui_enabled(False)
                return
            self._update_model_selection_label()
            self._set_model_dependent_ui_enabled(True)
            self._load_current_data()

    def _on_slide_changed(self, slide_name: str) -> None:
        """Load slide preview and apply any active model selection."""
        print(f"DEBUG: Slide changed to: {slide_name}")
        if not slide_name:
            print("DEBUG: No slide name provided")
            self.model_select_btn.setEnabled(False)
            return
        self.slide_thumbnail_list.set_current_slide(slide_name)
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: No slide info found for {slide_name}")
            self.model_select_btn.setEnabled(False)
            return

        self.model_select_btn.setEnabled(True)
        self._load_slide_image_only(slide_name)

        if self._model_selection:
            if self._apply_model_selection(self._model_selection):
                self._set_model_dependent_ui_enabled(True)
                self._load_current_data()
            else:
                self._model_selection = None
                self._update_model_selection_label()
                self._set_model_dependent_ui_enabled(False)
        else:
            self._set_model_dependent_ui_enabled(False)

    def _populate_model_selector(self, info: data_loader.SlideInfo) -> None:
        """Populate the hidden model selector for the current slide."""
        model_names = sorted(info.models.keys())
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(model_names)
        self.model_selector.blockSignals(False)

    def _apply_model_selection(self, selection: ModelSelection) -> bool:
        """Apply a stored model selection to the current slide."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return False
        info = self.slides.get(slide_name)
        if info is None:
            return False

        self._populate_model_selector(info)
        available_models = set(info.models.keys())
        missing = [model for model in selection.models if model not in available_models]
        if missing:
            QMessageBox.warning(
                self,
                "Models Unavailable",
                f"Selected models not available for {slide_name}: {', '.join(missing)}",
            )
            return False

        magnifications = self._get_common_magnifications(selection.models, info)
        if selection.magnification not in magnifications:
            QMessageBox.warning(
                self,
                "Magnification Unavailable",
                f"Magnification {selection.magnification} not available for selected models.",
            )
            return False

        patches = self._get_common_patches(selection.models, selection.magnification, info)
        if selection.patch_size not in patches:
            QMessageBox.warning(
                self,
                "Patch Size Unavailable",
                f"Patch size {selection.patch_size} not available for selected models.",
            )
            return False

        self.model_selector.blockSignals(True)
        self.model_selector.setSelectedModels(selection.models)
        self.model_selector.blockSignals(False)

        self.mag_combo.blockSignals(True)
        self.mag_combo.clear()
        self.mag_combo.addItems(magnifications)
        self.mag_combo.setCurrentText(selection.magnification)
        self.mag_combo.blockSignals(False)

        self.patch_combo.blockSignals(True)
        self.patch_combo.clear()
        self.patch_combo.addItems(patches)
        self.patch_combo.setCurrentText(selection.patch_size)
        self.patch_combo.blockSignals(False)

        self._update_model_selection_label()
        return True

    def _get_common_magnifications(
        self,
        models: List[str],
        info: data_loader.SlideInfo,
    ) -> List[str]:
        mag_sets = []
        for model_name in models:
            model_dict = info.models.get(model_name, {})
            mag_sets.append(set(model_dict.keys()))
        if not mag_sets:
            return []
        return sorted(set.intersection(*mag_sets))

    def _get_common_patches(
        self,
        models: List[str],
        magnification: str,
        info: data_loader.SlideInfo,
    ) -> List[str]:
        if not models or not magnification:
            return []
        patch_sets = []
        for model_name in models:
            patches = info.models.get(model_name, {}).get(magnification, {})
            patch_sets.append(set(patches.keys()))
        if not patch_sets:
            return []
        return sorted(set.intersection(*patch_sets))

    def _update_model_selection_label(self) -> None:
        """Update the summary label for the current model selection."""
        if self._model_selection is None:
            self.model_selection_label.setText("No model selected")
            self.model_selection_label.setToolTip("No model selected")
            return
        model_str = ", ".join(self._model_selection.models)
        summary = (
            f"Model: {model_str} | "
            f"{self._model_selection.magnification} | "
            f"{self._model_selection.patch_size}"
        )
        self.model_selection_label.setText(summary)
        self.model_selection_label.setToolTip(summary)

    def _set_model_dependent_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable model-dependent UI elements."""
        self.cluster_legend.setEnabled(enabled)
        self.local_region_widget.setEnabled(enabled)
        self.scatter_view.setEnabled(enabled)
        self.atlas_add_current_btn.setEnabled(enabled)
        self.cluster_spin.setEnabled(enabled)
        self.export_action.setEnabled(enabled and bool(self._selected_clusters))

    def _clear_scatter_view(self) -> None:
        """Clear scatter view contents when no model is selected."""
        self.scatter_view.scene().clear()
        self.scatter_view._scatter_items.clear()
        self.scatter_view.labels = None
        self.scatter_view.cluster_colors = []
        self.scatter_view.set_animation_active(False)

    def _show_preview_blocked_toast(self) -> None:
        """Show a short-lived toast when preview interactions are blocked."""
        if self._model_selection is not None:
            return

        if self._preview_toast is None:
            self._preview_toast = QLabel(self)
            self._preview_toast.setStyleSheet(
                "QLabel {"
                "background-color: rgba(30, 30, 30, 200);"
                "color: white;"
                "padding: 6px 10px;"
                "border-radius: 6px;"
                "font-size: 10pt;"
                "}"
            )
        self._preview_toast.setText("No model selected")
        self._preview_toast.adjustSize()

        cursor_pos = QCursor.pos()
        toast_pos = cursor_pos + QPoint(12, 12)
        self._preview_toast.move(toast_pos)
        self._preview_toast.show()
        self._preview_toast.raise_()

        QTimer.singleShot(2000, self._preview_toast.hide)

    def _load_slide_image_only(self, slide_name: str) -> None:
        """Load a slide preview without clustering."""
        info = self.slides.get(slide_name)
        if info is None:
            return

        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        adaptive_enabled = self.adaptive_zoom_action.isChecked()
        self.graphics_view.set_adaptive_mode(adaptive_enabled)
        use_adaptive = adaptive_enabled and data_loader.is_openslide_available()

        if use_adaptive:
            try:
                from openslide import OpenSlide  # type: ignore

                slide = OpenSlide(info.image_path)
                slide_dimensions = slide.dimensions
                slide.close()
                empty_coords = np.zeros((0, 2), dtype=float)
                empty_labels = np.zeros((0,), dtype=int)
                adaptive_loaded = self.graphics_view.load_slide_adaptive(
                    info.image_path,
                    empty_coords,
                    0.0,
                    empty_labels,
                    [],
                    slide_dimensions,
                )
                if adaptive_loaded:
                    self.cluster_legend.clear_cluster_names()
                    self.cluster_legend.update_clusters(np.zeros((0,), dtype=int), [])
                    self._clear_scatter_view()
                    self._current_embedding = None
                    self._current_features = None
                    self._current_coords_thumb = None
                    self._current_coords_lv0 = None
                    self._current_patch_size_thumb = None
                    self._current_patch_size_lv0 = None
                    self._coord_scale_factor = None
                    self._current_labels = None
                    self._clear_selected_clusters()
                    self.progress_bar.setVisible(False)
                    self.status_label.setText("Preview only — select a model to enable clustering.")
                    return
            except Exception as exc:
                print(f"DEBUG: Adaptive preview failed: {exc}")

        try:
            thumb_image = data_loader.load_thumbnail(info.image_path)
        except Exception as exc:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load slide thumbnail:\n{exc}")
            return

        empty_coords = np.zeros((0, 2), dtype=float)
        empty_labels = np.zeros((0,), dtype=int)
        self.graphics_view.load_slide(thumb_image, empty_coords, 0.0, empty_labels, [])
        self.cluster_legend.clear_cluster_names()
        self.cluster_legend.update_clusters(empty_labels, [])
        self._clear_scatter_view()

        self._current_embedding = None
        self._current_features = None
        self._current_coords_thumb = None
        self._current_coords_lv0 = None
        self._current_patch_size_thumb = None
        self._current_patch_size_lv0 = None
        self._coord_scale_factor = None
        self._current_labels = None
        self._clear_selected_clusters()
        self.status_label.setText("Select a model to enable clustering.")

        self.progress_bar.setVisible(False)

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
        if self._model_selection is None:
            return
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
        if self._model_selection is None:
            return
        # Validate model compatibility before loading
        self._validate_model_compatibility()
        self._load_current_data()

    def _on_cluster_changed(self, k: int) -> None:
        """Recompute clusters when the cluster count slider changes."""
        if self._model_selection is None:
            return
        print(f"DEBUG: Cluster spin changed to {k}; reclustering current features")
        self._load_current_data(recluster_only=True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        """Show or hide patch overlays based on menu action state."""
        print(f"DEBUG: Overlay toggled; visible={checked}")
        # Adjust visibility of patch rects
        for rect in getattr(self.graphics_view, 'rect_items', []):
            rect.setVisible(checked)
    
    def _on_adaptive_zoom_toggled(self, checked: bool) -> None:
        """Toggle adaptive zoom mode and reload the current slide."""
        print(f"DEBUG: Adaptive zoom toggled; enabled={checked}")
        self.graphics_view.set_adaptive_mode(checked)
        # Reload the current slide in the new mode
        if self.slide_combo.currentText():
            if self._model_selection is None:
                self._load_slide_image_only(self.slide_combo.currentText())
            else:
                self._load_current_data()

    def _on_slide_patch_hovered(self, idx: int, state: bool) -> None:
        """Highlight the scatter point corresponding to the hovered slide patch."""
        if not self.scatter_view:
            return
        # Skip scatter opacity changes if animation is in progress
        if self._animation_in_progress:
            return
        # When hover leaves or invalid index
        if idx == -1 or not state:
            # Hide slide popup
            self.slide_info_popup.hide_popup()
            # Clear hover state in scatter
            if self._hovered_scatter_idx is not None:
                print(f"DEBUG: Slide hover leave for scatter index {self._hovered_scatter_idx}")
            # If no selected clusters, reset all to default opacity
            if not self._selected_clusters:
                if self.scatter_view.labels is None:
                    return
                print("DEBUG: No selected clusters; resetting scatter opacities to default")
                for item in self.scatter_view._scatter_items:
                    item.setOpacity(0.6)
            else:
                # Apply persistent opacity reduction for selected clusters
                print("DEBUG: Maintaining selected cluster opacities on hover leave")
                self._apply_selected_cluster_styles()
            self._hovered_scatter_idx = None
        else:
            # Hover over a valid patch
            # FIRST: Restore the previous hovered scatter point if transitioning
            if self._hovered_scatter_idx is not None and self._hovered_scatter_idx != idx:
                prev_idx = self._hovered_scatter_idx
                # Determine the correct baseline opacity for the previous point
                if self._selected_clusters and self.scatter_view.labels is not None:
                    prev_label = int(self.scatter_view.labels[prev_idx])
                    if prev_label in self._selected_clusters:
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

            # Show slide popup in bottom-left corner
            self._update_and_show_slide_popup(idx)

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
            # Hide popup
            self.patch_info_popup.hide_popup()
            # Skip scatter opacity changes if animation is in progress
            # to avoid interrupting the cascade effect
            if self._animation_in_progress:
                return
            # Restore scatter opacities based on whether clusters are selected
            if not self._selected_clusters:
                # No active selection: return all points to medium opacity
                if self.scatter_view and self.scatter_view._scatter_items:
                    for item in self.scatter_view._scatter_items:
                        item.setOpacity(0.6)
            else:
                # Active selection persists low/high styling
                self._apply_selected_cluster_styles()
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

        # Show popup with patch info
        self._update_and_show_popup(idx)

    def _on_scatter_cluster_selected(self, cluster: int, ctrl_pressed: bool) -> None:
        """Respond to cluster selection on scatter plot: highlight slide patches."""
        print(f"DEBUG: Scatter cluster selected {cluster}")
        if self.graphics_view.labels is None:
            return
        if ctrl_pressed:
            if not self._add_selected_cluster(cluster):
                return
        else:
            self._set_selected_clusters({cluster})
        # Prepare scatter baseline for cascade
        self._prepare_scatter_for_cascade(cluster)
        # Use the first patch's center as click point approximate
        cluster_indices = np.where(self.graphics_view.labels == cluster)[0]
        if cluster_indices.size == 0:
            return
        idx0 = cluster_indices[0]
        x, y = self.graphics_view.coords[idx0] + self.graphics_view.patch_size / 2.0
        self._start_slide_cascade(cluster, (x, y))

    def _on_legend_cluster_clicked(self, cluster: int, ctrl_pressed: bool) -> None:
        """Handle left-click on cluster in legend - trigger cascade from centroid."""
        print(f"DEBUG: Legend cluster clicked {cluster}")

        # Validate data is loaded
        self._animation_in_progress = True
        self._animation_in_progress = True
        if self.graphics_view.labels is None or self.graphics_view.coords is None:
            return

        cluster_coords = self.graphics_view.coords[cluster_indices]
        centroid_x = cluster_coords[:, 0].mean() + self.graphics_view.patch_size / 2.0
        centroid_y = cluster_coords[:, 1].mean() + self.graphics_view.patch_size / 2.0

        self._start_slide_cascade(cluster, (centroid_x, centroid_y))

    def _on_legend_cluster_toggled(self, cluster: int, checked: bool) -> None:
        """Handle checkbox toggle for cluster selection."""
        if self.graphics_view.labels is None or self.graphics_view.coords is None:
            return
        if checked:
            if not self._add_selected_cluster(cluster):
                return
            self._prepare_scatter_for_cascade(cluster)
            cluster_indices = np.where(self.graphics_view.labels == cluster)[0]
            if cluster_indices.size == 0:
                return
            cluster_coords = self.graphics_view.coords[cluster_indices]
            centroid_x = cluster_coords[:, 0].mean() + self.graphics_view.patch_size / 2.0
            centroid_y = cluster_coords[:, 1].mean() + self.graphics_view.patch_size / 2.0
            self._start_slide_cascade(cluster, (centroid_x, centroid_y))
        else:
            if self._remove_selected_cluster(cluster):
                if not self._animation_in_progress:
                    self._apply_selected_cluster_styles()

    def _on_rename_cluster(self, cluster: int) -> None:
        """Handle rename cluster request from legend context menu."""
        current_name = self.cluster_legend.get_cluster_name(cluster)
        new_name, ok = QInputDialog.getText(
            self, "Rename Cluster", "Enter new name:",
            text=current_name
        )
        if ok and new_name.strip():
            self.cluster_legend.set_cluster_name(cluster, new_name.strip())
            # Refresh legend display
            if self._current_labels is not None and self.scatter_view.cluster_colors:
                self.cluster_legend.update_clusters(
                    self._current_labels,
                    [self._qcolor_to_hsl_string(c) for c in self.scatter_view.cluster_colors]
                )

    def _on_export_single_cluster(self, cluster: int) -> None:
        """Handle export single cluster request from legend context menu."""
        self._export_single_cluster(cluster)

    def _on_export_all_clusters(self) -> None:
        """Export all clusters to a single GeoJSON file."""
        # Validate data
        if self._current_labels is None:
            QMessageBox.warning(self, "Export Error", "No clustering data available.")
            return
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "Export Error", "Coordinate data not available.")
            return

        # Get file path
        default_name = "all_clusters.geojson"
        if hasattr(self, '_root_dir') and self._root_dir:
            default_path = os.path.join(self._root_dir, default_name)
        else:
            default_path = default_name

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Clusters", default_path,
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.endswith('.geojson'):
            file_path += '.geojson'

        # Build features for all clusters
        features = []
        unique_clusters = np.unique(self._current_labels)

        for cluster in unique_clusters:
            cluster = int(cluster)
            # Merge patches for this cluster
            merged_coords = self._merge_cluster_patches(cluster)
            if not merged_coords:
                continue

            # Get cluster color
            if cluster < len(self.scatter_view.cluster_colors):
                color_hsl = self._qcolor_to_hsl_string(self.scatter_view.cluster_colors[cluster])
                color_rgb = self._hsl_to_qupath_rgb(color_hsl)
            else:
                color_rgb = -16776961  # Default blue

            # Get cluster name
            cluster_name = self.cluster_legend.get_cluster_name(cluster)

            # Create feature for each polygon in merged result
            for poly_coords in merged_coords:
                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {
                            "name": cluster_name,
                            "colorRGB": color_rgb
                        },
                        "isLocked": False,
                        "measurements": []
                    }
                }
                features.append(feature)

        # Build GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Write file
        try:
            with open(file_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(unique_clusters)} clusters ({len(features)} polygons) to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write file:\n{str(e)}")

    def _qcolor_to_hsl_string(self, color: QColor) -> str:
        """Convert QColor to HSL string format."""
        h, s, l, _ = color.getHslF()
        return f"hsl({int(h * 360)}, {int(s * 100)}%, {int(l * 100)}%)"

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

    def _update_and_show_popup(self, index: int) -> None:
        """Update popup content and show near cursor.

        Parameters
        ----------
        index : int
            Patch index to display info for.
        """
        if index < 0:
            self.patch_info_popup.hide_popup()
            return

        # Get coordinates
        coords = None
        if self._current_coords_lv0 is not None and index < len(self._current_coords_lv0):
            x, y = self._current_coords_lv0[index]
            coords = (float(x), float(y))

        # Get cluster label and color
        cluster = None
        cluster_color = None
        if self._current_labels is not None and index < len(self._current_labels):
            cluster = int(self._current_labels[index])
            if self.scatter_view and hasattr(self.scatter_view, 'cluster_colors'):
                if cluster < len(self.scatter_view.cluster_colors):
                    cluster_color = self.scatter_view.cluster_colors[cluster]

        # Get distance to centroid
        distance = self._get_distance_to_centroid(index)

        # Update popup content
        self.patch_info_popup.update_info(index, coords, cluster, distance, cluster_color)

        # Show at current cursor position
        cursor_pos = QCursor.pos()
        self.patch_info_popup.show_at_cursor(cursor_pos)

    def _update_and_show_slide_popup(self, index: int) -> None:
        """Update slide popup content and show in bottom-left corner of slide view.

        Parameters
        ----------
        index : int
            Patch index to display info for.
        """
        if index < 0:
            self.slide_info_popup.hide_popup()
            return

        # Get coordinates
        coords = None
        if self._current_coords_lv0 is not None and index < len(self._current_coords_lv0):
            x, y = self._current_coords_lv0[index]
            coords = (float(x), float(y))

        # Get cluster label and color
        cluster = None
        cluster_color = None
        if self._current_labels is not None and index < len(self._current_labels):
            cluster = int(self._current_labels[index])
            if self.scatter_view and hasattr(self.scatter_view, 'cluster_colors'):
                if cluster < len(self.scatter_view.cluster_colors):
                    cluster_color = self.scatter_view.cluster_colors[cluster]

        # Get distance to centroid
        distance = self._get_distance_to_centroid(index)

        # Update popup content
        self.slide_info_popup.update_info(index, coords, cluster, distance, cluster_color)

        # Calculate bottom-left corner position of the graphics view
        view_rect = self.graphics_view.rect()
        popup_height = self.slide_info_popup.sizeHint().height()
        margin = 10
        bottom_left = self.graphics_view.mapToGlobal(
            QPoint(margin, view_rect.height() - popup_height - margin)
        )
        self.slide_info_popup.show_at_position(bottom_left)

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

    def _update_scatter_for_cluster(self, cluster: int, click_point: Tuple[float, float],
                                    ctrl_pressed: bool) -> None:
        """Prepare scatter plot for synchronized cascade animation.
        
        This method sets up the scatter plot's initial state before the slide
        animation begins. The actual cascade animation is handled through the
        patches_highlighted signal from SlideGraphicsView, which ensures both
        views animate in perfect synchronization.
        """
        if not hasattr(self, '_current_features'):
            return
        print(f"DEBUG: Preparing scatter for synchronized cascade, cluster {cluster}")
        if ctrl_pressed:
            if not self._add_selected_cluster(cluster):
                return
        else:
            self._set_selected_clusters({cluster})
        self._prepare_scatter_for_cascade(cluster)
        self._start_slide_cascade(cluster, click_point)

    def _prepare_scatter_for_cascade(self, cluster: int) -> None:
        """Prepare scatter points for a new cascade animation."""
        if not self.scatter_view or not self.scatter_view._scatter_items:
            return
        if self.scatter_view.labels is None:
            return
        self._animation_in_progress = True
        self.scatter_view.set_animation_active(True)
        for i, item in enumerate(self.scatter_view._scatter_items):
            label = int(self.scatter_view.labels[i])
            if label in self._selected_clusters and label != cluster:
                item.setOpacity(1.0)
            else:
                item.setOpacity(0.2)

    def _start_slide_cascade(self, cluster: int, click_point: Tuple[float, float]) -> None:
        """Start a slide cascade animation for a cluster."""
        if self.graphics_view.labels is None or self.graphics_view.coords is None:
            return
        cluster_indices = np.where(self.graphics_view.labels == cluster)[0]
        if cluster_indices.size == 0:
            return
        cluster_coords = self.graphics_view.coords[cluster_indices]
        order_local = radial_sweep_order(cluster_coords, click_point)
        order_global = cluster_indices[order_local]
        persisted = set(self._selected_clusters) - {cluster}
        self.graphics_view._start_animation(cluster, order_global.tolist(), persisted)

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

        # Apply correct styling based on current selection mode
        if self._selection_mode == SelectionMode.LOCAL_REGION:
            self._apply_local_region_cluster_styles()
        else:
            self._apply_selected_cluster_styles()

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
        """Handle export button click for selected clusters."""
        if not self._selected_clusters:
            QMessageBox.warning(
                self, "No Selection",
                "Please select a cluster first by clicking on the slide or scatter plot."
            )
            return

        # Check if we have the required coordinate data
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        selected = sorted(self._selected_clusters)
        if len(selected) == 1:
            self._export_single_cluster(selected[0])
            return

        prompt = QMessageBox(self)
        prompt.setWindowTitle("Export Selected Clusters")
        prompt.setText("Export selected clusters as one combined annotation or separate annotations?")
        combine_button = prompt.addButton("Combine", QMessageBox.AcceptRole)
        separate_button = prompt.addButton("Separate", QMessageBox.AcceptRole)
        prompt.addButton(QMessageBox.Cancel)
        prompt.exec()

        clicked = prompt.clickedButton()
        if clicked == combine_button:
            self._export_combined_clusters(selected)
        elif clicked == separate_button:
            self._export_separate_clusters(selected)

    def _export_single_cluster(self, cluster: int) -> None:
        """Export a single cluster to GeoJSON."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        default_name = f"Cluster_{cluster}"
        name, ok = QInputDialog.getText(
            self, "Annotation Name",
            "Enter a name for this annotation:",
            text=default_name
        )
        if not ok or not name.strip():
            return

        annotation_name = name.strip()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            f"{annotation_name}.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        polygons = self._merge_cluster_patches(cluster)
        if not polygons:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected cluster.")
            return

        color_rgb = self._get_cluster_color_rgb(cluster)
        features = self._build_geojson_features(polygons, annotation_name, color_rgb)
        self._write_geojson(file_path, features)

    def _export_combined_clusters(self, clusters: List[int]) -> None:
        """Export multiple clusters as a single combined annotation."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        name, ok = QInputDialog.getText(
            self, "Annotation Name",
            "Enter a name for the combined annotation:",
            text="Combined_Clusters"
        )
        if not ok or not name.strip():
            return
        annotation_name = name.strip()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            f"{annotation_name}.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        polygons = self._merge_selected_clusters_patches(clusters)
        if not polygons:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected clusters.")
            return

        color_rgb = self._get_cluster_color_rgb(clusters[0])
        features = self._build_geojson_features(polygons, annotation_name, color_rgb)
        self._write_geojson(file_path, features)

    def _export_separate_clusters(self, clusters: List[int]) -> None:
        """Export multiple clusters to a single GeoJSON file."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            "selected_clusters.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        features = []
        for cluster in clusters:
            polygons = self._merge_cluster_patches(cluster)
            if not polygons:
                continue
            cluster_name = self.cluster_legend.get_cluster_name(cluster)
            color_rgb = self._get_cluster_color_rgb(cluster)
            features.extend(self._build_geojson_features(polygons, cluster_name, color_rgb))

        if not features:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected clusters.")
            return

        self._write_geojson(file_path, features)

    def _build_geojson_features(self, polygons: List[List], name: str, color_rgb: int) -> List[Dict]:
        """Build GeoJSON features for a set of polygons."""
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
                        "name": name,
                        "colorRGB": color_rgb
                    },
                    "isLocked": False,
                    "measurements": []
                }
            }
            features.append(feature)
        return features

    def _write_geojson(self, file_path: str, features: List[Dict]) -> None:
        """Write GeoJSON features to a file."""
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        try:
            with open(file_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(features)} region(s) to:\n{file_path}"
            )
            print(f"DEBUG: Exported GeoJSON with {len(features)} features to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write file:\n{e}")
            print(f"DEBUG: Export failed: {e}")

    def _get_cluster_color_rgb(self, cluster: int) -> int:
        """Get QuPath color for a cluster."""
        if self.scatter_view and hasattr(self.scatter_view, 'cluster_colors'):
            if cluster < len(self.scatter_view.cluster_colors):
                color_hsl = self._qcolor_to_hsl_string(self.scatter_view.cluster_colors[cluster])
                return self._hsl_to_qupath_rgb(color_hsl)
        colours = generate_palette(int(self.graphics_view.labels.max()) + 1)
        return self._hsl_to_qupath_rgb(colours[cluster])

    def _merge_selected_clusters_patches(self, clusters: List[int]) -> List[List]:
        """Merge patches from multiple clusters into polygons."""
        if self.graphics_view.labels is None:
            return []
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            return []

        boxes = []
        for cluster in clusters:
            indices = np.where(self.graphics_view.labels == cluster)[0]
            for idx in indices:
                x, y = self._current_coords_thumb[idx]
                size = self._current_patch_size_thumb
                boxes.append(box(x, y, x + size, y + size))

        if not boxes:
            return []

        merged = unary_union(boxes)
        scale = self._coord_scale_factor

        if merged.geom_type == 'Polygon':
            polygons = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polygons = list(merged.geoms)
        else:
            print(f"DEBUG: Unexpected geometry type from unary_union: {merged.geom_type}")
            polygons = []

        result = []
        for poly in polygons:
            coords = []
            for ring in [poly.exterior] + list(poly.interiors):
                scaled_ring = [[int(x * scale), int(y * scale)] for x, y in ring.coords]
                coords.append(scaled_ring)
            result.append(coords)
        return result

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
    def _update_export_action(self) -> None:
        """Enable or disable export based on selection state."""
        if self._model_selection is None:
            self.export_action.setEnabled(False)
            return
        self.export_action.setEnabled(bool(self._selected_clusters))

    def _set_selected_clusters(self, clusters: set[int]) -> None:
        """Replace selected clusters and sync UI state."""
        self._selected_clusters = set(clusters)
        self._sync_legend_checkboxes()
        self._update_export_action()

    def _add_selected_cluster(self, cluster: int) -> bool:
        """Add a cluster to the selection if not present."""
        if cluster in self._selected_clusters:
            return False
        self._selected_clusters.add(cluster)
        self._sync_legend_checkboxes()
        self._update_export_action()
        return True

    def _remove_selected_cluster(self, cluster: int) -> bool:
        """Remove a cluster from the selection if present."""
        if cluster not in self._selected_clusters:
            return False
        self._selected_clusters.remove(cluster)
        self._sync_legend_checkboxes()
        self._update_export_action()
        return True

    def _clear_selected_clusters(self) -> None:
        """Clear all selected clusters."""
        self._selected_clusters.clear()
        self._sync_legend_checkboxes()
        self._update_export_action()

    def _sync_legend_checkboxes(self) -> None:
        """Sync legend checkbox states without triggering cascades."""
        for cluster_id, checkbox in self.cluster_legend._checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(cluster_id in self._selected_clusters)
            checkbox.blockSignals(False)

    def _apply_selected_cluster_styles(self) -> None:
        """Apply opacity styling to scatter points based on selected clusters."""
        if not self.scatter_view or not self.scatter_view._scatter_items:
            return
        if self.scatter_view.labels is None:
            return
        if not self._selected_clusters:
            for item in self.scatter_view._scatter_items:
                item.setOpacity(0.6)
        else:
            for i, item in enumerate(self.scatter_view._scatter_items):
                label = int(self.scatter_view.labels[i])
                item.setOpacity(1.0 if label in self._selected_clusters else 0.2)
        self._apply_selected_cluster_styles_to_slide()

    def _apply_selected_cluster_styles_to_slide(self) -> None:
        """Apply opacity styling to slide patches based on selected clusters."""
        if not self.graphics_view or not self.graphics_view.rect_items:
            return
        if self.graphics_view.labels is None:
            return
        if not self._selected_clusters:
            for rect in self.graphics_view.rect_items:
                rect.setOpacity(0.0)
        else:
            selected_opacity = self.graphics_view.highlight_opacity_on
            for i, rect in enumerate(self.graphics_view.rect_items):
                label = int(self.graphics_view.labels[i])
                rect.setOpacity(selected_opacity if label in self._selected_clusters else 0.0)

    def _load_current_data(self, recluster_only: bool = False) -> None:
        """Load or recompute data based on current selections.

        Parameters
        ----------
        recluster_only : bool, optional
            If True, only recompute clusters for existing data (e.g.,
            when the cluster count changes) without reloading images
            or coordinates.  Defaults to False.
        """
        # Reset selected clusters and disable export action when loading new data
        self._clear_selected_clusters()
        self._animation_in_progress = False
        if self.scatter_view:
            self.scatter_view.set_animation_active(False)
        
        if self._model_selection is None:
            print("DEBUG: No model selection; skipping data load")
            return

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

            # Clear custom cluster names on new data load
            self.cluster_legend.clear_cluster_names()

            # Update cluster legend
            self.cluster_legend.update_clusters(labels, colours)
            
            # Compute and store centroids
            self._cluster_centroids = self._compute_centroids(features, labels)
            self._current_labels = labels
            
            # Determine whether to use adaptive or thumbnail mode
            use_adaptive = (
                self.adaptive_zoom_action.isChecked() and 
                data_loader.is_openslide_available()
            )
            
            adaptive_success = False
            if use_adaptive:
                print("DEBUG: Attempting adaptive zoom mode")
                try:
                    adaptive_success = self.graphics_view.load_slide_adaptive(
                        info.image_path,
                        coords_lv0,
                        patch_size_lv0,
                        labels,
                        colours,
                        (slide_w, slide_h)
                    )
                    if adaptive_success:
                        print("DEBUG: Adaptive zoom mode loaded successfully")
                except Exception as e:
                    print(f"DEBUG: Adaptive zoom failed: {e}")
                    adaptive_success = False
            
            if not adaptive_success:
                # Fall back to thumbnail mode
                if use_adaptive:
                    print("DEBUG: Falling back to thumbnail mode")
                else:
                    print("DEBUG: Using thumbnail mode")
                self.graphics_view.load_slide(
                    thumb_image, coords_thumb, 
                    self._current_patch_size_thumb, labels, colours
                )
            
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

            # Clear custom cluster names (cluster IDs change meaning on recluster)
            self.cluster_legend.clear_cluster_names()

            # Update cluster legend
            self.cluster_legend.update_clusters(labels, colours)
            
            # Compute and store centroids
            self._cluster_centroids = self._compute_centroids(features, labels)
            self._current_labels = labels
            
            # Update both views with new labels and colours
            self.graphics_view.update_labels_and_colours(labels, colours)
            self.scatter_view.populate(self._current_embedding, labels, colours)
    
    def _compute_centroids(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute centroid of each cluster in feature space.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vectors (shape: n_patches x feature_dim).
        labels : np.ndarray
            Cluster labels for each patch.
            
        Returns
        -------
        np.ndarray
            Centroids array (shape: n_clusters x feature_dim).
        """
        n_clusters = int(labels.max()) + 1
        centroids = np.zeros((n_clusters, features.shape[1]))
        max_distances = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                cluster_features = features[mask]
                centroids[c] = cluster_features.mean(axis=0)
                # Compute max distance within this cluster
                distances = np.linalg.norm(cluster_features - centroids[c], axis=1)
                max_distances[c] = distances.max()
        # Store max distances for normalization
        self._max_cluster_distances = max_distances
        return centroids
    
    def _get_distance_to_centroid(self, index: int) -> Optional[float]:
        """Get normalized distance from a patch to its cluster centroid.
        
        Returns distance as a percentage (0-100%) where 100% represents
        the furthest point within the same cluster.
        
        Parameters
        ----------
        index : int
            Patch index.
            
        Returns
        -------
        Optional[float]
            Normalized distance as percentage (0-100), or None if data unavailable.
        """
        if (self._current_features is None or 
            self._cluster_centroids is None or
            self._max_cluster_distances is None or
            self._current_labels is None or
            index < 0 or index >= len(self._current_labels)):
            return None
        
        cluster = int(self._current_labels[index])
        feature_vec = self._current_features[index]
        centroid = self._cluster_centroids[cluster]
        distance = np.linalg.norm(feature_vec - centroid)
        
        # Normalize by max distance in cluster
        max_dist = self._max_cluster_distances[cluster]
        if max_dist > 0:
            return (distance / max_dist) * 100.0  # Return as percentage
        return 0.0

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
    
    def _snap_scatter_dock_to_corner(self) -> None:
        """Reposition scatter dock to upper-right corner of slide panel."""
        if not self.scatter_dock.isFloating():
            return  # Don't reposition if docked to edge
        
        # Get slide panel's global position
        slide_rect = self.graphics_view.geometry()
        slide_global = self.graphics_view.mapToGlobal(QPoint(0, 0))
        
        # Position dock in upper-right with margin
        dock_size = self.scatter_dock.size()
        margin = 10
        x = slide_global.x() + slide_rect.width() - dock_size.width() - margin
        y = slide_global.y() + margin
        
        self.scatter_dock.move(x, y)
    
    def _on_scatter_dock_visibility_changed(self, visible: bool) -> None:
        """Handle scatter dock visibility change to reposition when shown."""
        if visible and self.scatter_dock.isFloating():
            # Use QTimer to delay positioning until widget is fully shown
            QTimer.singleShot(0, self._snap_scatter_dock_to_corner)
    
    def resizeEvent(self, event) -> None:
        """Handle window resize to snap scatter dock."""
        super().resizeEvent(event)
        if hasattr(self, 'scatter_dock') and self.scatter_dock.isFloating():
            self._snap_scatter_dock_to_corner()
    
    def moveEvent(self, event) -> None:
        """Handle window move to snap scatter dock."""
        super().moveEvent(event)
        if hasattr(self, 'scatter_dock') and self.scatter_dock.isFloating():
            self._snap_scatter_dock_to_corner()
    
    def showEvent(self, event) -> None:
        """Position scatter dock in upper-right corner on first show."""
        super().showEvent(event)
        if not self._scatter_positioned:
            self._scatter_positioned = True
            self._snap_scatter_dock_to_corner()
    
    def closeEvent(self, event) -> None:
        """Handle application close event and clean up resources."""
        print("DEBUG: Application closing, cleaning up resources")
        # Clean up tile manager in graphics view
        if hasattr(self, 'graphics_view'):
            self.graphics_view.cleanup()
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Local Region Selection Mode
    # -------------------------------------------------------------------------

    def _on_sidebar_tab_changed(self, index: int) -> None:
        """Handle sidebar tab change to switch selection modes."""
        if index == 0:  # K-means tab
            self._set_selection_mode(SelectionMode.KMEANS)
        elif index == 1:  # Local Region tab
            self._set_selection_mode(SelectionMode.LOCAL_REGION)

    def _set_selection_mode(self, mode: SelectionMode) -> None:
        """Switch between K-means and Local Region selection modes."""
        self._selection_mode = mode
        print(f"DEBUG: Selection mode changed to {mode.value}")

        if mode == SelectionMode.LOCAL_REGION:
            # Calculate radius range based on current data
            min_r, max_r, default_r = self._calculate_radius_range()
            self.local_region_widget.set_radius_range(min_r, max_r, default_r)
            self._local_region_radius = float(default_r)

            # Enable local region mode in views
            self.graphics_view.set_local_region_mode(True, self._local_region_radius)

            # Translate radius to scatter space and enable in scatter view
            scatter_radius = self._translate_radius_to_scatter(self._local_region_radius)
            self.scatter_view.set_local_region_mode(True, scatter_radius)

            # Apply local region cluster styles (show user-defined regions)
            self._apply_local_region_cluster_styles()
        else:
            # Disable local region mode
            self.graphics_view.set_local_region_mode(False)
            self.scatter_view.set_local_region_mode(False)

            # Restore K-means cluster styles
            self._apply_selected_cluster_styles()

    def _on_local_region_radius_changed(self, radius: float) -> None:
        """Handle radius slider change."""
        self._local_region_radius = radius
        self.graphics_view.set_local_region_radius(radius)

        # Translate to scatter space
        scatter_radius = self._translate_radius_to_scatter(radius)
        self.scatter_view.set_local_region_radius(scatter_radius)

    def _calculate_radius_range(self) -> Tuple[int, int, int]:
        """Calculate intelligent radius range based on current data.

        Returns
        -------
        Tuple[int, int, int]
            (min_radius, max_radius, default_radius) in scene coordinates.
        """
        patch_size = self.graphics_view.patch_size
        if not patch_size or patch_size == 0:
            return (10, 500, 50)  # Defaults if no data

        # Minimum: 0.5x patch size (select single patch)
        min_radius = int(patch_size * 0.5)

        # Default: 2x patch size (select ~4-9 adjacent patches)
        default_radius = int(patch_size * 2)

        # Maximum: Estimate from coordinate bounds
        if self.graphics_view.coords is not None and len(self.graphics_view.coords) > 0:
            coords = self.graphics_view.coords
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            scene_diagonal = np.sqrt(x_range ** 2 + y_range ** 2)
            max_radius = int(scene_diagonal * 0.25)  # Max 25% of scene diagonal
        else:
            max_radius = int(patch_size * 20)  # Fallback: 20x patch size

        # Ensure sensible values
        min_radius = max(10, min_radius)
        max_radius = max(min_radius + 10, max_radius)
        default_radius = max(min_radius, min(default_radius, max_radius))

        return (min_radius, max_radius, default_radius)

    def _translate_radius_to_scatter(self, slide_radius: float) -> float:
        """Translate radius from slide scene coordinates to scatter coordinates.

        Parameters
        ----------
        slide_radius : float
            Radius in slide scene coordinates.

        Returns
        -------
        float
            Radius in scatter scene coordinates.
        """
        if self.graphics_view.coords is None or len(self.graphics_view.coords) == 0:
            return slide_radius

        # Get extent in slide coordinate space
        coords = self.graphics_view.coords
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        slide_extent = max(x_range, y_range)

        if slide_extent == 0:
            return slide_radius

        # Scatter is normalized to 400x400 with 20px padding = 360 usable
        scatter_extent = 360.0

        # Scale factor
        scale = scatter_extent / slide_extent
        return slide_radius * scale

    def _on_local_region_selected(self, click_point: Tuple[float, float], radius: float) -> None:
        """Handle local region selection from slide view.

        Parameters
        ----------
        click_point : Tuple[float, float]
            Click position in slide scene coordinates.
        radius : float
            Selection radius.
        """
        if self.graphics_view.coords is None or self.graphics_view.labels is None:
            return

        coords = self.graphics_view.coords
        labels = self.graphics_view.labels
        patch_size = self.graphics_view.patch_size

        # Find nearest patch to get K-means cluster
        patch_centers = coords + patch_size / 2.0
        diffs = patch_centers - np.array(click_point)
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        nearest_idx = int(np.argmin(dists_sq))
        kmeans_cluster = int(labels[nearest_idx])

        # Calculate effective radius accounting for cursor clamping
        # The cursor is clamped to 16-128 pixels, so we must match that in selection
        scale = self.graphics_view.transform().m11()
        cursor_diameter = radius * 2 * scale
        if cursor_diameter > 128:
            effective_radius = 64.0 / scale  # Match clamped cursor (128/2 = 64 pixels)
        elif cursor_diameter < 16:
            effective_radius = 8.0 / scale   # Match min cursor (16/2 = 8 pixels)
        else:
            effective_radius = radius

        # Find patches that are both in the K-means cluster AND within effective radius
        distances = np.sqrt(dists_sq)
        in_cluster = labels == kmeans_cluster
        in_radius = distances <= effective_radius
        selected_mask = in_cluster & in_radius
        selected_indices = set(np.where(selected_mask)[0])

        if not selected_indices:
            print("DEBUG: No patches selected (K-means cluster + radius filter)")
            return

        # Create new local region cluster
        self._create_local_region_cluster(
            selected_indices, click_point, radius, kmeans_cluster
        )

    def _on_scatter_local_region_selected(self, click_point: Tuple[float, float], radius: float) -> None:
        """Handle local region selection from scatter view.

        Parameters
        ----------
        click_point : Tuple[float, float]
            Click position in scatter scene coordinates.
        radius : float
            Selection radius in scatter coordinates.
        """
        if self._current_embedding is None or self.graphics_view.labels is None:
            return

        from utils import normalize_to_scene

        labels = self.graphics_view.labels
        scatter_coords = normalize_to_scene(self._current_embedding, 400, 400, 20)

        # Find nearest point to get K-means cluster
        diffs = scatter_coords - np.array(click_point)
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        nearest_idx = int(np.argmin(dists_sq))
        kmeans_cluster = int(labels[nearest_idx])

        # Calculate effective radius accounting for cursor clamping
        # The cursor is clamped to 16-128 pixels, so we must match that in selection
        scale = self.scatter_view.transform().m11()
        cursor_diameter = radius * 2 * scale
        if cursor_diameter > 128:
            effective_radius = 64.0 / scale  # Match clamped cursor (128/2 = 64 pixels)
        elif cursor_diameter < 16:
            effective_radius = 8.0 / scale   # Match min cursor (16/2 = 8 pixels)
        else:
            effective_radius = radius

        # Find points that are both in the K-means cluster AND within effective radius (in scatter space)
        distances = np.sqrt(dists_sq)
        in_cluster = labels == kmeans_cluster
        in_radius = distances <= effective_radius
        selected_mask = in_cluster & in_radius
        selected_indices = set(np.where(selected_mask)[0])

        if not selected_indices:
            print("DEBUG: No patches selected from scatter (K-means cluster + radius filter)")
            return

        # For the local region cluster, we need the slide coordinates
        if self.graphics_view.coords is not None:
            patch_centers = self.graphics_view.coords + self.graphics_view.patch_size / 2.0
            # Use centroid of selected patches in slide space as click point
            selected_coords = patch_centers[list(selected_indices)]
            slide_click_point = (float(selected_coords[:, 0].mean()), float(selected_coords[:, 1].mean()))
            slide_radius = self._local_region_radius  # Use slide radius for storage
        else:
            slide_click_point = click_point
            slide_radius = radius

        self._create_local_region_cluster(
            selected_indices, slide_click_point, slide_radius, kmeans_cluster
        )

    def _compute_local_region_center(self, patch_indices: Set[int]) -> Tuple[float, float]:
        """Compute the center point for a local region based on patches."""
        if not patch_indices or self.graphics_view.coords is None:
            return (0.0, 0.0)
        patch_centers = (
            self.graphics_view.coords[list(patch_indices)]
            + self.graphics_view.patch_size / 2.0
        )
        return (float(patch_centers[:, 0].mean()), float(patch_centers[:, 1].mean()))

    def _regions_are_connected(self, left: Set[int], right: Set[int]) -> bool:
        """Return True if two regions are connected by adjacent patches."""
        if self.graphics_view.coords is None or not left or not right:
            return False

        coords = self.graphics_view.coords
        step = int(round(self.graphics_view.patch_size))
        if step <= 0:
            step = 1

        left_coords = coords[list(left)]
        left_keys = {
            (int(round(x)), int(round(y)))
            for x, y in left_coords
        }

        right_coords = coords[list(right)]
        for x, y in right_coords:
            key = (int(round(x)), int(round(y)))
            if key in left_keys:
                return True
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                if (key[0] + dx, key[1] + dy) in left_keys:
                    return True

        return False

    def _find_connected_local_regions(
        self, patch_indices: Set[int], kmeans_cluster: int
    ) -> List[int]:
        """Find existing region IDs connected to the new selection."""
        connected_ids: List[int] = []
        merged_indices = set(patch_indices)
        changed = True

        while changed:
            changed = False
            for region_id, cluster in list(self._local_region_clusters.items()):
                if cluster.kmeans_cluster != kmeans_cluster:
                    continue
                if region_id in connected_ids:
                    continue
                if self._regions_are_connected(cluster.patch_indices, merged_indices):
                    connected_ids.append(region_id)
                    merged_indices.update(cluster.patch_indices)
                    changed = True

        return connected_ids

    def _create_local_region_cluster(self, patch_indices: Set[int],
                                      center_point: Tuple[float, float],
                                      radius: float, kmeans_cluster: int) -> None:
        """Create a new local region cluster.

        Parameters
        ----------
        patch_indices : Set[int]
            Indices of patches in this region.
        center_point : Tuple[float, float]
            Center point of the selection.
        radius : float
            Radius used for selection.
        kmeans_cluster : int
            The K-means cluster this region belongs to.
        """
        connected_ids = self._find_connected_local_regions(patch_indices, kmeans_cluster)
        if connected_ids:
            primary_id = connected_ids[0]
            merged_indices = set(patch_indices)
            merged_radius = radius

            for region_id in connected_ids:
                cluster = self._local_region_clusters.get(region_id)
                if cluster is None:
                    continue
                merged_indices.update(cluster.patch_indices)
                merged_radius = max(merged_radius, cluster.radius)

            primary_cluster = self._local_region_clusters.get(primary_id)
            if primary_cluster is None:
                return

            primary_cluster.patch_indices = merged_indices
            primary_cluster.center_point = self._compute_local_region_center(merged_indices)
            primary_cluster.radius = merged_radius

            self.local_region_widget.update_region(
                primary_id, len(merged_indices), primary_cluster.name
            )

            # Reapply styles to ensure all regions have correct opacities before cascade
            self._apply_local_region_cluster_styles()

            print(
                f"DEBUG: Merged local regions into {primary_id} with "
                f"{len(merged_indices)} patches"
            )

            # Collect all other region indices before deleting merged ones
            all_other_indices = set()
            for region_id, cluster in self._local_region_clusters.items():
                if region_id not in set(connected_ids) | {primary_id}:
                    all_other_indices.update(cluster.patch_indices)

            new_indices = set(patch_indices)
            for region_id in connected_ids:
                if region_id == primary_id:
                    continue
                cluster = self._local_region_clusters.get(region_id)
                if cluster is None:
                    continue
                new_indices -= cluster.patch_indices

            if new_indices:
                self._trigger_local_region_cascade(
                    primary_id, center_point, animate_indices=new_indices, preserve_indices=all_other_indices
                )
            else:
                self._apply_local_region_cluster_styles()

            # Now delete the merged regions
            for region_id in connected_ids:
                if region_id == primary_id:
                    continue
                if region_id in self._local_region_clusters:
                    del self._local_region_clusters[region_id]
                self.local_region_widget.remove_region(region_id)

            return

        cluster_id = self._next_local_cluster_id
        self._next_local_cluster_id += 1

        # Use K-means cluster color directly (no modification)
        if self.graphics_view.cluster_colors and kmeans_cluster < len(self.graphics_view.cluster_colors):
            color = self.graphics_view.cluster_colors[kmeans_cluster]
        else:
            color = QColor(128, 128, 128)  # Fallback gray

        # Create cluster data
        cluster = LocalRegionCluster(
            cluster_id=cluster_id,
            patch_indices=patch_indices,
            center_point=center_point,
            radius=radius,
            color=color,
            name=f"Region {cluster_id}",
            kmeans_cluster=kmeans_cluster
        )
        self._local_region_clusters[cluster_id] = cluster

        # Update UI
        self.local_region_widget.add_region(
            cluster_id, len(patch_indices), color, cluster.name
        )

        print(f"DEBUG: Created local region cluster {cluster_id} with {len(patch_indices)} patches")

        # Trigger cascade animation for the new cluster
        self._trigger_local_region_cascade(cluster_id, center_point)

    def _trigger_local_region_cascade(
        self,
        cluster_id: int,
        click_point: Tuple[float, float],
        animate_indices: Optional[Set[int]] = None,
        preserve_indices: Optional[Set[int]] = None,
    ) -> None:
        """Trigger cascade animation for a local region cluster.

        Parameters
        ----------
        cluster_id : int
            The local region cluster ID.
        click_point : Tuple[float, float]
            Click point for radial ordering.
        """
        cluster = self._local_region_clusters.get(cluster_id)
        if not cluster:
            return

        indices = list(cluster.patch_indices)
        if not indices:
            return

        if animate_indices is None:
            animate_indices = set(indices)
        else:
            animate_indices = set(animate_indices)

        if not animate_indices:
            self._apply_local_region_cluster_styles()
            return

        # Get coordinates for radial ordering
        patch_centers = self.graphics_view.coords + self.graphics_view.patch_size / 2.0
        animate_list = [idx for idx in indices if idx in animate_indices]
        if not animate_list:
            self._apply_local_region_cluster_styles()
            return
        cluster_coords = patch_centers[animate_list]

        # Order by distance from click point
        order_local = radial_sweep_order(cluster_coords, click_point)
        order_global = [animate_list[i] for i in order_local]

        # Build full set of local-region indices (all regions)
        all_local_indices = set()
        for cid, cl in self._local_region_clusters.items():
            all_local_indices.update(cl.patch_indices)

        # Use provided preserve_indices or fallback to all other regions
        preserve_indices = preserve_indices or set()
        if not preserve_indices:
            for region_id, other_cluster in self._local_region_clusters.items():
                if region_id == cluster_id:
                    continue
                preserve_indices.update(other_cluster.patch_indices)

        # Set patches to cluster color and dim before animation (only animated indices)
        for idx in animate_list:
            rect = self.graphics_view.rect_items[idx]
            rect.setBrush(QBrush(cluster.color))
            rect.setOpacity(self.graphics_view.highlight_opacity_off)

        # Dim only patches not in any local region
        for i, rect in enumerate(self.graphics_view.rect_items):
            if i not in all_local_indices:
                rect.setOpacity(0.0)

        # Prepare scatter view
        self._prepare_scatter_for_local_region(cluster_id)

        # Start animation
        self._animation_in_progress = True
        self.scatter_view.set_animation_active(True)
        self.graphics_view._start_animation(
            cluster.kmeans_cluster, order_global, set()
        )

    def _prepare_scatter_for_local_region(self, cluster_id: int) -> None:
        """Prepare scatter view for local region cascade animation.

        Parameters
        ----------
        cluster_id : int
            The local region cluster ID.
        """
        cluster = self._local_region_clusters.get(cluster_id)
        if not cluster:
            return

        # Dim only the selected cluster points; leave others as-is
        for item in self.scatter_view._scatter_items:
            if item.index in cluster.patch_indices:
                item.setOpacity(0.2)  # Will be animated to 1.0

    def _on_local_region_clicked(self, region_id: int) -> None:
        """Handle click on a local region in the widget list.

        Parameters
        ----------
        region_id : int
            The region ID that was clicked.
        """
        cluster = self._local_region_clusters.get(region_id)
        if not cluster:
            return

        # Highlight this region
        self._apply_local_region_cluster_styles()

        # Highlight selected region patches
        for idx in cluster.patch_indices:
            if 0 <= idx < len(self.graphics_view.rect_items):
                rect = self.graphics_view.rect_items[idx]
                rect.setBrush(QBrush(cluster.color))
                rect.setOpacity(0.8)

        # Highlight in scatter view
        for item in self.scatter_view._scatter_items:
            if item.index in cluster.patch_indices:
                item.setOpacity(1.0)

    def _on_local_region_deleted(self, region_id: int) -> None:
        """Handle deletion of a local region.

        Parameters
        ----------
        region_id : int
            The region ID to delete.
        """
        if region_id in self._local_region_clusters:
            del self._local_region_clusters[region_id]
            self.local_region_widget.remove_region(region_id)
            self._apply_local_region_cluster_styles()
            print(f"DEBUG: Deleted local region cluster {region_id}")

    def _clear_local_region_clusters(self) -> None:
        """Clear all local region clusters."""
        self._local_region_clusters.clear()
        self._next_local_cluster_id = 0
        self.local_region_widget.clear_regions()
        self._apply_local_region_cluster_styles()
        print("DEBUG: Cleared all local region clusters")

    def _apply_local_region_cluster_styles(self) -> None:
        """Apply styling to show local region clusters."""
        if not self.graphics_view.rect_items:
            return

        # Reset all patches to invisible
        for rect in self.graphics_view.rect_items:
            rect.setOpacity(0.0)

        # For each local region cluster, set patch colors and opacities
        for cluster_id, cluster in self._local_region_clusters.items():
            for idx in cluster.patch_indices:
                if 0 <= idx < len(self.graphics_view.rect_items):
                    rect = self.graphics_view.rect_items[idx]
                    rect.setBrush(QBrush(cluster.color))
                    rect.setOpacity(0.6)

        # Apply same to scatter view
        self._apply_local_region_scatter_styles()

    def _apply_local_region_scatter_styles(self) -> None:
        """Apply local region styles to scatter view."""
        if not self.scatter_view._scatter_items:
            return

        # Collect all selected indices
        all_selected = set()
        for cluster in self._local_region_clusters.values():
            all_selected.update(cluster.patch_indices)

        # Set opacities
        for item in self.scatter_view._scatter_items:
            if item.index in all_selected:
                item.setOpacity(1.0)
            else:
                item.setOpacity(0.2)

    def _export_local_region_clusters(self) -> None:
        """Export local region clusters to GeoJSON."""
        if not self._local_region_clusters:
            QMessageBox.warning(self, "No Regions", "No local regions to export.")
            return

        # Similar to _export_all_clusters but for local regions
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Local Regions as GeoJSON",
            "",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return

        self._write_local_regions_geojson(file_path)

    def _write_local_regions_geojson(self, file_path: str) -> None:
        """Write local region clusters to a GeoJSON file.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        """
        # Use level-0 coordinates if available, otherwise thumbnail coordinates
        if self._current_coords_lv0 is not None:
            coords = self._current_coords_lv0
            patch_size = self._current_patch_size_lv0
        else:
            coords = self.graphics_view.coords
            patch_size = self.graphics_view.patch_size

        if coords is None:
            QMessageBox.warning(self, "Export Error", "No coordinate data available.")
            return

        features = []

        for cluster_id, cluster in self._local_region_clusters.items():
            # Create boxes for each patch in the cluster
            boxes = []
            for idx in cluster.patch_indices:
                if 0 <= idx < len(coords):
                    x, y = coords[idx]
                    boxes.append(box(x, y, x + patch_size, y + patch_size))

            if boxes:
                # Merge overlapping boxes
                merged = unary_union(boxes)

                # Get color as hex
                color_hex = cluster.color.name()

                feature = {
                    "type": "Feature",
                    "geometry": merged.__geo_interface__,
                    "properties": {
                        "classification": {
                            "name": cluster.name,
                            "colorRGB": int(color_hex.replace('#', ''), 16)
                        },
                        "region_id": cluster_id,
                        "patch_count": len(cluster.patch_indices),
                        "kmeans_cluster": cluster.kmeans_cluster
                    }
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(file_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"DEBUG: Exported {len(features)} local regions to {file_path}")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {len(features)} local regions to:\n{file_path}"
        )

    # -------------------------------------------------------------------------
    # Cross-Slide Atlas Methods
    # -------------------------------------------------------------------------

    def _on_atlas_add_current(self) -> None:
        """Add the currently loaded slide to the atlas builder."""
        if self._current_features is None:
            QMessageBox.warning(
                self, "No Slide Loaded",
                "Please load a slide first before adding it to the atlas."
            )
            return

        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return

        # Check if already added
        if slide_name in self.atlas_slide_list.get_slide_names():
            QMessageBox.information(
                self, "Already Added",
                f"{slide_name} is already in the atlas."
            )
            return

        # Get the current H5 path for reference
        info = self.slides.get(slide_name)
        h5_path = ""
        if info:
            selected_models = self.model_selector.getSelectedModels()
            mag = self.mag_combo.currentText()
            patch = self.patch_combo.currentText()
            if selected_models and mag and patch:
                h5_path = info.models.get(selected_models[0], {}).get(mag, {}).get(patch, "")

        # Generate a color for this slide based on its index
        slide_idx = self.atlas_slide_list.count()
        hue = (slide_idx * 137) % 360  # Golden angle for good color distribution
        color = QColor.fromHslF(hue / 360.0, 0.7, 0.5)

        if self._current_coords_lv0 is None:
            QMessageBox.warning(
                self, "No Coordinates",
                "Current slide coordinates are unavailable for atlas creation."
            )
            return

        # Add to the list widget
        self.atlas_slide_list.add_slide(
            slide_name,
            self._current_features.copy(),
            self._current_coords_lv0.copy(),
            h5_path,
            color
        )

        print(f"DEBUG: Added {slide_name} to atlas ({len(self._current_features)} patches)")
        self._update_atlas_ui_state()

    def _on_atlas_slide_removed(self, slide_name: str) -> None:
        """Handle removal of a slide from the atlas builder."""
        print(f"DEBUG: Removed {slide_name} from atlas")
        self._update_atlas_ui_state()

    def _update_atlas_ui_state(self) -> None:
        """Update atlas UI controls based on current state."""
        slide_count = self.atlas_slide_list.count()

        # Enable/disable build button (need at least 2 slides)
        self.build_atlas_btn.setEnabled(slide_count >= 2)

        # Enable/disable clear button
        self.clear_atlas_btn.setEnabled(slide_count > 0 or self._cluster_atlas is not None)

        # Update info label
        if slide_count == 0:
            self.atlas_info_label.setText("Add at least 2 slides to build atlas")
        elif slide_count == 1:
            self.atlas_info_label.setText("Add 1 more slide to build atlas")
        else:
            total_patches = sum(
                len(data['features'])
                for data in self.atlas_slide_list.get_all_slide_data().values()
            )
            self.atlas_info_label.setText(
                f"{slide_count} slides, {total_patches:,} total patches"
            )

    def _build_atlas(self) -> None:
        """Build the cross-slide cluster atlas."""
        slide_data = self.atlas_slide_list.get_all_slide_data()

        if len(slide_data) < 2:
            QMessageBox.warning(
                self, "Not Enough Slides",
                "Add at least 2 slides to build an atlas."
            )
            return

        # Show progress
        self.atlas_progress.setVisible(True)
        self.atlas_progress.setValue(0)
        self.build_atlas_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Create builder
            n_clusters = self.atlas_k_spin.value()
            builder = AtlasBuilder(n_clusters=n_clusters, max_patches_per_slide=50000)

            # Add slides to builder
            for slide_name, data in slide_data.items():
                patches_added = builder.add_slide(
                    slide_name,
                    data['features'],
                    data['coords']
                )
                print(f"DEBUG: Atlas builder added {slide_name} with {patches_added} patches")

            self.atlas_progress.setValue(10)
            QApplication.processEvents()

            # Define progress callback
            def update_progress(pct: int) -> None:
                # Scale from 10-90%
                scaled = 10 + int(pct * 0.8)
                self.atlas_progress.setValue(scaled)
                QApplication.processEvents()

            # Build the atlas
            self._cluster_atlas = builder.build(progress_callback=update_progress)

            self.atlas_progress.setValue(95)
            QApplication.processEvents()

            # Populate the atlas scatter view
            self.atlas_scatter_view.populate(self._cluster_atlas)

            # Show the atlas scatter dock
            self.atlas_scatter_dock.show()

            # Position it near the regular scatter dock if visible
            if self.scatter_dock.isVisible():
                scatter_pos = self.scatter_dock.pos()
                self.atlas_scatter_dock.move(scatter_pos.x() + 20, scatter_pos.y() + 20)

            self.atlas_progress.setValue(100)

            # Update info label with atlas statistics
            self.atlas_info_label.setText(
                f"Atlas built: {len(self._cluster_atlas.slide_names)} slides, "
                f"{len(self._cluster_atlas.global_labels):,} patches, "
                f"{self._cluster_atlas.n_clusters} clusters"
            )

            print(f"DEBUG: Atlas built successfully with {self._cluster_atlas.n_clusters} clusters")

        except Exception as e:
            QMessageBox.critical(
                self, "Atlas Build Failed",
                f"Failed to build atlas:\n{str(e)}"
            )
            print(f"DEBUG: Atlas build failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.atlas_progress.setVisible(False)
            self._update_atlas_ui_state()

    def _clear_atlas(self) -> None:
        """Clear the atlas builder and scatter view."""
        # Clear the slide list
        self.atlas_slide_list.clear()

        # Clear the atlas
        self._cluster_atlas = None

        # Clear the scatter view
        self.atlas_scatter_view.scene().clear()

        # Hide the atlas dock
        self.atlas_scatter_dock.hide()

        # Update UI
        self._update_atlas_ui_state()
        print("DEBUG: Atlas cleared")

    def _on_atlas_cluster_selected(self, cluster_id: int) -> None:
        """Handle click on a cluster in the atlas scatter view."""
        if self._cluster_atlas is None:
            return

        print(f"DEBUG: Atlas cluster {cluster_id} selected")

        # Highlight the cluster in the atlas scatter view
        self.atlas_scatter_view.highlight_cluster(cluster_id)

        # Get statistics for this cluster
        total_count = self._cluster_atlas.get_cluster_count(cluster_id)
        per_slide = []
        for slide_name in self._cluster_atlas.slide_names:
            count = self._cluster_atlas.get_slide_cluster_count(slide_name, cluster_id)
            if count > 0:
                per_slide.append(f"{slide_name}: {count:,}")

        # Update info label
        self.atlas_info_label.setText(
            f"Cluster {cluster_id}: {total_count:,} patches\n" +
            "\n".join(per_slide[:5])  # Show top 5 slides
        )

    def _on_atlas_point_hovered(self, global_idx: int, slide_idx: int, entering: bool) -> None:
        """Handle hover on a point in the atlas scatter view."""
        if not entering or self._cluster_atlas is None:
            return

        # Get slide name and cluster info
        if slide_idx < len(self._cluster_atlas.slide_names):
            slide_name = self._cluster_atlas.slide_names[slide_idx]
            cluster_id = int(self._cluster_atlas.global_labels[global_idx])

            # Could show a tooltip or update status bar
            self.statusBar().showMessage(
                f"Slide: {slide_name} | Cluster: {cluster_id}",
                2000
            )
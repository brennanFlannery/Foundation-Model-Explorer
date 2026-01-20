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
from PySide6.QtCore import Qt, QTimer, QPointF, QObject, QPoint, QPropertyAnimation, QEasingCurve
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
        # Store level-0 coordinates for GeoJSON export
        self._current_coords_lv0: Optional[np.ndarray] = None
        self._current_patch_size_lv0: Optional[float] = None
        self._coord_scale_factor: Optional[float] = None
        # Store cluster centroids for distance calculations
        self._cluster_centroids: Optional[np.ndarray] = None
        self._max_cluster_distances: Optional[np.ndarray] = None
        self._current_labels: Optional[np.ndarray] = None
        # Track cascade animation state to prevent hover interference
        self._animation_in_progress: bool = False
        # Local region selection mode state
        self._selection_mode: SelectionMode = SelectionMode.KMEANS
        self._local_region_clusters: Dict[int, LocalRegionCluster] = {}
        self._next_local_cluster_id: int = 0
        self._local_region_radius: float = 50.0

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
        
        # Horizontal layout for legend + info panel + slide view
        content_layout = QHBoxLayout()

        # Left sidebar: tabbed widget with K-means clusters and Local Region selection
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(170)
        self.sidebar_tabs.setMaximumWidth(220)

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

        # Connect tab change to mode switch
        self.sidebar_tabs.currentChanged.connect(self._on_sidebar_tab_changed)

        content_layout.addWidget(self.sidebar_tabs)
        
        # Main slide view (takes remaining width)
        self.graphics_view = SlideGraphicsView()
        self.graphics_view.setMinimumWidth(400)
        # Connect signals: slide click selects cluster and updates scatter
        self.graphics_view.cluster_selected.connect(
            lambda cluster, pos, ctrl: self._update_scatter_for_cluster(cluster, pos, ctrl)
        )
        # Connect slide hover to scatter
        self.graphics_view.patch_hovered.connect(self._on_slide_patch_hovered)
        # Connect slide animation signals for synchronized scatter cascade
        self.graphics_view.patches_highlighted.connect(self._on_patches_highlighted)
        self.graphics_view.animation_completed.connect(self._on_animation_completed)
        # Connect local region selection signal
        self.graphics_view.local_region_selected.connect(self._on_local_region_selected)
        content_layout.addWidget(self.graphics_view, stretch=1)
        
        main_vbox.addLayout(content_layout, stretch=1)
        
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
    
    def _on_adaptive_zoom_toggled(self, checked: bool) -> None:
        """Toggle adaptive zoom mode and reload the current slide."""
        print(f"DEBUG: Adaptive zoom toggled; enabled={checked}")
        self.graphics_view.set_adaptive_mode(checked)
        # Reload the current slide in the new mode
        if self.slide_combo.currentText():
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

    def _trigger_local_region_cascade(self, cluster_id: int, click_point: Tuple[float, float]) -> None:
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

        # Get coordinates for radial ordering
        patch_centers = self.graphics_view.coords + self.graphics_view.patch_size / 2.0
        cluster_coords = patch_centers[indices]

        # Order by distance from click point
        order_local = radial_sweep_order(cluster_coords, click_point)
        order_global = [indices[i] for i in order_local]

        # Set patches to cluster color and dim before animation
        for idx in indices:
            rect = self.graphics_view.rect_items[idx]
            rect.setBrush(QBrush(cluster.color))
            rect.setOpacity(self.graphics_view.highlight_opacity_off)

        # Dim all other patches
        for i, rect in enumerate(self.graphics_view.rect_items):
            if i not in cluster.patch_indices:
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

        # Dim all scatter points
        for item in self.scatter_view._scatter_items:
            if item.index in cluster.patch_indices:
                item.setOpacity(0.2)  # Will be animated to 1.0
            else:
                item.setOpacity(0.1)  # Very dim for non-selected

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
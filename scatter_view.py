# Auto-split from original gui.py on 2025-08-17T19:45:46
"""
scatter_view.py
===============

Scatter plot visualization module for feature embeddings.

This module provides interactive scatter plot views for displaying 2D PCA
embeddings of patch features. Each point in the scatter plot represents
a patch from the whole-slide image, colored according to its cluster
assignment. The view supports zooming, clicking, and hovering interactions
that are synchronized with the slide view.

The scatter plot is automatically normalized to fit within a fixed scene
size (400x400 pixels with padding), and points are rendered as colored
ellipses. The view uses a cross-hair cursor for precise point selection
and supports mouse wheel zooming.

Classes
-------

ScatterGraphicsItem
    Individual scatter point with hover and click interactions.

ScatterGraphicsView
    Main view widget for displaying and interacting with scatter plots.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QPoint, QObject
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter, QCursor
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
from utils import generate_palette, cluster_features, infer_slide_dims, radial_sweep_order, normalize_to_scene
from PySide6.QtCore import Signal



# Shared color helper
def _hsl_to_qcolor(hsl_string: str) -> QColor:
    # Accept strings like 'hsl(210,70%,50%)'
    try:
        values = hsl_string.strip().lower().replace('hsl(', '').rstrip(')').split(',')
        h = float(values[0])
        s = float(values[1].strip(' %')) / 100.0
        l = float(values[2].strip(' %')) / 100.0
        c = QColor()
        c.setHslF(h/360.0, s, l)
        return c
    except Exception:
        return QColor('black')

class ScatterGraphicsItem(QObject, QGraphicsEllipseItem):
    """Custom ellipse item representing a point in the scatter plot.
    
    This class combines QObject and QGraphicsEllipseItem to create an
    interactive scatter point that can emit signals for user interactions.
    Each item represents a single patch from the whole-slide image, positioned
    according to its 2D PCA embedding coordinates and colored according to
    its cluster assignment.
    
    The item responds to hover and click events:
    - Hover: Opacity increases to 1.0 to indicate interactivity
    - Click: Emits a signal with the item's index for cluster selection
    
    Attributes
    ----------
    index : int
        The index of this point, corresponding to the patch index in the
        original feature array.
    _base_color : QColor
        The base color assigned to this point based on its cluster.
    _animation_active : bool
        When True, hover events will not change opacity to avoid
        interfering with cascade animations.
    
    Signals
    -------
    clicked(int)
        Emitted when the item is clicked with the left mouse button.
        The signal carries the item's index.
    hovered(int, bool)
        Emitted when the mouse enters or leaves the item. The first
        parameter is the item's index, the second is True on enter
        and False on leave.
    """
    clicked = Signal(int, bool)
    hovered = Signal(int, bool)  # index and hover state

    def __init__(self, index: int, x: float, y: float, radius: float, color: QColor):
        QGraphicsEllipseItem.__init__(self, -radius, -radius, 2*radius, 2*radius)
        QObject.__init__(self)
        self.index = index
        self.setPos(QPointF(x, y))
        self._base_color = color
        self.setBrush(QBrush(color))
        self.setPen(Qt.NoPen)
        self.setAcceptHoverEvents(True)
        self.setOpacity(0.6)
        # Track cascade animation state to prevent hover interference
        self._animation_active: bool = False

    def set_animation_active(self, active: bool) -> None:
        """Set whether a cascade animation is in progress.
        
        Parameters
        ----------
        active : bool
            True if cascade animation is in progress, False otherwise.
        """
        self._animation_active = active

    def hoverEnterEvent(self, event):
        # Skip opacity change during cascade animation
        if not self._animation_active:
            # Increase opacity to indicate hover
            self.setOpacity(1.0)
        self.hovered.emit(self.index, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        # Skip opacity change during cascade animation
        if not self._animation_active:
            # Restore opacity
            self.setOpacity(0.6)
        self.hovered.emit(self.index, False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
            self.clicked.emit(self.index, ctrl_pressed)
        super().mousePressEvent(event)

class ScatterGraphicsView(QGraphicsView):
    """View for displaying scatter plot of feature embeddings.
    
    This view displays a 2D scatter plot where each point represents a patch
    from the whole-slide image. Points are positioned according to their PCA
    embedding coordinates and colored according to cluster assignments. The
    view supports interactive exploration through clicking, hovering, and zooming.
    
    The scatter plot is automatically scaled to fit within a 400x400 pixel
    scene (with 20-pixel padding). Coordinates are normalized from the PCA
    embedding space to this fixed scene size.
    
    User Interactions
    -----------------
    
    - **Click**: Clicking a scatter point selects its cluster and highlights
      all patches in that cluster in the slide view. The selection triggers
      a radial sweep animation in the scatter plot.
    
    - **Hover**: Hovering over a scatter point highlights the corresponding
      patch in the slide view with a colored overlay.
    
    - **Zoom**: Mouse wheel scrolling zooms in/out around the cursor position.
      Zoom factor is 1.2x per scroll step.
    
    Attributes
    ----------
    _scatter_items : List[ScatterGraphicsItem]
        List of all scatter point graphics items in the scene.
    labels : Optional[np.ndarray]
        Cluster labels for each point (shape: n_patches,).
    cluster_colors : List[QColor]
        Color palette for clusters, converted from HSL strings to QColor.
    
    Signals
    -------
    cluster_selected(int)
        Emitted when a scatter point is clicked. The signal carries the
        cluster number of the clicked point.
    point_hovered(int, bool)
        Emitted when the mouse enters or leaves a scatter point. The first
        parameter is the point index, the second is True on enter and
        False on leave.
    """
    cluster_selected = Signal(int, bool)
    point_hovered = Signal(int, bool)  # index and hover state
    # Signal for local region selection mode
    local_region_selected = Signal(tuple, float)  # (click_point_in_scatter_coords, radius)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the scatter view.

        The default drag mode is set to ``NoDrag`` so that the cross‑hair
        cursor remains visible for precise clicking.  Panning is not
        enabled in this view since it is primarily used for selecting
        points rather than navigating the scene.  If panning is
        desired, it can be implemented similarly to ``SlideGraphicsView``.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        # Enable antialiasing for smooth points
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        # Anchor zooming around the mouse position
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Do not enable automatic hand drag; we will use a cross cursor
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Set a cross‑hair cursor for precise point selection
        self._current_cursor = Qt.CursorShape.CrossCursor
        self._override_pushed: bool = False
        self._set_cursor(Qt.CursorShape.CrossCursor)
        self._scatter_items: List[ScatterGraphicsItem] = []
        self.labels: Optional[np.ndarray] = None
        self.cluster_colors: List[QColor] = []
        # Track cascade animation state to prevent hover interference
        self._animation_active: bool = False
        # Variable to track middle-click panning state
        self._was_panning = False
        self._suppress_clicks = False
        # Local region selection mode state
        self._local_region_mode: bool = False
        self._local_region_radius: float = 50.0

    def _set_cursor(self, cursor) -> None:
        """Set cursor on viewport (correct for QAbstractScrollArea) and track it."""
        self._current_cursor = cursor
        self.viewport().setCursor(cursor)
        if self._override_pushed:
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(cursor)

    def enterEvent(self, event) -> None:
        """Re-assert cursor on viewport when mouse enters — recovers after macOS fullscreen."""
        super().enterEvent(event)
        # Clear any stale application-level override cursors from fullscreen transitions
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self._override_pushed = False
        self.viewport().unsetCursor()
        self.viewport().setCursor(self._current_cursor)
        # Push override cursor — maps to [NSCursor push] on macOS, which survives
        # NSTrackingArea resets unlike widget-level setCursor().
        QApplication.setOverrideCursor(self._current_cursor)
        self._override_pushed = True

    def leaveEvent(self, event) -> None:
        """Pop override cursor when mouse leaves so it doesn't bleed into other widgets."""
        super().leaveEvent(event)
        if self._override_pushed:
            QApplication.restoreOverrideCursor()
            self._override_pushed = False

    def set_animation_active(self, active: bool) -> None:
        """Set whether a cascade animation is in progress."""
        self._animation_active = active
        # Propagate to all scatter items
        for item in self._scatter_items:
            item.set_animation_active(active)

    def populate(self, coords_2d: np.ndarray, labels: np.ndarray, colors: List[str]) -> None:
        """Populate the scatter scene with points."""
        scene = self.scene()
        scene.clear()
        self._scatter_items.clear()

        if coords_2d is None or labels is None or len(coords_2d) == 0:
            return

        self.labels = labels
        self.cluster_colors = [
            _hsl_to_qcolor(color) if isinstance(color, str) else color for color in colors
        ]

        # Normalize coords into scene size
        coords = normalize_to_scene(coords_2d, width=400, height=400, padding=20)

        for idx, (x, y) in enumerate(coords):
            cluster = int(labels[idx])
            color = self.cluster_colors[cluster] if cluster < len(self.cluster_colors) else QColor("red")
            item = ScatterGraphicsItem(idx, x, y, radius=2.5, color=color)
            item.clicked.connect(self._on_item_clicked)
            item.hovered.connect(self._on_item_hovered)
            scene.addItem(item)
            self._scatter_items.append(item)

        padding = 20
        scene.setSceneRect(0, 0, 400 + 2 * padding, 400 + 2 * padding)
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def set_point_opacity(self, index: int, on: bool) -> None:
        """Manually adjust opacity of a scatter point."""
        if 0 <= index < len(self._scatter_items):
            target_opacity = 1.0 if on else 0.6
            self._scatter_items[index].setOpacity(target_opacity)

    def _on_item_clicked(self, idx: int, ctrl: bool) -> None:
        """Handle scatter point click and emit cluster selection."""
        if self._suppress_clicks:
            return
        if self.labels is None or idx < 0 or idx >= len(self.labels):
            return
        cluster = int(self.labels[idx])
        self.cluster_selected.emit(cluster, ctrl)

    def _on_item_hovered(self, idx: int, state: bool) -> None:
        """Handle scatter point hover and emit point hover signal."""
        self.point_hovered.emit(idx, state)

    def _forward_drag_event(self, event: QMouseEvent, button: Qt.MouseButton) -> None:
        """Forward a synthetic mouse event to Qt for native drag handling."""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent
        synthetic = QMouseEvent(
            event.type(),
            event.position(),
            event.globalPosition(),
            button,
            button | event.buttons(),
            event.modifiers(),
        )
        if event.type() == QEvent.Type.MouseButtonPress:
            super().mousePressEvent(synthetic)
        elif event.type() == QEvent.Type.MouseMove:
            super().mouseMoveEvent(synthetic)
        elif event.type() == QEvent.Type.MouseButtonRelease:
            super().mouseReleaseEvent(synthetic)

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for hover only; panning is handled by Qt natively."""
        # Ignore move events during Qt-native panning
        if self._was_panning and (event.buttons() & Qt.MouseButton.MiddleButton):
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            return  # Let Qt handle panning natively
        
        # If not panning, call base class for hover detection
        super().mouseMoveEvent(event)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for native panning with middle button."""
        if event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Scatter view middle mouse button pressed for native panning")
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._set_cursor(Qt.CursorShape.OpenHandCursor)
            self._was_panning = True
            self._suppress_clicks = True
            # Forward as left button press to trigger Qt's native drag
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            return

        # Handle local region mode clicks
        if self._local_region_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handle_local_region_click(scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Restore state after Qt-native panning or forward to base class."""
        if self._was_panning and event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Scatter view middle mouse button released; ending native panning")
            # Forward as left button release to complete Qt's native drag
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._set_cursor(Qt.CursorShape.CrossCursor)
            self._was_panning = False
            self._suppress_clicks = False
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """Zoom in/out with mouse wheel."""
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

        # Update local region cursor if in local region mode
        if self._local_region_mode:
            self._update_radius_cursor()

        event.accept()

    # -------------------------------------------------------------------------
    # Local Region Selection Mode
    # -------------------------------------------------------------------------

    def set_local_region_mode(self, enabled: bool, radius: float = 50.0) -> None:
        """Enable or disable local region selection mode.

        Parameters
        ----------
        enabled : bool
            Whether to enable local region mode.
        radius : float
            Initial selection radius in scatter scene coordinates.
        """
        self._local_region_mode = enabled
        self._local_region_radius = radius
        if enabled:
            self._update_radius_cursor()
        else:
            self._set_cursor(Qt.CursorShape.CrossCursor)

    def set_local_region_radius(self, radius: float) -> None:
        """Update the selection radius.

        Parameters
        ----------
        radius : float
            New selection radius in scatter scene coordinates.
        """
        self._local_region_radius = radius
        if self._local_region_mode:
            self._update_radius_cursor()

    def _update_radius_cursor(self) -> None:
        """Create a circle cursor matching the current radius and zoom level."""
        # Get current zoom scale
        scale = self.transform().m11()

        # Calculate cursor diameter in screen pixels
        cursor_diameter = int(self._local_region_radius * 2 * scale)

        # Clamp to reasonable cursor sizes (16-128 pixels)
        cursor_diameter = max(16, min(cursor_diameter, 128))

        # Create pixmap for cursor
        pixmap = QPixmap(cursor_diameter, cursor_diameter)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw circle outline
        pen = QPen(QColor(255, 100, 100, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 100, 100, 30)))

        margin = 2
        painter.drawEllipse(margin, margin,
                            cursor_diameter - 2 * margin,
                            cursor_diameter - 2 * margin)

        # Draw center crosshair
        center = cursor_diameter // 2
        crosshair_size = min(5, cursor_diameter // 6)
        painter.drawLine(center - crosshair_size, center, center + crosshair_size, center)
        painter.drawLine(center, center - crosshair_size, center, center + crosshair_size)

        painter.end()

        # Create cursor with hotspot at center
        cursor = QCursor(pixmap, cursor_diameter // 2, cursor_diameter // 2)
        self._set_cursor(cursor)

    def _handle_local_region_click(self, scene_pos: QPointF) -> None:
        """Handle a click in local region mode.

        Parameters
        ----------
        scene_pos : QPointF
            Click position in scene coordinates.
        """
        x, y = scene_pos.x(), scene_pos.y()
        print(f"DEBUG: Scatter local region click at ({x}, {y}) with radius {self._local_region_radius}")
        self.local_region_selected.emit((x, y), self._local_region_radius)


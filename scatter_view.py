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
    clicked = Signal(int)
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

    def hoverEnterEvent(self, event):
        # Increase opacity to indicate hover
        self.setOpacity(1.0)
        self.hovered.emit(self.index, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        # Restore opacity
        self.setOpacity(0.6)
        self.hovered.emit(self.index, False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
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
    cluster_selected = Signal(int)
    point_hovered = Signal(int, bool)  # index and hover state

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
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Do not enable automatic hand drag; we will use a cross cursor
        self.setDragMode(QGraphicsView.NoDrag)
        # Set a cross‑hair cursor for precise point selection
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._scatter_items: List[ScatterGraphicsItem] = []
        self.labels: Optional[np.ndarray] = None
        self.cluster_colors: List[QColor] = []

    def populate(self, coords_2d: np.ndarray, labels: np.ndarray, colors: List[str]) -> None:
        """Populate the scatter scene with points.

        Parameters
        ----------
        coords_2d : np.ndarray
            2D coordinates from PCA transformation.
        labels : np.ndarray
            Cluster labels for each point.
        colors : List[str]
            Colour palette for clusters.
        """
        # Clear any existing items before populating new scatter points
        scene = self.scene()
        scene.clear()
        self._scatter_items = []
        # Store labels and convert provided colours to QColours
        self.labels = labels.astype(int)
        self.cluster_colors = [_hsl_to_qcolor(c) for c in colors]
        # Debug: log basic statistics about the embedding
        print(f"DEBUG: Populating scatter with {coords_2d.shape[0]} points")
        # Normalize coordinates to view range
        if coords_2d.size == 0:
            return
        xs = coords_2d[:, 0]
        ys = coords_2d[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        # Add margin
        padding = 20
        width = max_x - min_x
        height = max_y - min_y
        for i, (x, y) in enumerate(coords_2d):
            # Map to scene coordinates
            sx = (x - min_x) / (width + 1e-9) * 400 + padding
            sy = (y - min_y) / (height + 1e-9) * 400 + padding
            lbl = int(labels[i])
            col = self.cluster_colors[lbl] if lbl < len(self.cluster_colors) else QColor('black')
            # Create a scatter item for this point and connect signals
            item = ScatterGraphicsItem(i, sx, sy, radius=4.0, color=col)
            item.clicked.connect(self._on_item_clicked)
            item.hovered.connect(self._on_item_hovered)
            scene.addItem(item)
            self._scatter_items.append(item)
        scene.setSceneRect(0, 0, 400 + 2*padding, 400 + 2*padding)
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def _on_item_clicked(self, index: int) -> None:
        """Emit signal with the cluster of the clicked point."""
        print(f"DEBUG: Scatter point clicked at index {index}")
        if self.labels is None:
            return
        cluster = int(self.labels[index])
        self.cluster_selected.emit(cluster)

    def _on_item_hovered(self, index: int, state: bool) -> None:
        """Forward hover state to external listeners."""
        print(f"DEBUG: Scatter point hover state changed for index {index}, state={state}")
        self.point_hovered.emit(index, state)

    def set_point_opacity(self, index: int, on: bool) -> None:
        """Manually adjust opacity of a scatter point."""
        if 0 <= index < len(self._scatter_items):
            target_opacity = 1.0 if on else 0.6
            print(f"DEBUG: Setting scatter point {index} opacity to {target_opacity}")
            self._scatter_items[index].setOpacity(target_opacity)

    def wheelEvent(self, event) -> None:
        """Zoom in/out with mouse wheel."""
        # Determine zoom factor based on scroll direction
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        print(f"DEBUG: Scatter view zoom with factor {factor}")
        self.scale(factor, factor)
        event.accept()


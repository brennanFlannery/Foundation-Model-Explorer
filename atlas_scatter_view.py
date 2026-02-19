"""
atlas_scatter_view.py
=====================

Scatter plot visualization for cross-slide cluster atlas.

This module provides the AtlasScatterView widget for displaying combined
PCA embeddings from multiple slides. Points are colored by global cluster
assignment and can be visually distinguished by slide through opacity and
hover interactions.

Classes
-------
AtlasScatterItem
    Individual scatter point representing a patch in the atlas.

AtlasScatterView
    Main view widget for the atlas scatter plot.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QObject
from PySide6.QtGui import QColor, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
)

if TYPE_CHECKING:
    from atlas_builder import ClusterAtlas


class AtlasScatterItem(QObject, QGraphicsEllipseItem):
    """Scatter point for atlas view with slide identification.

    Each item represents a single patch from one of the slides in the atlas.
    Points are colored by their global cluster assignment and can be
    distinguished by slide through opacity variations.

    Attributes
    ----------
    index : int
        Global index in the atlas (position in global_features array).
    slide_idx : int
        Index of the slide this patch belongs to.
    cluster_id : int
        Global cluster assignment.

    Signals
    -------
    clicked(int, int)
        Emitted when clicked: (global_index, slide_idx)
    hovered(int, int, bool)
        Emitted on hover: (global_index, slide_idx, entering)
    """
    clicked = Signal(int, int)
    hovered = Signal(int, int, bool)

    def __init__(self, index: int, slide_idx: int, cluster_id: int,
                 x: float, y: float, radius: float, color: QColor):
        QGraphicsEllipseItem.__init__(self, -radius, -radius, 2 * radius, 2 * radius)
        QObject.__init__(self)
        self.index = index
        self.slide_idx = slide_idx
        self.cluster_id = cluster_id
        self._base_color = color
        self._base_radius = radius

        self.setPos(QPointF(x, y))
        self.setBrush(QBrush(color))
        self.setPen(Qt.NoPen)
        self.setAcceptHoverEvents(True)

        # Different base opacity per slide for visual distinction
        # Cycles through 3 opacity levels for up to 9 slides
        base_opacity = 0.4 + (slide_idx % 3) * 0.15
        self.setOpacity(base_opacity)
        self._base_opacity = base_opacity

    def hoverEnterEvent(self, event):
        self.setOpacity(1.0)
        self.setZValue(10)
        self.hovered.emit(self.index, self.slide_idx, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setOpacity(self._base_opacity)
        self.setZValue(0)
        self.hovered.emit(self.index, self.slide_idx, False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index, self.slide_idx)
        super().mousePressEvent(event)


class AtlasScatterView(QGraphicsView):
    """Scatter view showing patches from multiple slides in the atlas.

    This view displays a 2D PCA projection of all patches across the atlas,
    colored by their global cluster assignment. Users can click to select
    clusters or hover to highlight individual points.

    Signals
    -------
    cluster_selected(int)
        Emitted when a point is clicked, carrying the cluster ID.
    point_hovered(int, int, bool)
        Emitted on hover: (global_index, slide_idx, entering)
    slide_highlight_requested(int)
        Emitted when user wants to highlight a specific slide.
    """
    cluster_selected = Signal(int)
    point_hovered = Signal(int, int, bool)
    slide_highlight_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._current_cursor = Qt.CursorShape.CrossCursor
        self._override_pushed: bool = False
        self._set_cursor(Qt.CursorShape.CrossCursor)

        self._items: List[AtlasScatterItem] = []
        self._atlas: Optional[ClusterAtlas] = None
        self._highlighted_slide: Optional[int] = None
        self._highlighted_cluster: Optional[int] = None

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

    def populate(self, atlas: 'ClusterAtlas') -> None:
        """Populate the scatter view with atlas data.

        Parameters
        ----------
        atlas : ClusterAtlas
            The cluster atlas to display.
        """
        scene = self.scene()
        scene.clear()
        self._items.clear()
        self._atlas = atlas

        if atlas is None or len(atlas.global_pca_coords) == 0:
            return

        # Normalize coordinates to scene
        coords = self._normalize_coords(atlas.global_pca_coords)

        print(f"DEBUG: Populating atlas scatter with {len(coords)} points")

        for i, (x, y) in enumerate(coords):
            cluster_id = int(atlas.global_labels[i])
            slide_idx = int(atlas.slide_indices[i])
            color = atlas.cluster_colors[cluster_id] if cluster_id < len(atlas.cluster_colors) else QColor("gray")

            item = AtlasScatterItem(i, slide_idx, cluster_id, x, y, radius=2.0, color=color)
            item.clicked.connect(self._on_item_clicked)
            item.hovered.connect(self._on_item_hovered)
            scene.addItem(item)
            self._items.append(item)

        scene.setSceneRect(0, 0, 440, 440)
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates to fit within scene bounds."""
        if len(coords) == 0:
            return coords

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # Prevent division by zero

        # Map to [20, 420] range (400x400 with 20px padding)
        normalized = (coords - mins) / ranges * 400 + 20
        return normalized

    def _on_item_clicked(self, global_idx: int, slide_idx: int) -> None:
        """Handle click on a scatter point."""
        if self._atlas is None:
            return
        cluster_id = int(self._atlas.global_labels[global_idx])
        self.cluster_selected.emit(cluster_id)

    def _on_item_hovered(self, global_idx: int, slide_idx: int, entering: bool) -> None:
        """Handle hover on a scatter point."""
        self.point_hovered.emit(global_idx, slide_idx, entering)

    def highlight_slide(self, slide_idx: int) -> None:
        """Highlight points from a specific slide.

        Parameters
        ----------
        slide_idx : int
            Index of the slide to highlight. Pass -1 to clear highlight.
        """
        self._highlighted_slide = slide_idx if slide_idx >= 0 else None

        for item in self._items:
            if slide_idx < 0:
                # Reset all to base opacity
                item.setOpacity(item._base_opacity)
                item.setZValue(0)
            elif item.slide_idx == slide_idx:
                item.setOpacity(1.0)
                item.setZValue(1)
            else:
                item.setOpacity(0.15)
                item.setZValue(0)

    def highlight_cluster(self, cluster_id: int) -> None:
        """Highlight points from a specific cluster.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster to highlight. Pass -1 to clear highlight.
        """
        self._highlighted_cluster = cluster_id if cluster_id >= 0 else None

        for item in self._items:
            if cluster_id < 0:
                # Reset all to base opacity
                item.setOpacity(item._base_opacity)
                item.setZValue(0)
            elif item.cluster_id == cluster_id:
                item.setOpacity(1.0)
                item.setZValue(1)
            else:
                item.setOpacity(0.15)
                item.setZValue(0)

    def reset_highlight(self) -> None:
        """Reset all points to their default opacity."""
        self._highlighted_slide = None
        self._highlighted_cluster = None
        for item in self._items:
            item.setOpacity(item._base_opacity)
            item.setZValue(0)

    def get_slide_points(self, slide_idx: int) -> List[int]:
        """Get global indices of all points from a specific slide.

        Parameters
        ----------
        slide_idx : int
            Index of the slide.

        Returns
        -------
        List[int]
            List of global indices.
        """
        return [item.index for item in self._items if item.slide_idx == slide_idx]

    def get_cluster_points(self, cluster_id: int) -> List[int]:
        """Get global indices of all points in a specific cluster.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster.

        Returns
        -------
        List[int]
            List of global indices.
        """
        return [item.index for item in self._items if item.cluster_id == cluster_id]

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zooming."""
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for panning with middle button."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._set_cursor(Qt.CursorShape.ClosedHandCursor)
            # Create a synthetic left-button event for Qt's drag handling
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtCore import QEvent
            synthetic = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(synthetic)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release to end panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._set_cursor(Qt.CursorShape.CrossCursor)
            # Create synthetic release event
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtCore import QEvent
            synthetic = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                event.modifiers()
            )
            super().mouseReleaseEvent(synthetic)
            return
        super().mouseReleaseEvent(event)

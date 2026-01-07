# Auto-split from original gui.py on 2025-08-17T19:45:46
"""
slide_view.py
=============

Whole-slide image visualization module with patch overlays.

This module provides an interactive view for displaying whole-slide image
thumbnails with color-coded patch overlays representing cluster assignments.
Each patch is rendered as a semi-transparent colored rectangle overlaid on
the slide image. The view supports clicking, hovering, panning, and zooming
interactions.

The view implements an animated radial sweep effect when clusters are selected:
patches are highlighted in order of increasing distance from the click point,
creating a visual cascade effect. This helps users understand spatial
relationships within clusters.

Classes
-------

SlideGraphicsView
    Main view widget for displaying whole-slide images with patch overlays.
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

from scatter_view import _hsl_to_qcolor
class SlideGraphicsView(QGraphicsView):
    """View for displaying whole-slide images with interactive patch overlays.
    
    This view displays a thumbnail of a whole-slide image with colored
    rectangular overlays representing patches. Each patch is colored according
    to its cluster assignment, and patches can be highlighted through user
    interactions. The view supports panning, zooming, clicking, and hovering.
    
    Animation System
    ----------------
    
    When a cluster is selected (via click), patches in that cluster are
    highlighted with an animated radial sweep effect. Patches are revealed
    in order of increasing distance from the click point, creating a
    cascading visual effect. The animation is controlled by a QTimer that
    updates multiple patches per step for smooth performance.
    
    User Interactions
    -----------------
    
    - **Left Click**: Clicking on a patch selects its cluster and triggers
      a radial sweep animation highlighting all patches in that cluster.
      The click position is used as the center for the radial ordering.
    
    - **Hover**: Hovering over a patch highlights it with a lighter color
      overlay and emits a signal to highlight the corresponding scatter point.
    
    - **Middle Click + Drag**: Temporarily enables panning mode. The cursor
      changes to an open hand, and dragging pans the view. Releasing the
      button restores the cross-hair cursor.
    
    - **Mouse Wheel**: Zooms in/out around the cursor position. Zoom factor
      is 1.25x per scroll step.
    
    Attributes
    ----------
    pixmap_item : Optional[QGraphicsPixmapItem]
        The graphics item displaying the slide thumbnail image.
    rect_items : List[QGraphicsRectItem]
        List of rectangular overlay items, one per patch. Each rectangle
        is colored according to its cluster assignment.
    coords : Optional[np.ndarray]
        Patch coordinates in thumbnail space (shape: n_patches x 2).
    labels : Optional[np.ndarray]
        Cluster labels for each patch (shape: n_patches,).
    patch_size : float
        Size of each patch in thumbnail coordinate space.
    cluster_colors : List[QColor]
        Color palette for clusters, converted from HSL strings.
    animation_interval : int
        Milliseconds between animation updates (default: 10).
    updates_per_step : int
        Number of patches to update per animation step (default: 3).
    highlight_opacity_on : float
        Opacity for highlighted patches (default: 0.6).
    highlight_opacity_off : float
        Opacity for non-highlighted patches in selected cluster (default: 0.05).
    
    Signals
    -------
    cluster_selected(int, tuple)
        Emitted when a patch is clicked. The first parameter is the cluster
        number, the second is the (x, y) click position in scene coordinates.
    patch_hovered(int, bool)
        Emitted when the mouse enters or leaves a patch. The first parameter
        is the patch index, the second is True on enter and False on leave.
    patches_highlighted(list)
        Emitted during cascade animation with the list of patch indices
        highlighted in the current animation step. Used for synchronization
        with the scatter view.
    animation_completed()
        Emitted when the cascade animation finishes. Used to apply persistent
        styling after the animation completes.
    """
    # Signals for cluster selection and hover
    cluster_selected = Signal(int, tuple)  # cluster number and click position
    patch_hovered = Signal(int, bool)  # patch index and hover state
    # Signals for animation synchronization
    patches_highlighted = Signal(list)  # list of patch indices highlighted this step
    animation_completed = Signal()  # emitted when cascade animation finishes
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the slide view.

        A cross‑hair cursor is used for precision clicking.  Panning
        with the left mouse button is disabled; instead, the middle
        mouse button activates panning (``ScrollHandDrag``) temporarily.
        Debug prints are included to aid in diagnosing unexpected
        behaviour.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        # Track mouse movements for hover events
        self.setMouseTracking(True)
        # Set cross‑hair cursor by default
        self.setCursor(Qt.CursorShape.CrossCursor)
        # Use NoDrag by default; panning will be enabled on middle click
        self.setDragMode(QGraphicsView.NoDrag)
        # Data structures to track slide patches
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.rect_items: List[QGraphicsRectItem] = []
        self.coords: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.patch_size: float = 0.0
        self.cluster_colors: List[QColor] = []
        # Animation state
        self._animation_order: List[int] = []
        self._animation_index: int = 0
        self.highlight_opacity_on = 0.6
        self.highlight_opacity_off = 0.05
        # Animation settings
        self.animation_interval = 10  # milliseconds between updates
        self.updates_per_step = 3    # number of patches per step
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_step)
        # For zoom support
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Variables to manage middle‑click panning
        self._was_panning = False
        self._orig_drag_mode = QGraphicsView.NoDrag
        self._orig_cursor: Optional[QCursor] = None


    def load_slide(self, image: Image.Image, coords: np.ndarray, patch_size: float,
                   labels: np.ndarray, cluster_colors: List[str]) -> None:
        """Load a new slide and associated patch data into the view.

        Parameters
        ----------
        image : PIL.Image.Image
            Thumbnail image to display.
        coords : np.ndarray
            Scaled coordinates of patch top‑left corners (shape `(n_patches, 2)`).
        patch_size : float
            Size of each patch in the thumbnail coordinate space.
        labels : np.ndarray
            Cluster labels for each patch.
        cluster_colors : List[str]
            Colour palette for clusters; each entry is a CSS colour string.
        """
        # Stop any ongoing animation and reset animation state
        self._timer.stop()
        self._animation_order = []
        self._animation_index = 0
        # Clear existing scene
        scene = self.scene()
        scene.clear()
        self.rect_items.clear()
        # Create pixmap from PIL image
        print(f"DEBUG: Loading slide thumbnail of size {image.size}")
        qim = Image.fromarray(np.array(image)) if not isinstance(image, Image.Image) else image
        im_data = qim.convert('RGB').tobytes('raw', 'RGB')
        qimage = QImage(im_data, qim.width, qim.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item = scene.addPixmap(pixmap)
        # Store data
        self.coords = coords
        self.labels = labels.astype(int)
        self.patch_size = float(patch_size)
        # Convert cluster colours to QColor
        self.cluster_colors = [_hsl_to_qcolor(col) for col in cluster_colors]
        # Create rect items for each patch
        for idx, (x, y) in enumerate(coords):
            label = int(labels[idx])
            color = self.cluster_colors[label] if label < len(self.cluster_colors) else QColor('red')
            rect = QGraphicsRectItem(x, y, self.patch_size, self.patch_size)
            # Use brush with full alpha; control opacity via item
            rect.setBrush(color)
            rect.setPen(Qt.NoPen)
            # Initial opacity is zero; will be increased during cascade animation
            rect.setOpacity(0.0)
            rect.setFlag(QGraphicsItem.ItemIsSelectable, False)
            scene.addItem(rect)
            self.rect_items.append(rect)
        # Resize scene rect to fit content
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def update_labels_and_colours(self, labels: np.ndarray, cluster_colors: List[str]) -> None:
        """Update cluster labels and colours for existing patch items.

        Parameters
        ----------
        labels : np.ndarray
            New cluster labels matching the number of rect items.
        cluster_colors : List[str]
            New palette; length should be >= max(label)+1.
        """
        if self.coords is None or len(self.rect_items) == 0:
            return
        # Update stored labels and cluster colours
        self.labels = labels.astype(int)
        self.cluster_colors = [_hsl_to_qcolor(col) for col in cluster_colors]
        # Update brush colours and reset opacity
        for idx, rect in enumerate(self.rect_items):
            lbl = int(self.labels[idx])
            color = self.cluster_colors[lbl] if lbl < len(self.cluster_colors) else QColor('red')
            rect.setBrush(color)
            # Reset opacity: start from invisible state for new clusters
            rect.setOpacity(0.0)

    def _highlight_cluster(self, cluster: int) -> None:
        """Highlight all patches belonging to a specific cluster.
        
        Parameters
        ----------
        cluster : int
            Cluster label to highlight
        """
        if self.labels is None:
            return
        # Find all patches in this cluster
        cluster_indices = np.where(self.labels == cluster)[0]
        if cluster_indices.size == 0:
            return
            
        # Get the coordinates of the currently selected patch
        if hasattr(self, '_last_click_pos'):
            x, y = self._last_click_pos
            # Compute radial order within cluster and begin animation
            cluster_coords = self.coords[cluster_indices]
            order_local = radial_sweep_order(cluster_coords, (x, y))
            order_global = cluster_indices[order_local]
            print(f"DEBUG: Highlighting cluster {cluster} with {len(order_global)} patches")
            # Start the radial sweep animation
            self._start_animation(cluster, order_global.tolist())

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events for highlighting or panning.

        * Left button: highlight the cluster under the cursor.
        * Middle button: enable panning by temporarily switching to
          ``ScrollHandDrag`` mode.  The original drag mode and cursor
          are restored on release.
        """
        print("DEBUG: SlideGraphicsView.mousePressEvent invoked")
        # Middle button initiates panning
        if event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Middle mouse button pressed for panning")
            # Save current drag mode and cursor
            self._orig_drag_mode = self.dragMode()
            self._orig_cursor = self.cursor()
            # Enable ScrollHandDrag and change cursor to open hand
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._was_panning = True
            # Delegate to base class to start the drag
            super().mousePressEvent(event)
            return

        # Only proceed with highlighting on left button
        if event.button() != Qt.MouseButton.LeftButton:
            print("DEBUG: Non-left and non-middle button pressed; passing to base")
            super().mousePressEvent(event)
            return

        # Ensure required data is loaded
        if self.pixmap_item is None or self.coords is None or self.labels is None:
            print("DEBUG: Missing required data for highlighting:", {
                "pixmap_item": self.pixmap_item is None,
                "coords": self.coords is None,
                "labels": self.labels is None
            })
            return

        # Map the click position to the scene
        scene_pos = self.mapToScene(event.position().toPoint())
        print(f"DEBUG: Click position mapped to scene: ({scene_pos.x()}, {scene_pos.y()})")
        # Ignore clicks outside the pixmap
        if not self.pixmap_item.contains(scene_pos):
            print("DEBUG: Click outside image bounds; ignoring")
            return

        x = scene_pos.x()
        y = scene_pos.y()
        # Compute nearest patch centre
        patch_centres = self.coords + self.patch_size / 2.0
        diffs = patch_centres - np.array([x, y])
        dists = np.einsum('ij,ij->i', diffs, diffs)
        idx = int(np.argmin(dists))
        cluster = int(self.labels[idx])
        print(f"DEBUG: Nearest patch index={idx}, cluster={cluster}")
        # Record last click position for radial ordering
        self._last_click_pos = (x, y)
        # Emit signal to main window (cluster number and click position)
        self.cluster_selected.emit(cluster, (x, y))
        # Compute animation order within the cluster and start animation
        cluster_indices = np.where(self.labels == cluster)[0]
        if cluster_indices.size > 0:
            cluster_coords = self.coords[cluster_indices]
            order_local = radial_sweep_order(cluster_coords, (x, y))
            order_global = cluster_indices[order_local]
            self._start_animation(cluster, order_global.tolist())

    def _start_animation(self, cluster: int, order: List[int]) -> None:
        """Reset opacities and begin the sweep animation for a cluster."""
        print(f"DEBUG: _start_animation called with cluster {cluster}")
        # Stop any previous animation
        self._timer.stop()
        self._animation_index = 0
        self._animation_order = order
        # Reset all opacities: cluster patches to low, others to 0
        for i, rect in enumerate(self.rect_items):
            if int(self.labels[i]) == cluster:
                # Slight opacity indicates pending highlight
                rect.setOpacity(self.highlight_opacity_off)
            else:
                rect.setOpacity(0.0)
        print("DEBUG: Starting animation timer with interval", self.animation_interval)
        self._timer.start(self.animation_interval)

    def _animate_step(self) -> None:
        """Animation step: increase opacity of multiple patches in the order."""
        # If animation has finished, stop the timer and emit completion signal
        if self._animation_index >= len(self._animation_order):
            print("DEBUG: Slide animation complete; stopping timer")
            self._timer.stop()
            self.animation_completed.emit()
            return

        # Update multiple patches per step
        end_idx = min(self._animation_index + self.updates_per_step,
                      len(self._animation_order))
        indices_this_step = self._animation_order[self._animation_index:end_idx]
        print(f"DEBUG: Slide animation step updating indices {self._animation_index} to {end_idx-1}")
        for idx in indices_this_step:
            rect = self.rect_items[idx]
            rect.setOpacity(self.highlight_opacity_on)

        self._animation_index = end_idx
        # Emit signal for synchronization with scatter view
        self.patches_highlighted.emit(list(indices_this_step))

    def mouseReleaseEvent(self, event) -> None:
        """Restore drag mode and cursor after panning or forward to base class."""
        if self._was_panning and event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Middle mouse button released; ending panning")
            # Restore original drag mode and cursor
            self.setDragMode(self._orig_drag_mode)
            if self._orig_cursor is not None:
                self.setCursor(self._orig_cursor)
            self._was_panning = False
            # Delegate to base class to finish the drag
            super().mouseReleaseEvent(event)
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Detect hover over patches and emit signal to highlight corresponding scatter point."""
        # Ignore move events if no pixmap or coordinates are loaded
        if self.pixmap_item is None or self.coords is None:
            super().mouseMoveEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        # If cursor is outside the image, reset hover
        if not self.pixmap_item.contains(scene_pos):
            # Notify hover off with index -1
            self.patch_hovered.emit(-1, False)
            super().mouseMoveEvent(event)
            return
        x = scene_pos.x()
        y = scene_pos.y()
        # Determine nearest patch by Euclidean distance
        patch_centres = self.coords + self.patch_size / 2.0
        diffs = patch_centres - np.array([x, y])
        dists = np.einsum('ij,ij->i', diffs, diffs)
        idx = int(np.argmin(dists))
        # Emit hover signal with the index
        self.patch_hovered.emit(idx, True)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        """Zoom in/out on scroll wheel."""
        # Determine zoom factor based on scroll direction
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        print(f"DEBUG: Slide view zoom with factor {factor}")
        self.scale(factor, factor)
        event.accept()


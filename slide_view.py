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

The view supports two rendering modes:
- **Thumbnail Mode**: Traditional single-image display (original behavior)
- **Adaptive Mode**: Multi-resolution tiled rendering using OpenSlide pyramid

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
from PySide6.QtCore import Qt, QTimer, QPointF, QPoint, QObject, QEvent
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter, QCursor, QMouseEvent
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

# Import tile manager for adaptive zoom
try:
    from tile_manager import TileManager
    TILE_MANAGER_AVAILABLE = True
except ImportError:
    TileManager = None
    TILE_MANAGER_AVAILABLE = False
class SlideGraphicsView(QGraphicsView):
    """View for displaying whole-slide images with interactive patch overlays.
    
    This view displays a whole-slide image with colored rectangular overlays
    representing patches. Each patch is colored according to its cluster
    assignment, and patches can be highlighted through user interactions.
    The view supports panning, zooming, clicking, and hovering.
    
    Rendering Modes
    ---------------
    
    The view supports two rendering modes:
    
    - **Thumbnail Mode** (adaptive_mode=False): Traditional single-image display
      using a downsampled thumbnail. Fast loading but limited zoom quality.
    
    - **Adaptive Mode** (adaptive_mode=True): Multi-resolution tiled rendering
      using OpenSlide's image pyramid. Enables seamless zoom from overview to
      full cellular resolution.
    
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
      is 1.25x per scroll step. In adaptive mode, also triggers LOD updates.
    
    Attributes
    ----------
    adaptive_mode : bool
        If True, use multi-resolution tiled rendering. If False, use thumbnail.
    tile_manager : Optional[TileManager]
        Manager for loading and caching tiles (adaptive mode only).
    tile_items : Dict[Tuple[int, int, int], QGraphicsPixmapItem]
        Mapping from (level, col, row) to tile graphics items.
    current_lod : int
        Current level-of-detail being displayed.
    pixmap_item : Optional[QGraphicsPixmapItem]
        The graphics item displaying the slide thumbnail image (thumbnail mode).
    rect_items : List[QGraphicsRectItem]
        List of rectangular overlay items, one per patch. Each rectangle
        is colored according to its cluster assignment.
    coords : Optional[np.ndarray]
        Patch coordinates in scene space (level-0 in adaptive mode, thumbnail in thumbnail mode).
    labels : Optional[np.ndarray]
        Cluster labels for each patch (shape: n_patches,).
    patch_size : float
        Size of each patch in scene coordinate space.
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
    cluster_selected = Signal(int, tuple, bool)  # cluster number, click position, ctrl pressed
    patch_hovered = Signal(int, bool)  # patch index and hover state
    preview_action_attempted = Signal()  # emitted when clicking without model data
    # Signals for animation synchronization
    patches_highlighted = Signal(list)  # list of patch indices highlighted this step
    animation_completed = Signal()  # emitted when cascade animation finishes
    # Signal for local region selection mode
    local_region_selected = Signal(tuple, float)  # (click_point, radius)
    
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
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # Adaptive zoom mode settings
        self.adaptive_mode: bool = False
        self.tile_manager: Optional[TileManager] = None
        self.tile_items: Dict[Tuple[int, int, int], QGraphicsPixmapItem] = {}
        self.current_lod: int = 0
        self._slide_dimensions: Optional[Tuple[int, int]] = None
        self._tile_update_pending: bool = False
        self._tile_update_timer = QTimer(self)
        self._tile_update_timer.setSingleShot(True)
        self._tile_update_timer.setInterval(50)  # Debounce tile updates
        self._tile_update_timer.timeout.connect(self._do_update_visible_tiles)
        
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
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Variable to track middle-click panning state
        self._was_panning = False
        self._suppress_clicks = False
        # Local region selection mode state
        self._local_region_mode: bool = False
        self._local_region_radius: float = 50.0

    def _forward_drag_event(self, event: QMouseEvent, button: Qt.MouseButton) -> None:
        """Forward a synthetic mouse event to Qt for native drag handling."""
        from PySide6.QtCore import QEvent
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


    def load_slide(self, image: Image.Image, coords: np.ndarray, patch_size: float,
                   labels: np.ndarray, cluster_colors: List[str]) -> None:
        """Load a new slide in thumbnail mode.

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
        # Close existing tile manager if any
        self._close_tile_manager()
        # Clear existing scene
        scene = self.scene()
        scene.clear()
        self.rect_items.clear()
        self.tile_items.clear()
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
    
    def load_slide_adaptive(self, slide_path: str, coords_lv0: np.ndarray,
                            patch_size_lv0: float, labels: np.ndarray,
                            cluster_colors: List[str],
                            slide_dimensions: Tuple[int, int]) -> bool:
        """Load a slide in adaptive multi-resolution mode.
        
        In adaptive mode, the scene uses level-0 coordinates and tiles are
        loaded dynamically based on the current zoom level and viewport.
        
        Parameters
        ----------
        slide_path : str
            Path to the whole-slide image file.
        coords_lv0 : np.ndarray
            Patch coordinates in level-0 pixel space (shape `(n_patches, 2)`).
        patch_size_lv0 : float
            Size of each patch in level-0 pixels.
        labels : np.ndarray
            Cluster labels for each patch.
        cluster_colors : List[str]
            Colour palette for clusters.
        slide_dimensions : Tuple[int, int]
            Full resolution (width, height) of the slide.
            
        Returns
        -------
        bool
            True if adaptive mode was successfully initialized, False otherwise.
        """
        if not TILE_MANAGER_AVAILABLE:
            print("DEBUG: TileManager not available, falling back to thumbnail mode")
            return False
        
        # Stop animations and close existing tile manager
        self._timer.stop()
        self._animation_order = []
        self._animation_index = 0
        self._close_tile_manager()
        
        # Clear scene
        scene = self.scene()
        scene.clear()
        self.rect_items.clear()
        self.tile_items.clear()
        self.pixmap_item = None
        
        # Try to create tile manager
        try:
            self.tile_manager = TileManager(slide_path, tile_size=256, cache_mb=250)
            self.tile_manager.tile_loaded.connect(self._on_tile_loaded)
            self._slide_dimensions = slide_dimensions
            print(f"DEBUG: TileManager created for {slide_path}")
        except Exception as e:
            print(f"DEBUG: Failed to create TileManager: {e}")
            return False
        
        # Store data using level-0 coordinates
        self.coords = coords_lv0
        self.labels = labels.astype(int)
        self.patch_size = float(patch_size_lv0)
        self.cluster_colors = [_hsl_to_qcolor(col) for col in cluster_colors]
        
        # Set scene rect to full slide dimensions
        slide_w, slide_h = slide_dimensions
        scene.setSceneRect(0, 0, slide_w, slide_h)
        
        # Create overlay rect items at level-0 coordinates
        for idx, (x, y) in enumerate(coords_lv0):
            label = int(labels[idx])
            color = self.cluster_colors[label] if label < len(self.cluster_colors) else QColor('red')
            rect = QGraphicsRectItem(x, y, self.patch_size, self.patch_size)
            rect.setBrush(color)
            rect.setPen(Qt.NoPen)
            rect.setOpacity(0.0)
            rect.setFlag(QGraphicsItem.ItemIsSelectable, False)
            # Set Z-value to ensure overlays render above tiles
            rect.setZValue(1)
            scene.addItem(rect)
            self.rect_items.append(rect)
        
        # Fit view to scene and trigger initial tile load
        self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._update_visible_tiles()
        
        print(f"DEBUG: Adaptive mode loaded with {len(self.rect_items)} overlays")
        return True
    
    def set_adaptive_mode(self, enabled: bool) -> None:
        """Enable or disable adaptive zoom mode.
        
        Note: This only sets the mode flag. The actual mode switch happens
        when the next slide is loaded.
        
        Parameters
        ----------
        enabled : bool
            True to enable adaptive mode, False for thumbnail mode.
        """
        self.adaptive_mode = enabled
        print(f"DEBUG: Adaptive mode set to {enabled}")
    
    def _close_tile_manager(self) -> None:
        """Close and clean up the tile manager."""
        if self.tile_manager is not None:
            try:
                self.tile_manager.tile_loaded.disconnect(self._on_tile_loaded)
            except RuntimeError:
                pass  # Signal was already disconnected
            self.tile_manager.close()
            self.tile_manager = None
    
    def _update_visible_tiles(self) -> None:
        """Schedule an update of visible tiles (debounced)."""
        if self.tile_manager is None:
            return
        if not self._tile_update_timer.isActive():
            self._tile_update_timer.start()
    
    def _do_update_visible_tiles(self) -> None:
        """Actually update visible tiles based on current viewport.
        
        Uses priority-based loading where tiles closer to the viewport center
        are loaded first, improving perceived performance. Also prefetches
        tiles just outside the viewport for smoother panning.
        """
        if self.tile_manager is None:
            return
        
        # Get current view scale (screen pixels / scene pixels)
        transform = self.transform()
        view_scale = transform.m11()  # Assuming uniform scaling
        
        # Determine appropriate LOD level
        new_lod = self.tile_manager.get_level_for_scale(view_scale)
        
        # Get viewport in scene coordinates
        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        
        # Clamp viewport to slide bounds
        slide_rect = QRectF(0, 0, self._slide_dimensions[0], self._slide_dimensions[1])
        viewport_rect = viewport_rect.intersected(slide_rect)
        
        if viewport_rect.isEmpty():
            return
        
        # Calculate viewport center for priority ordering
        viewport_center = viewport_rect.center()
        
        # Get visible tiles at this LOD
        visible_tiles = self.tile_manager.get_visible_tiles(viewport_rect, new_lod)
        
        # If LOD changed, cancel pending requests and clear old tiles
        if new_lod != self.current_lod:
            print(f"DEBUG: LOD changed from {self.current_lod} to {new_lod}")
            self.tile_manager.cancel_pending()
            self._clear_tiles_for_lod_change(new_lod)
            self.current_lod = new_lod
        
        # Add placeholder tiles for tiles not yet loaded
        self._add_placeholder_tiles(visible_tiles)
        
        # Request visible tiles with priority (center-first)
        if visible_tiles:
            self.tile_manager.request_tiles_prioritized(
                visible_tiles, 
                QPointF(viewport_center.x(), viewport_center.y())
            )
        
        # Prefetch neighboring tiles (lower priority)
        prefetch_tiles = self.tile_manager.get_prefetch_tiles(viewport_rect, new_lod)
        if prefetch_tiles:
            # Use a large priority offset to ensure prefetch tiles load after visible tiles
            # Max distance squared for a typical viewport is roughly viewport_size^2
            # Use a large constant offset to ensure visible tiles always come first
            prefetch_offset = 1e12  # Large offset to deprioritize prefetch
            self.tile_manager.request_tiles_prioritized(
                prefetch_tiles,
                QPointF(viewport_center.x(), viewport_center.y()),
                priority_offset=prefetch_offset
            )
    
    def _add_placeholder_tiles(self, tiles: List[Tuple[int, int, int]]) -> None:
        """Add placeholder graphics items for tiles that aren't loaded yet.
        
        Parameters
        ----------
        tiles : List[Tuple[int, int, int]]
            List of (level, col, row) tuples.
        """
        if self.tile_manager is None:
            return
        
        scene = self.scene()
        
        for level, col, row in tiles:
            key = (level, col, row)
            
            # Skip if tile already exists or is in cache
            if key in self.tile_items:
                continue
            if self.tile_manager.cache.contains(key):
                continue
            
            # Create placeholder
            bounds = self.tile_manager.get_tile_bounds_lv0(level, col, row)
            placeholder = self._create_placeholder_pixmap(256, 256)
            
            item = QGraphicsPixmapItem(placeholder)
            item.setPos(bounds.x(), bounds.y())
            scale_x = bounds.width() / placeholder.width()
            item.setScale(scale_x)
            item.setZValue(-1)  # Placeholders below real tiles
            scene.addItem(item)
            self.tile_items[key] = item
    
    def _clear_tiles_for_lod_change(self, new_lod: int) -> None:
        """Remove tiles from other LOD levels when switching."""
        scene = self.scene()
        keys_to_remove = []
        
        for key, item in self.tile_items.items():
            level, col, row = key
            if level != new_lod:
                scene.removeItem(item)
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.tile_items[key]
    
    def _create_placeholder_pixmap(self, width: int, height: int) -> QPixmap:
        """Create a placeholder pixmap for tiles that are loading.
        
        Parameters
        ----------
        width : int
            Width of the placeholder.
        height : int
            Height of the placeholder.
            
        Returns
        -------
        QPixmap
            A light gray placeholder pixmap.
        """
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(240, 240, 240))  # Light gray
        return pixmap
    
    def _on_tile_loaded(self, level: int, col: int, row: int, pixmap: QPixmap) -> None:
        """Handle a loaded tile from the TileManager.
        
        Parameters
        ----------
        level : int
            DeepZoom level of the tile.
        col : int
            Column index of the tile.
        row : int
            Row index of the tile.
        pixmap : QPixmap
            The tile image.
        """
        if self.tile_manager is None:
            return
        
        # Ignore tiles from wrong LOD level
        if level != self.current_lod:
            return
        
        key = (level, col, row)
        scene = self.scene()
        
        # Get tile bounds in level-0 coordinates
        bounds = self.tile_manager.get_tile_bounds_lv0(level, col, row)
        
        # Create or update tile item
        if key in self.tile_items:
            item = self.tile_items[key]
            item.setPixmap(pixmap)
        else:
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(bounds.x(), bounds.y())
            # Scale tile to cover level-0 area
            scale_x = bounds.width() / pixmap.width()
            scale_y = bounds.height() / pixmap.height()
            item.setScale(scale_x)  # Assuming square tiles
            # Tiles render below overlays
            item.setZValue(0)
            scene.addItem(item)
            self.tile_items[key] = item
    
    def _clear_all_tiles(self) -> None:
        """Remove all tile items from the scene."""
        scene = self.scene()
        for item in self.tile_items.values():
            scene.removeItem(item)
        self.tile_items.clear()
    
    def cleanup(self) -> None:
        """Clean up resources when the view is being destroyed.
        
        This should be called before the widget is deleted to ensure
        proper cleanup of the tile manager and background threads.
        """
        self._timer.stop()
        self._tile_update_timer.stop()
        self._close_tile_manager()
        self._clear_all_tiles()
    
    def closeEvent(self, event) -> None:
        """Handle widget close event."""
        self.cleanup()
        super().closeEvent(event)

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
        """Handle mouse press events for cluster selection and native panning.
        
        Uses Qt's native ScrollHandDrag for middle button panning to eliminate
        coordinate system issues causing acceleration. All other events are
        forwarded to base class for normal processing.
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Middle mouse button pressed for native panning")
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._was_panning = True
            self._suppress_clicks = True
            # Forward as left button press to trigger Qt's native drag
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            return
        super().mousePressEvent(event)

        if self._suppress_clicks:
            return

        # Ensure required data is loaded
        # In adaptive mode, we don't have a pixmap_item but we have tile_manager
        has_image = (self.pixmap_item is not None) or (self.tile_manager is not None)
        if not has_image or self.coords is None or self.labels is None:
            print("DEBUG: Missing required data for highlighting:", {
                "has_image": has_image,
                "coords": self.coords is None,
                "labels": self.labels is None
            })
            if has_image:
                self.preview_action_attempted.emit()
            return

        if len(self.coords) == 0 or len(self.labels) == 0:
            self.preview_action_attempted.emit()
            return

        # Map the click position to the scene
        scene_pos = self.mapToScene(event.position().toPoint())
        print(f"DEBUG: Click position mapped to scene: ({scene_pos.x()}, {scene_pos.y()})")
        
        # Check if click is within bounds
        if self.tile_manager is not None:
            # Adaptive mode: check against slide dimensions
            scene_rect = self.scene().sceneRect()
            if not scene_rect.contains(scene_pos):
                print("DEBUG: Click outside slide bounds; ignoring")
                return
        elif self.pixmap_item is not None:
            # Thumbnail mode: check against pixmap
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

        # Check if in local region mode
        if self._local_region_mode:
            # Emit local region selection signal
            print(f"DEBUG: Local region mode - emitting local_region_selected at ({x}, {y}) with radius {self._local_region_radius}")
            self.local_region_selected.emit((x, y), self._local_region_radius)
        else:
            # Normal K-means mode: emit cluster selection signal
            ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
            self.cluster_selected.emit(cluster, (x, y), ctrl_pressed)
        # Animation is triggered by the main window to support multi-select

    def _start_animation(self, cluster: int, order: List[int],
                         persisted_clusters: Optional[set[int]] = None) -> None:
        """Reset opacities and begin the sweep animation for a cluster."""
        print(f"DEBUG: _start_animation called with cluster {cluster}")
        persisted = persisted_clusters or set()
        # Stop any previous animation
        self._timer.stop()
        self._animation_index = 0
        self._animation_order = order
        # Reset all opacities: persisted clusters stay on, new cluster low, others off
        for i, rect in enumerate(self.rect_items):
            label = int(self.labels[i])
            if label in persisted and label != cluster:
                rect.setOpacity(self.highlight_opacity_on)
            elif label == cluster:
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
        """Restore drag mode after Qt-native panning or forward to base class."""
        if self._was_panning and event.button() == Qt.MouseButton.MiddleButton:
            print("DEBUG: Middle mouse button released; ending native panning")
            # Forward as left button release to complete Qt's native drag
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._was_panning = False
            self._suppress_clicks = False
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Detect hover over patches and emit signal to highlight corresponding scatter point."""
        # Ignore move events during Qt-native panning
        if self._was_panning and (event.buttons() & Qt.MouseButton.MiddleButton):
            # Forward as left drag to trigger Qt's native drag
            self._forward_drag_event(event, Qt.MouseButton.LeftButton)
            return
        
        # Ignore move events if no image or coordinates are loaded
        has_image = (self.pixmap_item is not None) or (self.tile_manager is not None)
        if not has_image or self.coords is None:
            super().mouseMoveEvent(event)
            return
        
        scene_pos = self.mapToScene(event.position().toPoint())
        
        # Check if cursor is within bounds
        is_inside = False
        if self.tile_manager is not None:
            # Adaptive mode: check against scene rect
            is_inside = self.scene().sceneRect().contains(scene_pos)
        elif self.pixmap_item is not None:
            # Thumbnail mode: check against pixmap
            is_inside = self.pixmap_item.contains(scene_pos)
        
        if not is_inside:
            # Notify hover off with index -1
            self.patch_hovered.emit(-1, False)
            super().mouseMoveEvent(event)
            return
        
        x = scene_pos.x()
        y = scene_pos.y()
        if self.coords is None or len(self.coords) == 0 or self.labels is None or len(self.labels) == 0:
            self.patch_hovered.emit(-1, False)
            super().mouseMoveEvent(event)
            return

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

        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = old_pos - new_pos
        self.translate(delta.x(), delta.y())

        # In adaptive mode, update visible tiles after zoom
        if self.tile_manager is not None:
            self._update_visible_tiles()

        # Update local region cursor if in local region mode
        if self._local_region_mode:
            self._update_radius_cursor()

        event.accept()
    
    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """Handle scrolling/panning and update tiles if needed."""
        super().scrollContentsBy(dx, dy)
        
        # In adaptive mode, update visible tiles after pan
        if self.tile_manager is not None:
            self._update_visible_tiles()
    
    def resizeEvent(self, event) -> None:
        """Handle view resize and update tiles if needed."""
        super().resizeEvent(event)

        # In adaptive mode, update visible tiles after resize
        if self.tile_manager is not None:
            self._update_visible_tiles()

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
            Initial selection radius in scene coordinates.
        """
        self._local_region_mode = enabled
        self._local_region_radius = radius
        if enabled:
            self._update_radius_cursor()
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_local_region_radius(self, radius: float) -> None:
        """Update the selection radius.

        Parameters
        ----------
        radius : float
            New selection radius in scene coordinates.
        """
        self._local_region_radius = radius
        if self._local_region_mode:
            self._update_radius_cursor()

    def _update_radius_cursor(self) -> None:
        """Create a circle cursor matching the current radius and zoom level."""
        # Get current zoom scale
        scale = self.transform().m11()

        # Calculate cursor diameter in screen pixels
        # Radius is in scene coordinates, scale converts to screen pixels
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
        painter.setBrush(QBrush(QColor(255, 100, 100, 30)))  # Semi-transparent fill

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
        self.setCursor(cursor)


"""
atlas_widgets.py
================

Widget classes for atlas slide list display and thumbnail panel.

Contains:
- AtlasSlideListWidget: Widget listing slides added to the atlas builder.
- ThumbnailLoadTask: Background QRunnable for loading slide thumbnails.
- SlideThumbnailItem: Clickable slide thumbnail card with loading/error states.
- SlideThumbnailListWidget: Horizontal scrollable list of SlideThumbnailItem cards.
- AtlasThumbnailView: Small QGraphicsView thumbnail with patch overlays.
- AtlasThumbnailLoadTask: Background QRunnable for loading atlas thumbnails.
- AtlasThumbnailPanel: Vertical scrollable panel of AtlasThumbnailView items.
- PatchExemplarPopup: Horizontally scrollable dialog for patch exemplar cards.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

import data_loader
from .atlas_builder import ClusterAtlas
from PySide6.QtCore import (
    Qt,
    QObject,
    QRunnable,
    QSize,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


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
                  h5_path: str = "", color: QColor = None,
                  alias: Optional[str] = None) -> None:
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
        display = alias if alias else (slide_name[:25] + "\u2026" if len(slide_name) > 25 else slide_name)
        name_label = QLabel(display)
        name_label.setStyleSheet("color: black;")
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

    def __init__(self, slide_name: str, thumb_size: QSize,
                 display_name: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._slide_name = slide_name
        self._display_name = display_name or slide_name
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
        elided = metrics.elidedText(self._display_name, Qt.ElideRight, available_width)
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
        self._thumb_size = QSize(120, 40)
        self._items: Dict[str, SlideThumbnailItem] = {}
        self._slide_numbers: Dict[str, int] = {}
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
        self._slide_numbers.clear()

        for idx, slide_name in enumerate(slide_names):
            number = idx + 1
            self._slide_numbers[slide_name] = number
            info = slide_infos.get(slide_name)
            image_path = info.image_path if info else ""
            self._slide_paths[slide_name] = image_path

            display_name = f"{number}  {slide_name}"
            item = SlideThumbnailItem(slide_name, self._thumb_size, display_name=display_name)
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

    def get_slide_number(self, slide_name: str) -> Optional[int]:
        """Return the 1-based position of a slide, or None if not found."""
        return self._slide_numbers.get(slide_name)

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


class AtlasThumbnailView(QGraphicsView):
    """Simplified thumbnail view with patch overlays for atlas display.

    Displays a small thumbnail of a slide with colored patch rectangles
    representing cluster assignments. Used in the atlas thumbnail panel
    to show cluster highlights across all atlas slides simultaneously.
    """
    clicked = Signal(str)  # Emits slide_name - placeholder for future

    THUMB_WIDTH = 150
    THUMB_HEIGHT = 100

    def __init__(self, slide_name: str, parent=None):
        super().__init__(parent)
        self._slide_name = slide_name
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._rect_items: List[QGraphicsRectItem] = []
        self._labels: Optional[np.ndarray] = None
        self._cluster_colors: List[QColor] = []
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

        # Fixed size for thumbnails
        self.setFixedSize(self.THUMB_WIDTH + 4, self.THUMB_HEIGHT + 4)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setStyleSheet("border: 1px solid #999;")

    def load_data(self, image: QImage, coords: np.ndarray,
                  labels: np.ndarray, cluster_colors: List[QColor],
                  original_dims: Tuple[int, int]) -> None:
        """Load thumbnail image and create patch overlay rectangles.

        Parameters
        ----------
        image : QImage
            The thumbnail image to display.
        coords : np.ndarray
            Patch coordinates in original slide dimensions (level 0).
        labels : np.ndarray
            Cluster labels for each patch.
        cluster_colors : List[QColor]
            Colors for each cluster.
        original_dims : Tuple[int, int]
            Original slide dimensions (width, height) for coordinate scaling.
        """
        self._scene.clear()
        self._rect_items.clear()
        self._labels = labels
        self._cluster_colors = cluster_colors

        # Set background image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.THUMB_WIDTH, self.THUMB_HEIGHT,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._pixmap_item = self._scene.addPixmap(scaled_pixmap)

        # Calculate scale factors
        thumb_w = scaled_pixmap.width()
        thumb_h = scaled_pixmap.height()
        scale_x = thumb_w / original_dims[0] if original_dims[0] > 0 else 1.0
        scale_y = thumb_h / original_dims[1] if original_dims[1] > 0 else 1.0

        # Estimate patch size from coords (assume square patches)
        if len(coords) > 1:
            # Find minimum distance between adjacent patches
            diffs = np.diff(np.sort(coords[:, 0]))
            diffs = diffs[diffs > 0]
            patch_size_lv0 = float(np.min(diffs)) if len(diffs) > 0 else 256.0
        else:
            patch_size_lv0 = 256.0

        thumb_patch_size = max(2, patch_size_lv0 * scale_x)

        # Create rect items for each patch
        for i, (x, y) in enumerate(coords[:, :2]):
            tx = x * scale_x
            ty = y * scale_y
            color = cluster_colors[labels[i]] if labels[i] < len(cluster_colors) else QColor(128, 128, 128)

            rect = self._scene.addRect(
                tx, ty, thumb_patch_size, thumb_patch_size,
                QPen(Qt.NoPen),
                QBrush(color)
            )
            rect.setOpacity(0.0)  # Initially hidden
            self._rect_items.append(rect)

        # Fit view to scene
        self._scene.setSceneRect(0, 0, thumb_w, thumb_h)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def highlight_cluster(self, cluster_id: int, selected_clusters: Set[int]) -> None:
        """Highlight patches belonging to selected clusters.

        Parameters
        ----------
        cluster_id : int
            The primary cluster that was clicked.
        selected_clusters : Set[int]
            All currently selected clusters (for multi-select support).
        """
        if self._labels is None:
            return

        for i, rect in enumerate(self._rect_items):
            label = int(self._labels[i])
            if label in selected_clusters:
                rect.setOpacity(0.6)
            else:
                rect.setOpacity(0.0)

    def clear_highlight(self) -> None:
        """Hide all patch overlays."""
        for rect in self._rect_items:
            rect.setOpacity(0.0)

    def mousePressEvent(self, event) -> None:
        """Emit clicked signal on mouse press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._slide_name)
        super().mousePressEvent(event)


class AtlasThumbnailLoadTask(QObject, QRunnable):
    """Background task for loading atlas slide thumbnails."""

    loaded = Signal(str, QImage, tuple)  # slide_name, image, original_dims
    failed = Signal(str, str)  # slide_name, error_msg

    def __init__(self, slide_name: str, image_path: str, max_size: int = 150) -> None:
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._slide_name = slide_name
        self._image_path = image_path
        self._max_size = max_size

    def run(self) -> None:
        try:
            # Load thumbnail and get original dimensions
            image = data_loader.load_thumbnail(self._image_path, max_size=self._max_size)

            # Get original slide dimensions
            import openslide
            slide = openslide.OpenSlide(self._image_path)
            original_dims = slide.dimensions
            slide.close()

            qimage = QImage(ImageQt(image))
            self.loaded.emit(self._slide_name, qimage, original_dims)
        except Exception as exc:
            self.failed.emit(self._slide_name, str(exc))


class AtlasThumbnailPanel(QWidget):
    """Vertical scrollable panel of atlas slide thumbnails with patch overlays.

    Displays thumbnails of all slides in the atlas. When a cluster is selected,
    patches belonging to that cluster are highlighted on all thumbnails
    simultaneously, enabling visual comparison across slides.
    """
    slide_clicked = Signal(str)  # Placeholder for future click handling

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumbnail_views: Dict[str, AtlasThumbnailView] = {}
        self._pending_tasks: Dict[str, AtlasThumbnailLoadTask] = {}
        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(2)
        self._atlas: Optional[ClusterAtlas] = None

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("Atlas Slides")
        header.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(header)

        # Scroll area for thumbnails
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for scroll area
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(6)
        self._container_layout.addStretch()

        self._scroll_area.setWidget(self._container)
        layout.addWidget(self._scroll_area)

        # Set fixed width for the panel
        self.setFixedWidth(AtlasThumbnailView.THUMB_WIDTH + 24)

    def populate(self, atlas: ClusterAtlas, slides: Dict[str, 'data_loader.SlideInfo'],
                 slide_numbers: Optional[Dict[str, int]] = None) -> None:
        """Populate the panel with thumbnails for all atlas slides.

        Parameters
        ----------
        atlas : ClusterAtlas
            The built cluster atlas containing slide entries.
        slides : Dict[str, SlideInfo]
            Dictionary mapping slide names to their info (for image paths).
        slide_numbers : Dict[str, int], optional
            Mapping of slide name to 1-based index for compact labels.
        """
        self.clear()
        self._atlas = atlas

        # Get cluster colors from atlas
        cluster_colors = [QColor(c) for c in atlas.cluster_colors]

        for slide_name in atlas.slide_names:
            # Create thumbnail view
            view = AtlasThumbnailView(slide_name)
            view.clicked.connect(self._on_thumbnail_clicked)

            # Add slide name label — show "#N" if a number mapping is available
            if slide_numbers and slide_name in slide_numbers:
                label_text = f"#{slide_numbers[slide_name]}"
            else:
                label_text = slide_name
            name_label = QLabel(label_text)
            name_label.setToolTip(slide_name)
            name_label.setStyleSheet("font-size: 9px;")
            name_label.setWordWrap(True)
            name_label.setMaximumWidth(AtlasThumbnailView.THUMB_WIDTH)

            # Container for view + label
            item_container = QWidget()
            item_layout = QVBoxLayout(item_container)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(2)
            item_layout.addWidget(view, alignment=Qt.AlignCenter)
            item_layout.addWidget(name_label, alignment=Qt.AlignCenter)

            # Insert before the stretch
            self._container_layout.insertWidget(
                self._container_layout.count() - 1,
                item_container
            )

            self._thumbnail_views[slide_name] = view

            # Start async thumbnail load
            slide_info = slides.get(slide_name)
            if slide_info and slide_info.image_path:
                task = AtlasThumbnailLoadTask(
                    slide_name,
                    slide_info.image_path,
                    max_size=AtlasThumbnailView.THUMB_WIDTH
                )
                task.loaded.connect(self._on_thumbnail_loaded)
                task.failed.connect(self._on_thumbnail_failed)
                self._pending_tasks[slide_name] = task
                self._thread_pool.start(task)

    def _on_thumbnail_loaded(self, slide_name: str, image: QImage,
                             original_dims: Tuple[int, int]) -> None:
        """Handle successful thumbnail load."""
        if slide_name not in self._thumbnail_views or self._atlas is None:
            return

        view = self._thumbnail_views[slide_name]
        entry = self._atlas.entries.get(slide_name)
        if entry is None:
            return

        # Get cluster colors from atlas
        cluster_colors = [QColor(c) for c in self._atlas.cluster_colors]

        view.load_data(
            image,
            entry.coords,
            entry.global_labels,
            cluster_colors,
            original_dims
        )

        # Clean up task reference
        self._pending_tasks.pop(slide_name, None)

    def _on_thumbnail_failed(self, slide_name: str, error_msg: str) -> None:
        """Handle failed thumbnail load."""
        print(f"DEBUG: Atlas thumbnail load failed for {slide_name}: {error_msg}")
        self._pending_tasks.pop(slide_name, None)

    def _on_thumbnail_clicked(self, slide_name: str) -> None:
        """Forward click to panel signal."""
        self.slide_clicked.emit(slide_name)

    def highlight_cluster(self, cluster_id: int, selected_clusters: Set[int]) -> None:
        """Highlight patches in selected clusters on all thumbnails.

        Parameters
        ----------
        cluster_id : int
            The primary cluster that was selected.
        selected_clusters : Set[int]
            All currently selected clusters.
        """
        for view in self._thumbnail_views.values():
            view.highlight_cluster(cluster_id, selected_clusters)

    def clear_highlight(self) -> None:
        """Clear highlights on all thumbnails."""
        for view in self._thumbnail_views.values():
            view.clear_highlight()

    def clear(self) -> None:
        """Remove all thumbnail views and reset state."""
        # Cancel pending tasks (they'll be ignored when they complete)
        self._pending_tasks.clear()
        self._thumbnail_views.clear()
        self._atlas = None

        # Remove all widgets except the stretch
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class PatchExemplarPopup(QDialog):
    """Horizontally scrollable popup that renders exemplar patch cards."""

    def __init__(
        self,
        popup_id: str,
        source_label: str,
        strategy: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.popup_id = popup_id
        self.setWindowTitle(f"Patch Exemplars - {source_label}")
        self.resize(1080, 360)
        self._items: List[Dict[str, Any]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        self._title = QLabel(f"{source_label} exemplars")
        self._title.setStyleSheet("font-weight: 600; font-size: 12pt;")
        self._meta = QLabel(f"Strategy: {strategy}")
        self._meta.setStyleSheet("color: #666;")
        self._export_btn = QPushButton("Export")
        self._close_btn = QPushButton("Close")
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._meta)
        top.addWidget(self._export_btn)
        top.addWidget(self._close_btn)
        root.addLayout(top)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(10)
        self._scroll.setWidget(self._container)
        root.addWidget(self._scroll, stretch=1)

        self._close_btn.clicked.connect(self.close)

    def set_items(
        self,
        items: List[Dict[str, Any]],
        include_metadata: bool = True,
    ) -> None:
        self._items = items
        while self._row.count():
            child = self._row.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for item in items:
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setMinimumWidth(180)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 6, 6, 6)
            card_layout.setSpacing(4)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            pix = item.get("pixmap")
            if isinstance(pix, QPixmap) and not pix.isNull():
                img_label.setPixmap(pix)
            else:
                img_label.setText("Image unavailable")
                img_label.setStyleSheet("color: #888;")
                img_label.setMinimumSize(QSize(160, 160))
            card_layout.addWidget(img_label)

            if include_metadata:
                idx = int(item.get("patch_index", -1))
                cluster = item.get("cluster_id")
                coords = item.get("coords_lv0", {})
                score = item.get("score")
                details = [
                    f"Patch {idx}",
                    f"Cluster: {cluster if cluster is not None else 'N/A'}",
                    f"({coords.get('x', '?')}, {coords.get('y', '?')})",
                ]
                if score is not None:
                    details.append(f"Score: {float(score):.3f}")
                for line in details:
                    lbl = QLabel(line)
                    lbl.setStyleSheet("font-size: 9pt;")
                    card_layout.addWidget(lbl)

            self._row.addWidget(card)

        self._row.addStretch(1)

    def items(self) -> List[Dict[str, Any]]:
        """Return currently displayed exemplar items."""
        return self._items

    def export_button(self) -> QPushButton:
        """Expose export button for parent wiring."""
        return self._export_btn

"""
view_mixin.py
=============
ViewMixin mixin for MainWindow.
Extracted from gui.py — contains related methods that are combined
back into MainWindow via multiple inheritance.
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
from PySide6.QtGui import QPen, QBrush
from PySide6.QtCore import QRectF, Signal
from sklearn.decomposition import PCA
from PIL import Image
from PIL.ImageQt import ImageQt
import data_loader
from utils import (
    generate_palette, cluster_features, infer_slide_dims, radial_sweep_order,
    compute_spatial_subclusters, compute_region_pca_embedding, RegionInfo,
)
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
from gui_types import SelectionMode, SourceMode, LocalRegionCluster, LabeledRegion, ModelSelection
from widgets import (
    CheckableComboBoxModel, ModelMultiSelector, ModelSelectionDialog,
    _qcolors_to_hsl_strings,
    PatchInfoPanel, PatchInfoPopup,
    ClusterLegendWidget, LocalRegionWidget, LabeledRegionsWidget,
)


class ViewMixin:

    def _set_model_dependent_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable model-dependent UI elements."""
        self.local_region_widget.setEnabled(enabled)
        self.labeled_regions_widget.setEnabled(enabled)
        self.scatter_view.setEnabled(enabled)
        self.atlas_add_current_btn.setEnabled(enabled)
        self.cluster_spin.setEnabled(enabled)
        self.export_action.setEnabled(enabled and bool(self._selected_clusters))
        if hasattr(self, 'label_mode_btn'):
            self.label_mode_btn.setEnabled(enabled)
            self.erase_mode_btn.setEnabled(enabled)


    def _clear_scatter_view(self) -> None:
        """Clear scatter view contents when no model is selected."""
        self.scatter_view.scene().clear()
        self.scatter_view._scatter_items.clear()
        self.scatter_view.labels = None
        self.scatter_view.cluster_colors = []
        self.scatter_view.set_animation_active(False)
        self.region_scatter_view.clear()
        self._current_regions = None
        self._current_region_embedding = None


    def _compute_and_populate_region_view(
        self,
        features: np.ndarray,
        coords_thumb: np.ndarray,
        labels: np.ndarray,
        colours: List[str],
    ) -> None:
        """Compute spatial subclusters and populate the region embedding view."""
        m = self._patches_per_region
        try:
            regions = compute_spatial_subclusters(features, coords_thumb, labels, colours, m)
        except Exception as e:
            print(f"DEBUG: compute_spatial_subclusters failed: {e}")
            self.region_scatter_view.clear()
            self._current_regions = None
            return

        if len(regions) < 2:
            self.region_scatter_view.clear()
            self._current_regions = None
            return

        coords_2d, _ = compute_region_pca_embedding(regions, features)
        if coords_2d is None:
            self.region_scatter_view.clear()
            self._current_regions = None
            return

        self._current_regions = regions
        self._current_region_embedding = coords_2d
        cluster_colors_q = [
            _hsl_to_qcolor(c) if isinstance(c, str) else c for c in colours
        ]
        self.region_scatter_view.populate(coords_2d, regions, cluster_colors_q)
        print(f"DEBUG: Region view populated with {len(regions)} regions (M={m})")


    def _show_toast(self, message: str) -> None:
        """Show a short-lived floating toast message near the cursor."""
        if self._preview_toast is None:
            self._preview_toast = QLabel(self)
            self._preview_toast.setStyleSheet(
                "QLabel { background-color: rgba(30,30,30,200); color: white;"
                "padding: 6px 10px; border-radius: 6px; font-size: 10pt; }"
            )
        self._preview_toast.setText(message)
        self._preview_toast.adjustSize()
        self._preview_toast.move(QCursor.pos() + QPoint(12, 12))
        self._preview_toast.show()
        self._preview_toast.raise_()
        QTimer.singleShot(2000, self._preview_toast.hide)


    def _show_preview_blocked_toast(self) -> None:
        """Show a short-lived toast when preview interactions are blocked."""
        if self._model_selection is not None:
            return
        self._show_toast("No model selected")


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
        if self._erase_mode:
            self._erase_kmeans_cluster(cluster)
            return
        if not self._create_kmeans_labeled_region(cluster):
            return
        # Prepare scatter baseline for cascade
        self._prepare_scatter_for_cascade(cluster)
        # Use the first patch's center as click point approximate
        cluster_indices = np.where(self.graphics_view.labels == cluster)[0]
        if cluster_indices.size == 0:
            return
        idx0 = cluster_indices[0]
        x, y = self.graphics_view.coords[idx0] + self.graphics_view.patch_size / 2.0
        self._start_slide_cascade(cluster, (x, y))


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
        if self._animation_in_progress:
            return
        if self._erase_mode:
            self._erase_kmeans_cluster(cluster)
            return
        print(f"DEBUG: Preparing scatter for synchronized cascade, cluster {cluster}")
        if not self._create_kmeans_labeled_region(cluster):
            return
        self._prepare_scatter_for_cascade(cluster)
        self._start_slide_cascade(cluster, click_point)

        # Also highlight on atlas thumbnails when atlas is active
        if self._is_atlas_active():
            self.atlas_thumbnail_panel.highlight_cluster(cluster, self._selected_clusters)


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
            self._apply_all_labeled_region_styles()


    def _on_label_mode_clicked(self) -> None:
        """Switch to Label mode."""
        self._erase_mode = False
        self.label_mode_btn.setChecked(True)
        self.erase_mode_btn.setChecked(False)
        self._update_view_cursors_for_mode()

    def _on_erase_mode_clicked(self) -> None:
        """Switch to Erase mode."""
        self._erase_mode = True
        self.erase_mode_btn.setChecked(True)
        self.label_mode_btn.setChecked(False)
        self._update_view_cursors_for_mode()

    def _update_view_cursors_for_mode(self) -> None:
        """Refresh cursor color in both views to reflect current erase/label mode."""
        if self._selection_mode == SelectionMode.LOCAL_REGION:
            self.graphics_view.set_erase_mode(self._erase_mode)
            self.scatter_view.set_erase_mode(self._erase_mode)

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

    def _set_selected_clusters(self, clusters: set[int]) -> None:
        """Replace selected clusters and sync UI state."""
        self._selected_clusters = set(clusters)
        self._update_export_action()


    def _add_selected_cluster(self, cluster: int) -> bool:
        """Add a cluster to the selection if not present."""
        if cluster in self._selected_clusters:
            return False
        self._selected_clusters.add(cluster)
        self._update_export_action()
        return True


    def _remove_selected_cluster(self, cluster: int) -> bool:
        """Remove a cluster from the selection if present."""
        if cluster not in self._selected_clusters:
            return False
        self._selected_clusters.remove(cluster)
        self._update_export_action()
        return True


    def _clear_selected_clusters(self) -> None:
        """Clear all selected clusters."""
        self._selected_clusters.clear()
        self._update_export_action()


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

        # Restore labeled regions so they stay visible during K-means selection
        for region in self._labeled_regions.values():
            for idx in region.patch_indices:
                if 0 <= idx < len(self.graphics_view.rect_items):
                    rect = self.graphics_view.rect_items[idx]
                    rect.setBrush(QBrush(region.color))
                    rect.setOpacity(0.55)


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

    def _on_region_scatter_clicked(self, region_index: int) -> None:
        """Handle click on a region X marker — creates a local region annotation."""
        if not self._current_regions or region_index >= len(self._current_regions):
            return
        region = self._current_regions[region_index]
        patch_set = set(region.patch_indices)
        center = self._compute_local_region_center(patch_set)
        self._create_local_region_cluster(patch_set, center, 0.0, region.kmeans_cluster)


    def _on_region_scatter_hovered(self, region_index: int, state: bool) -> None:
        """Handle hover on a region X marker — highlights patches in the slide view."""
        if not self._current_regions or region_index >= len(self._current_regions):
            return
        region = self._current_regions[region_index]
        if not self.graphics_view.rect_items:
            return
        n_rects = len(self.graphics_view.rect_items)
        if state:
            # Save current opacity for each patch then highlight
            for idx in region.patch_indices:
                if 0 <= idx < n_rects:
                    rect = self.graphics_view.rect_items[idx]
                    self._region_hover_saved_opacities[idx] = rect.opacity()
                    rect.setOpacity(1.0)
        else:
            # Restore saved opacities
            for idx in region.patch_indices:
                if 0 <= idx < n_rects:
                    rect = self.graphics_view.rect_items[idx]
                    saved = self._region_hover_saved_opacities.pop(idx, rect.opacity())
                    rect.setOpacity(saved)
            # Ensure no stale entries remain if regions changed mid-hover
            self._region_hover_saved_opacities.clear()


    def _show_preferences(self) -> None:
        """Display preferences dialog."""
        from preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self)
        if dialog.exec():
            # Reload preferences after dialog closes
            self.normalize_features = self.settings.value("normalize_features", True, type=bool)
            self._setup_chat_agent()
            # Reload M and recompute region view if data is loaded
            new_m = self.settings.value("patches_per_region", 15, type=int)
            if new_m != self._patches_per_region:
                self._patches_per_region = new_m
                if (self._current_features is not None
                        and self._current_labels is not None
                        and self._current_colours is not None):
                    self._compute_and_populate_region_view(
                        self._current_features, self._current_coords_thumb,
                        self._current_labels, self._current_colours,
                    )
    

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


    def changeEvent(self, event) -> None:
        from PySide6.QtCore import QEvent
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            # Delay to let macOS fullscreen/zoom animation complete before restoring cursors
            QTimer.singleShot(500, self._restore_view_cursors)


    def _restore_view_cursors(self) -> None:
        """Re-assert custom cursors after macOS fullscreen/maximize resets them."""
        # Clear all stale application-level override cursors
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        for view in (self.graphics_view, self.scatter_view, self.atlas_scatter_view):
            if hasattr(view, '_current_cursor'):
                view._override_pushed = False
                vp = view.viewport()
                # Edge case: mouse was already inside a view before/during fullscreen transition
                # (enterEvent won't fire again). Use underMouse() to detect and push override.
                if vp.underMouse():
                    QApplication.setOverrideCursor(view._current_cursor)
                    view._override_pushed = True


    def closeEvent(self, event) -> None:
        """Handle application close event and clean up resources."""
        print("DEBUG: Application closing, cleaning up resources")
        self._teardown_chat_agent()
        if self._active_exemplar_popup is not None:
            self._active_exemplar_popup.close()
            self._active_exemplar_popup.deleteLater()
            self._active_exemplar_popup = None
            self._active_exemplar_popup_id = None
        # Clean up tile manager in graphics view
        if hasattr(self, 'graphics_view'):
            self.graphics_view.cleanup()
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Local Region Selection Mode
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Atlas / local mode helpers
    # -------------------------------------------------------------------------


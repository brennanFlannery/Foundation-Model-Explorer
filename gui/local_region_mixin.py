"""
local_region_mixin.py
=====================
LocalRegionMixin mixin for MainWindow.
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


class LocalRegionMixin:

    @property
    def _local_region_clusters(self) -> Dict[int, "LabeledRegion"]:
        """Read-only backward-compat view of LOCAL-mode labeled regions."""
        return {rid: r for rid, r in self._labeled_regions.items()
                if r.source_mode == SourceMode.LOCAL}


    @property
    def _next_local_cluster_id(self) -> int:
        """Backward-compat alias for _next_region_id."""
        return self._next_region_id


    def _on_full_slide_toggled(self, checked: bool) -> None:
        """Handle Full Slide checkbox toggle — switches between KMEANS and LOCAL_REGION mode."""
        if checked:
            self._set_selection_mode(SelectionMode.KMEANS)
            self._restore_local_to_views()
            self._apply_all_labeled_region_styles()
        else:
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

            # Propagate current erase mode to views
            self.graphics_view.set_erase_mode(self._erase_mode)
            self.scatter_view.set_erase_mode(self._erase_mode)

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
        if self._animation_in_progress:
            return
        if self._erase_mode:
            self._erase_local_region_at_slide(click_point, radius)
            return
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
        if self._animation_in_progress:
            return
        if self._erase_mode:
            self._erase_local_region_at_scatter(click_point, radius)
            return
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


    def _erase_kmeans_cluster(self, cluster: int) -> None:
        """Delete all labeled regions (KMEANS or LOCAL) tied to this K-means cluster."""
        to_delete = [
            rid for rid, region in self._labeled_regions.items()
            if region.kmeans_cluster == cluster
        ]
        if not to_delete:
            return
        for rid in to_delete:
            region = self._labeled_regions.pop(rid, None)
            if region is None:
                continue
            if region.source_mode == SourceMode.LOCAL:
                self.local_region_widget.remove_region(rid)
            self.labeled_regions_widget.remove_region(rid)
        self._selected_clusters.discard(cluster)
        self._apply_all_labeled_region_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()

    def _erase_local_region_at_slide(self, click_point: Tuple[float, float], radius: float) -> None:
        """Remove patches within the brush from any labeled region (slide coords)."""
        if self.graphics_view.coords is None:
            return
        scale = self.graphics_view.transform().m11()
        cursor_diam = radius * 2 * scale
        if cursor_diam > 128:
            eff_r = 64.0 / scale
        elif cursor_diam < 16:
            eff_r = 8.0 / scale
        else:
            eff_r = radius
        patch_centers = self.graphics_view.coords + self.graphics_view.patch_size / 2.0
        diffs = patch_centers - np.array(click_point)
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        erased = set(int(i) for i in np.where(dists_sq <= eff_r ** 2)[0])
        self._apply_erase_to_regions(erased)

    def _erase_local_region_at_scatter(self, click_point: Tuple[float, float], radius: float) -> None:
        """Remove patches within the brush from any labeled region (scatter coords)."""
        if self._current_embedding is None:
            return
        from utils import normalize_to_scene
        scatter_coords = normalize_to_scene(self._current_embedding, 400, 400, 20)
        scale = self.scatter_view.transform().m11()
        cursor_diam = radius * 2 * scale
        if cursor_diam > 128:
            eff_r = 64.0 / scale
        elif cursor_diam < 16:
            eff_r = 8.0 / scale
        else:
            eff_r = radius
        diffs = scatter_coords - np.array(click_point)
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        erased = set(int(i) for i in np.where(dists_sq <= eff_r ** 2)[0])
        self._apply_erase_to_regions(erased)

    def _apply_erase_to_regions(self, erased_indices: Set[int]) -> None:
        """Remove patch indices from all regions; delete empty ones; convert partial KMEANS to LOCAL."""
        if not erased_indices:
            return
        to_delete: List[int] = []
        changed = False
        for rid, region in list(self._labeled_regions.items()):
            overlap = region.patch_indices & erased_indices
            if not overlap:
                continue
            changed = True
            region.patch_indices -= overlap
            if not region.patch_indices:
                to_delete.append(rid)
            else:
                # Partial erase: KMEANS region loses its cluster-wide status → LOCAL
                if region.source_mode == SourceMode.KMEANS:
                    region.source_mode = SourceMode.LOCAL
                    self._selected_clusters.discard(region.kmeans_cluster)
                    self.labeled_regions_widget.update_region_source(rid, SourceMode.LOCAL)
                    # Register in local_region_widget (wasn't there before)
                    self.local_region_widget.add_region(
                        rid, len(region.patch_indices), region.color, region.name
                    )
                # Update count display
                self.labeled_regions_widget.update_region(rid, len(region.patch_indices))
                if region.source_mode == SourceMode.LOCAL:
                    self.local_region_widget.update_region(rid, len(region.patch_indices), region.name)
        # Delete empty regions
        for rid in to_delete:
            region = self._labeled_regions.pop(rid, None)
            if region is None:
                continue
            if region.source_mode == SourceMode.LOCAL:
                self.local_region_widget.remove_region(rid)
            self.labeled_regions_widget.remove_region(rid)
        if changed:
            self._apply_all_labeled_region_styles()
            self._update_labeled_export_action()
            self._sync_app_state_regions()

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
                cluster = self._labeled_regions.get(region_id)
                if cluster is None:
                    continue
                merged_indices.update(cluster.patch_indices)
                merged_radius = max(merged_radius, cluster.radius)

            primary_cluster = self._labeled_regions.get(primary_id)
            if primary_cluster is None:
                return

            primary_cluster.patch_indices = merged_indices
            primary_cluster.center_point = self._compute_local_region_center(merged_indices)
            primary_cluster.radius = merged_radius

            self.local_region_widget.update_region(
                primary_id, len(merged_indices), primary_cluster.name
            )
            self.labeled_regions_widget.update_region(primary_id, len(merged_indices))

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
                cluster = self._labeled_regions.get(region_id)
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
                if region_id in self._labeled_regions:
                    del self._labeled_regions[region_id]
                self.local_region_widget.remove_region(region_id)
                self.labeled_regions_widget.remove_region(region_id)
            self._sync_app_state_regions()

            return

        cluster_id = self._next_region_id
        self._next_region_id += 1

        # Use K-means cluster color directly (no modification)
        if self.graphics_view.cluster_colors and kmeans_cluster < len(self.graphics_view.cluster_colors):
            color = self.graphics_view.cluster_colors[kmeans_cluster]
        else:
            color = QColor(128, 128, 128)  # Fallback gray

        slide_name = self.slide_combo.currentText() or None

        # Create region as LabeledRegion
        region = LabeledRegion(
            region_id=cluster_id,
            name=f"Region {cluster_id}",
            color=color,
            patch_indices=patch_indices,
            source_mode=SourceMode.LOCAL,
            kmeans_cluster=kmeans_cluster,
            center_point=center_point,
            radius=radius,
            slide_name=slide_name,
        )
        self._labeled_regions[cluster_id] = region
        self._last_labeled_slide = slide_name

        # Update UI
        self.local_region_widget.add_region(
            cluster_id, len(patch_indices), color, region.name
        )
        self.labeled_regions_widget.add_region(region)
        self._update_labeled_export_action()
        self._sync_app_state_regions()

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
        """Handle deletion of a local region (triggered from LocalRegionWidget).

        Parameters
        ----------
        region_id : int
            The region ID to delete.
        """
        if region_id in self._labeled_regions:
            del self._labeled_regions[region_id]
            self.local_region_widget.remove_region(region_id)
            self.labeled_regions_widget.remove_region(region_id)
            self._apply_local_region_cluster_styles()
            self._update_labeled_export_action()
            self._sync_app_state_regions()
            print(f"DEBUG: Deleted local region cluster {region_id}")


    def _on_labeled_region_deleted_from_panel(self, region_id: int) -> None:
        """Handle deletion triggered from the unified Labeled Regions panel.

        Parameters
        ----------
        region_id : int
            The region ID to delete.
        """
        region = self._labeled_regions.pop(region_id, None)
        if region is None:
            return
        if region.source_mode == SourceMode.LOCAL:
            self.local_region_widget.remove_region(region_id)
        elif region.source_mode == SourceMode.KMEANS:
            self._selected_clusters.discard(region.kmeans_cluster)
        self.labeled_regions_widget.remove_region(region_id)
        self._apply_all_labeled_region_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        print(f"DEBUG: Deleted labeled region {region_id} from panel")


    def _clear_local_region_clusters(self) -> None:
        """Clear all local region clusters."""
        local_ids = [rid for rid, r in self._labeled_regions.items()
                     if r.source_mode == SourceMode.LOCAL]
        for rid in local_ids:
            del self._labeled_regions[rid]
            self.labeled_regions_widget.remove_region(rid)
        self.local_region_widget.clear_regions()
        self._apply_local_region_cluster_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        print("DEBUG: Cleared all local region clusters")


    def _clear_all_labeled_regions(self) -> None:
        """Clear all labeled regions from all modes."""
        self._labeled_regions.clear()
        self._next_region_id = 0
        self._selected_clusters.clear()
        self.labeled_regions_widget.clear_regions()
        self.local_region_widget.clear_regions()
        self._apply_all_labeled_region_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        print("DEBUG: Cleared all labeled regions")

    # -------------------------------------------------------------------------
    # Unified labeled region style application
    # -------------------------------------------------------------------------


    def _update_labeled_export_action(self) -> None:
        """Enable/disable the Export Labeled Regions menu action."""
        if hasattr(self, 'export_labeled_action'):
            self.export_labeled_action.setEnabled(bool(self._labeled_regions))


    def _apply_all_labeled_region_styles(self) -> None:
        """Apply styles for all labeled regions across all modes.

        Hides all patches first, then illuminates patches belonging to any
        labeled region with their region color.
        """
        if not self.graphics_view.rect_items:
            return

        # Reset all patches to invisible
        for rect in self.graphics_view.rect_items:
            rect.setOpacity(0.0)

        # Illuminate patches for each labeled region
        for region in self._labeled_regions.values():
            for idx in region.patch_indices:
                if 0 <= idx < len(self.graphics_view.rect_items):
                    rect = self.graphics_view.rect_items[idx]
                    rect.setBrush(QBrush(region.color))
                    rect.setOpacity(0.6)

        self._apply_all_labeled_region_scatter_styles()


    def _apply_all_labeled_region_scatter_styles(self) -> None:
        """Apply scatter point styles for all labeled regions."""
        if not self.scatter_view or not self.scatter_view._scatter_items:
            return

        all_labeled_indices: Set[int] = set()
        for region in self._labeled_regions.values():
            all_labeled_indices.update(region.patch_indices)

        for item in self.scatter_view._scatter_items:
            if item.index in all_labeled_indices:
                item.setOpacity(1.0)
            else:
                item.setOpacity(0.2)

    # -------------------------------------------------------------------------
    # K-means labeled region helpers
    # -------------------------------------------------------------------------


    def _create_kmeans_labeled_region(self, cluster: int) -> bool:
        """Create a LabeledRegion for a K-means cluster immediately.

        Returns True if the region was created, False if it already exists or
        the cluster has no patches.
        """
        if self.graphics_view.labels is None:
            return False
        # Duplicate check
        for region in self._labeled_regions.values():
            if region.source_mode == SourceMode.KMEANS and region.kmeans_cluster == cluster:
                self._show_toast("Region already exists")
                return False
        indices = set(int(i) for i in np.where(self.graphics_view.labels == cluster)[0])
        if not indices:
            return False
        color = (self.graphics_view.cluster_colors[cluster]
                 if self.graphics_view.cluster_colors and cluster < len(self.graphics_view.cluster_colors)
                 else QColor(128, 128, 128))
        region_id = self._next_region_id
        self._next_region_id += 1
        region = LabeledRegion(
            region_id=region_id,
            name=f"Cluster {cluster}",
            color=color,
            patch_indices=indices,
            source_mode=SourceMode.KMEANS,
            kmeans_cluster=cluster,
            slide_name=self.slide_combo.currentText() or None,
        )
        self._labeled_regions[region_id] = region
        self._last_labeled_slide = region.slide_name
        self._selected_clusters.add(cluster)
        self.labeled_regions_widget.add_region(region)
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        print(f"DEBUG: Created K-means labeled region {region_id} for cluster {cluster}")
        return True


    def _delete_kmeans_labeled_region(self, cluster: int) -> None:
        """Delete the LabeledRegion for the given K-means cluster, if it exists."""
        for region_id, region in list(self._labeled_regions.items()):
            if region.source_mode == SourceMode.KMEANS and region.kmeans_cluster == cluster:
                del self._labeled_regions[region_id]
                self._selected_clusters.discard(cluster)
                self.labeled_regions_widget.remove_region(region_id)
                self._update_labeled_export_action()
                self._apply_all_labeled_region_styles()
                self._sync_app_state_regions()
                print(f"DEBUG: Deleted K-means labeled region {region_id} for cluster {cluster}")
                break


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


    def _export_labeled_regions(self) -> None:
        """Export all labeled regions (all modes) to a single GeoJSON file."""
        if not self._labeled_regions:
            QMessageBox.warning(self, "No Regions", "No labeled regions to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Labeled Regions as GeoJSON",
            "",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return

        self._write_labeled_regions_geojson(file_path)


    def _write_labeled_regions_geojson(self, file_path: str) -> None:
        """Write all labeled regions to a GeoJSON file.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        """
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

        for region in self._labeled_regions.values():
            boxes = []
            for idx in region.patch_indices:
                if 0 <= idx < len(coords):
                    x, y = coords[idx]
                    boxes.append(box(x, y, x + patch_size, y + patch_size))

            if boxes:
                merged = unary_union(boxes)
                color_hex = region.color.name()
                feature = {
                    "type": "Feature",
                    "geometry": merged.__geo_interface__,
                    "properties": {
                        "classification": {
                            "name": region.name,
                            "colorRGB": int(color_hex.replace('#', ''), 16)
                        },
                        "region_id": region.region_id,
                        "patch_count": len(region.patch_indices),
                        "kmeans_cluster": region.kmeans_cluster,
                        "source_mode": region.source_mode.value,
                    }
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(file_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"DEBUG: Exported {len(features)} labeled regions to {file_path}")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {len(features)} labeled regions to:\n{file_path}"
        )

    # -------------------------------------------------------------------------
    # Cross-Slide Atlas Methods
    # -------------------------------------------------------------------------


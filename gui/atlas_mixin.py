"""
atlas_mixin.py
==============
AtlasMixin mixin for MainWindow.
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


class AtlasMixin:

    def _get_atlas_entry_for_current_slide(self):
        """Return the SlideAtlasEntry for the currently displayed slide, or None.

        Returns None if:
        - No atlas has been built
        - The current slide was not added to the atlas
        - The patch count in the atlas entry doesn't match the loaded slide
          (can happen when the atlas was built from a subsampled version)
        """
        if self._cluster_atlas is None:
            return None
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return None
        entry = self._cluster_atlas.entries.get(slide_name)
        if entry is None:
            return None
        # Verify patch count matches so indices are consistent
        if self._current_labels is not None:
            if len(entry.global_labels) != len(self._current_labels):
                return None
        return entry


    def _is_atlas_active(self) -> bool:
        """Return True when atlas groups should supersede local K-means groups.

        True when:
        - An atlas has been built
        - The K-means tab (index 0) is NOT selected
        - The current slide is present in the atlas with a matching patch count
        """
        return (
            self._cluster_atlas is not None
            and self.sidebar_tabs.currentIndex() != 0
            and self._get_atlas_entry_for_current_slide() is not None
        )


    def _apply_atlas_to_views(self) -> None:
        """Push atlas labels and atlas PCA coords into the slide and scatter views."""
        entry = self._get_atlas_entry_for_current_slide()
        if entry is None or self._cluster_atlas is None:
            return
        atlas_labels = entry.global_labels
        atlas_colours = _qcolors_to_hsl_strings(self._cluster_atlas.cluster_colors)
        self.graphics_view.update_labels_and_colours(atlas_labels, atlas_colours)
        # Retrieve the PCA coords for this slide's patches from the global array
        pca_coords = self._cluster_atlas.global_pca_coords[entry.local_to_global]
        self.scatter_view.populate(pca_coords, atlas_labels, atlas_colours)


    def _restore_local_to_views(self) -> None:
        """Restore local K-means labels and PCA coords to the slide and scatter views."""
        if self._current_labels is None or self._current_colours is None:
            return
        self.graphics_view.update_labels_and_colours(
            self._current_labels, self._current_colours)
        if self._current_embedding is not None:
            self.scatter_view.populate(
                self._current_embedding, self._current_labels, self._current_colours)


    def _on_sidebar_tab_changed(self, index: int) -> None:
        """Handle sidebar tab change to switch selection modes."""
        if index == 0:  # Local Region tab — mode driven by Full Slide checkbox
            self._restore_local_to_views()
            self._apply_all_labeled_region_styles()
        else:  # Atlas tab (index 1) and any future tabs
            if self._is_atlas_active():
                self._apply_atlas_to_views()
        # Paint any KMEANS-mode labeled regions on top when on Atlas tab
        if index != 0 and self._labeled_regions and self.graphics_view.rect_items:
            for _region in self._labeled_regions.values():
                if _region.source_mode == SourceMode.KMEANS:
                    for _idx in _region.patch_indices:
                        if 0 <= _idx < len(self.graphics_view.rect_items):
                            _rect = self.graphics_view.rect_items[_idx]
                            _rect.setBrush(QBrush(_region.color))
                            _rect.setOpacity(0.6)


    def _on_atlas_thumbnail_clicked(self, slide_name: str) -> None:
        """Placeholder for future: handle click on atlas thumbnail.

        Parameters
        ----------
        slide_name : str
            Name of the slide whose thumbnail was clicked.
        """
        pass  # Reserved for future functionality


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
        number = self.slide_thumbnail_list.get_slide_number(slide_name)
        alias = f"#{number}" if number is not None else None
        self.atlas_slide_list.add_slide(
            slide_name,
            self._current_features.copy(),
            self._current_coords_lv0.copy(),
            h5_path,
            color,
            alias=alias
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

            # Populate Qt-free atlas state for MCP tools
            try:
                from app_state import AtlasState  # type: ignore
                import app_state as _app_state
                _app_state.update(
                    atlas_state=AtlasState(
                        global_labels=self._cluster_atlas.global_labels.copy(),
                        slide_indices=self._cluster_atlas.slide_indices.copy(),
                        slide_names=list(self._cluster_atlas.slide_names),
                        n_clusters=self._cluster_atlas.n_clusters,
                    )
                )
            except Exception:
                pass

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

            # Populate and show the atlas thumbnail panel
            self.atlas_thumbnail_panel.populate(
                self._cluster_atlas, self.slides,
                slide_numbers=self.slide_thumbnail_list._slide_numbers
            )
            self.atlas_thumbnail_panel.setVisible(True)

            self.atlas_progress.setValue(100)

            # Update info label with atlas statistics
            self.atlas_info_label.setText(
                f"Atlas built: {len(self._cluster_atlas.slide_names)} slides, "
                f"{len(self._cluster_atlas.global_labels):,} patches, "
                f"{self._cluster_atlas.n_clusters} clusters"
            )

            print(f"DEBUG: Atlas built successfully with {self._cluster_atlas.n_clusters} clusters")

            # If the user is not on the K-means tab, immediately apply atlas labels
            if self.sidebar_tabs.currentIndex() != 0:
                self._apply_atlas_to_views()

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

        # Clear and hide the atlas thumbnail panel
        self.atlas_thumbnail_panel.clear()
        self.atlas_thumbnail_panel.setVisible(False)

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

        # Cascade animation on the current slide using atlas cluster membership.
        # graphics_view.labels is already atlas labels when atlas is active, so
        # the existing cascade machinery works without modification.
        entry = self._get_atlas_entry_for_current_slide()
        if entry is not None and self.graphics_view.coords is not None:
            cluster_indices = np.where(entry.global_labels == cluster_id)[0]
            if cluster_indices.size > 0:
                idx0 = int(cluster_indices[0])
                x, y = (self.graphics_view.coords[idx0]
                        + self.graphics_view.patch_size / 2.0)
                self._set_selected_clusters({cluster_id})
                self._prepare_scatter_for_cascade(cluster_id)
                self._start_slide_cascade(cluster_id, (float(x), float(y)))

        # Highlight on all atlas thumbnails (no animation, instant)
        self.atlas_thumbnail_panel.highlight_cluster(cluster_id, self._selected_clusters)


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

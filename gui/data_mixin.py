"""
data_mixin.py
=============
DataMixin mixin for MainWindow.
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


class DataMixin:

    def _select_folder(self) -> None:
        """Prompt the user to select a root directory and parse its contents."""
        directory = QFileDialog.getExistingDirectory(self, "Select root folder", os.getcwd())
        if not directory:
            print("DEBUG: No directory selected")
            return
        print(f"DEBUG: Selected directory: {directory}")
        try:
            print("DEBUG: Attempting to parse directory...")
            self.slides = data_loader.parse_root_directory(directory)
            self._root_dir = directory
            print(f"DEBUG: Found {len(self.slides)} slides: {list(self.slides.keys())}")
            
            # Populate slide combo
            self.slide_combo.clear()
            slide_names = sorted(self.slides.keys())
            print(f"DEBUG: Adding slides to combo box: {slide_names}")
            self.slide_combo.addItems(slide_names)
            self.slide_thumbnail_list.set_slides(slide_names, self.slides)
            
            # Clear subsequent selectors and view
            self._model_selection = None
            self._update_model_selection_label()
            self._set_model_dependent_ui_enabled(False)
            self.model_select_btn.setEnabled(bool(slide_names))
            self.model_selector.clear()
            self.mag_combo.clear()
            self.patch_combo.clear()
            self.graphics_view.scene().clear()
            if hasattr(self.graphics_view, 'rect_items'):
                self.graphics_view.rect_items.clear()
            self._clear_scatter_view()
            
        except Exception as e:
            print(f"DEBUG: Error parsing directory: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to parse folder:\n{str(e)}")
            return


    def _on_thumbnail_slide_selected(self, slide_name: str) -> None:
        """Handle selection from the thumbnail list."""
        if slide_name and slide_name != self.slide_combo.currentText():
            self.slide_combo.setCurrentText(slide_name)


    def _open_model_selection_dialog(self) -> None:
        """Open the model selection dialog for the current slide."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            QMessageBox.warning(self, "Select Slide", "Please select a slide first.")
            return
        info = self.slides.get(slide_name)
        if info is None:
            QMessageBox.warning(self, "Select Slide", "Slide information is unavailable.")
            return

        dialog = ModelSelectionDialog(info, self._model_selection, self)
        if dialog.exec():
            selection = dialog.selection()
            if selection is None:
                QMessageBox.warning(self, "Model Selection", "Select models, magnification, and patch size.")
                return
            self._model_selection = selection
            if not self._apply_model_selection(selection):
                self._model_selection = None
                self._update_model_selection_label()
                self._set_model_dependent_ui_enabled(False)
                return
            self._update_model_selection_label()
            self._set_model_dependent_ui_enabled(True)
            self._load_current_data()


    def _on_slide_changed(self, slide_name: str) -> None:
        """Load slide preview and apply any active model selection."""
        print(f"DEBUG: Slide changed to: {slide_name}")
        if not slide_name:
            print("DEBUG: No slide name provided")
            self.model_select_btn.setEnabled(False)
            return
        self.slide_thumbnail_list.set_current_slide(slide_name)
        info = self.slides.get(slide_name)
        if info is None:
            print(f"DEBUG: No slide info found for {slide_name}")
            self.model_select_btn.setEnabled(False)
            return

        self.model_select_btn.setEnabled(True)
        self._load_slide_image_only(slide_name)

        if self._model_selection:
            if self._apply_model_selection(self._model_selection):
                self._set_model_dependent_ui_enabled(True)
                self._load_current_data()
            else:
                self._model_selection = None
                self._update_model_selection_label()
                self._set_model_dependent_ui_enabled(False)
        else:
            self._set_model_dependent_ui_enabled(False)


    def _populate_model_selector(self, info: data_loader.SlideInfo) -> None:
        """Populate the hidden model selector for the current slide."""
        model_names = sorted(info.models.keys())
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(model_names)
        self.model_selector.blockSignals(False)


    def _apply_model_selection(self, selection: ModelSelection) -> bool:
        """Apply a stored model selection to the current slide."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return False
        info = self.slides.get(slide_name)
        if info is None:
            return False

        self._populate_model_selector(info)
        available_models = set(info.models.keys())
        missing = [model for model in selection.models if model not in available_models]
        if missing:
            QMessageBox.warning(
                self,
                "Models Unavailable",
                f"Selected models not available for {slide_name}: {', '.join(missing)}",
            )
            return False

        magnifications = self._get_common_magnifications(selection.models, info)
        if selection.magnification not in magnifications:
            QMessageBox.warning(
                self,
                "Magnification Unavailable",
                f"Magnification {selection.magnification} not available for selected models.",
            )
            return False

        patches = self._get_common_patches(selection.models, selection.magnification, info)
        if selection.patch_size not in patches:
            QMessageBox.warning(
                self,
                "Patch Size Unavailable",
                f"Patch size {selection.patch_size} not available for selected models.",
            )
            return False

        self.model_selector.blockSignals(True)
        self.model_selector.setSelectedModels(selection.models)
        self.model_selector.blockSignals(False)

        self.mag_combo.blockSignals(True)
        self.mag_combo.clear()
        self.mag_combo.addItems(magnifications)
        self.mag_combo.setCurrentText(selection.magnification)
        self.mag_combo.blockSignals(False)

        self.patch_combo.blockSignals(True)
        self.patch_combo.clear()
        self.patch_combo.addItems(patches)
        self.patch_combo.setCurrentText(selection.patch_size)
        self.patch_combo.blockSignals(False)

        self._update_model_selection_label()
        return True


    def _get_common_magnifications(
        self,
        models: List[str],
        info: data_loader.SlideInfo,
    ) -> List[str]:
        mag_sets = []
        for model_name in models:
            model_dict = info.models.get(model_name, {})
            mag_sets.append(set(model_dict.keys()))
        if not mag_sets:
            return []
        return sorted(set.intersection(*mag_sets))


    def _get_common_patches(
        self,
        models: List[str],
        magnification: str,
        info: data_loader.SlideInfo,
    ) -> List[str]:
        if not models or not magnification:
            return []
        patch_sets = []
        for model_name in models:
            patches = info.models.get(model_name, {}).get(magnification, {})
            patch_sets.append(set(patches.keys()))
        if not patch_sets:
            return []
        return sorted(set.intersection(*patch_sets))


    def _update_model_selection_label(self) -> None:
        """Update the summary label for the current model selection."""
        if self._model_selection is None:
            self.model_selection_label.setText("No model selected")
            self.model_selection_label.setToolTip("No model selected")
            return
        model_str = ", ".join(self._model_selection.models)
        summary = (
            f"Model: {model_str} | "
            f"{self._model_selection.magnification} | "
            f"{self._model_selection.patch_size}"
        )
        self.model_selection_label.setText(summary)
        self.model_selection_label.setToolTip(summary)


    def _load_slide_image_only(self, slide_name: str) -> None:
        """Load a slide preview without clustering."""
        info = self.slides.get(slide_name)
        if info is None:
            return

        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        adaptive_enabled = self.adaptive_zoom_action.isChecked()
        self.graphics_view.set_adaptive_mode(adaptive_enabled)
        use_adaptive = adaptive_enabled and data_loader.is_openslide_available()

        if use_adaptive:
            try:
                from openslide import OpenSlide  # type: ignore

                slide = OpenSlide(info.image_path)
                slide_dimensions = slide.dimensions
                slide.close()
                empty_coords = np.zeros((0, 2), dtype=float)
                empty_labels = np.zeros((0,), dtype=int)
                adaptive_loaded = self.graphics_view.load_slide_adaptive(
                    info.image_path,
                    empty_coords,
                    0.0,
                    empty_labels,
                    [],
                    slide_dimensions,
                )
                if adaptive_loaded:
                    self._clear_scatter_view()
                    self._current_embedding = None
                    self._current_features = None
                    self._current_coords_thumb = None
                    self._current_coords_lv0 = None
                    self._current_patch_size_thumb = None
                    self._current_patch_size_lv0 = None
                    self._coord_scale_factor = None
                    self._current_labels = None
                    self._clear_selected_clusters()
                    self.progress_bar.setVisible(False)
                    self.status_label.setText("Preview only — select a model to enable clustering.")
                    return
            except Exception as exc:
                print(f"DEBUG: Adaptive preview failed: {exc}")

        try:
            thumb_image = data_loader.load_thumbnail(info.image_path)
        except Exception as exc:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load slide thumbnail:\n{exc}")
            return

        empty_coords = np.zeros((0, 2), dtype=float)
        empty_labels = np.zeros((0,), dtype=int)
        self.graphics_view.load_slide(thumb_image, empty_coords, 0.0, empty_labels, [])
        self._clear_scatter_view()

        self._current_embedding = None
        self._current_features = None
        self._current_coords_thumb = None
        self._current_coords_lv0 = None
        self._current_patch_size_thumb = None
        self._current_patch_size_lv0 = None
        self._coord_scale_factor = None
        self._current_labels = None
        self._clear_selected_clusters()
        self.status_label.setText("Select a model to enable clustering.")

        self.progress_bar.setVisible(False)


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
        if self._model_selection is None:
            return
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
        if self._model_selection is None:
            return
        # Validate model compatibility before loading
        self._validate_model_compatibility()
        self._load_current_data()


    def _on_cluster_changed(self, k: int) -> None:
        """Recompute clusters when the cluster count slider changes."""
        if self._model_selection is None:
            return
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
            if self._model_selection is None:
                self._load_slide_image_only(self.slide_combo.currentText())
            else:
                self._load_current_data()


    def _load_current_data(self, recluster_only: bool = False) -> None:
        """Load or recompute data based on current selections.

        Parameters
        ----------
        recluster_only : bool, optional
            If True, only recompute clusters for existing data (e.g.,
            when the cluster count changes) without reloading images
            or coordinates.  Defaults to False.
        """
        # Preserve selection so it can be restored after atlas labels are applied
        _prev_selected = set(self._selected_clusters)
        # Reset selected clusters and disable export action when loading new data
        self._clear_selected_clusters()
        self._animation_in_progress = False
        if self.scatter_view:
            self.scatter_view.set_animation_active(False)
        
        if self._model_selection is None:
            print("DEBUG: No model selection; skipping data load")
            return

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

        # --- Slide-switch safety: warn if labeled regions exist for a different slide ---
        if (not recluster_only
                and self._labeled_regions
                and self._last_labeled_slide is not None
                and self._last_labeled_slide != slide_name):
            ans = QMessageBox.question(
                self, "Switch Slide",
                "Switching slides will clear all labeled regions. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            self._clear_all_labeled_regions()

        # --- Recluster safety: warn if K-means labeled regions exist ---
        if recluster_only:
            kmeans_regions = [r for r in self._labeled_regions.values()
                              if r.source_mode == SourceMode.KMEANS]
            if kmeans_regions:
                ans = QMessageBox.question(
                    self, "Change K",
                    f"Changing the cluster count will invalidate "
                    f"{len(kmeans_regions)} K-means labeled region(s). Remove them?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    for region in kmeans_regions:
                        del self._labeled_regions[region.region_id]
                        self.labeled_regions_widget.remove_region(region.region_id)
                    self._update_labeled_export_action()
                    self._sync_app_state_regions()

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
                self._current_pca = pca
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
                raw_mpp = slide.properties.get("openslide.mpp-x")
                self._current_mpp = float(raw_mpp) if raw_mpp is not None else None
                slide.close()
                print(f"DEBUG: OpenSlide returned dimensions {slide_w}x{slide_h}")
            except Exception:
                # Fallback: infer from coordinate grid
                self._current_mpp = None
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

            # Compute and store centroids
            self._cluster_centroids = self._compute_centroids(features, labels)
            self._current_labels = labels
            self._current_colours = colours
            self.cluster_legend.clear_cluster_names()
            self.cluster_legend.update_clusters(labels, colours)

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
            self._compute_and_populate_region_view(
                self._current_features, self._current_coords_thumb, labels, colours
            )
            print("DEBUG: Data loading complete; views updated")
            # If atlas is active for this slide, apply atlas labels/colors over local ones
            if self._is_atlas_active():
                self._apply_atlas_to_views()
                # Re-apply cluster highlight if a selection was active before the slide switch
                if _prev_selected:
                    self._set_selected_clusters(_prev_selected)
                    self._apply_selected_cluster_styles()
                    self.atlas_thumbnail_panel.highlight_cluster(
                        next(iter(_prev_selected)), _prev_selected
                    )
            # Hide progress bar when done
            self.progress_bar.setVisible(False)
            self._sync_app_state()
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

            # Compute and store centroids
            self._cluster_centroids = self._compute_centroids(features, labels)
            self._current_labels = labels
            self._current_colours = colours
            self.cluster_legend.clear_cluster_names()
            self.cluster_legend.update_clusters(labels, colours)

            # Update both views with new labels and colours
            self.graphics_view.update_labels_and_colours(labels, colours)
            self.scatter_view.populate(self._current_embedding, labels, colours)
            self._compute_and_populate_region_view(
                self._current_features, self._current_coords_thumb, labels, colours
            )
            # If atlas is active for this slide, re-apply atlas overlay
            if self._is_atlas_active():
                self._apply_atlas_to_views()
                # Re-apply cluster highlight if a selection was active before the slide switch
                if _prev_selected:
                    self._set_selected_clusters(_prev_selected)
                    self._apply_selected_cluster_styles()
                    self.atlas_thumbnail_panel.highlight_cluster(
                        next(iter(_prev_selected)), _prev_selected
                    )
            self._sync_app_state()


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

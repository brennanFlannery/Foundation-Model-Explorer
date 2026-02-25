"""
chat_mixin.py
=============
ChatMixin mixin for MainWindow.
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


class ChatMixin:

    def _sync_app_state(self) -> None:
        """Push current GUI state into the app_state registry for MCP tools."""
        pca_ratio = None
        if hasattr(self, "_current_pca") and self._current_pca is not None:
            pca_ratio = self._current_pca.explained_variance_ratio_.tolist()

        region_data = {
            rid: LabeledRegionData(
                region_id=r.region_id,
                name=r.name,
                color_hex=r.color.name(),
                patch_indices=list(r.patch_indices),
                source_mode=r.source_mode.name.lower(),
                kmeans_cluster=r.kmeans_cluster,
            )
            for rid, r in self._labeled_regions.items()
        }

        app_state.update(
            slide_name=self.slide_combo.currentText() or None,
            root_dir=self._root_dir,
            selected_models=self._model_selection.models if self._model_selection else None,
            magnification=self._model_selection.magnification if self._model_selection else None,
            patch_size=self._model_selection.patch_size if self._model_selection else None,
            features=getattr(self, "_current_features", None),
            coords_lv0=getattr(self, "_current_coords_lv0", None),
            coords_thumb=getattr(self, "_current_coords_thumb", None),
            patch_size_lv0=getattr(self, "_current_patch_size_lv0", None),
            cluster_labels=getattr(self, "_current_labels", None),
            cluster_centroids=getattr(self, "_cluster_centroids", None),
            cluster_colours=getattr(self, "_current_colours", None),
            embedding_2d=getattr(self, "_current_embedding", None),
            pca_explained_variance_ratio=pca_ratio,
            labeled_regions=region_data,
            selected_clusters=set(self._selected_clusters),
            mpp=getattr(self, "_current_mpp", None),
        )


    def _sync_app_state_regions(self) -> None:
        """Push only annotation state into the app_state registry."""
        region_data = {
            rid: LabeledRegionData(
                region_id=r.region_id,
                name=r.name,
                color_hex=r.color.name(),
                patch_indices=list(r.patch_indices),
                source_mode=r.source_mode.name.lower(),
                kmeans_cluster=r.kmeans_cluster,
            )
            for rid, r in self._labeled_regions.items()
        }
        app_state.update(
            labeled_regions=region_data,
            selected_clusters=set(self._selected_clusters),
        )
        # Push region catalog to chat dock for @ mention autocomplete
        self.chat_dock.update_region_catalog([
            {"name": r.name, "region_id": rid}
            for rid, r in self._labeled_regions.items()
        ])

    # -------------------------------------------------------------------------
    # GUI action queue — called by MCP tools from background thread
    # -------------------------------------------------------------------------


    def _drain_gui_action_queue(self) -> None:
        """Consume pending GUI actions posted by MCP tools (main-thread QTimer)."""
        for action in app_state.drain_gui_actions():
            action_id = action["action_id"]
            try:
                result = self._handle_gui_action(action["action_type"], action["params"])
            except Exception as exc:
                result = {"error": str(exc)}
            app_state.set_gui_action_result(action_id, result)


    def _handle_gui_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a GUI action to the appropriate handler."""
        if action_type == "label_cluster":
            return self._gui_action_label_cluster(params)
        if action_type == "create_agent_region":
            return self._gui_action_create_agent_region(params)
        if action_type == "delete_region":
            return self._gui_action_delete_region(params)
        if action_type == "rename_region":
            return self._gui_action_rename_region(params)
        if action_type == "navigate_to_region":
            return self._gui_action_navigate_to_region(params)
        if action_type == "select_cluster":
            return self._gui_action_select_cluster(params)
        if action_type == "clear_all_regions":
            self._clear_all_labeled_regions()
            return {"success": True}
        if action_type == "deselect_all_clusters":
            return self._gui_action_deselect_all_clusters()
        if action_type == "switch_to_atlas_view":
            return self._gui_action_switch_to_atlas_view(params)
        if action_type == "highlight_atlas_cluster":
            return self._gui_action_highlight_atlas_cluster(params)
        if action_type == "set_cluster_count":
            return self._gui_action_set_cluster_count(params)
        if action_type == "load_slide":
            return self._gui_action_load_slide(params)
        if action_type == "export_regions_geojson":
            return self._gui_action_export_regions_geojson(params)
        if action_type == "open_patch_exemplar_popup":
            return self._gui_action_open_patch_exemplar_popup(params)
        if action_type == "export_current_exemplar_popup":
            return self._gui_action_export_current_exemplar_popup(params)
        if action_type == "close_exemplar_popup":
            return self._gui_action_close_exemplar_popup(params)
        return {"error": f"Unknown action_type: {action_type!r}"}


    def _gui_action_label_cluster(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cluster_id = int(params["cluster_id"])
        name = params.get("name")
        if self._animation_in_progress:
            return {"success": False, "reason": "animation_in_progress"}
        created = self._create_kmeans_labeled_region(cluster_id)
        if not created:
            for rid, r in self._labeled_regions.items():
                if r.source_mode == SourceMode.KMEANS and r.kmeans_cluster == cluster_id:
                    return {"success": False, "reason": "already_labeled", "region_id": rid}
            return {"success": False, "reason": "cluster_empty_or_unknown"}
        region_id = None
        for rid, r in self._labeled_regions.items():
            if r.source_mode == SourceMode.KMEANS and r.kmeans_cluster == cluster_id:
                region_id = rid
                if name:
                    r.name = name
                break
        if name and region_id is not None:
            self.labeled_regions_widget.update_region(
                region_id, len(self._labeled_regions[region_id].patch_indices), name
            )
            self._sync_app_state_regions()
        self._prepare_scatter_for_cascade(cluster_id)
        self._start_slide_cascade(cluster_id, self._get_cluster_screen_center(cluster_id))
        return {
            "success": True,
            "region_id": region_id,
            "patch_count": len(self._labeled_regions[region_id].patch_indices) if region_id is not None else 0,
        }


    def _gui_action_create_agent_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        patch_indices: Set[int] = set(int(i) for i in params["patch_indices"])
        name = params.get("name")
        select_dominant_cluster = bool(params.get("select_dominant_cluster", True))
        if not patch_indices or self._current_labels is None:
            return {"error": "No patches or no slide loaded"}
        labels_subset = self._current_labels[sorted(patch_indices)]
        counts = np.bincount(labels_subset.astype(int))
        kmeans_cluster = int(counts.argmax())
        region_id = self._next_region_id
        self._next_region_id += 1
        auto_name = name or f"Agent Region {region_id}"
        color = (
            QColor(self._current_colours[kmeans_cluster])
            if self._current_colours and kmeans_cluster < len(self._current_colours)
            else QColor("#888888")
        )
        region = LabeledRegion(
            region_id=region_id,
            name=auto_name,
            color=color,
            patch_indices=patch_indices,
            source_mode=SourceMode.LOCAL,
            kmeans_cluster=kmeans_cluster,
            slide_name=self.slide_combo.currentText() or None,
        )
        self._labeled_regions[region_id] = region
        self.labeled_regions_widget.add_region(region)
        if select_dominant_cluster:
            self._selected_clusters.add(kmeans_cluster)
        self._apply_all_labeled_region_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        return {
            "success": True,
            "region_id": region_id,
            "name": auto_name,
            "patch_count": len(patch_indices),
            "kmeans_cluster": kmeans_cluster,
            "select_dominant_cluster": select_dominant_cluster,
        }


    def _gui_action_delete_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        region_id = int(params["region_id"])
        if region_id not in self._labeled_regions:
            return {"error": f"Region {region_id} not found"}
        region = self._labeled_regions.pop(region_id)
        self.labeled_regions_widget.remove_region(region_id)
        if region.source_mode == SourceMode.KMEANS:
            self._selected_clusters.discard(region.kmeans_cluster)
        self._apply_all_labeled_region_styles()
        self._update_labeled_export_action()
        self._sync_app_state_regions()
        return {"success": True, "deleted_region_id": region_id}


    def _gui_action_rename_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        region_id = int(params["region_id"])
        new_name = str(params["new_name"])
        if region_id not in self._labeled_regions:
            return {"error": f"Region {region_id} not found"}
        old_name = self._labeled_regions[region_id].name
        self._labeled_regions[region_id].name = new_name
        self.labeled_regions_widget.update_region(
            region_id, len(self._labeled_regions[region_id].patch_indices), new_name
        )
        self._sync_app_state_regions()
        return {"success": True, "region_id": region_id, "old_name": old_name, "new_name": new_name}


    def _gui_action_navigate_to_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        region_id = int(params["region_id"])
        padding = float(params.get("padding_fraction", 0.15))
        if region_id not in self._labeled_regions:
            return {"error": f"Region {region_id} not found"}
        region = self._labeled_regions[region_id]
        if not region.patch_indices or self._current_coords_lv0 is None:
            return {"error": "No coordinates available"}
        idx = list(region.patch_indices)
        if self.graphics_view.adaptive_mode and self._current_coords_lv0 is not None:
            coords = self._current_coords_lv0[idx]
        else:
            coords = self.graphics_view.coords[idx]
        pad_sz = float(self.graphics_view.patch_size or 0)
        x_min = float(coords[:, 0].min())
        y_min = float(coords[:, 1].min())
        x_max = float(coords[:, 0].max()) + pad_sz
        y_max = float(coords[:, 1].max()) + pad_sz
        w, h = x_max - x_min, y_max - y_min
        margin_x, margin_y = w * padding, h * padding
        rect = QRectF(x_min - margin_x, y_min - margin_y, w + 2 * margin_x, h + 2 * margin_y)
        self.graphics_view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        return {
            "success": True,
            "region_id": region_id,
            "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        }


    def _gui_action_select_cluster(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cluster_id = int(params["cluster_id"])
        if self._current_labels is None:
            return {"error": "No slide loaded"}
        center = self._get_cluster_screen_center(cluster_id)
        self._update_scatter_for_cluster(cluster_id, center, ctrl_pressed=False)
        return {"success": True, "cluster_id": cluster_id}


    def _gui_action_deselect_all_clusters(self) -> Dict[str, Any]:
        self._selected_clusters.clear()
        self._apply_selected_cluster_styles()
        self._sync_app_state_regions()
        return {"success": True}


    def _gui_action_switch_to_atlas_view(self, params: Dict[str, Any]) -> Dict[str, Any]:
        del params
        if self._cluster_atlas is None:
            return {"error": "No atlas has been built yet"}
        self.sidebar_tabs.setCurrentIndex(1)
        return {"success": True}


    def _gui_action_highlight_atlas_cluster(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cluster_id = int(params["cluster_id"])
        if self._cluster_atlas is None:
            return {"error": "No atlas has been built yet"}
        self.atlas_thumbnail_panel.highlight_cluster(cluster_id, self._selected_clusters)
        if hasattr(self, "atlas_scatter_view") and self.atlas_scatter_view is not None:
            self.atlas_scatter_view.highlight_cluster(cluster_id)
        return {"success": True, "cluster_id": cluster_id}


    def _gui_action_set_cluster_count(self, params: Dict[str, Any]) -> Dict[str, Any]:
        k = int(params["k"])
        if k < self.cluster_spin.minimum() or k > self.cluster_spin.maximum():
            return {"error": f"k must be between {self.cluster_spin.minimum()} and {self.cluster_spin.maximum()}"}
        if self._model_selection is None:
            return {"error": "No model selection active — load a slide with a model first"}
        old_k = self.cluster_spin.value()
        self.cluster_spin.setValue(k)
        # If the value didn't actually change, the signal won't fire,
        # so trigger reclustering manually in that case.
        if old_k == k:
            self._load_current_data(recluster_only=True)
        return {"success": True, "previous_k": old_k, "new_k": k}


    def _gui_action_load_slide(self, params: Dict[str, Any]) -> Dict[str, Any]:
        slide_name = str(params["slide_name"])
        if not self.slides:
            return {"error": "No slides loaded — open a directory first"}
        if slide_name not in self.slides:
            available = sorted(self.slides.keys())
            return {"error": f"Slide '{slide_name}' not found. Available: {available}"}
        previous = self.slide_combo.currentText()
        self.slide_combo.setCurrentText(slide_name)
        return {"success": True, "previous_slide": previous, "loaded_slide": slide_name}


    def _gui_action_export_regions_geojson(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._labeled_regions:
            return {"error": "No labeled regions to export"}
        output_path = params.get("output_path")
        if not output_path:
            slide_name = self.slide_combo.currentText() or "regions"
            base = os.path.splitext(slide_name)[0]
            output_path = os.path.join(self._root_dir or ".", f"{base}_regions.geojson")

        if self._current_coords_lv0 is not None:
            coords = self._current_coords_lv0
            patch_size = self._current_patch_size_lv0
        else:
            coords = self.graphics_view.coords
            patch_size = self.graphics_view.patch_size

        if coords is None:
            return {"error": "No coordinate data available"}

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
                            "colorRGB": int(color_hex.replace('#', ''), 16),
                        },
                        "region_id": region.region_id,
                        "patch_count": len(region.patch_indices),
                        "kmeans_cluster": region.kmeans_cluster,
                        "source_mode": region.source_mode.value,
                    },
                }
                features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        return {"success": True, "file_path": output_path, "region_count": len(features)}


    def _gui_action_open_patch_exemplar_popup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        popup_id = str(params.get("popup_id") or f"exemplar-{uuid.uuid4()}")
        patch_indices = [int(i) for i in params.get("patch_indices", [])]
        include_metadata = bool(params.get("include_metadata", True))
        source_label = str(params.get("source_label") or "Exemplars")
        strategy = str(params.get("strategy") or "diverse")
        scores_by_patch = params.get("scores_by_patch", {}) or {}

        if not patch_indices:
            return {"error": "No patch indices provided"}
        if self._current_coords_lv0 is None or self._current_patch_size_lv0 is None:
            return {"error": "Level-0 coordinates are unavailable for exemplar rendering"}

        slide_path = self._current_slide_image_path()
        if not slide_path:
            return {"error": "Could not resolve current slide image path"}

        patch_size = int(max(1, round(float(self._current_patch_size_lv0))))
        target_size = 160
        exemplar_items: List[Dict[str, Any]] = []
        image_failures = 0

        backend = None
        backend_type = ""
        try:
            if data_loader.openslide is not None:
                backend = data_loader.openslide.OpenSlide(slide_path)  # type: ignore[attr-defined]
                backend_type = "openslide"
            else:
                backend = Image.open(slide_path)
                backend_type = "pil"
        except Exception as exc:
            return {"error": f"Failed to open slide image backend: {exc}"}

        for idx in patch_indices:
            pixmap = QPixmap()
            if idx < 0 or idx >= len(self._current_coords_lv0):
                image_failures += 1
                continue
            coords = self._current_coords_lv0[idx]
            x = int(coords[0])
            y = int(coords[1])
            try:
                if backend_type == "openslide":
                    tile = backend.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                else:
                    tile = backend.crop((x, y, x + patch_size, y + patch_size)).convert("RGB")
                if tile.size != (target_size, target_size):
                    tile = tile.resize((target_size, target_size), Image.BILINEAR)
                pixmap = QPixmap.fromImage(QImage(ImageQt(tile)))
            except Exception:
                image_failures += 1
                pixmap = QPixmap(target_size, target_size)
                pixmap.fill(QColor("#303030"))

            cluster_id = None
            if self._current_labels is not None and 0 <= idx < len(self._current_labels):
                cluster_id = int(self._current_labels[idx])
            exemplar_items.append(
                {
                    "patch_index": int(idx),
                    "cluster_id": cluster_id,
                    "coords_lv0": {"x": x, "y": y},
                    "score": float(scores_by_patch.get(str(idx), scores_by_patch.get(idx, 0.0))),
                    "pixmap": pixmap,
                }
            )

        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass

        if self._active_exemplar_popup is not None:
            self._active_exemplar_popup.close()
            self._active_exemplar_popup.deleteLater()

        popup = PatchExemplarPopup(
            popup_id=popup_id,
            source_label=source_label,
            strategy=strategy,
            parent=self,
        )
        popup.set_items(exemplar_items, include_metadata=include_metadata)
        popup.export_button().clicked.connect(
            lambda: self._gui_action_export_current_exemplar_popup({"popup_id": popup_id})
        )
        popup.finished.connect(lambda _result, pid=popup_id: self._on_exemplar_popup_closed(pid))
        popup.show()
        popup.raise_()

        self._active_exemplar_popup = popup
        self._active_exemplar_popup_id = popup_id
        self._active_exemplar_source_label = source_label

        warnings: List[str] = []
        if image_failures:
            warnings.append(f"{image_failures} exemplar patches used fallback placeholders")
        return {
            "success": True,
            "popup_id": popup_id,
            "shown_count": len(exemplar_items),
            "sample_indices": [int(item["patch_index"]) for item in exemplar_items],
            "warnings": warnings,
        }


    def _gui_action_export_current_exemplar_popup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        popup_id = str(params.get("popup_id") or "")
        if self._active_exemplar_popup is None or self._active_exemplar_popup_id != popup_id:
            return {"error": f"No active exemplar popup for popup_id='{popup_id}'"}

        requested_dir = params.get("output_dir")
        if requested_dir:
            output_dir = os.path.abspath(os.path.expanduser(str(requested_dir)))
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                self._root_dir or ".",
                "Reports",
                f"exemplars_{(self.slide_combo.currentText() or 'slide')}_{ts}",
            )
        os.makedirs(output_dir, exist_ok=True)

        image_paths: List[str] = []
        manifest_items: List[Dict[str, Any]] = []
        for item in self._active_exemplar_popup.items():
            idx = int(item.get("patch_index", -1))
            if idx < 0:
                continue
            path = os.path.join(output_dir, f"patch_{idx}.png")
            pix = item.get("pixmap")
            if isinstance(pix, QPixmap) and not pix.isNull():
                pix.save(path, "PNG")
                image_paths.append(path)
                manifest_items.append(
                    {
                        "patch_index": idx,
                        "cluster_id": item.get("cluster_id"),
                        "coords_lv0": item.get("coords_lv0"),
                        "score": item.get("score"),
                        "image_path": path,
                    }
                )

        manifest_path = None
        if bool(params.get("include_manifest", True)):
            manifest_path = os.path.join(output_dir, "manifest.json")
            payload = {
                "popup_id": popup_id,
                "slide_name": self.slide_combo.currentText(),
                "source_label": self._active_exemplar_source_label,
                "items": manifest_items,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        return {
            "success": True,
            "popup_id": popup_id,
            "output_dir": output_dir,
            "files": image_paths,
            "manifest_path": manifest_path,
            "count": len(image_paths),
        }


    def _gui_action_close_exemplar_popup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        popup_id = str(params.get("popup_id") or "")
        if self._active_exemplar_popup is None:
            return {"success": True, "closed": False}
        if popup_id and self._active_exemplar_popup_id != popup_id:
            return {"success": True, "closed": False, "reason": "popup_id_mismatch"}
        self._active_exemplar_popup.close()
        self._active_exemplar_popup.deleteLater()
        self._active_exemplar_popup = None
        self._active_exemplar_popup_id = None
        self._active_exemplar_source_label = ""
        return {"success": True, "closed": True}


    def _current_slide_image_path(self) -> Optional[str]:
        """Return the current slide image path if available."""
        slide_name = self.slide_combo.currentText()
        if not slide_name:
            return None
        info = self.slides.get(slide_name)
        if info is None:
            return None
        return info.image_path


    def _on_exemplar_popup_closed(self, popup_id: str) -> None:
        """Clear popup state when the exemplar dialog closes."""
        if self._active_exemplar_popup_id != popup_id:
            return
        self._active_exemplar_popup = None
        self._active_exemplar_popup_id = None
        self._active_exemplar_source_label = ""


    def _get_cluster_screen_center(self, cluster_id: int) -> Tuple[float, float]:
        """Return the mean scene position of all patches in a cluster."""
        if self._current_labels is None or self.graphics_view.coords is None:
            return (0.0, 0.0)
        mask = self._current_labels == cluster_id
        coords = self.graphics_view.coords[mask]
        if len(coords) == 0:
            return (0.0, 0.0)
        return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))


    def _get_distance_to_centroid(self, index: int) -> Optional[float]:
        """Get normalized distance from a patch to its cluster centroid.
        
        Returns distance as a percentage (0-100%) where 100% represents
        the furthest point within the same cluster.
        
        Parameters
        ----------
        index : int
            Patch index.
            
        Returns
        -------
        Optional[float]
            Normalized distance as percentage (0-100), or None if data unavailable.
        """
        if (self._current_features is None or 
            self._cluster_centroids is None or
            self._max_cluster_distances is None or
            self._current_labels is None or
            index < 0 or index >= len(self._current_labels)):
            return None
        
        cluster = int(self._current_labels[index])
        feature_vec = self._current_features[index]
        centroid = self._cluster_centroids[cluster]
        distance = np.linalg.norm(feature_vec - centroid)
        
        # Normalize by max distance in cluster
        max_dist = self._max_cluster_distances[cluster]
        if max_dist > 0:
            return (distance / max_dist) * 100.0  # Return as percentage
        return 0.0


    def _read_env_value(self, env_path: str, key: str) -> str:
        """Read a key from a simple .env file."""
        if not env_path:
            return ""
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    name, value = stripped.split("=", 1)
                    if name.strip() == key:
                        return value.strip().strip("\"").strip("'")
        except Exception:
            return ""
        return ""


    def _build_chat_config(self) -> Optional[ChatAgentConfig]:
        """Build chat agent config from persisted settings."""
        chat_enabled = self.settings.value("chat_enabled", True, type=bool)
        if not chat_enabled:
            return None

        default_secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        secrets_path = self.settings.value("chat_secrets_path", default_secrets_path, type=str).strip()
        if not secrets_path:
            secrets_path = default_secrets_path
        if not os.path.exists(secrets_path):
            raise FileNotFoundError(
                f"Secrets file not found: {secrets_path}. Create .env with OPENAI_API_KEY."
            )
        api_key = self._read_env_value(secrets_path, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY not found in secrets file: {secrets_path}"
            )

        model = self.settings.value("chat_model", "gpt-5-nano", type=str)
        llm_timeout = float(self.settings.value("chat_llm_timeout_s", 60, type=int))
        tool_timeout = float(self.settings.value("chat_tool_timeout_s", 30, type=int))

        return ChatAgentConfig(
            model=model,
            api_key=api_key,
            llm_timeout_s=llm_timeout,
            tool_timeout_s=tool_timeout,
        )


    def _teardown_chat_agent(self) -> None:
        """Stop chat worker thread and release resources."""
        if self._chat_worker is not None:
            try:
                self.chat_shutdown_requested.emit()
            except Exception:
                pass
            try:
                self.chat_submit_requested.disconnect(self._chat_worker.submit_user_message)
                self.chat_cancel_requested.disconnect(self._chat_worker.cancel_current)
                self.chat_shutdown_requested.disconnect(self._chat_worker.shutdown)
            except Exception:
                pass
            self._chat_worker.deleteLater()
        if self._chat_thread is not None:
            self._chat_thread.quit()
            self._chat_thread.wait(1000)
            self._chat_thread.deleteLater()
        self._chat_worker = None
        self._chat_thread = None


    def _setup_chat_agent(self) -> None:
        """Initialize chat worker in dedicated thread using current settings."""
        self._teardown_chat_agent()
        try:
            config = self._build_chat_config()
        except Exception as exc:
            if hasattr(self, "chat_dock"):
                self.chat_dock.set_status_text(str(exc))
                self.chat_dock.set_busy(False)
            return
        if config is None:
            if hasattr(self, "chat_dock"):
                self.chat_dock.set_status_text("Configure API key in Preferences to enable chat.")
                self.chat_dock.set_busy(False)
            return

        self._chat_thread = QThread()
        self._chat_worker = ChatAgentWorker(config)
        self._chat_worker.moveToThread(self._chat_thread)

        self.chat_submit_requested.connect(self._chat_worker.submit_user_message)
        self.chat_cancel_requested.connect(self._chat_worker.cancel_current)
        self.chat_shutdown_requested.connect(self._chat_worker.shutdown)

        self._chat_worker.response_started.connect(self._on_chat_response_started)
        self._chat_worker.response_delta.connect(self._on_chat_response_delta)
        self._chat_worker.tool_finished.connect(self._on_chat_tool_finished)
        self._chat_worker.response_finished.connect(self._on_chat_response_finished)
        self._chat_worker.error_emitted.connect(self._on_chat_error_emitted)
        self._chat_worker.state_changed.connect(self._on_chat_state_changed)

        self._chat_thread.start()
        if hasattr(self, "chat_dock"):
            self.chat_dock.set_status_text("Ready")
            self.chat_dock.set_busy(False)


    def _on_chat_send_requested(self, text: str, metadata: Optional[Dict[str, object]] = None) -> None:
        """Forward user chat message to worker with bounded app context."""
        if self._chat_worker is None:
            self.chat_dock.add_error(
                "Chat is not configured",
                "Set a valid .env path with OPENAI_API_KEY in Preferences.",
            )
            return
        self.chat_dock.add_user_message(text)
        slash_parse_error = (metadata or {}).get("slash_parse_error")
        if slash_parse_error:
            self.chat_dock.add_error(
                "Slash command parse error",
                str(slash_parse_error),
            )
            return

        at_regions = (metadata or {}).get("at_regions", [])
        at_unresolved = (metadata or {}).get("at_unresolved", [])

        # Warn about any unresolved @mentions
        if at_unresolved:
            self.chat_dock.add_error(
                "Unresolved @mention",
                f"No labeled region named: {', '.join('@' + n for n in at_unresolved)}",
            )

        # Inject region context inline so the agent knows region_id → tool mapping
        if at_regions:
            context_parts = []
            for entry in at_regions:
                rid = entry["region_id"]
                r = self._labeled_regions.get(rid)
                if r:
                    context_parts.append(
                        f"@{r.name} = region_id={rid} "
                        f"({len(r.patch_indices)} patches, {r.source_mode.name.lower()})"
                    )
            if context_parts:
                text = "[Region context: " + "; ".join(context_parts) + "]\n" + text

        context = {
            "root_dir": self._root_dir,
            "has_slide_loaded": getattr(self, "_current_features", None) is not None,
            "has_labeled_regions": bool(getattr(self, "_labeled_regions", {})),
            "has_selected_clusters": bool(getattr(self, "_selected_clusters", set())),
            "atlas_ready": getattr(self, "_cluster_atlas", None) is not None,
            "slash_command": (metadata or {}).get("slash_command"),
        }
        self.chat_submit_requested.emit(text, context)


    def _on_chat_cancel_requested(self) -> None:
        """Cancel current chat run."""
        self.chat_cancel_requested.emit()


    def _on_chat_response_started(self, _message_id: str) -> None:
        """Mark beginning of assistant response lifecycle."""
        return None


    def _on_chat_response_delta(self, _message_id: str, text_delta: str) -> None:
        """Render streaming response chunk."""
        self.chat_dock.stop_typing_indicator()
        self.chat_dock.append_assistant_delta(text_delta)


    def _on_chat_tool_finished(
        self,
        _call_id: str,
        tool_name: str,
        summary: Dict[str, object],
        raw: Dict[str, object],
    ) -> None:
        """Render a completed tool result card."""
        self.chat_dock.add_tool_card(tool_name, summary, raw)


    def _on_chat_response_finished(
        self,
        _message_id: str,
        _full_text: str,
        usage: Dict[str, object],
        latency_ms: int,
    ) -> None:
        """Finalize assistant response UI state and status line."""
        self.chat_dock.stop_typing_indicator()
        self.chat_dock.finish_assistant_message()
        self.chat_dock.record_response_usage(usage)
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            self.chat_dock.set_status_text(f"Done in {latency_ms} ms")
        else:
            self.chat_dock.set_status_text(
                f"Done in {latency_ms} ms | tokens: {total_tokens}"
            )


    def _on_chat_error_emitted(
        self,
        _message_id: str,
        _error_code: str,
        user_message: str,
        details: str,
    ) -> None:
        """Render worker errors in chat view."""
        self.chat_dock.stop_typing_indicator()
        self.chat_dock.add_error(user_message, details)
        self.chat_dock.finish_assistant_message()


    def _on_chat_state_changed(self, state: str) -> None:
        """Toggle chat controls for worker state."""
        busy = state in {"running", "cancelling"}
        self.chat_dock.set_busy(busy)
        if state == "running":
            self.chat_dock.start_typing_indicator()
            self.chat_dock.set_status_text("Running...")
        elif state == "cancelling":
            self.chat_dock.stop_typing_indicator()
            self.chat_dock.set_status_text("Cancelling...")
        elif state == "idle":
            self.chat_dock.stop_typing_indicator()


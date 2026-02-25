"""
cluster_mixin.py
================
ClusterMixin mixin for MainWindow.
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


class ClusterMixin:

    def _on_cluster_rename_requested(self, cluster_id: int) -> None:
        """Prompt the user to rename a cluster."""
        current_name = self.cluster_legend.get_cluster_name(cluster_id)
        name, ok = QInputDialog.getText(
            self, "Rename Cluster",
            f"New name for cluster {cluster_id}:",
            text=current_name
        )
        if ok and name.strip():
            self.cluster_legend.set_cluster_name(cluster_id, name.strip())


    def _on_export_all_clusters(self) -> None:
        """Export all clusters to a single GeoJSON file."""
        # Validate data
        if self._current_labels is None:
            QMessageBox.warning(self, "Export Error", "No clustering data available.")
            return
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "Export Error", "Coordinate data not available.")
            return

        # Get file path
        default_name = "all_clusters.geojson"
        if hasattr(self, '_root_dir') and self._root_dir:
            default_path = os.path.join(self._root_dir, default_name)
        else:
            default_path = default_name

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Clusters", default_path,
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.endswith('.geojson'):
            file_path += '.geojson'

        # Build features for all clusters
        features = []
        unique_clusters = np.unique(self._current_labels)

        for cluster in unique_clusters:
            cluster = int(cluster)
            # Merge patches for this cluster
            merged_coords = self._merge_cluster_patches(cluster)
            if not merged_coords:
                continue

            # Get cluster color
            if cluster < len(self.scatter_view.cluster_colors):
                color_hsl = self._qcolor_to_hsl_string(self.scatter_view.cluster_colors[cluster])
                color_rgb = self._hsl_to_qupath_rgb(color_hsl)
            else:
                color_rgb = -16776961  # Default blue

            # Get cluster name
            cluster_name = f"Cluster {cluster}"

            # Create feature for each polygon in merged result
            for poly_coords in merged_coords:
                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {
                            "name": cluster_name,
                            "colorRGB": color_rgb
                        },
                        "isLocked": False,
                        "measurements": []
                    }
                }
                features.append(feature)

        # Build GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Write file
        try:
            with open(file_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(unique_clusters)} clusters ({len(features)} polygons) to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write file:\n{str(e)}")


    def _merge_cluster_patches(self, cluster: int) -> List[List]:
        """Merge adjacent patches into continuous polygons.
        
        Works in thumbnail space for efficiency, then scales to level-0
        coordinates for QuPath-compatible GeoJSON export.
        
        Parameters
        ----------
        cluster : int
            The cluster number to merge patches for.
            
        Returns
        -------
        List[List]
            List of GeoJSON-ready polygon coordinate arrays. Each element
            is a list of rings (outer ring + any holes), where each ring
            is a list of [x, y] coordinate pairs in level-0 pixel units.
        """
        # Get patch indices for this cluster
        if self.graphics_view.labels is None:
            return []
        indices = np.where(self.graphics_view.labels == cluster)[0]
        if len(indices) == 0:
            return []
        
        # Create Shapely boxes in thumbnail space for efficiency
        boxes = []
        for idx in indices:
            x, y = self._current_coords_thumb[idx]
            size = self._current_patch_size_thumb
            boxes.append(box(x, y, x + size, y + size))
        
        # Merge all touching/overlapping boxes using unary_union
        merged = unary_union(boxes)
        
        # Scale factor from thumbnail to level-0 coordinates
        scale = self._coord_scale_factor
        
        # Handle Polygon vs MultiPolygon result
        if merged.geom_type == 'Polygon':
            polygons = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polygons = list(merged.geoms)
        else:
            # Unexpected geometry type (GeometryCollection, etc.)
            print(f"DEBUG: Unexpected geometry type from unary_union: {merged.geom_type}")
            polygons = []
        
        # Convert to GeoJSON coordinate format, scaled to level-0
        result = []
        for poly in polygons:
            coords = []
            # Process exterior ring and any interior rings (holes)
            for ring in [poly.exterior] + list(poly.interiors):
                # Scale coordinates to level-0 and convert to nested lists
                # Round to integers since QuPath expects pixel coordinates
                scaled_ring = [[int(x * scale), int(y * scale)] for x, y in ring.coords]
                coords.append(scaled_ring)
            result.append(coords)
        
        print(f"DEBUG: Merged {len(indices)} patches into {len(result)} polygon(s)")
        return result


    def _on_export_clicked(self) -> None:
        """Handle export button click for selected clusters."""
        if not self._selected_clusters:
            QMessageBox.warning(
                self, "No Selection",
                "Please select a cluster first by clicking on the slide or scatter plot."
            )
            return

        # Check if we have the required coordinate data
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        selected = sorted(self._selected_clusters)
        if len(selected) == 1:
            self._export_single_cluster(selected[0])
            return

        prompt = QMessageBox(self)
        prompt.setWindowTitle("Export Selected Clusters")
        prompt.setText("Export selected clusters as one combined annotation or separate annotations?")
        combine_button = prompt.addButton("Combine", QMessageBox.AcceptRole)
        separate_button = prompt.addButton("Separate", QMessageBox.AcceptRole)
        prompt.addButton(QMessageBox.Cancel)
        prompt.exec()

        clicked = prompt.clickedButton()
        if clicked == combine_button:
            self._export_combined_clusters(selected)
        elif clicked == separate_button:
            self._export_separate_clusters(selected)


    def _export_single_cluster(self, cluster: int) -> None:
        """Export a single cluster to GeoJSON."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        default_name = f"Cluster_{cluster}"
        name, ok = QInputDialog.getText(
            self, "Annotation Name",
            "Enter a name for this annotation:",
            text=default_name
        )
        if not ok or not name.strip():
            return

        annotation_name = name.strip()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            f"{annotation_name}.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        polygons = self._merge_cluster_patches(cluster)
        if not polygons:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected cluster.")
            return

        color_rgb = self._get_cluster_color_rgb(cluster)
        features = self._build_geojson_features(polygons, annotation_name, color_rgb)
        self._write_geojson(file_path, features)


    def _export_combined_clusters(self, clusters: List[int]) -> None:
        """Export multiple clusters as a single combined annotation."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        name, ok = QInputDialog.getText(
            self, "Annotation Name",
            "Enter a name for the combined annotation:",
            text="Combined_Clusters"
        )
        if not ok or not name.strip():
            return
        annotation_name = name.strip()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            f"{annotation_name}.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        polygons = self._merge_selected_clusters_patches(clusters)
        if not polygons:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected clusters.")
            return

        color_rgb = self._get_cluster_color_rgb(clusters[0])
        features = self._build_geojson_features(polygons, annotation_name, color_rgb)
        self._write_geojson(file_path, features)


    def _export_separate_clusters(self, clusters: List[int]) -> None:
        """Export multiple clusters to a single GeoJSON file."""
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            QMessageBox.warning(self, "No Data", "Please load a slide first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation",
            "selected_clusters.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.geojson'):
            file_path += '.geojson'

        features = []
        for cluster in clusters:
            polygons = self._merge_cluster_patches(cluster)
            if not polygons:
                continue
            cluster_name = f"Cluster {cluster}"
            color_rgb = self._get_cluster_color_rgb(cluster)
            features.extend(self._build_geojson_features(polygons, cluster_name, color_rgb))

        if not features:
            QMessageBox.warning(self, "No Regions", "No patches found for the selected clusters.")
            return

        self._write_geojson(file_path, features)


    def _build_geojson_features(self, polygons: List[List], name: str, color_rgb: int) -> List[Dict]:
        """Build GeoJSON features for a set of polygons."""
        features = []
        for coords in polygons:
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": name,
                        "colorRGB": color_rgb
                    },
                    "isLocked": False,
                    "measurements": []
                }
            }
            features.append(feature)
        return features


    def _write_geojson(self, file_path: str, features: List[Dict]) -> None:
        """Write GeoJSON features to a file."""
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        try:
            with open(file_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(features)} region(s) to:\n{file_path}"
            )
            print(f"DEBUG: Exported GeoJSON with {len(features)} features to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write file:\n{e}")
            print(f"DEBUG: Export failed: {e}")


    def _get_cluster_color_rgb(self, cluster: int) -> int:
        """Get QuPath color for a cluster."""
        if self.scatter_view and hasattr(self.scatter_view, 'cluster_colors'):
            if cluster < len(self.scatter_view.cluster_colors):
                color_hsl = self._qcolor_to_hsl_string(self.scatter_view.cluster_colors[cluster])
                return self._hsl_to_qupath_rgb(color_hsl)
        colours = generate_palette(int(self.graphics_view.labels.max()) + 1)
        return self._hsl_to_qupath_rgb(colours[cluster])


    def _merge_selected_clusters_patches(self, clusters: List[int]) -> List[List]:
        """Merge patches from multiple clusters into polygons."""
        if self.graphics_view.labels is None:
            return []
        if self._current_coords_thumb is None or self._coord_scale_factor is None:
            return []

        boxes = []
        for cluster in clusters:
            indices = np.where(self.graphics_view.labels == cluster)[0]
            for idx in indices:
                x, y = self._current_coords_thumb[idx]
                size = self._current_patch_size_thumb
                boxes.append(box(x, y, x + size, y + size))

        if not boxes:
            return []

        merged = unary_union(boxes)
        scale = self._coord_scale_factor

        if merged.geom_type == 'Polygon':
            polygons = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polygons = list(merged.geoms)
        else:
            print(f"DEBUG: Unexpected geometry type from unary_union: {merged.geom_type}")
            polygons = []

        result = []
        for poly in polygons:
            coords = []
            for ring in [poly.exterior] + list(poly.interiors):
                scaled_ring = [[int(x * scale), int(y * scale)] for x, y in ring.coords]
                coords.append(scaled_ring)
            result.append(coords)
        return result


    def _update_export_action(self) -> None:
        """Enable or disable export based on selection state."""
        if self._model_selection is None:
            self.export_action.setEnabled(False)
            return
        self.export_action.setEnabled(bool(self._selected_clusters))


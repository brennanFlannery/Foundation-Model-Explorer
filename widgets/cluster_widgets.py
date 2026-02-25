"""
cluster_widgets.py
==================

Widget classes for cluster legend display and region selection controls.

Contains:
- ClusterLegendWidget: Scrollable sidebar legend with cluster colors and patch counts.
- LocalRegionWidget: Radius slider + list of user-defined local regions.
- LabeledRegionsWidget: Persistent panel listing all labeled regions across modes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui_types import LabeledRegion, SourceMode


class ClusterLegendWidget(QWidget):
    """Widget displaying cluster color legend with patch counts.

    This widget shows a vertical list of cluster entries, each containing
    a colored square, cluster ID, and patch count. The legend automatically
    updates when clustering changes.

    Attributes
    ----------
    _cluster_rows : List[QWidget]
        List of row widgets for each cluster entry.
    """

    # Signals for cluster interactions
    cluster_clicked = Signal(int, bool)       # Emits cluster ID and ctrl state on left-click
    cluster_toggled = Signal(int, bool)       # Emits cluster ID and checked state
    cluster_rename = Signal(int)              # Emits cluster ID for rename request
    cluster_export = Signal(int)              # Emits cluster ID for single export
    export_all_requested = Signal()           # Emits when "Export All" selected

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)

        # Header
        header = QLabel("Clusters")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self._layout.addWidget(header)

        # Scrollable container for cluster rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        self._layout.addWidget(self._scroll)

        self._cluster_rows: List[QWidget] = []
        self._cluster_names: Dict[int, str] = {}  # Custom names for clusters
        self._checkboxes: Dict[int, QCheckBox] = {}

        # Set minimum width
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)

    def update_clusters(self, labels: np.ndarray, colors: List[str]) -> None:
        """Update legend with current cluster data.

        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for all patches.
        colors : List[str]
            HSL color strings for each cluster.
        """
        # Clear existing rows
        for row in self._cluster_rows:
            row.deleteLater()
        self._cluster_rows.clear()
        self._checkboxes.clear()

        if labels is None or len(labels) == 0:
            return

        # Count patches per cluster
        unique_labels = np.unique(labels)
        counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}

        # Create row for each cluster
        for lbl in sorted(unique_labels):
            lbl = int(lbl)
            count = counts[lbl]
            color_str = colors[lbl] if lbl < len(colors) else "hsl(0, 0%, 50%)"

            row = self._create_cluster_row(lbl, count, color_str)
            # Insert before the stretch
            self._content_layout.insertWidget(
                self._content_layout.count() - 1, row
            )
            self._cluster_rows.append(row)

    def _create_cluster_row(self, cluster_id: int, count: int,
                            color_str: str) -> QWidget:
        """Create a single cluster row widget.

        Parameters
        ----------
        cluster_id : int
            The cluster ID/number.
        count : int
            Number of patches in this cluster.
        color_str : str
            HSL color string for the cluster.

        Returns
        -------
        QWidget
            The row widget containing color square, label, and count.
        """
        row = QWidget()
        row.setProperty("cluster_id", cluster_id)  # Store cluster ID for event handling
        row.setCursor(Qt.PointingHandCursor)  # Indicate clickable
        row.installEventFilter(self)  # Handle mouse events

        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        # Color square
        color_square = QFrame()
        color_square.setFixedSize(14, 14)
        # Convert HSL to hex for CSS
        qcolor = self._hsl_to_qcolor(color_str)
        hex_color = qcolor.name()
        color_square.setStyleSheet(
            f"background-color: {hex_color}; "
            f"border: 1px solid #666; border-radius: 2px;"
        )
        color_square.installEventFilter(self)
        layout.addWidget(color_square)

        # Checkbox for multi-select
        checkbox = QCheckBox()
        checkbox.setToolTip("Select cluster")
        checkbox.toggled.connect(lambda checked, cid=cluster_id: self.cluster_toggled.emit(cid, checked))
        layout.addWidget(checkbox)
        self._checkboxes[cluster_id] = checkbox

        # Cluster label - use custom name if available
        display_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        label = QLabel(display_name)
        label.setStyleSheet("font-size: 9pt;")
        label.setObjectName("cluster_label")  # For later reference when renaming
        label.installEventFilter(self)
        layout.addWidget(label)

        # Spacer
        layout.addStretch()

        # Count
        count_label = QLabel(f"({count:,})")
        count_label.setStyleSheet("color: #888; font-size: 9pt;")
        count_label.installEventFilter(self)
        layout.addWidget(count_label)

        return row

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle mouse events on cluster rows."""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        if event.type() == QEvent.MouseButtonPress:
            cluster_id = obj.property("cluster_id")
            if cluster_id is None and obj.parent() is not None:
                cluster_id = obj.parent().property("cluster_id")
            if cluster_id is not None:
                if event.button() == Qt.LeftButton:
                    ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
                    self.cluster_clicked.emit(cluster_id, ctrl_pressed)
                    return True
                elif event.button() == Qt.RightButton:
                    self._show_context_menu(cluster_id, event.globalPosition().toPoint())
                    return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, cluster_id: int, pos) -> None:
        """Show context menu for a cluster row."""
        menu = QMenu(self)

        # Rename action
        rename_action = menu.addAction("Rename Cluster...")
        rename_action.triggered.connect(lambda: self.cluster_rename.emit(cluster_id))

        # Export single cluster
        export_action = menu.addAction("Export as GeoJSON...")
        export_action.triggered.connect(lambda: self.cluster_export.emit(cluster_id))

        menu.addSeparator()

        # Export all clusters
        export_all_action = menu.addAction("Export All Clusters...")
        export_all_action.triggered.connect(self.export_all_requested.emit)

        menu.exec_(pos)

    def set_cluster_checked(self, cluster_id: int, checked: bool,
                            block_signals: bool = True) -> None:
        """Set checkbox state for a cluster row."""
        checkbox = self._checkboxes.get(cluster_id)
        if checkbox is None:
            return
        if block_signals:
            checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        if block_signals:
            checkbox.blockSignals(False)

    def set_cluster_name(self, cluster_id: int, name: str) -> None:
        """Set a custom name for a cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID to rename.
        name : str
            The new display name.
        """
        self._cluster_names[cluster_id] = name
        for row in self._cluster_rows:
            if row.property("cluster_id") == cluster_id:
                label = row.findChild(QLabel, "cluster_label")
                if label is not None:
                    label.setText(name)
                break

    def get_cluster_name(self, cluster_id: int) -> str:
        """Get the display name for a cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID.

        Returns
        -------
        str
            Custom name if set, otherwise default "Cluster N" format.
        """
        return self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    def clear_cluster_names(self) -> None:
        """Clear all custom cluster names."""
        self._cluster_names.clear()

    @staticmethod
    def _hsl_to_qcolor(hsl_string: str) -> QColor:
        """Convert HSL string to QColor.

        Parameters
        ----------
        hsl_string : str
            Color in format 'hsl(H, S%, L%)'.

        Returns
        -------
        QColor
            Converted Qt color object.
        """
        try:
            values = hsl_string.strip().lower().replace('hsl(', '').rstrip(')').split(',')
            h = float(values[0])
            s = float(values[1].strip(' %')) / 100.0
            l = float(values[2].strip(' %')) / 100.0
            c = QColor()
            c.setHslF(h / 360.0, s, l)
            return c
        except Exception:
            return QColor('black')


class LocalRegionWidget(QWidget):
    """Widget for local region selection controls and cluster list.

    This widget provides controls for the local region selection mode,
    including a radius slider and a list of user-defined regions.
    """

    # Signals
    radius_changed = Signal(float)       # Emits new radius value
    region_clicked = Signal(int)         # Emits region ID for highlighting
    region_deleted = Signal(int)         # Emits region ID for deletion
    clear_all_requested = Signal()       # Request to clear all regions
    export_requested = Signal()          # Request to export regions
    full_slide_toggled = Signal(bool)    # Emits True when Full Slide mode enabled

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._region_rows: Dict[int, QWidget] = {}

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QLabel("Local Region Selection")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Click to select patches within radius\n"
            "(limited to same K-means cluster)"
        )
        instructions.setStyleSheet("color: gray; font-size: 9pt;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Full Slide checkbox — when checked, selects entire cluster (KMEANS mode)
        self.full_slide_check = QCheckBox("Full Slide")
        self.full_slide_check.setChecked(True)
        self.full_slide_check.setStyleSheet("font-size: 9pt;")
        self.full_slide_check.toggled.connect(self._on_full_slide_toggled)
        layout.addWidget(self.full_slide_check)

        # Slider row container — hidden when Full Slide is checked
        self._slider_row_widget = QWidget()
        slider_layout = QVBoxLayout(self._slider_row_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(4)

        radius_header = QLabel("Selection Radius:")
        radius_header.setStyleSheet("font-size: 9pt; margin-top: 4px;")
        slider_layout.addWidget(radius_header)

        radius_row = QHBoxLayout()
        radius_row.setSpacing(8)

        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(10, 500)  # Will be recalculated based on patch size
        self.radius_slider.setValue(50)
        self.radius_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radius_slider.setTickInterval(50)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        radius_row.addWidget(self.radius_slider, stretch=1)

        self.radius_label = QLabel("50")
        self.radius_label.setMinimumWidth(40)
        self.radius_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.radius_label.setStyleSheet("font-size: 9pt;")
        radius_row.addWidget(self.radius_label)

        slider_layout.addLayout(radius_row)
        layout.addWidget(self._slider_row_widget)
        self._slider_row_widget.setVisible(False)  # Hidden when Full Slide starts checked

        # Separator before regions list
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        # Regions header
        regions_header = QLabel("User-Defined Regions:")
        regions_header.setStyleSheet("font-size: 9pt; margin-top: 4px;")
        layout.addWidget(regions_header)

        # Scrollable region list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, stretch=1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("font-size: 9pt;")
        self.clear_btn.clicked.connect(self.clear_all_requested.emit)
        self.clear_btn.setEnabled(False)  # Disabled when no regions
        btn_row.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("font-size: 9pt;")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.export_btn.setEnabled(False)  # Disabled when no regions
        btn_row.addWidget(self.export_btn)

        layout.addLayout(btn_row)

        # Set size constraints
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)

    def _on_full_slide_toggled(self, checked: bool) -> None:
        """Handle Full Slide checkbox toggle — show/hide radius slider."""
        self._slider_row_widget.setVisible(not checked)
        self.full_slide_toggled.emit(checked)

    def _on_radius_changed(self, value: int) -> None:
        """Handle radius slider change."""
        self.radius_label.setText(str(value))
        self.radius_changed.emit(float(value))

    def set_radius_range(self, min_val: int, max_val: int, current: int) -> None:
        """Set radius slider range based on patch size.

        Parameters
        ----------
        min_val : int
            Minimum radius value.
        max_val : int
            Maximum radius value.
        current : int
            Current/default radius value.
        """
        self.radius_slider.blockSignals(True)
        self.radius_slider.setRange(min_val, max_val)
        self.radius_slider.setValue(current)
        self.radius_slider.setTickInterval(max(1, (max_val - min_val) // 10))
        self.radius_label.setText(str(current))
        self.radius_slider.blockSignals(False)

    def get_radius(self) -> float:
        """Get the current radius value."""
        return float(self.radius_slider.value())

    def add_region(self, region_id: int, patch_count: int,
                   color: QColor, name: str) -> None:
        """Add a new region row to the list.

        Parameters
        ----------
        region_id : int
            Unique identifier for the region.
        patch_count : int
            Number of patches in the region.
        color : QColor
            Color for the region display.
        name : str
            Display name for the region.
        """
        row = QWidget()
        row.setProperty("region_id", region_id)
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.installEventFilter(self)

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)
        row_layout.setSpacing(6)

        # Color square
        color_square = QFrame()
        color_square.setFixedSize(14, 14)
        hex_color = color.name()
        color_square.setStyleSheet(
            f"background-color: {hex_color}; "
            f"border: 1px solid #666; border-radius: 2px;"
        )
        row_layout.addWidget(color_square)

        # Region name label
        label = QLabel(name)
        label.setStyleSheet("font-size: 9pt;")
        label.setObjectName("region_label")
        row_layout.addWidget(label)

        # Spacer
        row_layout.addStretch()

        # Count
        count_label = QLabel(f"({patch_count:,})")
        count_label.setStyleSheet("color: #888; font-size: 9pt;")
        count_label.setObjectName("region_count")
        row_layout.addWidget(count_label)

        # Delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(18, 18)
        delete_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; color: #888; border: none; padding: 0; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        delete_btn.setToolTip("Delete region")
        delete_btn.clicked.connect(lambda: self.region_deleted.emit(region_id))
        row_layout.addWidget(delete_btn)

        # Insert before the stretch
        self._content_layout.insertWidget(
            self._content_layout.count() - 1, row
        )
        self._region_rows[region_id] = row

        # Enable buttons
        self.clear_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def update_region(self, region_id: int, patch_count: int, name: Optional[str] = None) -> None:
        """Update an existing region row."""
        row = self._region_rows.get(region_id)
        if row is None:
            return

        label = row.findChild(QLabel, "region_label")
        if label is not None and name is not None:
            label.setText(name)

        count_label = row.findChild(QLabel, "region_count")
        if count_label is not None:
            count_label.setText(f"({patch_count:,})")

    def remove_region(self, region_id: int) -> None:
        """Remove a region row from the list."""
        row = self._region_rows.pop(region_id, None)
        if row is not None:
            row.deleteLater()

        # Disable buttons if no regions
        if not self._region_rows:
            self.clear_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

    def clear_regions(self) -> None:
        """Remove all region rows."""
        for row in self._region_rows.values():
            row.deleteLater()
        self._region_rows.clear()
        self.clear_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle mouse events on region rows."""
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.MouseButtonPress:
            region_id = obj.property("region_id")
            if region_id is not None:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.region_clicked.emit(region_id)
                    return True
        return super().eventFilter(obj, event)


class LabeledRegionsWidget(QWidget):
    """Persistent sidebar panel showing all labeled regions across modes.

    Displays a unified list of labeled regions from K-means and Local Region
    modes, always visible below the tab widget.
    """

    region_clicked      = Signal(int)  # region_id
    region_deleted      = Signal(int)  # region_id
    export_requested    = Signal()
    clear_all_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._region_rows: Dict[int, QWidget] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QLabel("Labeled Regions")
        header.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.export_btn = QPushButton("Export All")
        self.export_btn.setStyleSheet("font-size: 9pt;")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.export_btn.setEnabled(False)
        btn_row.addWidget(self.export_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("font-size: 9pt;")
        self.clear_btn.clicked.connect(self.clear_all_requested.emit)
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

        self.setMinimumHeight(120)
        self.setMaximumHeight(300)

    def add_region(self, region: "LabeledRegion") -> None:
        """Add a new region row to the list."""
        row = QWidget()
        row.setProperty("region_id", region.region_id)
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.installEventFilter(self)

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)
        row_layout.setSpacing(4)

        # Color square
        color_square = QFrame()
        color_square.setFixedSize(14, 14)
        color_square.setStyleSheet(
            f"background-color: {region.color.name()}; "
            f"border: 1px solid #666; border-radius: 2px;"
        )
        row_layout.addWidget(color_square)

        # Mode badge ("K" blue / "L" orange)
        badge_text = "K" if region.source_mode == SourceMode.KMEANS else "L"
        badge = QLabel(badge_text)
        badge.setObjectName("mode_badge")
        badge.setFixedWidth(16)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if region.source_mode == SourceMode.KMEANS:
            badge.setStyleSheet(
                "background-color: #3a7bd5; color: white; font-size: 8pt; border-radius: 2px;"
            )
        else:
            badge.setStyleSheet(
                "background-color: #e08024; color: white; font-size: 8pt; border-radius: 2px;"
            )
        row_layout.addWidget(badge)

        # Name label
        label = QLabel(region.name)
        label.setStyleSheet("font-size: 9pt;")
        label.setObjectName("region_label")
        row_layout.addWidget(label, stretch=1)

        # Patch count
        count_label = QLabel(f"({len(region.patch_indices):,})")
        count_label.setStyleSheet("color: #888; font-size: 9pt;")
        count_label.setObjectName("region_count")
        row_layout.addWidget(count_label)

        # Delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(18, 18)
        delete_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; color: #888; border: none; padding: 0; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        delete_btn.setToolTip("Delete region")
        delete_btn.clicked.connect(lambda: self.region_deleted.emit(region.region_id))
        row_layout.addWidget(delete_btn)

        self._content_layout.insertWidget(self._content_layout.count() - 1, row)
        self._region_rows[region.region_id] = row

        self.export_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def remove_region(self, region_id: int) -> None:
        """Remove a region row."""
        row = self._region_rows.pop(region_id, None)
        if row is not None:
            row.deleteLater()
        if not self._region_rows:
            self.export_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)

    def update_region(self, region_id: int, patch_count: int, name: Optional[str] = None) -> None:
        """Update patch count (and optionally name) of an existing row."""
        row = self._region_rows.get(region_id)
        if row is None:
            return
        label = row.findChild(QLabel, "region_label")
        if label is not None and name is not None:
            label.setText(name)
        count_label = row.findChild(QLabel, "region_count")
        if count_label is not None:
            count_label.setText(f"({patch_count:,})")

    def update_region_source(self, region_id: int, source_mode: "SourceMode") -> None:
        """Update the K/L badge for an existing row when source mode changes."""
        row = self._region_rows.get(region_id)
        if row is None:
            return
        badge = row.findChild(QLabel, "mode_badge")
        if badge is None:
            return
        from gui_types import SourceMode
        if source_mode == SourceMode.LOCAL:
            badge.setText("L")
            badge.setStyleSheet(
                "background-color: #e08024; color: white; font-size: 8pt; border-radius: 2px;"
            )
        else:
            badge.setText("K")
            badge.setStyleSheet(
                "background-color: #3a7bd5; color: white; font-size: 8pt; border-radius: 2px;"
            )

    def clear_regions(self) -> None:
        """Remove all region rows."""
        for row in self._region_rows.values():
            row.deleteLater()
        self._region_rows.clear()
        self.export_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle mouse events on region rows."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.MouseButtonPress:
            region_id = obj.property("region_id")
            if region_id is not None:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.region_clicked.emit(region_id)
                    return True
        return super().eventFilter(obj, event)

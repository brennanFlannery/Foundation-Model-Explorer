"""
patch_widgets.py
================

Widget classes for displaying patch information panels and popups.

Contains:
- PatchInfoPanel: Sidebar panel showing hovered/selected patch details.
- PatchInfoPopup: Translucent animated popup that appears near the cursor.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class PatchInfoPanel(QWidget):
    """Widget displaying information about hovered/selected patch.

    This panel shows details about the currently hovered patch, including
    its index, coordinates, cluster assignment, and distance to the
    cluster centroid in feature space.

    Attributes
    ----------
    index_label : QLabel
        Displays the patch index.
    coords_label : QLabel
        Displays the patch coordinates.
    cluster_label : QLabel
        Displays the cluster assignment.
    distance_label : QLabel
        Displays distance to cluster centroid.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QLabel("Patch Info")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(header)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Info labels
        self.index_label = QLabel("Patch: -")
        self.index_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.index_label)

        self.coords_label = QLabel("Position: -")
        self.coords_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.coords_label)

        self.cluster_label = QLabel("Cluster: -")
        self.cluster_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.cluster_label)

        self.distance_label = QLabel("Centroid dist: -")
        self.distance_label.setStyleSheet("font-size: 9pt; color: #666;")
        layout.addWidget(self.distance_label)

        layout.addStretch()

        # Set size constraints
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)

    def update_patch_info(self, index: int, coords: Optional[Tuple[float, float]],
                          cluster: Optional[int], distance: Optional[float]) -> None:
        """Update displayed patch information.

        Parameters
        ----------
        index : int
            Patch index, or -1 to clear.
        coords : Optional[Tuple[float, float]]
            Patch coordinates (x, y) or None.
        cluster : Optional[int]
            Cluster assignment or None.
        distance : Optional[float]
            Distance to cluster centroid or None.
        """
        if index < 0:
            self.index_label.setText("Patch: -")
            self.coords_label.setText("Position: -")
            self.cluster_label.setText("Cluster: -")
            self.distance_label.setText("Centroid dist: -")
        else:
            self.index_label.setText(f"Patch: {index}")
            if coords is not None:
                self.coords_label.setText(f"Position: ({coords[0]:.0f}, {coords[1]:.0f})")
            else:
                self.coords_label.setText("Position: -")
            if cluster is not None:
                self.cluster_label.setText(f"Cluster: {cluster}")
            else:
                self.cluster_label.setText("Cluster: -")
            if distance is not None:
                self.distance_label.setText(f"Centroid dist: {distance:.1f}%")
            else:
                self.distance_label.setText("Centroid dist: -")

    def clear(self) -> None:
        """Clear all displayed information."""
        self.update_patch_info(-1, None, None, None)


class PatchInfoPopup(QWidget):
    """Translucent popup widget displaying patch information on hover.

    This popup appears near the cursor when hovering over scatter points,
    showing patch index, coordinates, cluster assignment, and centroid distance.

    The widget uses:
    - Qt.WindowFlags for frameless, translucent window
    - QPropertyAnimation for smooth fade in/out
    - Dynamic positioning to avoid viewport edge clipping
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Window flags for translucent, frameless, always-on-top popup
        self.setWindowFlags(
            Qt.ToolTip |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Setup UI
        self._setup_ui()

        # Animation for fade in/out
        self._opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_animation.setDuration(150)
        self._opacity_animation.setEasingCurve(QEasingCurve.OutCubic)

        # Hover delay timer (prevents flicker on fast mouse movement)
        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.timeout.connect(self._do_show)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._do_hide)

        # Pending position for delayed show
        self._pending_pos: Optional[QPoint] = None

        # Initially hidden
        self.hide()

    def _setup_ui(self) -> None:
        """Create the popup layout and labels."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container frame for styling
        self._frame = QFrame(self)
        self._frame.setObjectName("popupFrame")
        self._frame.setStyleSheet("""
            #popupFrame {
                background-color: rgba(40, 40, 45, 230);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(12, 10, 12, 10)
        frame_layout.setSpacing(4)

        # Header
        self._header = QLabel("Patch Info")
        self._header.setStyleSheet("""
            font-weight: bold;
            font-size: 10pt;
            color: rgba(255, 255, 255, 230);
        """)
        frame_layout.addWidget(self._header)

        # Info labels
        label_style = "font-size: 9pt; color: rgba(255, 255, 255, 200);"

        self.index_label = QLabel("Patch: -")
        self.index_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.index_label)

        self.coords_label = QLabel("Position: -")
        self.coords_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.coords_label)

        self.cluster_label = QLabel("Cluster: -")
        self.cluster_label.setStyleSheet(label_style)
        frame_layout.addWidget(self.cluster_label)

        self.distance_label = QLabel("Centroid dist: -")
        self.distance_label.setStyleSheet(
            "font-size: 9pt; color: rgba(200, 200, 200, 180);"
        )
        frame_layout.addWidget(self.distance_label)

        # Add frame to main layout
        layout.addWidget(self._frame)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(2, 2)
        self._frame.setGraphicsEffect(shadow)

        # Fixed width for consistent appearance
        self.setFixedWidth(180)

    def update_info(self, index: int, coords: Optional[Tuple[float, float]],
                    cluster: Optional[int], distance: Optional[float],
                    cluster_color: Optional[QColor] = None) -> None:
        """Update displayed patch information.

        Parameters
        ----------
        index : int
            Patch index.
        coords : Optional[Tuple[float, float]]
            Patch coordinates (x, y).
        cluster : Optional[int]
            Cluster assignment.
        distance : Optional[float]
            Distance to cluster centroid (percentage).
        cluster_color : Optional[QColor]
            Cluster color for visual indicator.
        """
        self.index_label.setText(f"Patch: {index}")

        if coords is not None:
            self.coords_label.setText(f"Position: ({coords[0]:.0f}, {coords[1]:.0f})")
        else:
            self.coords_label.setText("Position: -")

        if cluster is not None:
            if cluster_color is not None:
                rgb = f"rgb({cluster_color.red()}, {cluster_color.green()}, {cluster_color.blue()})"
                self.cluster_label.setText(f"Cluster: {cluster}")
                self.cluster_label.setStyleSheet(
                    f"font-size: 9pt; color: {rgb}; font-weight: bold;"
                )
            else:
                self.cluster_label.setText(f"Cluster: {cluster}")
                self.cluster_label.setStyleSheet("font-size: 9pt; color: rgba(255, 255, 255, 200);")
        else:
            self.cluster_label.setText("Cluster: -")
            self.cluster_label.setStyleSheet("font-size: 9pt; color: rgba(255, 255, 255, 200);")

        if distance is not None:
            self.distance_label.setText(f"Centroid dist: {distance:.1f}%")
        else:
            self.distance_label.setText("Centroid dist: -")

    def show_at_cursor(self, global_pos: QPoint, delay_ms: int = 100) -> None:
        """Show the popup near the given global position with optional delay.

        Parameters
        ----------
        global_pos : QPoint
            Global screen position (typically from QCursor.pos()).
        delay_ms : int
            Delay before showing (prevents flicker on fast movement).
        """
        # Cancel any pending hide
        self._hide_timer.stop()

        # Store position for delayed show
        self._pending_pos = global_pos

        if delay_ms > 0:
            self._show_timer.start(delay_ms)
        else:
            self._do_show()

    def show_at_position(self, global_pos: QPoint, delay_ms: int = 100) -> None:
        """Show the popup at a specific position without edge avoidance.

        Parameters
        ----------
        global_pos : QPoint
            Global screen position to place the popup.
        delay_ms : int
            Delay before showing (prevents flicker on fast movement).
        """
        # Cancel any pending hide
        self._hide_timer.stop()

        # Store position for delayed show (use directly, no edge calc)
        self._pending_pos = global_pos
        self._use_direct_position = True

        if delay_ms > 0:
            self._show_timer.start(delay_ms)
        else:
            self._do_show()

    def _do_show(self) -> None:
        """Internal method to actually show the popup."""
        if self._pending_pos is None:
            return

        # Calculate position with edge avoidance, or use direct position
        if getattr(self, '_use_direct_position', False):
            pos = self._pending_pos
            self._use_direct_position = False
        else:
            pos = self._calculate_position(self._pending_pos)
        self.move(pos)

        # Fade in
        self._opacity_animation.stop()
        self._opacity_animation.setStartValue(self.windowOpacity())
        self._opacity_animation.setEndValue(1.0)
        self._opacity_animation.start()

        self.show()
        self.raise_()

    def hide_popup(self, delay_ms: int = 50) -> None:
        """Hide the popup with optional delay and fade out.

        Parameters
        ----------
        delay_ms : int
            Delay before hiding (allows re-hover to cancel hide).
        """
        # Cancel any pending show
        self._show_timer.stop()

        if delay_ms > 0:
            self._hide_timer.start(delay_ms)
        else:
            self._do_hide()

    def _do_hide(self) -> None:
        """Internal method to actually hide the popup."""
        # Fade out then hide
        self._opacity_animation.stop()
        self._opacity_animation.setStartValue(self.windowOpacity())
        self._opacity_animation.setEndValue(0.0)

        # Disconnect any previous connection to avoid duplicates
        try:
            self._opacity_animation.finished.disconnect(self._on_fade_out_finished)
        except RuntimeError:
            pass

        self._opacity_animation.finished.connect(self._on_fade_out_finished)
        self._opacity_animation.start()

    def _on_fade_out_finished(self) -> None:
        """Called when fade-out animation completes."""
        try:
            self._opacity_animation.finished.disconnect(self._on_fade_out_finished)
        except RuntimeError:
            pass
        if self.windowOpacity() == 0.0:
            self.hide()

    def _calculate_position(self, cursor_pos: QPoint) -> QPoint:
        """Calculate popup position avoiding screen edges.

        Parameters
        ----------
        cursor_pos : QPoint
            Global cursor position.

        Returns
        -------
        QPoint
            Adjusted position for the popup.
        """
        # Offset from cursor
        offset_x = 15
        offset_y = 15

        # Get screen geometry
        screen = QApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()

        # Calculate popup dimensions
        popup_width = self.sizeHint().width()
        popup_height = self.sizeHint().height()

        # Default position: right and below cursor
        x = cursor_pos.x() + offset_x
        y = cursor_pos.y() + offset_y

        # Check right edge
        if x + popup_width > screen_rect.right():
            x = cursor_pos.x() - popup_width - offset_x

        # Check bottom edge
        if y + popup_height > screen_rect.bottom():
            y = cursor_pos.y() - popup_height - offset_y

        # Ensure not off left/top edges
        x = max(x, screen_rect.left())
        y = max(y, screen_rect.top())

        return QPoint(x, y)

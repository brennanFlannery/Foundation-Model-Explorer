"""Dockable chat UI components for the FoundationDetector agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TypingDotsWidget(QWidget):
    """Simple pulsing three-dot typing indicator."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(240)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(42, 14)
        self._timer.start()

    def _tick(self) -> None:
        self._phase = (self._phase + 1) % 3
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        base = QColor("#1f2937")
        alphas = [90, 90, 90]
        alphas[self._phase] = 230

        radius = 3
        y = self.height() // 2
        xs = [8, 21, 34]
        for index, x in enumerate(xs):
            color = QColor(base)
            color.setAlpha(alphas[index])
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)


class ChatBubbleRow(QWidget):
    """Single chat row with role-based bubble alignment and styling."""

    def __init__(
        self,
        role: str,
        text: str,
        details_text: Optional[str] = None,
        typing: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._role = role
        self._bubble_max_width = 420

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(8)

        self._bubble = QFrame(self)
        self._bubble.setObjectName(f"chat-bubble-{role}")
        self._bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        bubble_layout = QVBoxLayout(self._bubble)
        bubble_layout.setContentsMargins(10, 8, 10, 8)

        self._label: Optional[QLabel] = None
        self._typing_widget: Optional[TypingDotsWidget] = None
        if typing:
            self._typing_widget = TypingDotsWidget(self._bubble)
            bubble_layout.addWidget(self._typing_widget, 0, Qt.AlignmentFlag.AlignLeft)
        else:
            self._label = QLabel(text, self._bubble)
            self._label.setWordWrap(True)
            self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            bubble_layout.addWidget(self._label)

        self._details_toggle: Optional[QPushButton] = None
        self._details_view: Optional[QTextEdit] = None
        if details_text:
            self._details_toggle = QPushButton("Show details", self._bubble)
            self._details_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            self._details_toggle.setStyleSheet(
                "QPushButton {"
                "background: transparent;"
                "border: none;"
                "text-align: left;"
                "color: #0b5ed7;"
                "padding: 2px 0 0 0;"
                "}"
                "QPushButton:hover { text-decoration: underline; }"
            )
            self._details_toggle.clicked.connect(self._toggle_details)
            bubble_layout.addWidget(self._details_toggle)

            self._details_view = QTextEdit(self._bubble)
            self._details_view.setReadOnly(True)
            self._details_view.setPlainText(details_text)
            self._details_view.setVisible(False)
            self._details_view.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
            )
            self._details_view.setMaximumHeight(170)
            self._details_view.setStyleSheet(
                "QTextEdit {"
                "background-color: #ffffff;"
                "border: 1px solid #cbd5e1;"
                "border-radius: 6px;"
                "padding: 4px;"
                "font-family: Menlo, Monaco, 'Courier New', monospace;"
                "font-size: 11px;"
                "}"
            )
            bubble_layout.addWidget(self._details_view)

        if role == "user":
            self._layout.addStretch(1)
            self._layout.addWidget(self._bubble, 0, Qt.AlignmentFlag.AlignRight)
        else:
            self._layout.addWidget(self._bubble, 0, Qt.AlignmentFlag.AlignLeft)
            self._layout.addStretch(1)

        self._apply_style(role)
        self.set_max_width(self._bubble_max_width)

    def _toggle_details(self) -> None:
        if not self._details_view or not self._details_toggle:
            return
        show = not self._details_view.isVisible()
        self._details_view.setVisible(show)
        self._details_toggle.setText("Hide details" if show else "Show details")
        self.updateGeometry()

    def _apply_style(self, role: str) -> None:
        if role == "user":
            bubble_bg = "#2f6fed"
            text_fg = "#ffffff"
        elif role == "assistant":
            bubble_bg = "#f1f3f5"
            text_fg = "#1f2937"
        elif role == "tool":
            bubble_bg = "#e8f4ff"
            text_fg = "#0f2a43"
        else:
            bubble_bg = "#ffe9e9"
            text_fg = "#7f1d1d"

        self._bubble.setStyleSheet(
            f"QFrame#{self._bubble.objectName()} {{"
            f"background-color: {bubble_bg};"
            "border-radius: 12px;"
            "}"
        )
        if self._label is not None:
            self._label.setStyleSheet(f"color: {text_fg};")

    def set_max_width(self, width: int) -> None:
        self._bubble_max_width = max(200, width)
        self._bubble.setMaximumWidth(self._bubble_max_width)
        # Keep text width slightly narrower than bubble for padding.
        if self._label is not None:
            self._label.setMaximumWidth(self._bubble_max_width - 24)
        if self._details_view is not None:
            self._details_view.setMaximumWidth(self._bubble_max_width - 24)

    def set_text(self, text: str) -> None:
        if self._label is not None:
            self._label.setText(text)

    def append_text(self, text_delta: str) -> None:
        if self._label is not None:
            self._label.setText(self._label.text() + text_delta)


class ChatDockWidget(QDockWidget):
    """Dock widget containing the chat message stream and controls."""

    send_requested = Signal(str)
    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Agent Chat", parent)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.scroll_area = QScrollArea(container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.messages_container = QWidget(self.scroll_area)
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(2)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area, stretch=1)

        self.status_label = QLabel("Idle", container)
        self.status_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.status_label)

        self.input_box = QPlainTextEdit(container)
        self.input_box.setPlaceholderText("Ask about the currently loaded data...")
        self.input_box.setFixedHeight(80)
        layout.addWidget(self.input_box)

        button_row = QHBoxLayout()
        self.send_button = QPushButton("Send", container)
        self.cancel_button = QPushButton("Cancel", container)
        self.cancel_button.setEnabled(False)
        button_row.addWidget(self.send_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.send_button.clicked.connect(self._on_send)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        self.setWidget(container)

        self._active_assistant_row: Optional[ChatBubbleRow] = None
        self._typing_row: Optional[ChatBubbleRow] = None
        self._rows: List[ChatBubbleRow] = []

    def _on_send(self) -> None:
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self.input_box.clear()
        self.send_requested.emit(text)

    def _is_at_bottom(self) -> bool:
        bar = self.scroll_area.verticalScrollBar()
        return bar.value() >= (bar.maximum() - 8)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_row_widths(self) -> None:
        viewport_width = self.scroll_area.viewport().width()
        bubble_max = int(max(260, viewport_width * 0.75))
        for row in self._rows:
            row.set_max_width(bubble_max)

    def _add_row(
        self,
        role: str,
        text: str,
        details_text: Optional[str] = None,
        typing: bool = False,
    ) -> ChatBubbleRow:
        was_at_bottom = self._is_at_bottom()
        row = ChatBubbleRow(
            role=role,
            text=text,
            details_text=details_text,
            typing=typing,
            parent=self.messages_container,
        )
        self._rows.append(row)
        self.messages_layout.addWidget(row)
        self._update_row_widths()
        if was_at_bottom:
            QTimer.singleShot(0, self._scroll_to_bottom)
        return row

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_row_widths()

    def _remove_row(self, row: ChatBubbleRow) -> None:
        self.messages_layout.removeWidget(row)
        if row in self._rows:
            self._rows.remove(row)
        row.deleteLater()

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def add_user_message(self, text: str) -> None:
        self._add_row("user", text)

    def start_assistant_message(self) -> None:
        self._active_assistant_row = self._add_row("assistant", "")

    def append_assistant_delta(self, text_delta: str) -> None:
        if not self._active_assistant_row:
            self.start_assistant_message()
        was_at_bottom = self._is_at_bottom()
        self._active_assistant_row.append_text(text_delta)
        if was_at_bottom:
            QTimer.singleShot(0, self._scroll_to_bottom)

    def finish_assistant_message(self) -> None:
        self._active_assistant_row = None
        QTimer.singleShot(0, self._scroll_to_bottom)

    def start_typing_indicator(self) -> None:
        if self._typing_row is not None:
            return
        self._typing_row = self._add_row("assistant", "", typing=True)

    def stop_typing_indicator(self) -> None:
        if self._typing_row is None:
            return
        self._remove_row(self._typing_row)
        self._typing_row = None
        QTimer.singleShot(0, self._scroll_to_bottom)

    def add_tool_card(self, tool_name: str, summary: Dict[str, Any], raw: Dict[str, Any]) -> None:
        summary_text = json.dumps(summary, indent=2, default=str)
        raw_text = json.dumps(raw, indent=2, default=str)
        text = f"Tool: {tool_name}\n{summary_text}"
        self._add_row("tool", text, details_text=raw_text)

    def add_error(self, message: str, details: str = "") -> None:
        details_part = f"\nDetails: {details}" if details else ""
        self._add_row("error", f"Error: {message}{details_part}")

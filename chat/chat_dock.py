"""Dockable chat UI components for the FoundationDetector agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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


class ToolCallRow(QWidget):
    """Minimal collapsible row for a single tool invocation."""

    def __init__(self, tool_name: str, summary_text: str, raw_text: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        header = QHBoxLayout()
        self._toggle = QPushButton(f"\u25b8  used {tool_name}")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            "text-align: left; color: #94a3b8; font-size: 9pt; padding: 2px 0; }"
            "QPushButton:hover { color: #475569; }"
        )
        self._toggle.clicked.connect(self._toggle_details)
        header.addWidget(self._toggle)
        header.addStretch(1)
        layout.addLayout(header)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlainText(summary_text + "\n\n--- raw ---\n" + raw_text)
        self._details.setVisible(False)
        self._details.setMaximumHeight(170)
        self._details.setStyleSheet(
            "QTextEdit { background: #f8fafc; border: 1px solid #e2e8f0; "
            "border-radius: 6px; padding: 4px; "
            "font-family: Menlo, Monaco, 'Courier New', monospace; font-size: 11px; }"
        )
        layout.addWidget(self._details)

    def _toggle_details(self) -> None:
        show = not self._details.isVisible()
        self._details.setVisible(show)
        text = self._toggle.text()
        if show:
            self._toggle.setText(text.replace("\u25b8", "\u25be"))
        else:
            self._toggle.setText(text.replace("\u25be", "\u25b8"))
        self.updateGeometry()

    def set_max_width(self, width: int) -> None:
        self._details.setMaximumWidth(width - 16)


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

    send_requested = Signal(str, object)
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
        self.input_box.setPlaceholderText(
            "Ask about loaded data, or use /data, /slide, /models, /regions, /clusters, /atlas, /tool..."
        )
        self.input_box.setFixedHeight(80)
        layout.addWidget(self.input_box)
        self.input_box.installEventFilter(self)

        button_row = QHBoxLayout()
        self.send_button = QPushButton("Send", container)
        self.cancel_button = QPushButton("Cancel", container)
        self.cancel_button.setEnabled(False)
        button_row.addWidget(self.send_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.info_label = QLabel("Msgs: 0 | Session tokens: 0", container)
        self.info_label.setStyleSheet("color: #64748b; font-size: 9pt;")
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(40)
        layout.addWidget(self.info_label)

        self.send_button.clicked.connect(self._on_send)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        self.setWidget(container)

        self._active_assistant_row: Optional[ChatBubbleRow] = None
        self._typing_row: Optional[ChatBubbleRow] = None
        self._rows: List[QWidget] = []
        self._user_message_count: int = 0
        self._assistant_message_count: int = 0
        self._session_total_tokens: int = 0
        self._slash_catalog: List[Dict[str, str]] = [
            {"command": "data", "display": "/data", "description": "List available slides/models/mags/patch sizes"},
            {"command": "slide", "display": "/slide", "description": "Describe one slide/model/mag/patch"},
            {"command": "models", "display": "/models", "description": "Rank models by separability"},
            {"command": "regions", "display": "/regions", "description": "List labeled regions"},
            {"command": "clusters", "display": "/clusters", "description": "Compare selected clusters"},
            {"command": "atlas", "display": "/atlas", "description": "Atlas cluster representation"},
            {"command": "label", "display": "/label", "description": "Label a K-means cluster: /label {\"cluster_id\": 2}"},
            {"command": "select", "display": "/select", "description": "Highlight a cluster: /select {\"cluster_id\": 2}"},
            {"command": "navigate", "display": "/navigate", "description": "Pan/zoom to a region: /navigate {\"region_id\": 0}"},
            {"command": "delete", "display": "/delete", "description": "Delete a region: /delete {\"region_id\": 0}"},
            {"command": "rename", "display": "/rename", "description": "Rename a region: /rename {\"region_id\": 0, \"new_name\": \"Tumor\"}"},
            {"command": "clear", "display": "/clear", "description": "Remove all labeled regions"},
            {"command": "deselect", "display": "/deselect", "description": "Clear cluster selection highlighting"},
            {"command": "similar", "display": "/similar", "description": "Label cluster most similar to a region: /similar {\"region_id\": 0}"},
            {"command": "expand", "display": "/expand", "description": "Expand a region by N grid rings: /expand {\"region_id\": 0, \"n_rings\": 1}"},
            {"command": "different", "display": "/different", "description": "Find most different cluster: /different {\"region_ids\": [0]}"},
            {"command": "atlas-cluster", "display": "/atlas-cluster", "description": "Highlight atlas cluster: /atlas-cluster {\"cluster_id\": 3}"},
            {"command": "tool", "display": "/tool", "description": "Direct tool call: /tool <name> {json}"},
            {"command": "tools", "display": "/tools", "description": "Show slash command help"},
        ]
        self._slash_popup = QFrame(container)
        self._slash_popup.setVisible(False)
        self._slash_popup.setObjectName("slash-command-popup")
        self._slash_popup.setStyleSheet(
            "QFrame#slash-command-popup {"
            "background: #ffffff;"
            "border: 1px solid #cbd5e1;"
            "border-radius: 8px;"
            "}"
            "QListWidget {"
            "border: none;"
            "background: transparent;"
            "color: #0f172a;"
            "outline: none;"
            "}"
            "QListWidget::item {"
            "padding: 6px 8px;"
            "color: #0f172a;"
            "background: transparent;"
            "}"
            "QListWidget::item:selected {"
            "background: #dbeafe;"
            "color: #0f172a;"
            "}"
        )
        popup_layout = QVBoxLayout(self._slash_popup)
        popup_layout.setContentsMargins(2, 2, 2, 2)
        popup_layout.setSpacing(0)
        self._slash_list = QListWidget(self._slash_popup)
        self._slash_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slash_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._slash_list.setWordWrap(False)
        self._slash_list.setUniformItemSizes(True)
        self._slash_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._slash_list.itemClicked.connect(self._on_slash_item_clicked)
        popup_layout.addWidget(self._slash_list)

        # @ mention popup — mutually exclusive with slash popup
        self._at_catalog: List[Dict[str, Any]] = []  # [{name: str, region_id: int}, ...]
        self._at_popup = QFrame(container)
        self._at_popup.setVisible(False)
        self._at_popup.setObjectName("at-mention-popup")
        self._at_popup.setStyleSheet(
            "QFrame#at-mention-popup {"
            "background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; }"
            "QListWidget { border: none; background: transparent; color: #0f172a; outline: none; }"
            "QListWidget::item { padding: 6px 8px; color: #0f172a; background: transparent; }"
            "QListWidget::item:selected { background: #d1fae5; color: #0f172a; }"
        )
        at_popup_layout = QVBoxLayout(self._at_popup)
        at_popup_layout.setContentsMargins(2, 2, 2, 2)
        at_popup_layout.setSpacing(0)
        self._at_list = QListWidget(self._at_popup)
        self._at_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._at_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._at_list.setWordWrap(False)
        self._at_list.setUniformItemSizes(True)
        self._at_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._at_list.itemClicked.connect(self._on_at_item_clicked)
        at_popup_layout.addWidget(self._at_list)

        self.input_box.textChanged.connect(self._on_input_text_changed)

    def _on_send(self) -> None:
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self._hide_slash_popup()
        self._hide_at_popup()
        self.input_box.clear()
        slash_command, parse_error = self._parse_slash_command(text)
        at_regions, at_unresolved = self._resolve_at_mentions(text, self._at_catalog)
        metadata = {
            "slash_command": slash_command,
            "slash_parse_error": parse_error,
            "at_regions": at_regions,
            "at_unresolved": at_unresolved,
        }
        self.send_requested.emit(text, metadata)

    @staticmethod
    def _parse_slash_command(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse user-facing slash commands into a normalized command envelope."""
        if not text.startswith("/"):
            return None, None

        body = text[1:].strip()
        if not body:
            return None, "Empty slash command"

        if body in ("help", "tools"):
            return {"kind": "help", "raw": text}, None

        parts = body.split(None, 1)
        command = parts[0].lower().strip()
        arg_text = parts[1].strip() if len(parts) > 1 else ""

        alias_map = {
            "data": "list_data",
            "slide": "describe_slide",
            "models": "rank_models_by_separability",
            "regions": "list_labeled_regions",
            "clusters": "compare_selected_clusters",
            "atlas": "atlas_cluster_representation",
            "label": "label_cluster",
            "select": "select_cluster",
            "navigate": "navigate_to_region",
            "delete": "delete_region",
            "rename": "rename_region",
            "clear": "clear_all_regions",
            "deselect": "deselect_all_clusters",
            "similar": "label_similar_cluster",
            "expand": "expand_region",
            "different": "find_most_different_cluster",
            "atlas-cluster": "highlight_atlas_cluster",
        }

        if command == "tool":
            if not arg_text:
                return None, "Usage: /tool <tool_name> {json_args}"
            tool_parts = arg_text.split(None, 1)
            tool_name = tool_parts[0].strip()
            args_text = tool_parts[1].strip() if len(tool_parts) > 1 else ""
            arguments: Dict[str, Any] = {}
            if args_text:
                try:
                    parsed = json.loads(args_text)
                except Exception as exc:
                    return None, f"Invalid JSON arguments: {exc}"
                if not isinstance(parsed, dict):
                    return None, "Tool arguments must be a JSON object"
                arguments = parsed
            return {
                "kind": "tool",
                "raw": text,
                "tool_name": tool_name,
                "arguments": arguments,
            }, None

        if command not in alias_map:
            return None, f"Unknown slash command: /{command}"

        arguments = {}
        if arg_text:
            try:
                parsed = json.loads(arg_text)
            except Exception as exc:
                return None, f"Invalid JSON arguments: {exc}"
            if not isinstance(parsed, dict):
                return None, "Command arguments must be a JSON object"
            arguments = parsed

        return {
            "kind": "tool",
            "raw": text,
            "tool_name": alias_map[command],
            "arguments": arguments,
            "alias": command,
        }, None

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
        if self._slash_popup.isVisible():
            self._position_slash_popup()
        if self._at_popup.isVisible():
            self._position_at_popup()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.input_box:
            if event.type() == QEvent.Type.FocusOut:
                self._hide_slash_popup()
                self._hide_at_popup()
            if event.type() == QEvent.Type.KeyPress:
                if self._at_popup.isVisible():
                    key = event.key()
                    if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                        self._move_at_selection(+1 if key == Qt.Key.Key_Down else -1)
                        return True
                    if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                        self._commit_at_selection()
                        return True
                    if key == Qt.Key.Key_Escape:
                        self._hide_at_popup()
                        return True
                if self._slash_popup.isVisible():
                    key = event.key()
                    if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                        self._move_slash_selection(+1 if key == Qt.Key.Key_Down else -1)
                        return True
                    if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                        self._commit_slash_selection()
                        return True
                    if key == Qt.Key.Key_Escape:
                        self._hide_slash_popup()
                        return True
        return super().eventFilter(obj, event)

    def _on_input_text_changed(self) -> None:
        text = self.input_box.toPlainText()
        # Slash takes priority
        slash_prefix = self._extract_slash_prefix(text)
        if slash_prefix is not None:
            self._hide_at_popup()
            matches = self._get_slash_matches(slash_prefix, self._slash_catalog)
            self._update_slash_popup(matches)
            return
        self._hide_slash_popup()
        # @ popup
        cursor_pos = self.input_box.textCursor().position()
        at_prefix = self._extract_at_prefix(text, cursor_pos)
        if at_prefix is not None and self._at_catalog:
            matches = self._get_at_matches(at_prefix, self._at_catalog)
            self._update_at_popup(matches)
        else:
            self._hide_at_popup()

    def _update_slash_popup(self, matches: List[Dict[str, str]]) -> None:
        if not matches:
            self._hide_slash_popup()
            return

        self._slash_list.clear()
        for entry in matches:
            item = QListWidgetItem(
                f"{entry['display']}  {entry['description']}",
                self._slash_list,
            )
            item.setData(Qt.ItemDataRole.UserRole, entry["command"])
            item.setSizeHint(QSize(item.sizeHint().width(), 28))

        self._slash_list.setCurrentRow(0)
        self._position_slash_popup()
        self._slash_popup.setVisible(True)
        self._slash_popup.raise_()

    def _position_slash_popup(self) -> None:
        container = self.widget()
        if container is None:
            return

        row_count = self._slash_list.count()
        visible_rows = min(6, max(1, row_count))
        row_height = 28
        preferred_height = visible_rows * row_height + 6
        input_geom = self.input_box.geometry()
        space_below = max(0, container.height() - (input_geom.bottom() + 4))
        space_above = max(0, input_geom.y() - 4)
        below_viable = space_below >= 80

        if below_viable:
            target_height = min(preferred_height, space_below)
            y_pos = input_geom.bottom() + 4
        else:
            target_height = min(preferred_height, max(80, space_above))
            y_pos = max(0, input_geom.y() - 4 - target_height)

        self._slash_popup.setGeometry(
            input_geom.x(),
            y_pos,
            input_geom.width(),
            target_height,
        )

    def _hide_slash_popup(self) -> None:
        self._slash_popup.setVisible(False)
        self._slash_list.clear()

    def _move_slash_selection(self, delta: int) -> None:
        count = self._slash_list.count()
        if count <= 0:
            return
        index = self._slash_list.currentRow()
        if index < 0:
            index = 0
        next_index = (index + delta) % count
        self._slash_list.setCurrentRow(next_index)

    def _on_slash_item_clicked(self, item: QListWidgetItem) -> None:
        del item
        self._commit_slash_selection()

    def _commit_slash_selection(self) -> None:
        item = self._slash_list.currentItem()
        if item is None:
            return
        command = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not command:
            return

        text = self.input_box.toPlainText()
        insert_text = f"/{command} "
        if text.lstrip().startswith("/"):
            leading_ws = len(text) - len(text.lstrip())
            stripped = text.lstrip()
            command_token = stripped.split(None, 1)[0] if stripped else ""
            token_end = leading_ws + len(command_token)
            new_text = text[:leading_ws] + insert_text + text[token_end:].lstrip()
            cursor_pos = leading_ws + len(insert_text)
        else:
            new_text = insert_text + text
            cursor_pos = len(insert_text)

        self.input_box.blockSignals(True)
        self.input_box.setPlainText(new_text)
        self.input_box.blockSignals(False)
        cursor = self.input_box.textCursor()
        cursor.setPosition(cursor_pos)
        self.input_box.setTextCursor(cursor)
        self._hide_slash_popup()

    @staticmethod
    def _extract_slash_prefix(text: str) -> Optional[str]:
        """Return slash command prefix while user is typing the command token."""
        if not text:
            return None
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return None
        if any(ch.isspace() for ch in stripped[1:]):
            # Once command token is complete (space typed), hide suggestions.
            return None
        return stripped[1:].lower()

    @staticmethod
    def _get_slash_matches(prefix: str, catalog: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return slash command options matching the typed prefix."""
        prefix = prefix.strip().lower()
        return [
            entry for entry in catalog
            if entry.get("command", "").lower().startswith(prefix)
        ]

    # ------------------------------------------------------------------
    # @ mention support
    # ------------------------------------------------------------------

    def update_region_catalog(self, regions: List[Dict[str, Any]]) -> None:
        """Update the @ mention catalog. regions = [{name, region_id}, ...]"""
        self._at_catalog = regions
        # Re-filter if popup is currently open
        if self._at_popup.isVisible():
            text = self.input_box.toPlainText()
            cursor_pos = self.input_box.textCursor().position()
            prefix = self._extract_at_prefix(text, cursor_pos)
            if prefix is None:
                self._hide_at_popup()
            else:
                self._update_at_popup(self._get_at_matches(prefix, self._at_catalog))

    @staticmethod
    def _extract_at_prefix(text: str, cursor_pos: int) -> Optional[str]:
        """Return text after the last unambiguous @ before cursor, or None.

        @ must be at start-of-text or preceded by whitespace (avoids emails).
        """
        segment = text[:cursor_pos]
        at_idx = segment.rfind('@')
        if at_idx < 0:
            return None
        # @ must be at start or preceded by whitespace
        if at_idx > 0 and not segment[at_idx - 1].isspace():
            return None
        return segment[at_idx + 1:]  # "" means show all; "Reg" means filter

    @staticmethod
    def _get_at_matches(prefix: str, catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prefix_lower = prefix.lower()
        return [e for e in catalog if e["name"].lower().startswith(prefix_lower)]

    @staticmethod
    def _resolve_at_mentions(
        text: str, catalog: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Scan text for @name tokens; resolve against catalog using longest-match.

        Returns (resolved_list, unresolved_name_strings).
        """
        resolved: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        seen_ids: set = set()
        sorted_catalog = sorted(catalog, key=lambda e: len(e["name"]), reverse=True)
        i = 0
        while i < len(text):
            if text[i] == '@' and (i == 0 or text[i - 1].isspace()):
                after = text[i + 1:]
                matched = None
                for entry in sorted_catalog:
                    name = entry["name"]
                    if after.startswith(name):
                        tail_pos = len(name)
                        # Name must be followed by whitespace, @, newline, or end-of-string
                        if tail_pos == len(after) or after[tail_pos] in (' ', '\t', '\n', '@'):
                            matched = entry
                            break
                if matched:
                    if matched["region_id"] not in seen_ids:
                        resolved.append(matched)
                        seen_ids.add(matched["region_id"])
                    i += 1 + len(matched["name"])
                    continue
                else:
                    # Collect unresolved token up to next whitespace
                    end = i + 1
                    while end < len(text) and not text[end].isspace():
                        end += 1
                    token = text[i + 1:end].strip()
                    if token:
                        unresolved.append(token)
                    i = end
                    continue
            i += 1
        return resolved, unresolved

    def _update_at_popup(self, matches: List[Dict[str, Any]]) -> None:
        if not matches:
            self._hide_at_popup()
            return
        self._hide_slash_popup()  # mutually exclusive
        self._at_list.clear()
        for entry in matches:
            item = QListWidgetItem(f"@{entry['name']}", self._at_list)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setSizeHint(QSize(item.sizeHint().width(), 28))
        self._at_list.setCurrentRow(0)
        self._position_at_popup()
        self._at_popup.setVisible(True)
        self._at_popup.raise_()

    def _hide_at_popup(self) -> None:
        self._at_popup.setVisible(False)
        self._at_list.clear()

    def _position_at_popup(self) -> None:
        container = self.widget()
        if container is None:
            return
        row_count = self._at_list.count()
        visible_rows = min(6, max(1, row_count))
        row_height = 28
        preferred_height = visible_rows * row_height + 6
        input_geom = self.input_box.geometry()
        space_above = max(0, input_geom.y() - 4)
        space_below = max(0, container.height() - (input_geom.bottom() + 4))
        below_viable = space_below >= 80
        if below_viable:
            target_height = min(preferred_height, space_below)
            y_pos = input_geom.bottom() + 4
        else:
            target_height = min(preferred_height, max(80, space_above))
            y_pos = max(0, input_geom.y() - 4 - target_height)
        self._at_popup.setGeometry(input_geom.x(), y_pos, input_geom.width(), target_height)

    def _move_at_selection(self, delta: int) -> None:
        count = self._at_list.count()
        if count <= 0:
            return
        index = self._at_list.currentRow()
        self._at_list.setCurrentRow((max(0, index) + delta) % count)

    def _commit_at_selection(self) -> None:
        item = self._at_list.currentItem()
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not entry:
            return
        name = entry["name"]
        text = self.input_box.toPlainText()
        cursor_pos = self.input_box.textCursor().position()
        segment = text[:cursor_pos]
        at_idx = segment.rfind('@')
        if at_idx < 0:
            self._hide_at_popup()
            return
        insert_text = f"@{name} "
        new_text = text[:at_idx] + insert_text + text[cursor_pos:]
        new_cursor_pos = at_idx + len(insert_text)
        self.input_box.blockSignals(True)
        self.input_box.setPlainText(new_text)
        self.input_box.blockSignals(False)
        cursor = self.input_box.textCursor()
        cursor.setPosition(new_cursor_pos)
        self.input_box.setTextCursor(cursor)
        self._hide_at_popup()

    def _on_at_item_clicked(self, item: QListWidgetItem) -> None:
        self._at_list.setCurrentItem(item)
        self._commit_at_selection()

    def _remove_row(self, row: QWidget) -> None:
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
        self._user_message_count += 1
        self._refresh_info_label()

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
        if self._active_assistant_row is not None:
            self._assistant_message_count += 1
            self._refresh_info_label()
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
        was_at_bottom = self._is_at_bottom()
        row = ToolCallRow(tool_name, summary_text, raw_text, parent=self.messages_container)
        self._rows.append(row)
        self.messages_layout.addWidget(row)
        self._update_row_widths()
        if was_at_bottom:
            QTimer.singleShot(0, self._scroll_to_bottom)

    def add_error(self, message: str, details: str = "") -> None:
        details_part = f"\nDetails: {details}" if details else ""
        self._add_row("error", f"Error: {message}{details_part}")

    def record_response_usage(self, usage: Dict[str, Any]) -> None:
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int):
            self._session_total_tokens += total_tokens
            self._refresh_info_label()

    def _refresh_info_label(self) -> None:
        total_messages = self._user_message_count + self._assistant_message_count
        self.info_label.setText(
            f"Msgs: {total_messages} (U:{self._user_message_count}/A:{self._assistant_message_count})"
            f" | Session tokens: {self._session_total_tokens}"
        )

"""
model_selector.py
=================

Widget classes for model/magnification/patch-size selection.

Contains:
- CheckableComboBoxModel: QAbstractListModel backing a checkable combobox.
- ModelMultiSelector: Multi-select dropdown combobox for choosing models.
- ModelSelectionDialog: Dialog wrapping ModelMultiSelector with mag/patch combos.
- _qcolors_to_hsl_strings: Free helper converting QColor list to HSL strings.
"""
from __future__ import annotations

from typing import List, Optional

import data_loader
from PySide6.QtCore import QAbstractListModel, QModelIndex, Signal
from PySide6.QtCore import Qt as QtCore
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor

from gui_types import ModelSelection


class CheckableComboBoxModel(QAbstractListModel):
    """Model for QComboBox that supports checkable items."""

    def __init__(self, items: List[str] = None, parent=None):
        super().__init__(parent)
        self._items = items or []
        self._checked = {item: False for item in self._items}
        self._enabled = {item: True for item in self._items}
        self._tooltips = {item: "" for item in self._items}

    def rowCount(self, parent=QModelIndex()):
        return len(self._items)

    def data(self, index, role=QtCore.DisplayRole):
        if not index.isValid():
            return None

        item = self._items[index.row()]

        if role == QtCore.DisplayRole:
            return item
        elif role == QtCore.CheckStateRole:
            return QtCore.Checked if self._checked[item] else QtCore.Unchecked
        elif role == QtCore.ToolTipRole:
            return self._tooltips.get(item, "")

        return None

    def setData(self, index, value, role=QtCore.CheckStateRole):
        if not index.isValid():
            return False

        item = self._items[index.row()]

        if role == QtCore.CheckStateRole and self._enabled[item]:
            self._checked[item] = (value == QtCore.Checked)
            self.dataChanged.emit(index, index, [role])
            return True

        return False

    def flags(self, index):
        if not index.isValid():
            return QtCore.NoItemFlags

        item = self._items[index.row()]
        flags = QtCore.ItemIsUserCheckable

        if self._enabled[item]:
            flags |= QtCore.ItemIsEnabled

        return flags

    def setItems(self, items: List[str]):
        """Set the list of items."""
        self.beginResetModel()
        self._items = items
        self._checked = {item: False for item in items}
        self._enabled = {item: True for item in items}
        self._tooltips = {item: "" for item in items}
        self.endResetModel()

    def getCheckedItems(self) -> List[str]:
        """Return list of checked items."""
        return [item for item in self._items if self._checked[item]]

    def setItemEnabled(self, item: str, enabled: bool):
        """Enable or disable a specific item."""
        if item in self._enabled:
            self._enabled[item] = enabled
            if not enabled:
                self._checked[item] = False
            # Find index and emit dataChanged
            try:
                idx = self._items.index(item)
                model_index = self.index(idx)
                self.dataChanged.emit(model_index, model_index)
            except ValueError:
                pass

    def setItemToolTip(self, item: str, tooltip: str):
        """Set tooltip for a specific item."""
        if item in self._tooltips:
            self._tooltips[item] = tooltip

    def setCheckedItems(self, items: List[str]) -> None:
        """Set the checked state for multiple items."""
        target = set(items)
        for item in self._items:
            self._checked[item] = item in target and self._enabled[item]
        if self._items:
            top = self.index(0)
            bottom = self.index(len(self._items) - 1)
            self.dataChanged.emit(top, bottom, [QtCore.CheckStateRole])


class ModelMultiSelector(QComboBox):
    """Dropdown combobox with checkable items for multi-model selection.

    This widget looks like a standard QComboBox but allows multiple
    selections via checkboxes in the dropdown list. Selected models
    are displayed as comma-separated text when the dropdown is closed.

    Signals
    -------
    selectionChanged : Signal
        Emitted when the set of selected models changes.
    """
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up model
        self._model = CheckableComboBoxModel(parent=self)
        self.setModel(self._model)

        # Prevent default selection behavior
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select models...")

        # Connect model changes to update display text
        self._model.dataChanged.connect(self._update_text)

        # Prevent closing dropdown on item click
        self.view().viewport().installEventFilter(self)

        # Track selection state for deferred updates
        self._selection_changed_while_open = False
        self._previous_selection = set()

    def showPopup(self):
        """Track selection state when popup opens."""
        self._previous_selection = set(self.getSelectedModels())
        self._selection_changed_while_open = False
        print(f"DEBUG: Dropdown opened with selection: {self._previous_selection}")
        super().showPopup()

    def hidePopup(self):
        """Emit selectionChanged signal only when popup closes if selection changed."""
        super().hidePopup()

        # Check if selection actually changed
        current_selection = set(self.getSelectedModels())
        if current_selection != self._previous_selection:
            print(f"DEBUG: Dropdown closed, selection changed from {self._previous_selection} to {current_selection}")
            self.selectionChanged.emit()
        else:
            print("DEBUG: Dropdown closed, no selection change")

    def eventFilter(self, obj, event):
        """Prevent dropdown from closing when clicking checkboxes."""
        if obj == self.view().viewport():
            if event.type() == event.Type.MouseButtonRelease:
                # Get the index at click position
                index = self.view().indexAt(event.pos())
                if index.isValid():
                    # Toggle check state
                    current_state = self._model.data(index, QtCore.CheckStateRole)
                    new_state = QtCore.Unchecked if current_state == QtCore.Checked else QtCore.Checked
                    self._model.setData(index, new_state, QtCore.CheckStateRole)
                    self._selection_changed_while_open = True  # Mark that change occurred
                return True  # Prevent dropdown from closing

        return super().eventFilter(obj, event)

    def _update_text(self):
        """Update the display text to show selected items.

        Note: Signal emission is deferred until dropdown closes (see hidePopup).
        """
        selected = self._model.getCheckedItems()
        if selected:
            self.lineEdit().setText(", ".join(selected))
        else:
            self.lineEdit().setText("")
        # Signal will be emitted only when dropdown closes

    def clear(self):
        """Clear all items."""
        self._model.setItems([])
        self.lineEdit().clear()

    def addItems(self, items: List[str]):
        """Add model names as checkable items."""
        self._model.setItems(items)
        self.lineEdit().clear()

    def getSelectedModels(self) -> List[str]:
        """Return list of selected model names."""
        return self._model.getCheckedItems()

    def getAllModelNames(self) -> List[str]:
        """Return list of all available model names (not just selected)."""
        return self._model._items.copy()

    def setModelEnabled(self, model_name: str, enabled: bool):
        """Enable or disable a specific model item."""
        self._model.setItemEnabled(model_name, enabled)

    def setModelToolTip(self, model_name: str, tooltip: str):
        """Set tooltip for a specific model item."""
        self._model.setItemToolTip(model_name, tooltip)

    def setSelectedModels(self, models: List[str]) -> None:
        """Programmatically select a list of models."""
        self._model.setCheckedItems(models)
        self._update_text()


class ModelSelectionDialog(QDialog):
    """Dialog for selecting models, magnification, and patch size."""

    def __init__(
        self,
        slide_info: data_loader.SlideInfo,
        current_selection: Optional[ModelSelection] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Models")
        self._slide_info = slide_info
        self._current_selection = current_selection

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._model_selector = ModelMultiSelector()
        self._model_selector.addItems(sorted(slide_info.models.keys()))
        self._model_selector.selectionChanged.connect(self._on_model_selection_changed)

        self._mag_combo = QComboBox()
        self._mag_combo.currentTextChanged.connect(self._on_mag_changed)

        self._patch_combo = QComboBox()
        self._patch_combo.currentTextChanged.connect(self._update_ok_state)

        layout.addWidget(QLabel("Models:"))
        layout.addWidget(self._model_selector)
        layout.addWidget(QLabel("Magnification:"))
        layout.addWidget(self._mag_combo)
        layout.addWidget(QLabel("Patch size:"))
        layout.addWidget(self._patch_combo)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        if current_selection:
            available_models = set(slide_info.models.keys())
            preselect = [m for m in current_selection.models if m in available_models]
            self._model_selector.setSelectedModels(preselect)

        self._update_magnifications(prefer=current_selection.magnification if current_selection else None)
        self._update_patches(prefer=current_selection.patch_size if current_selection else None)
        self._update_ok_state()

    def selection(self) -> Optional[ModelSelection]:
        """Return the selected configuration if valid."""
        models = self._model_selector.getSelectedModels()
        mag = self._mag_combo.currentText()
        patch = self._patch_combo.currentText()
        if not models or not mag or not patch:
            return None
        return ModelSelection(models=models, magnification=mag, patch_size=patch)

    def _on_model_selection_changed(self) -> None:
        self._update_magnifications()
        self._update_patches()
        self._update_ok_state()

    def _on_mag_changed(self, mag: str) -> None:
        self._update_patches()
        self._update_ok_state()

    def _update_magnifications(self, prefer: Optional[str] = None) -> None:
        models = self._model_selector.getSelectedModels()
        mags = self._get_common_magnifications(models)
        self._mag_combo.blockSignals(True)
        self._mag_combo.clear()
        self._mag_combo.addItems(mags)
        if prefer and prefer in mags:
            self._mag_combo.setCurrentText(prefer)
        self._mag_combo.blockSignals(False)

    def _update_patches(self, prefer: Optional[str] = None) -> None:
        models = self._model_selector.getSelectedModels()
        mag = self._mag_combo.currentText()
        patches = self._get_common_patches(models, mag)
        self._patch_combo.blockSignals(True)
        self._patch_combo.clear()
        self._patch_combo.addItems(patches)
        if prefer and prefer in patches:
            self._patch_combo.setCurrentText(prefer)
        self._patch_combo.blockSignals(False)

    def _update_ok_state(self) -> None:
        selection = self.selection()
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(selection is not None)

    def _get_common_magnifications(self, models: List[str]) -> List[str]:
        mag_sets = []
        for model_name in models:
            model_dict = self._slide_info.models.get(model_name, {})
            mag_sets.append(set(model_dict.keys()))
        if not mag_sets:
            return []
        return sorted(set.intersection(*mag_sets))

    def _get_common_patches(self, models: List[str], mag: str) -> List[str]:
        if not models or not mag:
            return []
        patch_sets = []
        for model_name in models:
            patches = self._slide_info.models.get(model_name, {}).get(mag, {})
            patch_sets.append(set(patches.keys()))
        if not patch_sets:
            return []
        return sorted(set.intersection(*patch_sets))


def _qcolors_to_hsl_strings(colors: List[QColor]) -> List[str]:
    """Convert a list of QColor objects to HSL CSS strings understood by the views."""
    result = []
    for c in colors:
        h = max(0, c.hslHue())
        s = int(c.hslSaturationF() * 100)
        l = int(c.lightnessF() * 100)
        result.append(f"hsl({h},{s}%,{l}%)")
    return result

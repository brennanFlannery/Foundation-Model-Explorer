"""
preferences_dialog.py
=====================

Preferences dialog for FoundationDetector application.

This module provides a dialog for configuring application preferences,
including feature normalization settings for multi-model analysis.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QDialogButtonBox,
    QLabel,
    QGroupBox,
)
from PySide6.QtCore import QSettings


class PreferencesDialog(QDialog):
    """Preferences dialog for application settings.
    
    This dialog provides user-configurable settings for the application,
    with persistent storage using QSettings. Currently supports:
    
    - Feature normalization toggle for multi-model analysis
    
    Settings are automatically saved when the user clicks OK and loaded
    from QSettings on initialization.
    
    Attributes
    ----------
    settings : QSettings
        Persistent settings storage using organization "FoundationDetector"
        and application name "FoundationDetector".
    normalize_checkbox : QCheckBox
        Checkbox controlling whether features are z-score normalized
        before concatenation in multi-model mode.
    """
    
    def __init__(self, parent=None):
        """Initialize the preferences dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(400)
        self.settings = QSettings("FoundationDetector", "FoundationDetector")
        
        layout = QVBoxLayout(self)
        
        # Feature Processing Group
        feature_group = QGroupBox("Feature Processing")
        feature_layout = QVBoxLayout()
        
        # Normalization checkbox
        self.normalize_checkbox = QCheckBox("Normalize features across models (Z-score)")
        self.normalize_checkbox.setChecked(
            self.settings.value("normalize_features", True, type=bool)
        )
        feature_layout.addWidget(self.normalize_checkbox)
        
        # Help text
        help_text = QLabel(
            "When enabled, features from each model are z-score normalized "
            "(mean=0, std=1) before concatenation. This ensures equal "
            "contribution from all models regardless of feature dimension or scale.\n\n"
            "Recommended: ON for multi-model analysis."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray; font-size: 9pt;")
        feature_layout.addWidget(help_text)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        # Add stretch to push buttons to bottom
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def accept(self):
        """Save settings and close dialog."""
        # Save settings
        self.settings.setValue("normalize_features", self.normalize_checkbox.isChecked())
        super().accept()

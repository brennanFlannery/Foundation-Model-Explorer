"""
preferences_dialog.py
=====================

Preferences dialog for FoundationDetector application.

This module provides a dialog for configuring application preferences,
including feature normalization settings for multi-model analysis.
"""

import os

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QDialogButtonBox,
    QLabel,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QComboBox,
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
        default_secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        
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

        # Region Embeddings Group
        region_group = QGroupBox("Region Embeddings")
        region_layout = QVBoxLayout()

        patches_row = QHBoxLayout()
        patches_row.addWidget(QLabel("Patches per region (M):"))
        self.patches_per_region_spin = QSpinBox()
        self.patches_per_region_spin.setRange(3, 500)
        self.patches_per_region_spin.setValue(
            self.settings.value("patches_per_region", 15, type=int)
        )
        patches_row.addWidget(self.patches_per_region_spin)
        patches_row.addStretch()
        region_layout.addLayout(patches_row)

        region_help = QLabel(
            "N = ceil(cluster_size / M) subclusters per K-means group. "
            "Each subcluster appears as an X marker in the Region Embedding View."
        )
        region_help.setWordWrap(True)
        region_help.setStyleSheet("color: gray; font-size: 9pt;")
        region_layout.addWidget(region_help)

        region_group.setLayout(region_layout)
        layout.addWidget(region_group)

        # AI Agent Group
        agent_group = QGroupBox("AI Agent")
        agent_layout = QVBoxLayout()

        self.chat_enabled_checkbox = QCheckBox("Enable in-app chat agent")
        self.chat_enabled_checkbox.setChecked(
            self.settings.value("chat_enabled", True, type=bool)
        )
        agent_layout.addWidget(self.chat_enabled_checkbox)

        secrets_row = QHBoxLayout()
        secrets_row.addWidget(QLabel("Secrets (.env) path:"))
        self.chat_secrets_path = QLineEdit()
        self.chat_secrets_path.setText(
            self.settings.value("chat_secrets_path", default_secrets_path, type=str)
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_secrets_path)
        secrets_row.addWidget(self.chat_secrets_path, stretch=1)
        secrets_row.addWidget(browse_btn)
        agent_layout.addLayout(secrets_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.chat_model_combo = QComboBox()
        self.chat_model_combo.addItems([
            "gpt-4o-mini",
            "gpt-4o",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
        ])
        self.chat_model_combo.setEditable(True)
        selected_model = self.settings.value("chat_model", "gpt-4o-mini", type=str)
        index = self.chat_model_combo.findText(selected_model)
        if index >= 0:
            self.chat_model_combo.setCurrentIndex(index)
        else:
            self.chat_model_combo.setCurrentText(selected_model)
        model_row.addWidget(self.chat_model_combo, stretch=1)
        agent_layout.addLayout(model_row)

        timeout_row = QHBoxLayout()
        timeout_row.addWidget(QLabel("LLM timeout (s):"))
        self.chat_llm_timeout_spin = QSpinBox()
        self.chat_llm_timeout_spin.setRange(5, 600)
        self.chat_llm_timeout_spin.setValue(
            self.settings.value("chat_llm_timeout_s", 60, type=int)
        )
        timeout_row.addWidget(self.chat_llm_timeout_spin)
        timeout_row.addWidget(QLabel("Tool timeout (s):"))
        self.chat_tool_timeout_spin = QSpinBox()
        self.chat_tool_timeout_spin.setRange(5, 600)
        self.chat_tool_timeout_spin.setValue(
            self.settings.value("chat_tool_timeout_s", 30, type=int)
        )
        timeout_row.addWidget(self.chat_tool_timeout_spin)
        agent_layout.addLayout(timeout_row)

        help_text_agent = QLabel(
            "Chat uses LiteLLM with OpenAI-compatible settings. "
            "Provide a local .env file containing OPENAI_API_KEY. "
            "MCP tools are served in-process by FoundationDetector."
        )
        help_text_agent.setWordWrap(True)
        help_text_agent.setStyleSheet("color: gray; font-size: 9pt;")
        agent_layout.addWidget(help_text_agent)

        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
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
        self.settings.setValue("patches_per_region", self.patches_per_region_spin.value())
        self.settings.setValue("chat_enabled", self.chat_enabled_checkbox.isChecked())
        self.settings.setValue("chat_secrets_path", self.chat_secrets_path.text().strip())
        self.settings.setValue("chat_model", self.chat_model_combo.currentText().strip())
        self.settings.setValue("chat_llm_timeout_s", self.chat_llm_timeout_spin.value())
        self.settings.setValue("chat_tool_timeout_s", self.chat_tool_timeout_spin.value())
        super().accept()

    def _browse_secrets_path(self) -> None:
        """Prompt user to select a .env file path."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select .env File",
            "",
            "Env Files (*.env *.txt);;All Files (*)",
        )
        if selected:
            self.chat_secrets_path.setText(selected)

from __future__ import annotations

from pathlib import Path
from typing import List
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SettingsDialog(QDialog):
    """Application preferences editor."""

    recent_files_changed = pyqtSignal(list)

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Settings")
        self.resize(520, 420)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Follow System", "system")
        self._theme_combo.addItem("Light", "light")
        self._theme_combo.addItem("Dark", "dark")

        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout(theme_group)
        theme_layout.addRow(QLabel("Application theme:"), self._theme_combo)

        self._recent_list = QListWidget()
        self._recent_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        recent_buttons = QHBoxLayout()
        add_btn = QPushButton("Add...")
        add_btn.clicked.connect(self._on_add_recent)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_recent)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_recent)
        recent_buttons.addWidget(add_btn)
        recent_buttons.addWidget(remove_btn)
        recent_buttons.addWidget(clear_btn)
        recent_buttons.addStretch(1)

        recent_group = QGroupBox("Recent Files")
        recent_layout = QVBoxLayout(recent_group)
        recent_layout.addWidget(
            QLabel("Manage the list of files shown in the Open Recent menu.")
        )
        recent_layout.addWidget(self._recent_list)
        recent_layout.addLayout(recent_buttons)

        self._export_dir_edit = QLineEdit()
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._on_browse_export_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._export_dir_edit, 1)
        dir_row.addWidget(browse_dir_btn)

        self._export_name_edit = QLineEdit()
        self._export_name_edit.setPlaceholderText("spectra.xlsx")

        export_group = QGroupBox("Export Defaults")
        export_layout = QFormLayout(export_group)
        export_layout.addRow(QLabel("Default folder:"), self._wrap_row(dir_row))
        export_layout.addRow(QLabel("Default file name:"), self._export_name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(theme_group)
        layout.addWidget(recent_group, 1)
        layout.addWidget(export_group)
        layout.addStretch(1)
        layout.addWidget(buttons)

        self._load_from_settings()

    # ------------------------------------------------------------------
    # Helpers
    def values(self) -> dict:
        return {
            "theme": self._theme_combo.currentData(),
            "recent_files": self._collect_recent_files(),
            "export_default_dir": self._export_dir_edit.text().strip(),
            "export_default_name": self._export_name_edit.text().strip(),
        }

    def _load_from_settings(self):
        theme_value = str(self._settings.value("theme", "system") or "system").lower()
        index = self._theme_combo.findData(theme_value)
        if index >= 0:
            self._theme_combo.setCurrentIndex(index)

        recent = self._settings.value("recentFiles", [])
        for path in self._normalise_path_list(recent):
            self._recent_list.addItem(path)

        export_dir = self._settings.value("export/defaultDir", "") or ""
        export_name = self._settings.value("export/defaultName", "") or ""
        self._export_dir_edit.setText(str(export_dir))
        self._export_name_edit.setText(str(export_name))

    def _normalise_path_list(self, value: object) -> List[str]:
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value if v]
        elif isinstance(value, str):
            items = [v for v in value.split(";") if v]
        else:
            items = []
        # Deduplicate while preserving order
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _collect_recent_files(self) -> List[str]:
        return [self._recent_list.item(i).text() for i in range(self._recent_list.count())]

    def _on_add_recent(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add to Recent Files")
        if not paths:
            return
        current = self._collect_recent_files()
        for path in paths:
            if path and path not in current:
                self._recent_list.addItem(path)
                current.append(path)
        self._emit_recent_changed()

    def _on_remove_recent(self):
        for item in self._recent_list.selectedItems():
            row = self._recent_list.row(item)
            self._recent_list.takeItem(row)
        self._emit_recent_changed()

    def _on_clear_recent(self):
        self._recent_list.clear()
        self._emit_recent_changed()

    def _on_browse_export_dir(self):
        start_dir = self._export_dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Export Folder", start_dir)
        if directory:
            self._export_dir_edit.setText(directory)

    def _wrap_row(self, layout: QHBoxLayout) -> QWidget:
        container = QWidget()
        container.setLayout(layout)
        return container

    def _emit_recent_changed(self):
        self.recent_files_changed.emit(self._collect_recent_files())

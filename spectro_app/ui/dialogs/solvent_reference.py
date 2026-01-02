from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PyQt6 import QtCore, QtWidgets


@dataclass
class SolventReferenceMetadata:
    name: str
    tags: List[str]
    metadata: Dict[str, str]
    save_to_store: bool
    set_default: bool


class SolventReferenceMetadataDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        name_default: str = "",
        tags_default: Iterable[str] = (),
        allow_save_toggle: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Solvent Reference Metadata")
        self.resize(460, 320)

        self._name_edit = QtWidgets.QLineEdit(name_default)
        self._tags_edit = QtWidgets.QLineEdit(", ".join(str(tag) for tag in tags_default))
        self._identity_edit = QtWidgets.QLineEdit()
        self._mode_edit = QtWidgets.QLineEdit()
        self._temperature_edit = QtWidgets.QLineEdit()
        self._instrument_edit = QtWidgets.QLineEdit()
        self._date_edit = QtWidgets.QLineEdit()

        form = QtWidgets.QFormLayout()
        form.addRow("Name:", self._name_edit)
        form.addRow("Tags (comma-separated):", self._tags_edit)
        form.addRow("Solvent identity:", self._identity_edit)
        form.addRow("Measurement mode:", self._mode_edit)
        form.addRow("Temperature:", self._temperature_edit)
        form.addRow("Instrument:", self._instrument_edit)
        form.addRow("Date:", self._date_edit)

        self._save_checkbox = QtWidgets.QCheckBox("Save to solvent references")
        self._save_checkbox.setChecked(True)
        self._save_checkbox.setVisible(allow_save_toggle)

        self._default_checkbox = QtWidgets.QCheckBox("Mark as default")
        self._default_checkbox.setChecked(False)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._save_checkbox)
        layout.addWidget(self._default_checkbox)
        layout.addStretch(1)
        layout.addWidget(buttons)

    def metadata(self) -> SolventReferenceMetadata:
        name = self._name_edit.text().strip()
        tags = [tag.strip() for tag in self._tags_edit.text().split(",") if tag.strip()]
        metadata: Dict[str, str] = {}
        if self._identity_edit.text().strip():
            metadata["solvent_identity"] = self._identity_edit.text().strip()
        if self._mode_edit.text().strip():
            metadata["measurement_mode"] = self._mode_edit.text().strip()
        if self._temperature_edit.text().strip():
            metadata["temperature"] = self._temperature_edit.text().strip()
        if self._instrument_edit.text().strip():
            metadata["instrument"] = self._instrument_edit.text().strip()
        if self._date_edit.text().strip():
            metadata["date"] = self._date_edit.text().strip()
        return SolventReferenceMetadata(
            name=name,
            tags=tags,
            metadata=metadata,
            save_to_store=self._save_checkbox.isChecked(),
            set_default=self._default_checkbox.isChecked(),
        )


class SolventReferenceSelectionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Solvent Reference")
        self.resize(520, 360)
        self._entries = entries
        self._browse_path: Optional[str] = None

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._detail = QtWidgets.QTextEdit()
        self._detail.setReadOnly(True)

        default_index = None
        for idx, entry in enumerate(entries):
            name = str(entry.get("name") or "Untitled")
            tags = entry.get("tags") or []
            default_flag = " (default)" if entry.get("defaults") else ""
            label = name + default_flag
            if tags:
                label = f"{label} — {', '.join(str(tag) for tag in tags)}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            self._list.addItem(item)
            if default_index is None and entry.get("defaults"):
                default_index = idx

        self._list.currentItemChanged.connect(self._update_detail)
        if self._list.count():
            self._list.setCurrentRow(default_index or 0)

        self._default_checkbox = QtWidgets.QCheckBox("Mark selected as default")

        browse_button = QtWidgets.QPushButton("Browse File…")
        browse_button.clicked.connect(self._on_browse)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._list)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._default_checkbox)
        layout.addWidget(browse_button)
        layout.addWidget(buttons)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Solvent Reference File",
            str(Path.home()),
        )
        if not path:
            return
        self._browse_path = path
        self.accept()

    def _update_detail(self, current: Optional[QtWidgets.QListWidgetItem]) -> None:
        if current is None:
            self._detail.setPlainText("")
            return
        entry = current.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        lines = [
            f"ID: {entry.get('id') or ''}",
            f"Name: {entry.get('name') or ''}",
        ]
        if entry.get("tags"):
            lines.append(f"Tags: {', '.join(str(tag) for tag in entry.get('tags') or [])}")
        metadata = entry.get("metadata") or {}
        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
        if entry.get("source_path"):
            lines.append(f"Source: {entry.get('source_path')}")
        if entry.get("defaults"):
            lines.append("Default: yes")
        self._detail.setPlainText("\n".join(lines))

    def selected_entry(self) -> Optional[Dict[str, Any]]:
        item = self._list.currentItem()
        if not item:
            return None
        entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return entry if isinstance(entry, dict) else None

    def browse_path(self) -> Optional[str]:
        return self._browse_path

    def mark_default(self) -> bool:
        return self._default_checkbox.isChecked()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PyQt6 import QtCore, QtWidgets

from spectro_app.engine.solvent_reference import SolventReferenceEntry, SolventReferenceStore


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
        metadata_default: Optional[Dict[str, str]] = None,
        allow_save_toggle: bool = False,
        default_checked: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Solvent Reference Metadata")
        self.resize(460, 320)

        self._name_edit = QtWidgets.QLineEdit(name_default)
        self._tags_edit = QtWidgets.QLineEdit(", ".join(str(tag) for tag in tags_default))
        metadata_default = metadata_default or {}
        self._identity_edit = QtWidgets.QLineEdit(
            str(metadata_default.get("solvent_identity", ""))
        )
        self._mode_edit = QtWidgets.QLineEdit(
            str(metadata_default.get("measurement_mode", ""))
        )
        self._temperature_edit = QtWidgets.QLineEdit(
            str(metadata_default.get("temperature", ""))
        )
        self._instrument_edit = QtWidgets.QLineEdit(
            str(metadata_default.get("instrument", ""))
        )
        self._date_edit = QtWidgets.QLineEdit(str(metadata_default.get("date", "")))

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
        self._default_checkbox.setChecked(default_checked)

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
        store: Optional["SolventReferenceStore"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Solvent Reference")
        self.resize(520, 360)
        self._entries = entries
        self._browse_path: Optional[str] = None
        self._store = store

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._detail = QtWidgets.QTextEdit()
        self._detail.setReadOnly(True)

        self._edit_button = QtWidgets.QPushButton("Edit Metadata…")
        self._edit_button.clicked.connect(self._on_edit_metadata)

        default_index = None
        for idx, entry in enumerate(entries):
            label = self._format_entry_label(entry)
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

        self._update_edit_state()

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
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self._edit_button)
        action_row.addStretch(1)
        action_row.addWidget(browse_button)
        layout.addLayout(action_row)
        layout.addWidget(buttons)

    @staticmethod
    def _format_entry_label(entry: Dict[str, Any]) -> str:
        name = str(entry.get("name") or "Untitled")
        tags = entry.get("tags") or []
        default_flag = " (default)" if entry.get("defaults") else ""
        label = name + default_flag
        if tags:
            label = f"{label} — {', '.join(str(tag) for tag in tags)}"
        return label

    @staticmethod
    def _update_entry_spectrum(entry: Dict[str, Any]) -> None:
        spectrum = entry.get("spectrum")
        if not isinstance(spectrum, dict):
            return
        meta = spectrum.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            spectrum["meta"] = meta
        meta["reference_name"] = entry.get("name") or ""
        meta["reference_tags"] = list(entry.get("tags") or [])
        meta["reference_metadata"] = dict(entry.get("metadata") or {})

    def _update_edit_state(self) -> None:
        if not hasattr(self, "_edit_button"):
            return
        has_selection = self._list.currentItem() is not None
        self._edit_button.setEnabled(has_selection)

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

    def _on_edit_metadata(self) -> None:
        entry = self.selected_entry()
        if not entry:
            return
        metadata_dialog = SolventReferenceMetadataDialog(
            name_default=str(entry.get("name") or ""),
            tags_default=entry.get("tags") or (),
            metadata_default=entry.get("metadata") or {},
            default_checked=bool(entry.get("defaults")),
            parent=self,
        )
        if metadata_dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        metadata = metadata_dialog.metadata()
        if not metadata.name:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Name",
                "Provide a name for the solvent reference before saving changes.",
            )
            return
        entry["name"] = metadata.name
        entry["tags"] = metadata.tags
        entry["metadata"] = metadata.metadata
        entry["defaults"] = metadata.set_default
        self._update_entry_spectrum(entry)

        if self._store:
            stored = self._store.get(str(entry.get("id") or ""))
            if stored:
                entry["created_at"] = stored.created_at
            self._store.upsert(SolventReferenceEntry.from_dict(entry))

        current_item = self._list.currentItem()
        if current_item:
            current_item.setText(self._format_entry_label(entry))
            current_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
        self._update_detail(current_item)

    def _update_detail(self, current: Optional[QtWidgets.QListWidgetItem]) -> None:
        if current is None:
            self._detail.setPlainText("")
            self._update_edit_state()
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
        self._update_edit_state()

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

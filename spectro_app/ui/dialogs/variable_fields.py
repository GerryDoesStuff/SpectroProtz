from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6 import QtWidgets


class VariableFieldsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        fields_default: Sequence[Mapping[str, object]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Peak Variable Fields")
        self.resize(480, 320)

        self._table = QtWidgets.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Field", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )

        add_button = QtWidgets.QPushButton("Add field")
        remove_button = QtWidgets.QPushButton("Remove field")
        add_button.clicked.connect(self._on_add_row)
        remove_button.clicked.connect(self._on_remove_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Add name/value pairs to store with peak detections as custom metadata."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self._table, 1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(add_button)
        action_row.addWidget(remove_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addWidget(buttons)

        self._load_fields(fields_default)

    def fields(self) -> list[dict[str, str]]:
        fields: list[dict[str, str]] = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            value_item = self._table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            value = value_item.text().strip() if value_item else ""
            if not name:
                continue
            fields.append({"name": name, "value": value})
        return fields

    def _load_fields(self, fields_default: Sequence[Mapping[str, object]] | None) -> None:
        self._table.setRowCount(0)
        for entry in fields_default or []:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name") or entry.get("field") or entry.get("key")
            if name is None:
                continue
            value = entry.get("value", "")
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    def _on_add_row(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self._table.setCurrentCell(row, 0)

    def _on_remove_row(self) -> None:
        selection = self._table.selectionModel()
        if selection and selection.hasSelection():
            row = selection.selectedRows()[0].row()
        else:
            row = self._table.rowCount() - 1
        if row < 0:
            return
        self._table.removeRow(row)

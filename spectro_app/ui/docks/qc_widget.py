from __future__ import annotations

from typing import Dict, Iterable, List

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget

class QCDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("QC Panel", parent)
        self.table = QtWidgets.QTableView(self)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.model = QtGui.QStandardItemModel(self.table)
        self.table.setModel(self.model)
        self.setWidget(self.table)
        self.clear()

    def clear(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["QC"])
        self.model.appendRow([QtGui.QStandardItem("Awaiting quality control results…")])
        self.table.resizeColumnsToContents()

    def show_qc_table(self, rows: Iterable[Dict[str, object]]):
        data = list(rows)
        self.model.clear()

        if not data:
            self.model.setHorizontalHeaderLabels(["QC"])
            self.model.appendRow([QtGui.QStandardItem("No QC metrics were reported.")])
            self.table.resizeColumnsToContents()
            return

        headers = sorted({key for row in data for key in row.keys()})
        self.model.setColumnCount(len(headers))
        self.model.setHorizontalHeaderLabels(headers)

        for row in data:
            items: List[QtGui.QStandardItem] = []
            for header in headers:
                value = row.get(header, "")
                item = QtGui.QStandardItem(str(value))
                item.setEditable(False)
                items.append(item)
            self.model.appendRow(items)

        self.table.resizeColumnsToContents()

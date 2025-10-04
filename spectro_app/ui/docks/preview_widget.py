from __future__ import annotations

from typing import Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget

try:  # QtSvg is an optional dependency on some systems
    from PyQt6.QtSvg import QSvgRenderer  # type: ignore
except ImportError:  # pragma: no cover - fallback when QtSvg is unavailable
    QSvgRenderer = None

class PreviewDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Preview", parent)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False)
        self.tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.setWidget(self.tabs)
        self.clear()

    # ------------------------------------------------------------------
    # Public API used by the main window
    def clear(self):
        """Reset the preview area to its empty state."""
        self.tabs.clear()
        self._add_message_tab("Preview", "Waiting for results…")

    def show_error(self, message: str):
        self.tabs.clear()
        self._add_message_tab("Error", message)

    def show_figures(self, figures: Dict[str, bytes]):
        self.tabs.clear()
        if not figures:
            self._add_message_tab("Preview", "No figures were generated for this run.")
            return

        for idx, (name, data) in enumerate(figures.items(), start=1):
            title = name or f"Figure {idx}"
            pixmap = self._pixmap_from_bytes(data)
            if pixmap is None:
                self._add_message_tab(title, "Unable to render figure data.")
                continue

            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(pixmap)
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

            layout.addWidget(label)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)

            self.tabs.addTab(scroll, title)

    # ------------------------------------------------------------------
    # Internal helpers
    def _add_message_tab(self, title: str, message: str):
        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(label, title)

    def _pixmap_from_bytes(self, data: bytes) -> Optional[QtGui.QPixmap]:
        if not data:
            return None

        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(data):  # Handles PNG and other raster formats
            return pixmap

        if QSvgRenderer is None:
            return None

        renderer = QSvgRenderer(QtCore.QByteArray(data))
        if not renderer.isValid():
            return None

        size = renderer.defaultSize()
        if not size.isValid():
            size = QtCore.QSize(800, 600)

        image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()

        return QtGui.QPixmap.fromImage(image)

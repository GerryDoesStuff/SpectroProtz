from __future__ import annotations

from typing import Optional, Sequence

from PyQt6 import QtCore, QtWidgets

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.ui.docks.preview_widget import SpectraPlotWidget


class RawDataPreviewWindow(QtWidgets.QDialog):
    """Standalone window for showing raw spectra previews."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Raw Data Preview")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self._plot_widget: Optional[SpectraPlotWidget] = None
        self._message_label = QtWidgets.QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        try:
            self._plot_widget = SpectraPlotWidget(self)
        except Exception:
            self._plot_widget = None
            self._message_label.setText(
                "Interactive raw preview is unavailable because PyQtGraph could not be "
                "initialised. Install the 'pyqtgraph' package to enable plotting."
            )
            layout.addWidget(self._message_label)
        else:
            layout.addWidget(self._plot_widget, stretch=1)
            layout.addWidget(self._message_label)
            self._message_label.hide()

        self.resize(720, 480)

    # ------------------------------------------------------------------
    def plot_spectra(self, spectra: Sequence[Spectrum]) -> bool:
        """Populate the preview with spectra, returning ``True`` if shown."""

        if self._plot_widget is None:
            # Nothing to plot; the fallback label already explains why.
            self._message_label.show()
            return False

        try:
            displayed = bool(self._plot_widget.set_spectra(spectra))
        except Exception as exc:
            self._message_label.setText(
                "Unable to display spectra due to an unexpected error: " f"{exc}"
            )
            self._message_label.show()
            raise

        if displayed:
            self._message_label.hide()
        else:
            self._message_label.setText(
                "No previewable spectra were returned for this file."
            )
            self._message_label.show()
        return displayed


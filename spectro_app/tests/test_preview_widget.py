import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)

from PyQt6 import QtWidgets

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.ui.docks.preview_widget import SpectraPlotWidget


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_smoothed_fallback_renders_preview(qt_app):
    widget = SpectraPlotWidget()
    try:
        spectrum = Spectrum(
            wavelength=np.array([400.0, 500.0, 600.0], dtype=float),
            intensity=np.array([0.2, 0.4, 0.1], dtype=float),
            meta={},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        smoothed_checkbox = widget._stage_controls["smoothed"]
        assert smoothed_checkbox.isEnabled()
        assert smoothed_checkbox.isChecked()

        smoothed_datasets = widget._stage_datasets["smoothed"]
        assert len(smoothed_datasets) == 1
        np.testing.assert_allclose(smoothed_datasets[0].x, spectrum.wavelength)
        np.testing.assert_allclose(smoothed_datasets[0].y, spectrum.intensity)

        assert widget._curve_items["smoothed"]
        title_text = widget.plot.plotItem.titleLabel.text
        assert "Stepwise spectra unavailable" not in title_text
    finally:
        widget.deleteLater()
        qt_app.processEvents()

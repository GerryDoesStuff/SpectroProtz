import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)

import pyqtgraph as pg
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


def test_stage_default_skips_smoothed_fallback_when_other_stage_available(qt_app):
    widget = SpectraPlotWidget()
    try:
        wavelength = np.array([400.0, 500.0, 600.0], dtype=float)
        intensity = np.array([0.2, 0.5, 0.3], dtype=float)
        despiked = np.array([0.25, 0.55, 0.35], dtype=float)

        spectrum = Spectrum(
            wavelength=wavelength,
            intensity=intensity,
            meta={"channels": {"despiked": despiked}},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        despiked_checkbox = widget._stage_controls["despiked"]
        smoothed_checkbox = widget._stage_controls["smoothed"]

        assert despiked_checkbox.isEnabled()
        assert despiked_checkbox.isChecked()

        assert smoothed_checkbox.isEnabled()
        assert not smoothed_checkbox.isChecked()
        assert widget._smoothed_fallback_active
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_preserve_view_keeps_ranges_and_selection(qt_app):
    widget = SpectraPlotWidget()
    try:
        base_wavelength = np.linspace(400.0, 800.0, 50, dtype=float)
        spectra = [
            Spectrum(
                wavelength=base_wavelength,
                intensity=np.sin(base_wavelength / 50.0 + idx).astype(float),
                meta={"sample_id": f"Sample {idx + 1}"},
            )
            for idx in range(3)
        ]

        assert widget.set_spectra(spectra)
        qt_app.processEvents()

        widget.view_mode_button.click()
        qt_app.processEvents()
        widget._on_add_clicked()
        qt_app.processEvents()
        widget._on_previous_clicked()
        qt_app.processEvents()

        view_box = widget.plot.plotItem.getViewBox()
        view_box.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        view_box.setRange(xRange=(500.0, 600.0), yRange=(-0.75, 0.75), padding=0)

        selected_before = widget._current_single_labels()
        view_before = tuple(tuple(axis) for axis in view_box.viewRange())

        assert widget.set_spectra(spectra, preserve_view=True)
        qt_app.processEvents()

        selected_after = widget._current_single_labels()
        view_after = tuple(tuple(axis) for axis in view_box.viewRange())

        assert selected_after == selected_before
        for before_axis, after_axis in zip(view_before, view_after):
            np.testing.assert_allclose(after_axis, before_axis)

        assert widget.set_spectra(spectra, preserve_view=False)
        qt_app.processEvents()
        view_reset = tuple(tuple(axis) for axis in view_box.viewRange())

        assert not all(
            np.allclose(reset_axis, after_axis)
            for reset_axis, after_axis in zip(view_reset, view_after)
        )
    finally:
        widget.deleteLater()
        qt_app.processEvents()

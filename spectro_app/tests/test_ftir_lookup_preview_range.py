import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)

from pathlib import Path

from PyQt6 import QtWidgets

from spectro_app.app_context import AppContext
from spectro_app.ui.dialogs.ftir_lookup import (
    FtirLookupWindow,
    LookupResultEntry,
    _SelectedSpectrum,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _build_window(qt_app):
    context = AppContext()
    window = FtirLookupWindow(context)
    window._active_index_path = Path("dummy.duckdb")
    return window


def test_preview_range_prefers_longest_selected_spectrum(qt_app, monkeypatch):
    window = _build_window(qt_app)
    try:
        window._preview_entry = LookupResultEntry(
            file_id="1",
            spectrum_name="Ref",
            formula="",
            matched_peaks=0,
            match_score=0.0,
            metadata={},
        )
        window._preview_spectrum_overlay_enabled = True
        window._selected_spectra = [
            _SelectedSpectrum(label="Short", x=[100.0, 200.0, 300.0], y=[1.0, 2.0, 3.0]),
            _SelectedSpectrum(label="Wide", x=[50.0, 950.0], y=[0.1, 0.9]),
        ]

        monkeypatch.setattr(window, "_populate_reference_peaks", lambda file_ids: 0)
        monkeypatch.setattr(window, "_plot_reference_peaks", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            window,
            "_load_meta_json_spectrum",
            lambda entry: ([10.0, 20.0, 30.0], [0.1, 0.2, 0.3]),
        )

        captured = {}

        def _capture_x_range(min_x, max_x, padding=0):
            captured["range"] = (min_x, max_x)

        monkeypatch.setattr(window._preview_plot_widget, "setXRange", _capture_x_range)

        window._refresh_preview_plot()

        assert captured["range"] == (50.0, 950.0)
    finally:
        window.deleteLater()
        qt_app.processEvents()


def test_preview_range_uses_reference_when_no_selection(qt_app, monkeypatch):
    window = _build_window(qt_app)
    try:
        window._preview_entry = LookupResultEntry(
            file_id="1",
            spectrum_name="Ref",
            formula="",
            matched_peaks=0,
            match_score=0.0,
            metadata={},
        )
        window._preview_spectrum_overlay_enabled = True
        window._selected_spectra = []

        monkeypatch.setattr(window, "_populate_reference_peaks", lambda file_ids: 0)
        monkeypatch.setattr(window, "_plot_reference_peaks", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            window,
            "_load_meta_json_spectrum",
            lambda entry: ([200.0, 800.0, 600.0], [0.1, 0.4, 0.2]),
        )

        captured = {}

        def _capture_x_range(min_x, max_x, padding=0):
            captured["range"] = (min_x, max_x)

        monkeypatch.setattr(window._preview_plot_widget, "setXRange", _capture_x_range)

        window._refresh_preview_plot()

        assert captured["range"] == (200.0, 800.0)
    finally:
        window.deleteLater()
        qt_app.processEvents()

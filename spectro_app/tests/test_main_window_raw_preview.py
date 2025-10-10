import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6 import QtWidgets

from spectro_app.app_context import AppContext
from spectro_app.engine.plugin_api import Spectrum, SpectroscopyPlugin
from spectro_app.ui.main_window import MainWindow
import spectro_app.ui.dialogs.raw_preview as raw_preview


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


class _PreviewPlugin(SpectroscopyPlugin):
    id = "preview"
    label = "Preview"

    def detect(self, paths):
        return True

    def load(self, paths):
        return [
            Spectrum(
                wavelength=np.array([1.0, 2.0, 3.0]),
                intensity=np.array([0.1, 0.2, 0.3]),
                meta={"path": paths[0] if paths else ""},
            )
        ]


def test_queue_preview_opens_raw_window(qt_app, monkeypatch, tmp_path):
    appctx = AppContext()
    window = MainWindow(appctx)
    try:
        plugin = _PreviewPlugin()
        window._plugin_registry = {plugin.id: plugin}
        window._active_plugin = plugin
        window._active_plugin_id = plugin.id

        open_calls = []
        window._open_file_preview = lambda *args, **kwargs: open_calls.append((args, kwargs))

        captured = {}

        class FakePlot:
            def __init__(self, parent=None):
                self.last = None

            def set_spectra(self, spectra):
                self.last = list(spectra)
                captured["spectra"] = self.last
                return True

        monkeypatch.setattr(raw_preview, "SpectraPlotWidget", FakePlot)

        sample_path = tmp_path / "sample.spc"
        sample_path.write_text("dummy", encoding="utf-8")

        window._on_queue_preview_requested(str(sample_path))
        qt_app.processEvents()

        assert not open_calls
        assert window._raw_preview_window is not None
        assert isinstance(window._raw_preview_window._plot_widget, FakePlot)
        assert captured.get("spectra")
    finally:
        if window._raw_preview_window is not None:
            window._raw_preview_window.close()
        window.close()
        window.deleteLater()
        qt_app.processEvents()

import base64
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6 import QtCore, QtWidgets

from spectro_app.app_context import AppContext
from spectro_app.engine.plugin_api import BatchResult, Spectrum, SpectroscopyPlugin
from spectro_app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


class _DummyPlugin(SpectroscopyPlugin):
    id = "dummy"
    label = "Dummy"

    def load(self, paths):
        return [
            Spectrum(
                wavelength=np.array([1.0, 2.0, 3.0]),
                intensity=np.array([4.0, 5.0, 6.0]),
                meta={"path": paths[0] if paths else "dummy"},
            )
        ]

    def analyze(self, specs, recipe):
        return specs, [{"metric": "ok"}]

    def export(self, specs, qc, recipe):
        figure_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        )
        return BatchResult(
            processed=list(specs),
            qc_table=list(qc),
            figures={"Result": figure_bytes},
            audit=["analysis complete"],
        )


def test_run_controller_populates_docks(qt_app):
    appctx = AppContext()
    window = MainWindow(appctx)
    try:
        dummy = _DummyPlugin()
        window._plugin_registry = {dummy.id: dummy}
        window._active_plugin = dummy
        window._active_plugin_id = dummy.id

        finished = []
        loop = QtCore.QEventLoop()

        def _stop(result):
            finished.append(result)
            loop.quit()

        window.runctl.job_finished.connect(_stop)
        window.runctl.start(dummy, ["sample.spc"], {"module": dummy.id})

        QtCore.QTimer.singleShot(5000, loop.quit)
        loop.exec()

        assert finished, "RunController did not emit job_finished"
        qt_app.processEvents()

        assert window.previewDock.tabs.count() == 1
        assert window.previewDock.tabs.tabText(0) == "Result"

        assert window.qcDock.model.rowCount() == 1

        log_text = window.loggerDock.text.toPlainText()
        assert "analysis complete" in log_text
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()

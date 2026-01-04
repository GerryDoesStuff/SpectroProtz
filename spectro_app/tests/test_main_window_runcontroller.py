import base64
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6 import QtCore, QtWidgets

from spectro_app.app_context import AppContext
from spectro_app.engine.plugin_api import BatchResult, Spectrum, SpectroscopyPlugin
from spectro_app.engine.run_controller import PREVIEW_EXPORT_DISABLED_FLAG
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

    def load(self, paths, cancelled=None):
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
            report_text="Dummy narrative summary.",
        )


class _FileExportPlugin(SpectroscopyPlugin):
    id = "filewriter"
    label = "File Writer"

    def __init__(self) -> None:
        self.export_calls = []

    def detect(self, paths):
        return True

    def load(self, paths, cancelled=None):
        return [
            Spectrum(
                wavelength=np.array([1.0, 2.0, 3.0]),
                intensity=np.array([4.0, 5.0, 6.0]),
                meta={"path": paths[0] if paths else "dummy"},
            )
        ]

    def analyze(self, specs, recipe):
        return list(specs), [{"metric": "ok"}]

    def export(self, specs, qc, recipe):
        export_cfg = dict(recipe.get("export", {})) if recipe else {}
        preview_disabled = bool(export_cfg.get(PREVIEW_EXPORT_DISABLED_FLAG))
        target = export_cfg.get("path")
        pdf_target = None
        if target:
            pdf_target = str(Path(target).with_suffix(".pdf"))
        audit = []
        if preview_disabled:
            audit.append("Preview export disabled; workbook not written.")
        elif target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("workbook", encoding="utf-8")
            sidecar = path.with_suffix(".recipe.json")
            sidecar.write_text("recipe", encoding="utf-8")
            pdf_path = path.with_suffix(".pdf")
            pdf_path.write_text("pdf", encoding="utf-8")
            audit.append(f"Workbook written to {path}")
            audit.append(f"Recipe sidecar written to {sidecar}")
            audit.append(f"PDF report written to {pdf_path}")
        else:
            audit.append("No workbook path provided; workbook not written.")
        self.export_calls.append((target, dict(export_cfg)))
        return BatchResult(
            processed=list(specs),
            qc_table=list(qc),
            figures={},
            audit=audit,
            report_text=None,
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

        assert window.previewDock.tabs.count() >= 1
        assert window.previewDock.tabs.tabText(0) == "Spectra"

        assert window.qcDock.model.rowCount() == 1

        log_text = window.loggerDock.text.toPlainText()
        assert "Dummy narrative summary." in log_text
        assert "analysis complete" in log_text
        assert log_text.index("Dummy narrative summary.") < log_text.index("analysis complete")
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_auto_preview_does_not_export_files(qt_app, tmp_path, monkeypatch):
    appctx = AppContext()
    window = MainWindow(appctx)

    def run_job(auto_triggered: bool) -> BatchResult:
        finished = []
        loop = QtCore.QEventLoop()

        def _stop(result):
            finished.append(result)
            loop.quit()

        window.runctl.job_finished.connect(_stop)
        try:
            assert window._start_processing_job(auto_triggered=auto_triggered)
            QtCore.QTimer.singleShot(5000, loop.quit)
            loop.exec()
        finally:
            window.runctl.job_finished.disconnect(_stop)
        qt_app.processEvents()
        assert finished, "Processing job did not complete"
        assert isinstance(finished[-1], BatchResult)
        return finished[-1]

    def run_manual_preview() -> BatchResult:
        finished = []
        loop = QtCore.QEventLoop()

        def _stop(result):
            finished.append(result)
            loop.quit()

        window.runctl.job_finished.connect(_stop)
        try:
            window.on_run()
            QtCore.QTimer.singleShot(5000, loop.quit)
            loop.exec()
        finally:
            window.runctl.job_finished.disconnect(_stop)
        qt_app.processEvents()
        assert finished, "Processing job did not complete"
        assert isinstance(finished[-1], BatchResult)
        return finished[-1]

    try:
        plugin = _FileExportPlugin()
        window._plugin_registry = {plugin.id: plugin}
        window._queued_paths = [str(tmp_path / "sample.spc")]
        window._recipe_data["module"] = plugin.id
        window._recipe_data.setdefault("export", {})
        window._active_plugin = plugin
        window._active_plugin_id = plugin.id

        export_target = tmp_path / "manual_export.xlsx"
        sidecar_target = export_target.with_suffix(".recipe.json")
        pdf_target = export_target.with_suffix(".pdf")

        export_cfg = window._recipe_data.setdefault("export", {})
        export_cfg["path"] = str(export_target)

        run_manual_preview()
        assert isinstance(window._last_result, BatchResult)

        assert not export_target.exists()
        assert not sidecar_target.exists()
        assert not pdf_target.exists()

        assert plugin.export_calls, "Plugin export was not invoked"
        manual_call_target, manual_call_export = plugin.export_calls[-1]
        assert manual_call_target is None
        assert manual_call_export.get("path") is None
        assert manual_call_export.get(PREVIEW_EXPORT_DISABLED_FLAG)

        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_target), "xlsx"),
        )

        window.on_export()

        assert export_target.exists()
        assert sidecar_target.exists()
        assert pdf_target.exists()

        export_target.unlink()
        sidecar_target.unlink()
        pdf_target.unlink()

        export_cfg = window._recipe_data.get("export", {})
        assert export_cfg.get("path") == str(export_target)

        export_call_target, export_call_config = plugin.export_calls[-1]
        assert export_call_target == str(export_target)
        assert export_call_config.get("path") == str(export_target)
        assert not export_call_config.get(PREVIEW_EXPORT_DISABLED_FLAG)

        run_job(auto_triggered=True)
        assert isinstance(window._last_result, BatchResult)

        assert not export_target.exists()
        assert not sidecar_target.exists()
        assert not pdf_target.exists()

        assert plugin.export_calls, "Plugin export was not invoked"
        auto_call_target, auto_call_export = plugin.export_calls[-1]
        assert auto_call_target is None
        assert auto_call_export.get("path") is None
        assert auto_call_export.get("_preview_export_disabled")

        assert window._recipe_data.get("export", {}).get("path") == str(
            export_target
        )
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()

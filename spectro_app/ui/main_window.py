import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets

from spectro_app.engine.plugin_api import BatchResult
from spectro_app.engine.recipe_model import Recipe
from spectro_app.engine.run_controller import RunController
from spectro_app.plugins.ftir.plugin import FtirPlugin
from spectro_app.plugins.pees.plugin import PeesPlugin
from spectro_app.plugins.raman.plugin import RamanPlugin
from spectro_app.plugins.uvvis.plugin import UvVisPlugin
from spectro_app.ui.dialogs.about import AboutDialog
from spectro_app.ui.dialogs.help_viewer import HelpViewer
from spectro_app.ui.docks.file_queue import FileQueueDock
from spectro_app.ui.docks.logger_view import LoggerDock
from spectro_app.ui.docks.preview_widget import PreviewDock
from spectro_app.ui.docks.qc_widget import QCDock
from spectro_app.ui.docks.recipe_editor import RecipeEditorDock
from spectro_app.ui.menus import build_menus


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, appctx):
        super().__init__()
        self.appctx = appctx
        self.setWindowTitle("Spectroscopy Processing App")
        self.resize(1280, 800)
        build_menus(self)
        self._init_docks()
        self._init_status()
        self.restore_state()

        self.runctl = RunController(self)
        self.runctl.job_started.connect(self._on_job_started)
        self.runctl.job_finished.connect(self._on_job_finished)
        self.runctl.job_progress.connect(self._on_job_progress)
        self.runctl.job_message.connect(self._on_job_message)

        self._plugin_registry = {
            plugin.id: plugin
            for plugin in (
                UvVisPlugin(),
                FtirPlugin(),
                RamanPlugin(),
                PeesPlugin(),
            )
        }

        self._queued_paths: List[str] = []
        self._recipe_data: Dict[str, Any] = self._normalise_recipe_data({})
        self._current_recipe_path: Optional[Path] = None
        self._last_result: Optional[BatchResult] = None
        self._active_plugin = None
        self._active_plugin_id: Optional[str] = None
        self._export_default_dir: Optional[Path] = None
        self._job_running = False
        self._updating_ui = False

        self._collect_actions()
        self._connect_recipe_editor()
        self._select_module(self._recipe_data["module"])
        self.status.showMessage("Ready")
        self._update_action_states()

    def _init_docks(self):
        self.fileDock = FileQueueDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.fileDock)
        self.recipeDock = RecipeEditorDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.recipeDock)
        self.previewDock = PreviewDock(self)
        self.setCentralWidget(self.previewDock)
        self.qcDock = QCDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.qcDock)
        self.loggerDock = LoggerDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.loggerDock)

    def _init_status(self):
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.status.addPermanentWidget(self.progress)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def _collect_actions(self):
        actions = {action.text(): action for action in self.findChildren(QtWidgets.QAction)}
        self._open_action = actions.get("Open...")
        self._save_recipe_action = actions.get("Save Recipe")
        self._save_recipe_as_action = actions.get("Save Recipe As...")
        self._export_action = actions.get("Export Workbook...")
        self._run_action = actions.get("Run")
        self._cancel_action = actions.get("Cancel")

    def _connect_recipe_editor(self):
        self.recipeDock.module.currentTextChanged.connect(self._on_module_changed)
        self.recipeDock.smooth_enable.toggled.connect(self._on_recipe_widget_changed)
        self.recipeDock.smooth_window.valueChanged.connect(self._on_recipe_widget_changed)

    def on_open(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Spectra", str(self._last_data_dir())
        )
        if not paths:
            return

        recipe_paths = [p for p in paths if p.lower().endswith(".recipe.json")]
        data_paths = [p for p in paths if p not in recipe_paths]

        if recipe_paths:
            self._load_recipe(Path(recipe_paths[0]))

        if data_paths:
            self._set_queue(data_paths)
            self._export_default_dir = Path(data_paths[0]).parent
        else:
            self._set_queue([])

    def on_save_recipe(self):
        if self._current_recipe_path is None:
            self.on_save_recipe_as()
            return
        self._update_recipe_from_ui()
        self._write_recipe(self._current_recipe_path)

    def on_save_recipe_as(self):
        directory = (
            str(self._current_recipe_path.parent)
            if self._current_recipe_path
            else str(self._last_data_dir())
        )
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Recipe",
            directory,
            "Recipe Files (*.recipe.json);;JSON Files (*.json);;All Files (*)",
        )
        if not path_str:
            return
        target = Path(path_str)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".recipe.json")
        self._current_recipe_path = target
        self._update_recipe_from_ui()
        self._write_recipe(target)

    def on_export(self):
        if not self._last_result or not self._active_plugin:
            QtWidgets.QMessageBox.information(
                self, "No Results", "Run a job before exporting."
            )
            return
        default_dir = self._export_default_dir or Path.home()
        export_cfg = self._recipe_data.get("export", {})
        default_name = export_cfg.get("path") or "spectra.xlsx"
        default_path = (
            default_dir / default_name
            if not Path(default_name).is_absolute()
            else Path(default_name)
        )
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Workbook",
            str(default_path),
            "Excel Workbook (*.xlsx);;All Files (*)",
        )
        if not target:
            return

        export_config = dict(export_cfg)
        export_config["path"] = target
        self._recipe_data["export"] = export_config
        recipe_payload = self._build_recipe_payload()

        try:
            result = self._active_plugin.export(
                self._last_result.processed,
                self._last_result.qc_table,
                recipe_payload,
            )
        except Exception as exc:  # pragma: no cover - UI error path
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(exc))
            self.status.showMessage("Export failed.", 5000)
            return

        self._last_result = result
        self.status.showMessage(f"Exported workbook to {target}", 5000)
        self.appctx.set_dirty(True)
        self._update_action_states()

    def on_reset_layout(self):
        self.restoreState(self.saveState())

    def on_run(self):
        if self._job_running:
            return
        if not self._queued_paths:
            QtWidgets.QMessageBox.information(
                self, "No Files", "Add spectra to the queue before running."
            )
            return

        self._update_recipe_from_ui()
        recipe_payload = self._build_recipe_payload()
        recipe_model = Recipe(
            module=recipe_payload.get("module", "uvvis"),
            params=recipe_payload.get("params", {}),
            version=str(recipe_payload.get("version", Recipe().version)),
        )
        errors = recipe_model.validate()
        if errors:
            QtWidgets.QMessageBox.critical(
                self, "Invalid Recipe", "\n".join(errors)
            )
            return

        plugin = self._resolve_plugin(self._queued_paths, recipe_payload.get("module"))
        if not plugin:
            QtWidgets.QMessageBox.critical(
                self,
                "No Plugin",
                "Unable to determine a processing plugin for the selected files.",
            )
            return

        self._active_plugin = plugin
        self._active_plugin_id = plugin.id
        self._select_module(plugin.id)
        self._last_result = None
        self._update_action_states()
        self.status.showMessage(f"Running {plugin.label} pipeline...")
        self.runctl.start(plugin, self._queued_paths, recipe_payload)

    def on_cancel(self):
        if not self._job_running:
            return
        if self.runctl.cancel():
            self.status.showMessage("Cancelling job...")

    def on_help(self):
        dlg = HelpViewer(self)
        dlg.exec()

    def on_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def closeEvent(self, e):
        if not self.appctx.maybe_close():
            e.ignore()
            return
        self.save_state()
        super().closeEvent(e)

    def save_state(self):
        s = self.appctx.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())

    def restore_state(self):
        s = self.appctx.settings
        g = s.value("geometry")
        w = s.value("windowState")
        if g:
            self.restoreGeometry(g)
        if w:
            self.restoreState(w)

    def _update_action_states(self):
        running = self._job_running
        has_queue = bool(self._queued_paths)
        if self._run_action:
            self._run_action.setEnabled(not running and has_queue)
        if self._cancel_action:
            self._cancel_action.setEnabled(running)
        if self._open_action:
            self._open_action.setEnabled(not running)
        if self._export_action:
            self._export_action.setEnabled(not running and self._last_result is not None)

    def _last_data_dir(self) -> Path:
        if self._queued_paths:
            return Path(self._queued_paths[-1]).parent
        if self._current_recipe_path:
            return self._current_recipe_path.parent
        return Path.home()

    def _set_queue(self, paths: List[str]):
        self._queued_paths = [str(p) for p in paths]
        self.fileDock.list.clear()
        if self._queued_paths:
            self.fileDock.list.addItems(self._queued_paths)
        self._last_result = None
        self._update_action_states()
        if self._queued_paths:
            plugin = self._resolve_plugin(
                self._queued_paths, self._recipe_data.get("module")
            )
            if plugin:
                self._select_module(plugin.id)
        message = (
            f"Queued {len(self._queued_paths)} files"
            if self._queued_paths
            else "Queue cleared"
        )
        self.status.showMessage(message, 5000)

    def _normalise_recipe_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(data or {})
        default_version = Recipe().version
        module = base.get("module") or "uvvis"
        params = base.get("params") if isinstance(base.get("params"), dict) else {}
        export_cfg = (
            base.get("export") if isinstance(base.get("export"), dict) else {}
        )
        base.update(
            {
                "module": module,
                "params": params,
                "export": export_cfg,
                "version": str(base.get("version", default_version)),
            }
        )
        return base

    def _on_module_changed(self, module_text: str):
        if self._updating_ui:
            return
        module_id = (module_text or "").strip().lower()
        if module_id:
            self._recipe_data["module"] = module_id
            self._active_plugin_id = module_id
        self._update_recipe_from_ui()
        self.appctx.set_dirty(True)
        self._update_action_states()

    def _on_recipe_widget_changed(self, *_):
        if self._updating_ui:
            return
        self._update_recipe_from_ui()
        self.appctx.set_dirty(True)

    def _update_recipe_from_ui(self):
        params = dict(self._recipe_data.get("params") or {})
        smoothing = dict(params.get("smoothing") or {})
        smoothing["enabled"] = self.recipeDock.smooth_enable.isChecked()
        smoothing["window"] = int(self.recipeDock.smooth_window.value())
        params["smoothing"] = smoothing
        self._recipe_data["params"] = params
        module = (self.recipeDock.module.currentText() or "uvvis").strip().lower()
        if module:
            self._recipe_data["module"] = module
        if "version" not in self._recipe_data:
            self._recipe_data["version"] = Recipe().version

    def _build_recipe_payload(self) -> Dict[str, Any]:
        payload = copy.deepcopy(self._recipe_data)
        payload.setdefault("module", "uvvis")
        payload.setdefault("params", {})
        payload.setdefault("export", {})
        payload.setdefault("version", Recipe().version)
        return payload

    def _select_module(self, module_id: str):
        module_id = (module_id or "uvvis").strip().lower()
        if module_id not in self._plugin_registry:
            return
        previous_state = self._updating_ui
        self._updating_ui = True
        try:
            index = self.recipeDock.module.findText(module_id)
            if index >= 0:
                self.recipeDock.module.setCurrentIndex(index)
        finally:
            self._updating_ui = previous_state
        self._recipe_data["module"] = module_id
        self._active_plugin_id = module_id
        if not previous_state:
            self._update_recipe_from_ui()

    def _resolve_plugin(
        self, paths: List[str], preferred: Optional[str] = None
    ):
        if preferred:
            plugin = self._plugin_registry.get(preferred)
            if plugin:
                return plugin
        for plugin in self._plugin_registry.values():
            try:
                if plugin.detect(paths):
                    return plugin
            except Exception:  # pragma: no cover - plugin-specific failure
                continue
        return None

    def _on_job_started(self):
        self._job_running = True
        self.appctx.set_job_running(True)
        self.progress.setRange(0, 0)
        self._update_action_states()

    def _on_job_progress(self, value: int):
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def _on_job_message(self, message: str):
        self.status.showMessage(message)

    def _on_job_finished(self, result):
        self._job_running = False
        self.appctx.set_job_running(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if isinstance(result, Exception):
            if isinstance(result, RuntimeError) and str(result) == "Cancelled":
                self.status.showMessage("Job cancelled.", 5000)
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Processing Failed", str(result)
                )
                self.status.showMessage("Processing failed.", 5000)
            self._last_result = None
        else:
            self._last_result = result
            self.status.showMessage("Processing complete.", 5000)
        self._update_action_states()

    def _write_recipe(self, path: Path):
        recipe_payload = self._build_recipe_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(recipe_payload, handle, indent=2, sort_keys=True)
        self.status.showMessage(f"Recipe saved to {path}", 5000)
        self.appctx.set_dirty(False)

    def _load_recipe(self, path: Path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover - UI error path
            QtWidgets.QMessageBox.critical(self, "Recipe Load Failed", str(exc))
            return

        if not isinstance(data, dict):
            QtWidgets.QMessageBox.critical(
                self, "Recipe Load Failed", "Recipe file must contain a JSON object."
            )
            return

        self._recipe_data = self._normalise_recipe_data(data)
        self._current_recipe_path = path
        module_id = self._recipe_data.get("module", "uvvis")
        smoothing = (self._recipe_data.get("params") or {}).get("smoothing", {})

        self._updating_ui = True
        try:
            self._select_module(module_id)
            self.recipeDock.smooth_enable.setChecked(
                bool(smoothing.get("enabled", False))
            )
            window = smoothing.get(
                "window", self.recipeDock.smooth_window.value()
            )
            try:
                window_value = int(window)
            except (TypeError, ValueError):
                window_value = self.recipeDock.smooth_window.value()
            self.recipeDock.smooth_window.setValue(window_value)
        finally:
            self._updating_ui = False

        self._update_recipe_from_ui()
        self.appctx.set_dirty(False)
        self.status.showMessage(f"Loaded recipe from {path}", 5000)
        self._update_action_states()

import copy
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDesktopServices

from spectro_app.engine.plugin_api import BatchResult
from spectro_app.engine.recipe_model import Recipe
from spectro_app.engine.run_controller import RunController
from spectro_app.plugins.ftir.plugin import FtirPlugin
from spectro_app.plugins.pees.plugin import PeesPlugin
from spectro_app.plugins.raman.plugin import RamanPlugin
from spectro_app.plugins.uvvis.plugin import UvVisPlugin
from spectro_app.ui.dialogs.about import AboutDialog
from spectro_app.ui.dialogs.help_viewer import HelpViewer
from spectro_app.ui.dialogs.settings_dialog import SettingsDialog
from spectro_app.ui.docks.file_queue import FileQueueDock, QueueEntry
from spectro_app.ui.docks.logger_view import LoggerDock
from spectro_app.ui.docks.preview_widget import PreviewDock
from spectro_app.ui.docks.qc_widget import QCDock
from spectro_app.ui.docks.logger_view import LoggerDock
from spectro_app.ui.dialogs.about import AboutDialog
from spectro_app.ui.dialogs.help_viewer import HelpViewer
from spectro_app.engine.run_controller import RunController
from spectro_app.engine.plugin_api import BatchResult
from spectro_app.engine.recipe_model import Recipe
from spectro_app.ui.docks.recipe_editor import RecipeEditorDock
from spectro_app.ui.menus import build_menus


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, appctx):
        super().__init__()
        self.appctx = appctx
        self.setWindowTitle("Spectroscopy Processing App")
        self.resize(1280, 800)
        self._recent_menu = None
        self._recent_placeholder_text = "No Recent Files"
        build_menus(self)
        self._refresh_recent_menu()
        self._collect_actions()
        self._init_toolbar()
        self._init_docks()
        self._init_status()
        self.restore_state()

        self.runctl = RunController(self)
        self._connect_run_controller()

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
        self.fileDock.set_plugin_resolver(
            lambda paths: self._resolve_plugin(paths, self._recipe_data.get("module"))
        )
        self._current_recipe_path: Optional[Path] = None
        self._last_result: Optional[BatchResult] = None
        self._active_plugin = None
        self._active_plugin_id: Optional[str] = None
        self._export_default_dir: Optional[Path] = None
        self._default_palette: Optional[QtGui.QPalette] = None
        self._default_style_name: Optional[str] = None
        self._log_dir: Optional[Path] = None
        self._load_app_settings()
        self._job_running = False
        self._updating_ui = False

        self._connect_recipe_editor()
        self._select_module(self._recipe_data["module"])
        self.status.showMessage("Ready")
        self._update_action_states()

    def _init_docks(self):
        self.fileDock = FileQueueDock(self)
        self.fileDock.paths_dropped.connect(self._on_queue_paths_dropped)
        self.fileDock.inspect_requested.connect(self._on_queue_inspect_requested)
        self.fileDock.preview_requested.connect(self._on_queue_preview_requested)
        self.fileDock.locate_requested.connect(self._on_queue_locate_requested)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.fileDock)
        self.recipeDock = RecipeEditorDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.recipeDock)
        self.previewDock = PreviewDock(self)
        self.setCentralWidget(self.previewDock)
        self.qcDock = QCDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.qcDock)
        self.loggerDock = LoggerDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.loggerDock)

    def _init_toolbar(self):
        self._toolbar = self.addToolBar("Main")
        self._toolbar.setObjectName("MainToolbar")
        self._toolbar.setMovable(False)
        if self._open_action:
            self._toolbar.addAction(self._open_action)
        if self._save_recipe_action:
            self._toolbar.addAction(self._save_recipe_action)
        self._toolbar.addSeparator()
        if self._run_action:
            self._toolbar.addAction(self._run_action)
        if self._cancel_action:
            self._toolbar.addAction(self._cancel_action)
        self._toolbar.addSeparator()
        if self._export_action:
            self._toolbar.addAction(self._export_action)

    def _init_status(self):
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.status.addPermanentWidget(self.progress)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def _collect_actions(self):
        actions = {action.text(): action for action in self.findChildren(QtGui.QAction)}
        self._open_action = actions.get("Open...")
        self._save_recipe_action = actions.get("Save Recipe")
        self._save_recipe_as_action = actions.get("Save Recipe As...")
        self._export_action = actions.get("Export Workbook...")
        self._run_action = actions.get("Run")
        self._cancel_action = actions.get("Cancel")

    def _load_app_settings(self):
        settings = self.appctx.settings
        app = QtWidgets.QApplication.instance()
        if app:
            if self._default_palette is None:
                self._default_palette = QtGui.QPalette(app.palette())
            if self._default_style_name is None:
                self._default_style_name = app.style().objectName()

        export_dir_value = settings.value("export/defaultDir", "") or ""
        if export_dir_value:
            try:
                self._export_default_dir = Path(str(export_dir_value))
            except (TypeError, ValueError):
                self._export_default_dir = None

        export_name_value = settings.value("export/defaultName", "") or ""
        if export_name_value:
            export_cfg = self._recipe_data.setdefault("export", {})
            if not export_cfg.get("path"):
                export_cfg["path"] = export_name_value

        log_dir_value = settings.value("logDir", "") or ""
        if log_dir_value:
            try:
                self._log_dir = Path(str(log_dir_value))
            except (TypeError, ValueError):
                self._log_dir = Path.home() / "SpectroApp" / "logs"
        else:
            self._log_dir = Path.home() / "SpectroApp" / "logs"

        theme_value = str(settings.value("theme", "system") or "system")
        self._apply_theme(theme_value)

    def _apply_theme(self, theme: str):
        theme = (theme or "system").lower()
        app = QtWidgets.QApplication.instance()
        if not app:
            return

        if self._default_palette is None:
            self._default_palette = QtGui.QPalette(app.palette())
        if self._default_style_name is None:
            self._default_style_name = app.style().objectName()

        app.setProperty("spectro_theme", theme)

        if theme == "dark":
            app.setStyle("Fusion")
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
            palette.setColor(
                QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white
            )
            palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
            palette.setColor(
                QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53)
            )
            palette.setColor(
                QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white
            )
            palette.setColor(
                QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white
            )
            palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
            palette.setColor(
                QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white
            )
            palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
            palette.setColor(
                QtGui.QPalette.ColorRole.Highlight,
                QtGui.QColor(142, 45, 197).lighter(),
            )
            palette.setColor(
                QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black
            )
            app.setPalette(palette)
        elif theme == "light":
            app.setStyle("Fusion")
            if self._default_palette is not None:
                app.setPalette(QtGui.QPalette(self._default_palette))
            else:
                app.setPalette(app.style().standardPalette())
        else:
            if self._default_style_name:
                style = QtWidgets.QStyleFactory.create(self._default_style_name)
                if style:
                    app.setStyle(style)
                else:
                    app.setStyle(self._default_style_name)
            if self._default_palette is not None:
                app.setPalette(QtGui.QPalette(self._default_palette))
            else:
                app.setPalette(app.style().standardPalette())

    def _coerce_settings_list(self, value: object) -> List[str]:
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value if v]
        elif isinstance(value, str):
            items = [v for v in value.split(";") if v]
        else:
            items = []
        seen = set()
        ordered: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _refresh_recent_menu(self):
        menu = getattr(self, "_recent_menu", None)
        if not menu:
            return
        settings = self.appctx.settings
        stored = self._coerce_settings_list(settings.value("recentFiles", []))
        valid: List[str] = []
        for entry in stored:
            try:
                path = Path(entry)
            except (TypeError, ValueError):
                continue
            if path.exists():
                valid.append(str(path))
        limited = valid[:20]
        if len(stored) != len(limited) or stored[: len(limited)] != limited:
            settings.setValue("recentFiles", limited)
        self._populate_recent_menu(limited)

    def _populate_recent_menu(self, paths: List[str]):
        menu = getattr(self, "_recent_menu", None)
        if not menu:
            return
        menu.clear()
        if not paths:
            placeholder = menu.addAction(self._recent_placeholder_text)
            placeholder.setEnabled(False)
            return
        for path in paths:
            action = menu.addAction(path)
            action.setData(path)
            action.setToolTip(path)
            action.triggered.connect(partial(self._open_recent_file, path))

    def _preview_recent_menu(self, paths: List[str]):
        preview = self._coerce_settings_list(paths)[:20]
        self._populate_recent_menu(preview)

    def _open_recent_file(self, path_str: str):
        target = Path(path_str)
        if not target.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"The file '{path_str}' could not be found on disk.",
            )
            self._remove_recent_entry(path_str)
            return

        normalized = str(target)
        lowered = normalized.lower()
        if lowered.endswith(".recipe.json"):
            self._load_recipe(target)
        else:
            self._set_queue([normalized])
            base_dir = target.parent if target.is_file() else target
            if base_dir.is_dir():
                self._export_default_dir = base_dir
                self.appctx.settings.setValue("export/defaultDir", str(base_dir))

        self._record_recent_files([normalized])

    def _remove_recent_entry(self, path_str: str):
        settings = self.appctx.settings
        current = self._coerce_settings_list(settings.value("recentFiles", []))
        if path_str in current:
            current.remove(path_str)
            settings.setValue("recentFiles", current)
            self._refresh_recent_menu()

    def _record_recent_files(self, paths: List[str]):
        clean_paths = [str(p) for p in paths if p]
        if not clean_paths:
            return
        settings = self.appctx.settings
        current = self._coerce_settings_list(settings.value("recentFiles", []))
        for path in clean_paths:
            if path in current:
                current.remove(path)
            current.insert(0, path)
        settings.setValue("recentFiles", current[:20])
        self._refresh_recent_menu()

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
            try:
                self._export_default_dir = Path(data_paths[0]).parent
                self.appctx.settings.setValue(
                    "export/defaultDir", str(self._export_default_dir)
                )
            except (TypeError, ValueError):
                self._export_default_dir = None
        else:
            self._set_queue([])

        self._record_recent_files(paths)

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
        export_path = Path(target)
        self.appctx.settings.setValue("export/defaultDir", str(export_path.parent))
        self.appctx.settings.setValue("export/defaultName", export_path.name)
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

    def on_settings(self):
        dialog = SettingsDialog(self.appctx.settings, self)
        dialog.recent_files_changed.connect(self._preview_recent_menu)
        result = dialog.exec()
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            self._refresh_recent_menu()
            return

        values = dialog.values()
        settings = self.appctx.settings
        settings.setValue("theme", values["theme"])
        settings.setValue("recentFiles", values["recent_files"])
        settings.setValue("export/defaultDir", values["export_default_dir"])
        settings.setValue("export/defaultName", values["export_default_name"])

        self._apply_theme(values["theme"])

        export_dir = values["export_default_dir"]
        if export_dir:
            try:
                self._export_default_dir = Path(export_dir)
            except (TypeError, ValueError):
                self._export_default_dir = None
        else:
            self._export_default_dir = None
        export_name = values["export_default_name"]
        if export_name:
            export_cfg = self._recipe_data.setdefault("export", {})
            export_cfg["path"] = export_name

        self.status.showMessage("Settings updated.", 5000)
        self._refresh_recent_menu()

    def on_check_updates(self):
        QtWidgets.QMessageBox.information(
            self,
            "Check for Updates",
            "You are running the latest available version of SpectroApp.",
        )

    def on_open_log_folder(self):
        log_dir = self._log_dir or Path.home() / "SpectroApp" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem error path
            QtWidgets.QMessageBox.critical(
                self, "Log Folder", f"Unable to create log folder:\n{exc}"
            )
            return
        self._log_dir = log_dir
        self.appctx.settings.setValue("logDir", str(log_dir))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_dir)))
        self.status.showMessage(f"Opened log folder: {log_dir}", 5000)

    def closeEvent(self, e):
        if self.appctx.is_job_running():
            job_action = self._prompt_job_running_action()
            if job_action == "cancel":
                self.on_cancel()
                e.ignore()
                return
            if job_action != "force":
                e.ignore()
                return
            if self.runctl.cancel():
                self.status.showMessage("Force stopping job...", 5000)
            self._job_running = False
            self.appctx.set_job_running(False)

        if self.appctx.is_dirty():
            dirty_action = self._prompt_unsaved_changes()
            if dirty_action == "cancel":
                e.ignore()
                return
            if dirty_action == "save":
                self.on_save_recipe()
                if self.appctx.is_dirty():
                    e.ignore()
                    return
            elif dirty_action == "discard":
                self.appctx.set_dirty(False)

        self.save_state()
        super().closeEvent(e)

    def _prompt_job_running_action(self) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Job Running")
        box.setText(
            "A processing job is currently running. Choose an option to continue."
        )
        cancel_job = box.addButton(
            "Cancel Job", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        force_stop = box.addButton(
            "Force Stop", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        stay_open = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(stay_open)
        box.exec()
        clicked = box.clickedButton()
        if clicked == cancel_job:
            return "cancel"
        if clicked == force_stop:
            return "force"
        return "abort"

    def _prompt_unsaved_changes(self) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved Changes")
        box.setText(
            "You have unsaved recipe changes. Do you want to save before exiting?"
        )
        save_btn = box.addButton("Save", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(
            "Discard", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == save_btn:
            return "save"
        if clicked == discard_btn:
            return "discard"
        return "cancel"

    def save_state(self):
        s = self.appctx.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())

    def restore_state(self):
        s = self.appctx.settings
        g = s.value("geometry"); w = s.value("windowState")
        if g: self.restoreGeometry(g)
        if w: self.restoreState(w)

    # ------------------------------------------------------------------
    # Run controller plumbing
    def _connect_run_controller(self):
        self.runctl.job_started.connect(self._on_job_started)
        self.runctl.job_finished.connect(self._on_job_finished)
        self.runctl.job_progress.connect(self._on_job_progress)
        self.runctl.job_message.connect(self._on_job_message)

    def _update_action_states(self):
        running = self._job_running
        has_queue = bool(self._queued_paths)
        if self._run_action:
            self._run_action.setEnabled(not running and has_queue)
        if self._cancel_action:
            self._cancel_action.setEnabled(running)
        if self._open_action:
            self._open_action.setEnabled(not running)
        if self._save_recipe_action:
            self._save_recipe_action.setEnabled(not running)
        if self._save_recipe_as_action:
            self._save_recipe_as_action.setEnabled(not running)
        if self._export_action:
            self._export_action.setEnabled(not running and self._last_result is not None)

    def _last_data_dir(self) -> Path:
        if self._queued_paths:
            return Path(self._queued_paths[-1]).parent
        if self._current_recipe_path:
            return self._current_recipe_path.parent
        return Path.home()

    def _build_queue_entries(
        self,
        paths: List[str],
        plugin: Optional[object] = None,
    ) -> List[QueueEntry]:
        if not paths:
            return []

        normalized = [str(Path(p)) for p in paths]
        active_plugin = plugin
        if active_plugin is None:
            candidate = self._recipe_data.get("module")
            active_plugin = self._plugin_registry.get(candidate)
            if active_plugin is None and normalized:
                active_plugin = self._resolve_plugin(normalized, candidate)

        plugin_id = getattr(active_plugin, "id", None) if active_plugin else None
        plugin_label = getattr(active_plugin, "label", None) if active_plugin else None
        technique_label: Optional[str]
        if plugin_label:
            technique_label = str(plugin_label)
        elif isinstance(plugin_id, str):
            technique_label = plugin_id.upper()
        else:
            technique_label = None

        manifest_supported = bool(active_plugin and hasattr(active_plugin, "_is_manifest_file"))
        manifest_entries: List[Dict[str, object]] = []
        manifest_lookup: Dict[str, Dict[str, Dict[str, object]]] = {}
        manifest_files: set[str] = set()
        manifest_errors: set[str] = set()

        if manifest_supported:
            for path_str in normalized:
                path_obj = Path(path_str)
                try:
                    is_manifest = bool(active_plugin._is_manifest_file(path_obj))  # type: ignore[attr-defined]
                except Exception:
                    is_manifest = False
                if is_manifest:
                    manifest_files.add(path_str)
                    if hasattr(active_plugin, "_parse_manifest_file"):
                        try:
                            parsed = active_plugin._parse_manifest_file(path_obj)  # type: ignore[attr-defined]
                        except Exception:
                            manifest_errors.add(path_str)
                        else:
                            if parsed:
                                manifest_entries.extend(parsed)
            if manifest_entries and hasattr(active_plugin, "_build_manifest_lookup"):
                try:
                    manifest_lookup = active_plugin._build_manifest_lookup(manifest_entries)  # type: ignore[attr-defined]
                except Exception:
                    manifest_lookup = {}

        entries: List[QueueEntry] = []
        for path_str in normalized:
            path_obj = Path(path_str)
            is_manifest = path_str in manifest_files
            manifest_status: Optional[str]
            manifest_meta: Dict[str, object] = {}

            if active_plugin is None:
                manifest_status = None
            elif is_manifest:
                manifest_status = "manifest-error" if path_str in manifest_errors else "manifest"
            elif manifest_supported:
                if manifest_lookup and hasattr(active_plugin, "_lookup_manifest_entries"):
                    try:
                        manifest_meta = active_plugin._lookup_manifest_entries(  # type: ignore[attr-defined]
                            manifest_lookup,
                            path_obj,
                            {},
                        ) or {}
                    except Exception:
                        manifest_meta = {}
                if manifest_meta:
                    manifest_status = "linked"
                elif manifest_files:
                    manifest_status = "missing"
                else:
                    manifest_status = "none"
            else:
                manifest_status = "unsupported"

            role: Optional[str] = None
            mode: Optional[str] = None
            if manifest_meta:
                role_value = manifest_meta.get("role")
                if isinstance(role_value, str):
                    role = role_value.strip().lower() or None
                mode_value = manifest_meta.get("mode")
                if isinstance(mode_value, str):
                    mode = mode_value.strip().lower() or None

            metadata = dict(manifest_meta)
            display_name = path_obj.name or path_str
            entries.append(
                QueueEntry(
                    path=path_str,
                    display_name=display_name,
                    plugin_id=plugin_id,
                    technique=technique_label,
                    mode=mode,
                    role=role,
                    manifest_status=manifest_status,
                    is_manifest=is_manifest,
                    metadata=metadata,
                )
            )

        return entries

    def _refresh_queue_metadata(self) -> None:
        if not self._queued_paths:
            self.fileDock.set_entries([])
            return
        plugin = self._plugin_registry.get(self._active_plugin_id)
        if plugin is None:
            plugin = self._resolve_plugin(
                self._queued_paths, self._recipe_data.get("module")
            )
        entries = self._build_queue_entries(self._queued_paths, plugin)
        self.fileDock.apply_metadata(entries)

    def _set_queue(self, paths: List[str]):
        self._queued_paths = [str(p) for p in paths]
        plugin = None
        if self._queued_paths:
            plugin = self._resolve_plugin(
                self._queued_paths, self._recipe_data.get("module")
            )
        entries = self._build_queue_entries(self._queued_paths, plugin)
        self.fileDock.set_entries(entries)
        self._last_result = None
        self._update_action_states()
        if plugin:
            self._select_module(plugin.id)
        message = (
            f"Queued {len(self._queued_paths)} files"
            if self._queued_paths
            else "Queue cleared"
        )
        self.status.showMessage(message, 5000)

    # ------------------------------------------------------------------
    # File queue interactions
    def _on_queue_paths_dropped(self, paths: List[str]):
        if not paths:
            return
        self._set_queue(paths)
        first_existing = next((Path(p) for p in paths if Path(p).exists()), None)
        if first_existing:
            target_dir = first_existing.parent if first_existing.is_file() else first_existing
            if target_dir.is_dir():
                self._export_default_dir = target_dir

    def _on_queue_inspect_requested(self, path: str):
        self._open_file_preview(Path(path), "Inspect Header", 4096)

    def _on_queue_preview_requested(self, path: str):
        self._open_file_preview(Path(path), "Preview", 65536)

    def _on_queue_locate_requested(self, path: str):
        target = Path(path)
        if not target.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"The file '{path}' could not be found on disk.",
            )
            return
        directory = target.parent if target.is_file() else target
        QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory.resolve())))

    def _open_file_preview(self, path: Path, title: str, max_bytes: int):
        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"The file '{path}' could not be found on disk.",
            )
            return

        try:
            with path.open("rb") as handle:
                data = handle.read(max_bytes + 1)
        except Exception as exc:  # pragma: no cover - filesystem error path
            QtWidgets.QMessageBox.critical(
                self,
                f"Unable to Open {title}",
                f"Failed to open '{path}': {exc}",
            )
            return

        text = data.decode("utf-8", errors="replace")
        truncated = len(data) > max_bytes
        if truncated:
            text += "\n\n… output truncated …"

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{title} — {path.name}")
        dialog.setModal(True)

        layout = QtWidgets.QVBoxLayout(dialog)
        info = QtWidgets.QLabel(str(path))
        info.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        layout.addWidget(info)

        text_edit = QtWidgets.QPlainTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)

        if truncated:
            notice = QtWidgets.QLabel(
                "Only a portion of the file is shown to keep the preview responsive."
            )
            notice.setWordWrap(True)
            layout.addWidget(notice)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.resize(900, 600 if max_bytes > 8192 else 520)
        dialog.exec()

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
        self._refresh_queue_metadata()

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
        self.progress.setValue(0)
        self.previewDock.clear()
        self.qcDock.clear()
        self.loggerDock.clear()
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
            self._last_result = None
            if isinstance(result, RuntimeError) and str(result) == "Cancelled":
                self.status.showMessage("Job cancelled.", 5000)
                self.loggerDock.append_line("Job cancelled.")
            else:
                message = str(result) or result.__class__.__name__
                QtWidgets.QMessageBox.critical(
                    self, "Processing Failed", message
                )
                self.status.showMessage("Processing failed.", 5000)
                self.previewDock.show_error(message)
                self.loggerDock.append_line(f"Error: {message}")
            self._update_action_states()
            return

        if isinstance(result, BatchResult):
            self._last_result = result
            self.previewDock.show_batch_result(result)
            self.qcDock.show_qc_table(result.qc_table)
            if result.audit:
                self.loggerDock.stream_lines(result.audit)
            else:
                self.loggerDock.append_line("No audit messages were produced.")
            self.status.showMessage("Processing complete.", 5000)
        else:
            self._last_result = None
            self.previewDock.show_error("Unexpected job result.")
            self.loggerDock.append_line(
                f"Received unexpected job result: {result!r}"
            )
            self.status.showMessage(
                "Processing finished with unexpected result.", 5000
            )

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

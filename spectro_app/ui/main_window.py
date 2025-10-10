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
from spectro_app.ui.dialogs.raw_preview import RawDataPreviewWindow
from spectro_app.ui.docks.file_queue import FileQueueDock, QueueEntry
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
        self._recent_menu = None
        self._recent_placeholder_text = "No Recent Files"
        self._view_menu: Optional[QtWidgets.QMenu] = None
        self._panels_menu: Optional[QtWidgets.QMenu] = None
        self._default_layout_state: Optional[QtCore.QByteArray] = None
        build_menus(self)
        self._refresh_recent_menu()
        self._collect_actions()
        self._plugin_registry = {
            plugin.id: plugin
            for plugin in (
                UvVisPlugin(),
                FtirPlugin(),
                RamanPlugin(),
                PeesPlugin(),
            )
        }
        self._mode_selector: Optional[QtWidgets.QComboBox] = None
        self._init_toolbar()
        self._init_docks()
        self._default_geometry: Optional[QtCore.QByteArray] = QtCore.QByteArray(
            self.saveGeometry()
        )
        self._default_layout_state: Optional[QtCore.QByteArray] = QtCore.QByteArray(
            self.saveState()
        )
        self._init_view_menu()
        self._init_status()
        self.restore_state()

        self.runctl = RunController(self)
        self._connect_run_controller()

        self._queued_paths: List[str] = []
        self._queue_overrides: Dict[str, Dict[str, object]] = {}
        self._recipe_data: Dict[str, Any] = self._normalise_recipe_data({})
        self.fileDock.set_plugin_resolver(
            lambda paths: self._resolve_plugin(paths, self._recipe_data.get("module"))
        )
        self._current_recipe_path: Optional[Path] = None
        self._last_browsed_dir: Optional[Path] = None
        self._last_result: Optional[BatchResult] = None
        self._active_plugin = None
        self._active_plugin_id: Optional[str] = None
        self._export_default_dir: Optional[Path] = None
        self._default_palette: Optional[QtGui.QPalette] = None
        self._default_style_name: Optional[str] = None
        self._log_dir: Optional[Path] = None
        self._loading_session: bool = False
        self._updating_ui: bool = False
        self._job_running = False
        self._raw_preview_window: Optional[RawDataPreviewWindow] = None
        self._load_app_settings()

        self._populate_plugin_selectors()
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
        self.fileDock.overrides_changed.connect(self._on_queue_overrides_changed)
        self.fileDock.clear_requested.connect(self._on_queue_clear_requested)
        self.fileDock.remove_requested.connect(self._on_queue_remove_requested)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.fileDock)
        self.recipeDock = RecipeEditorDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.recipeDock)
        self.previewDock = PreviewDock(self)
        self.setCentralWidget(self.previewDock)
        self.qcDock = QCDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.qcDock)
        self.loggerDock = LoggerDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.loggerDock)

    def _init_view_menu(self):
        if not self._view_menu:
            # Fallback lookup in case build_menus didn't set the attribute for any reason
            for action in self.menuBar().actions():
                menu = action.menu()
                if menu and menu.objectName() == "viewMenu":
                    self._view_menu = menu
                    break

        if not self._view_menu:
            self._default_layout_state = QtCore.QByteArray(self.saveState())
            return

        self._view_menu.addSeparator()
        panels_menu = self._view_menu.addMenu("Panels")
        self._panels_menu = panels_menu

        for dock in (self.fileDock, self.recipeDock, self.qcDock, self.loggerDock):
            panels_menu.addAction(dock.toggleViewAction())

        self._default_layout_state = QtCore.QByteArray(self.saveState())

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
        if self._plugin_registry:
            self._toolbar.addSeparator()
            label = QtWidgets.QLabel("Mode:")
            label.setObjectName("ModeSelectorLabel")
            self._toolbar.addWidget(label)
            self._mode_selector = QtWidgets.QComboBox()
            self._mode_selector.setObjectName("ModeSelector")
            self._mode_selector.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
            )
            self._mode_selector.currentIndexChanged.connect(
                self._on_mode_selector_changed
            )
            self._toolbar.addWidget(self._mode_selector)

    def _init_status(self):
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.status.addPermanentWidget(self.progress)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def _collect_actions(self):
        actions = {action.text(): action for action in self.findChildren(QtGui.QAction)}
        self._open_action = actions.get("Open...")
        self._mass_load_action = actions.get("Mass Load...")
        self._save_recipe_action = actions.get("Save Recipe")
        self._save_recipe_as_action = actions.get("Save Recipe As...")
        self._export_action = actions.get("Export Workbook...")
        self._run_action = actions.get("Run")
        self._cancel_action = actions.get("Cancel")

    def _populate_plugin_selectors(self) -> None:
        modules: List[tuple[str, str]] = []
        for plugin_id, plugin in sorted(
            self._plugin_registry.items(),
            key=lambda item: getattr(item[1], "label", item[0]).lower(),
        ):
            label = getattr(plugin, "label", plugin_id) or plugin_id
            modules.append((plugin_id, str(label)))

        if self.recipeDock and hasattr(self.recipeDock, "set_available_modules"):
            self.recipeDock.set_available_modules(modules)

        if self._mode_selector:
            previous = self._mode_selector.blockSignals(True)
            try:
                self._mode_selector.clear()
                for module_id, display_label in modules:
                    self._mode_selector.addItem(display_label, module_id)
            finally:
                self._mode_selector.blockSignals(previous)

    def _load_app_settings(self):
        settings = self.appctx.settings
        self._restore_session_state(settings)
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

    def _restore_session_state(self, settings: QtCore.QSettings) -> None:
        raw_state = settings.value("session/state", "")
        if isinstance(raw_state, (bytes, bytearray)):
            raw_state = raw_state.decode("utf-8", errors="ignore")
        payload: Dict[str, Any] = {}
        if isinstance(raw_state, str) and raw_state.strip():
            try:
                parsed = json.loads(raw_state)
            except (TypeError, ValueError):
                parsed = {}
            if isinstance(parsed, dict):
                payload = parsed

        previous_loading = self._loading_session
        self._loading_session = True
        try:
            recipe_payload = (
                payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
            )
            self._recipe_data = self._normalise_recipe_data(recipe_payload)
            previous_state = self._updating_ui
            self._updating_ui = True
            try:
                self.recipeDock.set_recipe(self._recipe_data)
            finally:
                self._updating_ui = previous_state

            queue_items: List[str] = []
            stored_queue = payload.get("queue")
            if isinstance(stored_queue, list):
                for item in stored_queue:
                    if isinstance(item, str) and item:
                        queue_items.append(item)

            overrides: Dict[str, Dict[str, object]] = {}
            stored_overrides = payload.get("overrides")
            if isinstance(stored_overrides, dict):
                for path_str, values in stored_overrides.items():
                    if not isinstance(path_str, str) or not isinstance(values, dict):
                        continue
                    try:
                        normalized = str(Path(path_str))
                    except (TypeError, ValueError):
                        normalized = path_str
                    overrides[normalized] = {
                        str(key): value for key, value in values.items() if isinstance(key, str)
                    }
            self._queue_overrides = overrides
            self.fileDock.load_overrides(self._queue_overrides)

            stored_recipe_path = payload.get("current_recipe_path")
            if isinstance(stored_recipe_path, str) and stored_recipe_path.strip():
                try:
                    self._current_recipe_path = Path(stored_recipe_path)
                except (TypeError, ValueError):
                    self._current_recipe_path = None
            else:
                self._current_recipe_path = None

            stored_last_dir = payload.get("last_data_dir")
            if isinstance(stored_last_dir, str) and stored_last_dir.strip():
                try:
                    self._last_browsed_dir = Path(stored_last_dir)
                except (TypeError, ValueError):
                    self._last_browsed_dir = None

            if queue_items:
                self._set_queue(queue_items)
            else:
                self._set_queue([])

            if hasattr(self.appctx, "set_dirty"):
                try:
                    self.appctx.set_dirty(False)
                except Exception:
                    pass
        finally:
            self._loading_session = previous_loading

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
            self._add_to_queue([normalized])
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
            self._add_to_queue(data_paths)
            try:
                self._export_default_dir = Path(data_paths[0]).parent
                self.appctx.settings.setValue(
                    "export/defaultDir", str(self._export_default_dir)
                )
            except (TypeError, ValueError):
                self._export_default_dir = None

        self._record_recent_files(paths)

    def on_mass_load(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", str(self._last_data_dir())
        )
        if not directory:
            return

        try:
            root = Path(directory)
        except (TypeError, ValueError):
            return

        files: List[str] = [str(path) for path in root.rglob("*") if path.is_file()]

        if files:
            self._add_to_queue(files)

        if root.is_dir():
            self._export_default_dir = root
            self.appctx.settings.setValue("export/defaultDir", str(root))
            self._last_browsed_dir = root

        self._record_recent_files(files)

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
        default_geometry = getattr(self, "_default_geometry", None)
        restored_geometry = False
        if default_geometry:
            try:
                self.restoreGeometry(QtCore.QByteArray(default_geometry))
                restored_geometry = True
            except Exception:
                pass

        default_state = getattr(self, "_default_layout_state", None)
        restored_state = False
        if default_state:
            try:
                self.restoreState(QtCore.QByteArray(default_state))
                restored_state = True
            except Exception:
                pass

        if not (restored_geometry or restored_state):
            self.restore_state()
        if self._default_layout_state is not None:
            self.restoreState(QtCore.QByteArray(self._default_layout_state))
        else:
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
        if hasattr(plugin, "set_queue_overrides"):
            overrides = self._queue_overrides if isinstance(self._queue_overrides, dict) else {}
            try:
                plugin.set_queue_overrides(overrides)
            except Exception:
                try:
                    plugin.set_queue_overrides({})
                except Exception:
                    pass
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
        session_payload = {
            "recipe": copy.deepcopy(self._recipe_data),
            "queue": list(self._queued_paths),
            "overrides": {
                str(path): dict(values)
                for path, values in self._queue_overrides.items()
            },
            "current_recipe_path": (
                str(self._current_recipe_path) if self._current_recipe_path else None
            ),
            "last_data_dir": str(self._last_browsed_dir) if self._last_browsed_dir else None,
        }

        def _json_default(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

        try:
            encoded = json.dumps(session_payload, default=_json_default)
        except TypeError:
            encoded = json.dumps({})
        s.setValue("session/state", encoded)

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
        if self._mass_load_action:
            self._mass_load_action.setEnabled(not running)
        if self._save_recipe_action:
            self._save_recipe_action.setEnabled(not running)
        if self._save_recipe_as_action:
            self._save_recipe_as_action.setEnabled(not running)
        if self._export_action:
            self._export_action.setEnabled(not running and self._last_result is not None)

    def _last_data_dir(self) -> Path:
        if self._queued_paths:
            try:
                last_path = Path(self._queued_paths[-1])
            except (TypeError, ValueError):
                last_path = None
            else:
                if last_path.exists() and last_path.is_dir():
                    return last_path
                if last_path.parent != last_path:
                    return last_path.parent
        if self._current_recipe_path:
            return self._current_recipe_path.parent
        if self._last_browsed_dir:
            return self._last_browsed_dir
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

        manifest_supported = False
        if active_plugin is not None:
            manifest_supported = bool(
                getattr(active_plugin, "manifest_ui_capability_enabled", False)
            )
        manifest_detection_available = bool(
            active_plugin and hasattr(active_plugin, "_is_manifest_file")
        )
        manifest_supported = self._plugin_supports_manifest_ui(active_plugin)
        manifest_entries: List[Dict[str, object]] = []
        manifest_lookup: Dict[str, Dict[str, Dict[str, object]]] = {}
        manifest_files: set[str] = set()
        manifest_errors: set[str] = set()

        if manifest_detection_available:
            for path_str in normalized:
                path_obj = Path(path_str)
                try:
                    is_manifest = bool(active_plugin._is_manifest_file(path_obj))  # type: ignore[attr-defined]
                except Exception:
                    is_manifest = False
                if is_manifest:
                    manifest_files.add(path_str)
                    if manifest_supported and hasattr(active_plugin, "_parse_manifest_file"):
                        try:
                            parsed = active_plugin._parse_manifest_file(path_obj)  # type: ignore[attr-defined]
                        except Exception:
                            manifest_errors.add(path_str)
                        else:
                            if parsed:
                                manifest_entries.extend(parsed)
            if (
                manifest_supported
                and manifest_entries
                and hasattr(active_plugin, "_build_manifest_lookup")
            ):
                try:
                    manifest_lookup = active_plugin._build_manifest_lookup(manifest_entries)  # type: ignore[attr-defined]
                except Exception:
                    manifest_lookup = {}

        entries: List[QueueEntry] = []
        for path_str in normalized:
            path_obj = Path(path_str)
            is_manifest_candidate = path_str in manifest_files
            is_manifest = bool(is_manifest_candidate and manifest_supported)
            manifest_status: Optional[str] = None
            manifest_meta: Dict[str, object] = {}

            if active_plugin is None:
                manifest_status = None
            elif is_manifest_candidate and not manifest_supported:
                manifest_status = "unsupported"
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

            role: Optional[str] = None
            mode: Optional[str] = None
            if manifest_meta:
                role_value = manifest_meta.get("role")
                if isinstance(role_value, str):
                    role = role_value.strip().lower() or None
                mode_value = manifest_meta.get("mode")
                if isinstance(mode_value, str):
                    mode = mode_value.strip().lower() or None

            metadata = dict(manifest_meta) if manifest_supported else {}
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

    @staticmethod
    def _plugin_supports_manifest_ui(plugin: Optional[object]) -> bool:
        if not plugin or not hasattr(plugin, "_is_manifest_file"):
            return False

        supports_manifest = getattr(plugin, "supports_manifest_ui", None)
        if callable(supports_manifest):
            try:
                return bool(supports_manifest())
            except Exception:
                return True

        expose_attr = getattr(plugin, "expose_manifest_ui", None)
        if expose_attr is None:
            return True

        try:
            return bool(expose_attr()) if callable(expose_attr) else bool(expose_attr)
        except Exception:
            return True

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

    def _add_to_queue(self, paths: List[str], *, skip_duplicates: bool = True) -> None:
        if not paths:
            return
        combined = list(self._queued_paths)
        combined.extend(str(p) for p in paths if p)
        self._set_queue(combined, skip_duplicates=skip_duplicates)

    def _set_queue(self, paths: List[str], *, skip_duplicates: bool = False):
        cleaned_paths: List[str] = []
        seen: set[str] = set()
        for raw in paths:
            if not raw:
                continue
            try:
                normalized = str(raw)
            except Exception:
                continue
            if skip_duplicates:
                if normalized in seen:
                    continue
                seen.add(normalized)
            cleaned_paths.append(normalized)

        self._queued_paths = cleaned_paths
        if self._queued_paths and not self._loading_session:
            self._update_last_browsed_dir(self._queued_paths)
        plugin = None
        if self._queued_paths:
            plugin = self._resolve_plugin(
                self._queued_paths, self._recipe_data.get("module")
            )
        entries = self._build_queue_entries(self._queued_paths, plugin)
        self.fileDock.set_entries(entries)
        if not self._queued_paths:
            self.previewDock.clear()
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

    def _update_last_browsed_dir(self, paths: List[str]) -> None:
        if not paths:
            return
        candidate_dir: Optional[Path] = None
        for raw in paths:
            try:
                candidate = Path(raw)
            except (TypeError, ValueError):
                continue
            if candidate.exists():
                candidate_dir = candidate if candidate.is_dir() else candidate.parent
                break
        if candidate_dir is None:
            try:
                fallback = Path(paths[0])
            except (TypeError, ValueError):
                fallback = None
            else:
                if fallback.exists() and fallback.is_dir():
                    candidate_dir = fallback
                else:
                    candidate_dir = fallback.parent if fallback.parent != fallback else None
        if candidate_dir and str(candidate_dir):
            self._last_browsed_dir = candidate_dir

    # ------------------------------------------------------------------
    # File queue interactions
    def _on_queue_paths_dropped(self, paths: List[str]):
        if not paths:
            return
        self._add_to_queue(paths)
        first_existing = next((Path(p) for p in paths if Path(p).exists()), None)
        if first_existing:
            target_dir = first_existing.parent if first_existing.is_file() else first_existing
            if target_dir.is_dir():
                self._export_default_dir = target_dir

    def _on_queue_clear_requested(self) -> None:
        self._set_queue([])

    def _on_queue_remove_requested(self, paths: List[str]) -> None:
        if not paths:
            return

        removal_values: set[str] = set()
        normalized_values: set[str] = set()
        for raw in paths:
            if raw is None:
                continue
            try:
                as_str = str(raw)
            except Exception:
                continue
            if not as_str:
                continue
            removal_values.add(as_str)
            try:
                normalized_values.add(str(Path(as_str)))
            except (TypeError, ValueError):
                normalized_values.add(as_str)

        if not removal_values:
            return

        remaining = [path for path in self._queued_paths if path not in removal_values]
        if len(remaining) == len(self._queued_paths):
            return

        for key in list(self._queue_overrides.keys()):
            if key in normalized_values:
                self._queue_overrides.pop(key, None)

        self._set_queue(remaining)

        if not self._loading_session:
            self.appctx.set_dirty(True)

    def _on_queue_inspect_requested(self, path: str):
        self._open_file_data_dialog(Path(path))

    def _on_queue_preview_requested(self, path: str):
        candidate = Path(path)
        str_path = str(candidate)
        plugin = None
        if self._active_plugin_id:
            plugin = self._plugin_registry.get(self._active_plugin_id)
        if plugin is None:
            plugin = self._resolve_plugin([str_path])
        if plugin is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Preview Unavailable",
                "Could not determine a plugin to load the selected file for preview.",
            )
            return

        try:
            spectra = plugin.load([str_path])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Preview Failed",
                f"Loading spectra for preview failed: {exc}",
            )
            return

        if not spectra:
            QtWidgets.QMessageBox.warning(
                self,
                "Preview Unavailable",
                "The selected file did not produce any spectra to preview.",
            )
            return

        if self._raw_preview_window is None:
            self._raw_preview_window = RawDataPreviewWindow(self)
            self._raw_preview_window.destroyed.connect(self._on_raw_preview_destroyed)

        try:
            self._raw_preview_window.plot_spectra(spectra)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Preview Failed",
                f"Displaying spectra failed: {exc}",
            )
            return

        self._raw_preview_window.show()
        self._raw_preview_window.raise_()
        self._raw_preview_window.activateWindow()

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

    def _on_raw_preview_destroyed(self, *_args):
        self._raw_preview_window = None

    def _on_queue_overrides_changed(self, overrides: Dict[str, Dict[str, object]]):
        if overrides is None:
            self._queue_overrides = {}
        else:
            self._queue_overrides = {
                str(Path(path)): dict(values)
                for path, values in overrides.items()
            }
        if not self._loading_session:
            self.appctx.set_dirty(True)

    def _open_file_text_preview(self, path: Path, title: str, max_bytes: int):
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
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _open_file_data_dialog(self, path: Path) -> None:
        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"The file '{path}' could not be found on disk.",
            )
            return

        plugin = self._active_plugin
        if plugin is None and self._active_plugin_id:
            plugin = self._plugin_registry.get(self._active_plugin_id)
        if plugin is None:
            plugin = self._resolve_plugin([str(path)], self._recipe_data.get("module"))
        if plugin is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Compatible Plugin",
                "Unable to determine a plugin capable of loading this file for inspection.",
            )
            return

        try:
            spectra = plugin.load([str(path)])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to Inspect Data",
                f"Failed to load '{path}': {exc}",
            )
            return

        if not spectra:
            QtWidgets.QMessageBox.information(
                self,
                "No Data Found",
                f"No spectra were returned when loading '{path}'.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Inspect Data — {path.name}")
        dialog.setModal(True)

        layout = QtWidgets.QVBoxLayout(dialog)

        info_lines = [str(path.resolve())]
        plugin_label = getattr(plugin, "label", None) or getattr(plugin, "id", "")
        if plugin_label:
            info_lines.append(f"Plugin: {plugin_label}")
        info_label = QtWidgets.QLabel("\n".join(info_lines))
        info_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        layout.addWidget(info_label)

        selector: Optional[QtWidgets.QComboBox] = None
        if len(spectra) > 1:
            selector = QtWidgets.QComboBox()
            selector.setObjectName("InspectorSpectrumSelector")
            for index, spectrum in enumerate(spectra):
                label_candidates = [
                    spectrum.meta.get("sample_id") if isinstance(spectrum.meta, dict) else None,
                    spectrum.meta.get("name") if isinstance(spectrum.meta, dict) else None,
                    spectrum.meta.get("id") if isinstance(spectrum.meta, dict) else None,
                ]
                display = next((str(candidate) for candidate in label_candidates if candidate), None)
                if not display:
                    display = f"Spectrum {index + 1}"
                selector.addItem(display, index)
            layout.addWidget(selector)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        metadata_tree = QtWidgets.QTreeWidget()
        metadata_tree.setObjectName("InspectorMetadataTree")
        metadata_tree.setColumnCount(2)
        metadata_tree.setHeaderLabels(["Field", "Value"])
        metadata_tree.header().setStretchLastSection(True)
        splitter.addWidget(metadata_tree)

        data_table = QtWidgets.QTableWidget()
        data_table.setObjectName("InspectorDataTable")
        data_table.setColumnCount(2)
        data_table.setHorizontalHeaderLabels(["Wavelength", "Intensity"])
        data_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        data_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        data_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        data_table.verticalHeader().setVisible(False)
        data_table.horizontalHeader().setStretchLastSection(True)
        data_table.setAlternatingRowColors(True)
        splitter.addWidget(data_table)

        def _format_number(value: object) -> str:
            if value is None:
                return ""
            try:
                if isinstance(value, (int, bool)):
                    return str(int(value))
                numeric = float(value)
            except Exception:
                return str(value)
            else:
                return format(numeric, "g")

        def _coerce_sequence(values: object) -> List[object]:
            if values is None:
                return []
            if hasattr(values, "tolist") and callable(getattr(values, "tolist")):
                try:
                    values = values.tolist()
                except Exception:
                    pass
            if isinstance(values, (list, tuple)):
                return list(values)
            try:
                return list(values)
            except Exception:
                return [values]

        def _format_value(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, dict):
                return ""
            if isinstance(value, (list, tuple)):
                return ", ".join(_format_value(item) for item in value)
            if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
                try:
                    return ", ".join(
                        _format_number(item) for item in value.tolist()
                    )
                except Exception:
                    return str(value)
            if isinstance(value, (int, bool)):
                return str(int(value))
            try:
                return format(float(value), "g")
            except Exception:
                return str(value)

        def _populate_metadata(parent: QtWidgets.QTreeWidgetItem, value: object) -> None:
            if isinstance(value, dict):
                for key in sorted(value):
                    child = QtWidgets.QTreeWidgetItem(parent, [str(key), ""])
                    _populate_metadata(child, value[key])
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    child = QtWidgets.QTreeWidgetItem(parent, [str(index), ""])
                    _populate_metadata(child, item)
            else:
                parent.setText(1, _format_value(value))

        def _update_contents(index: int) -> None:
            if index < 0 or index >= len(spectra):
                return
            spectrum = spectra[index]
            metadata_tree.clear()
            title_label = f"Spectrum {index + 1}"
            if isinstance(spectrum.meta, dict):
                sample_id = spectrum.meta.get("sample_id")
                if sample_id:
                    title_label += f" — {sample_id}"
            root = QtWidgets.QTreeWidgetItem([title_label, ""])
            metadata_tree.addTopLevelItem(root)
            if isinstance(spectrum.meta, dict):
                _populate_metadata(root, spectrum.meta)
            metadata_tree.expandAll()

            wavelengths = _coerce_sequence(getattr(spectrum, "wavelength", []))
            intensities = _coerce_sequence(getattr(spectrum, "intensity", []))
            row_count = max(len(wavelengths), len(intensities))
            data_table.setRowCount(row_count)
            for row in range(row_count):
                if row < len(wavelengths):
                    data_table.setItem(
                        row,
                        0,
                        QtWidgets.QTableWidgetItem(
                            _format_number(wavelengths[row])
                        ),
                    )
                else:
                    data_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))

                if row < len(intensities):
                    data_table.setItem(
                        row,
                        1,
                        QtWidgets.QTableWidgetItem(
                            _format_number(intensities[row])
                        ),
                    )
                else:
                    data_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
            data_table.resizeColumnsToContents()

        _update_contents(0)

        if selector is not None:
            selector.currentIndexChanged.connect(_update_contents)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.resize(960, 640)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

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
        module_id = self.recipeDock.module.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if not isinstance(module_id, str) or not module_id.strip():
            module_id = (module_text or "").strip().lower()
        if not module_id:
            return
        self._select_module(module_id, user_initiated=True)

    def _on_mode_selector_changed(self, index: int):
        if self._updating_ui:
            return
        if not self._mode_selector:
            return
        module_id = self._mode_selector.itemData(
            index, QtCore.Qt.ItemDataRole.UserRole
        )
        if not isinstance(module_id, str) or not module_id.strip():
            module_id = (self._mode_selector.itemText(index) or "").strip().lower()
        if not module_id:
            return
        if module_id == self._recipe_data.get("module"):
            return
        self._select_module(module_id, user_initiated=True)

    def _on_recipe_widget_changed(self, *_):
        if self._updating_ui:
            return
        self._update_recipe_from_ui()
        self.appctx.set_dirty(True)

    def _sync_recipe_state(self) -> None:
        recipe_obj = self.recipeDock.get_recipe()
        if not isinstance(recipe_obj, Recipe):
            recipe_obj = Recipe()

        module = (recipe_obj.module or "uvvis").strip().lower()
        params_source = (
            recipe_obj.params if isinstance(recipe_obj.params, dict) else {}
        )
        params = copy.deepcopy(params_source)

        export_cfg = self._recipe_data.get("export")
        if not isinstance(export_cfg, dict):
            export_cfg = {}
        export_cfg = copy.deepcopy(export_cfg)

        version = str(recipe_obj.version or Recipe().version)

        combined = {
            "module": module,
            "params": params,
            "export": export_cfg,
            "version": version,
        }
        self._recipe_data = self._normalise_recipe_data(combined)
        self._active_plugin_id = module

    def _update_recipe_from_ui(self):
        self._sync_recipe_state()

    def _build_recipe_payload(self) -> Dict[str, Any]:
        self._sync_recipe_state()
        payload = copy.deepcopy(self._recipe_data)
        payload.setdefault("module", "uvvis")
        payload.setdefault("params", {})
        payload.setdefault("export", {})
        payload.setdefault("version", Recipe().version)
        self._recipe_data = copy.deepcopy(payload)
        return payload

    def _select_module(self, module_id: str, *, user_initiated: bool = False):
        module_id = (module_id or "uvvis").strip().lower()
        if module_id not in self._plugin_registry:
            return
        previous_state = self._updating_ui
        self._updating_ui = True
        try:
            index = self.recipeDock.module.findData(
                module_id, QtCore.Qt.ItemDataRole.UserRole
            )
            if index < 0:
                index = self.recipeDock.module.findText(module_id)
            if index >= 0:
                self.recipeDock.module.setCurrentIndex(index)
            if self._mode_selector:
                selector_previous = self._mode_selector.blockSignals(True)
                try:
                    selector_index = self._mode_selector.findData(
                        module_id, QtCore.Qt.ItemDataRole.UserRole
                    )
                    if selector_index < 0:
                        selector_index = self._mode_selector.findText(module_id)
                    if selector_index >= 0:
                        self._mode_selector.setCurrentIndex(selector_index)
                finally:
                    self._mode_selector.blockSignals(selector_previous)
        finally:
            self._updating_ui = previous_state
        self._recipe_data["module"] = module_id
        self._active_plugin_id = module_id
        if not previous_state:
            self._update_recipe_from_ui()
        if user_initiated:
            self.appctx.set_dirty(True)
            self._update_action_states()
            self._refresh_queue_metadata()

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
            narrative_displayed = False
            if result.report_text:
                for line in result.report_text.splitlines():
                    self.loggerDock.append_line(line)
                narrative_displayed = True
            if result.audit:
                if narrative_displayed:
                    self.loggerDock.append_line("")
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

        recipe_model = Recipe(
            module=self._recipe_data.get("module", "uvvis"),
            params=copy.deepcopy(self._recipe_data.get("params", {})),
            version=str(self._recipe_data.get("version", Recipe().version)),
        )

        self._updating_ui = True
        try:
            self.recipeDock.set_recipe(recipe_model)
        finally:
            self._updating_ui = False

        self._sync_recipe_state()
        self._refresh_queue_metadata()
        self.appctx.set_dirty(False)
        self.status.showMessage(f"Loaded recipe from {path}", 5000)
        self._update_action_states()

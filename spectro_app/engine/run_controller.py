import copy

from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable
from typing import Iterable, Optional


PREVIEW_EXPORT_DISABLED_FLAG = "_preview_export_disabled"


def _flatten_recipe(recipe: Optional[dict]) -> dict:
    """Return a copy of *recipe* with nested ``params`` merged into the top level."""

    if not recipe:
        return {}

    # Start with a shallow copy of the top-level keys except for the nested ``params``
    # mapping that holds the UI-shaped payload.
    flattened = {k: v for k, v in recipe.items() if k != "params"}

    params = recipe.get("params")
    if isinstance(params, dict):
        for key, value in params.items():
            # Preserve explicitly provided top-level overrides while exposing any
            # nested configuration (e.g. blank handling) to the plugin layer.
            flattened.setdefault(key, value)

    return flattened

class JobSignals(QObject):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object)  # BatchResult or Exception

def _sanitize_export_for_preview(recipe: dict) -> dict:
    """Return a preview-safe copy of *recipe* that disables file exports."""

    sanitized = copy.deepcopy(recipe) if recipe else {}
    export_cfg = sanitized.get("export")
    sanitized_export = {}
    if isinstance(export_cfg, dict):
        for key, value in export_cfg.items():
            key_text = str(key).strip().lower()
            if not key_text:
                sanitized_export[key] = value
                continue
            if key_text in {"path", "workbook"}:
                continue
            if key_text.startswith("pdf"):
                continue
            if key_text.startswith("recipe"):
                continue
            sanitized_export[key] = value
    sanitized_export[PREVIEW_EXPORT_DISABLED_FLAG] = True
    sanitized["export"] = sanitized_export
    return sanitized


class BatchRunnable(QRunnable):
    def __init__(
        self,
        plugin,
        paths: Iterable[str],
        recipe: dict,
        *,
        preview_export_disabled: bool = False,
    ):
        super().__init__()
        self.plugin, self.paths, self.recipe = plugin, list(paths), recipe
        self.signals = JobSignals()
        self._cancelled = False
        self._preview_export_disabled = preview_export_disabled

    def run(self):
        try:
            self._emit_message("Loading spectra...")
            self._raise_if_cancelled()
            specs = self.plugin.load(self.paths)
            self.signals.progress.emit(10)

            self._emit_message("Validating recipe...")
            flattened_recipe = _flatten_recipe(self.recipe)

            errs = self.plugin.validate(specs, flattened_recipe)
            if errs:
                raise RuntimeError("; ".join(errs))
            self.signals.progress.emit(20)

            self._emit_message("Preprocessing spectra...")
            self._raise_if_cancelled()
            specs = self.plugin.preprocess(specs, flattened_recipe)
            self.signals.progress.emit(40)

            self._emit_message("Analyzing spectra...")
            self._raise_if_cancelled()
            specs, qc = self.plugin.analyze(specs, flattened_recipe)
            self.signals.progress.emit(70)

            self._emit_message("Exporting results...")
            self._raise_if_cancelled()
            export_recipe = (
                _sanitize_export_for_preview(flattened_recipe)
                if self._preview_export_disabled
                else flattened_recipe
            )
            result = self.plugin.export(specs, qc, export_recipe)
            self.signals.progress.emit(100)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.finished.emit(e)

    def cancel(self):
        self._cancelled = True
        self._emit_message("Cancellation requested")

    def _raise_if_cancelled(self):
        if self._cancelled:
            raise RuntimeError("Cancelled")

    def _emit_message(self, message: str):
        self.signals.message.emit(message)

class RunController(QObject):
    job_started = pyqtSignal()
    job_finished = pyqtSignal(object)
    job_progress = pyqtSignal(int)
    job_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._current_runnable: Optional[BatchRunnable] = None

    def start(self, plugin, paths, recipe: dict, *, user_initiated: bool = False):
        runnable = BatchRunnable(
            plugin,
            paths,
            recipe,
            preview_export_disabled=bool(user_initiated),
        )
        runnable.signals.finished.connect(self._on_finished)
        runnable.signals.progress.connect(self.job_progress)
        runnable.signals.message.connect(self.job_message)
        self._current_runnable = runnable
        self.job_started.emit()
        self.pool.start(runnable)

    def cancel(self) -> bool:
        if self._current_runnable is None:
            return False
        self._current_runnable.cancel()
        return True

    def _on_finished(self, result):
        self._current_runnable = None
        self.job_finished.emit(result)

    def is_running(self) -> bool:
        return self._current_runnable is not None

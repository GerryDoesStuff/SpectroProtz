from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable
from typing import Iterable, Optional

class JobSignals(QObject):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object)  # BatchResult or Exception

class BatchRunnable(QRunnable):
    def __init__(self, plugin, paths: Iterable[str], recipe: dict):
        super().__init__()
        self.plugin, self.paths, self.recipe = plugin, list(paths), recipe
        self.signals = JobSignals()
        self._cancelled = False

    def run(self):
        try:
            self._emit_message("Loading spectra...")
            self._raise_if_cancelled()
            specs = self.plugin.load(self.paths)
            self.signals.progress.emit(10)

            self._emit_message("Validating recipe...")
            errs = self.plugin.validate(specs, self.recipe)
            if errs:
                raise RuntimeError("; ".join(errs))
            self.signals.progress.emit(20)

            self._emit_message("Preprocessing spectra...")
            self._raise_if_cancelled()
            specs = self.plugin.preprocess(specs, self.recipe)
            self.signals.progress.emit(40)

            self._emit_message("Analyzing spectra...")
            self._raise_if_cancelled()
            specs, qc = self.plugin.analyze(specs, self.recipe)
            self.signals.progress.emit(70)

            self._emit_message("Exporting results...")
            self._raise_if_cancelled()
            result = self.plugin.export(specs, qc, self.recipe)
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

    def start(self, plugin, paths, recipe: dict):
        runnable = BatchRunnable(plugin, paths, recipe)
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

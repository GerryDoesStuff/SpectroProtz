from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable
from typing import Iterable

class JobSignals(QObject):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object)  # BatchResult or Exception

class BatchRunnable(QRunnable):
    def __init__(self, plugin, paths: Iterable[str], recipe: dict):
        super().__init__()
        self.plugin, self.paths, self.recipe = plugin, list(paths), recipe
        self.signals = JobSignals()

    def run(self):
        try:
            specs = self.plugin.load(self.paths)
            errs = self.plugin.validate(specs, self.recipe)
            if errs:
                raise RuntimeError("; ".join(errs))
            specs = self.plugin.preprocess(specs, self.recipe)
            specs, qc = self.plugin.analyze(specs, self.recipe)
            result = self.plugin.export(specs, qc, self.recipe)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.finished.emit(e)

class RunController(QObject):
    job_started = pyqtSignal()
    job_finished = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()

    def start(self, plugin, paths, recipe: dict):
        runnable = BatchRunnable(plugin, paths, recipe)
        runnable.signals.finished.connect(self.job_finished)
        self.job_started.emit()
        self.pool.start(runnable)

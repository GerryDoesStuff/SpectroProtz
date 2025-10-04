from PyQt6.QtCore import QSettings


class AppContext:
    def __init__(self):
        # Company/app keys control where QSettings persists per OS
        self.settings = QSettings("SpectroLab", "SpectroApp")
        self._dirty = False
        self._job_running = False

    def set_dirty(self, dirty: bool):
        self._dirty = dirty

    def is_dirty(self) -> bool:
        return self._dirty

    def set_job_running(self, running: bool):
        self._job_running = running

    def is_job_running(self) -> bool:
        return self._job_running

    def maybe_close(self) -> bool:
        return not (self._job_running or self._dirty)

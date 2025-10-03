from PyQt6.QtCore import QSettings

class AppContext:
    def __init__(self):
        # Company/app keys control where QSettings persists per OS
        self.settings = QSettings("SpectroLab", "SpectroApp")
        self._dirty = False
        self._job_running = False

    def set_dirty(self, dirty: bool):
        self._dirty = dirty

    def set_job_running(self, running: bool):
        self._job_running = running

    def maybe_close(self) -> bool:
        # TODO: hook into recipe dirty state and running jobs
        return not self._job_running

from __future__ import annotations

from typing import Iterable, List

from PyQt6 import QtCore


class AppContext(QtCore.QObject):
    indexer_path_changed = QtCore.pyqtSignal(str)

    _INDEXER_RECENT_LIMIT = 8

    def __init__(self):
        super().__init__()
        # Company/app keys control where QSettings persists per OS
        self.settings = QtCore.QSettings("SpectroLab", "SpectroApp")
        self._dirty = False
        self._job_running = False
        self._indexer_last_index_path = ""
        self._indexer_recent_index_paths: List[str] = []
        self._load_indexer_settings()

    def _load_indexer_settings(self) -> None:
        stored_last = self.settings.value("indexer/lastIndexPath", "", type=str)
        self._indexer_last_index_path = stored_last or ""
        raw_recent = self.settings.value("indexer/recentIndexPaths", [], type=list)
        recent = self._coerce_settings_list(raw_recent)
        if self._indexer_last_index_path:
            recent = self._dedupe_recent_paths(
                [self._indexer_last_index_path, *recent]
            )
        self._indexer_recent_index_paths = recent[: self._INDEXER_RECENT_LIMIT]

    def _coerce_settings_list(self, value: object) -> List[str]:
        if isinstance(value, list):
            candidates = value
        elif value:
            candidates = [value]
        else:
            candidates = []
        return [str(item).strip() for item in candidates if str(item).strip()]

    def _dedupe_recent_paths(self, paths: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        deduped: List[str] = []
        for path in paths:
            if not path:
                continue
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def indexer_last_index_path(self) -> str:
        return self._indexer_last_index_path

    def indexer_recent_index_paths(self) -> List[str]:
        return list(self._indexer_recent_index_paths)

    def remember_indexer_index_path(self, path: str) -> None:
        path_text = str(path).strip()
        if not path_text:
            return
        recent = self._dedupe_recent_paths(
            [path_text, *self._indexer_recent_index_paths]
        )[: self._INDEXER_RECENT_LIMIT]
        self._indexer_recent_index_paths = recent
        self._indexer_last_index_path = path_text
        self.settings.setValue("indexer/lastIndexPath", path_text)
        self.settings.setValue("indexer/recentIndexPaths", recent)
        self.indexer_path_changed.emit(path_text)

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

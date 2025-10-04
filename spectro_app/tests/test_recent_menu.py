import os
from pathlib import Path

import pytest
from PyQt6 import QtCore, QtWidgets

from spectro_app.ui.main_window import MainWindow


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class DummyAppContext:
    def __init__(self, settings: QtCore.QSettings):
        self.settings = settings

    def set_dirty(self, dirty: bool):
        pass

    def set_job_running(self, running: bool):
        pass

    def maybe_close(self) -> bool:
        return True


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_settings(tmp_path: Path) -> QtCore.QSettings:
    settings_path = tmp_path / "test_settings.ini"
    if settings_path.exists():
        settings_path.unlink()
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)
    settings.clear()
    return settings


def test_recent_menu_shows_placeholder_when_empty(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    settings.setValue("recentFiles", [])
    window = MainWindow(DummyAppContext(settings))
    try:
        actions = window._recent_menu.actions()
        assert len(actions) == 1
        placeholder = actions[0]
        assert not placeholder.isEnabled()
        assert "No Recent Files" in placeholder.text()
    finally:
        window.close()
        window.deleteLater()


def test_recent_menu_prunes_missing_entries(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    existing = tmp_path / "data.txt"
    existing.write_text("sample")
    missing = tmp_path / "missing.txt"
    settings.setValue("recentFiles", [str(existing), str(missing)])

    window = MainWindow(DummyAppContext(settings))
    try:
        actions = [action for action in window._recent_menu.actions() if action.isEnabled()]
        assert len(actions) == 1
        assert actions[0].text() == str(existing)

        stored = window._coerce_settings_list(settings.value("recentFiles", []))
        assert stored == [str(existing)]
    finally:
        window.close()
        window.deleteLater()


def test_recent_menu_limits_to_twenty_items(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    paths = []
    for idx in range(25):
        path = tmp_path / f"file_{idx}.dat"
        path.write_text("data")
        paths.append(str(path))
    settings.setValue("recentFiles", paths)

    window = MainWindow(DummyAppContext(settings))
    try:
        actions = [action.text() for action in window._recent_menu.actions() if action.isEnabled()]
        assert actions == paths[:20]

        stored = window._coerce_settings_list(settings.value("recentFiles", []))
        assert len(stored) == 20
        assert stored == paths[:20]
    finally:
        window.close()
        window.deleteLater()

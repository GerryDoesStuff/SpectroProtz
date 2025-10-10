import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)

from PyQt6 import QtCore, QtWidgets

from spectro_app.ui.main_window import MainWindow


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


def test_mass_load_populates_queue_from_directory(qapp, tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    context = DummyAppContext(settings)
    window = MainWindow(context)

    try:
        root = tmp_path / "mass_root"
        nested = root / "nested" / "deep"
        nested.mkdir(parents=True)
        files = [
            root / "sample_one.dat",
            root / "nested" / "sample_two.spc",
            nested / "sample_three.txt",
        ]
        for file_path in files:
            file_path.write_text("data")

        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getExistingDirectory",
            lambda *args, **kwargs: str(root),
        )

        window.on_mass_load()
        qapp.processEvents()

        expected_paths = {str(path) for path in root.rglob("*") if path.is_file()}
        assert expected_paths, "Expected files were not created for the mass load test"

        assert set(window._queued_paths) == expected_paths
        assert len(window._queued_paths) == len(expected_paths)

        assert window._export_default_dir == root
        assert window._last_browsed_dir == root
        assert settings.value("export/defaultDir", "") == str(root)

        stored_recent = window._coerce_settings_list(settings.value("recentFiles", []))
        assert set(stored_recent) == expected_paths
    finally:
        window.close()
        window.deleteLater()


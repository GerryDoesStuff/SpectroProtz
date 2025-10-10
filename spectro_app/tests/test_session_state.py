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
        return True


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _make_settings(tmp_path: Path) -> QtCore.QSettings:
    settings_path = tmp_path / "session_state.ini"
    if settings_path.exists():
        settings_path.unlink()
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)
    settings.clear()
    return settings


def test_session_round_trip(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    ctx = DummyAppContext(settings)
    window = MainWindow(ctx)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_a = data_dir / "sample_a.spc"
    file_b = data_dir / "blank_b.spc"
    file_a.write_text("a")
    file_b.write_text("b")
    try:
        recipe_payload = {
            "module": "uvvis",
            "params": {
                "blank": {"subtract": True, "default": "blk-1"},
                "join": {"enabled": True, "window": 9},
            },
        }
        window._recipe_data = window._normalise_recipe_data(recipe_payload)
        window.recipeDock.set_recipe(window._recipe_data)
        window._sync_recipe_state()

        queue_paths = [str(file_a), str(file_b)]
        window._set_queue(queue_paths)
        overrides = {
            str(file_b): {"role": "blank", "blank_id": "B-001"},
            str(file_a): {"role": "sample"},
        }
        window._on_queue_overrides_changed(overrides)

        window._current_recipe_path = tmp_path / "saved.recipe.json"
        custom_dir = tmp_path / "last"
        custom_dir.mkdir()
        window._last_browsed_dir = custom_dir

        window.save_state()
        settings.sync()
        ctx.set_dirty(False)
    finally:
        ctx.set_dirty(False)
        window.close()
        window.deleteLater()

    settings_second = QtCore.QSettings(
        str(settings.fileName()), QtCore.QSettings.Format.IniFormat
    )
    ctx_second = DummyAppContext(settings_second)
    window_two = MainWindow(ctx_second)
    try:
        assert window_two._queued_paths == queue_paths
        assert window_two._queue_overrides[str(file_b)]["blank_id"] == "B-001"
        assert window_two.fileDock._overrides[str(file_b)]["blank_id"] == "B-001"

        restored_recipe = window_two.recipeDock.get_recipe()
        assert restored_recipe.params["blank"]["default"] == "blk-1"
        assert restored_recipe.params["join"]["window"] == 9

        assert window_two._current_recipe_path == tmp_path / "saved.recipe.json"
        assert window_two._last_browsed_dir == custom_dir

        window_two._set_queue([])
        window_two._current_recipe_path = None
        assert window_two._last_data_dir() == custom_dir
    finally:
        ctx_second.set_dirty(False)
        window_two.close()
        window_two.deleteLater()


def test_queue_appends_without_overwriting(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    ctx = DummyAppContext(settings)
    window = MainWindow(ctx)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_a = data_dir / "first.spc"
    file_b = data_dir / "second.spc"
    file_c = data_dir / "third.spc"
    file_a.write_text("a")
    file_b.write_text("b")
    file_c.write_text("c")

    try:
        window._set_queue([str(file_a)])
        window._on_queue_paths_dropped([str(file_b)])
        assert window._queued_paths == [str(file_a), str(file_b)]

        # Dropping an already queued file should keep the original ordering
        window._on_queue_paths_dropped([str(file_b), str(file_c)])
        assert window._queued_paths == [str(file_a), str(file_b), str(file_c)]

        # Explicitly allow duplicates when requested
        window._add_to_queue([str(file_a)], skip_duplicates=False)
        assert window._queued_paths == [
            str(file_a),
            str(file_b),
            str(file_c),
            str(file_a),
        ]

        window._on_queue_clear_requested()
        assert window._queued_paths == []
    finally:
        window.close()
        window.deleteLater()

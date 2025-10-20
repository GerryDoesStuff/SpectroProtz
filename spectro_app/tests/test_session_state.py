import os
from pathlib import Path

import pytest
import numpy as np

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


def test_queue_remove_single_entry_clears_override(qapp, tmp_path):
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
        queue = [str(file_a), str(file_b), str(file_c)]
        window._set_queue(queue)
        overrides = {
            str(file_a): {"role": "sample"},
            str(file_b): {"role": "blank"},
        }
        window._on_queue_overrides_changed(overrides)
        window.fileDock.load_overrides(overrides)

        window._on_queue_remove_requested([str(file_b)])

        assert window._queued_paths == [str(file_a), str(file_c)]
        assert str(file_b) not in window._queue_overrides
        assert str(file_b) not in window.fileDock._overrides
        assert ctx.is_dirty()
    finally:
        window.close()
        window.deleteLater()


def test_queue_remove_multiple_entries(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    ctx = DummyAppContext(settings)
    window = MainWindow(ctx)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = [data_dir / name for name in ("first.spc", "second.spc", "third.spc")]
    for idx, file in enumerate(files):
        file.write_text(str(idx))

    try:
        queue = [str(path) for path in files]
        window._set_queue(queue)
        overrides = {str(path): {"role": "sample"} for path in queue}
        window._on_queue_overrides_changed(overrides)
        window.fileDock.load_overrides(overrides)

        window._on_queue_remove_requested([str(files[0]), str(files[2])])

        assert window._queued_paths == [str(files[1])]
        assert str(files[0]) not in window._queue_overrides
        assert str(files[2]) not in window._queue_overrides
        assert str(files[0]) not in window.fileDock._overrides
        assert str(files[2]) not in window.fileDock._overrides
    finally:
        window.close()
        window.deleteLater()


def test_queue_normalises_forward_slash_paths(qapp, tmp_path):
    settings = _make_settings(tmp_path)
    ctx = DummyAppContext(settings)
    window = MainWindow(ctx)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "nested" / "sample.spc"
    file_path.parent.mkdir()
    file_path.write_text("content")

    try:
        raw_value = file_path.as_posix()
        expected = str(Path(raw_value))

        window._set_queue([raw_value])

        assert window._queued_paths == [expected]

        window._on_queue_remove_requested([expected])

        assert window._queued_paths == []
    finally:
        window.close()
        window.deleteLater()


def test_main_window_json_sanitise_handles_numpy():
    payload = {
        "array": np.array([np.array([1, 2]), np.array([3, 4])], dtype=object),
        "flag": np.bool_(True),
    }

    sanitised = MainWindow._json_sanitise(payload)

    assert sanitised["array"] == [[1, 2], [3, 4]]
    assert sanitised["flag"] is True

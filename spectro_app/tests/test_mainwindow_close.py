import pytest

try:  # pragma: no cover - environment-specific skip
    from PyQt6 import QtGui, QtWidgets
except ImportError:  # pragma: no cover - environment-specific skip
    pytest.skip("PyQt6 is unavailable", allow_module_level=True)

from spectro_app.app_context import AppContext
from spectro_app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def main_window(qapp):
    ctx = AppContext()
    window = MainWindow(ctx)
    yield window
    window.appctx.set_dirty(False)
    window.appctx.set_job_running(False)
    window.deleteLater()


def _close_event():
    return QtGui.QCloseEvent()


def test_close_event_cancel_job_allows_exit(main_window):
    main_window.appctx.set_job_running(True)

    called = {}

    def fake_prompt():
        return True

    def fake_cancel():
        called["cancelled"] = True
        return True

    main_window._prompt_job_running_action = fake_prompt
    main_window.runctl.cancel = fake_cancel
    main_window.runctl.pool.waitForDone = lambda: called.setdefault("waited", True)
    event = _close_event()

    main_window.closeEvent(event)

    assert called == {"cancelled": True, "waited": True}
    assert event.isAccepted()
    assert main_window.appctx.is_job_running() is False


def test_close_event_job_prompt_abort(main_window):
    main_window.appctx.set_job_running(True)

    called = {}

    def fake_prompt():
        return False

    main_window._prompt_job_running_action = fake_prompt
    event = _close_event()

    main_window.closeEvent(event)

    assert called == {}
    assert main_window.appctx.is_job_running() is True
    assert not event.isAccepted()


def test_close_event_dirty_save(main_window):
    main_window.appctx.set_dirty(True)

    def fake_prompt():
        return "save"

    def fake_save():
        main_window.appctx.set_dirty(False)

    main_window._prompt_unsaved_changes = fake_prompt
    main_window.on_save_recipe = fake_save
    event = _close_event()

    main_window.closeEvent(event)

    assert event.isAccepted()
    assert main_window.appctx.is_dirty() is False


def test_close_event_dirty_cancel(main_window):
    main_window.appctx.set_dirty(True)

    def fake_prompt():
        return "cancel"

    main_window._prompt_unsaved_changes = fake_prompt
    event = _close_event()

    main_window.closeEvent(event)

    assert not event.isAccepted()
    assert main_window.appctx.is_dirty()


def test_dirty_state_triggers_prompt(main_window):
    main_window.appctx.set_dirty(True)
    called = {}

    def fake_prompt():
        called["prompted"] = True
        return "cancel"

    main_window._prompt_unsaved_changes = fake_prompt
    event = _close_event()

    main_window.closeEvent(event)

    assert called == {"prompted": True}
    assert not event.isAccepted()

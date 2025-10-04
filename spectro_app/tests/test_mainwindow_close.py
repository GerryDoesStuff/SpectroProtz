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


def test_close_event_cancel_job_requests_cancel(main_window):
    main_window.appctx.set_job_running(True)

    called = {}

    def fake_prompt():
        return "cancel"

    def fake_cancel():
        called["cancelled"] = True

    main_window._prompt_job_running_action = fake_prompt
    main_window.on_cancel = fake_cancel
    event = _close_event()

    main_window.closeEvent(event)

    assert called == {"cancelled": True}
    assert not event.isAccepted()
    assert main_window.appctx.is_job_running()


def test_close_event_force_stop_allows_close(main_window):
    main_window.appctx.set_job_running(True)

    cancelled = {}

    def fake_prompt():
        return "force"

    def fake_cancel_job():
        cancelled["runctl"] = True
        return True

    main_window._prompt_job_running_action = fake_prompt
    main_window.runctl.cancel = fake_cancel_job
    event = _close_event()

    main_window.closeEvent(event)

    assert cancelled == {"runctl": True}
    assert main_window.appctx.is_job_running() is False
    assert event.isAccepted()


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

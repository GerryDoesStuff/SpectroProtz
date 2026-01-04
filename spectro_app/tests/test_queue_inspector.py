import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from spectro_app.app_context import AppContext
from spectro_app.engine.plugin_api import Spectrum, SpectroscopyPlugin
from spectro_app.ui.docks.file_queue import FileQueueDock, QueueEntry
from spectro_app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_context_menu_uses_inspect_data_label(qt_app, tmp_path, monkeypatch):
    dock = FileQueueDock()
    try:
        path = tmp_path / "sample.csv"
        path.write_text("wavelength,intensity\n1,10\n", encoding="utf-8")
        entry = QueueEntry(path=str(path), display_name=path.name)
        dock.set_entries([entry])
        dock.list.setCurrentRow(0)

        recorded: list[str] = []
        real_menu = QtWidgets.QMenu

        class RecordingMenu(real_menu):
            def addAction(self, *args):  # type: ignore[override]
                action = super().addAction(*args)
                if isinstance(action, QtGui.QAction):
                    recorded.append(action.text())
                return action

            def exec(self, *args, **kwargs):  # type: ignore[override]
                return None

        monkeypatch.setattr(QtWidgets, "QMenu", RecordingMenu)

        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse,
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 0),
        )
        dock.contextMenuEvent(event)

        assert "Inspect Data…" in recorded
    finally:
        dock.deleteLater()
        qt_app.processEvents()


class _InspectorPlugin(SpectroscopyPlugin):
    id = "inspector"
    label = "Inspector"

    def load(self, paths, cancelled=None):
        return [
            Spectrum(
                wavelength=np.array([1.0, 2.0, 3.0]),
                intensity=np.array([10.0, 11.0, 12.0]),
                meta={"sample_id": "Sample-1", "operator": "Analyst"},
            )
        ]


def test_inspector_dialog_shows_all_rows(qt_app, tmp_path, monkeypatch):
    ctx = AppContext()
    window = MainWindow(ctx)
    try:
        plugin = _InspectorPlugin()
        window._plugin_registry = {plugin.id: plugin}
        window._active_plugin = plugin
        window._active_plugin_id = plugin.id

        data_path = tmp_path / "fixture.csv"
        data_path.write_text("dummy", encoding="utf-8")

        captured: dict[str, object] = {}

        def fake_exec(self):  # type: ignore[override]
            table = self.findChild(QtWidgets.QTableWidget, "InspectorDataTable")
            assert table is not None
            tree = self.findChild(QtWidgets.QTreeWidget, "InspectorMetadataTree")
            assert tree is not None

            rows = table.rowCount()
            captured["rows"] = rows
            if rows:
                first = table.item(0, 0), table.item(0, 1)
                last = table.item(rows - 1, 0), table.item(rows - 1, 1)
                assert first[0] and first[1]
                assert last[0] and last[1]
                captured["first_row"] = (first[0].text(), first[1].text())
                captured["last_row"] = (last[0].text(), last[1].text())

            captured["metadata_titles"] = [
                tree.topLevelItem(i).text(0)
                for i in range(tree.topLevelItemCount())
            ]
            return 0

        monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec, raising=False)

        window._open_file_data_dialog(data_path)

        assert captured["rows"] == 3
        assert captured["first_row"] == ("1", "10")
        assert captured["last_row"] == ("3", "12")
        assert any("Sample-1" in title for title in captured["metadata_titles"])
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()

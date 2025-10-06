import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6 import QtWidgets  # type: ignore  # noqa: E402

from spectro_app.ui.docks.recipe_editor import RecipeEditorDock  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_recipe_editor_default_blank_flags(qt_app):
    dock = RecipeEditorDock()
    try:
        recipe = dock.get_recipe()
        blank_cfg = recipe.params.get("blank") or {}
        assert blank_cfg.get("subtract") is False
        assert blank_cfg.get("require") is False
    finally:
        dock.deleteLater()
        qt_app.processEvents()

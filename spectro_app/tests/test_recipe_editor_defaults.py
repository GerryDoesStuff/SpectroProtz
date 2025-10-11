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


def test_join_window_spikes_column_defaults(qt_app):
    dock = RecipeEditorDock()
    try:
        assert dock.join_windows_table.columnCount() == 3
        headers = [
            dock.join_windows_table.horizontalHeaderItem(i).text()
            for i in range(dock.join_windows_table.columnCount())
        ]
        assert headers == ["Min (nm)", "Max (nm)", "Spikes"]
        dock.join_enable.setChecked(True)
        qt_app.processEvents()
        dock.join_windows_add_row.click()
        qt_app.processEvents()
        assert dock.join_windows_table.rowCount() == 1
        spikes_widget = dock.join_windows_table.cellWidget(0, 2)
        assert isinstance(spikes_widget, QtWidgets.QSpinBox)
        assert spikes_widget.value() == 1
    finally:
        dock.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize(
    ("checkbox_attr", "widget_attrs", "expect_enabled_when_checked"),
    [
        ("smooth_enable", ["smooth_window", "smooth_poly"], True),
        ("despike_enable", ["despike_window", "despike_zscore"], True),
        (
            "join_enable",
            [
                "join_window",
                "join_threshold",
                "join_windows_instrument_combo",
                "join_windows_table",
                "join_windows_add_instrument",
                "join_windows_add_row",
            ],
            True,
        ),
        (
            "join_enable",
            ["join_windows_remove_instrument", "join_windows_remove_row"],
            False,
        ),
        (
            "blank_subtract",
            ["blank_require", "blank_fallback", "blank_match_strategy"],
            True,
        ),
        (
            "drift_enable",
            [
                "drift_window_min",
                "drift_window_max",
                "drift_max_slope",
                "drift_max_delta",
                "drift_max_residual",
            ],
            True,
        ),
    ],
)
def test_feature_controls_follow_checkbox_state(
    qt_app, checkbox_attr, widget_attrs, expect_enabled_when_checked
):
    dock = RecipeEditorDock()
    try:
        checkbox = getattr(dock, checkbox_attr)
        checkbox.setChecked(True)
        qt_app.processEvents()
        for widget_attr in widget_attrs:
            widget = getattr(dock, widget_attr)
            if expect_enabled_when_checked is True:
                assert widget.isEnabled()
            elif expect_enabled_when_checked is False:
                assert not widget.isEnabled()
        checkbox.setChecked(False)
        qt_app.processEvents()
        for widget_attr in widget_attrs:
            widget = getattr(dock, widget_attr)
            assert not widget.isEnabled()
    finally:
        dock.deleteLater()
        qt_app.processEvents()

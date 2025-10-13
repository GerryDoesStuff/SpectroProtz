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


def test_despike_exclusion_roundtrip(qt_app):
    dock = RecipeEditorDock()
    try:
        dock.despike_enable.setChecked(True)
        qt_app.processEvents()
        dock.despike_exclusions_add_row.click()
        qt_app.processEvents()
        dock.despike_exclusions_table.item(0, 0).setText("495")
        dock.despike_exclusions_table.item(0, 1).setText("505")
        dock.despike_exclusions_add_row.click()
        qt_app.processEvents()
        dock.despike_exclusions_table.item(1, 1).setText("420")
        qt_app.processEvents()
        dock._update_model_from_ui(force=True)
        params = dock.recipe.params.get("despike", {})
        exclusions = params.get("exclusions")
        assert isinstance(exclusions, dict)
        windows = exclusions.get("windows")
        assert isinstance(windows, list)
        assert windows and len(windows) == 2
        first, second = windows
        assert first.get("min_nm") == pytest.approx(495.0)
        assert first.get("max_nm") == pytest.approx(505.0)
        assert "min_nm" not in second or second.get("min_nm") is None
        assert second.get("max_nm") == pytest.approx(420.0)

        recipe_dict = dock.recipe_dict()
        dock.set_recipe(recipe_dict)
        qt_app.processEvents()
        assert dock.despike_exclusions_table.rowCount() == 2
        first_min = dock.despike_exclusions_table.item(0, 0)
        first_max = dock.despike_exclusions_table.item(0, 1)
        second_min = dock.despike_exclusions_table.item(1, 0)
        second_max = dock.despike_exclusions_table.item(1, 1)
        assert first_min is not None and float(first_min.text()) == pytest.approx(495.0)
        assert first_max is not None and float(first_max.text()) == pytest.approx(505.0)
        assert second_min is not None and second_min.text() == ""
        assert second_max is not None and float(second_max.text()) == pytest.approx(420.0)
    finally:
        dock.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize(
    ("checkbox_attr", "widget_attrs", "expect_enabled_when_checked"),
    [
        ("smooth_enable", ["smooth_window", "smooth_poly"], True),
        ("despike_enable", ["despike_window", "despike_zscore"], True),
        (
            "despike_enable",
            ["despike_exclusions_table", "despike_exclusions_add_row"],
            True,
        ),
        ("despike_enable", ["despike_exclusions_remove_row"], False),
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

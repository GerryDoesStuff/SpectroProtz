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


def test_peak_detection_roundtrip(qt_app):
    dock = RecipeEditorDock()
    try:
        dock.peaks_enable.setChecked(True)
        dock.peaks_prominence.setValue(0.25)
        dock.peaks_min_absorbance_threshold.setValue(0.12)
        dock.peaks_noise_sigma_multiplier.setValue(1.8)
        dock.peaks_noise_window_cm.setValue(250.0)
        dock.peaks_min_prominence_by_region.setText("400-1500:0.03,1500-1800:0.02")
        dock.peaks_min_distance.setValue(1.5)
        dock.peaks_min_distance_mode.setCurrentIndex(1)
        dock.peaks_min_distance_fwhm_fraction.setValue(0.35)
        dock.peaks_peak_width_min.setValue(5.0)
        dock.peaks_peak_width_max.setValue(20.0)
        dock.peaks_max_peak_candidates.setValue(12)
        dock.peaks_detect_negative.setChecked(True)
        dock.peaks_merge_tolerance.setValue(9.0)
        dock.peaks_close_peak_tolerance.setValue(2.5)
        dock.peaks_plateau_min_points.setValue(4)
        dock.peaks_plateau_prominence_factor.setValue(1.4)
        dock.peaks_shoulder_min_distance.setValue(1.1)
        dock.peaks_shoulder_merge_tolerance.setValue(2.2)
        dock.peaks_shoulder_curvature_prominence_factor.setValue(1.3)
        dock.peaks_cwt_enabled.setChecked(True)
        dock.peaks_cwt_width_min.setValue(4.0)
        dock.peaks_cwt_width_max.setValue(50.0)
        dock.peaks_cwt_width_step.setValue(4.0)
        dock.peaks_cwt_cluster_tolerance.setValue(7.0)
        variable_fields = [
            {"name": "Operator", "value": "Avery"},
            {"name": "Batch", "value": "B-12"},
        ]
        dock._set_peaks_variable_fields(variable_fields)
        qt_app.processEvents()

        dock._update_model_from_ui(force=True)
        features = dock.recipe.params.get("features", {})
        peaks = features.get("peaks")
        assert isinstance(peaks, dict)
        assert peaks.get("enabled") is True
        assert peaks.get("prominence") == pytest.approx(0.25)
        assert peaks.get("min_absorbance_threshold") == pytest.approx(0.12)
        assert peaks.get("noise_sigma_multiplier") == pytest.approx(1.8)
        assert peaks.get("noise_window_cm") == pytest.approx(250.0)
        assert peaks.get("min_prominence_by_region") == "400-1500:0.03,1500-1800:0.02"
        assert peaks.get("min_distance") == pytest.approx(1.5)
        assert peaks.get("min_distance_mode") == "adaptive"
        assert peaks.get("min_distance_fwhm_fraction") == pytest.approx(0.35)
        assert peaks.get("peak_width_min") == pytest.approx(5.0)
        assert peaks.get("peak_width_max") == pytest.approx(20.0)
        assert peaks.get("max_peak_candidates") == 12
        assert peaks.get("detect_negative_peaks") is True
        assert peaks.get("merge_tolerance") == pytest.approx(9.0)
        assert peaks.get("close_peak_tolerance_cm") == pytest.approx(2.5)
        assert peaks.get("plateau_min_points") == 4
        assert peaks.get("plateau_prominence_factor") == pytest.approx(1.4)
        assert peaks.get("shoulder_min_distance_cm") == pytest.approx(1.1)
        assert peaks.get("shoulder_merge_tolerance_cm") == pytest.approx(2.2)
        assert peaks.get("shoulder_curvature_prominence_factor") == pytest.approx(1.3)
        assert peaks.get("cwt_enabled") is True
        assert peaks.get("cwt_width_min") == pytest.approx(4.0)
        assert peaks.get("cwt_width_max") == pytest.approx(50.0)
        assert peaks.get("cwt_width_step") == pytest.approx(4.0)
        assert peaks.get("cwt_cluster_tolerance") == pytest.approx(7.0)
        assert peaks.get("variable_fields") == variable_fields

        recipe_dict = dock.recipe_dict()
        dock.set_recipe(recipe_dict)
        qt_app.processEvents()
        assert dock.peaks_enable.isChecked() is True
        assert dock.peaks_prominence.value() == pytest.approx(0.25)
        assert dock.peaks_min_absorbance_threshold.value() == pytest.approx(0.12)
        assert dock.peaks_noise_sigma_multiplier.value() == pytest.approx(1.8)
        assert dock.peaks_noise_window_cm.value() == pytest.approx(250.0)
        assert (
            dock.peaks_min_prominence_by_region.text()
            == "400-1500:0.03,1500-1800:0.02"
        )
        assert dock.peaks_min_distance.value() == pytest.approx(1.5)
        assert dock.peaks_min_distance_mode.currentData() == "adaptive"
        assert dock.peaks_min_distance_fwhm_fraction.value() == pytest.approx(0.35)
        assert dock.peaks_peak_width_min.value() == pytest.approx(5.0)
        assert dock.peaks_peak_width_max.value() == pytest.approx(20.0)
        assert dock.peaks_max_peak_candidates.value() == 12
        assert dock.peaks_detect_negative.isChecked() is True
        assert dock.peaks_merge_tolerance.value() == pytest.approx(9.0)
        assert dock.peaks_close_peak_tolerance.value() == pytest.approx(2.5)
        assert dock.peaks_plateau_min_points.value() == 4
        assert dock.peaks_plateau_prominence_factor.value() == pytest.approx(1.4)
        assert dock.peaks_shoulder_min_distance.value() == pytest.approx(1.1)
        assert dock.peaks_shoulder_merge_tolerance.value() == pytest.approx(2.2)
        assert dock.peaks_shoulder_curvature_prominence_factor.value() == pytest.approx(1.3)
        assert dock.peaks_cwt_enabled.isChecked() is True
        assert dock.peaks_cwt_width_min.value() == pytest.approx(4.0)
        assert dock.peaks_cwt_width_max.value() == pytest.approx(50.0)
        assert dock.peaks_cwt_width_step.value() == pytest.approx(4.0)
        assert dock.peaks_cwt_cluster_tolerance.value() == pytest.approx(7.0)
        assert dock._peaks_variable_fields == variable_fields

        dock.peaks_enable.setChecked(False)
        qt_app.processEvents()
        dock._update_model_from_ui(force=True)
        features_after_disable = dock.recipe.params.get("features")
        assert not features_after_disable or "peaks" not in features_after_disable
    finally:
        dock.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize(
    ("checkbox_attr", "widget_attrs", "expect_enabled_when_checked"),
    [
        ("smooth_enable", ["smooth_window", "smooth_poly"], True),
        (
            "peaks_enable",
            [
                "peaks_prominence",
                "peaks_min_absorbance_threshold",
                "peaks_noise_sigma_multiplier",
                "peaks_noise_window_cm",
                "peaks_min_prominence_by_region",
                "peaks_min_distance",
                "peaks_min_distance_mode",
                "peaks_min_distance_fwhm_fraction",
                "peaks_peak_width_min",
                "peaks_peak_width_max",
                "peaks_max_peak_candidates",
                "peaks_detect_negative",
                "peaks_merge_tolerance",
                "peaks_close_peak_tolerance",
                "peaks_plateau_min_points",
                "peaks_plateau_prominence_factor",
                "peaks_shoulder_min_distance",
                "peaks_shoulder_merge_tolerance",
                "peaks_shoulder_curvature_prominence_factor",
                "peaks_cwt_enabled",
                "peaks_cwt_width_min",
                "peaks_cwt_width_max",
                "peaks_cwt_width_step",
                "peaks_cwt_cluster_tolerance",
                "peaks_variable_fields_button",
            ],
            True,
        ),
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

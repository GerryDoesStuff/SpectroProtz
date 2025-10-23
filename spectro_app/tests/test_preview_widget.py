import builtins
import html
import importlib
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from spectro_app.engine.plugin_api import Spectrum
import spectro_app.ui.docks.preview_widget as preview_widget


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_smoothed_fallback_renders_preview(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        spectrum = Spectrum(
            wavelength=np.array([400.0, 500.0, 600.0], dtype=float),
            intensity=np.array([0.2, 0.4, 0.1], dtype=float),
            meta={},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        smoothed_checkbox = widget._stage_controls["smoothed"]
        assert smoothed_checkbox.isEnabled()
        assert smoothed_checkbox.isChecked()

        smoothed_datasets = widget._stage_datasets["smoothed"]
        assert len(smoothed_datasets) == 1
        np.testing.assert_allclose(smoothed_datasets[0].x, spectrum.wavelength)
        np.testing.assert_allclose(smoothed_datasets[0].y, spectrum.intensity)

        assert widget._curve_items["smoothed"]
        title_text = widget.plot.plotItem.titleLabel.text
        assert "Stepwise spectra unavailable" not in title_text
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_stage_default_skips_smoothed_fallback_when_other_stage_available(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        wavelength = np.array([400.0, 500.0, 600.0], dtype=float)
        intensity = np.array([0.2, 0.5, 0.3], dtype=float)
        despiked = np.array([0.25, 0.55, 0.35], dtype=float)

        spectrum = Spectrum(
            wavelength=wavelength,
            intensity=intensity,
            meta={"channels": {"despiked": despiked}},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        despiked_checkbox = widget._stage_controls["despiked"]
        smoothed_checkbox = widget._stage_controls["smoothed"]

        assert despiked_checkbox.isEnabled()
        assert despiked_checkbox.isChecked()

        assert smoothed_checkbox.isEnabled()
        assert not smoothed_checkbox.isChecked()
        assert widget._smoothed_fallback_active
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_peak_markers_created_and_follow_visibility(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        wavelength = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)
        intensity = np.array([0.2, 0.6, 0.3, 0.1], dtype=float)
        peaks = [
            {"wavelength": 500.0, "intensity": 0.6},
            {"wavelength": 600.0, "intensity": 0.3},
        ]
        spectrum = Spectrum(
            wavelength=wavelength,
            intensity=intensity,
            meta={"features": {"peaks": peaks}, "sample_id": "Sample with peaks"},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        label = widget._sample_labels[0]
        assert label in widget._peak_items
        scatter = widget._peak_items[label]
        x_data, y_data = scatter.getData()
        np.testing.assert_allclose(np.array(x_data), np.array([500.0, 600.0]))
        np.testing.assert_allclose(np.array(y_data), np.array([0.6, 0.3]))
        assert scatter.isVisible()

        assert widget.peaks_button.isEnabled()
        assert widget.peaks_button.isChecked()

        widget.peaks_button.setChecked(False)
        qt_app.processEvents()
        assert not scatter.isVisible()

        widget.peaks_button.setChecked(True)
        qt_app.processEvents()
        assert scatter.isVisible()

        smoothed_checkbox = widget._stage_controls["smoothed"]
        smoothed_checkbox.setChecked(False)
        qt_app.processEvents()
        assert not scatter.isVisible()

        smoothed_checkbox.setChecked(True)
        qt_app.processEvents()
        assert scatter.isVisible()

        curve = widget._curve_items["smoothed"][0]
        widget._select_curve(curve)
        widget._hide_selected_spectrum()
        qt_app.processEvents()
        assert not scatter.isVisible()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_update_peak_items_visibility_respects_state(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        wavelength = np.array([405.0, 505.0, 605.0], dtype=float)
        intensity = np.array([0.25, 0.65, 0.35], dtype=float)
        peaks = [
            {"wavelength": 505.0, "intensity": 0.65},
            {"wavelength": 605.0, "intensity": 0.35},
        ]
        spectrum = Spectrum(
            wavelength=wavelength,
            intensity=intensity,
            meta={"features": {"peaks": peaks}, "sample_id": "Update Peaks"},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        label = widget._sample_labels[0]
        scatter = widget._peak_items[label]
        assert scatter.isVisible()

        widget._peaks_enabled = False
        widget._update_peak_items_visibility()
        assert not scatter.isVisible()

        widget._peaks_enabled = True
        widget._hidden_labels.add(label)
        widget._update_peak_items_visibility()
        assert not scatter.isVisible()

        widget._hidden_labels.clear()
        widget._update_peak_items_visibility()
        assert scatter.isVisible()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_peaks_toggle_preserved_with_preserve_view(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        wavelength = np.array([410.0, 510.0, 610.0], dtype=float)
        intensity = np.array([0.3, 0.7, 0.2], dtype=float)
        peaks = [
            {"wavelength": 510.0, "intensity": 0.7},
            {"wavelength": 610.0, "intensity": 0.2},
        ]
        spectrum = Spectrum(
            wavelength=wavelength,
            intensity=intensity,
            meta={"features": {"peaks": peaks}, "sample_id": "Preserve Peaks"},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        assert widget.peaks_button.isChecked()
        widget.peaks_button.setChecked(False)
        qt_app.processEvents()
        assert not widget.peaks_button.isChecked()

        first_label = widget._sample_labels[0]
        first_scatter = widget._peak_items[first_label]
        assert not first_scatter.isVisible()

        assert widget.set_spectra([spectrum], preserve_view=True)
        qt_app.processEvents()

        assert not widget.peaks_button.isChecked()
        refreshed_label = widget._sample_labels[0]
        refreshed_scatter = widget._peak_items[refreshed_label]
        assert not refreshed_scatter.isVisible()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_blank_role_spectra_are_excluded(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        wavelength = np.array([400.0, 500.0, 600.0], dtype=float)
        blank = Spectrum(
            wavelength=wavelength,
            intensity=np.array([0.1, 0.2, 0.3], dtype=float),
            meta={"role": "Blank", "sample_id": "Blank sample"},
        )
        sample = Spectrum(
            wavelength=wavelength,
            intensity=np.array([0.3, 0.4, 0.2], dtype=float),
            meta={"sample_id": "Real sample"},
        )

        assert widget.set_spectra([blank, sample])
        qt_app.processEvents()

        assert widget._total_spectra == 1
        assert widget._sample_labels == ["Real sample"]
        assert all(ds.label != "Blank sample" for ds in widget._all_stage_datasets["smoothed"])

        summary_text = widget.selection_label.text()
        assert "Blank sample" not in summary_text
        assert "Real sample" in summary_text or "Showing the only spectrum" in summary_text
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_preserve_view_keeps_ranges_and_selection(qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        base_wavelength = np.linspace(400.0, 800.0, 50, dtype=float)
        spectra = [
            Spectrum(
                wavelength=base_wavelength,
                intensity=np.sin(base_wavelength / 50.0 + idx).astype(float),
                meta={"sample_id": f"Sample {idx + 1}"},
            )
            for idx in range(3)
        ]

        assert widget.set_spectra(spectra)
        qt_app.processEvents()

        widget.view_mode_button.click()
        qt_app.processEvents()
        widget._on_add_clicked()
        qt_app.processEvents()
        widget._on_previous_clicked()
        qt_app.processEvents()

        view_box = widget.plot.plotItem.getViewBox()
        view_box.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        view_box.setRange(xRange=(500.0, 600.0), yRange=(-0.75, 0.75), padding=0)

        selected_before = widget._current_single_labels()
        view_before = tuple(tuple(axis) for axis in view_box.viewRange())

        assert widget.set_spectra(spectra, preserve_view=True)
        qt_app.processEvents()

        selected_after = widget._current_single_labels()
        view_after = tuple(tuple(axis) for axis in view_box.viewRange())

        assert selected_after == selected_before
        for before_axis, after_axis in zip(view_before, view_after):
            np.testing.assert_allclose(after_axis, before_axis)

        assert widget.set_spectra(spectra, preserve_view=False)
        qt_app.processEvents()
        view_reset = tuple(tuple(axis) for axis in view_box.viewRange())

        assert not all(
            np.allclose(reset_axis, after_axis)
            for reset_axis, after_axis in zip(view_reset, view_after)
        )
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_export_action_disabled_when_exporters_missing(monkeypatch, qt_app):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyqtgraph.exporters":
            raise ImportError("exporters unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("pyqtgraph.exporters", None)

    reloaded = importlib.reload(preview_widget)

    try:
        assert reloaded.pg is not None
        assert not reloaded._EXPORTERS_AVAILABLE

        widget = reloaded.SpectraPlotWidget()
        try:
            spectrum = Spectrum(
                wavelength=np.array([400.0, 500.0, 600.0], dtype=float),
                intensity=np.array([0.2, 0.4, 0.1], dtype=float),
                meta={},
            )

            assert widget.set_spectra([spectrum])
            qt_app.processEvents()

            assert widget._curve_items["smoothed"]

            captured: dict[str, bool] = {}

            def fake_exec(self, pos):
                actions = {action.text(): action for action in self.actions()}
                captured["export_enabled"] = actions["Export figure…"].isEnabled()
                return None

            monkeypatch.setattr(QtWidgets.QMenu, "exec", fake_exec)
            widget._show_context_menu(QtCore.QPoint(0, 0))
            assert captured.get("export_enabled") is False

            messages: list[str] = []

            monkeypatch.setattr(
                reloaded.SpectraPlotWidget,
                "_show_export_unavailable_message",
                lambda self: messages.append("export-unavailable"),
            )

            widget._export_figure()
            assert messages == ["export-unavailable"]
        finally:
            widget.deleteLater()
            qt_app.processEvents()
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import, raising=False)
        importlib.reload(preview_widget)


def test_legend_handles_missing_sample_pen(monkeypatch, qt_app):
    sample_cls = pg.graphicsItems.LegendItem.ItemSample
    monkeypatch.delattr(sample_cls, "setPen", raising=False)

    widget = preview_widget.SpectraPlotWidget()
    try:
        spectrum = Spectrum(
            wavelength=np.array([400.0, 500.0, 600.0], dtype=float),
            intensity=np.array([0.1, 0.3, 0.2], dtype=float),
            meta={"sample_id": "Legend test"},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        curve = next(iter(widget._curve_metadata))
        dataset, original_pen, legend_text = widget._curve_metadata[curve]
        entry = widget._legend_entries[curve]
        sample_item, label_item = entry
        assert not hasattr(sample_item, "setPen")

        widget._select_curve(curve)
        qt_app.processEvents()

        assert widget._selected_curve is curve
        assert widget._selection_color.name() in label_item.text
        assert curve.opts["pen"].color().name() == widget._selection_color.name()

        widget._hidden_labels.add(dataset.label)
        widget._apply_hidden_visibility()
        qt_app.processEvents()

        assert "(hidden)" in label_item.text
        assert "#7f7f7f" in label_item.text

        widget._clear_selection()
        qt_app.processEvents()

        assert widget._selected_curve is None
        assert html.escape(legend_text) in label_item.text
        assert curve.opts["pen"].color().name() == original_pen.color().name()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_click_selected_curve_clears_selection(monkeypatch, qt_app):
    widget = preview_widget.SpectraPlotWidget()
    try:
        spectrum = Spectrum(
            wavelength=np.array([400.0, 500.0, 600.0], dtype=float),
            intensity=np.array([0.1, 0.2, 0.3], dtype=float),
            meta={"sample_id": "Selectable"},
        )

        assert widget.set_spectra([spectrum])
        qt_app.processEvents()

        curve = next(iter(widget._curve_metadata))
        widget._select_curve(curve)
        assert widget._selected_curve is curve
        assert widget._selected_label == "Selectable"

        scene = widget.plot.scene()
        monkeypatch.setattr(scene, "items", lambda pos: [curve])

        class FakeEvent:
            def __init__(self) -> None:
                self.accepted = False

            def button(self) -> QtCore.Qt.MouseButton:
                return QtCore.Qt.MouseButton.LeftButton

            def scenePos(self) -> QtCore.QPointF:
                return QtCore.QPointF(0, 0)

            def accept(self) -> None:
                self.accepted = True

        event = FakeEvent()
        widget._on_mouse_clicked(event)
        qt_app.processEvents()

        assert event.accepted is True
        assert widget._selected_curve is None
        assert widget._selected_label is None
    finally:
        widget.deleteLater()
        qt_app.processEvents()

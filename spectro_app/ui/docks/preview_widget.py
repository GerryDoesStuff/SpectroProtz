from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget

from spectro_app.engine.plugin_api import BatchResult, Spectrum

try:  # QtSvg is an optional dependency on some systems
    from PyQt6.QtSvg import QSvgRenderer  # type: ignore
except ImportError:  # pragma: no cover - fallback when QtSvg is unavailable
    QSvgRenderer = None

try:  # PyQtGraph provides the interactive plotting canvas
    import pyqtgraph as pg
    import pyqtgraph.exporters  # noqa: F401  # ensure exporters namespace is initialised
except Exception:  # pragma: no cover - headless / optional dependency missing
    pg = None  # type: ignore


@dataclass
class _StageDataset:
    stage: str
    label: str
    x: np.ndarray
    y: np.ndarray


class SpectraPlotWidget(QtWidgets.QWidget):
    """Interactive preview plot with stage toggles and crosshair readouts."""

    STAGE_LABELS: Dict[str, str] = {
        "raw": "Raw",
        "joined": "Joined",
        "despiked": "Despiked",
        "blanked": "Blanked",
        "baseline_corrected": "Baseline",
        "smoothed": "Smoothed",
    }

    DEFAULT_STAGE_PRIORITY: Sequence[str] = (
        "smoothed",
        "despiked",
        "joined",
        "baseline_corrected",
        "blanked",
        "raw",
    )

    STAGE_STYLES: Dict[str, QtCore.Qt.PenStyle] = {
        "raw": QtCore.Qt.PenStyle.DotLine,
        "blanked": QtCore.Qt.PenStyle.DashLine,
        "baseline_corrected": QtCore.Qt.PenStyle.DashDotLine,
        "joined": QtCore.Qt.PenStyle.DashDotDotLine,
        "despiked": QtCore.Qt.PenStyle.SolidLine,
        "smoothed": QtCore.Qt.PenStyle.SolidLine,
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        if pg is None:  # pragma: no cover - only triggered if dependency missing
            raise RuntimeError("PyQtGraph is required for SpectraPlotWidget")

        self._stage_controls: Dict[str, QtWidgets.QCheckBox] = {}
        self._stage_datasets: Dict[str, List[_StageDataset]] = {}
        self._all_stage_datasets: Dict[str, List[_StageDataset]] = {}
        self._curve_items: Dict[str, List[pg.PlotDataItem]] = {}
        self._visible_stages: set[str] = set()
        self._color_map: Dict[str, QtGui.QColor] = {}
        self._legend: Optional[pg.LegendItem] = None
        self._last_cursor_result: Optional[tuple[_StageDataset, int, float, float]] = None
        self._last_cursor_pos: Optional[tuple[float, float]] = None

        self._sample_labels: List[str] = []
        self._single_mode: bool = False
        self._single_window_start: int = 0
        self._single_window_size: int = 1

        self._total_spectra: int = 0
        self._smoothed_fallback_active: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 4)
        controls_layout.setSpacing(4)

        stages_row = QtWidgets.QHBoxLayout()
        stages_row.setSpacing(8)

        for key in self.STAGE_LABELS:
            checkbox = QtWidgets.QCheckBox(self.STAGE_LABELS[key])
            checkbox.setEnabled(False)
            checkbox.toggled.connect(lambda checked, stage=key: self._on_stage_toggled(stage, checked))
            stages_row.addWidget(checkbox)
            self._stage_controls[key] = checkbox

        stages_row.addStretch(1)

        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("Reset view")
        self.reset_button.clicked.connect(self._reset_view)
        stages_row.addWidget(self.reset_button)

        controls_layout.addLayout(stages_row)

        navigation_row = QtWidgets.QHBoxLayout()
        navigation_row.setSpacing(8)

        self.view_mode_button = QtWidgets.QToolButton()
        self.view_mode_button.setCheckable(True)
        self.view_mode_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.view_mode_button.toggled.connect(self._on_view_mode_toggled)
        navigation_row.addWidget(self.view_mode_button)

        self.previous_button = QtWidgets.QToolButton()
        self.previous_button.setText("Previous")
        self.previous_button.clicked.connect(self._on_previous_clicked)
        navigation_row.addWidget(self.previous_button)

        self.next_button = QtWidgets.QToolButton()
        self.next_button.setText("Next")
        self.next_button.clicked.connect(self._on_next_clicked)
        navigation_row.addWidget(self.next_button)

        self.remove_button = QtWidgets.QToolButton()
        self.remove_button.setText("Remove spectrum")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        navigation_row.addWidget(self.remove_button)

        self.add_button = QtWidgets.QToolButton()
        self.add_button.setText("Add spectrum")
        self.add_button.clicked.connect(self._on_add_clicked)
        navigation_row.addWidget(self.add_button)

        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("Zoom in")
        self.zoom_in_button.clicked.connect(self._on_zoom_in_clicked)
        navigation_row.addWidget(self.zoom_in_button)

        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("Zoom out")
        self.zoom_out_button.clicked.connect(self._on_zoom_out_clicked)
        navigation_row.addWidget(self.zoom_out_button)

        navigation_row.addStretch(1)

        self.selection_label = QtWidgets.QLabel()
        navigation_row.addWidget(self.selection_label)

        controls_layout.addLayout(navigation_row)

        layout.addWidget(controls)

        # Plot widget and crosshair readout area
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setMenuEnabled(False)
        self.plot.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._show_context_menu)

        self._legend = self.plot.addLegend(offset=(10, 10))

        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(120, 120, 120), width=1))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(120, 120, 120), width=1))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)

        container_layout.addWidget(self.plot, stretch=1)

        self.cursor_label = QtWidgets.QLabel("λ: ––– | I: –––")
        self.cursor_label.setContentsMargins(12, 4, 12, 8)
        container_layout.addWidget(self.cursor_label)

        layout.addWidget(container, stretch=1)

        self._mouse_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        self._update_navigation_ui()

    # ------------------------------------------------------------------
    # Public API
    def set_spectra(self, spectra: Sequence[Spectrum], preserve_view: bool = False) -> bool:
        """Populate the plot with spectra and return ``True`` if data was shown."""

        preserved_view_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        preserved_auto_range: Optional[tuple[object, object]] = None
        preserved_visible_stages: Optional[Set[str]] = None
        preserved_single_mode: Optional[bool] = None
        preserved_window_start: Optional[int] = None
        preserved_window_size: Optional[int] = None
        if preserve_view:
            view_box = self.plot.plotItem.getViewBox()
            if view_box is not None:
                try:
                    x_range, y_range = view_box.viewRange()
                except Exception:
                    preserved_view_range = None
                else:
                    preserved_view_range = (tuple(x_range), tuple(y_range))
                try:
                    auto_range_state = view_box.state.get("autoRange", (True, True))
                except Exception:
                    preserved_auto_range = None
                else:
                    preserved_auto_range = (
                        auto_range_state[0] if len(auto_range_state) > 0 else True,
                        auto_range_state[1] if len(auto_range_state) > 1 else True,
                    )
            preserved_visible_stages = {
                stage
                for stage, checkbox in self._stage_controls.items()
                if checkbox.isChecked()
            }
            preserved_single_mode = self._single_mode if self._total_spectra > 0 else False
            preserved_window_start = self._single_window_start
            preserved_window_size = self._single_window_size

        stage_datasets: Dict[str, List[_StageDataset]] = {key: [] for key in self.STAGE_LABELS}
        self._stage_datasets = {key: [] for key in self.STAGE_LABELS}
        smoothed_fallbacks: List[_StageDataset] = []
        self._smoothed_fallback_active = False
        self._curve_items.clear()
        self._visible_stages.clear()
        self._color_map.clear()
        if self._legend is not None:
            try:
                self.plot.removeItem(self._legend)
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._legend = None
        self.plot.clear()
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.addItem(self._hline, ignoreBounds=True)

        any_data = False
        has_stepwise = False
        label_counts: Dict[str, int] = {}

        for index, spec in enumerate(spectra):
            x = np.asarray(getattr(spec, "wavelength", ()), dtype=float)
            y = np.asarray(getattr(spec, "intensity", ()), dtype=float)
            if x.size < 2 or y.size != x.size:
                continue

            meta = getattr(spec, "meta", {}) or {}
            channels = dict(meta.get("channels") or {})

            label = str(
                meta.get("sample_id")
                or meta.get("id")
                or meta.get("label")
                or meta.get("path")
                or f"Sample {index + 1}"
            )
            # Ensure labels unique for legend readability
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] > 1:
                label = f"{label} ({label_counts[label]})"

            for stage in self.STAGE_LABELS:
                values: Optional[np.ndarray] = None
                from_channels = False
                if stage in channels:
                    values = np.asarray(channels[stage], dtype=float)
                    from_channels = True
                elif stage == "smoothed":
                    values = y
                if values is None or values.shape != x.shape:
                    continue
                dataset = _StageDataset(stage=stage, label=label, x=x.copy(), y=values.copy())
                if from_channels:
                    stage_datasets.setdefault(stage, []).append(dataset)
                    any_data = True
                    if stage != "smoothed":
                        has_stepwise = True
                elif stage == "smoothed":
                    smoothed_fallbacks.append(dataset)
                    any_data = True

        if smoothed_fallbacks:
            stage_datasets.setdefault("smoothed", []).extend(smoothed_fallbacks)
            self._smoothed_fallback_active = True

        has_usable_stage_data = any(stage_datasets[stage] for stage in self.STAGE_LABELS)

        if not has_stepwise and not has_usable_stage_data:
            self._all_stage_datasets = {stage: [] for stage in self.STAGE_LABELS}
            self._sample_labels = []
            self._total_spectra = 0
            self._single_mode = False
            self._smoothed_fallback_active = False
            for checkbox in self._stage_controls.values():
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
                checkbox.blockSignals(False)
            self.cursor_label.setText("λ: ––– | I: –––")
            message = (
                "No spectra available"
                if not any_data
                else "Stepwise spectra unavailable; use exported overlays."
            )
            self.plot.setTitle(message)
            self.reset_button.setEnabled(False)
            self._update_navigation_ui()
            return False

        self._all_stage_datasets = {stage: list(datasets) for stage, datasets in stage_datasets.items()}
        self.reset_button.setEnabled(True)

        self._sample_labels = self._collect_labels(self._all_stage_datasets)
        self._total_spectra = len(self._sample_labels)

        self._color_map.clear()
        if self._sample_labels:
            hues = max(len(self._sample_labels), 8)
            for idx, label in enumerate(self._sample_labels):
                color = pg.intColor(idx, hues=hues, values=255)
                self._color_map[label] = pg.mkColor(color)

        if self._total_spectra == 0:
            self._single_window_start = 0
            self._single_window_size = 1
            self._single_mode = False
        else:
            if preserve_view and preserved_window_start is not None and preserved_window_size is not None:
                if preserved_single_mode:
                    self._single_window_start = preserved_window_start % self._total_spectra
                    self._single_window_size = min(
                        max(1, preserved_window_size), self._total_spectra
                    )
                    self._single_mode = True
                else:
                    self._single_window_start %= self._total_spectra
                    self._single_window_size = min(
                        max(1, self._single_window_size), self._total_spectra
                    )
                    self._single_mode = False
            else:
                self._single_window_start %= self._total_spectra
                self._single_window_size = min(
                    max(1, self._single_window_size), self._total_spectra
                )
                self._single_mode = self.view_mode_button.isChecked()

        if preserve_view and preserved_visible_stages is not None:
            self._visible_stages = set(preserved_visible_stages)

        self.plot.setTitle("")
        self._apply_view_mode(initial=not preserve_view)
        self.cursor_label.setText("λ: ––– | I: –––")

        if preserve_view and preserved_view_range is not None:
            view_box = self.plot.plotItem.getViewBox()
            if view_box is not None:
                auto_range_state = preserved_auto_range or (False, False)
                axis_map = (pg.ViewBox.XAxis, pg.ViewBox.YAxis)
                for index, axis in enumerate(axis_map):
                    state = auto_range_state[index] if index < len(auto_range_state) else True
                    enable = state is not False
                    try:
                        view_box.enableAutoRange(axis=axis, enable=enable)
                    except Exception:
                        continue
                x_range, y_range = preserved_view_range
                if len(auto_range_state) > 0 and auto_range_state[0] is False:
                    view_box.setXRange(*x_range, padding=0)
                if len(auto_range_state) > 1 and auto_range_state[1] is False:
                    view_box.setYRange(*y_range, padding=0)
                if (
                    (len(auto_range_state) == 0 or auto_range_state[0] is False)
                    and (len(auto_range_state) <= 1 or auto_range_state[1] is False)
                ):
                    view_box.setRange(xRange=x_range, yRange=y_range, padding=0)

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    def _choose_default_stage(self, available: Optional[Set[str]] = None) -> Optional[str]:
        if available is None:
            available = {stage for stage, datasets in self._stage_datasets.items() if datasets}
        candidates = set(available)
        if (
            self._smoothed_fallback_active
            and "smoothed" in candidates
            and any(stage != "smoothed" for stage in candidates)
        ):
            candidates.discard("smoothed")

        for stage in self.DEFAULT_STAGE_PRIORITY:
            if stage in candidates:
                return stage
        return next(iter(candidates), None)

    def _collect_labels(self, stage_datasets: Dict[str, List[_StageDataset]]) -> List[str]:
        labels: List[str] = []
        for datasets in stage_datasets.values():
            for dataset in datasets:
                if dataset.label not in labels:
                    labels.append(dataset.label)
        return labels

    def _current_single_labels(self) -> List[str]:
        if not self._sample_labels or self._total_spectra == 0:
            return []
        labels: List[str] = []
        total = self._total_spectra
        start = self._single_window_start % total
        for offset in range(min(self._single_window_size, total)):
            index = (start + offset) % total
            label = self._sample_labels[index]
            labels.append(label)
        return labels

    def _apply_view_mode(self, *, initial: bool = False) -> None:
        if self._total_spectra == 0:
            self._single_mode = False

        stage_datasets: Dict[str, List[_StageDataset]] = {stage: [] for stage in self.STAGE_LABELS}
        if self._single_mode and self._total_spectra > 0:
            selected_labels = set(self._current_single_labels())
            for stage in self.STAGE_LABELS:
                stage_datasets[stage] = [
                    dataset
                    for dataset in self._all_stage_datasets.get(stage, [])
                    if dataset.label in selected_labels
                ]
        else:
            for stage in self.STAGE_LABELS:
                stage_datasets[stage] = list(self._all_stage_datasets.get(stage, []))

        self._stage_datasets = stage_datasets
        self._rebuild_plot(initial=initial)
        self._update_navigation_ui()

    def _rebuild_plot(self, *, initial: bool) -> None:
        for curves in self._curve_items.values():
            for curve in curves:
                try:
                    self.plot.removeItem(curve)
                except Exception:
                    pass
        self._curve_items = {stage: [] for stage in self.STAGE_LABELS}

        self._update_stage_controls(initial=initial)
        self._render_curves()
        self._sync_visible_stages()
        self._update_cursor_label(None)
        if initial:
            self._reset_view()

    def _update_stage_controls(self, *, initial: bool) -> None:
        available = {stage for stage, datasets in self._stage_datasets.items() if datasets}
        desired_visible: Set[str] = set()
        default_stage = self._choose_default_stage(available) if initial else None

        for stage, checkbox in self._stage_controls.items():
            has_stage = stage in available
            checkbox.blockSignals(True)
            if not has_stage:
                checkbox.setChecked(False)
            elif initial:
                should_check = default_stage is not None and stage == default_stage
                checkbox.setChecked(should_check)
                if should_check:
                    desired_visible.add(stage)
            else:
                should_check = checkbox.isChecked() or stage in self._visible_stages
                should_check = should_check and has_stage
                checkbox.setChecked(should_check)
                if should_check:
                    desired_visible.add(stage)
            checkbox.setEnabled(has_stage)
            checkbox.blockSignals(False)

        if not desired_visible and available:
            fallback = self._choose_default_stage(available)
            if fallback:
                checkbox = self._stage_controls[fallback]
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                desired_visible.add(fallback)

        self._visible_stages = desired_visible

    def _sync_visible_stages(self) -> None:
        self._visible_stages = {
            stage
            for stage, checkbox in self._stage_controls.items()
            if checkbox.isEnabled() and checkbox.isChecked()
        }
        for stage, curves in self._curve_items.items():
            visible = stage in self._visible_stages
            for curve in curves:
                curve.setVisible(visible)

    def _update_navigation_ui(self) -> None:
        total = self._total_spectra
        is_single = self._single_mode and total > 0

        self.view_mode_button.blockSignals(True)
        self.view_mode_button.setChecked(is_single)
        self.view_mode_button.blockSignals(False)
        self.view_mode_button.setEnabled(total > 0)
        self.view_mode_button.setText("Show all spectra" if is_single else "Single spectrum view")

        self.previous_button.setEnabled(is_single and total > 1)
        self.next_button.setEnabled(is_single and total > 1)
        self.add_button.setEnabled(is_single and self._single_window_size < total)
        self.remove_button.setEnabled(is_single and self._single_window_size > 1)
        self.zoom_in_button.setEnabled(total > 0)
        self.zoom_out_button.setEnabled(total > 0)

        if is_single:
            labels = self._current_single_labels()
            if not labels:
                summary = f"Showing 0 of {total} spectra"
            elif len(labels) == 1:
                summary = f"Showing 1 of {total}: {labels[0]}"
            else:
                summary = f"Showing {len(labels)} of {total}: {', '.join(labels)}"
        else:
            if total == 0:
                summary = "No spectra available"
            elif total == 1:
                summary = "Showing the only spectrum"
            else:
                summary = f"Showing all {total} spectra"

        self.selection_label.setText(summary)

    def _on_view_mode_toggled(self, checked: bool) -> None:
        if self._total_spectra == 0:
            self.view_mode_button.blockSignals(True)
            self.view_mode_button.setChecked(False)
            self.view_mode_button.blockSignals(False)
            self._single_mode = False
            self._update_navigation_ui()
            return

        self._single_mode = checked
        if self._single_mode:
            self._single_window_size = min(max(1, self._single_window_size), self._total_spectra)
        self._apply_view_mode()

    def _on_previous_clicked(self) -> None:
        if not self._single_mode or self._total_spectra <= 1:
            return
        self._single_window_start = (self._single_window_start - 1) % self._total_spectra
        self._apply_view_mode()

    def _on_next_clicked(self) -> None:
        if not self._single_mode or self._total_spectra <= 1:
            return
        self._single_window_start = (self._single_window_start + 1) % self._total_spectra
        self._apply_view_mode()

    def _on_add_clicked(self) -> None:
        if not self._single_mode or self._total_spectra == 0:
            return
        if self._single_window_size >= self._total_spectra:
            return
        self._single_window_size += 1
        self._apply_view_mode()

    def _on_remove_clicked(self) -> None:
        if not self._single_mode or self._single_window_size <= 1:
            return
        self._single_window_size -= 1
        self._apply_view_mode()

    def _on_zoom_in_clicked(self) -> None:
        self._apply_zoom(0.8, self._last_cursor_pos)

    def _on_zoom_out_clicked(self) -> None:
        self._apply_zoom(1.25, self._last_cursor_pos)

    def _apply_zoom(self, factor: float, center: Optional[tuple[float, float]] = None) -> None:
        view_box = self.plot.plotItem.getViewBox()
        if view_box is None:
            return
        self.plot.plotItem.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        if center is not None:
            target_center = pg.Point(*center)
        elif self._last_cursor_pos is not None:
            target_center = pg.Point(*self._last_cursor_pos)
        else:
            rect = view_box.viewRect()
            center_qpoint = rect.center()
            target_center = pg.Point(center_qpoint.x(), center_qpoint.y())
        view_box.scaleBy((factor, factor), center=target_center)

    def _render_curves(self) -> None:
        if self._legend is not None:
            self._legend.clear()
        else:
            self._legend = self.plot.addLegend(offset=(10, 10))

        for stage, datasets in self._stage_datasets.items():
            stage_label = self.STAGE_LABELS.get(stage, stage)
            pen_style = self.STAGE_STYLES.get(stage, QtCore.Qt.PenStyle.SolidLine)
            self._curve_items.setdefault(stage, [])
            for dataset in datasets:
                color = self._color_map.get(dataset.label, pg.mkColor("#1f77b4"))
                pen = pg.mkPen(color=color, width=2, style=pen_style)
                curve = self.plot.plot(dataset.x, dataset.y, pen=pen, name=f"{dataset.label} · {stage_label}")
                curve.setVisible(self._stage_controls.get(stage, QtWidgets.QCheckBox()).isChecked())
                self._curve_items[stage].append(curve)

    def _reset_view(self) -> None:
        self.plot.plotItem.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot.plotItem.autoRange()

    def _on_stage_toggled(self, stage: str, checked: bool) -> None:
        if checked:
            self._visible_stages.add(stage)
        else:
            self._visible_stages.discard(stage)
        for curve in self._curve_items.get(stage, []):
            curve.setVisible(checked)
        self._update_cursor_label(None)

    def _on_mouse_moved(self, event: Iterable[object]) -> None:
        if not event:
            return
        pos = event[0]
        if not isinstance(pos, QtCore.QPointF):
            return
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot.plotItem.vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        self._last_cursor_pos = (x, y)
        self._vline.setPos(x)
        self._hline.setPos(y)
        self._update_cursor_label((x, y))

    def _update_cursor_label(self, coords: Optional[tuple[float, float]]) -> None:
        if coords is None or not self._visible_stages:
            self._last_cursor_result = None
            self.cursor_label.setText("λ: ––– | I: –––")
            return

        x, y = coords
        nearest = self._find_nearest_dataset(x)
        if nearest is None:
            self._last_cursor_result = None
            self.cursor_label.setText("λ: ––– | I: –––")
            return

        dataset, index, x_val, y_val = nearest
        stage_label = self.STAGE_LABELS.get(dataset.stage, dataset.stage)
        self._last_cursor_result = nearest
        self.cursor_label.setText(
            f"λ: {x_val:.4g} | I: {y:.4g} · {dataset.label} ({stage_label})"
        )

    def _find_nearest_dataset(
        self, target_x: float
    ) -> Optional[tuple[_StageDataset, int, float, float]]:
        best: Optional[tuple[_StageDataset, int, float, float]] = None
        best_distance = float("inf")
        for stage in self._visible_stages:
            for dataset in self._stage_datasets.get(stage, []):
                if dataset.x.size == 0:
                    continue
                distances = np.abs(dataset.x - target_x)
                index = int(distances.argmin())
                distance = float(distances[index])
                if distance < best_distance:
                    x_val = float(dataset.x[index])
                    y_val = float(dataset.y[index])
                    if not np.isfinite(x_val) or not np.isfinite(y_val):
                        continue
                    best = (dataset, index, x_val, y_val)
                    best_distance = distance
        return best

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("Copy data at cursor")
        export_action = menu.addAction("Export figure…")
        reset_action = menu.addAction("Reset view")
        menu.addSeparator()
        zoom_in_action = menu.addAction("Zoom in here")
        zoom_out_action = menu.addAction("Zoom out here")
        action = menu.exec(self.plot.mapToGlobal(pos))
        if action == copy_action:
            self._copy_data_at_cursor()
        elif action == export_action:
            self._export_figure()
        elif action == reset_action:
            self._reset_view()
        elif action in {zoom_in_action, zoom_out_action}:
            view_box = self.plot.plotItem.getViewBox()
            if view_box is None:
                return
            scene_pos = self.plot.mapToScene(pos)
            view_pos = view_box.mapSceneToView(scene_pos)
            center = (float(view_pos.x()), float(view_pos.y()))
            if action == zoom_in_action:
                self._apply_zoom(0.8, center)
            else:
                self._apply_zoom(1.25, center)

    def _copy_data_at_cursor(self) -> None:
        if not self._last_cursor_result:
            return
        dataset, index, x_val, y_val = self._last_cursor_result
        stage_label = self.STAGE_LABELS.get(dataset.stage, dataset.stage)
        text = f"{x_val:.6g}\t{y_val:.6g}\t{dataset.label}\t{stage_label}"
        QtWidgets.QApplication.clipboard().setText(text)

    def _export_figure(self) -> None:
        if pg is None:  # pragma: no cover - only when dependency missing
            return
        dialog = QtWidgets.QFileDialog(self)
        dialog.setWindowTitle("Export figure")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters([
            "PNG Image (*.png)",
            "Scalable Vector Graphics (*.svg)",
        ])
        if not dialog.exec():  # pragma: no cover - user cancelled
            return
        path = dialog.selectedFiles()[0]
        name_filter = dialog.selectedNameFilter()
        if not path:
            return
        if name_filter.endswith("*.svg") and not path.lower().endswith(".svg"):
            path += ".svg"
        elif name_filter.endswith("*.png") and not path.lower().endswith(".png"):
            path += ".png"
        if path.lower().endswith(".svg"):
            exporter = pg.exporters.SVGExporter(self.plot.plotItem)
        else:
            exporter = pg.exporters.ImageExporter(self.plot.plotItem)
        exporter.export(path)


class PreviewDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Preview", parent)
        self.setObjectName("PreviewDock")
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False)
        self.tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.setWidget(self.tabs)
        self._spectra_widget: Optional[SpectraPlotWidget] = None
        self.clear()

    # ------------------------------------------------------------------
    # Public API used by the main window
    def clear(self):
        """Reset the preview area to its empty state."""
        self.tabs.clear()
        self._add_message_tab("Preview", "Waiting for results…")

    def prepare_for_job(self) -> None:
        """Ensure the dock is ready for a new job without clearing previews.

        When new results are being generated we keep the previous preview
        visible so the user is not presented with placeholder text.  Only when
        no tabs are currently shown do we fall back to the empty-state message.
        """

        if self.tabs.count() == 0:
            self.clear()

    def show_error(self, message: str):
        self.tabs.clear()
        self._add_message_tab("Error", message)

    def show_batch_result(self, result: BatchResult):
        self.tabs.clear()

        spectra_widget: Optional[SpectraPlotWidget] = None
        shown = False
        if pg is not None and result.processed:
            preserve_view = self._spectra_widget is not None
            if self._spectra_widget is None:
                try:
                    self._spectra_widget = SpectraPlotWidget()
                except RuntimeError:
                    self._spectra_widget = None
            spectra_widget = self._spectra_widget
            if spectra_widget is not None:
                try:
                    shown = spectra_widget.set_spectra(
                        result.processed, preserve_view=preserve_view
                    )
                except RuntimeError:
                    spectra_widget.deleteLater()
                    spectra_widget = None
                    self._spectra_widget = None
                    shown = False
        if shown and spectra_widget is not None:
            self.tabs.addTab(spectra_widget, "Spectra")
        else:
            message = "Interactive plotting is unavailable for these results."
            if result.figures:
                message += " Showing generated figures instead."
            self._add_message_tab("Spectra", message)

        if result.figures:
            for idx, (name, data) in enumerate(result.figures.items(), start=1):
                title = name or f"Figure {idx}"
                pixmap = self._pixmap_from_bytes(data)
                if pixmap is None:
                    self._add_message_tab(title, "Unable to render figure data.")
                    continue
                self.tabs.addTab(self._build_pixmap_tab(pixmap), title)

        if self.tabs.count() == 0:
            self._add_message_tab("Preview", "No preview data was produced for this run.")

    # ------------------------------------------------------------------
    # Internal helpers
    def _build_pixmap_tab(self, pixmap: QtGui.QPixmap) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        layout.addWidget(label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _add_message_tab(self, title: str, message: str):
        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(label, title)

    def _pixmap_from_bytes(self, data: bytes) -> Optional[QtGui.QPixmap]:
        if not data:
            return None

        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(data):  # Handles PNG and other raster formats
            return pixmap

        if QSvgRenderer is None:
            return None

        renderer = QSvgRenderer(QtCore.QByteArray(data))
        if not renderer.isValid():
            return None

        size = renderer.defaultSize()
        if not size.isValid():
            size = QtCore.QSize(800, 600)

        image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()

        return QtGui.QPixmap.fromImage(image)

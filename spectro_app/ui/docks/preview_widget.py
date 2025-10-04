from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

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
        "blanked": "Blanked",
        "baseline_corrected": "Baseline",
        "joined": "Joined",
        "despiked": "Despiked",
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
        self._curve_items: Dict[str, List[pg.PlotDataItem]] = {}
        self._visible_stages: set[str] = set()
        self._color_map: Dict[str, QtGui.QColor] = {}
        self._legend: Optional[pg.LegendItem] = None
        self._last_cursor_result: Optional[tuple[_StageDataset, int, float, float]] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 4)
        controls_layout.setSpacing(8)

        for key in self.STAGE_LABELS:
            checkbox = QtWidgets.QCheckBox(self.STAGE_LABELS[key])
            checkbox.setEnabled(False)
            checkbox.toggled.connect(lambda checked, stage=key: self._on_stage_toggled(stage, checked))
            controls_layout.addWidget(checkbox)
            self._stage_controls[key] = checkbox

        controls_layout.addStretch(1)

        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("Reset view")
        self.reset_button.clicked.connect(self._reset_view)
        controls_layout.addWidget(self.reset_button)

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

    # ------------------------------------------------------------------
    # Public API
    def set_spectra(self, spectra: Sequence[Spectrum]) -> bool:
        """Populate the plot with spectra and return ``True`` if data was shown."""

        stage_datasets: Dict[str, List[_StageDataset]] = {key: [] for key in self.STAGE_LABELS}
        self._stage_datasets = {key: [] for key in self.STAGE_LABELS}
        smoothed_fallbacks: List[_StageDataset] = []
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

        if not any_data:
            for checkbox in self._stage_controls.values():
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
                checkbox.blockSignals(False)
            self.cursor_label.setText("λ: ––– | I: –––")
            self.plot.setTitle("No spectra available")
            self.reset_button.setEnabled(False)
            return False

        if not has_stepwise:
            for checkbox in self._stage_controls.values():
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
                checkbox.blockSignals(False)
            self.cursor_label.setText("λ: ––– | I: –––")
            self.plot.setTitle("Stepwise spectra unavailable; use exported overlays.")
            self.reset_button.setEnabled(False)
            return False

        if smoothed_fallbacks:
            stage_datasets.setdefault("smoothed", []).extend(smoothed_fallbacks)

        self._stage_datasets = stage_datasets
        self.reset_button.setEnabled(True)

        # Build a colour palette per sample label
        labels: List[str] = []
        for datasets in self._stage_datasets.values():
            for dataset in datasets:
                if dataset.label not in labels:
                    labels.append(dataset.label)
        hues = max(len(labels), 8)
        for idx, label in enumerate(labels):
            color = pg.intColor(idx, hues=hues, values=255)
            self._color_map[label] = pg.mkColor(color)

        for stage, checkbox in self._stage_controls.items():
            has_stage = bool(self._stage_datasets.get(stage))
            checkbox.blockSignals(True)
            checkbox.setEnabled(has_stage)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self._curve_items[stage] = []

        default_stage = self._choose_default_stage()
        if default_stage:
            self._stage_controls[default_stage].setChecked(True)
            self._visible_stages.add(default_stage)

        self.plot.setTitle("")
        self._render_curves()
        self._reset_view()
        self.cursor_label.setText("λ: ––– | I: –––")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    def _choose_default_stage(self) -> Optional[str]:
        available = {stage for stage, datasets in self._stage_datasets.items() if datasets}
        for stage in self.DEFAULT_STAGE_PRIORITY:
            if stage in available:
                return stage
        return next(iter(available), None)

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
        self._vline.setPos(x)
        self._hline.setPos(y)
        self._update_cursor_label((x, y))

    def _update_cursor_label(self, coords: Optional[tuple[float, float]]) -> None:
        if coords is None or not self._visible_stages:
            self._last_cursor_result = None
            self.cursor_label.setText("λ: ––– | I: –––")
            return

        x, _ = coords
        nearest = self._find_nearest_dataset(x)
        if nearest is None:
            self._last_cursor_result = None
            self.cursor_label.setText("λ: ––– | I: –––")
            return

        dataset, index, x_val, y_val = nearest
        stage_label = self.STAGE_LABELS.get(dataset.stage, dataset.stage)
        self._last_cursor_result = nearest
        self.cursor_label.setText(
            f"λ: {x_val:.4g} | I: {y_val:.4g} · {dataset.label} ({stage_label})"
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
        action = menu.exec(self.plot.mapToGlobal(pos))
        if action == copy_action:
            self._copy_data_at_cursor()
        elif action == export_action:
            self._export_figure()
        elif action == reset_action:
            self._reset_view()

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
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False)
        self.tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.setWidget(self.tabs)
        self.clear()

    # ------------------------------------------------------------------
    # Public API used by the main window
    def clear(self):
        """Reset the preview area to its empty state."""
        self.tabs.clear()
        self._add_message_tab("Preview", "Waiting for results…")

    def show_error(self, message: str):
        self.tabs.clear()
        self._add_message_tab("Error", message)

    def show_batch_result(self, result: BatchResult):
        self.tabs.clear()

        spectra_widget: Optional[SpectraPlotWidget] = None
        shown = False
        if pg is not None and result.processed:
            spectra_widget = SpectraPlotWidget()
            try:
                shown = spectra_widget.set_spectra(result.processed)
            except RuntimeError:
                spectra_widget.deleteLater()
                spectra_widget = None
                shown = False
        if shown and spectra_widget is not None:
            self.tabs.addTab(spectra_widget, "Spectra")
        else:
            if spectra_widget is not None:
                spectra_widget.deleteLater()
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

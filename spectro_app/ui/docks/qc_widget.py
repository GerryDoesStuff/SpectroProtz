from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget


@dataclass
class AggregatedQCSeries:
    numeric_series: Dict[str, Tuple[np.ndarray, np.ndarray]]
    hist_series: Dict[str, np.ndarray]
    flagged_rows: Set[int]
    row_tooltips: Dict[int, str]
    flagged_points: Dict[str, Tuple[np.ndarray, np.ndarray]]
    axis_label: str
    x_labels: List[str]
    x_positions: np.ndarray

    def has_numeric(self) -> bool:
        return any(series[1].size > 0 for series in self.numeric_series.values())


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result):
            return None
        return result
    if isinstance(value, np.generic):
        result = float(value)
        if math.isnan(result):
            return None
        return result
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        cleaned = stripped.replace(",", "").replace("%", "")
        while cleaned and cleaned[-1] not in "0123456789eE.-":
            cleaned = cleaned[:-1]
        while cleaned and cleaned[0] not in "+-0123456789.":
            cleaned = cleaned[1:]
        if not cleaned:
            return None
        try:
            result = float(cleaned)
        except ValueError:
            return None
        if math.isnan(result):
            return None
        return result
    return None


def _match_key_case_insensitive(row: Dict[str, Any], candidate: str) -> Optional[str]:
    candidate_lower = candidate.lower()
    for key in row.keys():
        if key.lower() == candidate_lower:
            return key
    return None


def _row_flag_reasons(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for key, value in row.items():
        key_lower = key.lower()
        if key_lower in {"flag", "flagged", "warning", "status", "alert"}:
            if isinstance(value, bool) and value:
                reasons.append(f"{key}: flagged")
            elif isinstance(value, (int, float)) and value:
                reasons.append(f"{key}: {value}")
            elif isinstance(value, str):
                text = value.strip()
                if text:
                    reasons.append(f"{key}: {text}")
            continue
        if isinstance(value, str):
            lowered = value.lower()
            if any(token in lowered for token in ("fail", "flag", "warn", "alert", "violation", "threshold")):
                reasons.append(f"{key}: {value}")
    return reasons


def aggregate_qc_metrics(rows: Iterable[Dict[str, Any]]) -> Optional[AggregatedQCSeries]:
    data = list(rows)
    if not data:
        return None

    label_candidates = ("spectrum", "sample", "id", "name", "filename", "file")
    axis_candidates = ("time", "timestamp", "elapsed", "scan", "index", "order", "sequence")

    first_row = data[0]
    label_key: Optional[str] = None
    for candidate in label_candidates:
        matched = _match_key_case_insensitive(first_row, candidate)
        if matched:
            label_key = matched
            break

    axis_key: Optional[str] = None
    for candidate in axis_candidates:
        matched = _match_key_case_insensitive(first_row, candidate)
        if matched:
            axis_key = matched
            break

    axis_label = "Sample #"
    axis_values_pre: List[Optional[float]] = []
    if axis_key:
        for row in data:
            axis_values_pre.append(_to_float(row.get(axis_key)))
        if not any(value is not None for value in axis_values_pre):
            axis_key = None
            axis_values_pre = []
        else:
            axis_label = axis_key.title()

    x_positions: List[float] = []
    x_labels: List[str] = []
    flagged_rows: Set[int] = set()
    row_tooltips: Dict[int, str] = {}
    numeric_entries: Dict[str, List[Dict[str, Any]]] = {}
    skip_keys = {"notes", "comment", "comments", "threshold", "limit"}
    if label_key:
        skip_keys.add(label_key.lower())
    if axis_key:
        skip_keys.add(axis_key.lower())

    for index, row in enumerate(data):
        axis_value: Optional[float] = None
        if axis_key and axis_values_pre:
            axis_value = axis_values_pre[index]
        if axis_value is None:
            axis_value = float(index)
        x_positions.append(axis_value)

        if label_key:
            label_value = row.get(label_key)
            x_labels.append(str(label_value) if label_value is not None else str(index + 1))
        else:
            x_labels.append(str(index + 1))

        reasons = _row_flag_reasons(row)
        if reasons:
            flagged_rows.add(index)
            row_tooltips[index] = "; ".join(reasons)

        for key, value in row.items():
            key_lower = key.lower()
            if key_lower in skip_keys:
                continue
            numeric_value = _to_float(value)
            if numeric_value is None:
                continue
            numeric_entries.setdefault(key, []).append(
                {
                    "row_index": index,
                    "x": axis_value,
                    "y": numeric_value,
                }
            )

    if not numeric_entries:
        return AggregatedQCSeries(
            numeric_series={},
            hist_series={},
            flagged_rows=flagged_rows,
            row_tooltips=row_tooltips,
            flagged_points={},
            axis_label=axis_label,
            x_labels=x_labels,
            x_positions=np.array(x_positions, dtype=float) if x_positions else np.array([]),
        )

    numeric_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hist_series: Dict[str, np.ndarray] = {}
    flagged_points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    histogram_keywords = ("rsd", "noise", "stdev", "std", "variance", "var")

    for key, entries in numeric_entries.items():
        if not entries:
            continue
        entries.sort(key=lambda item: item["row_index"])
        xs = np.array([item["x"] for item in entries], dtype=float)
        ys = np.array([item["y"] for item in entries], dtype=float)
        numeric_series[key] = (xs, ys)
        key_lower = key.lower()
        if any(token in key_lower for token in histogram_keywords):
            hist_series[key] = ys
        row_indices = np.array([item["row_index"] for item in entries], dtype=int)
        flagged_mask = np.array([idx in flagged_rows for idx in row_indices])
        if flagged_mask.any():
            flagged_points[key] = (xs[flagged_mask], ys[flagged_mask])

    return AggregatedQCSeries(
        numeric_series=numeric_series,
        hist_series=hist_series,
        flagged_rows=flagged_rows,
        row_tooltips=row_tooltips,
        flagged_points=flagged_points,
        axis_label=axis_label,
        x_labels=x_labels,
        x_positions=np.array(x_positions, dtype=float) if x_positions else np.array([]),
    )


class QCDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("QC Panel", parent)
        self.setObjectName("QCDock")
        self._qc_series: Optional[AggregatedQCSeries] = None
        self._desired_mode = "table"
        self._current_mode = "table"
        self._chart_message: Optional[str] = "Awaiting quality control results…"

        self.table = QtWidgets.QTableView(self)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.model = QtGui.QStandardItemModel(self.table)
        self.table.setModel(self.model)

        self.figure = Figure(figsize=(6, 4))
        self._supports_constrained_layout = bool(
            getattr(self.figure, "set_layout_engine", None)
        )
        self._uses_constrained_layout = False
        self._pending_constrained_layout_activation = False
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()
        self.canvas.installEventFilter(self)
        self.chart_toolbar = NavigationToolbar(self.canvas, self)

        self.table_container = QtWidgets.QWidget(self)
        table_layout = QtWidgets.QVBoxLayout(self.table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)

        self.chart_container = QtWidgets.QWidget(self)
        chart_layout = QtWidgets.QVBoxLayout(self.chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.chart_toolbar)
        chart_layout.addWidget(self.canvas)

        self.stacked = QtWidgets.QStackedWidget(self)
        self.stacked.addWidget(self.table_container)
        self.stacked.addWidget(self.chart_container)

        self.toolbar = QtWidgets.QToolBar(self)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setMovable(False)

        self.view_group = QtGui.QActionGroup(self.toolbar)
        self.view_group.setExclusive(True)

        self.table_action = self.toolbar.addAction("Table")
        self.table_action.setCheckable(True)
        self.table_action.setToolTip("Show QC results in table form")
        self.view_group.addAction(self.table_action)
        self.table_action.triggered.connect(lambda: self._set_view_mode("table"))

        self.chart_action = self.toolbar.addAction("Charts")
        self.chart_action.setCheckable(True)
        self.chart_action.setToolTip("Show QC metric trends and distributions")
        self.view_group.addAction(self.chart_action)
        self.chart_action.triggered.connect(lambda: self._set_view_mode("charts"))
        self.chart_action.setEnabled(False)

        container = QtWidgets.QWidget(self)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.toolbar)
        container_layout.addWidget(self.stacked)

        self.setWidget(container)
        self._apply_view_mode("table")
        self.clear()

    def clear(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["QC"])
        self.model.appendRow([QtGui.QStandardItem("Awaiting quality control results…")])
        self.table.resizeColumnsToContents()
        self._chart_message = "Awaiting quality control results…"
        self._set_series(None)

    def show_qc_table(self, rows: Iterable[Dict[str, object]]):
        data = list(rows)
        self.model.clear()

        if not data:
            self.model.setHorizontalHeaderLabels(["QC"])
            self.model.appendRow([QtGui.QStandardItem("No QC metrics were reported.")])
            self.table.resizeColumnsToContents()
            self._chart_message = "No QC metrics were reported."
            self._set_series(None)
            return

        headers = sorted({key for row in data for key in row.keys()})
        self.model.setColumnCount(len(headers))
        self.model.setHorizontalHeaderLabels(headers)

        series = aggregate_qc_metrics(data)
        flagged_rows = series.flagged_rows if series else set()
        row_tooltips = series.row_tooltips if series else {}

        for row_index, row in enumerate(data):
            items: List[QtGui.QStandardItem] = []
            for header in headers:
                value = row.get(header, "")
                item = QtGui.QStandardItem(str(value))
                item.setEditable(False)
                if row_index in row_tooltips:
                    item.setToolTip(row_tooltips[row_index])
                if row_index in flagged_rows:
                    item.setBackground(QtGui.QBrush(QtGui.QColor("#ffe6e6")))
                items.append(item)
            self.model.appendRow(items)

        self.table.resizeColumnsToContents()
        self._chart_message = None if series and series.has_numeric() else "No numeric QC metrics were reported."
        self._set_series(series)

    def _set_view_mode(self, mode: str):
        self._desired_mode = mode
        if mode == "charts" and not self._can_show_charts():
            mode = "table"
        self._apply_view_mode(mode)

    def _apply_view_mode(self, mode: str):
        if mode == "charts" and not self._can_show_charts():
            mode = "table"
        self._current_mode = mode
        if mode == "table":
            self.stacked.setCurrentWidget(self.table_container)
            self.table_action.setChecked(True)
            self.chart_action.setChecked(False)
            self._disable_constrained_layout()
        else:
            self.stacked.setCurrentWidget(self.chart_container)
            self.table_action.setChecked(False)
            self.chart_action.setChecked(True)
            self._ensure_constrained_layout_active()

    def _can_show_charts(self) -> bool:
        return bool(self._qc_series and self._qc_series.has_numeric())

    def _set_series(self, series: Optional[AggregatedQCSeries]):
        self._qc_series = series
        self._update_toolbar_state()
        self._update_chart_view()
        target_mode = self._desired_mode if self._can_show_charts() else "table"
        self._apply_view_mode(target_mode)

    def _update_toolbar_state(self):
        self.chart_action.setEnabled(self._can_show_charts())

    def _update_chart_view(self):
        self.figure.clear()
        if not self._can_show_charts():
            message = self._chart_message or "No numeric QC metrics were reported."
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            self.canvas.draw_idle()
            return

        series = self._qc_series
        assert series is not None

        has_hist = any(values.size > 0 for values in series.hist_series.values())
        has_trend = any(values[1].size > 0 for values in series.numeric_series.values())

        ax_hist = None
        ax_trend = None
        if has_hist and has_trend:
            ax_hist, ax_trend = self.figure.subplots(2, 1, sharex=False)
        elif has_hist:
            ax_hist = self.figure.add_subplot(111)
        else:
            ax_trend = self.figure.add_subplot(111)

        if ax_hist is not None:
            ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
            for label, values in series.hist_series.items():
                clean = values[~np.isnan(values)]
                if clean.size == 0:
                    continue
                bins = min(15, max(5, int(math.sqrt(clean.size))))
                ax_hist.hist(clean, bins=bins, alpha=0.6, label=label)
            ax_hist.set_title("Distribution of QC Metrics")
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Count")
            handles, labels = ax_hist.get_legend_handles_labels()
            if handles:
                ax_hist.legend(loc="best", frameon=False)

        if ax_trend is not None:
            ax_trend.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
            flagged_label_added = False
            for label, (xs, ys) in series.numeric_series.items():
                if ys.size == 0:
                    continue
                ax_trend.plot(xs, ys, marker="o", label=label)
                flagged = series.flagged_points.get(label)
                if flagged:
                    flag_x, flag_y = flagged
                    valid_mask = ~np.isnan(flag_y)
                    flag_x = flag_x[valid_mask]
                    flag_y = flag_y[valid_mask]
                    if flag_x.size:
                        ax_trend.scatter(
                            flag_x,
                            flag_y,
                            color="#d62728",
                            edgecolors="white",
                            linewidths=0.5,
                            s=60,
                            zorder=5,
                            label="Flagged" if not flagged_label_added else None,
                        )
                        flagged_label_added = True
            ax_trend.set_title("QC Metric Trends")
            ax_trend.set_xlabel(series.axis_label)
            ax_trend.set_ylabel("Value")
            if series.x_positions.size and len(series.x_labels) == series.x_positions.size:
                ax_trend.set_xticks(series.x_positions)
                ax_trend.set_xticklabels(series.x_labels, rotation=45, ha="right")
            handles, labels = ax_trend.get_legend_handles_labels()
            handles = [handle for handle, label in zip(handles, labels) if label]
            labels = [label for label in labels if label]
            if handles:
                ax_trend.legend(handles, labels, loc="best", frameon=False)

        if not self._uses_constrained_layout:
            # Ensure there is enough padding for tick labels when constrained
            # layout is unavailable on the current Matplotlib version.
            self.figure.subplots_adjust(bottom=0.18)
        self.canvas.draw_idle()

    def _ensure_constrained_layout_active(self):
        if not self._supports_constrained_layout or self._uses_constrained_layout:
            return
        if self.canvas.size().isEmpty():
            self._pending_constrained_layout_activation = True
            return
        try:
            self.figure.set_layout_engine("constrained")
        except AttributeError:
            self._supports_constrained_layout = False
            self._uses_constrained_layout = False
            self._pending_constrained_layout_activation = False
            return
        self._uses_constrained_layout = True
        self._pending_constrained_layout_activation = False
        self.canvas.draw_idle()

    def _disable_constrained_layout(self):
        self._pending_constrained_layout_activation = False
        if not self._supports_constrained_layout or not self._uses_constrained_layout:
            return
        try:
            self.figure.set_layout_engine(None)
        except AttributeError:
            self._supports_constrained_layout = False
            self._uses_constrained_layout = False
            return
        self._uses_constrained_layout = False
        self.canvas.draw_idle()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.canvas and event.type() == QtCore.QEvent.Type.Resize:
            if self._pending_constrained_layout_activation and not self.canvas.size().isEmpty():
                self._ensure_constrained_layout_active()
        return super().eventFilter(obj, event)

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from spectro_app.app_context import AppContext
from spectro_app.engine.ftir_index_schema import validate_ftir_index_schema
from spectro_app.engine.ftir_lookup import LookupCriteria, PeakCriterion, build_lookup_queries, parse_lookup_text


class FtirLookupWindow(QtWidgets.QDialog):
    """Reference lookup window for FTIR peak matching."""

    def __init__(self, appctx: AppContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("FTIR Reference Lookup")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(980, 640)

        self._appctx = appctx
        self._auto_peak_centers: List[float] = []
        self._auto_peak_label: Optional[str] = None
        self._auto_peak_axis_key: Optional[str] = None
        self._selected_spectrum_x: List[float] = []
        self._selected_spectrum_y: List[float] = []
        self._selected_spectrum_label: Optional[str] = None
        self._active_index_path: Optional[Path] = None
        self._selected_entries: List[LookupResultEntry] = []
        self._plot_legend: Optional[pg.LegendItem] = None

        self._index_path_edit = QtWidgets.QLineEdit()
        self._index_path_edit.setPlaceholderText("Select peaks.duckdb")
        self._index_path_edit.setToolTip("Path to the FTIR index DuckDB file.")

        index_button = QtWidgets.QToolButton()
        index_button.setText("...")
        index_button.setToolTip("Browse for a DuckDB index file")
        index_button.clicked.connect(self._on_pick_index_path)

        index_row = QtWidgets.QHBoxLayout()
        index_row.addWidget(self._index_path_edit, 1)
        index_row.addWidget(index_button)
        index_container = QtWidgets.QWidget()
        index_container.setLayout(index_row)

        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("e.g. 1720±5 1600±8 title:acetone")
        self._search_edit.setToolTip("Enter peak positions and metadata filters.")
        self._search_button = QtWidgets.QPushButton("Search")
        self._search_button.clicked.connect(self._on_manual_search)

        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(self._search_edit, 1)
        search_row.addWidget(self._search_button)

        self._auto_status_label = QtWidgets.QLabel("No preview peaks received yet.")
        self._auto_status_label.setWordWrap(True)

        self._tolerance_spin = QtWidgets.QDoubleSpinBox()
        self._tolerance_spin.setDecimals(2)
        self._tolerance_spin.setRange(0.01, 200.0)
        self._tolerance_spin.setSingleStep(0.5)
        self._tolerance_spin.setValue(5.0)
        self._tolerance_spin.setToolTip("Tolerance applied to preview peak centers (cm⁻¹).")

        self._apply_auto_button = QtWidgets.QPushButton("Apply preview peaks")
        self._apply_auto_button.setEnabled(False)
        self._apply_auto_button.clicked.connect(self._on_apply_auto_search)

        auto_row = QtWidgets.QHBoxLayout()
        auto_row.addWidget(QtWidgets.QLabel("Tolerance (cm⁻¹)"))
        auto_row.addWidget(self._tolerance_spin)
        auto_row.addStretch(1)
        auto_row.addWidget(self._apply_auto_button)

        auto_box = QtWidgets.QGroupBox("Auto-search from preview peaks")
        auto_layout = QtWidgets.QVBoxLayout(auto_box)
        auto_layout.addWidget(self._auto_status_label)
        auto_layout.addLayout(auto_row)

        self._status_label = QtWidgets.QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #444;")

        self._results_summary = QtWidgets.QLabel("Matches: 0")
        self._results_summary.setStyleSheet("font-weight: 600;")

        self._results_list = QtWidgets.QListWidget()
        self._results_list.setWordWrap(True)
        self._results_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._results_list.setToolTip("Reference spectra matching the active search criteria.")
        self._results_list.itemClicked.connect(self._on_result_item_clicked)
        self._results_list.itemSelectionChanged.connect(self._update_add_button_state)
        self._results_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._results_list.customContextMenuRequested.connect(self._show_results_context_menu)

        self._add_to_plot_button = QtWidgets.QToolButton()
        self._add_to_plot_button.setText("Add to plot →")
        self._add_to_plot_button.setEnabled(False)
        self._add_to_plot_button.setToolTip("Add selected lookup results to the plotting sidebar.")
        self._add_to_plot_button.clicked.connect(self._on_add_selected_results)

        self._results_page_label = QtWidgets.QLabel()
        self._results_page_label.setStyleSheet("color: #666;")
        self._prev_page_button = QtWidgets.QToolButton()
        self._prev_page_button.setText("◀")
        self._prev_page_button.setToolTip("Previous page")
        self._prev_page_button.clicked.connect(self._on_prev_page)
        self._next_page_button = QtWidgets.QToolButton()
        self._next_page_button.setText("▶")
        self._next_page_button.setToolTip("Next page")
        self._next_page_button.clicked.connect(self._on_next_page)

        pager_row = QtWidgets.QHBoxLayout()
        pager_row.addWidget(self._results_page_label, 1)
        pager_row.addWidget(self._prev_page_button)
        pager_row.addWidget(self._next_page_button)

        add_row = QtWidgets.QHBoxLayout()
        add_row.addStretch(1)
        add_row.addWidget(self._add_to_plot_button)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self._results_summary)
        left_layout.addWidget(self._results_list, 1)
        left_layout.addLayout(add_row)
        left_layout.addLayout(pager_row)
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_layout)

        right_layout = QtWidgets.QVBoxLayout()
        right_header = QtWidgets.QLabel("Selected references")
        right_header.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(right_header)

        self._selected_list = QtWidgets.QListWidget()
        self._selected_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._selected_list.setToolTip("References included in the comparison plot.")
        self._selected_list.itemSelectionChanged.connect(self._update_remove_button_state)
        self._selected_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._selected_list.customContextMenuRequested.connect(
            self._show_selected_context_menu
        )
        right_layout.addWidget(self._selected_list, 1)

        self._remove_from_plot_button = QtWidgets.QToolButton()
        self._remove_from_plot_button.setText("Remove selected")
        self._remove_from_plot_button.setEnabled(False)
        self._remove_from_plot_button.setToolTip("Remove selected references from the plot.")
        self._remove_from_plot_button.clicked.connect(self._on_remove_selected_references)
        right_layout.addWidget(self._remove_from_plot_button)
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_layout)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([280, 600])

        form = QtWidgets.QFormLayout()
        form.addRow("Index DB", index_container)
        form.addRow("Manual search", search_row)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(auto_box)
        layout.addWidget(self._status_label)
        self._plot_widget = pg.PlotWidget(background="w")
        self._plot_widget.setMinimumHeight(220)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "Wavenumber", units="cm⁻¹")
        self._plot_widget.setLabel("left", "Normalized intensity")
        self._plot_widget.setTitle("Selected reference peaks")
        layout.addWidget(self._plot_widget, 1)
        layout.addWidget(splitter, 1)

        self._result_entries: List[LookupResultEntry] = []
        self._page_size = 150
        self._current_page = 0
        self._last_selected_row: Optional[int] = None

        self._restore_index_path()

    def set_auto_search_peaks(self, payload: Dict[str, object]) -> None:
        peaks_raw = payload.get("peaks")
        label = payload.get("label")
        axis_key = payload.get("axis_key")
        spectrum_x_raw = payload.get("spectrum_x")
        spectrum_y_raw = payload.get("spectrum_y")

        peaks: List[float] = []
        if isinstance(peaks_raw, Iterable) and not isinstance(peaks_raw, (str, bytes)):
            for value in peaks_raw:
                try:
                    center = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(center):
                    continue
                peaks.append(center)

        axis_text = str(axis_key or "").strip().lower()
        self._auto_peak_centers = peaks
        self._auto_peak_label = str(label) if label is not None else None
        self._auto_peak_axis_key = axis_text or None
        self._selected_spectrum_x = []
        self._selected_spectrum_y = []
        self._selected_spectrum_label = None
        if (
            isinstance(spectrum_x_raw, Iterable)
            and isinstance(spectrum_y_raw, Iterable)
            and not isinstance(spectrum_x_raw, (str, bytes))
            and not isinstance(spectrum_y_raw, (str, bytes))
        ):
            for x_value, y_value in zip(spectrum_x_raw, spectrum_y_raw):
                try:
                    x_float = float(x_value)
                    y_float = float(y_value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(x_float) and math.isfinite(y_float)):
                    continue
                self._selected_spectrum_x.append(x_float)
                self._selected_spectrum_y.append(y_float)
        if self._selected_spectrum_x and self._selected_spectrum_y:
            self._selected_spectrum_label = self._auto_peak_label

        self._refresh_auto_status()
        self._refresh_plot()

    # ------------------------------------------------------------------
    def _restore_index_path(self) -> None:
        stored = self._appctx.settings.value("indexer/lastIndexPath", "", type=str)
        if stored:
            self._index_path_edit.setText(stored)

    def _on_pick_index_path(self) -> None:
        start = self._index_path_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select FTIR index database",
            start,
            "DuckDB (*.duckdb);;All Files (*)",
        )
        if path:
            self._index_path_edit.setText(path)
            self._appctx.settings.setValue("indexer/lastIndexPath", path)
            self._status_label.setText(f"Index path set to {path}.")

    def _on_manual_search(self) -> None:
        text = self._search_edit.text().strip()
        criteria = parse_lookup_text(text)
        self._run_lookup(criteria, source_label="Manual search")

    def _on_apply_auto_search(self) -> None:
        if not self._auto_peak_centers:
            self._status_label.setText("No preview peaks are available to search.")
            return
        if not self._is_wavenumber_axis(self._auto_peak_axis_key):
            self._status_label.setText(
                "Preview peaks are not in wavenumber units; auto-search only supports FTIR wavenumber peaks."
            )
            return

        tolerance = float(self._tolerance_spin.value())
        criteria = LookupCriteria(
            peaks=[PeakCriterion(center=center, tolerance=tolerance) for center in self._auto_peak_centers],
            filters={},
            errors=[],
        )
        self._run_lookup(criteria, source_label="Preview peak auto-search")

    def _refresh_auto_status(self) -> None:
        if not self._auto_peak_centers:
            self._auto_status_label.setText("No preview peaks received yet.")
            self._apply_auto_button.setEnabled(False)
            return

        count = len(self._auto_peak_centers)
        label = self._auto_peak_label or "Selected spectrum"
        axis_ok = self._is_wavenumber_axis(self._auto_peak_axis_key)
        axis_note = "" if axis_ok else " (non-wavenumber axis)"
        self._auto_status_label.setText(f"{label}: {count} peaks captured{axis_note}.")
        self._apply_auto_button.setEnabled(axis_ok)

    def _is_wavenumber_axis(self, axis_key: Optional[str]) -> bool:
        if not axis_key:
            return False
        normalized = axis_key.strip().lower()
        return normalized in {"wavenumber", "wavenumbers", "cm-1", "cm^-1", "cm⁻¹"}

    def _resolve_index_path(self) -> Optional[Path]:
        raw = self._index_path_edit.text().strip()
        if not raw:
            self._status_label.setText("Select an FTIR index database before searching.")
            return None
        path = Path(raw).expanduser()
        if not path.exists() or not path.is_file():
            self._status_label.setText(f"Index database not found: {path}")
            return None
        return path

    def _run_lookup(self, criteria: LookupCriteria, *, source_label: str) -> None:
        path = self._resolve_index_path()
        if path is None:
            return
        self._active_index_path = path
        errors = validate_ftir_index_schema(path)
        if errors:
            joined = "\n".join(errors)
            self._status_label.setText(f"Index schema errors:\n{joined}")
            return

        if criteria.is_empty:
            self._status_label.setText("Enter a search query or apply preview peaks to run a lookup.")
            return

        error_note = "; ".join(criteria.errors) if criteria.errors else ""

        queries = build_lookup_queries(criteria)
        entries: List[Dict[str, Any]] = []
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                result = con.execute(queries.spectra_sql, queries.spectra_params)
                columns = [col[0] for col in result.description]
                for row in result.fetchall():
                    entries.append(dict(zip(columns, row)))
                peak_counts = self._fetch_peak_counts(con, queries)
            finally:
                con.close()
        except Exception as exc:
            self._status_label.setText(f"Lookup failed: {exc}")
            return

        self._populate_results(entries, peak_counts)
        status = f"{source_label}: {len(entries)} matches"
        if error_note:
            status = f"{status} (warnings: {error_note})"
        self._status_label.setText(status)
        self._refresh_plot()

    def _fetch_peak_counts(self, con: duckdb.DuckDBPyConnection, queries: Any) -> Dict[str, int]:
        peak_counts: Dict[str, int] = {}
        grouped_sql = (
            "SELECT file_id, COUNT(*) AS matched_peaks "
            f"FROM ({queries.peaks_sql}) AS matches "
            "GROUP BY file_id"
        )
        result = con.execute(grouped_sql, queries.peaks_params)
        for file_id, count in result.fetchall():
            if file_id is None:
                continue
            peak_counts[str(file_id)] = int(count)
        return peak_counts

    def _populate_results(
        self,
        entries: List[Dict[str, Any]],
        peak_counts: Dict[str, int],
    ) -> None:
        self._result_entries = []
        for entry in entries:
            file_id = str(entry.get("file_id") or "").strip()
            self._result_entries.append(
                LookupResultEntry(
                    file_id=file_id,
                    spectrum_name=self._format_entry_name(entry),
                    formula=(entry.get("molform") or "").strip(),
                    matched_peaks=peak_counts.get(file_id, 0),
                    metadata=entry,
                )
            )
        self._current_page = 0
        self._last_selected_row = None
        self._render_results_page()

    def _render_results_page(self) -> None:
        self._results_list.clear()
        total = len(self._result_entries)
        if total == 0:
            self._results_summary.setText("Matches: 0")
            self._results_page_label.setText("No results")
            self._prev_page_button.setEnabled(False)
            self._next_page_button.setEnabled(False)
            self._update_add_button_state()
            return

        start = self._current_page * self._page_size
        end = min(start + self._page_size, total)
        for entry in self._result_entries[start:end]:
            label = self._format_entry_label(entry)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            self._results_list.addItem(item)

        self._results_summary.setText(f"Matches: {total}")
        self._results_page_label.setText(f"Showing {start + 1}-{end} of {total}")
        self._prev_page_button.setEnabled(self._current_page > 0)
        self._next_page_button.setEnabled(end < total)
        self._update_add_button_state()

    def _on_prev_page(self) -> None:
        if self._current_page <= 0:
            return
        self._current_page -= 1
        self._last_selected_row = None
        self._render_results_page()

    def _on_next_page(self) -> None:
        total = len(self._result_entries)
        if (self._current_page + 1) * self._page_size >= total:
            return
        self._current_page += 1
        self._last_selected_row = None
        self._render_results_page()

    def _on_result_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        row = self._results_list.row(item)
        if self._last_selected_row == row:
            self._results_list.clearSelection()
            self._results_list.setCurrentRow(-1)
            self._last_selected_row = None
            return
        self._last_selected_row = row

    def _show_results_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._results_list.itemAt(pos)
        if item is None:
            return
        entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(entry, LookupResultEntry):
            return
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction("Add to plot")
        action = menu.exec(self._results_list.mapToGlobal(pos))
        if action == add_action:
            self._add_entries_to_plot([entry])

    def _show_selected_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._selected_list.itemAt(pos)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        remove_action = menu.addAction("Remove from plot")
        action = menu.exec(self._selected_list.mapToGlobal(pos))
        if action == remove_action:
            self._remove_selected_references([item])

    def _update_add_button_state(self) -> None:
        if not hasattr(self, "_add_to_plot_button"):
            return
        self._add_to_plot_button.setEnabled(bool(self._results_list.selectedItems()))

    def _update_remove_button_state(self) -> None:
        if not hasattr(self, "_remove_from_plot_button"):
            return
        self._remove_from_plot_button.setEnabled(bool(self._selected_list.selectedItems()))

    def _on_add_selected_results(self) -> None:
        items = self._results_list.selectedItems()
        if not items:
            return
        entries: List[LookupResultEntry] = []
        for item in items:
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        self._add_entries_to_plot(entries)

    def _add_entries_to_plot(self, entries: List[LookupResultEntry]) -> None:
        if not entries:
            return
        existing_ids = {entry.file_id for entry in self._selected_entries}
        for entry in entries:
            if entry.file_id in existing_ids:
                continue
            item = QtWidgets.QListWidgetItem(self._format_entry_label(entry))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            self._selected_list.addItem(item)
            existing_ids.add(entry.file_id)
        self._sync_selected_entries()
        self._refresh_plot()

    def _on_remove_selected_references(self) -> None:
        items = self._selected_list.selectedItems()
        if not items:
            return
        self._remove_selected_references(items)

    def _remove_selected_references(
        self,
        items: List[QtWidgets.QListWidgetItem],
    ) -> None:
        for item in items:
            row = self._selected_list.row(item)
            if row >= 0:
                self._selected_list.takeItem(row)
        self._sync_selected_entries()
        self._refresh_plot()
        self._update_remove_button_state()

    def _sync_selected_entries(self) -> None:
        entries: List[LookupResultEntry] = []
        for idx in range(self._selected_list.count()):
            item = self._selected_list.item(idx)
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        self._selected_entries = entries

    def _refresh_plot(self) -> None:
        self._plot_widget.clear()
        if self._plot_legend is not None:
            self._plot_widget.removeItem(self._plot_legend)
        self._plot_legend = self._plot_widget.addLegend(offset=(10, 10))

        has_selected_spectrum = bool(self._selected_spectrum_x and self._selected_spectrum_y)
        if has_selected_spectrum:
            normalized_y = self._normalize_spectrum_intensities(self._selected_spectrum_y)
            label = self._selected_spectrum_label or "Selected spectrum"
            pen = pg.mkPen(color="#444444", width=2)
            self._plot_widget.plot(
                self._selected_spectrum_x,
                normalized_y,
                pen=pen,
                name=label,
            )

        if not self._selected_entries:
            title = "Selected reference peaks"
            if has_selected_spectrum:
                title = "Selected spectrum"
            self._plot_widget.setTitle(title)
            return
        if self._active_index_path is None:
            self._plot_widget.setTitle("Select an index database to plot peaks")
            return

        file_ids = [entry.file_id for entry in self._selected_entries if entry.file_id]
        if not file_ids:
            self._plot_widget.setTitle("Selected reference peaks")
            return

        peaks_by_file: Dict[str, List[tuple[float, float]]] = {}
        placeholders = ", ".join(["?"] * len(file_ids))
        sql = (
            "SELECT file_id, center, amplitude FROM peaks "
            f"WHERE file_id IN ({placeholders})"
        )
        try:
            con = duckdb.connect(str(self._active_index_path), read_only=True)
            try:
                rows = con.execute(sql, file_ids).fetchall()
            finally:
                con.close()
        except Exception as exc:
            self._status_label.setText(f"Plot refresh failed: {exc}")
            return

        for file_id, center, amplitude in rows:
            if file_id is None or center is None:
                continue
            try:
                center_value = float(center)
                amplitude_value = float(amplitude) if amplitude is not None else 1.0
            except (TypeError, ValueError):
                continue
            peaks_by_file.setdefault(str(file_id), []).append(
                (center_value, amplitude_value)
            )

        for idx, entry in enumerate(self._selected_entries):
            peaks = peaks_by_file.get(entry.file_id, [])
            if not peaks:
                continue
            normalized_heights = self._normalize_peak_heights(peaks)
            x_values: List[float] = []
            y_values: List[float] = []
            for (center, _amplitude), height in zip(peaks, normalized_heights):
                x_values.extend([center, center, float("nan")])
                y_values.extend([0.0, height, float("nan")])
            pen = pg.mkPen(color=pg.intColor(idx), width=2)
            self._plot_widget.plot(
                x_values,
                y_values,
                pen=pen,
                name=entry.spectrum_name,
            )

        if has_selected_spectrum:
            self._plot_widget.setTitle("Selected spectrum + reference peaks")
        else:
            self._plot_widget.setTitle("Selected reference peaks")

    def _normalize_spectrum_intensities(self, values: List[float]) -> List[float]:
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if not (math.isfinite(min_val) and math.isfinite(max_val)):
            return [0.0 for _ in values]
        span = max_val - min_val
        if span == 0:
            return [0.0 for _ in values]
        return [(value - min_val) / span for value in values]

    def _normalize_peak_heights(self, peaks: List[tuple[float, float]]) -> List[float]:
        if not peaks:
            return []
        max_amp = max(abs(amplitude) for _center, amplitude in peaks) or 1.0
        return [abs(amplitude) / max_amp for _center, amplitude in peaks]

    def _format_entry_name(self, entry: Dict[str, Any]) -> str:
        title = (entry.get("title") or "").strip()
        names = (entry.get("names") or "").strip()
        path = (entry.get("path") or "").strip()

        file_id = str(entry.get("file_id") or "").strip()
        fallback = f"File {file_id}".strip() if file_id else "File"
        return title or names or path or fallback

    def _format_entry_label(self, entry: "LookupResultEntry") -> str:
        formula = entry.formula or "—"
        peak_text = "peak" if entry.matched_peaks == 1 else "peaks"
        return (
            f"{entry.spectrum_name}\n"
            f"Formula: {formula} • Matched {entry.matched_peaks} {peak_text}"
        )


@dataclass(frozen=True)
class LookupResultEntry:
    file_id: str
    spectrum_name: str
    formula: str
    matched_peaks: int
    metadata: Dict[str, Any]

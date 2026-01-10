from __future__ import annotations

import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from scripts.jdxIndexBuilder import parse_jcamp_multispec
from spectro_app.app_context import AppContext
from spectro_app.engine.ftir_index_schema import validate_ftir_index_schema
from spectro_app.engine.ftir_lookup import LookupCriteria, PeakCriterion, build_lookup_queries, parse_lookup_text


@dataclass
class _SelectedSpectrum:
    label: Optional[str]
    x: List[float]
    y: List[float]


class FtirLookupWindow(QtWidgets.QDialog):
    """Reference lookup window for FTIR peak matching."""

    identified_overlay_requested = QtCore.pyqtSignal(list)
    _AUTO_SEARCH_TOLERANCE_CM = 2.0

    def __init__(self, appctx: AppContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("FTIR Reference Lookup")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(980, 640)

        self._appctx = appctx
        self._auto_peak_centers: List[float] = []
        self._auto_peak_label: Optional[str] = None
        self._auto_peak_axis_key: Optional[str] = None
        self._selected_spectra: List[_SelectedSpectrum] = []
        self._active_index_path: Optional[Path] = None
        self._selected_entries: List[LookupResultEntry] = []
        self._plot_legend: Optional[pg.LegendItem] = None
        self._reference_spectra_cache: "OrderedDict[str, tuple[List[float], List[float]]]" = OrderedDict()
        self._reference_spectra_cache_limit = 48
        self._reference_peaks_cache: "OrderedDict[str, List[tuple[float, float]]]" = OrderedDict()
        self._reference_peaks_cache_limit = 120
        self._reference_spectra_errors: Dict[str, str] = {}
        self._search_in_progress = False
        self._preview_entry: Optional[LookupResultEntry] = None
        self._pending_search_text: Optional[str] = None
        self._last_search_text: Optional[str] = None
        self._search_debounce_timer = QtCore.QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.setInterval(350)
        self._search_debounce_timer.timeout.connect(self._run_debounced_search)
        self._plot_refresh_timer = QtCore.QTimer(self)
        self._plot_refresh_timer.setSingleShot(True)
        self._plot_refresh_timer.setInterval(90)
        self._plot_refresh_timer.timeout.connect(self._refresh_plot)
        self._last_auto_signature: Optional[tuple[tuple[float, ...], float]] = None
        self._last_search_source: Optional[str] = None
        self._last_lookup_criteria: Optional[LookupCriteria] = None

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
        self._search_edit.textChanged.connect(self._on_search_text_changed)
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
        self._tolerance_spin.setValue(self._AUTO_SEARCH_TOLERANCE_CM)
        self._tolerance_spin.setReadOnly(True)
        self._tolerance_spin.setToolTip(
            "Auto-search tolerance applied to preview peak centers (cm⁻¹)."
        )

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

        self._export_button = QtWidgets.QToolButton()
        self._export_button.setText("Export CSV")
        self._export_button.setToolTip("Export lookup matches to CSV.")
        self._export_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        export_menu = QtWidgets.QMenu(self._export_button)
        self._export_all_action = export_menu.addAction("All matches")
        self._export_selected_action = export_menu.addAction("Selected references")
        self._export_all_action.triggered.connect(
            lambda: self._export_lookup_csv(scope="all")
        )
        self._export_selected_action.triggered.connect(
            lambda: self._export_lookup_csv(scope="selected")
        )
        self._export_button.setMenu(export_menu)
        self._export_button.setEnabled(False)

        self._results_list = QtWidgets.QListWidget()
        self._results_list.setWordWrap(True)
        self._results_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._results_list.setToolTip("Reference spectra matching the active search criteria.")
        self._results_list.itemClicked.connect(self._on_result_item_clicked)
        self._results_list.itemSelectionChanged.connect(self._on_results_selection_changed)
        self._results_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._results_list.customContextMenuRequested.connect(self._show_results_context_menu)

        self._add_to_plot_button = QtWidgets.QToolButton()
        self._add_to_plot_button.setText("Add →")
        self._add_to_plot_button.setEnabled(False)
        self._add_to_plot_button.setToolTip("Move selected lookup results into the plotting sidebar.")
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

        left_layout = QtWidgets.QVBoxLayout()
        summary_row = QtWidgets.QHBoxLayout()
        summary_row.addWidget(self._results_summary)
        summary_row.addStretch(1)
        summary_row.addWidget(self._export_button)
        left_layout.addLayout(summary_row)
        left_layout.addWidget(self._results_list, 1)
        left_layout.addLayout(pager_row)
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_layout)

        self._remove_from_plot_button = QtWidgets.QToolButton()
        self._remove_from_plot_button.setText("← Remove")
        self._remove_from_plot_button.setEnabled(False)
        self._remove_from_plot_button.setToolTip(
            "Remove selected references from the plotting sidebar."
        )
        self._remove_from_plot_button.clicked.connect(self._on_remove_selected_references)

        transfer_layout = QtWidgets.QVBoxLayout()
        transfer_layout.addStretch(1)
        transfer_layout.addWidget(self._add_to_plot_button)
        transfer_layout.addWidget(self._remove_from_plot_button)
        transfer_layout.addStretch(1)
        transfer_container = QtWidgets.QWidget()
        transfer_container.setLayout(transfer_layout)

        right_layout = QtWidgets.QVBoxLayout()
        right_header = QtWidgets.QLabel("Selected references")
        right_header.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(right_header)

        self._selected_list = QtWidgets.QListWidget()
        self._selected_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._selected_list.setToolTip("References included in the comparison plot.")
        self._selected_list.itemSelectionChanged.connect(self._on_selected_list_selection_changed)
        self._selected_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._selected_list.customContextMenuRequested.connect(
            self._show_selected_context_menu
        )
        right_layout.addWidget(self._selected_list, 1)
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_layout)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(transfer_container)
        splitter.addWidget(right_container)
        splitter.setSizes([280, 80, 600])

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
        self._metadata_label = QtWidgets.QLabel("No reference selected.")
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._metadata_label.setStyleSheet("color: #444;")
        metadata_box = QtWidgets.QGroupBox("Reference metadata")
        metadata_box.setMinimumWidth(240)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_box)
        metadata_layout.addWidget(self._metadata_label)

        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._plot_widget, 3)
        plot_layout.addWidget(metadata_box, 1)
        layout.addWidget(plot_container, 1)

        self._send_to_main_plot_button = QtWidgets.QPushButton()
        self._send_to_main_plot_button.setText("Send overlay to main preview")
        self._send_to_main_plot_button.setToolTip(
            "Overlay the current reference plot in the main preview window."
        )
        self._send_to_main_plot_button.setEnabled(False)
        self._send_to_main_plot_button.clicked.connect(self._on_send_to_main_plot)

        send_row = QtWidgets.QHBoxLayout()
        send_row.addStretch(1)
        send_row.addWidget(self._send_to_main_plot_button)
        layout.addLayout(send_row)

        layout.addWidget(splitter, 1)

        self._result_entries: List[LookupResultEntry] = []
        self._page_size = 150
        self._current_page = 0
        self._last_selected_row: Optional[int] = None
        self._max_results = 1000
        self._result_limit_hit = False

        self._restore_index_path()
        self._update_send_button_state()

    def set_auto_search_peaks(self, payload: Dict[str, object]) -> None:
        peaks_raw = payload.get("peaks")
        label = payload.get("label")
        axis_key = payload.get("axis_key")
        spectrum_x_raw = payload.get("spectrum_x")
        spectrum_y_raw = payload.get("spectrum_y")
        spectra_raw = payload.get("selected_spectra")

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
        self._selected_spectra = self._parse_selected_spectra(
            spectra_raw,
            fallback_label=self._auto_peak_label,
            spectrum_x_raw=spectrum_x_raw,
            spectrum_y_raw=spectrum_y_raw,
        )

        self._refresh_auto_status()
        self._request_plot_refresh()
        self._auto_search_from_preview()

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
        self._search_debounce_timer.stop()
        text = self._search_edit.text().strip()
        self._last_search_text = text or None
        criteria = parse_lookup_text(text)
        self._run_lookup(criteria, source_label="Manual search")

    def _on_apply_auto_search(self) -> None:
        self._auto_search_from_preview(force=True)

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
        try:
            with path.open("rb"):
                pass
        except OSError as exc:
            self._status_label.setText(
                f"Index database cannot be read: {path}\n{exc}"
            )
            return None
        return path

    def _run_lookup(self, criteria: LookupCriteria, *, source_label: str) -> None:
        path = self._resolve_index_path()
        if path is None:
            self._show_results_message(
                "Select a valid FTIR index database to run a lookup."
            )
            return
        if self._active_index_path != path:
            self._reference_spectra_cache.clear()
            self._reference_peaks_cache.clear()
            self._reference_spectra_errors.clear()
        self._active_index_path = path
        errors = validate_ftir_index_schema(path)
        if errors:
            joined = "\n".join(errors)
            self._status_label.setText(f"Index schema errors:\n{joined}")
            self._show_results_message(
                "The selected index is missing required tables or columns."
            )
            return

        if criteria.is_empty:
            self._status_label.setText("Enter a search query or apply preview peaks to run a lookup.")
            self._last_lookup_criteria = None
            return
        self._last_lookup_criteria = criteria

        error_note = "; ".join(criteria.errors) if criteria.errors else ""

        queries = build_lookup_queries(criteria)
        entries: List[Dict[str, Any]] = []
        limit = max(1, int(self._max_results))
        self._search_in_progress = True
        malformed_rows = 0
        try:
            try:
                con = duckdb.connect(str(path), read_only=True)
            except Exception as exc:
                self._status_label.setText(
                    f"Lookup failed to open index database: {path}\n{exc}"
                )
                return
            try:
                limited_sql = f"{queries.spectra_sql} LIMIT {limit + 1}"
                result = con.execute(limited_sql, queries.spectra_params)
                columns = [col[0] for col in result.description]
                for row in result.fetchall():
                    entry = dict(zip(columns, row))
                    file_id = entry.get("file_id")
                    if file_id is None or not str(file_id).strip():
                        malformed_rows += 1
                        continue
                    entries.append(entry)
                self._result_limit_hit = len(entries) > limit
                if self._result_limit_hit:
                    entries = entries[:limit]
                file_ids = [str(entry.get("file_id") or "").strip() for entry in entries]
                peak_stats = self._fetch_peak_stats(con, queries, file_ids=file_ids)
            finally:
                con.close()
        except Exception as exc:
            self._status_label.setText(f"Lookup failed: {exc}")
            return
        finally:
            self._search_in_progress = False

        self._populate_results(entries, peak_stats)
        self._last_search_source = source_label
        self._update_export_button_state()
        match_count = len(entries)
        if match_count == 0:
            status = f"{source_label}: no matches found."
        else:
            status = f"{source_label}: {match_count} matches"
            if self._result_limit_hit:
                status = f"{source_label}: showing first {match_count} matches (cap {limit})."
        if error_note:
            status = f"{status} (warnings: {error_note})"
        self._status_label.setText(status)
        if malformed_rows:
            self._append_status_warning(
                f"Skipped {malformed_rows} malformed spectra rows missing file IDs."
            )

    def _fetch_peak_stats(
        self,
        con: duckdb.DuckDBPyConnection,
        queries: Any,
        *,
        file_ids: List[str],
    ) -> Dict[str, "PeakMatchStats"]:
        peak_stats: Dict[str, PeakMatchStats] = {}
        grouped_sql = (
            "SELECT file_id, COUNT(*) AS matched_peaks, "
            "SUM(COALESCE(ABS(amplitude), 0) + COALESCE(ABS(area), 0)) AS match_score "
            f"FROM ({queries.peaks_sql}) AS matches "
            "GROUP BY file_id"
        )
        params = list(queries.peaks_params)
        if file_ids:
            placeholders = ", ".join(["?"] * len(file_ids))
            grouped_sql = f"{grouped_sql} HAVING file_id IN ({placeholders})"
            params.extend(file_ids)
        result = con.execute(grouped_sql, params)
        for file_id, count, match_score in result.fetchall():
            if file_id is None:
                continue
            peak_stats[str(file_id)] = PeakMatchStats(
                matched_peaks=int(count),
                match_score=float(match_score or 0.0),
            )
        return peak_stats

    def _populate_results(
        self,
        entries: List[Dict[str, Any]],
        peak_stats: Dict[str, "PeakMatchStats"],
    ) -> None:
        self._result_entries = []
        self._set_preview_entry(None, refresh=False)
        for entry in entries:
            file_id = str(entry.get("file_id") or "").strip()
            stats = peak_stats.get(file_id)
            self._result_entries.append(
                LookupResultEntry(
                    file_id=file_id,
                    spectrum_name=self._format_entry_name(entry),
                    formula=(entry.get("molform") or "").strip(),
                    matched_peaks=stats.matched_peaks if stats else 0,
                    match_score=stats.match_score if stats else 0.0,
                    metadata=entry,
                )
            )
        self._result_entries.sort(
            key=lambda entry: (
                -entry.match_score,
                -entry.matched_peaks,
                (entry.spectrum_name or "").lower(),
            )
        )
        self._current_page = 0
        self._last_selected_row = None
        self._render_results_page()

    def _render_results_page(self) -> None:
        self._results_list.clear()
        total = len(self._result_entries)
        if total == 0:
            self._show_results_message(
                "No matches found. Try adjusting the query or selecting a different index."
            )
            return

        start = self._current_page * self._page_size
        end = min(start + self._page_size, total)
        for entry in self._result_entries[start:end]:
            label = self._format_entry_label(entry)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            self._results_list.addItem(item)

        summary = f"Matches: {total}"
        if self._result_limit_hit:
            summary = f"Matches: {total} (capped)"
        self._results_summary.setText(summary)
        self._results_page_label.setText(f"Showing {start + 1}-{end} of {total}")
        self._prev_page_button.setEnabled(self._current_page > 0)
        self._next_page_button.setEnabled(end < total)
        self._update_add_button_state()
        self._update_export_button_state()

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
        self._set_preview_entry(entry)
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction("Add to plot")
        action = menu.exec(self._results_list.mapToGlobal(pos))
        if action == add_action:
            self._add_entries_to_plot([entry])

    def _show_selected_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._selected_list.itemAt(pos)
        if item is None:
            return
        entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(entry, LookupResultEntry):
            self._set_preview_entry(entry)
        menu = QtWidgets.QMenu(self)
        remove_action = menu.addAction("Remove from plot")
        action = menu.exec(self._selected_list.mapToGlobal(pos))
        if action == remove_action:
            self._remove_selected_references([item])

    def _update_add_button_state(self) -> None:
        if not hasattr(self, "_add_to_plot_button"):
            return
        self._add_to_plot_button.setEnabled(bool(self._results_list.selectedItems()))

    def _on_results_selection_changed(self) -> None:
        items = self._results_list.selectedItems()
        if not items:
            if self._preview_entry is not None:
                self._set_preview_entry(None)
            self._update_add_button_state()
            return
        entry = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(entry, LookupResultEntry):
            self._set_preview_entry(entry)
        self._update_add_button_state()

    def _update_remove_button_state(self) -> None:
        if not hasattr(self, "_remove_from_plot_button"):
            return
        self._remove_from_plot_button.setEnabled(bool(self._selected_list.selectedItems()))

    def _on_selected_list_selection_changed(self) -> None:
        self._update_remove_button_state()
        items = self._selected_list.selectedItems()
        if len(items) == 1:
            entry = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                self._set_preview_entry(entry)
                return
        self._set_preview_entry(None)

    def _on_add_selected_results(self) -> None:
        entries = self._selected_results_in_row_order()
        if not entries:
            return
        self._add_entries_to_plot(entries)

    def _selected_results_in_row_order(self) -> List[LookupResultEntry]:
        items = self._results_list.selectedItems()
        if not items:
            return []
        ordered_items = sorted(items, key=self._results_list.row)
        entries: List[LookupResultEntry] = []
        for item in ordered_items:
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        return entries

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
        self._request_plot_refresh(confirmed=True)

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
        self._request_plot_refresh(confirmed=True)
        self._update_remove_button_state()

    def _sync_selected_entries(self) -> None:
        entries: List[LookupResultEntry] = []
        for idx in range(self._selected_list.count()):
            item = self._selected_list.item(idx)
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        self._selected_entries = entries
        self._update_send_button_state()
        self._update_export_button_state()

    def _on_send_to_main_plot(self) -> None:
        overlays = self._collect_plot_overlays()
        if not overlays:
            self._status_label.setText("No reference spectra are available to send.")
            return
        self.identified_overlay_requested.emit(overlays)
        self._status_label.setText("Sent reference overlays to the main preview plot.")
        self._update_send_button_state()

    def _update_send_button_state(self) -> None:
        if not hasattr(self, "_send_to_main_plot_button"):
            return
        has_plot_entries = self._preview_entry is not None or bool(self._selected_entries)
        self._send_to_main_plot_button.setEnabled(has_plot_entries)
        self._update_export_button_state()

    def _collect_plot_overlays(self) -> List[Dict[str, object]]:
        preview_entry = self._preview_entry
        plot_entries = [preview_entry] if preview_entry is not None else self._selected_entries_for_plot()
        overlays: List[Dict[str, object]] = []
        for entry in plot_entries:
            reference_trace = self._load_reference_spectrum(entry)
            if reference_trace is None:
                continue
            x_vals, y_vals = reference_trace
            normalized_trace = self._normalize_spectrum_intensities(y_vals)
            if not normalized_trace or not x_vals:
                continue
            overlays.append(
                {
                    "label": entry.spectrum_name or entry.file_id or "Reference",
                    "x": x_vals,
                    "y": normalized_trace,
                }
            )
        return overlays

    def _request_plot_refresh(self, *, confirmed: bool = False) -> None:
        if self._search_in_progress and not confirmed:
            return
        if confirmed:
            self._plot_refresh_timer.start(0)
            return
        self._plot_refresh_timer.start()

    def _refresh_plot(self) -> None:
        self._plot_widget.clear()
        if self._plot_legend is not None:
            self._plot_widget.removeItem(self._plot_legend)
        self._plot_legend = self._plot_widget.addLegend(offset=(10, 10))

        has_selected_spectra = bool(self._selected_spectra)
        if has_selected_spectra:
            for idx, spectrum in enumerate(self._selected_spectra):
                normalized_y = self._normalize_spectrum_intensities(spectrum.y)
                label = spectrum.label or "Selected spectrum"
                pen = pg.mkPen(color=pg.intColor(idx), width=2)
                self._plot_widget.plot(
                    spectrum.x,
                    normalized_y,
                    pen=pen,
                    name=label,
                )

        preview_entry = self._preview_entry
        plot_entries = [preview_entry] if preview_entry is not None else self._selected_entries_for_plot()
        if not plot_entries:
            title = "Selected reference peaks"
            if has_selected_spectra:
                title = "Selected spectra"
            self._plot_widget.setTitle(title)
            self._update_metadata_panel(None, count=0)
            return
        if self._active_index_path is None:
            self._plot_widget.setTitle("Select an index database to plot peaks")
            self._update_metadata_panel(None, count=len(plot_entries))
            return

        file_ids = [entry.file_id for entry in plot_entries if entry.file_id]
        if not file_ids:
            self._plot_widget.setTitle("Selected reference peaks")
            self._update_metadata_panel(None, count=len(plot_entries))
            return

        peaks_by_file: Dict[str, List[tuple[float, float]]] = {}
        malformed_rows = 0
        missing_ids = [file_id for file_id in file_ids if file_id not in self._reference_peaks_cache]
        if missing_ids:
            placeholders = ", ".join(["?"] * len(missing_ids))
            sql = (
                "SELECT file_id, center, amplitude FROM peaks "
                f"WHERE file_id IN ({placeholders})"
            )
            try:
                con = duckdb.connect(str(self._active_index_path), read_only=True)
                try:
                    rows = con.execute(sql, missing_ids).fetchall()
                finally:
                    con.close()
            except Exception as exc:
                self._status_label.setText(f"Plot refresh failed: {exc}")
                return

            peaks_by_file = {file_id: [] for file_id in missing_ids}
            for file_id, center, amplitude in rows:
                if file_id is None or center is None or amplitude is None:
                    malformed_rows += 1
                    continue
                try:
                    center_value = float(center)
                    amplitude_value = float(amplitude)
                except (TypeError, ValueError):
                    malformed_rows += 1
                    continue
                if not (math.isfinite(center_value) and math.isfinite(amplitude_value)):
                    malformed_rows += 1
                    continue
                peaks_by_file.setdefault(str(file_id), []).append(
                    (center_value, amplitude_value)
                )
            for missing_id in missing_ids:
                self._cache_peak_data(missing_id, peaks_by_file.get(missing_id, []))

        if malformed_rows:
            self._append_status_warning(
                f"Skipped {malformed_rows} malformed peak rows while plotting."
            )

        for idx, entry in enumerate(plot_entries):
            reference_trace = self._load_reference_spectrum(entry)
            if reference_trace is not None:
                x_vals, y_vals = reference_trace
                normalized_trace = self._normalize_spectrum_intensities(y_vals)
                pen = pg.mkPen(color=pg.intColor(idx), width=2)
                self._plot_widget.plot(
                    x_vals,
                    normalized_trace,
                    pen=pen,
                    name=entry.spectrum_name,
                )
            peaks = self._get_cached_peaks(entry.file_id)
            if not peaks:
                continue
            normalized_heights = self._normalize_peak_heights(peaks)
            x_values: List[float] = []
            y_values: List[float] = []
            for (center, _amplitude), height in zip(peaks, normalized_heights):
                x_values.extend([center, center, float("nan")])
                y_values.extend([0.0, height, float("nan")])
            pen = pg.mkPen(color=pg.intColor(idx), width=1, style=QtCore.Qt.PenStyle.DashLine)
            self._plot_widget.plot(
                x_values,
                y_values,
                pen=pen,
            )

        if preview_entry is not None:
            self._plot_widget.setTitle("Reference preview")
            self._update_metadata_panel(preview_entry, count=1)
        else:
            if has_selected_spectra:
                self._plot_widget.setTitle("Selected spectra + reference traces")
            else:
                self._plot_widget.setTitle("Selected reference traces")
            if len(plot_entries) == 1:
                self._update_metadata_panel(plot_entries[0], count=1)
            else:
                self._update_metadata_panel(None, count=len(plot_entries))

    def _selected_entries_for_plot(self) -> List[LookupResultEntry]:
        if not self._selected_entries:
            return []
        selected_items = self._selected_list.selectedItems()
        if not selected_items:
            return list(self._selected_entries)
        entries: List[LookupResultEntry] = []
        for item in selected_items:
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        return entries

    def _load_reference_spectrum(
        self,
        entry: LookupResultEntry,
    ) -> Optional[tuple[List[float], List[float]]]:
        if not entry.file_id:
            return None
        cached = self._reference_spectra_cache.get(entry.file_id)
        if cached is not None:
            self._reference_spectra_cache.move_to_end(entry.file_id)
            return cached
        if entry.file_id in self._reference_spectra_errors:
            return None
        raw_path = entry.metadata.get("path")
        if not raw_path:
            self._reference_spectra_errors[entry.file_id] = "Missing path metadata."
            return None
        path = Path(str(raw_path)).expanduser()
        if not path.exists():
            self._reference_spectra_errors[entry.file_id] = f"Missing file: {path}"
            return None
        try:
            x_values, spectra, _headers = parse_jcamp_multispec(str(path))
        except Exception as exc:
            self._reference_spectra_errors[entry.file_id] = f"Failed to parse {path.name}: {exc}"
            return None
        if not spectra:
            self._reference_spectra_errors[entry.file_id] = f"No spectra in {path.name}"
            return None
        y_values = spectra[0]
        if len(x_values) == 0 or len(y_values) == 0:
            self._reference_spectra_errors[entry.file_id] = f"Empty spectrum in {path.name}"
            return None
        x_list = [float(value) for value in x_values.tolist()]
        y_list = [float(value) for value in y_values.tolist()]
        cached = (x_list, y_list)
        self._cache_reference_spectrum(entry.file_id, cached)
        return cached

    def _set_preview_entry(
        self,
        entry: Optional["LookupResultEntry"],
        *,
        refresh: bool = True,
    ) -> None:
        self._preview_entry = entry
        self._update_send_button_state()
        if refresh:
            self._request_plot_refresh(confirmed=True)

    def _update_metadata_panel(
        self,
        entry: Optional["LookupResultEntry"],
        *,
        count: int,
    ) -> None:
        if entry is None:
            if count > 1:
                self._metadata_label.setText(f"{count} references selected.")
            elif count == 1:
                self._metadata_label.setText("1 reference selected.")
            else:
                self._metadata_label.setText("No reference selected.")
            return

        meta = entry.metadata
        name = entry.spectrum_name or "—"
        formula = entry.formula or meta.get("molform") or "—"
        matched_peaks = entry.matched_peaks
        match_score = entry.match_score
        title = meta.get("title") or "—"
        names = meta.get("names") or "—"
        origin = meta.get("origin") or "—"
        cas = meta.get("cas") or "—"
        path = meta.get("path") or "—"
        file_id = entry.file_id or "—"
        match_score_text = self._format_match_score(match_score)

        def fmt(value: object) -> str:
            text = str(value).strip()
            return html.escape(text) if text else "—"

        summary = (
            f"<b>Name:</b> {fmt(name)}<br>"
            f"<b>Formula:</b> {fmt(formula)}<br>"
            f"<b>Matched peaks:</b> {fmt(matched_peaks)}<br>"
            f"<b>Match score:</b> {fmt(match_score_text)}<br>"
            f"<b>File ID:</b> {fmt(file_id)}<br>"
            f"<b>Title:</b> {fmt(title)}<br>"
            f"<b>Names:</b> {fmt(names)}<br>"
            f"<b>Origin:</b> {fmt(origin)}<br>"
            f"<b>CAS:</b> {fmt(cas)}<br>"
            f"<b>Path:</b> {fmt(path)}"
        )
        self._metadata_label.setText(summary)

    def _append_status_warning(self, warning: str) -> None:
        if not warning:
            return
        current = self._status_label.text().strip()
        warning_text = f"Warning: {warning}"
        if warning_text in current:
            return
        if current:
            self._status_label.setText(f"{current}\n{warning_text}")
        else:
            self._status_label.setText(warning_text)

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

    def _parse_selected_spectra(
        self,
        spectra_raw: object,
        *,
        fallback_label: Optional[str],
        spectrum_x_raw: object,
        spectrum_y_raw: object,
    ) -> List[_SelectedSpectrum]:
        parsed: List[_SelectedSpectrum] = []
        if isinstance(spectra_raw, Iterable) and not isinstance(spectra_raw, (str, bytes)):
            for entry in spectra_raw:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                x_raw = entry.get("x")
                y_raw = entry.get("y")
                spectrum = self._parse_selected_spectrum(label, x_raw, y_raw)
                if spectrum is not None:
                    parsed.append(spectrum)

        if parsed:
            return parsed

        spectrum = self._parse_selected_spectrum(fallback_label, spectrum_x_raw, spectrum_y_raw)
        return [spectrum] if spectrum is not None else []

    def _parse_selected_spectrum(
        self,
        label: object,
        x_raw: object,
        y_raw: object,
    ) -> Optional[_SelectedSpectrum]:
        if (
            not isinstance(x_raw, Iterable)
            or not isinstance(y_raw, Iterable)
            or isinstance(x_raw, (str, bytes))
            or isinstance(y_raw, (str, bytes))
        ):
            return None
        x_vals: List[float] = []
        y_vals: List[float] = []
        for x_value, y_value in zip(x_raw, y_raw):
            try:
                x_float = float(x_value)
                y_float = float(y_value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x_float) and math.isfinite(y_float)):
                continue
            x_vals.append(x_float)
            y_vals.append(y_float)
        if not x_vals or not y_vals:
            return None
        return _SelectedSpectrum(label=str(label) if label is not None else None, x=x_vals, y=y_vals)

    def _format_entry_name(self, entry: Dict[str, Any]) -> str:
        title = (entry.get("title") or "").strip()
        names = (entry.get("names") or "").strip()
        path = (entry.get("path") or "").strip()

        file_id = str(entry.get("file_id") or "").strip()
        fallback = f"File {file_id}".strip() if file_id else "File"
        return title or names or path or fallback

    def _format_entry_label(self, entry: "LookupResultEntry") -> str:
        formula = entry.formula or "—"
        cas = (entry.metadata.get("cas") or "").strip()
        peak_text = "peak" if entry.matched_peaks == 1 else "peaks"
        score_text = self._format_match_score(entry.match_score)
        summary_parts = [f"Formula: {formula}"]
        if cas:
            summary_parts.append(f"CAS: {cas}")
        summary_parts.append(f"Matches: {entry.matched_peaks} {peak_text}")
        summary_parts.append(f"Score: {score_text}")
        return (
            f"{entry.spectrum_name}\n"
            f"{' • '.join(summary_parts)}"
        )

    def _format_match_score(self, score: float) -> str:
        if not math.isfinite(score):
            return "—"
        return f"{score:.2f}"

    def _on_search_text_changed(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self._search_debounce_timer.stop()
            self._pending_search_text = None
            self._clear_results()
            return
        self._pending_search_text = cleaned
        self._search_debounce_timer.start()

    def _run_debounced_search(self) -> None:
        text = (self._pending_search_text or "").strip()
        if not text:
            return
        if text == self._last_search_text:
            return
        self._last_search_text = text
        criteria = parse_lookup_text(text)
        self._run_lookup(criteria, source_label="Manual search")

    def _clear_results(self) -> None:
        self._result_entries = []
        self._result_limit_hit = False
        self._current_page = 0
        self._last_selected_row = None
        self._last_search_text = None
        self._last_search_source = None
        self._last_lookup_criteria = None
        self._results_list.clear()
        self._results_summary.setText("Matches: 0")
        self._results_page_label.setText("No results")
        self._prev_page_button.setEnabled(False)
        self._next_page_button.setEnabled(False)
        self._status_label.setText("Enter a search query or apply preview peaks to run a lookup.")
        self._update_add_button_state()
        self._update_export_button_state()
        self._auto_search_from_preview()

    def _show_results_message(self, message: str) -> None:
        self._result_entries = []
        self._result_limit_hit = False
        self._current_page = 0
        self._last_selected_row = None
        self._results_list.clear()
        self._results_summary.setText("Matches: 0")
        self._results_page_label.setText("No results")
        self._prev_page_button.setEnabled(False)
        self._next_page_button.setEnabled(False)
        empty_item = QtWidgets.QListWidgetItem(message)
        empty_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        self._results_list.addItem(empty_item)
        self._update_add_button_state()
        self._update_export_button_state()

    def _cache_reference_spectrum(
        self,
        file_id: str,
        payload: tuple[List[float], List[float]],
    ) -> None:
        if not file_id:
            return
        self._reference_spectra_cache[file_id] = payload
        self._reference_spectra_cache.move_to_end(file_id)
        while len(self._reference_spectra_cache) > self._reference_spectra_cache_limit:
            self._reference_spectra_cache.popitem(last=False)

    def _cache_peak_data(self, file_id: str, peaks: List[tuple[float, float]]) -> None:
        if not file_id:
            return
        self._reference_peaks_cache[file_id] = peaks
        self._reference_peaks_cache.move_to_end(file_id)
        while len(self._reference_peaks_cache) > self._reference_peaks_cache_limit:
            self._reference_peaks_cache.popitem(last=False)

    def _get_cached_peaks(self, file_id: str) -> List[tuple[float, float]]:
        if not file_id:
            return []
        cached = self._reference_peaks_cache.get(file_id)
        if cached is None:
            return []
        self._reference_peaks_cache.move_to_end(file_id)
        return cached

    def _auto_search_from_preview(self, *, force: bool = False) -> None:
        if not self._auto_peak_centers:
            if force:
                self._status_label.setText("No preview peaks are available to search.")
            return
        if not self._is_wavenumber_axis(self._auto_peak_axis_key):
            if force:
                self._status_label.setText(
                    "Preview peaks are not in wavenumber units; auto-search only supports FTIR wavenumber peaks."
                )
            return
        if not force and self._search_edit.text().strip():
            return
        if not force and self._last_search_source == "Manual search":
            return
        tolerance = self._AUTO_SEARCH_TOLERANCE_CM
        signature = (
            tuple(round(center, 6) for center in self._auto_peak_centers),
            round(tolerance, 6),
        )
        if (
            not force
            and signature == self._last_auto_signature
            and self._last_search_source == "Preview peak auto-search"
        ):
            return
        criteria = LookupCriteria(
            peaks=[
                PeakCriterion(center=center, tolerance=tolerance)
                for center in self._auto_peak_centers
            ],
            filters={},
            errors=[],
        )
        self._run_lookup(criteria, source_label="Preview peak auto-search")
        self._last_auto_signature = signature

    def _update_export_button_state(self) -> None:
        if not hasattr(self, "_export_button"):
            return
        has_results = bool(self._result_entries)
        has_criteria = self._last_lookup_criteria is not None
        self._export_button.setEnabled(has_results and has_criteria)
        if hasattr(self, "_export_all_action"):
            self._export_all_action.setEnabled(has_results and has_criteria)
        has_selected = bool(self._selected_list.selectedItems())
        if hasattr(self, "_export_selected_action"):
            self._export_selected_action.setEnabled(has_selected and has_criteria)

    def _selected_entries_for_export(self) -> List["LookupResultEntry"]:
        items = self._selected_list.selectedItems()
        if not items:
            return []
        entries: List[LookupResultEntry] = []
        for item in items:
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry, LookupResultEntry):
                entries.append(entry)
        return entries

    def _export_lookup_csv(self, *, scope: str) -> None:
        criteria = self._last_lookup_criteria
        if criteria is None:
            self._status_label.setText("Run a lookup before exporting results.")
            return
        path = self._resolve_index_path()
        if path is None:
            return
        if not self._result_entries:
            self._status_label.setText("No lookup results are available to export.")
            return

        file_ids: List[str] = []
        if scope == "selected":
            selected_entries = self._selected_entries_for_export()
            if not selected_entries:
                self._status_label.setText(
                    "Select references in the right sidebar to export them."
                )
                return
            file_ids = [entry.file_id for entry in selected_entries if entry.file_id]
        else:
            file_ids = [entry.file_id for entry in self._result_entries if entry.file_id]

        if not file_ids:
            self._status_label.setText("No lookup results are available to export.")
            return

        default_name = "ftir_lookup_matches.csv" if scope == "all" else "ftir_lookup_selected.csv"
        target_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export lookup matches",
            str(Path.home() / default_name),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not target_path:
            return

        queries = build_lookup_queries(criteria)
        params = list(queries.peaks_params)
        placeholders = ", ".join(["?"] * len(file_ids))
        sql = (
            "SELECT matches.file_id, matches.spectrum_id, matches.peak_id, "
            "matches.center, matches.amplitude, matches.area, matches.fwhm, matches.r2, "
            "s.title, s.molform, s.cas "
            f"FROM ({queries.peaks_sql}) AS matches "
            "JOIN spectra s ON s.file_id = matches.file_id "
            f"WHERE matches.file_id IN ({placeholders}) "
            "ORDER BY matches.file_id, matches.peak_id"
        )
        params.extend(file_ids)

        rows: List[Dict[str, object]] = []
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                result = con.execute(sql, params)
                columns = [col[0] for col in result.description]
                for row in result.fetchall():
                    rows.append(dict(zip(columns, row)))
            finally:
                con.close()
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            return

        fieldnames = [
            "file_id",
            "spectrum_id",
            "peak_id",
            "center",
            "amplitude",
            "area",
            "fwhm",
            "r2",
            "title",
            "molform",
            "cas",
        ]
        try:
            with open(target_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
        except OSError as exc:
            self._status_label.setText(f"Export failed: {exc}")
            return

        result_note = f"{len(rows)} peak rows"
        if scope == "selected":
            scope_note = f"{len(file_ids)} selected references"
        else:
            scope_note = f"{len(file_ids)} matches"
        status = f"Exported {result_note} for {scope_note} to {target_path}."
        if scope == "all" and self._result_limit_hit:
            status = (
                f"{status} Matches list is capped; run a narrower search to export all hits."
            )
        self._status_label.setText(status)


@dataclass(frozen=True)
class LookupResultEntry:
    file_id: str
    spectrum_name: str
    formula: str
    matched_peaks: int
    match_score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PeakMatchStats:
    matched_peaks: int
    match_score: float

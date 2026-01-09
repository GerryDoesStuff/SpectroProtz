from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
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
        self._results_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._results_list.setToolTip("Reference spectra matching the active search criteria.")

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self._results_summary)
        left_layout.addWidget(self._results_list, 1)
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_layout)

        placeholder = QtWidgets.QLabel(
            "Select a reference spectrum to view details.\n"
            "(Detailed view and plotting are handled elsewhere.)"
        )
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet("color: #777;")

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(placeholder, 1)
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
        layout.addWidget(splitter, 1)

        self._restore_index_path()

    def set_auto_search_peaks(self, payload: Dict[str, object]) -> None:
        peaks_raw = payload.get("peaks")
        label = payload.get("label")
        axis_key = payload.get("axis_key")

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

        self._refresh_auto_status()

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
            finally:
                con.close()
        except Exception as exc:
            self._status_label.setText(f"Lookup failed: {exc}")
            return

        self._populate_results(entries)
        status = f"{source_label}: {len(entries)} matches"
        if error_note:
            status = f"{status} (warnings: {error_note})"
        self._status_label.setText(status)

    def _populate_results(self, entries: List[Dict[str, Any]]) -> None:
        self._results_list.clear()
        for entry in entries:
            label = self._format_entry_label(entry)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            self._results_list.addItem(item)
        self._results_summary.setText(f"Matches: {len(entries)}")

    def _format_entry_label(self, entry: Dict[str, Any]) -> str:
        title = (entry.get("title") or "").strip()
        names = (entry.get("names") or "").strip()
        path = (entry.get("path") or "").strip()
        cas = (entry.get("cas") or "").strip()
        molform = (entry.get("molform") or "").strip()

        file_id = str(entry.get("file_id") or "").strip()
        fallback = f"File {file_id}".strip() if file_id else "File"
        main_label = title or names or path or fallback
        details = ""
        parts = [part for part in [cas or None, molform or None] if part]
        if parts:
            details = f" ({', '.join(parts)})"
        return f"{main_label}{details}"

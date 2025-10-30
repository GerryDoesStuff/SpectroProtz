from __future__ import annotations

from pathlib import Path
import ast
import traceback
from types import SimpleNamespace
from typing import Dict, Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scripts.jdxIndexBuilder import (
    build_file_consensus,
    build_global_consensus,
    index_file,
    init_db,
    persist_consensus,
)


def _load_cli_defaults() -> Dict[str, Any]:
    """Extract the jdxIndexBuilder CLI defaults via AST parsing.

    This keeps the GUI in sync without importing heavy dependencies that the
    script requires (e.g. duckdb).
    """

    module_path = Path(__file__).resolve().parents[2] / "scripts" / "jdxIndexBuilder.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    parser_name: str | None = None
    defaults: Dict[str, Any] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Name)
                            and isinstance(stmt.value, ast.Call)
                            and isinstance(stmt.value.func, ast.Attribute)
                            and isinstance(stmt.value.func.value, ast.Name)
                            and stmt.value.func.attr == "ArgumentParser"
                        ):
                            parser_name = target.id
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.attr == "add_argument"
                    and parser_name
                    and stmt.value.func.value.id == parser_name
                ):
                    call = stmt.value
                    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
                    dest: str | None = None
                    dest_node = keywords.get("dest")
                    if isinstance(dest_node, ast.Constant) and isinstance(dest_node.value, str):
                        dest = dest_node.value
                    if dest is None:
                        for arg in call.args:
                            if (
                                isinstance(arg, ast.Constant)
                                and isinstance(arg.value, str)
                                and arg.value.startswith("--")
                            ):
                                dest = arg.value.lstrip("-").replace("-", "_")
                                break
                    default_node = keywords.get("default")
                    if dest and default_node is not None:
                        try:
                            defaults[dest] = ast.literal_eval(default_node)
                        except Exception:
                            pass
                    elif dest and keywords.get("action") and dest not in defaults:
                        action = keywords["action"]
                        if isinstance(action, ast.Constant) and action.value == "store_true":
                            defaults[dest] = False
            break
    return defaults


class IndexerWorker(QObject):
    """Background worker that wraps the jdxIndexBuilder script functions."""

    progress_update = pyqtSignal(int, int)
    stats_update = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str, str)
    cancelled = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, params: Dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params = params
        self._cancelled = False

    def request_cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        data_dir = Path(self._params["data_dir"]).resolve()
        index_dir = Path(self._params["index_dir"]).resolve()
        peaks_path = (index_dir / "peaks.duckdb").resolve()

        con = None
        try:
            files = sorted(
                p
                for p in data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jdx", ".dx"}
            )
            total_files = len(files)
            self.progress_update.emit(0, total_files)
            self.stats_update.emit(0, 0)
            if total_files:
                self.log_message.emit(
                    f"Discovered {total_files} JCAMP-DX files under {data_dir}."
                )
            else:
                self.log_message.emit(
                    f"No JCAMP-DX files were found under {data_dir}."
                )

            args = SimpleNamespace(**self._params)
            if not hasattr(args, "model") or args.model is None:
                args.model = "Gaussian"

            con = init_db(str(index_dir))

            total_specs = 0
            total_peaks = 0
            processed_files = 0

            for path in files:
                if self._cancelled:
                    break
                processed_files += 1
                self.log_message.emit(f"Indexing {path}...")
                n_specs, n_peaks = index_file(str(path), con, args)
                total_specs += n_specs
                total_peaks += n_peaks
                self.log_message.emit(
                    f"Processed {path.name}: spectra={n_specs}, peaks={n_peaks}"
                )
                self.progress_update.emit(processed_files, total_files)
                self.stats_update.emit(total_specs, total_peaks)

            if self._cancelled:
                self.log_message.emit("Indexing cancelled by user.")
                summary = (
                    f"Indexed spectra: {total_specs} | Peaks: {total_peaks} | "
                    "File clusters: 0 | Global: 0 (cancelled)"
                )
                self.cancelled.emit(summary, str(peaks_path))
                return

            fc = build_file_consensus(con, args)
            gc = build_global_consensus(con, fc, args)
            persist_consensus(con, fc, gc)

            summary = (
                f"Indexed spectra: {total_specs} | Peaks: {total_peaks} | "
                f"File clusters: {len(fc)} | Global: {len(gc)}"
            )
            self.log_message.emit(summary)
            self.finished.emit(summary, str(peaks_path))
        except Exception as exc:
            tb = traceback.format_exc()
            self.log_message.emit(tb)
            self.error_occurred.emit(str(exc))
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass


class FtirIndexerDialog(QDialog):
    """Configuration dialog for building FTIR indexes via jdxIndexBuilder."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("FTIR Index Builder")
        self.setModal(True)
        self._defaults = _load_cli_defaults()

        self._source_dir_edit = QLineEdit()
        self._source_dir_edit.setPlaceholderText("Select JCAMP-DX source folder")
        self._source_dir_edit.setToolTip("Folder containing JCAMP-DX spectra to index.")
        self._source_dir_edit.setWhatsThis(
            "Choose the directory with the .jdx/.dx files that will be processed."
        )

        self._index_dir_edit = QLineEdit()
        self._index_dir_edit.setPlaceholderText("Select index output folder")
        self._index_dir_edit.setToolTip(
            "Directory that will store DuckDB/Parquet outputs from indexing."
        )
        self._index_dir_edit.setWhatsThis(
            "Pick where the generated database and parquet files should be saved."
        )

        source_layout = self._build_path_row(self._source_dir_edit, self._on_pick_source_dir)
        index_layout = self._build_path_row(self._index_dir_edit, self._on_pick_index_dir)

        paths_form = QFormLayout()
        paths_form.addRow("JCAMP source folder", source_layout)
        paths_form.addRow("Index output folder", index_layout)

        paths_tab = QWidget()
        paths_tab.setLayout(paths_form)

        # Peak detection tab -------------------------------------------------
        self._prominence_spin = QDoubleSpinBox()
        self._prominence_spin.setDecimals(4)
        self._prominence_spin.setSingleStep(0.001)
        self._prominence_spin.setRange(0.0, 1.0)
        self._set_numeric_help(
            self._prominence_spin,
            "Minimum peak prominence (absorbance units) passed to scipy.signal.find_peaks.",
        )

        self._min_distance_spin = QSpinBox()
        self._min_distance_spin.setRange(1, 10000)
        self._min_distance_spin.setSingleStep(1)
        self._set_numeric_help(
            self._min_distance_spin,
            "Minimum point distance between neighbouring peaks (cm⁻¹) used in find_peaks.",
        )

        self._sg_window_spin = QSpinBox()
        self._sg_window_spin.setRange(3, 999)
        self._sg_window_spin.setSingleStep(2)
        self._sg_window_spin.setToolTip(
            "Savitzky–Golay filter window length in points (must be odd)."
        )
        self._sg_window_spin.setWhatsThis(
            "Number of points in the Savitzky–Golay smoothing window applied before peak finding."
        )

        self._sg_poly_spin = QSpinBox()
        self._sg_poly_spin.setRange(1, 10)
        self._sg_poly_spin.setSingleStep(1)
        self._set_numeric_help(
            self._sg_poly_spin,
            "Savitzky–Golay polynomial order controlling smoothing stiffness.",
        )

        self._fit_window_spin = QSpinBox()
        self._fit_window_spin.setRange(5, 2000)
        self._fit_window_spin.setSingleStep(1)
        self._set_numeric_help(
            self._fit_window_spin,
            "Number of points centred on a peak used for profile fitting.",
        )

        self._min_r2_spin = QDoubleSpinBox()
        self._min_r2_spin.setDecimals(3)
        self._min_r2_spin.setSingleStep(0.01)
        self._min_r2_spin.setRange(0.0, 1.0)
        self._set_numeric_help(
            self._min_r2_spin,
            "Minimum coefficient of determination required to keep a fitted peak.",
        )

        detection_form = QFormLayout()
        detection_form.addRow("Prominence", self._prominence_spin)
        detection_form.addRow("Min distance", self._min_distance_spin)
        detection_form.addRow("SG window", self._sg_window_spin)
        detection_form.addRow("SG order", self._sg_poly_spin)
        detection_form.addRow("Fit window pts", self._fit_window_spin)
        detection_form.addRow("Minimum R²", self._min_r2_spin)

        detection_tab = QWidget()
        detection_tab.setLayout(detection_form)

        # Baseline tab -------------------------------------------------------
        self._als_lambda_spin = QDoubleSpinBox()
        self._als_lambda_spin.setDecimals(1)
        self._als_lambda_spin.setRange(1.0, 1e9)
        self._als_lambda_spin.setSingleStep(1000.0)
        self._als_lambda_spin.setToolTip(
            "Asymmetric least squares smoothness penalty (λ) for baseline removal."
        )
        self._als_lambda_spin.setWhatsThis(
            "Larger λ enforces a smoother baseline when running ALS background correction."
        )

        self._als_p_spin = QDoubleSpinBox()
        self._als_p_spin.setDecimals(4)
        self._als_p_spin.setRange(0.0001, 0.5)
        self._als_p_spin.setSingleStep(0.001)
        self._als_p_spin.setToolTip(
            "Asymmetric least squares asymmetry parameter (p) controlling negative residual weight."
        )
        self._als_p_spin.setWhatsThis(
            "Smaller p places more weight on positive residuals, yielding a tighter baseline under peaks."
        )

        baseline_form = QFormLayout()
        baseline_form.addRow("ALS λ", self._als_lambda_spin)
        baseline_form.addRow("ALS p", self._als_p_spin)

        baseline_tab = QWidget()
        baseline_tab.setLayout(baseline_form)

        # Clustering tab -----------------------------------------------------
        self._file_min_samples_spin = QSpinBox()
        self._file_min_samples_spin.setRange(1, 1000)
        self._set_numeric_help(
            self._file_min_samples_spin,
            "Minimum number of peaks required per local DBSCAN cluster (per file).",
        )

        self._file_eps_factor_spin = QDoubleSpinBox()
        self._file_eps_factor_spin.setDecimals(3)
        self._file_eps_factor_spin.setRange(0.01, 10.0)
        self._file_eps_factor_spin.setSingleStep(0.05)
        self._set_numeric_help(
            self._file_eps_factor_spin,
            "Multiplier applied to adaptive file-level DBSCAN epsilon relative to fitted peak width.",
        )

        self._file_eps_min_spin = QDoubleSpinBox()
        self._file_eps_min_spin.setDecimals(3)
        self._file_eps_min_spin.setRange(0.01, 20.0)
        self._file_eps_min_spin.setSingleStep(0.1)
        self._set_numeric_help(
            self._file_eps_min_spin,
            "Lower bound (cm⁻¹) for file-level DBSCAN epsilon after scaling by peak width.",
        )

        self._global_min_samples_spin = QSpinBox()
        self._global_min_samples_spin.setRange(1, 1000)
        self._set_numeric_help(
            self._global_min_samples_spin,
            "Minimum spectra contributing to a global consensus peak cluster.",
        )

        self._global_eps_abs_spin = QDoubleSpinBox()
        self._global_eps_abs_spin.setDecimals(3)
        self._global_eps_abs_spin.setRange(0.01, 50.0)
        self._global_eps_abs_spin.setSingleStep(0.1)
        self._set_numeric_help(
            self._global_eps_abs_spin,
            "Absolute epsilon (cm⁻¹) for clustering global consensus peak positions.",
        )

        clustering_form = QFormLayout()
        clustering_form.addRow("File min samples", self._file_min_samples_spin)
        clustering_form.addRow("File eps factor", self._file_eps_factor_spin)
        clustering_form.addRow("File eps min", self._file_eps_min_spin)
        clustering_form.addRow("Global min samples", self._global_min_samples_spin)
        clustering_form.addRow("Global eps abs", self._global_eps_abs_spin)

        clustering_tab = QWidget()
        clustering_tab.setLayout(clustering_form)

        # Options ------------------------------------------------------------
        self._strict_check = QCheckBox("Strict error handling")
        self._strict_check.setToolTip(
            "Raise exceptions instead of skipping files when indexing encounters issues."
        )
        self._strict_check.setWhatsThis(
            "When enabled the indexer aborts on errors rather than skipping problematic spectra."
        )

        options_widget = QWidget()
        options_layout = QVBoxLayout(options_widget)
        options_layout.addWidget(self._strict_check)
        options_layout.addStretch(1)
        self._options_widget = options_widget

        tabs = QTabWidget()
        tabs.addTab(paths_tab, "Paths")
        tabs.addTab(detection_tab, "Peaks")
        tabs.addTab(baseline_tab, "Baseline")
        tabs.addTab(clustering_tab, "Consensus")
        self._tabs = tabs

        restore_btn = QPushButton("Restore defaults")
        restore_btn.clicked.connect(self._restore_defaults)
        self._restore_btn = restore_btn

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("Status messages will appear here during indexing.")
        self._log_output.setMinimumHeight(120)

        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)

        self._button_box = QDialogButtonBox()
        self._run_button = self._button_box.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        self._cancel_button = self._button_box.addButton(
            QDialogButtonBox.StandardButton.Cancel
        )
        self._run_button.clicked.connect(self._on_run_clicked)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(options_widget)
        layout.addWidget(restore_btn)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._log_output, 1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._button_box)

        self._worker_thread: QThread | None = None
        self._worker: IndexerWorker | None = None
        self._is_running = False
        self._total_spectra = 0
        self._total_peaks = 0
        self._total_files = 0
        self._processed_files = 0
        self._current_index_path: Path | None = None

        self._apply_defaults()

    # ------------------------------------------------------------------
    def _build_path_row(self, editor: QLineEdit, picker) -> QWidget:
        button = QToolButton()
        button.setText("...")
        button.setToolTip("Browse for a directory")
        button.clicked.connect(picker)

        row = QHBoxLayout()
        row.addWidget(editor, 1)
        row.addWidget(button)
        container = QWidget()
        container.setLayout(row)
        return container

    def _set_numeric_help(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
        widget.setWhatsThis(text)

    def _on_pick_source_dir(self) -> None:
        start = self._source_dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select JCAMP Source Folder", start)
        if directory:
            self._source_dir_edit.setText(directory)

    def _on_pick_index_dir(self) -> None:
        start = self._index_dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Index Output Folder", start)
        if directory:
            self._index_dir_edit.setText(directory)

    def _apply_defaults(self) -> None:
        defaults = self._defaults
        self._prominence_spin.setValue(float(defaults.get("prominence", 0.02)))
        self._min_distance_spin.setValue(int(defaults.get("min_distance", 5)))
        self._sg_window_spin.setValue(int(defaults.get("sg_win", 9)))
        self._sg_poly_spin.setValue(int(defaults.get("sg_poly", 3)))
        self._als_lambda_spin.setValue(float(defaults.get("als_lam", 1e5)))
        self._als_p_spin.setValue(float(defaults.get("als_p", 0.01)))
        self._fit_window_spin.setValue(int(defaults.get("fit_window_pts", 50)))
        self._min_r2_spin.setValue(float(defaults.get("min_r2", 0.9)))
        self._file_min_samples_spin.setValue(int(defaults.get("file_min_samples", 2)))
        self._file_eps_factor_spin.setValue(float(defaults.get("file_eps_factor", 0.5)))
        self._file_eps_min_spin.setValue(float(defaults.get("file_eps_min", 2.0)))
        self._global_min_samples_spin.setValue(int(defaults.get("global_min_samples", 2)))
        self._global_eps_abs_spin.setValue(float(defaults.get("global_eps_abs", 4.0)))
        self._strict_check.setChecked(bool(defaults.get("strict", False)))

    def _restore_defaults(self) -> None:
        self._apply_defaults()

    # ------------------------------------------------------------------
    def values(self) -> Dict[str, Any]:
        return {
            "data_dir": self._source_dir_edit.text().strip(),
            "index_dir": self._index_dir_edit.text().strip(),
            "prominence": float(self._prominence_spin.value()),
            "min_distance": int(self._min_distance_spin.value()),
            "sg_win": int(self._sg_window_spin.value()),
            "sg_poly": int(self._sg_poly_spin.value()),
            "als_lam": float(self._als_lambda_spin.value()),
            "als_p": float(self._als_p_spin.value()),
            "fit_window_pts": int(self._fit_window_spin.value()),
            "min_r2": float(self._min_r2_spin.value()),
            "file_min_samples": int(self._file_min_samples_spin.value()),
            "file_eps_factor": float(self._file_eps_factor_spin.value()),
            "file_eps_min": float(self._file_eps_min_spin.value()),
            "global_min_samples": int(self._global_min_samples_spin.value()),
            "global_eps_abs": float(self._global_eps_abs_spin.value()),
            "strict": bool(self._strict_check.isChecked()),
        }

    def _validate_inputs(self) -> bool:
        source_text = self._source_dir_edit.text().strip()
        index_text = self._index_dir_edit.text().strip()

        missing = []
        if not source_text:
            missing.append("JCAMP source folder")
        else:
            source = Path(source_text)
            if not source.exists() or not source.is_dir():
                QMessageBox.warning(
                    self,
                    "Invalid folder",
                    "The selected JCAMP source folder does not exist or is not a directory.",
                )
                return False
        if not index_text:
            missing.append("Index output folder")
        else:
            index = Path(index_text)
            if not index.exists() or not index.is_dir():
                QMessageBox.warning(
                    self,
                    "Invalid folder",
                    "The selected index output folder does not exist or is not a directory.",
                )
                return False
        if missing:
            QMessageBox.warning(
                self,
                "Missing folders",
                "Please choose both the JCAMP source and index output directories before continuing.",
            )
            return False
        return True

    def accept(self) -> None:
        if not self._validate_inputs():
            return
        super().accept()

    def reject(self) -> None:
        if self._is_running and self._worker:
            self._on_cancel_clicked()
            return
        super().reject()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        widgets = [
            self._source_dir_edit,
            self._index_dir_edit,
            self._prominence_spin,
            self._min_distance_spin,
            self._sg_window_spin,
            self._sg_poly_spin,
            self._als_lambda_spin,
            self._als_p_spin,
            self._fit_window_spin,
            self._min_r2_spin,
            self._file_min_samples_spin,
            self._file_eps_factor_spin,
            self._file_eps_min_spin,
            self._global_min_samples_spin,
            self._global_eps_abs_spin,
            self._strict_check,
            self._tabs,
            self._options_widget,
            self._restore_btn,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def _append_log(self, message: str) -> None:
        text = message.rstrip()
        if not text:
            return
        self._log_output.appendPlainText(text)

    def _on_run_clicked(self) -> None:
        if self._is_running:
            return
        if not self._validate_inputs():
            return
        params = self.values()
        self._start_worker(params)

    def _on_cancel_clicked(self) -> None:
        if self._is_running and self._worker:
            self._cancel_button.setEnabled(False)
            self._cancel_button.setText("Cancelling…")
            self._append_log("Cancellation requested by user.")
            self._worker.request_cancel()
            return
        super().reject()

    def _start_worker(self, params: Dict[str, Any]) -> None:
        self._is_running = True
        self._set_inputs_enabled(False)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._cancel_button.setText("Cancel")
        self._log_output.clear()
        self._total_spectra = 0
        self._total_peaks = 0
        self._total_files = 0
        self._processed_files = 0
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setValue(0)
        self._status_label.setText("Preparing to index…")
        self._current_index_path = (Path(params["index_dir"]).resolve() / "peaks.duckdb")

        self._worker = IndexerWorker(params)
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress_update.connect(self._on_worker_progress)
        self._worker.stats_update.connect(self._on_worker_stats)
        self._worker.log_message.connect(self._append_log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker_thread.start()

    def _cleanup_worker(self) -> None:
        if self._worker_thread:
            if self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait()
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._worker_thread:
            self._worker_thread.deleteLater()
            self._worker_thread = None

    def _finish_worker(self) -> None:
        self._cleanup_worker()
        self._is_running = False
        self._set_inputs_enabled(True)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(True)
        self._cancel_button.setText("Close")
        self._current_index_path = None

    def _on_worker_progress(self, processed: int, total: int) -> None:
        self._total_files = total
        self._processed_files = processed
        if total > 0:
            self._progress_bar.setRange(0, total)
        else:
            self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(processed)
        self._update_status_label()

    def _on_worker_stats(self, spectra: int, peaks: int) -> None:
        self._total_spectra = spectra
        self._total_peaks = peaks
        self._update_status_label()

    def _update_status_label(self) -> None:
        if not self._is_running:
            return
        if self._total_files:
            progress_text = f"Processed {self._processed_files}/{self._total_files} files"
        else:
            progress_text = "Processed 0/0 files"
        self._status_label.setText(
            f"{progress_text} | Spectra: {self._total_spectra} | Peaks: {self._total_peaks}"
        )

    def _on_worker_finished(self, summary: str, db_path: str) -> None:
        self._progress_bar.setValue(self._progress_bar.maximum())
        footer = f"{summary}\nDatabase: {db_path}"
        self._status_label.setText(footer)
        self._finish_worker()

    def _on_worker_cancelled(self, summary: str, db_path: str) -> None:
        self._progress_bar.setValue(self._progress_bar.value())
        footer = f"{summary}\nDatabase: {db_path}"
        self._status_label.setText(footer)
        self._append_log(summary)
        self._finish_worker()

    def _on_worker_error(self, message: str) -> None:
        db_path = str(self._current_index_path) if self._current_index_path else None
        self._finish_worker()
        if db_path:
            self._status_label.setText(f"Error: {message}\nDatabase: {db_path}")
        else:
            self._status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Indexing failed", message)

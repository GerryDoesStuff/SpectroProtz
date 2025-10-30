from __future__ import annotations

from pathlib import Path
import ast
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
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

        tabs = QTabWidget()
        tabs.addTab(paths_tab, "Paths")
        tabs.addTab(detection_tab, "Peaks")
        tabs.addTab(baseline_tab, "Baseline")
        tabs.addTab(clustering_tab, "Consensus")

        restore_btn = QPushButton("Restore defaults")
        restore_btn.clicked.connect(self._restore_defaults)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(options_widget)
        layout.addWidget(restore_btn)
        layout.addWidget(buttons)

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

    def accept(self) -> None:
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
                return
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
                return
        if missing:
            QMessageBox.warning(
                self,
                "Missing folders",
                "Please choose both the JCAMP source and index output directories before continuing.",
            )
            return
        super().accept()

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping, Sequence

import yaml
from PyQt6 import QtCore
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spectro_app.engine.recipe_model import Recipe


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle_button = QToolButton()
        self._toggle_button.setText(title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(True)
        self._toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)

        self._content_area = QWidget()
        self._content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._content_area)

        self._toggle_button.clicked.connect(self._on_toggled)

    @property
    def content_widget(self) -> QWidget:
        return self._content_area

    def setContentLayout(self, layout: QVBoxLayout | QFormLayout) -> None:
        layout.setContentsMargins(0, 0, 0, 0)
        self._content_area.setLayout(layout)

    def _on_toggled(self, checked: bool) -> None:
        self._content_area.setVisible(checked)
        self._toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if checked
            else QtCore.Qt.ArrowType.RightArrow
        )


class RecipeEditorDock(QDockWidget):
    config_changed = QtCore.pyqtSignal()
    module_changed = QtCore.pyqtSignal(str)
    solvent_reference_requested = QtCore.pyqtSignal()
    _CUSTOM_SENTINEL = "__custom__"
    _JOIN_WINDOWS_GLOBAL_KEY = "__global__"

    def __init__(self, parent=None):
        super().__init__("Recipe Editor", parent)
        self.setObjectName("RecipeEditorDock")
        self.recipe = Recipe()
        self._updating = False
        self._project_root = Path(__file__).resolve().parents[2]
        self._join_windows_store: dict[str, list[dict[str, str]]] = {
            self._JOIN_WINDOWS_GLOBAL_KEY: []
        }
        self._join_windows_active_key = self._JOIN_WINDOWS_GLOBAL_KEY
        self._join_windows_loading = False
        self._join_windows_errors: list[str] = []
        self._stitch_windows_metadata: list[dict[str, object]] = []
        self._stitch_windows_loading = False
        self._stitch_windows_errors: list[str] = []
        self._baseline_anchor_errors: list[str] = []
        self._despike_exclusion_errors: list[str] = []
        self._solvent_reference_entries: dict[str, dict[str, object]] = {}

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)

        preset_form = QFormLayout()
        preset_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        preset_row_widget = QWidget()
        preset_row = QHBoxLayout(preset_row_widget)
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(6)

        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        preset_row.addWidget(self.preset_combo, 1)

        self.load_preset_button = QPushButton("Load Preset")
        preset_row.addWidget(self.load_preset_button)

        self.save_preset_button = QPushButton("Save as Preset…")
        preset_row.addWidget(self.save_preset_button)

        preset_form.addRow("Preset", preset_row_widget)
        layout.addLayout(preset_form)

        self.module = QComboBox()
        self.module.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.module.setToolTip(
            "Select the spectroscopy module so preprocessing uses the matching"
            " recipe keys for UV-Vis, FTIR, Raman, or PEES data."
        )
        self.set_available_modules(
            [
                ("uvvis", "uvvis"),
                ("ftir", "ftir"),
                ("raman", "raman"),
                ("pees", "pees"),
            ]
        )
        module_form = QFormLayout()
        module_form.addRow("Module", self.module)
        layout.addLayout(module_form)

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #0a0;")

        validation_row = QHBoxLayout()
        validation_row.setContentsMargins(0, 0, 0, 0)
        validation_row.setSpacing(6)
        validation_row.addWidget(self.validation_label, 1)

        self.auto_update_preview_checkbox = QCheckBox("Auto-update preview")
        self.auto_update_preview_checkbox.setToolTip(
            "When enabled, processing restarts automatically a short moment after "
            "recipe edits are made."
        )
        self.auto_update_preview_checkbox.setChecked(True)
        validation_row.addWidget(self.auto_update_preview_checkbox)

        layout.addLayout(validation_row)

        # --- Stitching ---
        stitch_section = CollapsibleSection("Stitching")
        stitch_form = QFormLayout()
        stitch_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        stitch_section.setContentLayout(stitch_form)

        self.stitch_enable = QCheckBox("Enable detector stitching")
        self.stitch_enable.setToolTip(
            "Fill masked detector overlap windows using shoulder samples so"
            " multi-detector spectra present a continuous trace before join"
            " correction and despiking."
        )

        self.stitch_shoulder_points = QSpinBox()
        self.stitch_shoulder_points.setRange(0, 999)
        self.stitch_shoulder_points.setValue(5)
        self.stitch_shoulder_points.setToolTip(
            "Number of shoulder samples to draw from each side of a masked"
            " window. Start near 5 when overlaps are dense; reduce only when"
            " shoulder regions are short."
        )

        self.stitch_method = QComboBox()
        self.stitch_method.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.stitch_method.addItem("Linear (degree 1)", "linear")
        self.stitch_method.addItem("Quadratic (degree 2)", "quadratic")
        self.stitch_method.addItem("Cubic (degree 3)", "cubic")
        self.stitch_method.setToolTip(
            "Interpolation method for filling the masked window. Higher degree"
            " polynomials need longer shoulders and can overfit noise."
        )

        self.stitch_fallback_policy = QComboBox()
        self.stitch_fallback_policy.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.stitch_fallback_policy.addItem(
            "Preserve stitched correction", "preserve"
        )
        self.stitch_fallback_policy.addItem(
            "Keep raw segment when stitching fails", "raw"
        )
        self.stitch_fallback_policy.setToolTip(
            "Policy when insufficient shoulders or invalid fits prevent"
            " stitching. 'Preserve' keeps stitched corrections where"
            " possible; 'Raw' falls back to the original detector data."
        )

        stitch_windows_container = QWidget()
        stitch_windows_layout = QVBoxLayout(stitch_windows_container)
        stitch_windows_layout.setContentsMargins(0, 0, 0, 0)
        stitch_windows_layout.setSpacing(6)

        self.stitch_windows_table = QTableWidget(0, 2)
        self.stitch_windows_table.setHorizontalHeaderLabels([
            "Lower (nm)",
            "Upper (nm)",
        ])
        stitch_header = self.stitch_windows_table.horizontalHeader()
        stitch_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        stitch_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.stitch_windows_table.verticalHeader().setVisible(False)
        self.stitch_windows_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.stitch_windows_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        stitch_windows_layout.addWidget(self.stitch_windows_table)

        stitch_windows_buttons = QHBoxLayout()
        stitch_windows_buttons.setContentsMargins(0, 0, 0, 0)
        stitch_windows_buttons.setSpacing(6)
        stitch_windows_buttons.addStretch(1)

        self.stitch_windows_add_row = QPushButton("Add window")
        stitch_windows_buttons.addWidget(self.stitch_windows_add_row)

        self.stitch_windows_remove_row = QPushButton("Remove selected")
        stitch_windows_buttons.addWidget(self.stitch_windows_remove_row)

        stitch_windows_layout.addLayout(stitch_windows_buttons)

        stitch_form.addRow(self.stitch_enable)
        stitch_form.addRow("Shoulder points", self.stitch_shoulder_points)
        stitch_form.addRow("Interpolation", self.stitch_method)
        stitch_form.addRow("Fallback policy", self.stitch_fallback_policy)
        stitch_form.addRow("Windows", stitch_windows_container)
        layout.addWidget(stitch_section)

        # --- Join correction ---
        join_section = CollapsibleSection("Join correction")
        join_form = QFormLayout()
        join_section.setContentLayout(join_form)
        self.join_enable = QCheckBox("Enable detector join correction")
        self.join_enable.setToolTip(
            "Detect and correct step offsets across detector overlaps; enable when"
            " multi-detector instruments need plateau joins stitched."
        )
        self.join_window = QSpinBox()
        self.join_window.setRange(1, 999)
        self.join_window.setValue(10)
        self.join_window.setToolTip(
            "Number of points on each side used to measure join offsets; keep"
            " custom values above 3 so the detector overlap provides enough"
            " context."
        )
        self.join_threshold = QLineEdit()
        self.join_threshold.setPlaceholderText("Leave blank for default")
        self.join_threshold.setClearButtonEnabled(True)
        self.join_threshold.setToolTip(
            "Maximum absorbance offset permitted when stitching detectors. The"
            " UV-Vis default is 0.2 and QC falls back to 0.5, so keep overrides"
            " within that band for consistent flagging."
        )

        join_form.addRow(self.join_enable)
        join_form.addRow("Window", self.join_window)
        join_form.addRow("Threshold", self.join_threshold)

        join_windows_container = QWidget()
        join_windows_layout = QVBoxLayout(join_windows_container)
        join_windows_layout.setContentsMargins(0, 0, 0, 0)
        join_windows_layout.setSpacing(6)

        join_windows_scope_row = QHBoxLayout()
        join_windows_scope_row.setContentsMargins(0, 0, 0, 0)
        join_windows_scope_row.setSpacing(6)

        scope_label = QLabel("Instrument")
        join_windows_scope_row.addWidget(scope_label)

        self.join_windows_instrument_combo = QComboBox()
        self.join_windows_instrument_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        join_windows_scope_row.addWidget(self.join_windows_instrument_combo, 1)

        self.join_windows_add_instrument = QPushButton("Add instrument…")
        join_windows_scope_row.addWidget(self.join_windows_add_instrument)

        self.join_windows_remove_instrument = QPushButton("Remove")
        join_windows_scope_row.addWidget(self.join_windows_remove_instrument)

        join_windows_layout.addLayout(join_windows_scope_row)

        self.join_windows_table = QTableWidget(0, 3)
        self.join_windows_table.setHorizontalHeaderLabels([
            "Min (nm)",
            "Max (nm)",
            "Spikes",
        ])
        header = self.join_windows_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.join_windows_table.verticalHeader().setVisible(False)
        self.join_windows_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.join_windows_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        join_windows_layout.addWidget(self.join_windows_table)

        join_windows_buttons = QHBoxLayout()
        join_windows_buttons.setContentsMargins(0, 0, 0, 0)
        join_windows_buttons.setSpacing(6)
        join_windows_buttons.addStretch(1)

        self.join_windows_add_row = QPushButton("Add window")
        join_windows_buttons.addWidget(self.join_windows_add_row)

        self.join_windows_remove_row = QPushButton("Remove selected")
        join_windows_buttons.addWidget(self.join_windows_remove_row)

        join_windows_layout.addLayout(join_windows_buttons)

        join_form.addRow("Windows", join_windows_container)
        self._refresh_join_windows_combo()
        layout.addWidget(join_section)

        # --- Despiking ---
        despike_section = CollapsibleSection("Despiking")
        despike_form = QFormLayout()
        despike_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        despike_section.setContentLayout(despike_form)

        self.despike_enable = QCheckBox("Enable spike removal")
        self.despike_enable.setToolTip(
            "Detect and replace transient spikes anywhere in the spectrum using a"
            " z-score threshold on a sliding window. Join correction handles plateau"
            " discontinuities, while despiking targets impulsive noise; start from"
            " the documented default window=5/z-score≈5 and only deviate when peak"
            " fidelity or spike persistence needs tighter tuning."
        )
        self.despike_window = QSpinBox()
        self.despike_window.setRange(3, 999)
        self.despike_window.setValue(5)
        self.despike_window.setToolTip(
            "Number of points in the despiking window. Begin at the default 5 and"
            " tighten only if spikes linger, typically ending between 7 and 10 when"
            " broader spikes require more context."
        )
        self.despike_zscore = QDoubleSpinBox()
        self.despike_zscore.setRange(0.1, 99.9)
        self.despike_zscore.setDecimals(2)
        self.despike_zscore.setSingleStep(0.1)
        self.despike_zscore.setValue(5.0)
        self.despike_zscore.setToolTip(
            "Z-score threshold for spike detection. Start near the default ≈5 and"
            " tighten toward 2.5–6 when spikes blend into noise or peaks are"
            " delicate."
        )

        self.despike_baseline_window = QSpinBox()
        self.despike_baseline_window.setRange(0, 999)
        self.despike_baseline_window.setSpecialValueText("Auto")
        self.despike_baseline_window.setValue(0)
        self.despike_baseline_window.setToolTip(
            "Odd-length window for the rolling baseline. Auto (0) mirrors the"
            " detection window—keep that default unless curvature demands manual"
            " spans such as 9–13."
        )

        self.despike_spread_window = QSpinBox()
        self.despike_spread_window.setRange(0, 999)
        self.despike_spread_window.setSpecialValueText("Auto")
        self.despike_spread_window.setValue(0)
        self.despike_spread_window.setToolTip(
            "Odd-length window for local spread estimation. Auto (0) tracks the"
            " detection window by default, with manual choices like 9–13 when"
            " curvature or baseline drift needs extra smoothing."
        )

        self.despike_spread_method = QComboBox()
        self.despike_spread_method.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.despike_spread_method.addItem(
            "Median absolute deviation (MAD)", "mad"
        )
        self.despike_spread_method.addItem(
            "Standard deviation (STD)", "std"
        )
        self.despike_spread_method.setToolTip(
            "Statistic used for the spread term in the z-score. MAD is the default"
            " for robust spike rejection; switch to STD only when smooth,"
            " Gaussian-like noise dominates."
        )

        self.despike_spread_epsilon = QDoubleSpinBox()
        self.despike_spread_epsilon.setDecimals(8)
        self.despike_spread_epsilon.setRange(1e-9, 1.0)
        self.despike_spread_epsilon.setSingleStep(1e-6)
        self.despike_spread_epsilon.setValue(1e-6)
        self.despike_spread_epsilon.setToolTip(
            "Minimum spread used when MAD/STD collapses. Leave at the default"
            " 1e-6 unless flat baselines still trigger spikes, then enlarge"
            " cautiously."
        )

        self.despike_residual_floor = QDoubleSpinBox()
        self.despike_residual_floor.setDecimals(4)
        self.despike_residual_floor.setRange(0.0, 1.0)
        self.despike_residual_floor.setSingleStep(1e-3)
        self.despike_residual_floor.setValue(2e-2)
        self.despike_residual_floor.setToolTip(
            "Floor applied to the residual-based fallback when windows collapse."
            " The 0.02 default keeps isolated spikes removable without"
            " flattening peaks."
        )

        self.despike_tail_decay_ratio = QDoubleSpinBox()
        self.despike_tail_decay_ratio.setDecimals(3)
        self.despike_tail_decay_ratio.setRange(0.1, 1.0)
        self.despike_tail_decay_ratio.setSingleStep(0.01)
        self.despike_tail_decay_ratio.setValue(0.85)
        self.despike_tail_decay_ratio.setToolTip(
            "Controls how quickly spike tails must decay before they are"
            " absorbed into the correction window. Increase toward 1.0 to"
            " follow broader shoulders; decrease to demand steeper drop-offs."
        )

        self.despike_isolation_ratio = QDoubleSpinBox()
        self.despike_isolation_ratio.setDecimals(2)
        self.despike_isolation_ratio.setRange(1.0, 1e3)
        self.despike_isolation_ratio.setSingleStep(0.5)
        self.despike_isolation_ratio.setValue(20.0)
        self.despike_isolation_ratio.setToolTip(
            "Minimum ratio between spike amplitude and the residual floor before"
            " a point is considered isolated and replaced. Stay near the"
            " ≈20 default unless clusters masquerade as noise."
        )

        self.despike_max_passes = QSpinBox()
        self.despike_max_passes.setRange(1, 100)
        self.despike_max_passes.setValue(10)
        self.despike_max_passes.setToolTip(
            "Upper bound on iterative despiking passes per segment. The ≈10"
            " default handles most data; raise only when clustered spikes"
            " persist."
        )

        self.despike_leading_padding = QSpinBox()
        self.despike_leading_padding.setRange(0, 999)
        self.despike_leading_padding.setValue(0)
        self.despike_leading_padding.setToolTip(
            "Samples to protect at the start of each segment before despiking"
            " engages. Leave at the zero-padding default unless sharp leading"
            " edges smear."
        )

        self.despike_trailing_padding = QSpinBox()
        self.despike_trailing_padding.setRange(0, 999)
        self.despike_trailing_padding.setValue(0)
        self.despike_trailing_padding.setToolTip(
            "Samples to protect at the end of each segment so trailing"
            " curvature survives. Retain the zero-padding default unless"
            " peaks roll off too aggressively."
        )

        self.despike_noise_scale_multiplier = QDoubleSpinBox()
        self.despike_noise_scale_multiplier.setDecimals(3)
        self.despike_noise_scale_multiplier.setRange(0.0, 10.0)
        self.despike_noise_scale_multiplier.setSingleStep(0.1)
        self.despike_noise_scale_multiplier.setValue(1.0)
        self.despike_noise_scale_multiplier.setToolTip(
            "Multiplier for the synthetic noise injected into corrected samples."
            " Hold the 1.0 default unless instrument noise statistics are"
            " mis-modeled."
        )

        self.despike_rng_seed = QLineEdit()
        self.despike_rng_seed.setPlaceholderText("Leave blank for random seed")
        self.despike_rng_seed.setClearButtonEnabled(True)
        self.despike_rng_seed.setToolTip(
            "Optional RNG seed for reproducible spike replacement whenever"
            " stochastic inpainting runs; leave blank for randomized fills."
        )

        activation_group = QGroupBox("Activation & detection")
        activation_layout = QFormLayout()
        activation_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        activation_layout.setContentsMargins(6, 6, 6, 6)
        activation_layout.setSpacing(6)
        activation_layout.addRow(self.despike_enable)
        activation_layout.addRow("Window", self.despike_window)
        activation_layout.addRow("Z-score", self.despike_zscore)
        activation_group.setLayout(activation_layout)

        baseline_group = QGroupBox("Baseline & spread")
        baseline_layout = QFormLayout()
        baseline_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        baseline_layout.setContentsMargins(6, 6, 6, 6)
        baseline_layout.setSpacing(6)
        baseline_layout.addRow("Baseline window", self.despike_baseline_window)
        baseline_layout.addRow("Spread window", self.despike_spread_window)
        baseline_layout.addRow("Spread method", self.despike_spread_method)
        baseline_layout.addRow("Spread epsilon", self.despike_spread_epsilon)
        baseline_layout.addRow("Residual floor", self.despike_residual_floor)
        baseline_layout.addRow("Tail decay ratio", self.despike_tail_decay_ratio)
        baseline_group.setLayout(baseline_layout)

        isolation_group = QGroupBox("Isolation & passes")
        isolation_layout = QFormLayout()
        isolation_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        isolation_layout.setContentsMargins(6, 6, 6, 6)
        isolation_layout.setSpacing(6)
        isolation_layout.addRow("Isolation ratio", self.despike_isolation_ratio)
        isolation_layout.addRow("Max passes", self.despike_max_passes)
        isolation_group.setLayout(isolation_layout)

        padding_group = QGroupBox("Padding & noise")
        padding_layout = QFormLayout()
        padding_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        padding_layout.setContentsMargins(6, 6, 6, 6)
        padding_layout.setSpacing(6)
        padding_layout.addRow("Leading padding", self.despike_leading_padding)
        padding_layout.addRow("Trailing padding", self.despike_trailing_padding)
        padding_layout.addRow("Noise scale", self.despike_noise_scale_multiplier)
        padding_layout.addRow("RNG seed", self.despike_rng_seed)
        padding_group.setLayout(padding_layout)

        exclusion_group = QGroupBox("Spike exclusion regions")
        exclusion_layout = QVBoxLayout(exclusion_group)
        exclusion_layout.setContentsMargins(6, 6, 6, 6)
        exclusion_layout.setSpacing(6)

        self.despike_exclusions_table = QTableWidget(0, 2)
        self.despike_exclusions_table.setHorizontalHeaderLabels(["Min (nm)", "Max (nm)"])
        header = self.despike_exclusions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.despike_exclusions_table.verticalHeader().setVisible(False)
        self.despike_exclusions_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.despike_exclusions_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        exclusion_layout.addWidget(self.despike_exclusions_table)

        exclusion_buttons = QHBoxLayout()
        exclusion_buttons.setContentsMargins(0, 0, 0, 0)
        exclusion_buttons.setSpacing(6)
        exclusion_buttons.addStretch(1)

        self.despike_exclusions_add_row = QPushButton("Add window")
        exclusion_buttons.addWidget(self.despike_exclusions_add_row)

        self.despike_exclusions_remove_row = QPushButton("Remove selected")
        exclusion_buttons.addWidget(self.despike_exclusions_remove_row)

        exclusion_layout.addLayout(exclusion_buttons)

        despike_form.addRow(activation_group)
        despike_form.addRow(baseline_group)
        despike_form.addRow(isolation_group)
        despike_form.addRow(padding_group)
        despike_form.addRow(exclusion_group)
        self._despike_control_groups = (
            baseline_group,
            isolation_group,
            padding_group,
            exclusion_group,
        )
        layout.addWidget(despike_section)

        # --- Blank handling ---
        blank_section = CollapsibleSection("Blank handling")
        blank_form = QFormLayout()
        blank_section.setContentLayout(blank_form)
        self.blank_subtract = QCheckBox("Subtract blank from samples")
        self.blank_subtract.setToolTip(
            "Remove instrument/background signal using matched blanks; enable"
            " when blank spectra are available for subtraction."
        )
        self.blank_require = QCheckBox("Require blank for each batch")
        self.blank_require.setToolTip(
            "Fail batches without a paired blank. Leave disabled for automated"
            " workflows that cannot guarantee blanks every run."
        )
        self.blank_fallback = QLineEdit()
        self.blank_fallback.setPlaceholderText("Optional fallback/default blank path")
        self.blank_fallback.setClearButtonEnabled(True)
        self.blank_fallback.setToolTip(
            "Optional path to a default blank used when no matched file is"
            " found; keep it empty when on-the-fly blanks should be mandatory."
        )
        self.blank_match_strategy = QComboBox()
        self.blank_match_strategy.addItem("Match by blank identifier", "blank_id")
        self.blank_match_strategy.addItem("Match by cuvette slot", "cuvette_slot")
        self.blank_match_strategy.setToolTip(
            "Use blank identifiers when files share stable IDs with samples;"
            " switch to cuvette slots when pairing by instrument position so"
            " slot metadata drives the association."
        )

        blank_form.addRow(self.blank_subtract)
        blank_form.addRow(self.blank_require)
        blank_form.addRow("Fallback", self.blank_fallback)
        blank_form.addRow("Match strategy", self.blank_match_strategy)
        layout.addWidget(blank_section)

        # --- Baseline correction ---
        baseline_section = CollapsibleSection("Baseline correction")
        baseline_form = QFormLayout()
        baseline_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        baseline_section.setContentLayout(baseline_form)

        self.baseline_enable = QCheckBox("Enable baseline correction")
        self.baseline_enable.setToolTip(
            "Subtract a modelled background using the selected baseline method;"
            " enable this whenever gradual drift or curvature needs removal."
        )
        self.baseline_method = QComboBox()
        self.baseline_method.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.baseline_method.addItem("Asymmetric least squares (AsLS)", "asls")
        self.baseline_method.addItem("Rubber-band", "rubberband")
        self.baseline_method.addItem("SNIP", "snip")
        self.baseline_method.setToolTip(
            "Choose the baseline solver: AsLS for gradual drift, rubber-band for"
            " broad curvature, or SNIP for fluorescence streaks that need many"
            " iterations."
        )

        self.baseline_lambda = QDoubleSpinBox()
        self.baseline_lambda.setDecimals(3)
        self.baseline_lambda.setRange(0.0, 1e12)
        self.baseline_lambda.setSingleStep(1000.0)
        self._baseline_default_lambda = 1e5
        self.baseline_lambda.setValue(self._baseline_default_lambda)
        self.baseline_lambda.setToolTip(
            "Smoothness penalty (λ) for AsLS baselines; start at 1e5 and explore"
            " roughly 1e3–1e7 depending on how aggressively the background"
            " should be flattened."
        )

        self.baseline_p = QDoubleSpinBox()
        self.baseline_p.setDecimals(5)
        self.baseline_p.setRange(0.0, 1.0)
        self.baseline_p.setSingleStep(0.001)
        self._baseline_default_p = 0.01
        self.baseline_p.setValue(self._baseline_default_p)
        self.baseline_p.setToolTip(
            "Asymmetry parameter (p) for AsLS baselines; typical UV-Vis recipes"
            " stay between 0.001 and 0.05 with 0.01 as a reliable default."
        )

        self.baseline_niter = QSpinBox()
        self.baseline_niter.setRange(1, 999)
        self._baseline_default_niter = 10
        self.baseline_niter.setValue(self._baseline_default_niter)
        self.baseline_niter.setToolTip(
            "Number of refinement passes for AsLS; keep between 10 and 30 to"
            " balance convergence and runtime."
        )

        self.baseline_iterations = QSpinBox()
        self.baseline_iterations.setRange(1, 999)
        self._baseline_default_iterations = 24
        self.baseline_iterations.setValue(self._baseline_default_iterations)
        self.baseline_iterations.setToolTip(
            "Iteration count for SNIP baselines; start around 24 and adjust"
            " within 20–60 for fluorescence-heavy spectra."
        )

        baseline_form.addRow(self.baseline_enable)
        baseline_form.addRow("Method", self.baseline_method)
        baseline_form.addRow("λ (lambda)", self.baseline_lambda)
        baseline_form.addRow("p", self.baseline_p)
        baseline_form.addRow("AsLS iterations", self.baseline_niter)
        baseline_form.addRow("SNIP iterations", self.baseline_iterations)

        baseline_anchor_container = QWidget()
        baseline_anchor_layout = QVBoxLayout(baseline_anchor_container)
        baseline_anchor_layout.setContentsMargins(0, 0, 0, 0)
        baseline_anchor_layout.setSpacing(6)

        self.baseline_anchor_enable = QCheckBox("Apply anchor window zeroing")
        self.baseline_anchor_enable.setToolTip(
            "Zero or level specific wavelength windows after baseline removal by"
            " pinning them to target intensities."
        )
        baseline_anchor_layout.addWidget(self.baseline_anchor_enable)

        self.baseline_anchor_table = QTableWidget(0, 4)
        self.baseline_anchor_table.setHorizontalHeaderLabels(
            ["Min (nm)", "Max (nm)", "Target", "Label"]
        )
        anchor_header = self.baseline_anchor_table.horizontalHeader()
        anchor_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        anchor_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        anchor_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        anchor_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.baseline_anchor_table.verticalHeader().setVisible(False)
        self.baseline_anchor_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.baseline_anchor_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        baseline_anchor_layout.addWidget(self.baseline_anchor_table)

        anchor_buttons = QHBoxLayout()
        anchor_buttons.setContentsMargins(0, 0, 0, 0)
        anchor_buttons.setSpacing(6)
        anchor_buttons.addStretch(1)

        self.baseline_anchor_add_row = QPushButton("Add window")
        anchor_buttons.addWidget(self.baseline_anchor_add_row)

        self.baseline_anchor_remove_row = QPushButton("Remove selected")
        anchor_buttons.addWidget(self.baseline_anchor_remove_row)

        baseline_anchor_layout.addLayout(anchor_buttons)

        baseline_form.addRow("Anchor windows", baseline_anchor_container)
        layout.addWidget(baseline_section)

        # --- Solvent subtraction (FTIR) ---
        self.solvent_section = CollapsibleSection("Solvent subtraction (FTIR)")
        solvent_form = QFormLayout()
        self._solvent_form = solvent_form
        solvent_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.solvent_section.setContentLayout(solvent_form)

        self.solvent_enable = QCheckBox("Enable solvent subtraction")
        self.solvent_enable.setToolTip(
            "Subtract solvent/reference signatures from FTIR spectra before smoothing."
        )

        reference_row = QWidget()
        reference_layout = QHBoxLayout(reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(6)

        self.solvent_reference_combo = QComboBox()
        self.solvent_reference_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.solvent_reference_combo.setToolTip(
            "Select a stored solvent reference to subtract from samples."
        )
        reference_layout.addWidget(self.solvent_reference_combo, 1)

        self.solvent_reference_load = QPushButton("Load…")
        self.solvent_reference_load.setToolTip(
            "Load a solvent reference from file or choose from stored references."
        )
        reference_layout.addWidget(self.solvent_reference_load)

        self.solvent_shift_enable = QCheckBox("Enable shift compensation")
        self.solvent_shift_enable.setToolTip(
            "Enable spectral alignment by scanning small shift offsets."
        )
        self.solvent_shift_min = QDoubleSpinBox()
        self.solvent_shift_min.setDecimals(3)
        self.solvent_shift_min.setRange(-999.0, 999.0)
        self.solvent_shift_min.setSingleStep(0.1)
        self.solvent_shift_min.setValue(-2.0)
        self.solvent_shift_min.setToolTip(
            "Minimum wavenumber shift (cm⁻¹) applied when aligning references."
        )
        self.solvent_shift_max = QDoubleSpinBox()
        self.solvent_shift_max.setDecimals(3)
        self.solvent_shift_max.setRange(-999.0, 999.0)
        self.solvent_shift_max.setSingleStep(0.1)
        self.solvent_shift_max.setValue(2.0)
        self.solvent_shift_max.setToolTip(
            "Maximum wavenumber shift (cm⁻¹) applied when aligning references."
        )
        self.solvent_shift_step = QDoubleSpinBox()
        self.solvent_shift_step.setDecimals(3)
        self.solvent_shift_step.setRange(0.001, 100.0)
        self.solvent_shift_step.setSingleStep(0.05)
        self.solvent_shift_step.setValue(0.1)
        self.solvent_shift_step.setToolTip(
            "Step size (cm⁻¹) used to scan shift offsets."
        )

        shift_bounds_row = QWidget()
        shift_bounds_layout = QHBoxLayout(shift_bounds_row)
        shift_bounds_layout.setContentsMargins(0, 0, 0, 0)
        shift_bounds_layout.setSpacing(6)
        shift_bounds_layout.addWidget(self.solvent_shift_min)
        shift_bounds_layout.addWidget(self.solvent_shift_max)
        shift_bounds_layout.addWidget(self.solvent_shift_step)

        self.solvent_multi_reference_enable = QCheckBox(
            "Select best reference (RMSE)"
        )
        self.solvent_multi_reference_enable.setToolTip(
            "Each selected reference is fit individually (shift/scale/offset) and "
            "the reference with the lowest RMSE is chosen. Candidate scores are "
            "recorded in metadata."
        )
        self.solvent_selection_metric = QComboBox()
        self.solvent_selection_metric.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.solvent_selection_metric.addItem("RMSE (global)", "rmse")
        self.solvent_selection_metric.addItem(
            "Pattern correlation", "pattern_correlation"
        )
        self.solvent_selection_metric.setToolTip(
            "Metric used to select the best solvent reference when multiple "
            "candidates are evaluated."
        )

        self.solvent_selection_region_min = QLineEdit()
        self.solvent_selection_region_min.setPlaceholderText("Min cm⁻¹")
        self.solvent_selection_region_min.setToolTip(
            "Optional minimum wavenumber for reference selection scoring."
        )
        self.solvent_selection_region_min.setValidator(
            QDoubleValidator(-99999.0, 99999.0, 6, self)
        )
        self.solvent_selection_region_max = QLineEdit()
        self.solvent_selection_region_max.setPlaceholderText("Max cm⁻¹")
        self.solvent_selection_region_max.setToolTip(
            "Optional maximum wavenumber for reference selection scoring."
        )
        self.solvent_selection_region_max.setValidator(
            QDoubleValidator(-99999.0, 99999.0, 6, self)
        )
        selection_region_row = QWidget()
        selection_region_layout = QHBoxLayout(selection_region_row)
        selection_region_layout.setContentsMargins(0, 0, 0, 0)
        selection_region_layout.setSpacing(6)
        selection_region_layout.addWidget(self.solvent_selection_region_min, 1)
        selection_region_layout.addWidget(self.solvent_selection_region_max, 1)

        self.solvent_selection_weight_slider = QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.solvent_selection_weight_slider.setRange(0, 100)
        self.solvent_selection_weight_slider.setValue(100)
        self.solvent_selection_weight_slider.setToolTip(
            "Weight applied to the selection region scoring (0–1)."
        )
        self.solvent_selection_weight_spin = QDoubleSpinBox()
        self.solvent_selection_weight_spin.setDecimals(2)
        self.solvent_selection_weight_spin.setRange(0.0, 1.0)
        self.solvent_selection_weight_spin.setSingleStep(0.05)
        self.solvent_selection_weight_spin.setValue(1.0)
        self.solvent_selection_weight_spin.setToolTip(
            "Weight applied to the selection region scoring (0–1)."
        )
        selection_weight_row = QWidget()
        selection_weight_layout = QHBoxLayout(selection_weight_row)
        selection_weight_layout.setContentsMargins(0, 0, 0, 0)
        selection_weight_layout.setSpacing(6)
        selection_weight_layout.addWidget(self.solvent_selection_weight_slider, 1)
        selection_weight_layout.addWidget(self.solvent_selection_weight_spin)

        self.solvent_reference_list = QListWidget()
        self.solvent_reference_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.solvent_reference_list.setToolTip(
            "Select additional solvent references to include in the fit."
        )

        self.solvent_ridge_enable = QCheckBox("Enable ridge regularization")
        self.solvent_ridge_enable.setToolTip(
            "Apply ridge regularization to stabilize multi-reference fits."
        )
        self.solvent_ridge_alpha = QDoubleSpinBox()
        self.solvent_ridge_alpha.setDecimals(6)
        self.solvent_ridge_alpha.setRange(0.0, 1e6)
        self.solvent_ridge_alpha.setSingleStep(0.1)
        self.solvent_ridge_alpha.setValue(0.0)
        self.solvent_ridge_alpha.setToolTip(
            "Regularization strength (α). Increase to reduce coefficient noise."
        )

        self.solvent_offset_enable = QCheckBox("Include constant offset")
        self.solvent_offset_enable.setToolTip(
            "Fit a constant offset term alongside reference coefficients."
        )

        solvent_form.addRow(self.solvent_enable)
        solvent_form.addRow("Reference", reference_row)
        solvent_form.addRow(self.solvent_shift_enable)
        solvent_form.addRow("Shift bounds (min / max / step)", shift_bounds_row)
        solvent_form.addRow(self.solvent_multi_reference_enable)
        solvent_form.addRow("Selection metric", self.solvent_selection_metric)
        solvent_form.addRow(
            "Selection region (min / max cm⁻¹)", selection_region_row
        )
        solvent_form.addRow("Selection region weight", selection_weight_row)
        solvent_form.addRow("Additional references", self.solvent_reference_list)
        self._solvent_reference_list_row = solvent_form.rowCount() - 1
        solvent_form.addRow(self.solvent_ridge_enable)
        solvent_form.addRow("Ridge α", self.solvent_ridge_alpha)
        solvent_form.addRow(self.solvent_offset_enable)
        layout.addWidget(self.solvent_section)

        # --- Smoothing ---
        smoothing_section = CollapsibleSection("Smoothing")
        smoothing_form = QFormLayout()
        smoothing_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        smoothing_section.setContentLayout(smoothing_form)

        self.smooth_enable = QCheckBox("Enable Savitzky–Golay smoothing")
        self.smooth_enable.setToolTip(
            "Toggle global Savitzky–Golay smoothing; defaults of window 15 and"
            " polyorder 3 suit 1 nm UV-Vis sampling and can be reduced for"
            " sparse data."
        )
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 999)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(15)
        self.smooth_window.setToolTip(
            "Number of points in the Savitzky–Golay window. Use odd values larger"
            " than the polynomial order; start around 15 for 1 nm data and shrink"
            " only when working with sparse sampling."
        )
        self.smooth_poly = QSpinBox()
        self.smooth_poly.setRange(1, 15)
        self.smooth_poly.setValue(3)
        self.smooth_poly.setToolTip(
            "Polynomial order used by the Savitzky–Golay fit. Keep it lower than"
            " the window size; an order of 3 is the recommended starting point"
            " for 1 nm sampling."
        )

        smoothing_form.addRow(self.smooth_enable)
        smoothing_form.addRow("Window", self.smooth_window)
        smoothing_form.addRow("Poly order", self.smooth_poly)
        layout.addWidget(smoothing_section)

        # --- Peak detection ---
        peaks_section = CollapsibleSection("Peak detection")
        peaks_form = QFormLayout()
        peaks_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        peaks_section.setContentLayout(peaks_form)

        self.peaks_enable = QCheckBox("Enable peak detection")
        self.peaks_enable.setToolTip(
            "Detect prominent spectral peaks and report their wavelength, height,"
            " and width statistics in analysis outputs."
        )
        self.peaks_prominence = QDoubleSpinBox()
        self.peaks_prominence.setDecimals(4)
        self.peaks_prominence.setRange(0.0, 1e6)
        self.peaks_prominence.setSingleStep(0.01)
        self.peaks_prominence.setValue(0.003)
        self.peaks_prominence.setToolTip(
            "Base prominence floor for peak detection (normalized absorbance units)."
        )
        self.peaks_min_absorbance_threshold = QDoubleSpinBox()
        self.peaks_min_absorbance_threshold.setDecimals(4)
        self.peaks_min_absorbance_threshold.setRange(0.0, 1e6)
        self.peaks_min_absorbance_threshold.setSingleStep(0.01)
        self.peaks_min_absorbance_threshold.setValue(0.05)
        self.peaks_min_absorbance_threshold.setToolTip(
            "Minimum baseline-corrected absorbance to keep a peak candidate."
        )
        self.peaks_noise_sigma_multiplier = QDoubleSpinBox()
        self.peaks_noise_sigma_multiplier.setDecimals(4)
        self.peaks_noise_sigma_multiplier.setRange(0.0, 1e6)
        self.peaks_noise_sigma_multiplier.setSingleStep(0.1)
        self.peaks_noise_sigma_multiplier.setValue(1.5)
        self.peaks_noise_sigma_multiplier.setToolTip(
            "Multiplier applied to estimated noise sigma to set the prominence floor."
        )
        self.peaks_noise_window_cm = QDoubleSpinBox()
        self.peaks_noise_window_cm.setDecimals(3)
        self.peaks_noise_window_cm.setRange(0.0, 1e6)
        self.peaks_noise_window_cm.setSingleStep(10.0)
        self.peaks_noise_window_cm.setValue(400.0)
        self.peaks_noise_window_cm.setToolTip(
            "Window size (cm^-1) for local noise estimation. Set to 0 to use a"
            " single global noise estimate."
        )
        self.peaks_min_prominence_by_region = QLineEdit()
        self.peaks_min_prominence_by_region.setPlaceholderText(
            'e.g. "400-1500:0.03,1500-1800:0.02"'
        )
        self.peaks_min_prominence_by_region.setClearButtonEnabled(True)
        self.peaks_min_prominence_by_region.setToolTip(
            "Comma-separated regional prominence floors, e.g. "
            '"400-1500:0.03,1500-1800:0.02". Use this to suppress low-value '
            "artifacts in targeted ranges."
        )
        self.peaks_min_distance = QDoubleSpinBox()
        self.peaks_min_distance.setDecimals(3)
        self.peaks_min_distance.setRange(0.0, 1e6)
        self.peaks_min_distance.setSingleStep(0.1)
        self.peaks_min_distance.setValue(0.8)
        self.peaks_min_distance.setToolTip(
            "Minimum peak separation in cm^-1 (converted to points using median spacing)."
        )
        self.peaks_min_distance_mode = QComboBox()
        self.peaks_min_distance_mode.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.peaks_min_distance_mode.addItem("Fixed", "fixed")
        self.peaks_min_distance_mode.addItem("Adaptive (median FWHM)", "adaptive")
        self.peaks_min_distance_mode.setToolTip(
            "Use a fixed min distance or clamp it to a fraction of the median peak FWHM"
            " (adaptive)."
        )
        self.peaks_min_distance_fwhm_fraction = QDoubleSpinBox()
        self.peaks_min_distance_fwhm_fraction.setDecimals(3)
        self.peaks_min_distance_fwhm_fraction.setRange(0.0, 10.0)
        self.peaks_min_distance_fwhm_fraction.setSingleStep(0.05)
        self.peaks_min_distance_fwhm_fraction.setValue(0.5)
        self.peaks_min_distance_fwhm_fraction.setToolTip(
            "Fraction of median peak FWHM to clamp min-distance in adaptive mode."
        )
        self.peaks_peak_width_min = QDoubleSpinBox()
        self.peaks_peak_width_min.setDecimals(3)
        self.peaks_peak_width_min.setRange(0.0, 1e6)
        self.peaks_peak_width_min.setSingleStep(0.5)
        self.peaks_peak_width_min.setValue(0.0)
        self.peaks_peak_width_min.setToolTip(
            "Minimum peak width in cm^-1; set to 0 to disable width filtering."
        )
        self.peaks_peak_width_max = QDoubleSpinBox()
        self.peaks_peak_width_max.setDecimals(3)
        self.peaks_peak_width_max.setRange(0.0, 1e6)
        self.peaks_peak_width_max.setSingleStep(0.5)
        self.peaks_peak_width_max.setValue(0.0)
        self.peaks_peak_width_max.setToolTip(
            "Maximum peak width in cm^-1; set to 0 to disable width filtering."
        )
        self.peaks_max_peak_candidates = QSpinBox()
        self.peaks_max_peak_candidates.setRange(0, 100000)
        self.peaks_max_peak_candidates.setValue(0)
        self.peaks_max_peak_candidates.setToolTip(
            "Maximum number of peak candidates to retain per spectrum (0 disables)."
        )
        self.peaks_detect_negative = QCheckBox("Detect negative peaks")
        self.peaks_detect_negative.setToolTip(
            "Also detect negative peaks by searching inverted spectra."
        )
        self.peaks_merge_tolerance = QDoubleSpinBox()
        self.peaks_merge_tolerance.setDecimals(3)
        self.peaks_merge_tolerance.setRange(0.0, 1e6)
        self.peaks_merge_tolerance.setSingleStep(0.5)
        self.peaks_merge_tolerance.setValue(8.0)
        self.peaks_merge_tolerance.setToolTip(
            "Cluster tolerance (cm^-1) for merging nearby peak detections."
        )
        self.peaks_close_peak_tolerance = QDoubleSpinBox()
        self.peaks_close_peak_tolerance.setDecimals(3)
        self.peaks_close_peak_tolerance.setRange(0.0, 1e6)
        self.peaks_close_peak_tolerance.setSingleStep(0.1)
        self.peaks_close_peak_tolerance.setValue(1.5)
        self.peaks_close_peak_tolerance.setToolTip(
            "Tolerance (cm^-1) for labeling peaks that are unusually close together."
        )
        self.peaks_plateau_min_points = QSpinBox()
        self.peaks_plateau_min_points.setRange(2, 999)
        self.peaks_plateau_min_points.setValue(3)
        self.peaks_plateau_min_points.setToolTip(
            "Minimum number of consecutive samples required to flag a plateau peak."
        )
        self.peaks_plateau_prominence_factor = QDoubleSpinBox()
        self.peaks_plateau_prominence_factor.setDecimals(3)
        self.peaks_plateau_prominence_factor.setRange(0.0, 10.0)
        self.peaks_plateau_prominence_factor.setSingleStep(0.1)
        self.peaks_plateau_prominence_factor.setValue(1.0)
        self.peaks_plateau_prominence_factor.setToolTip(
            "Multiplier applied to the median prominence when detecting plateau peaks."
        )
        self.peaks_shoulder_min_distance = QDoubleSpinBox()
        self.peaks_shoulder_min_distance.setDecimals(3)
        self.peaks_shoulder_min_distance.setRange(0.0, 1e6)
        self.peaks_shoulder_min_distance.setSingleStep(0.1)
        self.peaks_shoulder_min_distance.setValue(0.0)
        self.peaks_shoulder_min_distance.setSpecialValueText("Auto")
        self.peaks_shoulder_min_distance.setToolTip(
            "Minimum distance between shoulder detections in cm^-1. Leave at Auto to"
            " derive from the sampling step and min-distance."
        )
        self.peaks_shoulder_merge_tolerance = QDoubleSpinBox()
        self.peaks_shoulder_merge_tolerance.setDecimals(3)
        self.peaks_shoulder_merge_tolerance.setRange(0.0, 1e6)
        self.peaks_shoulder_merge_tolerance.setSingleStep(0.1)
        self.peaks_shoulder_merge_tolerance.setValue(1.5)
        self.peaks_shoulder_merge_tolerance.setToolTip(
            "Merge tolerance (cm^-1) for shoulder detections."
        )
        self.peaks_shoulder_curvature_prominence_factor = QDoubleSpinBox()
        self.peaks_shoulder_curvature_prominence_factor.setDecimals(3)
        self.peaks_shoulder_curvature_prominence_factor.setRange(0.0, 10.0)
        self.peaks_shoulder_curvature_prominence_factor.setSingleStep(0.1)
        self.peaks_shoulder_curvature_prominence_factor.setValue(1.2)
        self.peaks_shoulder_curvature_prominence_factor.setToolTip(
            "Multiplier applied to curvature prominence when detecting shoulders."
        )
        self.peaks_cwt_enabled = QCheckBox("Enable CWT detection")
        self.peaks_cwt_enabled.setToolTip(
            "Use continuous wavelet transform (CWT) detection to supplement peak picks."
        )
        self.peaks_cwt_width_min = QDoubleSpinBox()
        self.peaks_cwt_width_min.setDecimals(3)
        self.peaks_cwt_width_min.setRange(0.0, 1e6)
        self.peaks_cwt_width_min.setSingleStep(0.5)
        self.peaks_cwt_width_min.setValue(0.0)
        self.peaks_cwt_width_min.setToolTip(
            "Minimum CWT width in cm^-1 (0 uses the main width minimum)."
        )
        self.peaks_cwt_width_max = QDoubleSpinBox()
        self.peaks_cwt_width_max.setDecimals(3)
        self.peaks_cwt_width_max.setRange(0.0, 1e6)
        self.peaks_cwt_width_max.setSingleStep(0.5)
        self.peaks_cwt_width_max.setValue(0.0)
        self.peaks_cwt_width_max.setToolTip(
            "Maximum CWT width in cm^-1 (0 uses the main width maximum)."
        )
        self.peaks_cwt_width_step = QDoubleSpinBox()
        self.peaks_cwt_width_step.setDecimals(3)
        self.peaks_cwt_width_step.setRange(0.0, 1e6)
        self.peaks_cwt_width_step.setSingleStep(0.5)
        self.peaks_cwt_width_step.setValue(0.0)
        self.peaks_cwt_width_step.setToolTip(
            "CWT width step in cm^-1 (0 derives the step from the width span)."
        )
        self.peaks_cwt_cluster_tolerance = QDoubleSpinBox()
        self.peaks_cwt_cluster_tolerance.setDecimals(3)
        self.peaks_cwt_cluster_tolerance.setRange(0.0, 1e6)
        self.peaks_cwt_cluster_tolerance.setSingleStep(0.5)
        self.peaks_cwt_cluster_tolerance.setValue(8.0)
        self.peaks_cwt_cluster_tolerance.setToolTip(
            "Cluster tolerance (cm^-1) used to merge adjacent CWT detections."
        )

        peaks_form.addRow(self.peaks_enable)
        peaks_form.addRow("Prominence", self.peaks_prominence)
        peaks_form.addRow("Min absorbance threshold", self.peaks_min_absorbance_threshold)
        peaks_form.addRow("Noise sigma multiplier", self.peaks_noise_sigma_multiplier)
        peaks_form.addRow("Noise window (cm^-1)", self.peaks_noise_window_cm)
        peaks_form.addRow("Min prominence by region", self.peaks_min_prominence_by_region)
        peaks_form.addRow("Min distance", self.peaks_min_distance)
        peaks_form.addRow("Min distance mode", self.peaks_min_distance_mode)
        peaks_form.addRow("Min distance FWHM fraction", self.peaks_min_distance_fwhm_fraction)
        peaks_form.addRow("Peak width min (cm^-1)", self.peaks_peak_width_min)
        peaks_form.addRow("Peak width max (cm^-1)", self.peaks_peak_width_max)
        peaks_form.addRow("Max peak candidates", self.peaks_max_peak_candidates)
        peaks_form.addRow(self.peaks_detect_negative)
        peaks_form.addRow("Merge tolerance (cm^-1)", self.peaks_merge_tolerance)
        peaks_form.addRow("Close-peak tolerance (cm^-1)", self.peaks_close_peak_tolerance)
        peaks_form.addRow("Plateau min points", self.peaks_plateau_min_points)
        peaks_form.addRow("Plateau prominence factor", self.peaks_plateau_prominence_factor)
        peaks_form.addRow("Shoulder min distance (cm^-1)", self.peaks_shoulder_min_distance)
        peaks_form.addRow("Shoulder merge tolerance (cm^-1)", self.peaks_shoulder_merge_tolerance)
        peaks_form.addRow(
            "Shoulder curvature prominence factor",
            self.peaks_shoulder_curvature_prominence_factor,
        )
        peaks_form.addRow(self.peaks_cwt_enabled)
        peaks_form.addRow("CWT width min (cm^-1)", self.peaks_cwt_width_min)
        peaks_form.addRow("CWT width max (cm^-1)", self.peaks_cwt_width_max)
        peaks_form.addRow("CWT width step (cm^-1)", self.peaks_cwt_width_step)
        peaks_form.addRow("CWT cluster tolerance (cm^-1)", self.peaks_cwt_cluster_tolerance)
        layout.addWidget(peaks_section)

        # --- QC / Drift ---
        qc_section = CollapsibleSection("Quality control – drift limits")
        qc_layout = QVBoxLayout()
        qc_section.setContentLayout(qc_layout)
        self.drift_enable = QCheckBox("Enable drift monitoring")
        self.drift_enable.setToolTip(
            "Track baseline drift over time and flag batches when slope, delta,"
            " or residual limits are exceeded."
        )
        qc_layout.addWidget(self.drift_enable)

        drift_bounds_row = QHBoxLayout()
        self.drift_window_min = QLineEdit()
        self.drift_window_min.setPlaceholderText("Quiet window min")
        self.drift_window_min.setClearButtonEnabled(True)
        self.drift_window_min.setToolTip(
            "Lower wavelength bound (nm) for the quiet drift window. Align this"
            " with a flat region such as 210–290 nm when modelling drift."
        )
        self.drift_window_max = QLineEdit()
        self.drift_window_max.setPlaceholderText("Quiet window max")
        self.drift_window_max.setClearButtonEnabled(True)
        self.drift_window_max.setToolTip(
            "Upper wavelength bound (nm) for the quiet drift window; pair with"
            " the minimum to enclose the same flat region."
        )
        drift_bounds_row.addWidget(self.drift_window_min)
        drift_bounds_row.addWidget(self.drift_window_max)
        qc_layout.addLayout(drift_bounds_row)

        qc_limits_form = QFormLayout()
        self.drift_max_slope = QLineEdit()
        self.drift_max_slope.setPlaceholderText("Max slope per hour")
        self.drift_max_slope.setClearButtonEnabled(True)
        self.drift_max_slope.setToolTip(
            "Largest acceptable drift slope (absorbance/hour); start around 0.5"
            " to flag trends exceeding typical QC tolerance."
        )
        self.drift_max_delta = QLineEdit()
        self.drift_max_delta.setPlaceholderText("Max delta allowed")
        self.drift_max_delta.setClearButtonEnabled(True)
        self.drift_max_delta.setToolTip(
            "Maximum total baseline change allowed across the batch; 0.6 is a"
            " representative limit for UV-Vis drift checks."
        )
        self.drift_max_residual = QLineEdit()
        self.drift_max_residual.setPlaceholderText("Max residual allowed")
        self.drift_max_residual.setClearButtonEnabled(True)
        self.drift_max_residual.setToolTip(
            "Largest acceptable residual error after drift fitting; values"
            " near 0.4 catch noisy baselines without over-flagging."
        )

        qc_limits_form.addRow("Slope", self.drift_max_slope)
        qc_limits_form.addRow("Delta", self.drift_max_delta)
        qc_limits_form.addRow("Residual", self.drift_max_residual)
        qc_layout.addLayout(qc_limits_form)
        layout.addWidget(qc_section)

        layout.addStretch(1)

        self.setWidget(scroll_area)

        self._refresh_preset_list()
        self._connect_signals()
        self._update_ui_from_recipe()

    def _connect_signals(self):
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.stitch_windows_add_row.clicked.connect(
            self._on_stitch_windows_add_row
        )
        self.stitch_windows_remove_row.clicked.connect(
            self._on_stitch_windows_remove_row
        )
        self.stitch_windows_table.itemSelectionChanged.connect(
            self._update_stitch_window_buttons
        )
        self.join_windows_instrument_combo.currentIndexChanged.connect(
            self._on_join_windows_instrument_changed
        )
        self.join_windows_add_instrument.clicked.connect(
            self._on_join_windows_add_instrument
        )
        self.join_windows_remove_instrument.clicked.connect(
            self._on_join_windows_remove_instrument
        )
        self.join_windows_add_row.clicked.connect(self._on_join_windows_add_row)
        self.join_windows_remove_row.clicked.connect(
            self._on_join_windows_remove_row
        )
        self.despike_exclusions_add_row.clicked.connect(
            self._on_despike_exclusions_add_row
        )
        self.despike_exclusions_remove_row.clicked.connect(
            self._on_despike_exclusions_remove_row
        )
        self.despike_exclusions_table.itemChanged.connect(
            self._on_despike_exclusions_item_changed
        )
        self.despike_exclusions_table.itemSelectionChanged.connect(
            self._update_despike_exclusion_buttons
        )
        self.baseline_enable.toggled.connect(self._on_baseline_enable_toggled)
        self.baseline_method.currentIndexChanged.connect(
            self._on_baseline_method_changed
        )
        self.baseline_anchor_enable.toggled.connect(
            self._on_baseline_anchor_enable_toggled
        )
        self.baseline_anchor_add_row.clicked.connect(
            self._on_baseline_anchor_add_row
        )
        self.baseline_anchor_remove_row.clicked.connect(
            self._on_baseline_anchor_remove_row
        )
        self.baseline_anchor_table.itemChanged.connect(
            self._on_baseline_anchor_item_changed
        )
        self.baseline_anchor_table.itemSelectionChanged.connect(
            self._update_baseline_anchor_buttons
        )
        self.solvent_reference_load.clicked.connect(
            self.solvent_reference_requested.emit
        )
        self.solvent_enable.toggled.connect(self._update_solvent_controls_enabled)
        self.solvent_multi_reference_enable.toggled.connect(
            self._update_solvent_controls_enabled
        )
        self.solvent_selection_weight_slider.valueChanged.connect(
            self._on_solvent_weight_slider_changed
        )
        self.solvent_selection_weight_spin.valueChanged.connect(
            self._on_solvent_weight_spin_changed
        )
        self.solvent_ridge_enable.toggled.connect(
            self._update_solvent_controls_enabled
        )
        self.solvent_shift_enable.toggled.connect(
            self._update_solvent_controls_enabled
        )
        self.smooth_enable.toggled.connect(self._update_feature_controls_enabled)
        self.stitch_enable.toggled.connect(self._update_feature_controls_enabled)
        self.peaks_enable.toggled.connect(self._update_feature_controls_enabled)
        self.despike_enable.toggled.connect(self._update_feature_controls_enabled)
        self.join_enable.toggled.connect(self._update_feature_controls_enabled)
        self.blank_subtract.toggled.connect(self._update_feature_controls_enabled)
        self.drift_enable.toggled.connect(self._update_feature_controls_enabled)
        for signal in (
            self.module.currentTextChanged,
            self.smooth_enable.toggled,
            self.smooth_window.valueChanged,
            self.smooth_poly.valueChanged,
            self.stitch_enable.toggled,
            self.stitch_shoulder_points.valueChanged,
            self.stitch_method.currentIndexChanged,
            self.stitch_fallback_policy.currentIndexChanged,
            self.peaks_enable.toggled,
            self.peaks_prominence.valueChanged,
            self.peaks_min_distance.valueChanged,
            self.peaks_min_absorbance_threshold.valueChanged,
            self.peaks_noise_sigma_multiplier.valueChanged,
            self.peaks_noise_window_cm.valueChanged,
            self.peaks_min_prominence_by_region.textChanged,
            self.peaks_min_distance_mode.currentIndexChanged,
            self.peaks_min_distance_fwhm_fraction.valueChanged,
            self.peaks_peak_width_min.valueChanged,
            self.peaks_peak_width_max.valueChanged,
            self.peaks_max_peak_candidates.valueChanged,
            self.peaks_detect_negative.toggled,
            self.peaks_merge_tolerance.valueChanged,
            self.peaks_close_peak_tolerance.valueChanged,
            self.peaks_plateau_min_points.valueChanged,
            self.peaks_plateau_prominence_factor.valueChanged,
            self.peaks_shoulder_min_distance.valueChanged,
            self.peaks_shoulder_merge_tolerance.valueChanged,
            self.peaks_shoulder_curvature_prominence_factor.valueChanged,
            self.peaks_cwt_enabled.toggled,
            self.peaks_cwt_width_min.valueChanged,
            self.peaks_cwt_width_max.valueChanged,
            self.peaks_cwt_width_step.valueChanged,
            self.peaks_cwt_cluster_tolerance.valueChanged,
            self.baseline_lambda.valueChanged,
            self.baseline_p.valueChanged,
            self.baseline_niter.valueChanged,
            self.baseline_iterations.valueChanged,
            self.solvent_enable.toggled,
            self.solvent_reference_combo.currentIndexChanged,
            self.solvent_shift_enable.toggled,
            self.solvent_shift_min.valueChanged,
            self.solvent_shift_max.valueChanged,
            self.solvent_shift_step.valueChanged,
            self.solvent_multi_reference_enable.toggled,
            self.solvent_selection_metric.currentIndexChanged,
            self.solvent_selection_region_min.textChanged,
            self.solvent_selection_region_max.textChanged,
            self.solvent_selection_weight_spin.valueChanged,
            self.solvent_reference_list.itemSelectionChanged,
            self.solvent_ridge_enable.toggled,
            self.solvent_ridge_alpha.valueChanged,
            self.solvent_offset_enable.toggled,
            self.despike_enable.toggled,
            self.despike_window.valueChanged,
            self.despike_zscore.valueChanged,
            self.despike_baseline_window.valueChanged,
            self.despike_spread_window.valueChanged,
            self.despike_spread_method.currentIndexChanged,
            self.despike_spread_epsilon.valueChanged,
            self.despike_residual_floor.valueChanged,
            self.despike_tail_decay_ratio.valueChanged,
            self.despike_isolation_ratio.valueChanged,
            self.despike_max_passes.valueChanged,
            self.despike_leading_padding.valueChanged,
            self.despike_trailing_padding.valueChanged,
            self.despike_noise_scale_multiplier.valueChanged,
            self.despike_rng_seed.textChanged,
            self.join_enable.toggled,
            self.join_window.valueChanged,
            self.join_threshold.textChanged,
            self.blank_subtract.toggled,
            self.blank_require.toggled,
            self.blank_fallback.textChanged,
            self.blank_match_strategy.currentIndexChanged,
            self.drift_enable.toggled,
            self.drift_window_min.textChanged,
            self.drift_window_max.textChanged,
            self.drift_max_slope.textChanged,
            self.drift_max_delta.textChanged,
            self.drift_max_residual.textChanged,
        ):
            signal.connect(self._update_model_from_ui)
        self.module.currentTextChanged.connect(self._on_module_changed)

    def _on_module_changed(self, text: str) -> None:
        if self._updating:
            return
        module_id = self.module.currentData(QtCore.Qt.ItemDataRole.UserRole)
        module_id = module_id if isinstance(module_id, str) else text
        self._update_solvent_section_visibility(module_id)
        self._update_solvent_controls_enabled()
        self.module_changed.emit(text)

    def set_available_modules(self, modules: list[tuple[str, str]] | None) -> None:
        cleaned: list[tuple[str, str]] = []
        seen: set[str] = set()
        if modules:
            for module_id, label in modules:
                raw_id = str(module_id or "").strip().lower()
                if not raw_id or raw_id in seen:
                    continue
                seen.add(raw_id)
                display = str(label or module_id or raw_id).strip() or raw_id
                cleaned.append((raw_id, display))

        if not cleaned:
            default_id = (self.recipe.module or "uvvis").strip().lower()
            cleaned.append((default_id, default_id))

        with self._suspend_updates():
            current_id = self.module.currentData(QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(current_id, str) or not current_id:
                current_id = (self.recipe.module or "uvvis").strip().lower()
            self.module.clear()
            for module_id, display_label in cleaned:
                self.module.addItem(display_label, module_id)
            index = self.module.findData(
                current_id, QtCore.Qt.ItemDataRole.UserRole
            )
            if index < 0 and cleaned:
                index = 0
            if index >= 0:
                self.module.setCurrentIndex(index)

    def set_solvent_reference_entries(
        self, entries: list[dict[str, object]] | None
    ) -> None:
        cleaned: dict[str, dict[str, object]] = {}
        if entries:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    continue
                name = str(entry.get("name") or entry_id).strip()
                payload = dict(entry)
                payload["id"] = entry_id
                payload["name"] = name or entry_id
                cleaned[entry_id] = payload

        active_id = self._current_solvent_reference_id()
        if active_id and active_id in self._solvent_reference_entries:
            cleaned.setdefault(
                active_id, self._solvent_reference_entries[active_id]
            )

        self._solvent_reference_entries = cleaned
        self._refresh_solvent_reference_widgets(
            selected_id=active_id, selected_ids=[active_id] if active_id else None
        )

    def set_selected_solvent_reference(self, entry: dict[str, object]) -> None:
        if not isinstance(entry, dict):
            return
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            return
        payload = dict(entry)
        payload.setdefault("name", entry_id)
        self._solvent_reference_entries[entry_id] = payload
        self._refresh_solvent_reference_widgets(
            selected_id=entry_id, selected_ids=[entry_id]
        )
        if not self._updating:
            self._update_model_from_ui()

    @contextmanager
    def _suspend_updates(self):
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def set_recipe(self, data: Recipe | dict[str, object]):
        """Load a recipe object or dictionary into the editor."""
        if isinstance(data, Recipe):
            self.recipe = data
        elif isinstance(data, dict):
            params = data.get("params") if isinstance(data.get("params"), dict) else None
            if params is None:
                params = {k: v for k, v in data.items() if k not in {"module", "version"}}
            module = data.get("module", self.recipe.module) if isinstance(data.get("module"), str) else self.recipe.module
            version = data.get("version", self.recipe.version) if isinstance(data.get("version"), str) else self.recipe.version
            self.recipe = Recipe(module=module, params=copy.deepcopy(params or {}), version=version)
        else:
            raise TypeError("Recipe data must be a Recipe instance or mapping")
        self._update_ui_from_recipe()

    def recipe_dict(self) -> dict[str, object]:
        params = self.recipe.params if isinstance(self.recipe.params, dict) else {}
        return {
            "module": self.recipe.module,
            "params": copy.deepcopy(params),
            "version": self.recipe.version,
        }

    def _update_ui_from_recipe(self):
        with self._suspend_updates():
            module = (self.recipe.module or "uvvis").strip().lower()
            index = self.module.findData(module, QtCore.Qt.ItemDataRole.UserRole)
            if index < 0:
                self.module.addItem(module, module)
                index = self.module.findData(
                    module, QtCore.Qt.ItemDataRole.UserRole
                )
            if index >= 0:
                self.module.setCurrentIndex(index)
            self._update_solvent_section_visibility(module)

            params = self.recipe.params if isinstance(self.recipe.params, dict) else {}
            smoothing = params.get("smoothing", {}) if isinstance(params.get("smoothing"), dict) else {}
            self.smooth_enable.setChecked(bool(smoothing.get("enabled", False)))
            self.smooth_window.setValue(
                self._safe_int(smoothing.get("window"), self.smooth_window.value())
            )
            self.smooth_poly.setValue(
                self._safe_int(smoothing.get("polyorder"), self.smooth_poly.value())
            )

            features_cfg = (
                params.get("features", {})
                if isinstance(params.get("features"), dict)
                else {}
            )
            peaks_cfg_raw = features_cfg.get("peaks") if isinstance(features_cfg, Mapping) else None
            peaks_cfg = peaks_cfg_raw if isinstance(peaks_cfg_raw, Mapping) else {}
            peaks_enabled = bool(peaks_cfg.get("enabled", False))
            self.peaks_enable.setChecked(peaks_enabled)
            self.peaks_prominence.setValue(
                self._safe_float(
                    peaks_cfg.get("prominence"),
                    self.peaks_prominence.value(),
                )
            )
            self.peaks_min_absorbance_threshold.setValue(
                self._safe_float(
                    peaks_cfg.get("min_absorbance_threshold"),
                    self.peaks_min_absorbance_threshold.value(),
                )
            )
            self.peaks_noise_sigma_multiplier.setValue(
                self._safe_float(
                    peaks_cfg.get("noise_sigma_multiplier"),
                    self.peaks_noise_sigma_multiplier.value(),
                )
            )
            self.peaks_noise_window_cm.setValue(
                self._safe_float(
                    peaks_cfg.get("noise_window_cm"),
                    self.peaks_noise_window_cm.value(),
                )
            )
            region_prominence = peaks_cfg.get("min_prominence_by_region")
            self.peaks_min_prominence_by_region.setText(
                "" if region_prominence in (None, "") else str(region_prominence)
            )
            self.peaks_min_distance.setValue(
                self._safe_float(
                    peaks_cfg.get("min_distance", peaks_cfg.get("distance")),
                    self.peaks_min_distance.value(),
                )
            )
            min_distance_mode = str(peaks_cfg.get("min_distance_mode") or "fixed").lower()
            if min_distance_mode not in {"fixed", "adaptive"}:
                min_distance_mode = "fixed"
            min_distance_mode_index = self.peaks_min_distance_mode.findData(
                min_distance_mode, QtCore.Qt.ItemDataRole.UserRole
            )
            if min_distance_mode_index < 0:
                min_distance_mode_index = 0
            self.peaks_min_distance_mode.setCurrentIndex(min_distance_mode_index)
            self.peaks_min_distance_fwhm_fraction.setValue(
                self._safe_float(
                    peaks_cfg.get("min_distance_fwhm_fraction"),
                    self.peaks_min_distance_fwhm_fraction.value(),
                )
            )
            self.peaks_peak_width_min.setValue(
                self._safe_float(
                    peaks_cfg.get("peak_width_min"),
                    self.peaks_peak_width_min.value(),
                )
            )
            self.peaks_peak_width_max.setValue(
                self._safe_float(
                    peaks_cfg.get("peak_width_max"),
                    self.peaks_peak_width_max.value(),
                )
            )
            self.peaks_max_peak_candidates.setValue(
                self._safe_int(
                    peaks_cfg.get(
                        "max_peak_candidates",
                        peaks_cfg.get("max_peaks", peaks_cfg.get("num_peaks")),
                    ),
                    self.peaks_max_peak_candidates.value(),
                )
            )
            self.peaks_detect_negative.setChecked(
                bool(peaks_cfg.get("detect_negative_peaks", False))
            )
            self.peaks_merge_tolerance.setValue(
                self._safe_float(
                    peaks_cfg.get("merge_tolerance"),
                    self.peaks_merge_tolerance.value(),
                )
            )
            self.peaks_close_peak_tolerance.setValue(
                self._safe_float(
                    peaks_cfg.get("close_peak_tolerance_cm"),
                    self.peaks_close_peak_tolerance.value(),
                )
            )
            self.peaks_plateau_min_points.setValue(
                self._safe_int(
                    peaks_cfg.get("plateau_min_points"),
                    self.peaks_plateau_min_points.value(),
                )
            )
            self.peaks_plateau_prominence_factor.setValue(
                self._safe_float(
                    peaks_cfg.get("plateau_prominence_factor"),
                    self.peaks_plateau_prominence_factor.value(),
                )
            )
            self.peaks_shoulder_min_distance.setValue(
                self._safe_float(
                    peaks_cfg.get("shoulder_min_distance_cm"),
                    self.peaks_shoulder_min_distance.value(),
                )
            )
            self.peaks_shoulder_merge_tolerance.setValue(
                self._safe_float(
                    peaks_cfg.get("shoulder_merge_tolerance_cm"),
                    self.peaks_shoulder_merge_tolerance.value(),
                )
            )
            self.peaks_shoulder_curvature_prominence_factor.setValue(
                self._safe_float(
                    peaks_cfg.get("shoulder_curvature_prominence_factor"),
                    self.peaks_shoulder_curvature_prominence_factor.value(),
                )
            )
            self.peaks_cwt_enabled.setChecked(
                bool(peaks_cfg.get("cwt_enabled", False))
            )
            self.peaks_cwt_width_min.setValue(
                self._safe_float(
                    peaks_cfg.get("cwt_width_min"),
                    self.peaks_cwt_width_min.value(),
                )
            )
            self.peaks_cwt_width_max.setValue(
                self._safe_float(
                    peaks_cfg.get("cwt_width_max"),
                    self.peaks_cwt_width_max.value(),
                )
            )
            self.peaks_cwt_width_step.setValue(
                self._safe_float(
                    peaks_cfg.get("cwt_width_step"),
                    self.peaks_cwt_width_step.value(),
                )
            )
            self.peaks_cwt_cluster_tolerance.setValue(
                self._safe_float(
                    peaks_cfg.get("cwt_cluster_tolerance"),
                    self.peaks_cwt_cluster_tolerance.value(),
                )
            )

            baseline_cfg = (
                params.get("baseline", {}) if isinstance(params.get("baseline"), dict) else {}
            )
            method = str(baseline_cfg.get("method") or "").strip().lower()
            if method not in {"asls", "rubberband", "snip"}:
                method = ""
            self.baseline_enable.setChecked(bool(method))
            method_index = self.baseline_method.findData(
                method or "asls", QtCore.Qt.ItemDataRole.UserRole
            )
            if method_index < 0:
                method_index = 0
            self.baseline_method.setCurrentIndex(method_index)
            self.baseline_lambda.setValue(
                self._safe_float(
                    baseline_cfg.get("lam", baseline_cfg.get("lambda")),
                    self._baseline_default_lambda,
                )
            )
            self.baseline_p.setValue(
                self._safe_float(
                    baseline_cfg.get("p"),
                    self._baseline_default_p,
                )
            )
            self.baseline_niter.setValue(
                self._safe_int(
                    baseline_cfg.get("niter"),
                    self._baseline_default_niter,
                )
            )
            self.baseline_iterations.setValue(
                self._safe_int(
                    baseline_cfg.get("iterations"),
                    self._baseline_default_iterations,
                )
            )
            anchor_cfg = (
                baseline_cfg.get("anchor")
                or baseline_cfg.get("anchor_windows")
                or baseline_cfg.get("anchors")
                or baseline_cfg.get("zeroing")
            )
            anchor_enabled, _ = self._load_baseline_anchor_windows(anchor_cfg)
            self.baseline_anchor_enable.setChecked(anchor_enabled)

            solvent_cfg = (
                params.get("solvent_subtraction", {})
                if isinstance(params.get("solvent_subtraction"), dict)
                else {}
            )
            self.solvent_enable.setChecked(bool(solvent_cfg.get("enabled", False)))
            reference_entry = solvent_cfg.get("reference_entry")
            if isinstance(reference_entry, dict):
                entry_id = str(reference_entry.get("id") or "").strip()
                if entry_id:
                    self._solvent_reference_entries[entry_id] = dict(reference_entry)
            reference_id = solvent_cfg.get("reference_id") or solvent_cfg.get("reference")
            reference_id = str(reference_id).strip() if reference_id else ""
            reference_ids = solvent_cfg.get("reference_ids")
            if isinstance(reference_ids, Sequence) and not isinstance(
                reference_ids, (str, bytes)
            ):
                reference_ids_list = [
                    str(ref).strip()
                    for ref in reference_ids
                    if str(ref).strip()
                ]
            else:
                reference_ids_list = []
            shift_cfg = (
                solvent_cfg.get("shift_compensation", {})
                if isinstance(solvent_cfg.get("shift_compensation"), dict)
                else {}
            )
            self.solvent_shift_enable.setChecked(
                bool(shift_cfg.get("enabled", True))
            )
            self.solvent_shift_min.setValue(
                self._safe_float(shift_cfg.get("min"), self.solvent_shift_min.value())
            )
            self.solvent_shift_max.setValue(
                self._safe_float(shift_cfg.get("max"), self.solvent_shift_max.value())
            )
            self.solvent_shift_step.setValue(
                self._safe_float(shift_cfg.get("step"), self.solvent_shift_step.value())
            )
            self.solvent_multi_reference_enable.setChecked(
                bool(solvent_cfg.get("multi_reference", False))
            )
            selection_metric = str(
                solvent_cfg.get("selection_metric") or "rmse"
            ).strip().lower()
            if selection_metric not in {"rmse", "pattern_correlation"}:
                selection_metric = "rmse"
            selection_metric_index = self.solvent_selection_metric.findData(
                selection_metric, QtCore.Qt.ItemDataRole.UserRole
            )
            if selection_metric_index < 0:
                selection_metric_index = 0
            self.solvent_selection_metric.setCurrentIndex(selection_metric_index)
            selection_region = solvent_cfg.get("selection_region")
            selection_min = None
            selection_max = None
            if isinstance(selection_region, Mapping):
                selection_min = selection_region.get("min", selection_region.get("min_cm"))
                selection_max = selection_region.get("max", selection_region.get("max_cm"))
            elif isinstance(selection_region, Sequence) and not isinstance(
                selection_region, (str, bytes)
            ):
                if len(selection_region) > 0:
                    selection_min = selection_region[0]
                if len(selection_region) > 1:
                    selection_max = selection_region[1]
            self.solvent_selection_region_min.setText(
                self._format_optional(selection_min)
            )
            self.solvent_selection_region_max.setText(
                self._format_optional(selection_max)
            )
            selection_weight = self._safe_float(
                solvent_cfg.get("selection_region_weight"),
                self.solvent_selection_weight_spin.value(),
            )
            self._set_solvent_selection_weight(selection_weight)
            ridge_alpha = solvent_cfg.get("ridge_alpha")
            self.solvent_ridge_enable.setChecked(ridge_alpha is not None)
            self.solvent_ridge_alpha.setValue(
                self._safe_float(
                    ridge_alpha, self.solvent_ridge_alpha.value()
                )
            )
            self.solvent_offset_enable.setChecked(
                bool(solvent_cfg.get("include_offset", False))
            )
            self._refresh_solvent_reference_widgets(
                selected_id=reference_id,
                selected_ids=reference_ids_list
                or ([reference_id] if reference_id else None),
            )

            despike_cfg = params.get("despike", {}) if isinstance(params.get("despike"), dict) else {}
            self.despike_enable.setChecked(bool(despike_cfg.get("enabled", False)))
            self.despike_window.setValue(
                self._safe_int(despike_cfg.get("window"), self.despike_window.value())
            )
            self.despike_zscore.setValue(
                self._safe_float(despike_cfg.get("zscore"), self.despike_zscore.value())
            )
            baseline_window = despike_cfg.get("baseline_window")
            if baseline_window in (None, ""):
                self.despike_baseline_window.setValue(
                    self.despike_baseline_window.minimum()
                )
            else:
                self.despike_baseline_window.setValue(
                    self._safe_int(
                        baseline_window, self.despike_baseline_window.minimum()
                    )
                )
            spread_window = despike_cfg.get("spread_window")
            if spread_window in (None, ""):
                self.despike_spread_window.setValue(
                    self.despike_spread_window.minimum()
                )
            else:
                self.despike_spread_window.setValue(
                    self._safe_int(
                        spread_window, self.despike_spread_window.minimum()
                    )
                )
            spread_method = str(despike_cfg.get("spread_method") or "mad").strip().lower()
            if spread_method not in {"mad", "std"}:
                spread_method = "mad"
            spread_index = self.despike_spread_method.findData(
                spread_method, QtCore.Qt.ItemDataRole.UserRole
            )
            if spread_index < 0:
                spread_index = 0
            self.despike_spread_method.setCurrentIndex(spread_index)
            self.despike_spread_epsilon.setValue(
                self._safe_float(
                    despike_cfg.get("spread_epsilon"),
                    self.despike_spread_epsilon.value(),
                )
            )
            self.despike_residual_floor.setValue(
                self._safe_float(
                    despike_cfg.get("residual_floor"),
                    self.despike_residual_floor.value(),
                )
            )
            self.despike_tail_decay_ratio.setValue(
                self._safe_float(
                    despike_cfg.get("tail_decay_ratio"),
                    self.despike_tail_decay_ratio.value(),
                )
            )
            self.despike_isolation_ratio.setValue(
                self._safe_float(
                    despike_cfg.get("isolation_ratio"),
                    self.despike_isolation_ratio.value(),
                )
            )
            self.despike_max_passes.setValue(
                self._safe_int(
                    despike_cfg.get("max_passes"),
                    self.despike_max_passes.value(),
                )
            )
            self.despike_leading_padding.setValue(
                self._safe_int(
                    despike_cfg.get("leading_padding"),
                    self.despike_leading_padding.value(),
                )
            )
            self.despike_trailing_padding.setValue(
                self._safe_int(
                    despike_cfg.get("trailing_padding"),
                    self.despike_trailing_padding.value(),
                )
            )
            self._load_despike_exclusions(despike_cfg.get("exclusions"))
            self.despike_noise_scale_multiplier.setValue(
                self._safe_float(
                    despike_cfg.get("noise_scale_multiplier"),
                    self.despike_noise_scale_multiplier.value(),
                )
            )
            rng_seed = despike_cfg.get("rng_seed")
            self.despike_rng_seed.setText(
                "" if rng_seed in (None, "") else str(rng_seed)
            )
            stitch_cfg = (
                params.get("stitch", {})
                if isinstance(params.get("stitch"), dict)
                else {}
            )
            self.stitch_enable.setChecked(bool(stitch_cfg.get("enabled", False)))
            stitch_points = (
                stitch_cfg.get("shoulder_points")
                if stitch_cfg.get("shoulder_points") not in (None, "")
                else stitch_cfg.get("points", stitch_cfg.get("shoulders"))
            )
            self.stitch_shoulder_points.setValue(
                self._safe_int(
                    stitch_points, self.stitch_shoulder_points.value()
                )
            )
            method_value = (
                stitch_cfg.get("method")
                if stitch_cfg.get("method") not in (None, "")
                else stitch_cfg.get("mode", stitch_cfg.get("interpolation"))
            )
            method_token = (
                str(method_value).strip().lower()
                if method_value not in (None, "")
                else "linear"
            )
            method_index = self.stitch_method.findData(
                method_token, QtCore.Qt.ItemDataRole.UserRole
            )
            if method_index < 0:
                self.stitch_method.addItem(method_token, method_token)
                method_index = self.stitch_method.findData(
                    method_token, QtCore.Qt.ItemDataRole.UserRole
                )
            if method_index >= 0:
                self.stitch_method.setCurrentIndex(method_index)
            fallback_value = stitch_cfg.get("fallback_policy")
            if fallback_value in (None, ""):
                fallback_value = stitch_cfg.get("fallback")
            fallback_token = (
                str(fallback_value).strip().lower() if fallback_value else "preserve"
            )
            fallback_index = self.stitch_fallback_policy.findData(
                fallback_token, QtCore.Qt.ItemDataRole.UserRole
            )
            if fallback_index < 0 and fallback_token:
                self.stitch_fallback_policy.addItem(
                    fallback_token, fallback_token
                )
                fallback_index = self.stitch_fallback_policy.findData(
                    fallback_token, QtCore.Qt.ItemDataRole.UserRole
                )
            if fallback_index >= 0:
                self.stitch_fallback_policy.setCurrentIndex(fallback_index)
            self._load_stitch_windows(stitch_cfg.get("windows"))
            join_cfg = params.get("join", {}) if isinstance(params.get("join"), dict) else {}
            self.join_enable.setChecked(bool(join_cfg.get("enabled", False)))
            self.join_window.setValue(
                self._safe_int(join_cfg.get("window"), self.join_window.value())
            )
            self.join_threshold.setText(self._format_optional(join_cfg.get("threshold")))
            self._load_join_windows(join_cfg.get("windows"))

            blank_cfg = params.get("blank", {}) if isinstance(params.get("blank"), dict) else {}
            subtract = bool(blank_cfg.get("subtract", blank_cfg.get("enabled", False)))
            self.blank_subtract.setChecked(subtract)
            self.blank_require.setChecked(bool(blank_cfg.get("require", subtract)))
            fallback_value = blank_cfg.get("default", blank_cfg.get("fallback"))
            self.blank_fallback.setText(self._format_optional(fallback_value))
            match_strategy = str(blank_cfg.get("match_strategy") or "blank_id").strip().lower()
            match_strategy = match_strategy.replace("-", "_")
            if match_strategy not in {"blank_id", "cuvette_slot"}:
                match_strategy = "blank_id"
            index = self.blank_match_strategy.findData(
                match_strategy, QtCore.Qt.ItemDataRole.UserRole
            )
            if index < 0:
                self.blank_match_strategy.addItem(match_strategy, match_strategy)
                index = self.blank_match_strategy.findData(
                    match_strategy, QtCore.Qt.ItemDataRole.UserRole
                )
            if index >= 0:
                self.blank_match_strategy.setCurrentIndex(index)

            qc_cfg = params.get("qc", {}) if isinstance(params.get("qc"), dict) else {}
            drift_cfg = qc_cfg.get("drift", {}) if isinstance(qc_cfg.get("drift"), dict) else {}
            self.drift_enable.setChecked(bool(drift_cfg.get("enabled", False)))
            window = drift_cfg.get("window", {}) if isinstance(drift_cfg.get("window"), dict) else {}
            self.drift_window_min.setText(self._format_optional(window.get("min")))
            self.drift_window_max.setText(self._format_optional(window.get("max")))
            self.drift_max_slope.setText(self._format_optional(drift_cfg.get("max_slope_per_hour")))
            self.drift_max_delta.setText(self._format_optional(drift_cfg.get("max_delta")))
            self.drift_max_residual.setText(self._format_optional(drift_cfg.get("max_residual")))

        self._sync_recipe_from_ui()
        self._update_baseline_controls_enabled()
        self._update_solvent_controls_enabled()
        self._update_feature_controls_enabled()

    def _format_optional(self, value) -> str:
        if value is None:
            return ""
        return str(value)

    def _load_despike_exclusions(self, config) -> None:
        self.despike_exclusions_table.setRowCount(0)
        windows_iter: Sequence[object] | None = None
        if isinstance(config, Mapping):
            if "windows" in config:
                candidate = config.get("windows")
                if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                    windows_iter = candidate
            if windows_iter is None and any(
                key in config for key in ("min", "max", "min_nm", "max_nm", "lower", "upper", "lower_nm", "upper_nm")
            ):
                windows_iter = [config]
        elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            windows_iter = config

        if windows_iter is None:
            windows_iter = []

        for entry in windows_iter:
            if not isinstance(entry, Mapping):
                continue
            min_val = None
            max_val = None
            for key in ("min_nm", "lower_nm", "min", "lower"):
                if entry.get(key) is not None:
                    min_val = entry.get(key)
                    break
            for key in ("max_nm", "upper_nm", "max", "upper"):
                if entry.get(key) is not None:
                    max_val = entry.get(key)
                    break
            row = self.despike_exclusions_table.rowCount()
            self.despike_exclusions_table.insertRow(row)
            min_item = QTableWidgetItem(self._format_optional(min_val))
            max_item = QTableWidgetItem(self._format_optional(max_val))
            self.despike_exclusions_table.setItem(row, 0, min_item)
            self.despike_exclusions_table.setItem(row, 1, max_item)

        self._despike_exclusion_errors = []
        self._update_despike_exclusion_buttons()

    def _load_stitch_windows(self, config) -> None:
        try:
            from spectro_app.plugins.uvvis import pipeline as uvvis_pipeline
        except Exception:
            normalised: list[dict[str, object]] = []
        else:
            try:
                normalised = uvvis_pipeline.normalise_stitch_windows(config)
            except Exception:
                normalised = []

        self._stitch_windows_loading = True
        try:
            self.stitch_windows_table.setRowCount(0)
            self._stitch_windows_metadata = []
            for entry in normalised:
                lower_val = self._format_optional(entry.get("lower_nm"))
                upper_val = self._format_optional(entry.get("upper_nm"))
                extras = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"lower_nm", "upper_nm"}
                }
                self._insert_stitch_window_row(lower_val, upper_val, extras)
        finally:
            self._stitch_windows_loading = False
        self._stitch_windows_errors = []
        self._update_stitch_window_buttons()

    def _insert_stitch_window_row(
        self,
        lower_text: str = "",
        upper_text: str = "",
        extras: Mapping[str, object] | None = None,
    ) -> None:
        row_index = self.stitch_windows_table.rowCount()
        self.stitch_windows_table.insertRow(row_index)
        lower_field = self._create_stitch_window_field("Lower (nm)", lower_text)
        upper_field = self._create_stitch_window_field("Upper (nm)", upper_text)
        self.stitch_windows_table.setCellWidget(row_index, 0, lower_field)
        self.stitch_windows_table.setCellWidget(row_index, 1, upper_field)
        extras_map = dict(extras) if isinstance(extras, Mapping) else {}
        if row_index >= len(self._stitch_windows_metadata):
            self._stitch_windows_metadata.append(extras_map)
        else:
            self._stitch_windows_metadata.insert(row_index, extras_map)

    def _create_stitch_window_field(self, placeholder: str, text: str) -> QLineEdit:
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        field.setClearButtonEnabled(True)
        validator = QDoubleValidator(0.0, 10000.0, 3, field)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        field.setValidator(validator)
        if text:
            field.setText(text)
        field.textChanged.connect(self._on_stitch_window_value_changed)
        return field

    def _on_stitch_window_value_changed(self, *_):
        if self._stitch_windows_loading:
            return
        self._update_model_from_ui()

    def _on_stitch_windows_add_row(self):
        self._insert_stitch_window_row()
        row = self.stitch_windows_table.rowCount() - 1
        if row >= 0:
            self.stitch_windows_table.selectRow(row)
            widget = self.stitch_windows_table.cellWidget(row, 0)
            if isinstance(widget, QLineEdit):
                widget.setFocus()
                widget.selectAll()
        self._update_stitch_window_buttons()
        self._update_model_from_ui()

    def _on_stitch_windows_remove_row(self):
        if self.stitch_windows_table.rowCount() <= 0:
            return
        selection = self.stitch_windows_table.selectionModel()
        if selection and selection.hasSelection():
            row = selection.selectedRows()[0].row()
        else:
            row = self.stitch_windows_table.rowCount() - 1
        if row < 0:
            return
        self.stitch_windows_table.removeRow(row)
        if row < len(self._stitch_windows_metadata):
            self._stitch_windows_metadata.pop(row)
        self._update_stitch_window_buttons()
        self._update_model_from_ui()

    def _update_stitch_window_buttons(self) -> None:
        stitch_enabled = self.stitch_enable.isChecked()
        row_count = self.stitch_windows_table.rowCount()
        has_rows = row_count > 0
        self.stitch_windows_add_row.setEnabled(stitch_enabled)
        self.stitch_windows_remove_row.setEnabled(stitch_enabled and has_rows)

    def _load_join_windows(self, config) -> None:
        store: dict[str, list[dict[str, str]]] = {
            self._JOIN_WINDOWS_GLOBAL_KEY: []
        }
        if isinstance(config, Mapping):
            for key, value in config.items():
                if key is None:
                    continue
                key_str = str(key).strip()
                if not key_str:
                    continue
                target_key = (
                    self._JOIN_WINDOWS_GLOBAL_KEY
                    if key_str.lower() == "default"
                    else key_str
                )
                rows = self._normalise_join_window_rows(value)
                if target_key in store:
                    store[target_key] = rows
                else:
                    store[target_key] = rows
        elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            rows = self._normalise_join_window_rows(config)
            store[self._JOIN_WINDOWS_GLOBAL_KEY] = rows
        self._join_windows_store = store
        self._join_windows_active_key = self._JOIN_WINDOWS_GLOBAL_KEY
        self._join_windows_errors = []
        self._refresh_join_windows_combo()

    def _normalise_join_window_rows(self, value) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return rows
        for entry in value:
            if not isinstance(entry, Mapping):
                continue
            spikes_value = entry.get("spikes")
            spikes_text = self._format_optional(spikes_value)
            spikes_text = spikes_text if spikes_text else "1"
            rows.append(
                {
                    "min": self._format_optional(entry.get("min_nm")),
                    "max": self._format_optional(entry.get("max_nm")),
                    "spikes": spikes_text,
                }
            )
        return rows

    def _on_despike_exclusions_add_row(self) -> None:
        row = self.despike_exclusions_table.rowCount()
        self.despike_exclusions_table.insertRow(row)
        for column in range(2):
            item = QTableWidgetItem("")
            self.despike_exclusions_table.setItem(row, column, item)
        self.despike_exclusions_table.setCurrentCell(row, 0)
        self._update_despike_exclusion_buttons()
        self._update_model_from_ui()

    def _on_despike_exclusions_remove_row(self) -> None:
        selection = self.despike_exclusions_table.selectionModel()
        if selection and selection.hasSelection():
            row = selection.selectedRows()[0].row()
        else:
            row = self.despike_exclusions_table.rowCount() - 1
        if row < 0:
            return
        self.despike_exclusions_table.removeRow(row)
        self._update_despike_exclusion_buttons()
        self._update_model_from_ui()

    def _on_despike_exclusions_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        self._update_model_from_ui()

    def _update_despike_exclusion_buttons(self) -> None:
        enabled = self.despike_enable.isChecked()
        row_count = self.despike_exclusions_table.rowCount()
        self.despike_exclusions_add_row.setEnabled(enabled)
        self.despike_exclusions_remove_row.setEnabled(enabled and row_count > 0)

    def _on_baseline_enable_toggled(self, checked: bool) -> None:
        self._update_baseline_controls_enabled()
        self._update_model_from_ui()

    def _on_baseline_method_changed(self) -> None:
        self._update_baseline_controls_enabled()
        self._update_model_from_ui()

    def _on_baseline_anchor_enable_toggled(self, checked: bool) -> None:
        self._update_baseline_controls_enabled()
        self._update_model_from_ui()

    def _on_baseline_anchor_add_row(self) -> None:
        row = self.baseline_anchor_table.rowCount()
        self.baseline_anchor_table.insertRow(row)
        for column, default in enumerate(("", "", "0", "")):
            item = QTableWidgetItem(default)
            self.baseline_anchor_table.setItem(row, column, item)
        self.baseline_anchor_table.setCurrentCell(row, 0)
        self._update_baseline_anchor_buttons()
        self._update_model_from_ui()

    def _on_baseline_anchor_remove_row(self) -> None:
        selection = self.baseline_anchor_table.selectionModel()
        if selection and selection.hasSelection():
            row = selection.selectedRows()[0].row()
        else:
            row = self.baseline_anchor_table.rowCount() - 1
        if row < 0:
            return
        self.baseline_anchor_table.removeRow(row)
        self._update_baseline_anchor_buttons()
        self._update_model_from_ui()

    def _on_baseline_anchor_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        self._update_model_from_ui()

    def _update_baseline_controls_enabled(self) -> None:
        enabled = self.baseline_enable.isChecked()
        method = self.baseline_method.currentData(QtCore.Qt.ItemDataRole.UserRole)
        method = str(method).strip().lower() if isinstance(method, str) else ""
        asls_active = enabled and method == "asls"
        snip_active = enabled and method == "snip"

        self.baseline_method.setEnabled(enabled)
        self.baseline_lambda.setEnabled(asls_active)
        self.baseline_p.setEnabled(asls_active)
        self.baseline_niter.setEnabled(asls_active)
        self.baseline_iterations.setEnabled(snip_active)

        anchor_enabled = enabled and self.baseline_anchor_enable.isChecked()
        self.baseline_anchor_enable.setEnabled(enabled)
        self.baseline_anchor_table.setEnabled(anchor_enabled)
        self.baseline_anchor_add_row.setEnabled(anchor_enabled)
        self.baseline_anchor_remove_row.setEnabled(
            anchor_enabled and self.baseline_anchor_table.rowCount() > 0
        )

    def _update_baseline_anchor_buttons(self) -> None:
        anchor_enabled = (
            self.baseline_enable.isChecked()
            and self.baseline_anchor_enable.isChecked()
        )
        row_count = self.baseline_anchor_table.rowCount()
        self.baseline_anchor_remove_row.setEnabled(anchor_enabled and row_count > 0)

    def _update_feature_controls_enabled(self) -> None:
        smoothing_enabled = self.smooth_enable.isChecked()
        self.smooth_window.setEnabled(smoothing_enabled)
        self.smooth_poly.setEnabled(smoothing_enabled)

        peaks_enabled = self.peaks_enable.isChecked()
        self.peaks_prominence.setEnabled(peaks_enabled)
        self.peaks_min_distance.setEnabled(peaks_enabled)
        self.peaks_min_absorbance_threshold.setEnabled(peaks_enabled)
        self.peaks_noise_sigma_multiplier.setEnabled(peaks_enabled)
        self.peaks_noise_window_cm.setEnabled(peaks_enabled)
        self.peaks_min_prominence_by_region.setEnabled(peaks_enabled)
        self.peaks_min_distance_mode.setEnabled(peaks_enabled)
        self.peaks_min_distance_fwhm_fraction.setEnabled(peaks_enabled)
        self.peaks_peak_width_min.setEnabled(peaks_enabled)
        self.peaks_peak_width_max.setEnabled(peaks_enabled)
        self.peaks_max_peak_candidates.setEnabled(peaks_enabled)
        self.peaks_detect_negative.setEnabled(peaks_enabled)
        self.peaks_merge_tolerance.setEnabled(peaks_enabled)
        self.peaks_close_peak_tolerance.setEnabled(peaks_enabled)
        self.peaks_plateau_min_points.setEnabled(peaks_enabled)
        self.peaks_plateau_prominence_factor.setEnabled(peaks_enabled)
        self.peaks_shoulder_min_distance.setEnabled(peaks_enabled)
        self.peaks_shoulder_merge_tolerance.setEnabled(peaks_enabled)
        self.peaks_shoulder_curvature_prominence_factor.setEnabled(peaks_enabled)
        self.peaks_cwt_enabled.setEnabled(peaks_enabled)
        self.peaks_cwt_width_min.setEnabled(peaks_enabled)
        self.peaks_cwt_width_max.setEnabled(peaks_enabled)
        self.peaks_cwt_width_step.setEnabled(peaks_enabled)
        self.peaks_cwt_cluster_tolerance.setEnabled(peaks_enabled)

        stitch_enabled = self.stitch_enable.isChecked()
        self.stitch_shoulder_points.setEnabled(stitch_enabled)
        self.stitch_method.setEnabled(stitch_enabled)
        self.stitch_fallback_policy.setEnabled(stitch_enabled)
        self.stitch_windows_table.setEnabled(stitch_enabled)

        despike_enabled = self.despike_enable.isChecked()
        self.despike_window.setEnabled(despike_enabled)
        self.despike_zscore.setEnabled(despike_enabled)
        self.despike_baseline_window.setEnabled(despike_enabled)
        self.despike_spread_window.setEnabled(despike_enabled)
        self.despike_spread_method.setEnabled(despike_enabled)
        self.despike_spread_epsilon.setEnabled(despike_enabled)
        self.despike_residual_floor.setEnabled(despike_enabled)
        self.despike_tail_decay_ratio.setEnabled(despike_enabled)
        self.despike_isolation_ratio.setEnabled(despike_enabled)
        self.despike_max_passes.setEnabled(despike_enabled)
        self.despike_leading_padding.setEnabled(despike_enabled)
        self.despike_trailing_padding.setEnabled(despike_enabled)
        self.despike_noise_scale_multiplier.setEnabled(despike_enabled)
        self.despike_rng_seed.setEnabled(despike_enabled)
        for group in getattr(self, "_despike_control_groups", ()):
            group.setEnabled(despike_enabled)
        self._update_despike_exclusion_buttons()

        join_enabled = self.join_enable.isChecked()
        self.join_window.setEnabled(join_enabled)
        self.join_threshold.setEnabled(join_enabled)
        self.join_windows_instrument_combo.setEnabled(join_enabled)
        self.join_windows_table.setEnabled(join_enabled)
        self.join_windows_add_instrument.setEnabled(join_enabled)
        self.join_windows_add_row.setEnabled(join_enabled)

        blank_enabled = self.blank_subtract.isChecked()
        self.blank_require.setEnabled(blank_enabled)
        self.blank_fallback.setEnabled(blank_enabled)
        self.blank_match_strategy.setEnabled(blank_enabled)

        drift_enabled = self.drift_enable.isChecked()
        self.drift_window_min.setEnabled(drift_enabled)
        self.drift_window_max.setEnabled(drift_enabled)
        self.drift_max_slope.setEnabled(drift_enabled)
        self.drift_max_delta.setEnabled(drift_enabled)
        self.drift_max_residual.setEnabled(drift_enabled)

        self._update_join_window_buttons()
        self._update_stitch_window_buttons()

    def _update_solvent_controls_enabled(self) -> None:
        solvent_enabled = self.solvent_enable.isChecked()
        self.solvent_reference_combo.setEnabled(solvent_enabled)
        self.solvent_reference_load.setEnabled(solvent_enabled)
        self.solvent_shift_enable.setEnabled(solvent_enabled)
        shift_enabled = solvent_enabled and self.solvent_shift_enable.isChecked()
        self.solvent_shift_min.setEnabled(shift_enabled)
        self.solvent_shift_max.setEnabled(shift_enabled)
        self.solvent_shift_step.setEnabled(shift_enabled)
        self.solvent_multi_reference_enable.setEnabled(solvent_enabled)
        multi_enabled = (
            solvent_enabled and self.solvent_multi_reference_enable.isChecked()
        )
        self.solvent_selection_metric.setEnabled(multi_enabled)
        self.solvent_selection_region_min.setEnabled(multi_enabled)
        self.solvent_selection_region_max.setEnabled(multi_enabled)
        self.solvent_selection_weight_slider.setEnabled(multi_enabled)
        self.solvent_selection_weight_spin.setEnabled(multi_enabled)
        self.solvent_reference_list.setEnabled(multi_enabled)
        self.solvent_reference_list.setVisible(multi_enabled)
        if hasattr(self, "_solvent_form"):
            try:
                self._solvent_form.setRowVisible(
                    self._solvent_reference_list_row, multi_enabled
                )
            except AttributeError:
                pass
        self.solvent_ridge_enable.setEnabled(solvent_enabled)
        ridge_enabled = (
            solvent_enabled and self.solvent_ridge_enable.isChecked()
        )
        self.solvent_ridge_alpha.setEnabled(ridge_enabled)
        self.solvent_offset_enable.setEnabled(solvent_enabled)

    def _set_solvent_selection_weight(self, value: float) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        slider_value = int(round(clamped * 100))
        spin_blocked = self.solvent_selection_weight_spin.blockSignals(True)
        slider_blocked = self.solvent_selection_weight_slider.blockSignals(True)
        try:
            self.solvent_selection_weight_spin.setValue(clamped)
            self.solvent_selection_weight_slider.setValue(slider_value)
        finally:
            self.solvent_selection_weight_spin.blockSignals(spin_blocked)
            self.solvent_selection_weight_slider.blockSignals(slider_blocked)

    def _on_solvent_weight_slider_changed(self, value: int) -> None:
        self.solvent_selection_weight_spin.setValue(value / 100.0)

    def _on_solvent_weight_spin_changed(self, value: float) -> None:
        slider_value = int(round(max(0.0, min(1.0, float(value))) * 100))
        slider_blocked = self.solvent_selection_weight_slider.blockSignals(True)
        try:
            self.solvent_selection_weight_slider.setValue(slider_value)
        finally:
            self.solvent_selection_weight_slider.blockSignals(slider_blocked)

    def _current_solvent_reference_id(self) -> str | None:
        current = self.solvent_reference_combo.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if isinstance(current, str) and current.strip():
            return current.strip()
        return None

    def _selected_solvent_reference_ids(self) -> list[str]:
        selected: list[str] = []
        for item in self.solvent_reference_list.selectedItems():
            entry_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry_id, str) and entry_id.strip():
                selected.append(entry_id.strip())
        return selected

    def _refresh_solvent_reference_widgets(
        self,
        *,
        selected_id: str | None = None,
        selected_ids: Sequence[str] | None = None,
    ) -> None:
        entries = list(self._solvent_reference_entries.values())
        entries.sort(
            key=lambda entry: (not entry.get("defaults", False), entry.get("name", ""))
        )
        current_id = selected_id or self._current_solvent_reference_id()
        self.solvent_reference_combo.blockSignals(True)
        try:
            self.solvent_reference_combo.clear()
            self.solvent_reference_combo.addItem("None", None)
            for entry in entries:
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    continue
                name = str(entry.get("name") or entry_id).strip()
                label = name or entry_id
                if entry.get("defaults"):
                    label = f"{label} (default)"
                self.solvent_reference_combo.addItem(label, entry_id)
            if current_id:
                index = self.solvent_reference_combo.findData(
                    current_id, QtCore.Qt.ItemDataRole.UserRole
                )
                if index >= 0:
                    self.solvent_reference_combo.setCurrentIndex(index)
        finally:
            self.solvent_reference_combo.blockSignals(False)

        selected_set = {
            str(value).strip()
            for value in (selected_ids or [])
            if str(value).strip()
        }
        self.solvent_reference_list.blockSignals(True)
        try:
            self.solvent_reference_list.clear()
            for entry in entries:
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    continue
                name = str(entry.get("name") or entry_id).strip()
                label = name or entry_id
                if entry.get("defaults"):
                    label = f"{label} (default)"
                item = QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, entry_id)
                if entry_id in selected_set:
                    item.setSelected(True)
                self.solvent_reference_list.addItem(item)
        finally:
            self.solvent_reference_list.blockSignals(False)

    def _update_solvent_section_visibility(self, module_id: str) -> None:
        is_ftir = str(module_id or "").strip().lower() == "ftir"
        self.solvent_section.setVisible(is_ftir)

    def _refresh_join_windows_combo(self) -> None:
        self._join_windows_loading = True
        try:
            if self._JOIN_WINDOWS_GLOBAL_KEY not in self._join_windows_store:
                self._join_windows_store[self._JOIN_WINDOWS_GLOBAL_KEY] = []
            self.join_windows_instrument_combo.blockSignals(True)
            self.join_windows_instrument_combo.clear()
            self.join_windows_instrument_combo.addItem(
                "Global (all instruments)", self._JOIN_WINDOWS_GLOBAL_KEY
            )
            for key in self._join_windows_store.keys():
                if key == self._JOIN_WINDOWS_GLOBAL_KEY:
                    continue
                self.join_windows_instrument_combo.addItem(key, key)
            current_key = (
                self._join_windows_active_key
                if self._join_windows_active_key in self._join_windows_store
                else self._JOIN_WINDOWS_GLOBAL_KEY
            )
            index = self.join_windows_instrument_combo.findData(
                current_key, QtCore.Qt.ItemDataRole.UserRole
            )
            if index < 0:
                index = 0
            self.join_windows_instrument_combo.setCurrentIndex(index)
        finally:
            self.join_windows_instrument_combo.blockSignals(False)
            self._join_windows_loading = False
        current_data = self.join_windows_instrument_combo.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if not isinstance(current_data, str):
            current_data = self._JOIN_WINDOWS_GLOBAL_KEY
        self._join_windows_active_key = current_data
        self._populate_join_windows_table(current_data)
        self._update_join_window_buttons()

    def _populate_join_windows_table(self, key: str) -> None:
        self._join_windows_loading = True
        try:
            self.join_windows_table.setRowCount(0)
            rows = self._join_windows_store.get(key, [])
            for row in rows:
                self._insert_join_window_row(
                    row.get("min", ""),
                    row.get("max", ""),
                    row.get("spikes", ""),
                )
        finally:
            self._join_windows_loading = False
        self._update_join_window_buttons()

    def _insert_join_window_row(
        self, min_text: str = "", max_text: str = "", spikes_value: str | int = ""
    ) -> None:
        row_index = self.join_windows_table.rowCount()
        self.join_windows_table.insertRow(row_index)
        min_field = self._create_join_window_field("Min (nm)", min_text)
        max_field = self._create_join_window_field("Max (nm)", max_text)
        spikes_field = self._create_join_window_spikes_field(spikes_value)
        self.join_windows_table.setCellWidget(row_index, 0, min_field)
        self.join_windows_table.setCellWidget(row_index, 1, max_field)
        self.join_windows_table.setCellWidget(row_index, 2, spikes_field)

    def _create_join_window_field(self, placeholder: str, text: str) -> QLineEdit:
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        field.setClearButtonEnabled(True)
        validator = QDoubleValidator(0.0, 10000.0, 3, field)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        field.setValidator(validator)
        if text:
            field.setText(text)
        field.textChanged.connect(self._on_join_window_value_changed)
        return field

    def _create_join_window_spikes_field(
        self, value: str | int | float | None
    ) -> QSpinBox:
        field = QSpinBox()
        field.setRange(1, 999)
        if isinstance(value, (int, float)):
            int_value = int(value)
        else:
            text_value = str(value).strip() if value not in (None, "") else ""
            if text_value:
                try:
                    int_value = int(text_value)
                except ValueError:
                    int_value = 1
            else:
                int_value = 1
        if int_value < 1:
            int_value = 1
        field.setValue(int_value)
        field.valueChanged.connect(self._on_join_window_spikes_changed)
        return field

    def _on_join_window_value_changed(self, *_):
        if self._join_windows_loading:
            return
        self._save_current_join_windows()
        self._update_model_from_ui()

    def _on_join_window_spikes_changed(self, *_):
        if self._join_windows_loading:
            return
        self._save_current_join_windows()
        self._update_model_from_ui()

    def _save_current_join_windows(self) -> None:
        key = self._join_windows_active_key
        if not isinstance(key, str):
            key = self._JOIN_WINDOWS_GLOBAL_KEY
        rows: list[dict[str, str]] = []
        for index in range(self.join_windows_table.rowCount()):
            min_widget = self.join_windows_table.cellWidget(index, 0)
            max_widget = self.join_windows_table.cellWidget(index, 1)
            spikes_widget = self.join_windows_table.cellWidget(index, 2)
            min_text = min_widget.text().strip() if isinstance(min_widget, QLineEdit) else ""
            max_text = max_widget.text().strip() if isinstance(max_widget, QLineEdit) else ""
            if isinstance(spikes_widget, QSpinBox):
                spikes_value = str(max(1, int(spikes_widget.value())))
            else:
                spikes_value = "1"
            rows.append({"min": min_text, "max": max_text, "spikes": spikes_value})
        self._join_windows_store[key] = rows

    def _on_join_windows_instrument_changed(self):
        if self._join_windows_loading:
            return
        self._save_current_join_windows()
        new_key = self.join_windows_instrument_combo.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if not isinstance(new_key, str):
            new_key = self._JOIN_WINDOWS_GLOBAL_KEY
        self._join_windows_active_key = new_key
        if new_key not in self._join_windows_store:
            self._join_windows_store[new_key] = []
        self._populate_join_windows_table(new_key)
        self._update_model_from_ui()

    def _on_join_windows_add_instrument(self):
        self._save_current_join_windows()
        name, ok = QInputDialog.getText(
            self,
            "Add Instrument Profile",
            "Instrument name (e.g. model, serial fragment, or 'default'):",
        )
        if not ok:
            return
        key = name.strip()
        if not key:
            return
        if key.lower() == "default":
            index = self.join_windows_instrument_combo.findData(
                self._JOIN_WINDOWS_GLOBAL_KEY, QtCore.Qt.ItemDataRole.UserRole
            )
            if index >= 0:
                self.join_windows_instrument_combo.setCurrentIndex(index)
            return
        existing = {
            (stored_key or "").strip().lower(): stored_key
            for stored_key in self._join_windows_store.keys()
        }
        lookup = key.strip().lower()
        if lookup in existing:
            index = self.join_windows_instrument_combo.findData(
                existing[lookup], QtCore.Qt.ItemDataRole.UserRole
            )
            if index >= 0:
                self.join_windows_instrument_combo.setCurrentIndex(index)
            return
        self._join_windows_store[key] = []
        self._join_windows_active_key = key
        self._refresh_join_windows_combo()
        index = self.join_windows_instrument_combo.findData(
            key, QtCore.Qt.ItemDataRole.UserRole
        )
        if index >= 0:
            self.join_windows_instrument_combo.setCurrentIndex(index)
        self._update_model_from_ui()

    def _on_join_windows_remove_instrument(self):
        key = self.join_windows_instrument_combo.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if not isinstance(key, str) or key == self._JOIN_WINDOWS_GLOBAL_KEY:
            return
        self._join_windows_store.pop(key, None)
        self._join_windows_active_key = self._JOIN_WINDOWS_GLOBAL_KEY
        self._refresh_join_windows_combo()
        index = self.join_windows_instrument_combo.findData(
            self._JOIN_WINDOWS_GLOBAL_KEY, QtCore.Qt.ItemDataRole.UserRole
        )
        if index >= 0:
            self.join_windows_instrument_combo.setCurrentIndex(index)
        self._update_model_from_ui()

    def _on_join_windows_add_row(self):
        self._insert_join_window_row()
        row = self.join_windows_table.rowCount() - 1
        if row >= 0:
            self.join_windows_table.selectRow(row)
            widget = self.join_windows_table.cellWidget(row, 0)
            if isinstance(widget, QLineEdit):
                widget.setFocus()
                widget.selectAll()
        self._save_current_join_windows()
        self._update_join_window_buttons()
        self._update_model_from_ui()

    def _on_join_windows_remove_row(self):
        selection = self.join_windows_table.selectionModel()
        if selection and selection.hasSelection():
            row = selection.selectedRows()[0].row()
        else:
            row = self.join_windows_table.rowCount() - 1
        if row < 0:
            return
        self.join_windows_table.removeRow(row)
        self._save_current_join_windows()
        self._update_join_window_buttons()
        self._update_model_from_ui()

    def _update_join_window_buttons(self) -> None:
        join_enabled = self.join_enable.isChecked()
        has_rows = self.join_windows_table.rowCount() > 0
        self.join_windows_add_row.setEnabled(join_enabled)
        self.join_windows_remove_row.setEnabled(join_enabled and has_rows)
        removable = (
            self.join_windows_instrument_combo.count() > 1
            and self.join_windows_instrument_combo.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
            != self._JOIN_WINDOWS_GLOBAL_KEY
        )
        self.join_windows_add_instrument.setEnabled(join_enabled)
        self.join_windows_remove_instrument.setEnabled(join_enabled and removable)

    def _build_despike_exclusions_payload(self) -> tuple[object | None, list[str]]:
        errors: list[str] = []
        windows: list[dict[str, float]] = []
        for row in range(self.despike_exclusions_table.rowCount()):
            min_item = self.despike_exclusions_table.item(row, 0)
            max_item = self.despike_exclusions_table.item(row, 1)
            min_text = min_item.text().strip() if isinstance(min_item, QTableWidgetItem) else ""
            max_text = max_item.text().strip() if isinstance(max_item, QTableWidgetItem) else ""
            if not min_text and not max_text:
                continue
            min_value: float | None = None
            max_value: float | None = None
            if min_text:
                try:
                    min_value = float(min_text)
                except ValueError:
                    errors.append(
                        f"Despike exclusion row {row + 1} minimum must be numeric"
                    )
                    continue
            if max_text:
                try:
                    max_value = float(max_text)
                except ValueError:
                    errors.append(
                        f"Despike exclusion row {row + 1} maximum must be numeric"
                    )
                    continue
            if min_value is not None and max_value is not None and min_value > max_value:
                errors.append(
                    f"Despike exclusion row {row + 1} minimum must be ≤ maximum"
                )
                continue
            payload: dict[str, float] = {}
            if min_value is not None:
                payload["min_nm"] = float(min_value)
            if max_value is not None:
                payload["max_nm"] = float(max_value)
            windows.append(payload)

        if not windows:
            return (None, errors)
        return ({"windows": windows}, errors)

    def _build_stitch_windows_payload(self) -> tuple[object | None, list[str]]:
        errors: list[str] = []
        windows: list[dict[str, object]] = []
        for row in range(self.stitch_windows_table.rowCount()):
            lower_widget = self.stitch_windows_table.cellWidget(row, 0)
            upper_widget = self.stitch_windows_table.cellWidget(row, 1)
            lower_text = (
                lower_widget.text().strip()
                if isinstance(lower_widget, QLineEdit)
                else ""
            )
            upper_text = (
                upper_widget.text().strip()
                if isinstance(upper_widget, QLineEdit)
                else ""
            )
            if not lower_text and not upper_text:
                continue

            lower_value: float | None = None
            upper_value: float | None = None
            if lower_text:
                try:
                    lower_value = float(lower_text)
                except ValueError:
                    errors.append(
                        f"Stitch window row {row + 1} lower bound must be numeric"
                    )
                    continue
            if upper_text:
                try:
                    upper_value = float(upper_text)
                except ValueError:
                    errors.append(
                        f"Stitch window row {row + 1} upper bound must be numeric"
                    )
                    continue

            if lower_value is None and upper_value is None:
                continue

            if (
                lower_value is not None
                and upper_value is not None
                and lower_value > upper_value
            ):
                lower_value, upper_value = upper_value, lower_value

            extras = (
                dict(self._stitch_windows_metadata[row])
                if row < len(self._stitch_windows_metadata)
                else {}
            )
            extras.pop("lower_nm", None)
            extras.pop("upper_nm", None)
            payload: dict[str, object] = dict(extras)
            payload["lower_nm"] = lower_value
            payload["upper_nm"] = upper_value
            windows.append(payload)

        if not windows:
            return (None, errors)
        return (windows, errors)

    def _build_join_windows_payload(self) -> tuple[object | None, list[str]]:
        errors: list[str] = []
        cleaned: dict[str, list[dict[str, float | int]]] = {}
        for key, rows in self._join_windows_store.items():
            cleaned_rows: list[dict[str, float | int]] = []
            for index, row in enumerate(rows, start=1):
                min_text = str(row.get("min", "")).strip()
                max_text = str(row.get("max", "")).strip()
                spikes_text = str(row.get("spikes", "")).strip()
                if not min_text and not max_text:
                    continue
                if not min_text or not max_text:
                    label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                    errors.append(
                        f"Join window for {label} row {index} must include both min and max"
                    )
                    continue
                min_value = self._parse_optional_float(min_text)
                max_value = self._parse_optional_float(max_text)
                if not isinstance(min_value, (int, float)) or not isinstance(
                    max_value, (int, float)
                ):
                    label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                    errors.append(
                        f"Join window bounds for {label} row {index} must be numeric"
                    )
                    continue
                min_float = float(min_value)
                max_float = float(max_value)
                if min_float < 0.0 or max_float < 0.0:
                    label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                    errors.append(
                        f"Join window bounds for {label} row {index} must be positive"
                    )
                    continue
                if min_float > max_float:
                    label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                    errors.append(
                        f"Join window min must be ≤ max for {label} row {index}"
                    )
                    continue
                if not spikes_text:
                    spikes_int = 1
                else:
                    try:
                        spikes_int = int(spikes_text)
                    except (TypeError, ValueError):
                        label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                        errors.append(
                            f"Join window spikes for {label} row {index} must be a positive integer"
                        )
                        continue
                    if spikes_int < 1:
                        label = "Global" if key == self._JOIN_WINDOWS_GLOBAL_KEY else key
                        errors.append(
                            f"Join window spikes for {label} row {index} must be a positive integer"
                        )
                        continue
                cleaned_rows.append(
                    {"min_nm": min_float, "max_nm": max_float, "spikes": spikes_int}
                )
            if cleaned_rows:
                cleaned[key] = cleaned_rows
        global_rows = cleaned.get(self._JOIN_WINDOWS_GLOBAL_KEY, [])
        non_global_keys = [
            key
            for key in self._join_windows_store.keys()
            if key != self._JOIN_WINDOWS_GLOBAL_KEY and key in cleaned
        ]
        if non_global_keys:
            payload: dict[str, list[dict[str, float | int]]] = {}
            if global_rows:
                payload["default"] = global_rows
            for key in non_global_keys:
                payload[key] = cleaned[key]
            return payload or None, errors
        if global_rows:
            return global_rows, errors
        return None, errors

    def _load_baseline_anchor_windows(self, config) -> tuple[bool, bool]:
        windows_data: list[dict[str, object]] = []
        enabled = False
        if isinstance(config, Mapping):
            enabled = bool(config.get("enabled", True))
            source = config.get("windows")
        else:
            source = config
            enabled = bool(config)

        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            for entry in source:
                if isinstance(entry, Mapping):
                    windows_data.append(
                        {
                            "min": self._format_optional(
                                entry.get("min_nm")
                                or entry.get("lower_nm")
                                or entry.get("min")
                                or entry.get("lower")
                            ),
                            "max": self._format_optional(
                                entry.get("max_nm")
                                or entry.get("upper_nm")
                                or entry.get("max")
                                or entry.get("upper")
                            ),
                            "target": self._format_optional(
                                entry.get("target")
                                or entry.get("level")
                                or entry.get("value")
                            ),
                            "label": self._format_optional(entry.get("label")),
                        }
                    )
                elif isinstance(entry, Sequence) and len(entry) >= 2:
                    min_val = self._format_optional(entry[0])
                    max_val = self._format_optional(entry[1])
                    target_val = self._format_optional(entry[2]) if len(entry) > 2 else ""
                    label_val = self._format_optional(entry[3]) if len(entry) > 3 else ""
                    windows_data.append(
                        {
                            "min": min_val,
                            "max": max_val,
                            "target": target_val,
                            "label": label_val,
                        }
                    )

        self.baseline_anchor_table.blockSignals(True)
        try:
            self.baseline_anchor_table.setRowCount(0)
            for entry in windows_data:
                row = self.baseline_anchor_table.rowCount()
                self.baseline_anchor_table.insertRow(row)
                self.baseline_anchor_table.setItem(row, 0, QTableWidgetItem(entry.get("min", "")))
                self.baseline_anchor_table.setItem(row, 1, QTableWidgetItem(entry.get("max", "")))
                target_text = entry.get("target", "")
                if target_text in {None, "None"}:
                    target_text = ""
                self.baseline_anchor_table.setItem(row, 2, QTableWidgetItem(str(target_text)))
                label_text = entry.get("label", "")
                self.baseline_anchor_table.setItem(row, 3, QTableWidgetItem(str(label_text or "")))
        finally:
            self.baseline_anchor_table.blockSignals(False)
        self._update_baseline_anchor_buttons()
        return enabled, bool(windows_data)

    def _build_baseline_anchor_payload(self) -> tuple[object | None, list[str]]:
        errors: list[str] = []
        windows: list[dict[str, object]] = []
        for row in range(self.baseline_anchor_table.rowCount()):
            min_item = self.baseline_anchor_table.item(row, 0)
            max_item = self.baseline_anchor_table.item(row, 1)
            target_item = self.baseline_anchor_table.item(row, 2)
            label_item = self.baseline_anchor_table.item(row, 3)

            min_text = min_item.text().strip() if min_item else ""
            max_text = max_item.text().strip() if max_item else ""
            target_text = target_item.text().strip() if target_item else ""
            label_text = label_item.text().strip() if label_item else ""

            if not min_text and not max_text:
                continue
            if not min_text or not max_text:
                errors.append(
                    f"Baseline anchor row {row + 1} must include both min and max"
                )
                continue

            min_value = self._parse_optional_float(min_text)
            max_value = self._parse_optional_float(max_text)
            if not isinstance(min_value, (int, float)) or not isinstance(
                max_value, (int, float)
            ):
                errors.append(
                    f"Baseline anchor row {row + 1} bounds must be numeric"
                )
                continue

            if float(min_value) > float(max_value):
                errors.append(
                    f"Baseline anchor row {row + 1} min must be ≤ max"
                )
                continue

            if target_text:
                target_value = self._parse_optional_float(target_text)
                if not isinstance(target_value, (int, float)):
                    errors.append(
                        f"Baseline anchor row {row + 1} target must be numeric"
                    )
                    continue
                target_float: float = float(target_value)
            else:
                target_float = 0.0

            window_entry: dict[str, object] = {
                "min_nm": float(min_value),
                "max_nm": float(max_value),
                "target": target_float,
            }
            if label_text:
                window_entry["label"] = label_text
            windows.append(window_entry)

        if not windows:
            return None, errors

        payload = {"enabled": True, "windows": windows}
        return payload, errors

    def _sync_recipe_from_ui(self) -> None:
        with self._suspend_updates():
            self._update_model_from_ui(force=True)

    def _update_model_from_ui(self, *_, force: bool = False, run_validation: bool = True):
        if self._updating and not force:
            return

        params_source = self.recipe.params if isinstance(self.recipe.params, dict) else {}
        params = copy.deepcopy(params_source)

        baseline_cfg = self._ensure_dict(params, "baseline")
        method_data = self.baseline_method.currentData(QtCore.Qt.ItemDataRole.UserRole)
        method_value = str(method_data).strip().lower() if isinstance(method_data, str) else ""
        if self.baseline_enable.isChecked() and method_value:
            baseline_cfg["method"] = method_value
            baseline_cfg["lam"] = float(self.baseline_lambda.value())
            baseline_cfg.pop("lambda", None)
            baseline_cfg["p"] = float(self.baseline_p.value())
            baseline_cfg["niter"] = int(self.baseline_niter.value())
            baseline_cfg["iterations"] = int(self.baseline_iterations.value())
        else:
            baseline_cfg.pop("method", None)

        if self.baseline_enable.isChecked() and self.baseline_anchor_enable.isChecked():
            anchor_payload, anchor_errors = self._build_baseline_anchor_payload()
            self._baseline_anchor_errors = anchor_errors
            if anchor_payload is None:
                for key in ("anchor", "anchor_windows", "anchors", "zeroing"):
                    baseline_cfg.pop(key, None)
            else:
                baseline_cfg["anchor"] = anchor_payload
                for key in ("anchor_windows", "anchors", "zeroing"):
                    baseline_cfg.pop(key, None)
        else:
            self._baseline_anchor_errors = []
            for key in ("anchor", "anchor_windows", "anchors", "zeroing"):
                baseline_cfg.pop(key, None)

        module_id = self.module.currentData(QtCore.Qt.ItemDataRole.UserRole)
        module_id = str(module_id or "").strip().lower()
        if module_id == "ftir":
            solvent_cfg = self._ensure_dict(params, "solvent_subtraction")
            solvent_cfg["enabled"] = bool(self.solvent_enable.isChecked())
            reference_id = self._current_solvent_reference_id()
            if reference_id:
                solvent_cfg["reference_id"] = reference_id
            else:
                solvent_cfg.pop("reference_id", None)
                solvent_cfg.pop("reference_entry", None)
            entry = (
                self._solvent_reference_entries.get(reference_id)
                if reference_id
                else None
            )
            if isinstance(entry, dict):
                solvent_cfg["reference_entry"] = copy.deepcopy(entry)
            elif not reference_id:
                solvent_cfg.pop("reference_entry", None)
            multi_enabled = bool(self.solvent_multi_reference_enable.isChecked())
            solvent_cfg["multi_reference"] = multi_enabled
            if multi_enabled:
                reference_ids = self._selected_solvent_reference_ids()
                if reference_id and reference_id not in reference_ids:
                    reference_ids.insert(0, reference_id)
                if reference_ids:
                    solvent_cfg["reference_ids"] = reference_ids
                else:
                    solvent_cfg.pop("reference_ids", None)
            else:
                solvent_cfg.pop("reference_ids", None)
            selection_metric_data = self.solvent_selection_metric.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
            selection_metric_value = (
                str(selection_metric_data).strip().lower()
                if isinstance(selection_metric_data, str)
                else ""
            )
            if selection_metric_value:
                solvent_cfg["selection_metric"] = selection_metric_value
            else:
                solvent_cfg.pop("selection_metric", None)
            selection_region_min = self._parse_optional_float(
                self.solvent_selection_region_min.text()
            )
            selection_region_max = self._parse_optional_float(
                self.solvent_selection_region_max.text()
            )
            if selection_region_min is None and selection_region_max is None:
                solvent_cfg.pop("selection_region", None)
            else:
                solvent_cfg["selection_region"] = {
                    "min": selection_region_min,
                    "max": selection_region_max,
                }
            solvent_cfg["selection_region_weight"] = float(
                self.solvent_selection_weight_spin.value()
            )
            shift_cfg = self._ensure_dict(solvent_cfg, "shift_compensation")
            shift_cfg["enabled"] = bool(self.solvent_shift_enable.isChecked())
            shift_cfg["min"] = float(self.solvent_shift_min.value())
            shift_cfg["max"] = float(self.solvent_shift_max.value())
            shift_cfg["step"] = float(self.solvent_shift_step.value())
            if self.solvent_ridge_enable.isChecked():
                solvent_cfg["ridge_alpha"] = float(
                    self.solvent_ridge_alpha.value()
                )
            else:
                solvent_cfg.pop("ridge_alpha", None)
            solvent_cfg["include_offset"] = bool(
                self.solvent_offset_enable.isChecked()
            )

        smoothing_cfg = self._ensure_dict(params, "smoothing")
        smoothing_cfg.update(
            {
                "enabled": self.smooth_enable.isChecked(),
                "window": int(self.smooth_window.value()),
                "polyorder": int(self.smooth_poly.value()),
            }
        )

        features_cfg = self._ensure_dict(params, "features")
        peaks_enabled = self.peaks_enable.isChecked()
        if peaks_enabled:
            min_distance_mode_data = self.peaks_min_distance_mode.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
            min_distance_mode = (
                str(min_distance_mode_data).strip().lower()
                if isinstance(min_distance_mode_data, str)
                else "fixed"
            )
            if min_distance_mode not in {"fixed", "adaptive"}:
                min_distance_mode = "fixed"
            peak_payload: dict[str, object] = {
                "enabled": True,
                "prominence": float(self.peaks_prominence.value()),
                "min_absorbance_threshold": float(self.peaks_min_absorbance_threshold.value()),
                "noise_sigma_multiplier": float(self.peaks_noise_sigma_multiplier.value()),
                "noise_window_cm": float(self.peaks_noise_window_cm.value()),
                "min_distance": float(self.peaks_min_distance.value()),
                "min_distance_mode": min_distance_mode,
                "min_distance_fwhm_fraction": float(
                    self.peaks_min_distance_fwhm_fraction.value()
                ),
                "peak_width_min": float(self.peaks_peak_width_min.value()),
                "peak_width_max": float(self.peaks_peak_width_max.value()),
                "max_peak_candidates": int(self.peaks_max_peak_candidates.value()),
                "detect_negative_peaks": bool(self.peaks_detect_negative.isChecked()),
                "merge_tolerance": float(self.peaks_merge_tolerance.value()),
                "close_peak_tolerance_cm": float(self.peaks_close_peak_tolerance.value()),
                "plateau_min_points": int(self.peaks_plateau_min_points.value()),
                "plateau_prominence_factor": float(
                    self.peaks_plateau_prominence_factor.value()
                ),
                "shoulder_merge_tolerance_cm": float(
                    self.peaks_shoulder_merge_tolerance.value()
                ),
                "shoulder_curvature_prominence_factor": float(
                    self.peaks_shoulder_curvature_prominence_factor.value()
                ),
                "cwt_enabled": bool(self.peaks_cwt_enabled.isChecked()),
                "cwt_width_min": float(self.peaks_cwt_width_min.value()),
                "cwt_width_max": float(self.peaks_cwt_width_max.value()),
                "cwt_width_step": float(self.peaks_cwt_width_step.value()),
                "cwt_cluster_tolerance": float(self.peaks_cwt_cluster_tolerance.value()),
            }
            region_text = self.peaks_min_prominence_by_region.text().strip()
            peak_payload["min_prominence_by_region"] = region_text or None
            shoulder_min_distance = float(self.peaks_shoulder_min_distance.value())
            if shoulder_min_distance > 0:
                peak_payload["shoulder_min_distance_cm"] = shoulder_min_distance
            features_cfg["peaks"] = peak_payload
        else:
            features_cfg["peaks"] = {"enabled": False}

        if not features_cfg:
            params.pop("features", None)

        despike_cfg = self._ensure_dict(params, "despike")
        despike_enabled = self.despike_enable.isChecked()
        despike_cfg["enabled"] = despike_enabled
        if despike_enabled:
            despike_cfg["window"] = int(self.despike_window.value())
            despike_cfg["zscore"] = float(self.despike_zscore.value())
            baseline_window = int(self.despike_baseline_window.value())
            if baseline_window > 0:
                despike_cfg["baseline_window"] = baseline_window
            else:
                despike_cfg.pop("baseline_window", None)
            spread_window = int(self.despike_spread_window.value())
            if spread_window > 0:
                despike_cfg["spread_window"] = spread_window
            else:
                despike_cfg.pop("spread_window", None)
            spread_method_data = self.despike_spread_method.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
            spread_method = (
                str(spread_method_data).strip().lower()
                if isinstance(spread_method_data, str)
                else ""
            )
            if spread_method and spread_method != "mad":
                despike_cfg["spread_method"] = spread_method
            else:
                despike_cfg.pop("spread_method", None)
            despike_cfg["spread_epsilon"] = float(
                self.despike_spread_epsilon.value()
            )
            despike_cfg["residual_floor"] = float(
                self.despike_residual_floor.value()
            )
            despike_cfg["tail_decay_ratio"] = float(
                self.despike_tail_decay_ratio.value()
            )
            despike_cfg["isolation_ratio"] = float(
                self.despike_isolation_ratio.value()
            )
            despike_cfg["max_passes"] = int(self.despike_max_passes.value())
            despike_cfg["leading_padding"] = int(
                self.despike_leading_padding.value()
            )
            despike_cfg["trailing_padding"] = int(
                self.despike_trailing_padding.value()
            )
            despike_cfg["noise_scale_multiplier"] = float(
                self.despike_noise_scale_multiplier.value()
            )
            exclusions_payload, exclusion_errors = self._build_despike_exclusions_payload()
            self._despike_exclusion_errors = exclusion_errors
            if exclusions_payload is None:
                despike_cfg.pop("exclusions", None)
            else:
                despike_cfg["exclusions"] = exclusions_payload
            seed_text = self.despike_rng_seed.text().strip()
            if seed_text:
                try:
                    despike_cfg["rng_seed"] = int(seed_text)
                except ValueError:
                    despike_cfg["rng_seed"] = seed_text
            else:
                despike_cfg.pop("rng_seed", None)
        else:
            despike_cfg.pop("window", None)
            despike_cfg.pop("zscore", None)
            for key in (
                "baseline_window",
                "spread_window",
                "spread_method",
                "spread_epsilon",
                "residual_floor",
                "tail_decay_ratio",
                "isolation_ratio",
                "max_passes",
                "leading_padding",
                "trailing_padding",
                "noise_scale_multiplier",
                "rng_seed",
            ):
                despike_cfg.pop(key, None)
            despike_cfg.pop("exclusions", None)
            self._despike_exclusion_errors = []

        stitch_cfg = self._ensure_dict(params, "stitch")
        stitch_cfg["enabled"] = self.stitch_enable.isChecked()
        stitch_cfg["shoulder_points"] = int(self.stitch_shoulder_points.value())
        method_data = self.stitch_method.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        method_value = (
            str(method_data).strip().lower()
            if isinstance(method_data, str)
            else ""
        )
        if method_value:
            stitch_cfg["method"] = method_value
        else:
            stitch_cfg.pop("method", None)
        fallback_data = self.stitch_fallback_policy.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        fallback_value = (
            str(fallback_data).strip().lower()
            if isinstance(fallback_data, str)
            else ""
        )
        if fallback_value:
            stitch_cfg["fallback_policy"] = fallback_value
        else:
            stitch_cfg.pop("fallback_policy", None)
        windows_payload, stitch_window_errors = self._build_stitch_windows_payload()
        self._stitch_windows_errors = stitch_window_errors
        if windows_payload is None:
            stitch_cfg.pop("windows", None)
        else:
            stitch_cfg["windows"] = windows_payload

        join_cfg = self._ensure_dict(params, "join")
        threshold_text = self.join_threshold.text().strip()
        if threshold_text:
            try:
                threshold_value: object = float(threshold_text)
            except ValueError:
                threshold_value = threshold_text
        else:
            threshold_value = None
        join_cfg.update(
            {
                "enabled": self.join_enable.isChecked(),
                "window": int(self.join_window.value()),
            }
        )
        if threshold_value is None:
            join_cfg.pop("threshold", None)
        else:
            join_cfg["threshold"] = threshold_value
        self._save_current_join_windows()
        windows_payload, window_errors = self._build_join_windows_payload()
        self._join_windows_errors = window_errors
        if windows_payload is None:
            join_cfg.pop("windows", None)
        else:
            join_cfg["windows"] = windows_payload

        blank_cfg = self._ensure_dict(params, "blank")
        fallback_text = self.blank_fallback.text().strip()
        blank_cfg.update(
            {
                "subtract": self.blank_subtract.isChecked(),
                "require": self.blank_require.isChecked(),
            }
        )
        if fallback_text:
            blank_cfg["default"] = fallback_text
        else:
            blank_cfg.pop("default", None)
            blank_cfg.pop("fallback", None)
        match_data = self.blank_match_strategy.currentData(QtCore.Qt.ItemDataRole.UserRole)
        match_value = str(match_data).strip().lower().replace("-", "_") if match_data is not None else ""
        if match_value not in {"blank_id", "cuvette_slot"}:
            match_value = "blank_id"
        if match_value:
            blank_cfg["match_strategy"] = match_value
        else:
            blank_cfg.pop("match_strategy", None)

        qc_cfg = self._ensure_dict(params, "qc")
        drift_cfg = self._ensure_dict(qc_cfg, "drift")
        drift_cfg["enabled"] = self.drift_enable.isChecked()
        drift_cfg["window"] = {
            "min": self._parse_optional_float(self.drift_window_min.text()),
            "max": self._parse_optional_float(self.drift_window_max.text()),
        }
        drift_cfg["max_slope_per_hour"] = self._parse_optional_float(self.drift_max_slope.text())
        drift_cfg["max_delta"] = self._parse_optional_float(self.drift_max_delta.text())
        drift_cfg["max_residual"] = self._parse_optional_float(self.drift_max_residual.text())

        module_id = self.module.currentData(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(module_id, str) and module_id:
            self.recipe.module = module_id
        self.recipe.params = params
        self._mark_custom_preset()
        if run_validation:
            self._run_validation()
        if not self._updating:
            self.config_changed.emit()

    def set_auto_update_preview_enabled(self, enabled: bool) -> None:
        previous = self.auto_update_preview_checkbox.blockSignals(True)
        try:
            self.auto_update_preview_checkbox.setChecked(bool(enabled))
        finally:
            self.auto_update_preview_checkbox.blockSignals(previous)

    def auto_update_preview_enabled(self) -> bool:
        return self.auto_update_preview_checkbox.isChecked()

    def _parse_optional_float(self, text: str):
        value = text.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return value

    def _safe_int(self, value, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    def _safe_float(self, value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _ensure_dict(self, parent: dict, key: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            value = {}
        parent[key] = value
        return value

    def _run_validation(self):
        errors = list(self.recipe.validate())
        if self._stitch_windows_errors:
            errors.extend(self._stitch_windows_errors)
        if self._join_windows_errors:
            errors.extend(self._join_windows_errors)
        if self._baseline_anchor_errors:
            errors.extend(self._baseline_anchor_errors)
        if self._despike_exclusion_errors:
            errors.extend(self._despike_exclusion_errors)
        if errors:
            self.validation_label.setStyleSheet("color: #b00020;")
            formatted = "\n".join(f"• {err}" for err in errors)
            self.validation_label.setText(f"Configuration errors:\n{formatted}")
        else:
            self.validation_label.setStyleSheet("color: #0a0;")
            self.validation_label.setText("Recipe configuration is valid.")

    def get_recipe(self) -> Recipe:
        return self.recipe

    def _refresh_preset_list(self):
        current_data = self.preset_combo.currentData()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— Custom —", self._CUSTOM_SENTINEL)

        for label, path in self._discover_presets():
            self.preset_combo.addItem(label, str(path))

        if current_data:
            index = self.preset_combo.findData(current_data)
            if index != -1:
                self.preset_combo.setCurrentIndex(index)
            else:
                self.preset_combo.setCurrentIndex(0)
        else:
            self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _discover_presets(self) -> list[tuple[str, Path]]:
        presets: list[tuple[str, Path]] = []
        base_dir = self._project_root / "config" / "presets"
        presets.extend(self._collect_presets_from_dir(base_dir, "Core"))

        plugin_root = self._project_root / "plugins"
        if plugin_root.is_dir():
            for plugin_dir in sorted(plugin_root.iterdir()):
                if not plugin_dir.is_dir():
                    continue
                preset_dir = plugin_dir / "presets"
                label_prefix = plugin_dir.name
                presets.extend(self._collect_presets_from_dir(preset_dir, label_prefix))
        return presets

    def _collect_presets_from_dir(self, directory: Path, label_prefix: str) -> list[tuple[str, Path]]:
        if not directory.is_dir():
            return []
        entries: list[tuple[str, Path]] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            label = f"{label_prefix}: {path.stem}"
            entries.append((label, path))
        return entries

    def _load_selected_preset(self):
        data = self.preset_combo.currentData()
        if data == self._CUSTOM_SENTINEL:
            self.set_recipe(Recipe())
            self._mark_custom_preset()
            return
        if not data:
            return

        path = Path(str(data))
        try:
            with path.open("r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            self._show_validation_error(f"Failed to load preset: {exc}")
            return

        if not isinstance(content, dict):
            self._show_validation_error("Preset file must contain a mapping at the top level.")
            return

        try:
            self.set_recipe(content)
        except Exception as exc:  # pragma: no cover - defensive
            self._show_validation_error(f"Invalid preset contents: {exc}")
            return

        index = self.preset_combo.findData(str(path))
        if index != -1:
            self.preset_combo.setCurrentIndex(index)

    def _save_preset(self):
        base_dir = self._project_root / "config" / "presets"
        base_dir.mkdir(parents=True, exist_ok=True)

        default_name = f"{self.recipe.module}_preset"
        name, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Preset file name (will be saved under config/presets):",
            text=default_name,
        )
        if not ok:
            return
        filename = name.strip()
        if not filename:
            self._show_validation_error("Preset name cannot be empty.")
            return
        if not filename.lower().endswith(('.yaml', '.yml')):
            filename += ".yaml"

        path = base_dir / filename
        try:
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.recipe_dict(), handle, sort_keys=False)
        except OSError as exc:
            self._show_validation_error(f"Failed to save preset: {exc}")
            return

        self._refresh_preset_list()
        index = self.preset_combo.findData(str(path))
        if index != -1:
            self.preset_combo.setCurrentIndex(index)

    def _show_validation_error(self, message: str):
        self.validation_label.setStyleSheet("color: #b00020;")
        self.validation_label.setText(message)

    def _mark_custom_preset(self):
        if self.preset_combo.currentData() == self._CUSTOM_SENTINEL:
            return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

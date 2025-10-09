from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping, Sequence

import yaml
from PyQt6 import QtCore
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from spectro_app.engine.recipe_model import Recipe


class RecipeEditorDock(QDockWidget):
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

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

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
        layout.addWidget(self.validation_label)

        # --- Smoothing ---
        smoothing_group = QGroupBox("Smoothing")
        smoothing_form = QFormLayout(smoothing_group)
        smoothing_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.smooth_enable = QCheckBox("Enable Savitzky–Golay smoothing")
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 999)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(15)
        self.smooth_window.setToolTip("Smoothing window (must be odd and > 2×poly order)")
        self.smooth_poly = QSpinBox()
        self.smooth_poly.setRange(1, 15)
        self.smooth_poly.setValue(3)
        self.smooth_poly.setToolTip("Polynomial order for Savitzky–Golay smoothing")

        smoothing_form.addRow(self.smooth_enable)
        smoothing_form.addRow("Window", self.smooth_window)
        smoothing_form.addRow("Poly order", self.smooth_poly)
        layout.addWidget(smoothing_group)

        # --- Despiking ---
        despike_group = QGroupBox("Despiking")
        despike_form = QFormLayout(despike_group)
        despike_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.despike_enable = QCheckBox("Enable spike removal")
        self.despike_enable.setToolTip(
            "Detect and replace narrow spikes using a z-score threshold on a sliding window."
        )
        self.despike_window = QSpinBox()
        self.despike_window.setRange(3, 999)
        self.despike_window.setValue(5)
        self.despike_window.setToolTip("Window size (number of points) used for spike detection")
        self.despike_zscore = QDoubleSpinBox()
        self.despike_zscore.setRange(0.1, 99.9)
        self.despike_zscore.setDecimals(2)
        self.despike_zscore.setSingleStep(0.1)
        self.despike_zscore.setValue(5.0)
        self.despike_zscore.setToolTip("Z-score threshold used to flag spikes")

        despike_form.addRow(self.despike_enable)
        despike_form.addRow("Window", self.despike_window)
        despike_form.addRow("Z-score", self.despike_zscore)
        layout.addWidget(despike_group)

        # --- Join correction ---
        join_group = QGroupBox("Join correction")
        join_form = QFormLayout(join_group)
        self.join_enable = QCheckBox("Enable detector join correction")
        self.join_window = QSpinBox()
        self.join_window.setRange(1, 999)
        self.join_window.setValue(10)
        self.join_threshold = QLineEdit()
        self.join_threshold.setPlaceholderText("Leave blank for default")
        self.join_threshold.setClearButtonEnabled(True)
        self.join_threshold.setToolTip(
            "Absolute absorbance offset allowed when joining detectors. "
            "Default join threshold is 0.2; quality control fallback uses 0.5."
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

        self.join_windows_table = QTableWidget(0, 2)
        self.join_windows_table.setHorizontalHeaderLabels(["Min (nm)", "Max (nm)"])
        header = self.join_windows_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
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
        layout.addWidget(join_group)

        # --- Blank handling ---
        blank_group = QGroupBox("Blank handling")
        blank_form = QFormLayout(blank_group)
        self.blank_subtract = QCheckBox("Subtract blank from samples")
        self.blank_require = QCheckBox("Require blank for each batch")
        self.blank_fallback = QLineEdit()
        self.blank_fallback.setPlaceholderText("Optional fallback/default blank path")
        self.blank_fallback.setClearButtonEnabled(True)
        self.blank_match_strategy = QComboBox()
        self.blank_match_strategy.addItem("Match by blank identifier", "blank_id")
        self.blank_match_strategy.addItem("Match by cuvette slot", "cuvette_slot")
        self.blank_match_strategy.setToolTip(
            "Use blank identifiers when blank files share a named ID with their samples. "
            "Choose cuvette slots when pairing by instrument position; blank IDs may be left empty."
        )

        blank_form.addRow(self.blank_subtract)
        blank_form.addRow(self.blank_require)
        blank_form.addRow("Fallback", self.blank_fallback)
        blank_form.addRow("Match strategy", self.blank_match_strategy)
        layout.addWidget(blank_group)

        # --- QC / Drift ---
        qc_group = QGroupBox("Quality control – drift limits")
        qc_layout = QVBoxLayout(qc_group)
        self.drift_enable = QCheckBox("Enable drift monitoring")
        qc_layout.addWidget(self.drift_enable)

        drift_bounds_row = QHBoxLayout()
        self.drift_window_min = QLineEdit()
        self.drift_window_min.setPlaceholderText("Quiet window min")
        self.drift_window_min.setClearButtonEnabled(True)
        self.drift_window_max = QLineEdit()
        self.drift_window_max.setPlaceholderText("Quiet window max")
        self.drift_window_max.setClearButtonEnabled(True)
        drift_bounds_row.addWidget(self.drift_window_min)
        drift_bounds_row.addWidget(self.drift_window_max)
        qc_layout.addLayout(drift_bounds_row)

        qc_limits_form = QFormLayout()
        self.drift_max_slope = QLineEdit()
        self.drift_max_slope.setPlaceholderText("Max slope per hour")
        self.drift_max_slope.setClearButtonEnabled(True)
        self.drift_max_delta = QLineEdit()
        self.drift_max_delta.setPlaceholderText("Max delta allowed")
        self.drift_max_delta.setClearButtonEnabled(True)
        self.drift_max_residual = QLineEdit()
        self.drift_max_residual.setPlaceholderText("Max residual allowed")
        self.drift_max_residual.setClearButtonEnabled(True)

        qc_limits_form.addRow("Slope", self.drift_max_slope)
        qc_limits_form.addRow("Delta", self.drift_max_delta)
        qc_limits_form.addRow("Residual", self.drift_max_residual)
        qc_layout.addLayout(qc_limits_form)
        layout.addWidget(qc_group)

        layout.addStretch(1)

        self.setWidget(container)

        self._refresh_preset_list()
        self._connect_signals()
        self._update_ui_from_recipe()

    def _connect_signals(self):
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
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
        for signal in (
            self.module.currentTextChanged,
            self.smooth_enable.toggled,
            self.smooth_window.valueChanged,
            self.smooth_poly.valueChanged,
            self.despike_enable.toggled,
            self.despike_window.valueChanged,
            self.despike_zscore.valueChanged,
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

            params = self.recipe.params if isinstance(self.recipe.params, dict) else {}
            smoothing = params.get("smoothing", {}) if isinstance(params.get("smoothing"), dict) else {}
            self.smooth_enable.setChecked(bool(smoothing.get("enabled", False)))
            self.smooth_window.setValue(
                self._safe_int(smoothing.get("window"), self.smooth_window.value())
            )
            self.smooth_poly.setValue(
                self._safe_int(smoothing.get("polyorder"), self.smooth_poly.value())
            )

            despike_cfg = params.get("despike", {}) if isinstance(params.get("despike"), dict) else {}
            self.despike_enable.setChecked(bool(despike_cfg.get("enabled", False)))
            self.despike_window.setValue(
                self._safe_int(despike_cfg.get("window"), self.despike_window.value())
            )
            self.despike_zscore.setValue(
                self._safe_float(despike_cfg.get("zscore"), self.despike_zscore.value())
            )

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

    def _format_optional(self, value) -> str:
        if value is None:
            return ""
        return str(value)

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
            rows.append(
                {
                    "min": self._format_optional(entry.get("min_nm")),
                    "max": self._format_optional(entry.get("max_nm")),
                }
            )
        return rows

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
                self._insert_join_window_row(row.get("min", ""), row.get("max", ""))
        finally:
            self._join_windows_loading = False
        self._update_join_window_buttons()

    def _insert_join_window_row(self, min_text: str = "", max_text: str = "") -> None:
        row_index = self.join_windows_table.rowCount()
        self.join_windows_table.insertRow(row_index)
        min_field = self._create_join_window_field("Min (nm)", min_text)
        max_field = self._create_join_window_field("Max (nm)", max_text)
        self.join_windows_table.setCellWidget(row_index, 0, min_field)
        self.join_windows_table.setCellWidget(row_index, 1, max_field)

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

    def _on_join_window_value_changed(self, *_):
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
            min_text = min_widget.text().strip() if isinstance(min_widget, QLineEdit) else ""
            max_text = max_widget.text().strip() if isinstance(max_widget, QLineEdit) else ""
            rows.append({"min": min_text, "max": max_text})
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
        has_rows = self.join_windows_table.rowCount() > 0
        self.join_windows_remove_row.setEnabled(has_rows)
        removable = (
            self.join_windows_instrument_combo.count() > 1
            and self.join_windows_instrument_combo.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
            != self._JOIN_WINDOWS_GLOBAL_KEY
        )
        self.join_windows_remove_instrument.setEnabled(removable)

    def _build_join_windows_payload(self) -> tuple[object | None, list[str]]:
        errors: list[str] = []
        cleaned: dict[str, list[dict[str, float]]] = {}
        for key, rows in self._join_windows_store.items():
            cleaned_rows: list[dict[str, float]] = []
            for index, row in enumerate(rows, start=1):
                min_text = str(row.get("min", "")).strip()
                max_text = str(row.get("max", "")).strip()
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
                cleaned_rows.append({"min_nm": min_float, "max_nm": max_float})
            if cleaned_rows:
                cleaned[key] = cleaned_rows
        global_rows = cleaned.get(self._JOIN_WINDOWS_GLOBAL_KEY, [])
        non_global_keys = [
            key
            for key in self._join_windows_store.keys()
            if key != self._JOIN_WINDOWS_GLOBAL_KEY and key in cleaned
        ]
        if non_global_keys:
            payload: dict[str, list[dict[str, float]]] = {}
            if global_rows:
                payload["default"] = global_rows
            for key in non_global_keys:
                payload[key] = cleaned[key]
            return payload or None, errors
        if global_rows:
            return global_rows, errors
        return None, errors

    def _sync_recipe_from_ui(self) -> None:
        with self._suspend_updates():
            self._update_model_from_ui(force=True)

    def _update_model_from_ui(self, *_, force: bool = False, run_validation: bool = True):
        if self._updating and not force:
            return

        params_source = self.recipe.params if isinstance(self.recipe.params, dict) else {}
        params = copy.deepcopy(params_source)

        smoothing_cfg = self._ensure_dict(params, "smoothing")
        smoothing_cfg.update(
            {
                "enabled": self.smooth_enable.isChecked(),
                "window": int(self.smooth_window.value()),
                "polyorder": int(self.smooth_poly.value()),
            }
        )

        despike_cfg = self._ensure_dict(params, "despike")
        despike_enabled = self.despike_enable.isChecked()
        despike_cfg["enabled"] = despike_enabled
        if despike_enabled:
            despike_cfg["window"] = int(self.despike_window.value())
            despike_cfg["zscore"] = float(self.despike_zscore.value())
        else:
            despike_cfg.pop("window", None)
            despike_cfg.pop("zscore", None)

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
        if self._join_windows_errors:
            errors.extend(self._join_windows_errors)
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

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

import yaml
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectro_app.engine.recipe_model import Recipe


class RecipeEditorDock(QDockWidget):
    _CUSTOM_SENTINEL = "__custom__"

    def __init__(self, parent=None):
        super().__init__("Recipe Editor", parent)
        self.recipe = Recipe()
        self._updating = False
        self._project_root = Path(__file__).resolve().parents[2]

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
        self.module.addItems(["uvvis", "ftir", "raman", "pees"])
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

        join_form.addRow(self.join_enable)
        join_form.addRow("Window", self.join_window)
        join_form.addRow("Threshold", self.join_threshold)
        layout.addWidget(join_group)

        # --- Blank handling ---
        blank_group = QGroupBox("Blank handling")
        blank_form = QFormLayout(blank_group)
        self.blank_subtract = QCheckBox("Subtract blank from samples")
        self.blank_require = QCheckBox("Require blank for each batch")
        self.blank_fallback = QLineEdit()
        self.blank_fallback.setPlaceholderText("Optional fallback/default blank path")
        self.blank_fallback.setClearButtonEnabled(True)

        blank_form.addRow(self.blank_subtract)
        blank_form.addRow(self.blank_require)
        blank_form.addRow("Fallback", self.blank_fallback)
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
        self._run_validation()

    def _connect_signals(self):
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
        for signal in (
            self.module.currentTextChanged,
            self.smooth_enable.toggled,
            self.smooth_window.valueChanged,
            self.smooth_poly.valueChanged,
            self.join_enable.toggled,
            self.join_window.valueChanged,
            self.join_threshold.textChanged,
            self.blank_subtract.toggled,
            self.blank_require.toggled,
            self.blank_fallback.textChanged,
            self.drift_enable.toggled,
            self.drift_window_min.textChanged,
            self.drift_window_max.textChanged,
            self.drift_max_slope.textChanged,
            self.drift_max_delta.textChanged,
            self.drift_max_residual.textChanged,
        ):
            signal.connect(self._update_model_from_ui)

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
        self._run_validation()

    def recipe_dict(self) -> dict[str, object]:
        params = self.recipe.params if isinstance(self.recipe.params, dict) else {}
        return {
            "module": self.recipe.module,
            "params": copy.deepcopy(params),
            "version": self.recipe.version,
        }

    def _update_ui_from_recipe(self):
        with self._suspend_updates():
            module = self.recipe.module
            if module not in [self.module.itemText(i) for i in range(self.module.count())]:
                self.module.addItem(module)
            self.module.setCurrentText(module)

            params = self.recipe.params if isinstance(self.recipe.params, dict) else {}
            smoothing = params.get("smoothing", {}) if isinstance(params.get("smoothing"), dict) else {}
            self.smooth_enable.setChecked(bool(smoothing.get("enabled", False)))
            self.smooth_window.setValue(
                self._safe_int(smoothing.get("window"), self.smooth_window.value())
            )
            self.smooth_poly.setValue(
                self._safe_int(smoothing.get("polyorder"), self.smooth_poly.value())
            )

            join_cfg = params.get("join", {}) if isinstance(params.get("join"), dict) else {}
            self.join_enable.setChecked(bool(join_cfg.get("enabled", False)))
            self.join_window.setValue(
                self._safe_int(join_cfg.get("window"), self.join_window.value())
            )
            self.join_threshold.setText(self._format_optional(join_cfg.get("threshold")))

            blank_cfg = params.get("blank", {}) if isinstance(params.get("blank"), dict) else {}
            subtract = bool(blank_cfg.get("subtract", blank_cfg.get("enabled", False)))
            self.blank_subtract.setChecked(subtract)
            self.blank_require.setChecked(bool(blank_cfg.get("require", subtract)))
            fallback_value = blank_cfg.get("default", blank_cfg.get("fallback"))
            self.blank_fallback.setText(self._format_optional(fallback_value))

            qc_cfg = params.get("qc", {}) if isinstance(params.get("qc"), dict) else {}
            drift_cfg = qc_cfg.get("drift", {}) if isinstance(qc_cfg.get("drift"), dict) else {}
            self.drift_enable.setChecked(bool(drift_cfg.get("enabled", False)))
            window = drift_cfg.get("window", {}) if isinstance(drift_cfg.get("window"), dict) else {}
            self.drift_window_min.setText(self._format_optional(window.get("min")))
            self.drift_window_max.setText(self._format_optional(window.get("max")))
            self.drift_max_slope.setText(self._format_optional(drift_cfg.get("max_slope_per_hour")))
            self.drift_max_delta.setText(self._format_optional(drift_cfg.get("max_delta")))
            self.drift_max_residual.setText(self._format_optional(drift_cfg.get("max_residual")))

    def _format_optional(self, value) -> str:
        if value is None:
            return ""
        return str(value)

    def _update_model_from_ui(self):
        if self._updating:
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

        self.recipe.module = self.module.currentText()
        self.recipe.params = params
        self._mark_custom_preset()
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

    def _ensure_dict(self, parent: dict, key: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            value = {}
        parent[key] = value
        return value

    def _run_validation(self):
        errors = self.recipe.validate()
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

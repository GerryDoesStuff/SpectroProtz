from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Recipe:
    module: str = "uvvis"
    params: Dict[str, Any] = field(default_factory=dict)
    version: str = "0.1.0"

    def validate(self) -> list[str]:
        errs = []
        smoothing = self.params.get("smoothing", {})
        if smoothing.get("enabled"):
            window = int(smoothing.get("window", 15))
            poly = int(smoothing.get("polyorder", 3))
            if window % 2 == 0:
                errs.append("Savitzky–Golay window must be odd")
            if window <= poly:
                errs.append("Savitzky–Golay window must exceed polynomial order")
            if window <= poly * 2:
                errs.append("Savitzky–Golay window must be > 2*polyorder")
            if window < 3:
                errs.append("Savitzky–Golay window must be at least 3 points")

        join_cfg = self.params.get("join", {})
        if join_cfg.get("enabled"):
            window = int(join_cfg.get("window", 0))
            if window <= 0:
                errs.append("Join correction window must be positive")
            threshold = join_cfg.get("threshold")
            if threshold is not None and float(threshold) <= 0:
                errs.append("Join detection threshold must be positive")

        blank_cfg = self.params.get("blank", {})
        subtract = blank_cfg.get("subtract", blank_cfg.get("enabled", False))
        require_blank = blank_cfg.get("require", subtract)
        fallback = blank_cfg.get("default") or blank_cfg.get("fallback")
        if subtract and not require_blank and not fallback:
            errs.append("Blank subtraction allows missing blanks but provides no fallback/default blank")
        qc_cfg = self.params.get("qc", {})
        drift_cfg = qc_cfg.get("drift", {}) if isinstance(qc_cfg, dict) else {}
        if drift_cfg.get("enabled"):
            window = drift_cfg.get("window", {})
            if isinstance(window, dict):
                w_min = window.get("min")
                w_max = window.get("max")
                try:
                    if w_min is not None and w_max is not None and float(w_min) >= float(w_max):
                        errs.append("Drift window min must be less than max")
                except (TypeError, ValueError):
                    errs.append("Drift window bounds must be numeric")
            else:
                errs.append("Drift window must be a mapping with min/max")
            for key, label in (
                ("max_slope_per_hour", "slope limit"),
                ("max_delta", "delta limit"),
                ("max_residual", "residual limit"),
            ):
                value = drift_cfg.get(key)
                if value is None:
                    continue
                try:
                    if float(value) <= 0:
                        errs.append(f"Drift {label} must be positive")
                except (TypeError, ValueError):
                    errs.append(f"Drift {label} must be numeric")
        return errs

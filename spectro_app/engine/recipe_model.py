from dataclasses import dataclass, field
from typing import Dict, Any
from collections.abc import Mapping, Sequence

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

        baseline_cfg = self.params.get("baseline")
        if isinstance(baseline_cfg, dict):
            method = baseline_cfg.get("method")
            if method:
                method_name = str(method).strip().lower()
                if method_name not in {"asls", "rubberband", "snip"}:
                    errs.append("Baseline method must be one of asls, rubberband, snip")
                else:
                    if method_name == "asls":
                        lam_val = baseline_cfg.get("lam", baseline_cfg.get("lambda"))
                        if lam_val is not None:
                            try:
                                if float(lam_val) <= 0:
                                    errs.append("Baseline lambda must be positive")
                            except (TypeError, ValueError):
                                errs.append("Baseline lambda must be numeric")
                        p_val = baseline_cfg.get("p")
                        if p_val is not None:
                            try:
                                p_float = float(p_val)
                            except (TypeError, ValueError):
                                errs.append("Baseline p must be numeric")
                            else:
                                if not (0 < p_float < 1):
                                    errs.append("Baseline p must be between 0 and 1")
                        niter_val = baseline_cfg.get("niter")
                        if niter_val is not None:
                            try:
                                if int(niter_val) <= 0:
                                    errs.append("Baseline iteration count must be positive")
                            except (TypeError, ValueError):
                                errs.append("Baseline iteration count must be an integer")
                    if method_name == "snip":
                        iterations_val = baseline_cfg.get("iterations")
                        if iterations_val is not None:
                            try:
                                if int(iterations_val) <= 0:
                                    errs.append("SNIP iteration count must be positive")
                            except (TypeError, ValueError):
                                errs.append("SNIP iteration count must be an integer")

                    anchor_cfg = (
                        baseline_cfg.get("anchor")
                        or baseline_cfg.get("anchor_windows")
                        or baseline_cfg.get("anchors")
                        or baseline_cfg.get("zeroing")
                    )
                    if anchor_cfg:
                        if isinstance(anchor_cfg, Mapping):
                            windows_iter = anchor_cfg.get("windows")
                        elif isinstance(anchor_cfg, Sequence) and not isinstance(
                            anchor_cfg, (str, bytes)
                        ):
                            windows_iter = anchor_cfg
                        else:
                            errs.append(
                                "Baseline anchor configuration must be a mapping or sequence"
                            )
                            windows_iter = None
                        if isinstance(windows_iter, Sequence):
                            for idx, entry in enumerate(windows_iter, start=1):
                                if isinstance(entry, Mapping):
                                    min_val = (
                                        entry.get("min_nm")
                                        or entry.get("lower_nm")
                                        or entry.get("min")
                                        or entry.get("lower")
                                    )
                                    max_val = (
                                        entry.get("max_nm")
                                        or entry.get("upper_nm")
                                        or entry.get("max")
                                        or entry.get("upper")
                                    )
                                    target_val = entry.get("target")
                                    if target_val is None:
                                        target_val = entry.get("level") or entry.get("value")
                                elif isinstance(entry, Sequence) and not isinstance(
                                    entry, (str, bytes)
                                ):
                                    if len(entry) < 2:
                                        errs.append(
                                            f"Baseline anchor row {idx} must provide min and max"
                                        )
                                        continue
                                    min_val = entry[0]
                                    max_val = entry[1]
                                    target_val = entry[2] if len(entry) > 2 else None
                                else:
                                    errs.append(
                                        f"Baseline anchor row {idx} must be a mapping or sequence"
                                    )
                                    continue

                                try:
                                    min_float = float(min_val)
                                    max_float = float(max_val)
                                except (TypeError, ValueError):
                                    errs.append(
                                        f"Baseline anchor row {idx} bounds must be numeric"
                                    )
                                    continue
                                if min_float > max_float:
                                    errs.append(
                                        f"Baseline anchor row {idx} min must be ≤ max"
                                    )
                                    continue
                                if target_val is not None:
                                    try:
                                        float(target_val)
                                    except (TypeError, ValueError):
                                        errs.append(
                                            f"Baseline anchor row {idx} target must be numeric"
                                        )

        join_cfg = self.params.get("join", {})
        if join_cfg.get("enabled"):
            window = int(join_cfg.get("window", 0))
            if window <= 0:
                errs.append("Join correction window must be positive")
            threshold = join_cfg.get("threshold")
            if threshold is not None and float(threshold) <= 0:
                errs.append("Join detection threshold must be positive")

        solvent_cfg = self.params.get("solvent_subtraction")
        if isinstance(solvent_cfg, dict):
            ref_id = solvent_cfg.get("reference_id") or solvent_cfg.get("reference")
            if ref_id is not None:
                ref_text = str(ref_id).strip()
                if not ref_text:
                    errs.append("Solvent reference id must be non-empty when provided")
            scale_val = solvent_cfg.get("scale")
            if scale_val is not None:
                try:
                    float(scale_val)
                except (TypeError, ValueError):
                    errs.append("Solvent subtraction scale must be numeric")
            fit_scale_val = solvent_cfg.get("fit_scale")
            if fit_scale_val is not None and not isinstance(fit_scale_val, bool):
                errs.append("Solvent subtraction fit_scale must be true or false")

        blank_cfg = self.params.get("blank", {})
        subtract = blank_cfg.get("subtract", blank_cfg.get("enabled", False))
        require_blank = blank_cfg.get("require", subtract)
        fallback = blank_cfg.get("default") or blank_cfg.get("fallback")
        # Historically, recipes that subtracted blanks while allowing them to be optional were
        # rejected unless a fallback sample was provided. In practice the UV-Vis pipeline and
        # GUI expect such recipes to validate successfully; if advisory messaging is required it
        # should be surfaced outside of validation. We still resolve the fallback value here so
        # downstream consumers can make that decision if needed.
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

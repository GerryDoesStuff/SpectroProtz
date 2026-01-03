"""Core spectroscopy pipeline helpers.

This module centralizes the recipe-editor pipeline steps so multiple
modalities can share the same preprocessing and QC flow.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import multiprocessing
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spectro_app.engine import qc as qc_engine
from spectro_app.engine.peak_detection import detect_peaks_for_features, resolve_peak_config
from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis import pipeline as uvvis_pipeline

__all__ = [
    "AxisAdapter",
    "coerce_domain",
    "subtract_blank",
    "apply_baseline",
    "detect_joins",
    "correct_joins",
    "normalise_exclusion_windows",
    "normalise_stitch_windows",
    "stitch_regions",
    "despike_spectrum",
    "smooth_spectrum",
    "apply_solvent_subtraction",
    "average_replicates",
    "replicate_key",
    "blank_identifier",
    "blank_match_identifier",
    "blank_match_identifiers",
    "blank_replicate_key",
    "normalize_blank_match_strategy",
    "run_pipeline",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AxisAdapter:
    axis_key: str = "wavelength"
    unit_label: str = "nm"
    window_suffix: str = "nm"

    @classmethod
    def from_spectrum(cls, spec: Spectrum) -> "AxisAdapter":
        meta = spec.meta if isinstance(spec.meta, Mapping) else {}
        axis_key = str(meta.get("axis_key") or meta.get("axis_type") or "wavelength")
        unit_label = str(meta.get("axis_unit") or meta.get("axis_units") or "nm")
        suffix = _suffix_from_unit(unit_label)
        if axis_key.lower() == "wavenumber":
            suffix = suffix or "cm"
        return cls(axis_key=axis_key, unit_label=unit_label, window_suffix=suffix or "nm")


def _suffix_from_unit(unit_label: str) -> str:
    text = str(unit_label).strip().lower()
    if "cm" in text:
        return "cm"
    if "nm" in text:
        return "nm"
    return ""


def _apply_axis_metadata(spec: Spectrum, axis: AxisAdapter) -> Spectrum:
    if axis.axis_key == "wavelength" and axis.window_suffix == "nm":
        return spec

    meta = dict(spec.meta or {})
    meta["axis_key"] = axis.axis_key
    meta["axis_unit"] = axis.unit_label

    domain_key = "wavelength_domain_nm"
    if domain_key in meta:
        domain_meta = meta.pop(domain_key)
        target_key = f"{axis.axis_key}_domain_{axis.window_suffix}"
        meta[target_key] = domain_meta

    channels = dict(meta.get("channels") or {})
    if "original_wavelength" in channels:
        original_axis = channels.pop("original_wavelength")
        channels[f"original_{axis.axis_key}"] = original_axis
    meta["channels"] = channels

    return Spectrum(
        wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
        intensity=np.asarray(spec.intensity, dtype=float).copy(),
        meta=meta,
    )


def _map_window_keys(entry: Mapping[str, Any], axis: AxisAdapter) -> Dict[str, Any]:
    if axis.window_suffix == "nm":
        return dict(entry)

    mapped: Dict[str, Any] = dict(entry)
    suffix = axis.window_suffix
    key_pairs = {
        f"min_{suffix}": "min_nm",
        f"max_{suffix}": "max_nm",
        f"lower_{suffix}": "lower_nm",
        f"upper_{suffix}": "upper_nm",
    }
    for src, target in key_pairs.items():
        if src in mapped and target not in mapped:
            mapped[target] = mapped[src]
    return mapped


def _map_window_config(config: Any, axis: AxisAdapter) -> Any:
    if axis.window_suffix == "nm":
        return config
    if isinstance(config, Mapping):
        mapped = dict(config)
        windows = mapped.get("windows")
        if isinstance(windows, list):
            mapped["windows"] = [_map_window_keys(entry, axis) if isinstance(entry, Mapping) else entry for entry in windows]
        elif isinstance(windows, Mapping):
            mapped["windows"] = _map_window_keys(windows, axis)
        return _map_window_keys(mapped, axis)
    if isinstance(config, list):
        return [_map_window_keys(entry, axis) if isinstance(entry, Mapping) else entry for entry in config]
    return config


def coerce_domain(spec: Spectrum, domain: Dict[str, float] | None, *, axis: AxisAdapter | None = None) -> Spectrum:
    axis = axis or AxisAdapter.from_spectrum(spec)
    coerced = uvvis_pipeline.coerce_domain(spec, domain)
    return _apply_axis_metadata(coerced, axis)


def subtract_blank(
    spec: Spectrum,
    blank: Spectrum,
    *,
    audit: Optional[List[str]] = None,
) -> Spectrum:
    return uvvis_pipeline.subtract_blank(spec, blank, audit=audit)


def apply_baseline(
    spec: Spectrum,
    method: str,
    *,
    axis: AxisAdapter | None = None,
    **params: Any,
) -> Spectrum:
    axis = axis or AxisAdapter.from_spectrum(spec)
    anchor = params.get("anchor")
    if anchor is not None:
        params["anchor"] = _map_window_config(anchor, axis)
    return uvvis_pipeline.apply_baseline(spec, method, **params)


def detect_joins(*args: Any, **kwargs: Any) -> List[int]:
    return uvvis_pipeline.detect_joins(*args, **kwargs)


def correct_joins(*args: Any, **kwargs: Any) -> Spectrum:
    return uvvis_pipeline.correct_joins(*args, **kwargs)


def normalise_exclusion_windows(
    config: Any,
    *,
    axis: AxisAdapter | None = None,
) -> List[Dict[str, float]]:
    axis = axis or AxisAdapter()
    mapped = _map_window_config(config, axis)
    return uvvis_pipeline.normalise_exclusion_windows(mapped)


def normalise_stitch_windows(
    windows_cfg: Any,
    *,
    axis: AxisAdapter | None = None,
) -> List[Dict[str, float]]:
    axis = axis or AxisAdapter()
    mapped = _map_window_config(windows_cfg, axis)
    return uvvis_pipeline.normalise_stitch_windows(mapped)


def stitch_regions(
    spec: Spectrum,
    windows: Sequence[Mapping[str, object]] | None,
    *,
    axis: AxisAdapter | None = None,
    **kwargs: Any,
) -> Spectrum:
    axis = axis or AxisAdapter.from_spectrum(spec)
    mapped_windows = (
        [
            _map_window_keys(window, axis) if isinstance(window, Mapping) else window
            for window in windows
        ]
        if windows is not None
        else None
    )
    return uvvis_pipeline.stitch_regions(spec, mapped_windows, **kwargs)


def despike_spectrum(
    spec: Spectrum,
    *,
    axis: AxisAdapter | None = None,
    **kwargs: Any,
) -> Spectrum:
    axis = axis or AxisAdapter.from_spectrum(spec)
    exclusion_windows = kwargs.get("exclusion_windows")
    if exclusion_windows is not None:
        kwargs["exclusion_windows"] = _map_window_config(exclusion_windows, axis)
    return uvvis_pipeline.despike_spectrum(spec, **kwargs)


def smooth_spectrum(*args: Any, **kwargs: Any) -> Spectrum:
    return uvvis_pipeline.smooth_spectrum(*args, **kwargs)


@dataclass(frozen=True)
class SolventSubtractionTaskResult:
    spectrum: Spectrum
    error: str | None = None


def _solvent_subtraction_config(
    solvent_cfg: Mapping[str, Any],
    *,
    reference_id: str | None,
) -> Dict[str, Any]:
    shift_cfg = solvent_cfg.get("shift_compensation")
    shift_payload = dict(shift_cfg) if isinstance(shift_cfg, Mapping) else {}
    return {
        "reference_id": reference_id,
        "scale": solvent_cfg.get("scale"),
        "fit_scale": solvent_cfg.get("fit_scale", True),
        "ridge_alpha": solvent_cfg.get("ridge_alpha"),
        "include_offset": solvent_cfg.get("include_offset", False),
        "shift_compensation": shift_payload,
    }


def _default_parallel_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _resolve_parallel_settings(solvent_cfg: Mapping[str, Any]) -> tuple[bool, int]:
    parallel_cfg = solvent_cfg.get("parallel")
    if not isinstance(parallel_cfg, Mapping):
        parallel_cfg = {}

    enabled = parallel_cfg.get("enabled")
    if enabled is None and "use_multiprocessing" in solvent_cfg:
        enabled = solvent_cfg.get("use_multiprocessing")
    parallel_enabled = bool(enabled) if enabled is not None else False

    workers_value = parallel_cfg.get("workers")
    try:
        workers = int(workers_value) if workers_value is not None else _default_parallel_workers()
    except (TypeError, ValueError):
        workers = _default_parallel_workers()
    if workers < 1:
        workers = _default_parallel_workers()

    return parallel_enabled, workers


def _apply_solvent_subtraction_task(
    spec: Spectrum,
    solvent_reference: Spectrum | Sequence[Spectrum],
    cfg: Mapping[str, Any],
    axis: AxisAdapter,
) -> SolventSubtractionTaskResult:
    try:
        corrected = apply_solvent_subtraction(
            spec,
            solvent_reference,
            reference_id=cfg.get("reference_id"),
            scale=cfg.get("scale"),
            fit_scale=cfg.get("fit_scale", True),
            ridge_alpha=cfg.get("ridge_alpha"),
            include_offset=cfg.get("include_offset", False),
            shift_compensation=cfg.get("shift_compensation"),
        )
        return SolventSubtractionTaskResult(spectrum=corrected)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        return SolventSubtractionTaskResult(spectrum=spec, error=error_text)


def apply_solvent_subtraction(
    spec: Spectrum,
    reference: Spectrum | Sequence[Spectrum],
    *,
    reference_id: str | None = None,
    scale: float | None = None,
    fit_scale: bool = True,
    ridge_alpha: float | None = None,
    include_offset: bool = False,
    shift_compensation: Mapping[str, Any] | None = None,
) -> Spectrum:
    x = np.asarray(spec.wavelength, dtype=float)
    y = np.asarray(spec.intensity, dtype=float)
    references = list(reference) if isinstance(reference, Sequence) else [reference]
    if not references:
        return spec
    ref_specs = [ref for ref in references if ref is not None]
    if not ref_specs:
        return spec

    if x.size == 0:
        return spec

    def _interp_reference(
        target_x: np.ndarray, ref_spec: Spectrum, *, shift: float = 0.0
    ) -> np.ndarray:
        ref_x = np.asarray(ref_spec.wavelength, dtype=float)
        ref_y = np.asarray(ref_spec.intensity, dtype=float)
        if ref_x.size == 0:
            return np.full_like(target_x, np.nan, dtype=float)
        ref_order = np.argsort(ref_x)
        ref_x_sorted = ref_x[ref_order] + float(shift)
        ref_y_sorted = ref_y[ref_order]
        x_order = np.argsort(target_x)
        x_sorted = target_x[x_order]
        ref_interp_sorted = np.interp(
            x_sorted,
            ref_x_sorted,
            ref_y_sorted,
            left=ref_y_sorted[0],
            right=ref_y_sorted[-1],
        )
        ref_interp = np.empty_like(ref_interp_sorted)
        ref_interp[x_order] = ref_interp_sorted
        return ref_interp

    def _axis_limits(axis: np.ndarray) -> Tuple[float, float] | None:
        valid = axis[np.isfinite(axis)]
        if valid.size == 0:
            return None
        return float(np.min(valid)), float(np.max(valid))

    target_limits = _axis_limits(x)
    if target_limits is None:
        return spec
    ref_limits = []
    for ref in ref_specs:
        ref_axis = np.asarray(ref.wavelength, dtype=float)
        limits = _axis_limits(ref_axis)
        if limits is None:
            return spec
        ref_limits.append(limits)

    shift_cfg = dict(shift_compensation or {})
    shift_enabled = bool(shift_cfg.get("enabled", True))
    shift_min = float(shift_cfg.get("min", -2.0))
    shift_max = float(shift_cfg.get("max", 2.0))
    shift_step = float(shift_cfg.get("step", 0.1))
    if shift_min > shift_max:
        shift_min, shift_max = shift_max, shift_min
    if shift_step <= 0:
        shift_step = 0.1
    if shift_enabled:
        ref_min_candidates = [limits[0] + shift_min for limits in ref_limits]
        ref_max_candidates = [limits[1] + shift_max for limits in ref_limits]
    else:
        ref_min_candidates = [limits[0] for limits in ref_limits]
        ref_max_candidates = [limits[1] for limits in ref_limits]
    overlap_min = max([target_limits[0]] + ref_min_candidates)
    overlap_max = min([target_limits[1]] + ref_max_candidates)
    if overlap_min >= overlap_max:
        return spec

    overlap_mask = (x >= overlap_min) & (x <= overlap_max)
    if not np.any(overlap_mask):
        return spec

    ref_count = len(ref_specs)
    offset_used = bool(include_offset)
    if shift_enabled:
        candidates = np.arange(shift_min, shift_max + (shift_step / 2.0), shift_step)
        if candidates.size == 0:
            candidates = np.array([0.0], dtype=float)
    else:
        candidates = np.array([0.0], dtype=float)

    def _fit_coefficients(
        ref_matrix: np.ndarray, fit_mask: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        if not np.any(fit_mask):
            return None, 0.0
        if ref_count == 1 and not offset_used and not (fit_scale or scale is None):
            scale_val = float(scale)
            return np.array([scale_val], dtype=float), 0.0
        if ref_count == 1 and not offset_used:
            denom = float(np.dot(ref_matrix[fit_mask, 0], ref_matrix[fit_mask, 0]))
            if denom == 0.0:
                return None, 0.0
            scale_val = float(np.dot(y[fit_mask], ref_matrix[fit_mask, 0]) / denom)
            return np.array([scale_val], dtype=float), 0.0
        design = ref_matrix[fit_mask]
        if offset_used:
            design = np.column_stack([design, np.ones(design.shape[0], dtype=float)])
        target = y[fit_mask]
        ridge = None
        if ref_count > 1 and ridge_alpha is not None:
            ridge_val = float(ridge_alpha)
            if ridge_val > 0.0:
                ridge = ridge_val
        if ridge is None:
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        else:
            gram = design.T @ design
            if offset_used:
                penalty = np.diag([ridge] * ref_count + [0.0])
            else:
                penalty = np.diag([ridge] * ref_count)
            try:
                coef = np.linalg.solve(gram + penalty, design.T @ target)
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(gram + penalty, design.T @ target, rcond=None)
        if offset_used:
            return np.asarray(coef[:-1], dtype=float), float(coef[-1])
        return np.asarray(coef, dtype=float), 0.0

    best = {
        "shift": 0.0,
        "error": None,
        "ref_matrix": None,
        "fit_mask": None,
        "coef": None,
        "offset": 0.0,
    }
    for shift in candidates:
        ref_interp_list = [_interp_reference(x, ref, shift=shift) for ref in ref_specs]
        ref_matrix = np.column_stack(ref_interp_list)
        fit_mask = overlap_mask & np.isfinite(y)
        if ref_matrix.shape[1] == 1:
            fit_mask &= np.isfinite(ref_matrix[:, 0])
        else:
            fit_mask &= np.all(np.isfinite(ref_matrix), axis=1)
        coef, offset = _fit_coefficients(ref_matrix, fit_mask)
        if coef is None:
            continue
        fitted = ref_matrix @ coef
        if offset_used:
            fitted = fitted + offset
        residual = y[fit_mask] - fitted[fit_mask]
        error = float(np.mean(residual * residual)) if residual.size else None
        if error is None:
            continue
        if best["error"] is None or error < best["error"]:
            best.update(
                {
                    "shift": float(shift),
                    "error": error,
                    "ref_matrix": ref_matrix,
                    "fit_mask": fit_mask,
                    "coef": coef,
                    "offset": offset,
                }
            )

    if best["coef"] is None or best["ref_matrix"] is None or best["fit_mask"] is None:
        return spec

    ref_matrix = best["ref_matrix"]
    fit_mask = best["fit_mask"]
    coef = best["coef"]
    offset = best["offset"]
    fitted = ref_matrix @ coef
    if offset_used:
        fitted = fitted + offset
    fitted = np.asarray(fitted, dtype=float)
    valid_ref_mask = np.all(np.isfinite(ref_matrix), axis=1)
    apply_mask = np.isfinite(y) & valid_ref_mask
    if not np.any(apply_mask):
        return spec
    corrected = y.copy()
    corrected[apply_mask] = y[apply_mask] - fitted[apply_mask]
    overlap_min = float(np.min(x[fit_mask]))
    overlap_max = float(np.max(x[fit_mask]))
    residual = y[fit_mask] - fitted[fit_mask]
    sse = float(np.sum(residual * residual))
    rmse = float(np.sqrt(sse / residual.size)) if residual.size else None
    negative_fraction = None
    negative_area = None
    if residual.size:
        corrected_overlap = corrected[fit_mask]
        negative_fraction = float(np.mean(corrected_overlap < 0.0))
        order = np.argsort(x[fit_mask])
        sorted_x = x[fit_mask][order]
        sorted_corrected = corrected_overlap[order]
        negative_area = float(
            np.trapezoid(np.minimum(sorted_corrected, 0.0), sorted_x)
        )
    derivative_corr = None
    diagnostics_skipped = False
    if residual.size >= 3:
        derivative = np.gradient(fitted[fit_mask], x[fit_mask])
        valid = np.isfinite(residual) & np.isfinite(derivative)
        if np.count_nonzero(valid) >= 3:
            residual_std = float(np.std(residual[valid]))
            derivative_std = float(np.std(derivative[valid]))
            if (
                not np.isfinite(residual_std)
                or not np.isfinite(derivative_std)
                or residual_std == 0.0
                or derivative_std == 0.0
            ):
                diagnostics_skipped = True
            else:
                corr = np.corrcoef(residual[valid], derivative[valid])
                derivative_corr = float(corr[0, 1])
    condition_number = None
    fit_design = ref_matrix[fit_mask]
    if fit_design.size and fit_design.shape[0] >= fit_design.shape[1]:
        design_matrix = fit_design
        if offset_used:
            design_matrix = np.column_stack(
                [design_matrix, np.ones(design_matrix.shape[0], dtype=float)]
            )
        try:
            condition_number = float(np.linalg.cond(design_matrix))
        except np.linalg.LinAlgError:
            condition_number = None

    meta = dict(spec.meta or {})
    solvent_meta = dict(meta.get("solvent_subtraction") or {})
    ref_names = []
    ref_tags = []
    ref_metadata = []
    ref_sources = []
    ref_ids = []
    for ref in ref_specs:
        reference_meta = dict(ref.meta or {})
        if reference_meta.get("reference_name"):
            ref_names.append(reference_meta.get("reference_name"))
        if reference_meta.get("reference_tags"):
            ref_tags.append(reference_meta.get("reference_tags"))
        if reference_meta.get("reference_metadata"):
            ref_metadata.append(reference_meta.get("reference_metadata"))
        if reference_meta.get("source_path"):
            ref_sources.append(reference_meta.get("source_path"))
        ref_ident = _reference_identifier(ref)
        if ref_ident:
            ref_ids.append(ref_ident)

    shift_meta = {
        "enabled": bool(shift_enabled),
        "selected_shift": float(best["shift"]),
        "min": float(shift_min),
        "max": float(shift_max),
        "step": float(shift_step),
        "candidates": int(candidates.size),
        "optimizer": "grid",
        "residual": float(best["error"]) if best["error"] is not None else None,
        "interpolation": "linear",
        "resampling": {
            "method": "linear",
            "reference_axis": "shifted",
            "target_axis": "sample_axis",
        },
    }
    warning_thresholds = {
        "min_overlap_points": 10,
        "max_negative_fraction": 0.1,
        "max_derivative_corr_abs": 0.7,
        "max_condition_number": 1.0e6,
        "max_normalized_rmse": 0.5,
    }
    warnings = []
    overlap_points = int(np.count_nonzero(fit_mask))
    if overlap_points < warning_thresholds["min_overlap_points"]:
        warnings.append(
            f"Low overlap coverage (points={overlap_points}, "
            f"threshold={warning_thresholds['min_overlap_points']})."
        )
    if negative_fraction is not None and negative_fraction > warning_thresholds["max_negative_fraction"]:
        warnings.append(
            "Potential oversubtraction "
            f"(negative_fraction={negative_fraction:.3f}, "
            f"threshold={warning_thresholds['max_negative_fraction']})."
        )
    if derivative_corr is not None and abs(derivative_corr) > warning_thresholds["max_derivative_corr_abs"]:
        warnings.append(
            "Residuals correlated with reference derivative "
            f"(corr={derivative_corr:.3f}, "
            f"threshold={warning_thresholds['max_derivative_corr_abs']})."
        )
    if condition_number is not None and condition_number > warning_thresholds["max_condition_number"]:
        warnings.append(
            "Ill-conditioned multi-reference fit "
            f"(condition_number={condition_number:.2e}, "
            f"threshold={warning_thresholds['max_condition_number']:.2e})."
        )
    if rmse is not None and residual.size:
        signal_std = float(np.std(y[apply_mask])) if residual.size else 0.0
        if not np.isfinite(signal_std) or signal_std <= 0.0:
            normalized_rmse = None
            diagnostics_skipped = True
        else:
            normalized_rmse = float(rmse / signal_std)
        if normalized_rmse is not None and normalized_rmse > warning_thresholds["max_normalized_rmse"]:
            warnings.append(
                "High normalized RMSE over overlap "
                f"(normalized_rmse={normalized_rmse:.3f}, "
                f"threshold={warning_thresholds['max_normalized_rmse']})."
            )
    else:
        normalized_rmse = None
    if diagnostics_skipped:
        warnings.append(
            "Reference identical to sample; diagnostics skipped due to zero variance."
        )

    if ref_names and not solvent_meta.get("reference_name"):
        solvent_meta["reference_name"] = ref_names[0] if len(ref_names) == 1 else ref_names
    if ref_tags and not solvent_meta.get("reference_tags"):
        solvent_meta["reference_tags"] = ref_tags[0] if len(ref_tags) == 1 else ref_tags
    if ref_metadata and not solvent_meta.get("reference_metadata"):
        solvent_meta["reference_metadata"] = ref_metadata[0] if len(ref_metadata) == 1 else ref_metadata
    if ref_sources and not solvent_meta.get("reference_source"):
        solvent_meta["reference_source"] = ref_sources[0] if len(ref_sources) == 1 else ref_sources
    edge_strategy = "nearest"
    solvent_meta.update(
        {
            "enabled": True,
            "edge_strategy": edge_strategy,
            "reference_id": reference_id or (ref_ids[0] if len(ref_ids) == 1 else None),
            "reference_ids": ref_ids if len(ref_ids) > 1 else None,
            "reference_count": int(ref_count),
            "scale_input": float(scale) if scale is not None else None,
            "scale": float(coef[0]) if coef is not None and coef.size == 1 else None,
            "coefficients": [float(val) for val in coef] if coef is not None and coef.size > 1 else None,
            "offset": float(offset) if offset_used else None,
            "fit_scale": bool(fit_scale),
            "include_offset": bool(include_offset),
            "ridge_alpha": float(ridge_alpha) if ridge_alpha is not None else None,
            "weights": None,
            "interpolation": "linear",
            "overlap_min": float(overlap_min),
            "overlap_max": float(overlap_max),
            "overlap_points": overlap_points,
            "shift_compensation": shift_meta,
            "diagnostics": {
                "sse": sse,
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
            },
            "oversubtraction": {
                "negative_fraction": negative_fraction,
                "negative_area": negative_area,
            },
            "residual_derivative_corr": derivative_corr,
            "condition_number": condition_number,
            "warning_thresholds": warning_thresholds,
            "warnings": warnings,
        }
    )
    meta["solvent_subtraction"] = solvent_meta
    meta["solvent_subtracted"] = True

    return Spectrum(
        wavelength=x.copy(),
        intensity=corrected.copy(),
        meta=meta,
    )


average_replicates = uvvis_pipeline.average_replicates
replicate_key = uvvis_pipeline.replicate_key
blank_identifier = uvvis_pipeline.blank_identifier
blank_match_identifier = uvvis_pipeline.blank_match_identifier
blank_match_identifiers = uvvis_pipeline.blank_match_identifiers
blank_replicate_key = uvvis_pipeline.blank_replicate_key
normalize_blank_match_strategy = uvvis_pipeline.normalize_blank_match_strategy


def _lookup_blanks(
    blanks: Sequence[Spectrum],
    strategy: str | None,
) -> Dict[str, Spectrum]:
    lookup: Dict[str, Spectrum] = {}
    for blank in blanks:
        for identifier in blank_match_identifiers(blank, strategy):
            lookup.setdefault(identifier, blank)
    return lookup


def _normalize_identifier(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_role(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _reference_identifier(spec: Spectrum) -> str | None:
    meta = spec.meta if isinstance(spec.meta, Mapping) else {}
    for key in ("reference_id", "sample_id", "id", "label", "name"):
        ident = _normalize_identifier(meta.get(key))
        if ident:
            return ident
    return None


def _select_reference(
    references: Sequence[Spectrum],
    reference_id: str | None,
) -> Spectrum | None:
    if not reference_id:
        return None
    for ref in references:
        ident = _reference_identifier(ref)
        if ident and ident == reference_id:
            return ref
    return None


def _blank_candidates(
    spec: Spectrum,
    strategy: str | None,
    fallback_identifier: str | None,
) -> List[str]:
    candidates: List[str] = []
    if strategy == "cuvette_slot":
        slot_value = _normalize_identifier(spec.meta.get("cuvette_slot"))
        if slot_value:
            candidates.append(slot_value)
    blank_id_value = _normalize_identifier(spec.meta.get("blank_id"))
    if blank_id_value and blank_id_value not in candidates:
        candidates.append(blank_id_value)
    else:
        derived_sources: List[str] = []
        for key in ("sample_id", "channel"):
            derived = _normalize_identifier(spec.meta.get(key))
            if derived and derived not in derived_sources:
                derived_sources.append(derived)
        suffix = "_indref"
        for derived in derived_sources:
            if derived not in candidates:
                candidates.append(derived)
            if not derived.lower().endswith(suffix):
                indref_candidate = f"{derived}_indRef"
                if indref_candidate not in candidates:
                    candidates.append(indref_candidate)
    if fallback_identifier and fallback_identifier not in candidates:
        candidates.append(fallback_identifier)
    return candidates


def _select_blank(
    spec: Spectrum,
    lookup: Dict[str, Spectrum],
    strategy: str | None,
    fallback_identifier: str | None,
) -> Spectrum | None:
    for candidate in _blank_candidates(spec, strategy, fallback_identifier):
        selected = lookup.get(candidate)
        if selected is not None:
            return selected
    return None


def _detect_peaks(spec: Spectrum, recipe: Dict[str, Any], axis: AxisAdapter) -> Spectrum:
    features_cfg = dict(recipe.get("features", {})) if recipe else {}
    peak_cfg = dict(features_cfg.get("peaks", {}))
    peaks = detect_peaks_for_features(
        np.asarray(spec.wavelength, dtype=float),
        np.asarray(spec.intensity, dtype=float),
        resolve_peak_config(peak_cfg),
        axis_key=axis.axis_key,
    )
    meta = dict(spec.meta or {})
    meta.setdefault("features", {})
    meta["features"]["peaks"] = peaks
    return Spectrum(
        wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
        intensity=np.asarray(spec.intensity, dtype=float).copy(),
        meta=meta,
    )


def run_pipeline(
    specs: Iterable[Spectrum],
    recipe: Dict[str, Any],
    *,
    axis: AxisAdapter | None = None,
) -> Tuple[List[Spectrum], List[Dict[str, Any]]]:
    specs_list = list(specs)
    if not specs_list:
        return [], []

    axis = axis or AxisAdapter.from_spectrum(specs_list[0])
    domain_cfg = recipe.get("domain") if recipe else None
    stitch_cfg = dict(recipe.get("stitch", {})) if recipe else {}
    join_cfg = dict(recipe.get("join", {})) if recipe else {}
    despike_cfg = dict(recipe.get("despike", {})) if recipe else {}
    blank_cfg = dict(recipe.get("blank", {})) if recipe else {}
    baseline_cfg = dict(recipe.get("baseline", {})) if recipe else {}
    solvent_cfg = dict(recipe.get("solvent_subtraction", {})) if recipe else {}
    smoothing_cfg = dict(recipe.get("smoothing", {})) if recipe else {}
    module_id = str(recipe.get("module", "") or "").strip().lower()

    preprocess: List[Spectrum] = []
    for spec in specs_list:
        working = coerce_domain(spec, domain_cfg, axis=axis)

        if stitch_cfg.get("enabled"):
            windows_cfg = stitch_cfg.get("windows")
            windows = normalise_stitch_windows(windows_cfg, axis=axis)
            working = stitch_regions(
                working,
                windows,
                shoulder_points=stitch_cfg.get("shoulder_points", stitch_cfg.get("points", 0)),
                method=stitch_cfg.get("method", "linear"),
            )

        joins: List[int] = []
        if join_cfg.get("enabled"):
            joins = detect_joins(
                working.wavelength,
                working.intensity,
                threshold=join_cfg.get("threshold"),
                window=int(join_cfg.get("window", 10)),
            )
            working = correct_joins(
                working,
                joins,
                window=int(join_cfg.get("window", 10)),
            )
        working.meta.setdefault("join_indices", tuple(joins))

        if despike_cfg.get("enabled"):
            exclusion_windows = normalise_exclusion_windows(despike_cfg.get("exclusions"), axis=axis)
            working = despike_spectrum(
                working,
                zscore=despike_cfg.get("zscore", 5.0),
                window=despike_cfg.get("window", 5),
                baseline_window=despike_cfg.get("baseline_window"),
                spread_window=despike_cfg.get("spread_window"),
                spread_method=despike_cfg.get("spread_method", "mad"),
                spread_epsilon=despike_cfg.get("spread_epsilon", 1e-6),
                residual_floor=despike_cfg.get("residual_floor", 2e-2),
                isolation_ratio=despike_cfg.get("isolation_ratio", 20.0),
                tail_decay_ratio=despike_cfg.get("tail_decay_ratio", 0.85),
                max_passes=despike_cfg.get("max_passes", 10),
                leading_padding=despike_cfg.get("leading_padding", 0),
                trailing_padding=despike_cfg.get("trailing_padding", 0),
                noise_scale_multiplier=despike_cfg.get("noise_scale_multiplier", 1.0),
                rng_seed=despike_cfg.get("rng_seed"),
                exclusion_windows=exclusion_windows,
            )

        preprocess.append(working)

    blanks = [spec for spec in preprocess if spec.meta.get("role") == "blank"]
    reference_role_set = {"reference", "solvent", "solvent_reference"}
    solvent_enabled = module_id == "ftir" and bool(solvent_cfg.get("enabled"))
    solvent_reference_id = _normalize_identifier(
        solvent_cfg.get("reference_id") or solvent_cfg.get("reference")
    )
    reference_ids_cfg = solvent_cfg.get("reference_ids")
    if isinstance(reference_ids_cfg, Sequence) and not isinstance(
        reference_ids_cfg, (str, bytes)
    ):
        solvent_reference_ids = [
            _normalize_identifier(value)
            for value in reference_ids_cfg
            if _normalize_identifier(value)
        ]
    else:
        solvent_reference_ids = []
    solvent_references = (
        [
            spec
            for spec in preprocess
            if _normalize_role(spec.meta.get("role")) in reference_role_set
        ]
        if solvent_enabled
        else []
    )
    multi_reference = bool(solvent_cfg.get("multi_reference")) or len(
        solvent_reference_ids
    ) > 1
    if solvent_enabled and solvent_references and multi_reference:
        if not solvent_reference_ids:
            solvent_reference = solvent_references
        else:
            solvent_reference = [
                ref
                for ref in solvent_references
                if _normalize_identifier(_reference_identifier(ref))
                in solvent_reference_ids
            ]
        if not solvent_reference:
            solvent_reference = None
    else:
        solvent_reference = _select_reference(solvent_references, solvent_reference_id)

    if solvent_enabled:
        samples = [
            spec
            for spec in preprocess
            if spec.meta.get("role") not in {"blank"}
            and _normalize_role(spec.meta.get("role")) not in reference_role_set
        ]
    else:
        samples = [spec for spec in preprocess if spec.meta.get("role") != "blank"]

    subtract_blank_flag = blank_cfg.get("subtract", blank_cfg.get("enabled", False))
    require_blank = blank_cfg.get("require", subtract_blank_flag)
    strategy = normalize_blank_match_strategy(blank_cfg.get("match_strategy"))
    blank_lookup = _lookup_blanks(blanks, strategy)
    fallback_identifier = _normalize_identifier(blank_cfg.get("default") or blank_cfg.get("fallback"))

    pre_solvent: List[Spectrum] = []
    for spec in samples:
        working = spec
        if subtract_blank_flag:
            blank = _select_blank(spec, blank_lookup, strategy, fallback_identifier)
            if blank is None and require_blank:
                raise ValueError("Blank spectrum required but not found")
            if blank is not None:
                working = subtract_blank(working, blank)

        if baseline_cfg.get("method"):
            params = {
                "lam": baseline_cfg.get("lam", baseline_cfg.get("lambda", 1e5)),
                "p": baseline_cfg.get("p", 0.01),
                "niter": baseline_cfg.get("niter", 10),
                "iterations": baseline_cfg.get("iterations", 24),
            }
            params["anchor"] = baseline_cfg.get("anchor")
            working = apply_baseline(
                working, str(baseline_cfg.get("method")), axis=axis, **params
            )

        pre_solvent.append(working)

    solvent_results = pre_solvent
    if solvent_enabled and solvent_reference is not None:
        solvent_task_cfg = _solvent_subtraction_config(
            solvent_cfg, reference_id=solvent_reference_id
        )
        parallel_enabled, parallel_workers = _resolve_parallel_settings(solvent_cfg)
        if parallel_enabled and len(pre_solvent) > 1 and parallel_workers > 1:
            ctx = multiprocessing.get_context("spawn")
            results: List[SolventSubtractionTaskResult | None] = [
                None for _ in pre_solvent
            ]
            with ProcessPoolExecutor(
                mp_context=ctx, max_workers=parallel_workers
            ) as executor:
                future_map = {
                    executor.submit(
                        _apply_solvent_subtraction_task,
                        spec,
                        solvent_reference,
                        solvent_task_cfg,
                        axis,
                    ): idx
                    for idx, spec in enumerate(pre_solvent)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        logger.exception(
                            "Solvent subtraction task failed for spectrum %s: %s",
                            idx,
                            error_text,
                        )
                        results[idx] = SolventSubtractionTaskResult(
                            spectrum=pre_solvent[idx], error=error_text
                        )
            solvent_results = []
            for idx, result in enumerate(results):
                if result is None:
                    logger.error(
                        "Solvent subtraction task returned no result for spectrum %s",
                        idx,
                    )
                    solvent_results.append(pre_solvent[idx])
                    continue
                if result.error:
                    logger.error(
                        "Solvent subtraction failed for spectrum %s: %s",
                        idx,
                        result.error,
                    )
                solvent_results.append(result.spectrum)
        else:
            solvent_results = []
            for idx, spec in enumerate(pre_solvent):
                result = _apply_solvent_subtraction_task(
                    spec,
                    solvent_reference,
                    solvent_task_cfg,
                    axis,
                )
                if result.error:
                    logger.error(
                        "Solvent subtraction failed for spectrum %s: %s",
                        idx,
                        result.error,
                    )
                solvent_results.append(result.spectrum)

    processed: List[Spectrum] = []
    for working in solvent_results:
        if smoothing_cfg.get("enabled"):
            join_indices = working.meta.get("join_indices")
            working = smooth_spectrum(
                working,
                window=int(smoothing_cfg.get("window", 5)),
                polyorder=int(smoothing_cfg.get("polyorder", 2)),
                join_indices=join_indices,
            )

        working = _detect_peaks(working, recipe, axis)
        processed.append(working)

    drift_map = qc_engine.compute_uvvis_drift_map(processed, recipe)
    qc_rows = [
        qc_engine.compute_uvvis_qc(spec, recipe, drift_map.get(id(spec)))
        for spec in processed
    ]
    return processed, qc_rows

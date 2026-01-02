"""Core spectroscopy pipeline helpers.

This module centralizes the recipe-editor pipeline steps so multiple
modalities can share the same preprocessing and QC flow.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "average_replicates",
    "replicate_key",
    "blank_identifier",
    "blank_match_identifier",
    "blank_match_identifiers",
    "blank_replicate_key",
    "normalize_blank_match_strategy",
    "run_pipeline",
]


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
    smoothing_cfg = dict(recipe.get("smoothing", {})) if recipe else {}

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
    samples = [spec for spec in preprocess if spec.meta.get("role") != "blank"]

    subtract_blank_flag = blank_cfg.get("subtract", blank_cfg.get("enabled", False))
    require_blank = blank_cfg.get("require", subtract_blank_flag)
    strategy = normalize_blank_match_strategy(blank_cfg.get("match_strategy"))
    blank_lookup = _lookup_blanks(blanks, strategy)
    fallback_identifier = _normalize_identifier(blank_cfg.get("default") or blank_cfg.get("fallback"))

    processed: List[Spectrum] = []
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
            working = apply_baseline(working, str(baseline_cfg.get("method")), axis=axis, **params)

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

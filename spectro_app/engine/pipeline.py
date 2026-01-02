"""Core pipeline execution for spectroscopy processing."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from spectro_app.engine.peak_detection import detect_peaks_for_features, resolve_peak_config
from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis import pipeline as uvvis_pipeline


def _ensure_features(meta: Mapping[str, object] | None) -> Dict[str, object]:
    if isinstance(meta, Mapping):
        features = meta.get("features")
        if isinstance(features, dict):
            return features
    return {}


def _set_features(meta: Dict[str, object], features: Dict[str, object]) -> None:
    if features:
        meta["features"] = features


def _core_peak_detection(
    specs: Iterable[Spectrum],
    peak_cfg: Mapping[str, object] | None,
    *,
    axis_key: str = "wavelength",
) -> List[Spectrum]:
    processed: List[Spectrum] = []
    resolved = resolve_peak_config(dict(peak_cfg) if peak_cfg else None)
    for spec in specs:
        wl = np.asarray(spec.wavelength, dtype=float)
        intensity = np.asarray(spec.intensity, dtype=float)
        peaks = detect_peaks_for_features(
            wl,
            intensity,
            resolved,
            axis_key=axis_key,
        )
        meta = dict(spec.meta)
        features = _ensure_features(meta)
        features["peaks"] = peaks
        _set_features(meta, features)
        processed.append(
            Spectrum(
                wavelength=spec.wavelength.copy(),
                intensity=spec.intensity.copy(),
                meta=meta,
            )
        )
    return processed


def _core_preprocess(specs: Iterable[Spectrum], recipe: Mapping[str, object]) -> List[Spectrum]:
    if not specs:
        return []

    domain_cfg = recipe.get("domain")
    stitch_cfg = dict(recipe.get("stitch", {})) if recipe else {}
    join_cfg = dict(recipe.get("join", {})) if recipe else {}
    despike_cfg = dict(recipe.get("despike", {})) if recipe else {}
    blank_cfg = dict(recipe.get("blank", {})) if recipe else {}
    baseline_cfg = dict(recipe.get("baseline", {})) if recipe else {}
    smoothing_cfg = dict(recipe.get("smoothing", {})) if recipe else {}

    blanks = [spec for spec in specs if spec.meta.get("role") == "blank"]
    samples = [spec for spec in specs if spec.meta.get("role") != "blank"]

    def _apply_steps(targets: Iterable[Spectrum], *, allow_blank_subtraction: bool) -> List[Spectrum]:
        processed: List[Spectrum] = []
        for spec in targets:
            working = uvvis_pipeline.coerce_domain(
                spec, domain_cfg if isinstance(domain_cfg, dict) else None
            )
            if stitch_cfg.get("enabled"):
                windows = uvvis_pipeline.normalise_stitch_windows(
                    stitch_cfg.get("windows") or []
                )
                working = uvvis_pipeline.stitch_regions(
                    working,
                    windows,
                    shoulder_points=int(stitch_cfg.get("shoulder_points", 5)),
                    method=str(stitch_cfg.get("method") or "linear"),
                    fallback_policy=str(stitch_cfg.get("fallback_policy") or "preserve"),
                )
            if join_cfg.get("enabled"):
                joins = uvvis_pipeline.detect_joins(
                    working.wavelength,
                    working.intensity,
                    window=int(join_cfg.get("window", 10)),
                    threshold=join_cfg.get("threshold"),
                )
                working = uvvis_pipeline.correct_joins(
                    working,
                    joins,
                    window=int(join_cfg.get("window", 10)),
                )
            if despike_cfg.get("enabled"):
                working = uvvis_pipeline.despike_spectrum(
                    working,
                    zscore=float(despike_cfg.get("zscore", 3.0)),
                    window=int(despike_cfg.get("window", 7)),
                    noise_scale_multiplier=float(
                        despike_cfg.get("noise_scale_multiplier", 1.0)
                    ),
                )
            if allow_blank_subtraction and blank_cfg.get("subtract"):
                blank_spec = None
                if blanks:
                    averaged_blanks, _ = uvvis_pipeline.average_replicates(blanks)
                    if averaged_blanks:
                        blank_spec = averaged_blanks[0]
                if blank_spec is not None:
                    working = uvvis_pipeline.subtract_blank(working, blank_spec)
            method = baseline_cfg.get("method")
            if method:
                working = uvvis_pipeline.apply_baseline(
                    working,
                    str(method),
                    lam=baseline_cfg.get("lam", baseline_cfg.get("lambda")),
                    p=baseline_cfg.get("p"),
                    niter=baseline_cfg.get("iterations", baseline_cfg.get("niter")),
                    **{
                        k: v
                        for k, v in baseline_cfg.items()
                        if k
                        not in {"method", "lam", "lambda", "p", "iterations", "niter"}
                    },
                )
            if smoothing_cfg.get("enabled"):
                working = uvvis_pipeline.smooth_spectrum(
                    working,
                    window=int(smoothing_cfg.get("window", 7)),
                    polyorder=int(smoothing_cfg.get("polyorder", 3)),
                )
            processed.append(working)
        return processed

    processed_blanks = _apply_steps(blanks, allow_blank_subtraction=False)
    processed_samples = _apply_steps(samples, allow_blank_subtraction=True)
    return processed_blanks + processed_samples


def _generic_analyze(
    specs: Iterable[Spectrum],
    recipe: Mapping[str, object] | None,
    *,
    axis_key: str = "wavelength",
) -> Tuple[List[Spectrum], List[Dict[str, object]]]:
    feature_cfg = dict(recipe.get("features", {})) if recipe else {}
    peak_cfg = dict(feature_cfg.get("peaks", {}))
    specs_with_peaks = _core_peak_detection(specs, peak_cfg, axis_key=axis_key)
    qc_rows = []
    for spec in specs_with_peaks:
        features = spec.meta.get("features") if isinstance(spec.meta, Mapping) else None
        peaks = []
        if isinstance(features, Mapping):
            peaks = list(features.get("peaks") or [])
        qc_rows.append({"peaks": len(peaks)})
    return specs_with_peaks, qc_rows


def run_pipeline(
    plugin,
    specs: Iterable[Spectrum],
    recipe: Mapping[str, object],
) -> Tuple[List[Spectrum], List[Dict[str, object]]]:
    if getattr(plugin, "id", "") != "uvvis":
        specs = _core_preprocess(specs, recipe)
        return _generic_analyze(specs, recipe, axis_key="wavelength")

    preprocessed = plugin.preprocess(specs, recipe)
    feature_cfg = dict(recipe.get("features", {})) if recipe else {}
    peak_cfg = dict(feature_cfg.get("peaks", {}))
    preprocessed = _core_peak_detection(preprocessed, peak_cfg, axis_key="wavelength")
    return plugin.analyze(preprocessed, recipe)


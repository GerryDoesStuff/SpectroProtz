"""Quality control helpers shared across spectroscopy plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import medfilt

from spectro_app.engine.plugin_api import Spectrum


@dataclass
class SaturationResult:
    flag: bool
    minimum: float
    maximum: float
    low_fraction: float
    high_fraction: float
    limits: Tuple[float, float]


@dataclass
class NoiseResult:
    rsd: float
    window: Tuple[float | None, float | None]
    used_points: int


@dataclass
class JoinResult:
    count: int
    max_offset: float
    mean_offset: float
    max_overlap_error: float
    indices: List[int]


@dataclass
class SpikeResult:
    count: int
    threshold: float | None


@dataclass
class SmoothingGuardResult:
    flag: bool
    requested: int | None
    spectrum_points: int


def _infer_mode(spec: Spectrum) -> str:
    mode = str(spec.meta.get("mode", "")).strip().lower()
    if "abs" in mode:
        return "absorbance"
    if "%t" in mode or "trans" in mode:
        return "transmittance"
    if "%r" in mode or "reflect" in mode:
        return "reflectance"

    inten = np.asarray(spec.intensity, dtype=float)
    if not np.any(np.isfinite(inten)):
        return "unknown"
    finite = inten[np.isfinite(inten)]
    if finite.size == 0:
        return "unknown"
    max_val = float(np.nanmax(finite))
    min_val = float(np.nanmin(finite))
    if max_val > 2.0 and min_val >= 0:
        return "transmittance"
    return "absorbance"


def _saturation_thresholds(mode: str, intensity: np.ndarray) -> Tuple[float, float]:
    if mode == "absorbance":
        return -0.05, 3.0
    if mode in {"transmittance", "reflectance"}:
        finite = intensity[np.isfinite(intensity)]
        if finite.size == 0:
            return 0.0, 1.0
        max_val = float(np.nanmax(finite))
        scale = 100.0 if max_val > 2.0 else 1.0
        if scale > 10:
            return 0.5, 99.5
        return 0.005, 0.995
    return float(np.nanmin(intensity)), float(np.nanmax(intensity))


def compute_saturation(spec: Spectrum) -> SaturationResult:
    intensity = np.asarray(spec.intensity, dtype=float)
    mode = _infer_mode(spec)
    low_limit, high_limit = _saturation_thresholds(mode, intensity)
    finite = np.isfinite(intensity)
    count = int(np.count_nonzero(finite))
    if count == 0:
        return SaturationResult(False, float("nan"), float("nan"), 0.0, 0.0, (low_limit, high_limit))

    high_hits = np.logical_and(finite, intensity >= high_limit)
    low_hits = np.logical_and(finite, intensity <= low_limit)
    high_fraction = float(np.count_nonzero(high_hits)) / count if count else 0.0
    low_fraction = float(np.count_nonzero(low_hits)) / count if count else 0.0
    flag = bool(high_fraction > 0 or low_fraction > 0)
    return SaturationResult(
        flag=flag,
        minimum=float(np.nanmin(intensity)),
        maximum=float(np.nanmax(intensity)),
        low_fraction=low_fraction,
        high_fraction=high_fraction,
        limits=(low_limit, high_limit),
    )


def compute_quiet_window_noise(
    spec: Spectrum,
    quiet_window: Dict[str, float] | None,
) -> NoiseResult:
    wl = np.asarray(spec.wavelength, dtype=float)
    intensity = np.asarray(spec.intensity, dtype=float)

    mask = np.isfinite(wl) & np.isfinite(intensity)
    if quiet_window:
        q_min = quiet_window.get("min")
        q_max = quiet_window.get("max")
    else:
        q_min = float(wl[np.isfinite(wl)].min()) if np.any(np.isfinite(wl)) else None
        q_max = None

    if q_min is not None:
        mask &= wl >= float(q_min)
    if quiet_window and quiet_window.get("max") is not None:
        q_max = float(quiet_window["max"])
        mask &= wl <= q_max

    if mask.sum() < 3:
        return NoiseResult(rsd=float("nan"), window=(q_min, q_max), used_points=int(mask.sum()))

    window_values = intensity[mask]
    denominator = float(np.nanmean(np.abs(window_values)))
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        rsd = float("nan")
    else:
        rsd = float(np.nanstd(window_values, ddof=1) / denominator * 100.0)

    return NoiseResult(rsd=rsd, window=(q_min, q_max), used_points=int(mask.sum()))


def compute_join_diagnostics(
    spec: Spectrum,
    join_cfg: Dict[str, Any] | None,
) -> JoinResult:
    from spectro_app.plugins.uvvis import pipeline

    join_cfg = join_cfg or {}
    if not join_cfg.get("enabled"):
        return JoinResult(count=0, max_offset=0.0, mean_offset=0.0, max_overlap_error=0.0, indices=[])

    window = int(join_cfg.get("window", 10))
    if window < 1:
        window = 1

    y = np.asarray(spec.intensity, dtype=float)
    joins = pipeline.detect_joins(
        spec.wavelength,
        y,
        threshold=join_cfg.get("threshold"),
        window=window,
    )
    if not joins:
        return JoinResult(count=0, max_offset=0.0, mean_offset=0.0, max_overlap_error=0.0, indices=[])

    offsets: List[float] = []
    overlap_errors: List[float] = []
    for join_idx in joins:
        left = y[max(0, join_idx - window) : join_idx]
        right = y[join_idx : min(y.size, join_idx + window)]
        if left.size == 0 or right.size == 0:
            continue
        left_mean = float(np.nanmean(left))
        right_mean = float(np.nanmean(right))
        offsets.append(right_mean - left_mean)
        overlap_len = min(left.size, right.size)
        if overlap_len:
            left_tail = left[-overlap_len:]
            right_head = right[:overlap_len]
            overlap_errors.append(float(np.nanmean(np.abs(right_head - left_tail))))

    if not offsets:
        return JoinResult(count=len(joins), max_offset=0.0, mean_offset=0.0, max_overlap_error=0.0, indices=list(joins))

    max_offset = float(np.nanmax(np.abs(offsets))) if offsets else 0.0
    mean_offset = float(np.nanmean(offsets)) if offsets else 0.0
    max_overlap_error = float(np.nanmax(overlap_errors)) if overlap_errors else 0.0
    return JoinResult(
        count=len(joins),
        max_offset=max_offset,
        mean_offset=mean_offset,
        max_overlap_error=max_overlap_error,
        indices=list(joins),
    )


def count_spikes(spec: Spectrum, despike_cfg: Dict[str, Any] | None) -> SpikeResult:
    y = np.asarray(spec.intensity, dtype=float)
    if y.size < 3:
        return SpikeResult(count=0, threshold=None)

    despike_cfg = despike_cfg or {}
    window = int(despike_cfg.get("window", 5))
    if window % 2 == 0:
        window += 1
    if window < 3:
        window = 3
    if window > y.size:
        window = y.size if y.size % 2 else y.size - 1
    if window < 3:
        return SpikeResult(count=0, threshold=None)

    baseline = medfilt(y, kernel_size=window)
    residual = y - baseline
    median = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - median))
    zscore = float(despike_cfg.get("zscore", 5.0))
    if np.isfinite(mad) and mad > 0:
        threshold = float(zscore) * 1.4826 * mad
    else:
        std = np.nanstd(residual)
        if not np.isfinite(std) or std == 0:
            return SpikeResult(count=0, threshold=None)
        threshold = float(zscore) * std
    mask = np.abs(residual) > threshold
    count = int(np.count_nonzero(mask))
    return SpikeResult(count=count, threshold=threshold)


def smoothing_guard(spec: Spectrum, smoothing_cfg: Dict[str, Any] | None) -> SmoothingGuardResult:
    smoothing_cfg = smoothing_cfg or {}
    if not smoothing_cfg.get("enabled"):
        return SmoothingGuardResult(flag=False, requested=None, spectrum_points=spec.intensity.size)

    requested = int(smoothing_cfg.get("window", 5))
    spectrum_points = int(np.asarray(spec.intensity).size)
    flag = requested >= max(9, spectrum_points // 2)
    if requested <= 0:
        flag = True
    return SmoothingGuardResult(flag=flag, requested=requested, spectrum_points=spectrum_points)


def summarise_uvvis_metrics(metrics: Dict[str, Any]) -> str:
    parts: List[str] = []
    saturation: SaturationResult = metrics["saturation"]
    noise: NoiseResult = metrics["noise"]
    join: JoinResult = metrics["join"]
    spikes: SpikeResult = metrics["spikes"]
    smoothing: SmoothingGuardResult = metrics["smoothing"]

    if saturation.flag:
        parts.append("Saturation detected")
    parts.append(f"Noise RSD {noise.rsd:.2f}%" if np.isfinite(noise.rsd) else "Noise RSD n/a")
    if join.count:
        parts.append(f"Joins {join.count} (max offset {join.max_offset:.3g})")
    if spikes.count:
        parts.append(f"Spikes {spikes.count}")
    if smoothing.flag:
        parts.append("Smoothing guard")
    return "; ".join(parts)


def compute_uvvis_qc(spec: Spectrum, recipe: Dict[str, Any]) -> Dict[str, Any]:
    qc_cfg = recipe.get("qc", {})
    saturation = compute_saturation(spec)
    noise = compute_quiet_window_noise(spec, qc_cfg.get("quiet_window"))
    join = compute_join_diagnostics(spec, recipe.get("join"))
    spikes = count_spikes(spec, recipe.get("despike"))
    smoothing = smoothing_guard(spec, recipe.get("smoothing"))

    flags: List[str] = []
    noise_limit = float(qc_cfg.get("noise_rsd_limit", 5.0))
    join_limit = float(qc_cfg.get("join_offset_limit", recipe.get("join", {}).get("threshold", 0.0) or 0.5))
    spike_limit = int(qc_cfg.get("spike_limit", 0))

    if saturation.flag:
        flags.append("saturation")
    if np.isfinite(noise.rsd) and noise.rsd > noise_limit:
        flags.append("noise")
    if join.count and abs(join.max_offset) > join_limit:
        flags.append("join_offset")
    if spikes.count > spike_limit:
        flags.append("spikes")
    if smoothing.flag:
        flags.append("smoothing_guard")

    summary = summarise_uvvis_metrics(
        {
            "saturation": saturation,
            "noise": noise,
            "join": join,
            "spikes": spikes,
            "smoothing": smoothing,
        }
    )

    return {
        "mode": _infer_mode(spec),
        "saturation": saturation,
        "noise": noise,
        "join": join,
        "spikes": spikes,
        "smoothing": smoothing,
        "flags": flags,
        "summary": summary,
    }


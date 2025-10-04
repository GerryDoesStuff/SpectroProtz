"""Quality control helpers shared across spectroscopy plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.signal import medfilt

from spectro_app.engine.plugin_api import Spectrum

STAGE_CHANNEL_NAMES: Tuple[str, ...] = (
    "raw",
    "blanked",
    "baseline_corrected",
    "joined",
    "despiked",
    "smoothed",
)


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
class RollingNoiseResult:
    enabled: bool
    window_points: int
    step_points: int
    windows: int
    max_rsd: float
    median_rsd: float


@dataclass
class JoinResult:
    count: int = 0
    max_offset: float = 0.0
    mean_offset: float = 0.0
    max_overlap_error: float = 0.0
    indices: List[int] = field(default_factory=list)
    offsets: List[float] = field(default_factory=list)
    pre_deltas: List[float] = field(default_factory=list)
    post_deltas: List[float] = field(default_factory=list)
    pre_overlap_errors: List[float] = field(default_factory=list)
    post_overlap_errors: List[float] = field(default_factory=list)


@dataclass
class SpikeResult:
    count: int
    threshold: float | None


@dataclass
class SmoothingGuardResult:
    flag: bool
    requested: int | None
    spectrum_points: int


@dataclass
class DriftResult:
    available: bool
    timestamp: datetime | None
    order: int | None
    baseline: float
    baseline_delta: float
    slope_per_hour: float | None
    residual: float | None
    predicted: float | None
    span_minutes: float | None
    flag: bool
    reasons: List[str]
    batch_reasons: List[str]
    window: Tuple[float | None, float | None]


def _mean_abs_first_difference(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        return float("nan")
    diffs = np.diff(arr)
    if diffs.size == 0:
        return float("nan")
    abs_diffs = np.abs(diffs)
    finite = np.isfinite(abs_diffs)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmean(abs_diffs[finite]))
@dataclass
class NegativeIntensityResult:
    processed_count: int
    processed_total: int
    processed_fraction: float
    channels: Dict[str, Dict[str, float | int]]
    flag: bool = False
    tolerance: float = 0.0


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


def compute_rolling_rsd(spec: Spectrum, rolling_cfg: Dict[str, Any] | None) -> RollingNoiseResult:
    rolling_cfg = rolling_cfg or {}
    window = int(rolling_cfg.get("window", 0) or 0)
    step = int(rolling_cfg.get("step", 1) or 1)

    if window <= 1 or step <= 0:
        return RollingNoiseResult(
            enabled=bool(rolling_cfg),
            window_points=max(window, 0),
            step_points=max(step, 1),
            windows=0,
            max_rsd=float("nan"),
            median_rsd=float("nan"),
        )

    intensity = np.asarray(spec.intensity, dtype=float)
    finite_mask = np.isfinite(intensity)
    if not np.any(finite_mask):
        return RollingNoiseResult(
            enabled=True,
            window_points=window,
            step_points=step,
            windows=0,
            max_rsd=float("nan"),
            median_rsd=float("nan"),
        )

    rsd_values: List[float] = []
    total_points = intensity.size
    for start in range(0, total_points - window + 1, step):
        window_values = intensity[start : start + window]
        finite = window_values[np.isfinite(window_values)]
        if finite.size < 3:
            continue
        denominator = float(np.nanmean(np.abs(finite)))
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            continue
        if finite.size < 2:
            continue
        rsd = float(np.nanstd(finite, ddof=1) / denominator * 100.0)
        if np.isfinite(rsd):
            rsd_values.append(rsd)

    if not rsd_values:
        return RollingNoiseResult(
            enabled=True,
            window_points=window,
            step_points=step,
            windows=0,
            max_rsd=float("nan"),
            median_rsd=float("nan"),
        )

    rsd_array = np.array(rsd_values, dtype=float)
    max_rsd = float(np.nanmax(rsd_array)) if rsd_array.size else float("nan")
    median_rsd = float(np.nanmedian(rsd_array)) if rsd_array.size else float("nan")

    return RollingNoiseResult(
        enabled=True,
        window_points=window,
        step_points=step,
        windows=len(rsd_values),
        max_rsd=max_rsd,
        median_rsd=median_rsd,
    )


def compute_join_diagnostics(
    spec: Spectrum,
    join_cfg: Dict[str, Any] | None,
) -> JoinResult:
    from spectro_app.plugins.uvvis import pipeline

    join_cfg = join_cfg or {}
    if not join_cfg.get("enabled"):
        return JoinResult()

    window = int(join_cfg.get("window", 10))
    if window < 1:
        window = 1

    meta = spec.meta if isinstance(spec.meta, dict) else {}

    def _as_float_list(values: Iterable[Any] | None) -> List[float]:
        if values is None:
            return []
        result: List[float] = []
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(float("nan"))
        return result

    indices: List[int] = []
    offsets: List[float] = []
    pre_deltas: List[float] = []
    post_deltas: List[float] = []
    pre_overlap_errors: List[float] = []
    post_overlap_errors: List[float] = []

    stats_meta = meta.get("join_statistics") if isinstance(meta, dict) else None
    segments_meta = meta.get("join_segments") if isinstance(meta, dict) else None

    if isinstance(stats_meta, dict):
        indices = [int(idx) for idx in stats_meta.get("indices", [])]
        offsets = _as_float_list(stats_meta.get("offsets"))
        pre_deltas = _as_float_list(stats_meta.get("pre_deltas"))
        post_deltas = _as_float_list(stats_meta.get("post_deltas"))
        pre_overlap_errors = _as_float_list(stats_meta.get("pre_overlap_errors"))
        post_overlap_errors = _as_float_list(stats_meta.get("post_overlap_errors"))

        if (not pre_deltas or len(pre_deltas) != len(indices)) and isinstance(stats_meta.get("pre_means"), dict):
            left_means = _as_float_list(stats_meta["pre_means"].get("left"))
            right_means = _as_float_list(stats_meta["pre_means"].get("right"))
            pre_deltas = [
                right_means[i] - left_means[i]
                if i < len(left_means) and i < len(right_means)
                else float("nan")
                for i in range(len(indices))
            ]
        if (not post_deltas or len(post_deltas) != len(indices)) and isinstance(stats_meta.get("post_means"), dict):
            left_means = _as_float_list(stats_meta["post_means"].get("left"))
            right_means = _as_float_list(stats_meta["post_means"].get("right"))
            post_deltas = [
                right_means[i] - left_means[i]
                if i < len(left_means) and i < len(right_means)
                else float("nan")
                for i in range(len(indices))
            ]

    if (not indices or len(offsets) != len(indices)) and isinstance(segments_meta, dict):
        seg_indices = [int(idx) for idx in segments_meta.get("indices", [])]
        raw_segments = segments_meta.get("raw") or []
        corrected_segments = segments_meta.get("corrected") or []
        segment_window = int(segments_meta.get("window") or window)
        if segment_window < 1:
            segment_window = 1
        if raw_segments and corrected_segments and len(raw_segments) >= len(seg_indices) + 1:
            indices = seg_indices
            offsets = []
            pre_deltas = []
            post_deltas = []
            pre_overlap_errors = []
            post_overlap_errors = []
            for idx in range(len(seg_indices)):
                left_raw = np.asarray(raw_segments[idx], dtype=float)
                right_raw = np.asarray(raw_segments[idx + 1], dtype=float)
                left_post = np.asarray(corrected_segments[idx], dtype=float)
                right_post = np.asarray(corrected_segments[idx + 1], dtype=float)
                if left_raw.size == 0 or right_raw.size == 0:
                    offsets.append(float("nan"))
                    pre_deltas.append(float("nan"))
                    post_deltas.append(float("nan"))
                    pre_overlap_errors.append(float("nan"))
                    post_overlap_errors.append(float("nan"))
                    continue

                left_window_raw = left_raw[-segment_window:]
                right_window_raw = right_raw[:segment_window]
                left_window_post = left_post[-segment_window:]
                right_window_post = right_post[:segment_window]

                left_mean_pre = float(np.nanmean(left_window_raw)) if left_window_raw.size else float("nan")
                right_mean_pre = float(np.nanmean(right_window_raw)) if right_window_raw.size else float("nan")
                pre_deltas.append(right_mean_pre - left_mean_pre)

                left_mean_post = float(np.nanmean(left_window_post)) if left_window_post.size else float("nan")
                right_mean_post = float(np.nanmean(right_window_post)) if right_window_post.size else float("nan")
                post_deltas.append(right_mean_post - left_mean_post)

                offsets.append(
                    float(np.nanmedian(right_window_raw) - np.nanmedian(left_window_raw))
                )

                overlap_len = min(left_window_raw.size, right_window_raw.size)
                if overlap_len:
                    left_tail_pre = left_window_raw[-overlap_len:]
                    right_head_pre = right_window_raw[:overlap_len]
                    pre_overlap_errors.append(float(np.nanmean(np.abs(right_head_pre - left_tail_pre))))
                else:
                    pre_overlap_errors.append(float("nan"))

                overlap_len_post = min(left_window_post.size, right_window_post.size)
                if overlap_len_post:
                    left_tail_post = left_window_post[-overlap_len_post:]
                    right_head_post = right_window_post[:overlap_len_post]
                    post_overlap_errors.append(float(np.nanmean(np.abs(right_head_post - left_tail_post))))
                else:
                    post_overlap_errors.append(float("nan"))

    if not indices:
        y = np.asarray(spec.intensity, dtype=float)
        joins = pipeline.detect_joins(
            spec.wavelength,
            y,
            threshold=join_cfg.get("threshold"),
            window=window,
        )
        if not joins:
            return JoinResult()

        offsets = []
        pre_deltas = []
        post_deltas = []
        pre_overlap_errors = []
        post_overlap_errors = []
        for join_idx in joins:
            left = y[max(0, join_idx - window) : join_idx]
            right = y[join_idx : min(y.size, join_idx + window)]
            if left.size == 0 or right.size == 0:
                offsets.append(float("nan"))
                pre_deltas.append(float("nan"))
                post_deltas.append(float("nan"))
                pre_overlap_errors.append(float("nan"))
                post_overlap_errors.append(float("nan"))
                continue
            left_mean = float(np.nanmean(left))
            right_mean = float(np.nanmean(right))
            offsets.append(right_mean - left_mean)
            pre_deltas.append(right_mean - left_mean)
            post_deltas.append(right_mean - left_mean)
            overlap_len = min(left.size, right.size)
            if overlap_len:
                left_tail = left[-overlap_len:]
                right_head = right[:overlap_len]
                overlap_val = float(np.nanmean(np.abs(right_head - left_tail)))
            else:
                overlap_val = float("nan")
            pre_overlap_errors.append(overlap_val)
            post_overlap_errors.append(overlap_val)
        indices = list(joins)

    offsets_arr = np.asarray(offsets, dtype=float)
    finite_offsets = offsets_arr[np.isfinite(offsets_arr)]
    if finite_offsets.size:
        max_offset = float(np.nanmax(np.abs(finite_offsets)))
        mean_offset = float(np.nanmean(finite_offsets))
    else:
        max_offset = 0.0
        mean_offset = 0.0

    post_overlap_arr = np.asarray(post_overlap_errors, dtype=float)
    finite_post_overlap = post_overlap_arr[np.isfinite(post_overlap_arr)]
    if finite_post_overlap.size:
        max_overlap_error = float(np.nanmax(finite_post_overlap))
    else:
        max_overlap_error = 0.0

    return JoinResult(
        count=len(indices),
        max_offset=max_offset,
        mean_offset=mean_offset,
        max_overlap_error=max_overlap_error,
        indices=list(indices),
        offsets=list(offsets),
        pre_deltas=list(pre_deltas),
        post_deltas=list(post_deltas),
        pre_overlap_errors=list(pre_overlap_errors),
        post_overlap_errors=list(post_overlap_errors),
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


def _parse_timestamp(meta: Dict[str, Any]) -> datetime | None:
    for key in ("acquired_datetime", "timestamp", "datetime"):
        value = meta.get(key) if meta else None
        if isinstance(value, datetime):
            return value
        if value:
            try:
                return datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                continue
    return None


def _baseline_in_window(spec: Spectrum, window: Tuple[float | None, float | None]) -> float:
    wl = np.asarray(spec.wavelength, dtype=float)
    intensity = np.asarray(spec.intensity, dtype=float)
    mask = np.isfinite(wl) & np.isfinite(intensity)
    lower, upper = window
    if lower is not None:
        mask &= wl >= float(lower)
    if upper is not None:
        mask &= wl <= float(upper)
    if np.count_nonzero(mask) == 0:
        mask = np.isfinite(intensity)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.nanmedian(intensity[mask]))


def compute_uvvis_drift_map(specs: Iterable[Spectrum], recipe: Dict[str, Any]) -> Dict[int, DriftResult]:
    qc_cfg = recipe.get("qc", {}) if recipe else {}
    drift_cfg = dict(qc_cfg.get("drift", {}))
    if not drift_cfg.get("enabled"):
        return {}

    window_cfg = drift_cfg.get("window") or {}
    window = (
        float(window_cfg.get("min")) if window_cfg.get("min") is not None else None,
        float(window_cfg.get("max")) if window_cfg.get("max") is not None else None,
    )

    max_slope = drift_cfg.get("max_slope_per_hour")
    slope_limit = float(max_slope) if max_slope is not None else None
    max_delta = drift_cfg.get("max_delta")
    delta_limit = float(max_delta) if max_delta is not None else None
    max_residual = drift_cfg.get("max_residual")
    residual_limit = float(max_residual) if max_residual is not None else None

    specs_list = list(specs)
    baseline_map: Dict[int, float] = {id(spec): _baseline_in_window(spec, window) for spec in specs_list}
    timestamp_map: Dict[int, datetime] = {}
    for spec in specs_list:
        ts = _parse_timestamp(spec.meta)
        if ts is not None:
            timestamp_map[id(spec)] = ts

    records: List[Tuple[int, datetime, float]] = []
    for spec in specs_list:
        spec_id = id(spec)
        ts = timestamp_map.get(spec_id)
        baseline = baseline_map.get(spec_id, float("nan"))
        if ts is not None and np.isfinite(baseline):
            records.append((spec_id, ts, baseline))

    if not records:
        span_minutes: float | None = None
        slope_per_hour = float("nan")
        batch_reasons: List[str] = []
        drift_map: Dict[int, DriftResult] = {}
        for spec in specs_list:
            baseline = baseline_map.get(id(spec), float("nan"))
            drift_map[id(spec)] = DriftResult(
                available=False,
                timestamp=None,
                order=None,
                baseline=baseline,
                baseline_delta=float("nan"),
                slope_per_hour=slope_per_hour,
                residual=float("nan"),
                predicted=float("nan"),
                span_minutes=span_minutes,
                flag=False,
                reasons=[],
                batch_reasons=batch_reasons,
                window=window,
            )
        return drift_map

    records.sort(key=lambda item: item[1])
    base_time = records[0][1]
    times = np.array(
        [((ts - base_time).total_seconds() / 60.0) for _, ts, _ in records],
        dtype=float,
    )
    baselines = np.array([baseline for _, _, baseline in records], dtype=float)
    span_minutes = float(times.max() - times.min()) if times.size else 0.0

    if times.size >= 2 and span_minutes > 0:
        A = np.vstack([times, np.ones_like(times)]).T
        slope_per_minute, intercept = np.linalg.lstsq(A, baselines, rcond=None)[0]
        slope_per_minute = float(slope_per_minute)
        intercept = float(intercept)
    else:
        slope_per_minute = 0.0
        intercept = float(baselines[0]) if baselines.size else float("nan")

    predicted = slope_per_minute * times + intercept
    residuals = baselines - predicted
    slope_per_hour = slope_per_minute * 60.0

    batch_reasons: List[str] = []
    if slope_limit is not None and np.isfinite(slope_per_hour) and slope_limit > 0:
        if abs(slope_per_hour) > slope_limit:
            batch_reasons.append("slope")

    baseline_deltas = baselines - baselines[0]
    if delta_limit is not None and delta_limit > 0 and np.any(np.isfinite(baseline_deltas)):
        if float(np.nanmax(np.abs(baseline_deltas))) > delta_limit:
            if "delta" not in batch_reasons:
                batch_reasons.append("delta")

    order_map = {spec_id: idx for idx, (spec_id, _, _) in enumerate(records)}
    time_map = {spec_id: times[idx] for idx, (spec_id, _, _) in enumerate(records)}
    delta_map = {spec_id: baseline_deltas[idx] for idx, (spec_id, _, _) in enumerate(records)}
    residual_map = {spec_id: residuals[idx] for idx, (spec_id, _, _) in enumerate(records)}
    predicted_map = {spec_id: predicted[idx] for idx, (spec_id, _, _) in enumerate(records)}

    drift_map: Dict[int, DriftResult] = {}
    for spec in specs_list:
        spec_id = id(spec)
        baseline = baseline_map.get(spec_id, float("nan"))
        ts = timestamp_map.get(spec_id)
        if spec_id in order_map:
            spec_reasons = list(batch_reasons)
            delta_val = float(delta_map[spec_id])
            residual_val = float(residual_map[spec_id])
            if delta_limit is not None and delta_limit > 0 and abs(delta_val) > delta_limit:
                if "delta" not in spec_reasons:
                    spec_reasons.append("delta")
            if residual_limit is not None and residual_limit > 0 and np.isfinite(residual_val):
                if abs(residual_val) > residual_limit:
                    if "residual" not in spec_reasons:
                        spec_reasons.append("residual")
            flag = bool(spec_reasons)
            drift_map[spec_id] = DriftResult(
                available=True,
                timestamp=ts,
                order=order_map[spec_id],
                baseline=baseline,
                baseline_delta=delta_val,
                slope_per_hour=slope_per_hour if np.isfinite(slope_per_hour) else float("nan"),
                residual=residual_val,
                predicted=float(predicted_map[spec_id]),
                span_minutes=span_minutes,
                flag=flag,
                reasons=spec_reasons,
                batch_reasons=list(batch_reasons),
                window=window,
            )
        else:
            spec_reasons = list(batch_reasons)
            flag = bool(spec_reasons)
            drift_map[spec_id] = DriftResult(
                available=False,
                timestamp=None,
                order=None,
                baseline=baseline,
                baseline_delta=float("nan"),
                slope_per_hour=slope_per_hour if np.isfinite(slope_per_hour) else float("nan"),
                residual=float("nan"),
                predicted=float("nan"),
                span_minutes=span_minutes,
                flag=flag,
                reasons=spec_reasons,
                batch_reasons=list(batch_reasons),
                window=window,
            )

    return drift_map


def detect_negative_intensity(spec: Spectrum) -> NegativeIntensityResult:
    intensity = np.asarray(spec.intensity, dtype=float)
    finite = np.isfinite(intensity)
    total = int(np.count_nonzero(finite))
    negatives = int(np.count_nonzero(intensity[finite] < 0)) if total else 0
    fraction = float(negatives / total) if total else 0.0

    channel_stats: Dict[str, Dict[str, float | int]] = {}
    channels_meta = spec.meta.get("channels") if isinstance(spec.meta, dict) else None
    if isinstance(channels_meta, dict):
        for name, values in channels_meta.items():
            try:
                arr = np.asarray(values, dtype=float)
            except Exception:  # pragma: no cover - unexpected metadata types
                continue
            mask = np.isfinite(arr)
            total_ch = int(np.count_nonzero(mask))
            if total_ch == 0:
                count_ch = 0
                frac_ch = 0.0
            else:
                count_ch = int(np.count_nonzero(arr[mask] < 0))
                frac_ch = float(count_ch / total_ch)
            channel_stats[name] = {
                "count": count_ch,
                "total": total_ch,
                "fraction": frac_ch,
            }

    return NegativeIntensityResult(
        processed_count=negatives,
        processed_total=total,
        processed_fraction=fraction,
        channels=channel_stats,
    )


def summarise_uvvis_metrics(metrics: Dict[str, Any]) -> str:
    parts: List[str] = []
    saturation: SaturationResult = metrics["saturation"]
    noise: NoiseResult = metrics["noise"]
    rolling: RollingNoiseResult = metrics.get("rolling_noise")
    join: JoinResult = metrics["join"]
    spikes: SpikeResult = metrics["spikes"]
    smoothing: SmoothingGuardResult = metrics["smoothing"]
    drift: DriftResult = metrics.get("drift")
    negative: NegativeIntensityResult | None = metrics.get("negative_intensity")

    if saturation.flag:
        parts.append("Saturation detected")
    parts.append(f"Noise RSD {noise.rsd:.2f}%" if np.isfinite(noise.rsd) else "Noise RSD n/a")
    if rolling and rolling.enabled and rolling.windows and np.isfinite(rolling.max_rsd):
        parts.append(f"Rolling noise max {rolling.max_rsd:.2f}%")
    if join.count:
        parts.append(f"Joins {join.count} (max offset {join.max_offset:.3g})")
    if spikes.count:
        parts.append(f"Spikes {spikes.count}")
    if smoothing.flag:
        parts.append("Smoothing guard")
    if drift and (drift.flag or (drift.available and drift.slope_per_hour is not None)):
        if drift.flag and np.isfinite(drift.slope_per_hour if drift.slope_per_hour is not None else float("nan")):
            parts.append(f"Drift {drift.slope_per_hour:.3g}/h")
        elif drift.available and drift.slope_per_hour is not None and np.isfinite(drift.slope_per_hour):
            parts.append(f"Drift slope {drift.slope_per_hour:.3g}/h")
    if negative and negative.processed_count:
        parts.append(f"Negative intensity {negative.processed_fraction:.1%}")
    return "; ".join(parts)


def compute_uvvis_qc(
    spec: Spectrum,
    recipe: Dict[str, Any],
    drift: DriftResult | None = None,
) -> Dict[str, Any]:
    qc_cfg = recipe.get("qc", {})
    saturation = compute_saturation(spec)
    noise = compute_quiet_window_noise(spec, qc_cfg.get("quiet_window"))
    rolling_noise = compute_rolling_rsd(spec, qc_cfg.get("rolling_rsd"))
    join = compute_join_diagnostics(spec, recipe.get("join"))
    spikes = count_spikes(spec, recipe.get("despike"))
    smoothing = smoothing_guard(spec, recipe.get("smoothing"))
    negative = detect_negative_intensity(spec)

    processed_roughness = _mean_abs_first_difference(spec.intensity)
    channel_roughness: Dict[str, float] = {}
    channels_meta = spec.meta.get("channels") if isinstance(spec.meta, dict) else None
    if isinstance(channels_meta, dict):
        for name in STAGE_CHANNEL_NAMES:
            if name not in channels_meta:
                continue
            try:
                channel_roughness[name] = _mean_abs_first_difference(channels_meta[name])
            except (TypeError, ValueError):
                continue
    roughness_metrics = {"processed": processed_roughness, "channels": channel_roughness}

    if drift is None:
        drift_cfg = qc_cfg.get("drift", {}) if qc_cfg else {}
        window_cfg = drift_cfg.get("window") or {}
        window = (
            float(window_cfg.get("min")) if window_cfg.get("min") is not None else None,
            float(window_cfg.get("max")) if window_cfg.get("max") is not None else None,
        )
        drift = DriftResult(
            available=False,
            timestamp=_parse_timestamp(spec.meta),
            order=None,
            baseline=_baseline_in_window(spec, window),
            baseline_delta=float("nan"),
            slope_per_hour=float("nan"),
            residual=float("nan"),
            predicted=float("nan"),
            span_minutes=None,
            flag=False,
            reasons=[],
            batch_reasons=[],
            window=window,
        )

    flags: List[str] = []
    noise_limit = float(qc_cfg.get("noise_rsd_limit", 5.0))
    join_limit = float(qc_cfg.get("join_offset_limit", recipe.get("join", {}).get("threshold", 0.0) or 0.5))
    spike_limit = int(qc_cfg.get("spike_limit", 0))
    rolling_cfg = qc_cfg.get("rolling_rsd") or {}
    rolling_limit = rolling_cfg.get("limit")
    rolling_limit_val = float(rolling_limit) if rolling_limit is not None else None
    negative_tolerance = float(qc_cfg.get("negative_intensity_tolerance", 0.0))
    if negative_tolerance < 0:
        negative_tolerance = 0.0
    negative.tolerance = negative_tolerance
    negative.flag = negative.processed_fraction > negative_tolerance

    if saturation.flag:
        flags.append("saturation")
    if np.isfinite(noise.rsd) and noise.rsd > noise_limit:
        flags.append("noise")
    if (
        rolling_noise.enabled
        and rolling_noise.windows
        and rolling_limit_val is not None
        and np.isfinite(rolling_noise.max_rsd)
        and rolling_noise.max_rsd > rolling_limit_val
    ):
        flags.append("rolling_noise")
    if join.count and abs(join.max_offset) > join_limit:
        flags.append("join_offset")
    if spikes.count > spike_limit:
        flags.append("spikes")
    if smoothing.flag:
        flags.append("smoothing_guard")
    if drift.flag:
        flags.append("drift")
    if negative.flag:
        flags.append("negative_intensity")

    summary = summarise_uvvis_metrics(
        {
            "saturation": saturation,
            "noise": noise,
            "rolling_noise": rolling_noise,
            "join": join,
            "spikes": spikes,
            "smoothing": smoothing,
            "drift": drift,
            "negative_intensity": negative,
        }
    )

    return {
        "mode": _infer_mode(spec),
        "saturation": saturation,
        "noise": noise,
        "rolling_noise": rolling_noise,
        "join": join,
        "spikes": spikes,
        "smoothing": smoothing,
        "drift": drift,
        "roughness": roughness_metrics,
        "negative_intensity": negative,
        "flags": flags,
        "summary": summary,
    }


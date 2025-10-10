"""UV-Vis preprocessing helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

from spectro_app.engine.plugin_api import Spectrum

__all__ = [
    "coerce_domain",
    "subtract_blank",
    "apply_baseline",
    "detect_joins",
    "correct_joins",
    "despike_spectrum",
    "smooth_spectrum",
    "average_replicates",
    "replicate_key",
    "blank_identifier",
    "blank_match_identifier",
    "blank_match_identifiers",
    "blank_replicate_key",
    "normalize_blank_match_strategy",
]


def _nan_safe(values: np.ndarray) -> np.ndarray:
    """Interpolate NaNs in ``values`` so that downstream algorithms behave."""

    arr = np.asarray(values, dtype=float)
    if np.all(np.isfinite(arr)):
        return arr

    x = np.arange(arr.size)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr)

    arr = arr.copy()
    arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return arr


def _update_channel(meta: Dict[str, object], name: str, values: np.ndarray, *, overwrite: bool = True) -> None:
    """Store a processing channel in ``meta`` preserving existing ones."""

    channels = dict(meta.get("channels") or {})
    if not overwrite and name in channels:
        return
    channels[name] = np.asarray(values, dtype=float).copy()
    meta["channels"] = channels


def coerce_domain(spec: Spectrum, domain: Dict[str, float] | None) -> Spectrum:
    """Ensure wavelength axis is sorted, clipped, and optionally interpolated."""

    wl = np.asarray(spec.wavelength, dtype=float)
    inten = np.asarray(spec.intensity, dtype=float)

    order = np.argsort(wl)
    wl = wl[order]
    inten = inten[order]

    mask = np.isfinite(wl) & np.isfinite(inten)
    wl = wl[mask]
    inten = inten[mask]

    if wl.size == 0:
        raise ValueError("Spectrum does not contain finite wavelength samples")

    preserved_wl = wl.copy()
    preserved_inten = inten.copy()
    unique_wl, inverse, counts = np.unique(wl, return_inverse=True, return_counts=True)
    aggregated_inten = np.zeros_like(unique_wl, dtype=float)
    np.add.at(aggregated_inten, inverse, inten)
    inten = aggregated_inten / counts
    wl = unique_wl

    original_min = float(wl[0])
    original_max = float(wl[-1])

    output_wl = wl
    output_inten = inten
    requested_min: float | None = None
    requested_max: float | None = None
    resampled = False

    if domain:
        start = float(domain.get("min", wl.min())) if domain.get("min") is not None else float(wl.min())
        stop = float(domain.get("max", wl.max())) if domain.get("max") is not None else float(wl.max())
        if start >= stop:
            raise ValueError("Domain minimum must be smaller than maximum")

        requested_min = float(start) if domain.get("min") is not None else None
        requested_max = float(stop) if domain.get("max") is not None else None

        if "step" in domain and domain["step"] is not None:
            step = float(domain["step"])
            if step <= 0:
                raise ValueError("Domain step must be positive")
            count = int(np.floor((stop - start) / step + 0.5)) + 1
            axis = start + np.arange(count) * step
            if axis[-1] > stop + 1e-9:
                axis = axis[axis <= stop + 1e-9]
            if axis.size < 2:
                axis = np.linspace(start, stop, 2)
            output_wl = axis
            output_inten = np.interp(axis, wl, inten, left=np.nan, right=np.nan)
            resampled = True
        elif domain.get("num") is not None:
            num = int(domain.get("num", wl.size))
            if num < 2:
                raise ValueError("Domain must request at least two points")
            axis = np.linspace(start, stop, num)
            output_wl = axis
            output_inten = np.interp(axis, wl, inten, left=np.nan, right=np.nan)
            resampled = True
        else:
            clip_mask = (wl >= start) & (wl <= stop)
            output_wl = wl[clip_mask]
            output_inten = inten[clip_mask]
            if output_wl.size == 0:
                raise ValueError("Spectrum does not overlap requested domain")

    meta = dict(spec.meta)
    if resampled:
        channels = dict(meta.get("channels") or {})
        channels["original_wavelength"] = np.asarray(preserved_wl, dtype=float).copy()
        channels["original_intensity"] = np.asarray(preserved_inten, dtype=float).copy()
        meta["channels"] = channels
    if domain:
        domain_meta = {
            "original_min_nm": original_min,
            "original_max_nm": original_max,
            "output_min_nm": float(output_wl[0]) if output_wl.size else float("nan"),
            "output_max_nm": float(output_wl[-1]) if output_wl.size else float("nan"),
        }
        if requested_min is not None:
            domain_meta["requested_min_nm"] = requested_min
        if requested_max is not None:
            domain_meta["requested_max_nm"] = requested_max
        meta["wavelength_domain_nm"] = domain_meta

    _update_channel(meta, "raw", output_inten, overwrite=True)
    return Spectrum(
        wavelength=np.asarray(output_wl, dtype=float).copy(),
        intensity=np.asarray(output_inten, dtype=float).copy(),
        meta=meta,
    )


def subtract_blank(
    sample: Spectrum,
    blank: Spectrum,
    *,
    audit: Dict[str, object] | None = None,
) -> Spectrum:
    """Subtract a blank from the sample after verifying domain overlap.

    Parameters
    ----------
    sample:
        Sample spectrum that will have the blank removed.
    blank:
        Blank spectrum that should be subtracted from ``sample``.
    audit:
        Optional metadata recorded for quality-control auditing. Any supplied
        fields will be merged with statistics gathered during subtraction.
    """

    sample_wl = np.asarray(sample.wavelength, dtype=float)
    blank_interp = np.interp(
        sample_wl,
        np.asarray(blank.wavelength, dtype=float),
        np.asarray(blank.intensity, dtype=float),
        left=np.nan,
        right=np.nan,
    )

    valid = np.isfinite(blank_interp)
    if valid.sum() < 2:
        raise ValueError("Blank spectrum does not overlap sample domain sufficiently")

    corrected = sample.intensity.astype(float).copy()
    corrected[valid] = corrected[valid] - blank_interp[valid]
    corrected[~valid] = np.nan

    overlap_points = int(valid.sum())
    overlap_fraction = float(overlap_points) / float(sample_wl.size)
    blank_mean = float(np.nanmean(blank_interp[valid])) if overlap_points else float("nan")

    meta = dict(sample.meta)
    meta["blank_subtracted"] = True
    audit_payload: Dict[str, object] = {
        "overlap_points": overlap_points,
        "overlap_fraction": overlap_fraction,
        "blank_mean_intensity": blank_mean,
    }
    if audit:
        audit_payload.update(audit)

    existing_audit = meta.get("blank_audit")
    if isinstance(existing_audit, dict):
        merged_audit = dict(existing_audit)
        for key, value in audit_payload.items():
            merged_audit.setdefault(key, value)
    else:
        merged_audit = audit_payload

    meta["blank_audit"] = merged_audit
    _update_channel(meta, "blanked", corrected, overwrite=True)
    return Spectrum(
        wavelength=np.asarray(sample.wavelength, dtype=float).copy(),
        intensity=np.asarray(corrected, dtype=float).copy(),
        meta=meta,
    )


def _baseline_asls(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = _nan_safe(y)
    L = y.size
    if L < 3:
        return np.zeros_like(y)

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(max(1, int(niter))):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def _lower_hull(points: np.ndarray) -> List[Tuple[float, float]]:
    hull: List[Tuple[float, float]] = []
    for x, y in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2:]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    return hull


def _baseline_rubberband(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = _nan_safe(np.asarray(y, dtype=float))
    if x.size < 2:
        return np.zeros_like(y)

    points = np.column_stack((x, y))
    hull = _lower_hull(points)
    hull_x, hull_y = zip(*hull)
    baseline = np.interp(x, np.asarray(hull_x), np.asarray(hull_y))
    return baseline


def _baseline_snip(y: np.ndarray, iterations: int = 24) -> np.ndarray:
    y = _nan_safe(np.asarray(y, dtype=float))
    n = y.size
    if n < 3:
        return np.zeros_like(y)

    baseline = y.copy()
    max_k = min(int(iterations), n // 2)
    for k in range(1, max_k + 1):
        left = np.roll(baseline, k)
        right = np.roll(baseline, -k)
        left[:k] = baseline[:k]
        right[-k:] = baseline[-k:]
        avg = (left + right) / 2.0
        baseline = np.minimum(baseline, avg)
    return baseline


def apply_baseline(
    spec: Spectrum,
    method: str,
    *,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
    iterations: int = 24,
    anchor: Dict[str, object] | None = None,
) -> Spectrum:
    """Baseline correction entry point."""

    y = np.asarray(spec.intensity, dtype=float)
    method = (method or "").lower()
    if method == "asls":
        baseline = _baseline_asls(y, lam=lam, p=p, niter=niter)
    elif method == "rubberband":
        baseline = _baseline_rubberband(spec.wavelength, y)
    elif method == "snip":
        baseline = _baseline_snip(y, iterations=iterations)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    corrected = y - baseline
    anchor_meta: Dict[str, object] | None = None
    if anchor:
        corrected, anchor_meta = _apply_baseline_anchor(
            np.asarray(spec.wavelength, dtype=float), corrected, anchor
        )
    meta = dict(spec.meta)
    meta.setdefault("baseline", method)
    if anchor_meta is not None:
        meta["baseline_anchor"] = anchor_meta
    _update_channel(meta, "baseline_corrected", corrected, overwrite=True)
    return Spectrum(
        wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
        intensity=np.asarray(corrected, dtype=float).copy(),
        meta=meta,
    )


def _apply_baseline_anchor(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    anchor_cfg: Dict[str, object],
) -> tuple[np.ndarray, Dict[str, object]]:
    """Adjust ``intensity`` so that anchor windows hit their targets."""

    enabled = True
    if isinstance(anchor_cfg, dict):
        enabled = bool(anchor_cfg.get("enabled", True))
        windows_cfg = (
            anchor_cfg.get("windows")
            or anchor_cfg.get("ranges")
            or anchor_cfg.get("anchors")
        )
    else:
        windows_cfg = anchor_cfg

    if not enabled:
        return intensity, {
            "enabled": False,
            "applied": False,
            "windows": [],
            "config": anchor_cfg,
        }

    norm_windows = _normalise_anchor_windows(windows_cfg)
    if not norm_windows:
        return intensity, {
            "enabled": True,
            "applied": False,
            "windows": [],
            "config": anchor_cfg,
        }

    wl = np.asarray(wavelength, dtype=float)
    corrected = np.asarray(intensity, dtype=float).copy()

    anchor_records: List[Dict[str, object]] = []
    centers: List[float] = []
    offsets: List[float] = []

    for window in norm_windows:
        lo = window.get("lower_nm")
        hi = window.get("upper_nm")
        target = float(window.get("target", 0.0) or 0.0)

        mask = np.ones_like(wl, dtype=bool)
        if lo is not None:
            mask &= wl >= lo
        if hi is not None:
            mask &= wl <= hi

        if not np.any(mask):
            record = {
                "lower_nm": lo,
                "upper_nm": hi,
                "target_level": target,
                "points": 0,
                "window_mean": float("nan"),
                "applied_offset": float("nan"),
                "center_nm": float("nan"),
            }
            if window.get("label") is not None:
                record["label"] = window["label"]
            anchor_records.append(record)
            continue

        window_values = corrected[mask]
        finite_mask = np.isfinite(window_values)
        points = int(finite_mask.sum())
        if points == 0:
            record = {
                "lower_nm": lo,
                "upper_nm": hi,
                "target_level": target,
                "points": 0,
                "window_mean": float("nan"),
                "applied_offset": float("nan"),
                "center_nm": float("nan"),
            }
            if window.get("label") is not None:
                record["label"] = window["label"]
            anchor_records.append(record)
            continue

        window_wl = wl[mask][finite_mask]
        window_vals = window_values[finite_mask]
        window_mean = float(np.nanmean(window_vals))
        offset = float(window_mean - target)
        center = float(np.nanmean(window_wl)) if window_wl.size else float("nan")

        centers.append(center if np.isfinite(center) else float(np.nanmean(wl)))
        offsets.append(offset)

        record = {
            "lower_nm": lo,
            "upper_nm": hi,
            "target_level": target,
            "points": points,
            "window_mean": window_mean,
            "applied_offset": offset,
            "center_nm": center,
        }
        if window.get("label") is not None:
            record["label"] = window["label"]
        anchor_records.append(record)

    valid = [idx for idx, center in enumerate(centers) if np.isfinite(center)]
    if not valid:
        meta = {
            "enabled": True,
            "applied": False,
            "windows": anchor_records,
            "config": anchor_cfg,
        }
        return corrected, meta

    centers_arr = np.asarray([centers[idx] for idx in valid], dtype=float)
    offsets_arr = np.asarray([offsets[idx] for idx in valid], dtype=float)
    order = np.argsort(centers_arr)
    centers_sorted = centers_arr[order]
    offsets_sorted = offsets_arr[order]

    correction = np.interp(
        wl,
        centers_sorted,
        offsets_sorted,
        left=offsets_sorted[0],
        right=offsets_sorted[-1],
    )
    corrected -= correction

    for record in anchor_records:
        lo = record["lower_nm"]
        hi = record["upper_nm"]
        target_level = record["target_level"]
        mask = np.ones_like(wl, dtype=bool)
        if lo is not None:
            mask &= wl >= lo
        if hi is not None:
            mask &= wl <= hi
        if np.any(mask):
            corrected[mask] = target_level

    meta = {
        "enabled": True,
        "applied": True,
        "windows": anchor_records,
        "config": anchor_cfg,
        "correction_min": float(np.nanmin(correction)),
        "correction_max": float(np.nanmax(correction)),
        "correction_mean": float(np.nanmean(correction)),
    }
    return corrected, meta


def _normalise_anchor_windows(windows_cfg: object) -> List[Dict[str, object]]:
    """Normalise user-supplied window definitions."""

    if windows_cfg is None:
        return []

    if isinstance(windows_cfg, dict):
        windows_iterable = windows_cfg.get("windows")
        if windows_iterable is None:
            return []
    else:
        windows_iterable = windows_cfg

    if not isinstance(windows_iterable, Sequence):
        return []

    normalised: List[Dict[str, object]] = []
    for entry in windows_iterable:
        if entry is None:
            continue
        if isinstance(entry, dict):
            lo = (
                entry.get("min_nm")
                if entry.get("min_nm") is not None
                else entry.get("lower_nm")
            )
            if lo is None:
                lo = entry.get("min")
            if lo is None:
                lo = entry.get("lower")
            lo_val = float(lo) if lo is not None else None

            hi = (
                entry.get("max_nm")
                if entry.get("max_nm") is not None
                else entry.get("upper_nm")
            )
            if hi is None:
                hi = entry.get("max")
            if hi is None:
                hi = entry.get("upper")
            hi_val = float(hi) if hi is not None else None

            target = entry.get("target")
            if target is None:
                target = entry.get("level")
            if target is None:
                target = entry.get("value")
            target_val = float(target) if target is not None else 0.0

            label = entry.get("label")
        elif isinstance(entry, Sequence):
            data = list(entry)
            if len(data) < 2:
                continue
            lo_val = float(data[0]) if data[0] is not None else None
            hi_val = float(data[1]) if data[1] is not None else None
            target_val = float(data[2]) if len(data) > 2 and data[2] is not None else 0.0
            label = None
        else:
            continue

        if lo_val is not None and hi_val is not None and lo_val > hi_val:
            lo_val, hi_val = hi_val, lo_val

        window_payload: Dict[str, object] = {
            "lower_nm": lo_val,
            "upper_nm": hi_val,
            "target": target_val,
        }
        if label is not None:
            window_payload["label"] = str(label)
        normalised.append(window_payload)

    return normalised


def normalise_join_windows(windows_cfg: object) -> List[Dict[str, float | None]]:
    """Normalise a join window configuration into explicit numeric bounds."""

    if windows_cfg is None:
        return []

    windows = _normalise_anchor_windows(windows_cfg)
    normalised: List[Dict[str, float | None]] = []
    for entry in windows:
        lower = entry.get("lower_nm")
        upper = entry.get("upper_nm")
        lower_val = float(lower) if lower is not None else None
        upper_val = float(upper) if upper is not None else None
        normalised.append({"lower_nm": lower_val, "upper_nm": upper_val})
    return normalised


def detect_joins(
    wavelength: Sequence[float],
    intensity: Sequence[float],
    *,
    threshold: float | None = None,
    window: int = 5,
    windows: Sequence[Mapping[str, float | int | None]] | None = None,
) -> List[int]:
    """Detect detector joins using change-point analysis of overlap windows.

    When ``windows`` are supplied only change-points whose wavelength falls
    within the specified bounds are considered.
    """

    y = _nan_safe(np.asarray(intensity, dtype=float))
    n = y.size
    raw_window = max(1, int(window))
    half_window = max(3, raw_window)
    if half_window > n // 2:
        half_window = max(2, n // 2)
    if n < 2 * half_window:
        return []

    # Pre-compute rolling window medians for the change statistic.
    scores = np.full(n, np.nan, dtype=float)
    deltas = np.full(n, np.nan, dtype=float)
    for idx in range(half_window, n - half_window + 1):
        left = y[idx - half_window : idx]
        right = y[idx : idx + half_window]
        if left.size == 0 or right.size == 0:
            continue
        left_median = float(np.nanmedian(left))
        right_median = float(np.nanmedian(right))
        if not np.isfinite(left_median) or not np.isfinite(right_median):
            continue
        delta = right_median - left_median
        deltas[idx] = delta
        scores[idx] = abs(delta)

    valid_positions = np.where(np.isfinite(scores))[0]
    if windows is not None:
        bounds = normalise_join_windows(windows)
        if not bounds:
            return []
        wl = np.asarray(wavelength, dtype=float)
        if wl.size != n:
            raise ValueError("Wavelength axis and intensity must be the same length")
        if not np.any(np.isfinite(wl)):
            return []
        mask = np.zeros_like(wl, dtype=bool)
        for entry in bounds:
            lo = entry.get("lower_nm")
            hi = entry.get("upper_nm")
            if lo is None and hi is None:
                mask |= np.isfinite(wl)
                break
            lo_val = float(lo) if lo is not None else None
            hi_val = float(hi) if hi is not None else None
            if lo_val is not None and hi_val is not None and lo_val > hi_val:
                lo_val, hi_val = hi_val, lo_val
            window_condition = np.isfinite(wl)
            if lo_val is not None:
                window_condition &= wl >= lo_val
            if hi_val is not None:
                window_condition &= wl <= hi_val
            mask |= window_condition
        mask &= np.isfinite(wl)
        if not np.any(mask):
            return []
        valid_positions = valid_positions[mask[valid_positions]]
    if valid_positions.size == 0:
        return []

    candidate_scores = scores[valid_positions]
    if threshold is None:
        median = float(np.nanmedian(candidate_scores))
        mad = float(np.nanmedian(np.abs(candidate_scores - median)))
        peak = float(np.nanmax(candidate_scores))
        if not np.isfinite(peak) or peak <= 0:
            return []
        if not np.isfinite(mad) or mad == 0:
            threshold = peak
        else:
            computed = median + 6.0 * 1.4826 * mad
            if not np.isfinite(computed) or computed <= 0:
                threshold = peak
            elif peak <= computed:
                ratio = abs(mad / peak) if peak else float("inf")
                if ratio < 0.25:
                    return []
                threshold = peak
            else:
                threshold = computed
    else:
        threshold = float(threshold)

    if not np.isfinite(threshold) or threshold <= 0:
        return []

    result: list[int] = []

    def _binary_segment(start: int, stop: int) -> None:
        segment_mask = (valid_positions >= start) & (valid_positions < stop)
        segment_positions = valid_positions[segment_mask]
        if segment_positions.size == 0:
            return
        segment_scores = scores[segment_positions]
        best_score = float(np.nanmax(segment_scores))
        if not np.isfinite(best_score):
            return
        maxima_mask = np.isclose(segment_scores, best_score, rtol=0.0, atol=1e-12)
        maxima_positions = segment_positions[maxima_mask]
        if maxima_positions.size == 0:
            return
        center = (start + stop) / 2.0
        representative_delta = float(deltas[int(maxima_positions[0])])
        if np.isfinite(representative_delta) and representative_delta != 0:
            median_idx = int(np.median(maxima_positions))
            best_idx = median_idx
        else:
            best_idx = int(maxima_positions[int(np.argmin(np.abs(maxima_positions - center)))])
        eligible_positions = [
            int(pos)
            for pos in maxima_positions
            if pos - start >= half_window and stop - pos >= half_window
        ]
        if (best_idx - start < half_window or stop - best_idx < half_window) and eligible_positions:
            best_idx = int(np.median(eligible_positions))
        if not np.isfinite(best_score) or best_score < threshold:
            return
        if best_idx - start < half_window or stop - best_idx < half_window:
            return
        delta_value = float(deltas[best_idx])
        sign = 0
        if np.isfinite(delta_value):
            if delta_value > 0:
                sign = 1
            elif delta_value < 0:
                sign = -1
        left_idx = best_idx
        while (
            left_idx - 1 >= start
            and left_idx - 1 >= best_idx - (half_window - 1)
            and scores[left_idx - 1] >= threshold
            and (sign == 0 or not np.isfinite(deltas[left_idx - 1]) or np.sign(deltas[left_idx - 1]) == sign)
        ):
            left_idx -= 1
        right_idx = best_idx
        max_right = best_idx + (half_window - 1)
        while (
            right_idx + 1 < stop
            and right_idx + 1 <= max_right
            and scores[right_idx + 1] >= threshold
            and (sign == 0 or not np.isfinite(deltas[right_idx + 1]) or np.sign(deltas[right_idx + 1]) == sign)
        ):
            right_idx += 1
        if sign > 0:
            if raw_window <= 2:
                chosen = right_idx
            elif raw_window == 3:
                chosen = left_idx
            else:
                chosen = best_idx
        elif sign < 0:
            if raw_window <= 3:
                chosen = left_idx
            else:
                chosen = best_idx
        else:
            chosen = int(round((left_idx + right_idx) / 2.0))
        result.append(chosen)
        _binary_segment(start, chosen)
        _binary_segment(chosen, stop)

    _binary_segment(half_window, n - half_window + 1)

    if not result:
        return []

    result = sorted(set(idx for idx in result if half_window <= idx < n))
    filtered: list[int] = []
    min_spacing = max(1, raw_window)
    for idx in result:
        if filtered and idx - filtered[-1] < min_spacing:
            prev = filtered[-1]
            if scores[idx] > scores[prev]:
                filtered[-1] = idx
            continue
        filtered.append(idx)

    if not filtered:
        return []

    diffs = np.abs(np.diff(y))
    refined: list[int] = []
    for idx in filtered:
        lo = max(1, idx - half_window)
        hi = min(n - 1, idx + half_window)
        window_diffs = diffs[lo - 1 : hi]
        if window_diffs.size == 0:
            refined.append(idx)
            continue
        rel = int(np.nanargmax(window_diffs))
        refined_idx = lo - 1 + rel + 1
        delta_value = deltas[idx] if 0 <= idx < deltas.size else float("nan")
        if np.isfinite(delta_value):
            if delta_value > 0 and refined_idx < idx:
                refined_idx = idx
            elif delta_value < 0 and refined_idx > idx:
                refined_idx = idx
        refined.append(refined_idx)

    refined_sorted: list[int] = []
    for idx in sorted(set(refined)):
        if 0 < idx < n:
            if refined_sorted and idx - refined_sorted[-1] < min_spacing:
                if diffs[idx - 1] > diffs[refined_sorted[-1] - 1]:
                    refined_sorted[-1] = idx
                continue
            refined_sorted.append(idx)

    if refined_sorted:
        filtered_candidates: list[int] = []
        for idx in refined_sorted:
            if idx >= scores.size:
                continue
            score = float(scores[idx])
            if not np.isfinite(score) or score >= threshold:
                filtered_candidates.append(idx)
        refined_sorted = filtered_candidates

    return refined_sorted


def correct_joins(
    spec: Spectrum,
    join_indices: Iterable[int],
    *,
    window: int = 10,
    offset_bounds: tuple[float | None, float | None] | None = None,
) -> Spectrum:
    """Apply offset corrections after joins using neighbouring medians."""

    raw_values = np.asarray(spec.intensity, dtype=float)
    corrected = raw_values.astype(float).copy()
    n = corrected.size
    window = max(1, int(window))

    min_offset = max_offset = None
    if offset_bounds is not None:
        min_offset, max_offset = offset_bounds
        if min_offset is not None:
            min_offset = float(min_offset)
        if max_offset is not None:
            max_offset = float(max_offset)

    joins = sorted({int(idx) for idx in join_indices if 0 < int(idx) < n})

    offsets: list[float] = []
    raw_offsets: list[float] = []
    pre_left_means: list[float] = []
    pre_right_means: list[float] = []
    post_left_means: list[float] = []
    post_right_means: list[float] = []
    pre_deltas: list[float] = []
    post_deltas: list[float] = []
    pre_overlap_errors: list[float] = []
    post_overlap_errors: list[float] = []

    for join in joins:
        left_slice = slice(max(0, join - window), join)
        right_slice = slice(join, min(n, join + window))
        left_window_raw = raw_values[left_slice]
        right_window_raw = raw_values[right_slice]
        if left_window_raw.size == 0 or right_window_raw.size == 0:
            offsets.append(float("nan"))
            pre_left_means.append(float("nan"))
            pre_right_means.append(float("nan"))
            post_left_means.append(float("nan"))
            post_right_means.append(float("nan"))
            pre_deltas.append(float("nan"))
            post_deltas.append(float("nan"))
            pre_overlap_errors.append(float("nan"))
            post_overlap_errors.append(float("nan"))
            continue

        left_mean_pre = float(np.nanmean(left_window_raw))
        right_mean_pre = float(np.nanmean(right_window_raw))
        pre_left_means.append(left_mean_pre)
        pre_right_means.append(right_mean_pre)
        pre_delta = float(right_mean_pre - left_mean_pre)
        pre_deltas.append(pre_delta)

        overlap_len = min(left_window_raw.size, right_window_raw.size)
        if overlap_len:
            left_tail_pre = left_window_raw[-overlap_len:]
            right_head_pre = right_window_raw[:overlap_len]
            pre_overlap = float(np.nanmean(np.abs(right_head_pre - left_tail_pre)))
        else:
            pre_overlap = float("nan")
        pre_overlap_errors.append(pre_overlap)

        offset = np.nanmedian(right_window_raw) - np.nanmedian(left_window_raw)
        raw_offsets.append(float(offset) if np.isfinite(offset) else float("nan"))
        if np.isfinite(offset):
            applied_offset = float(offset)
            if min_offset is not None and np.isfinite(min_offset):
                applied_offset = max(applied_offset, min_offset)
            if max_offset is not None and np.isfinite(max_offset):
                applied_offset = min(applied_offset, max_offset)
            corrected[join:] -= applied_offset
            offsets.append(applied_offset)
        else:
            offsets.append(float("nan"))

        left_window_post = corrected[left_slice]
        right_window_post = corrected[right_slice]
        left_mean_post = float(np.nanmean(left_window_post)) if left_window_post.size else float("nan")
        right_mean_post = float(np.nanmean(right_window_post)) if right_window_post.size else float("nan")
        post_left_means.append(left_mean_post)
        post_right_means.append(right_mean_post)
        post_delta = float(right_mean_post - left_mean_post)
        post_deltas.append(post_delta)

        if overlap_len:
            left_tail_post = left_window_post[-overlap_len:]
            right_head_post = right_window_post[:overlap_len]
            post_overlap = float(np.nanmean(np.abs(right_head_post - left_tail_post)))
        else:
            post_overlap = float("nan")
        post_overlap_errors.append(post_overlap)

    boundaries = [0, *joins, n]
    raw_segments = [
        np.asarray(raw_values[start:stop], dtype=float).tolist()
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    corrected_segments = [
        np.asarray(corrected[start:stop], dtype=float).tolist()
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]

    meta = dict(spec.meta)
    if joins:
        meta.setdefault("join_corrected", True)

    join_segments_meta = dict(meta.get("join_segments") or {})
    join_segments_meta.setdefault("raw", raw_segments)
    join_segments_meta["corrected"] = corrected_segments
    join_segments_meta["indices"] = [int(idx) for idx in joins]
    join_segments_meta["window"] = window
    meta["join_segments"] = join_segments_meta

    join_stats = {
        "indices": [int(idx) for idx in joins],
        "window": window,
        "offsets": offsets,
        "raw_offsets": raw_offsets,
        "pre_means": {"left": pre_left_means, "right": pre_right_means},
        "post_means": {"left": post_left_means, "right": post_right_means},
        "pre_deltas": pre_deltas,
        "post_deltas": post_deltas,
        "pre_overlap_errors": pre_overlap_errors,
        "post_overlap_errors": post_overlap_errors,
    }
    if offset_bounds is not None:
        join_stats["offset_bounds"] = {
            "min": float(min_offset) if min_offset is not None else None,
            "max": float(max_offset) if max_offset is not None else None,
        }
    meta["join_statistics"] = join_stats

    channels = dict(meta.get("channels") or {})
    if "join_raw" not in channels:
        channels["join_raw"] = np.asarray(raw_values, dtype=float).copy()
    channels["join_corrected"] = np.asarray(corrected, dtype=float).copy()
    meta["channels"] = channels
    _update_channel(meta, "joined", corrected, overwrite=True)
    return Spectrum(
        wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
        intensity=np.asarray(corrected, dtype=float).copy(),
        meta=meta,
    )


def despike_spectrum(
    spec: Spectrum,
    *,
    zscore: float = 5.0,
    window: int = 5,
    join_indices: Sequence[int] | None = None,
) -> Spectrum:
    """Replace spikes using a moving median filter with robust detection.

    When ``join_indices`` are supplied the spectrum is segmented and each
    portion is processed independently so spikes near detector joins are
    corrected without smoothing across the discontinuity.
    """

    y = np.asarray(spec.intensity, dtype=float)

    def _final_spectrum(values: np.ndarray, *, mask: np.ndarray | None = None) -> Spectrum:
        meta = dict(spec.meta)
        if mask is not None and np.any(mask):
            meta.setdefault("despiked", True)
        _update_channel(meta, "despiked", values, overwrite=True)
        return Spectrum(
            wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
            intensity=np.asarray(values, dtype=float).copy(),
            meta=meta,
        )

    if y.size < 3:
        return _final_spectrum(y.copy())

    base_window = int(window)
    if base_window <= 1:
        base_window = 3
    if base_window % 2 == 0:
        base_window += 1
    if base_window < 3:
        return _final_spectrum(y.copy())

    n = y.size

    def _effective_window(length: int) -> int | None:
        if length < 3:
            return None
        eff = base_window
        if eff > length:
            eff = length if length % 2 else length - 1
        if eff < 3:
            return None
        return eff

    joins: list[int] = []
    if join_indices:
        joins = sorted({int(idx) for idx in join_indices if 0 < int(idx) < n})

    if not joins:
        window_size = _effective_window(n)
        if window_size is None:
            return _final_spectrum(y.copy())

        baseline = medfilt(y, kernel_size=window_size)
        residual = y - baseline
        mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
        if not np.isfinite(mad) or mad == 0:
            return _final_spectrum(y.copy())

        threshold = float(zscore) * 1.4826 * mad
        mask = np.abs(residual) > threshold
        corrected = y.copy()
        if np.any(mask):
            indices = np.arange(n)
            good = indices[~mask]
            if good.size >= 2:
                corrected[mask] = np.interp(indices[mask], good, corrected[~mask])
            else:
                corrected[mask] = baseline[mask]
        return _final_spectrum(corrected, mask=mask)

    boundaries = [0, *joins, n]
    corrected = y.copy()
    overall_mask = np.zeros(n, dtype=bool)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = y[start:stop]
        seg_len = segment.size
        seg_window = _effective_window(seg_len)
        if seg_window is None:
            continue

        baseline = medfilt(segment, kernel_size=seg_window)
        residual = segment - baseline
        mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
        if np.isfinite(mad) and mad > 0:
            threshold = float(zscore) * 1.4826 * mad
            mask = np.abs(residual) > threshold
        else:
            nonzero = np.abs(residual) > 1e-12
            if not np.any(nonzero) or np.count_nonzero(nonzero) >= seg_len:
                continue
            mask = nonzero
        if not np.any(mask):
            continue

        indices = np.arange(seg_len)
        good = indices[~mask]
        target = corrected[start:stop]
        if good.size >= 2:
            target[mask] = np.interp(indices[mask], good, target[~mask])
        else:
            target[mask] = baseline[mask]
        overall_mask[start:stop] = mask

    return _final_spectrum(corrected, mask=overall_mask)


def smooth_spectrum(
    spec: Spectrum,
    *,
    window: int,
    polyorder: int,
    join_indices: Sequence[int] | None = None,
) -> Spectrum:
    """Savitzky–Golay smoothing wrapper."""

    y = np.asarray(spec.intensity, dtype=float)
    window = int(window)
    polyorder = int(polyorder)
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd")
    if window <= polyorder:
        raise ValueError("Savitzky-Golay window must exceed polynomial order")
    if window > y.size:
        raise ValueError("Savitzky-Golay window larger than spectrum length")

    n = y.size

    joins: list[int] = []
    if join_indices:
        joins = sorted({int(idx) for idx in join_indices if 0 < int(idx) < n})

    segments: list[tuple[int, int]] = []
    last = 0
    for join in joins:
        if join <= last or join >= n:
            continue
        segments.append((last, join))
        last = join
    if last < n:
        segments.append((last, n))
    if not segments:
        segments = [(0, n)]

    def _effective_window(length: int) -> int | None:
        eff = min(window, length)
        if eff % 2 == 0:
            eff -= 1
        while eff >= 3:
            if eff > polyorder:
                return eff
            eff -= 2
        return None

    smoothed = y.copy()
    for start, stop in segments:
        seg_len = stop - start
        if seg_len <= polyorder or seg_len < 3:
            continue
        seg_window = _effective_window(seg_len)
        if seg_window is None:
            continue
        segment = y[start:stop]
        smoothed[start:stop] = savgol_filter(
            segment, window_length=seg_window, polyorder=polyorder, mode="interp"
        )

    segmented = len(segments) > 1

    meta = dict(spec.meta)
    meta.setdefault("smoothed", True)
    if segmented:
        meta.setdefault("smoothed_segmented", True)
    _update_channel(meta, "smoothed", smoothed, overwrite=True)
    return Spectrum(
        wavelength=np.asarray(spec.wavelength, dtype=float).copy(),
        intensity=np.asarray(smoothed, dtype=float).copy(),
        meta=meta,
    )


def normalize_blank_match_strategy(strategy: object | None) -> str:
    if isinstance(strategy, str):
        text = strategy.strip().lower().replace("-", "_")
    else:
        text = ""
    if text in {"cuvette_slot", "slot", "cuvette"}:
        return "cuvette_slot"
    return "blank_id"


def _coerce_identifier(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def blank_match_identifiers(spec: Spectrum, strategy: str | None = None) -> List[str]:
    role = str(spec.meta.get("role", "sample"))
    if role != "blank":
        return []

    canonical = normalize_blank_match_strategy(strategy)
    identifiers: List[str] = []

    if canonical == "cuvette_slot":
        slot = _coerce_identifier(spec.meta.get("cuvette_slot"))
        if slot is not None:
            identifiers.append(slot)

    suffix = "_indref"
    for candidate in (
        spec.meta.get("blank_id"),
        spec.meta.get("sample_id"),
        spec.meta.get("channel"),
    ):
        coerced = _coerce_identifier(candidate)
        if coerced is None:
            continue
        if coerced not in identifiers:
            identifiers.append(coerced)
        if coerced.lower().endswith(suffix):
            base = coerced[: -len(suffix)]
            if base and base not in identifiers:
                identifiers.append(base)

    return identifiers


def blank_match_identifier(spec: Spectrum, strategy: str | None = None) -> str | None:
    identifiers = blank_match_identifiers(spec, strategy)
    return identifiers[0] if identifiers else None


def blank_replicate_key(spec: Spectrum, strategy: str | None = None) -> Tuple[str, str]:
    role = str(spec.meta.get("role", "sample"))
    if role != "blank":
        return replicate_key(spec)

    identifiers = blank_match_identifiers(spec, strategy)
    ident: str | None
    if identifiers:
        ident = identifiers[0]
    else:
        ident = None
    if ident is None:
        ident = f"blank_{id(spec)}"
    return "blank", ident


def replicate_key(spec: Spectrum) -> Tuple[str, str]:
    role = str(spec.meta.get("role", "sample"))
    if role == "blank":
        ident = blank_match_identifier(spec)
    else:
        ident = _coerce_identifier(spec.meta.get("sample_id"))
        if ident is None:
            ident = _coerce_identifier(spec.meta.get("channel"))
        if ident is None:
            ident = _coerce_identifier(spec.meta.get("id"))
    if ident is None:
        ident = f"{role}_{id(spec)}"
    return role, str(ident)


def _replicate_feature_matrix(
    wavelength: np.ndarray,
    stack: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Derive a compact feature matrix characterising replicate spectra."""

    features: List[List[float]] = []
    wl = np.asarray(wavelength, dtype=float)
    for row, mask in zip(stack, valid_mask):
        row = np.asarray(row, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            features.append([1.0, 0.0, 0.0, 0.0, 0.0])
            continue

        finite_row = row[mask]
        finite_wl = wl[mask]
        mean_val = float(np.mean(finite_row))
        std_val = float(np.std(finite_row))
        if finite_row.size >= 2:
            span = float(finite_wl[-1] - finite_wl[0])
            slope_val = float((finite_row[-1] - finite_row[0]) / span) if span else 0.0
            energy_val = float(np.mean(np.abs(np.diff(finite_row))))
        else:
            slope_val = 0.0
            energy_val = 0.0
        area_val = (
            float(np.trapezoid(finite_row, finite_wl)) if finite_row.size >= 2 else mean_val
        )
        features.append([1.0, mean_val, std_val, slope_val, energy_val, area_val])

    matrix = np.asarray(features, dtype=float)
    n_obs = matrix.shape[0]
    if n_obs > 1 and matrix.shape[1] >= n_obs:
        matrix = matrix[:, : max(1, n_obs - 1)]
    if matrix.shape[1] > 1:
        others = matrix[:, 1:]
        other_mean = np.mean(others, axis=0)
        other_std = np.std(others, axis=0)
        safe_std = np.where(other_std < 1e-12, 1.0, other_std)
        matrix[:, 1:] = (others - other_mean) / safe_std
    return matrix


def _compute_cooksd_scores(
    wavelength: np.ndarray, stack: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Cook's distance scores and leverage for each replicate."""

    if stack.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    valid_mask = np.isfinite(stack)
    n_obs = stack.shape[0]
    if n_obs == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    with np.errstate(invalid="ignore"):
        column_means = np.nanmean(stack, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    filled = np.where(valid_mask, stack, column_means)

    feature_matrix = _replicate_feature_matrix(wavelength, filled, valid_mask)
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] != n_obs:
        return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=float)

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    hat = X @ XtX_inv @ X.T
    leverage = np.clip(np.diag(hat).real, 0.0, 1.0)

    rank = int(np.linalg.matrix_rank(X))
    rank = max(rank, 1)

    # Solve the multivariate least-squares system using the filled values.
    coeffs, _, _, _ = np.linalg.lstsq(X, filled, rcond=None)
    fitted = X @ coeffs
    residual = filled - fitted
    residual[~valid_mask] = 0.0
    residual_sq = residual**2
    rss_per_obs = residual_sq.sum(axis=1)
    sse_total = float(residual_sq.sum())

    effective_q = int(np.sum(valid_mask) / max(n_obs, 1))
    effective_q = max(effective_q, 1)
    dof = max(n_obs - rank, 1)
    mse = sse_total / (dof * effective_q)

    scores = np.zeros(n_obs, dtype=float)
    if mse <= 0 or not np.isfinite(mse):
        return scores, leverage

    denom = (1.0 - leverage) ** 2
    safe = denom > 1e-12
    p_eff = max(rank, 1)

    scores[safe] = (
        rss_per_obs[safe] / (p_eff * mse)
    ) * (leverage[safe] / denom[safe])
    scores[~safe] = np.inf
    scores = np.where(np.isfinite(scores), scores, np.inf)
    return scores, leverage


def average_replicates(
    specs: Sequence[Spectrum],
    *,
    return_mapping: bool = False,
    outlier: Dict[str, object] | bool | None = None,
    key_func: Callable[[Spectrum], Tuple[str, str]] | None = None,
) -> Tuple[List[Spectrum], Dict[Tuple[str, str], Spectrum]] | List[Spectrum]:
    """Average replicate spectra with optional outlier rejection.

    Parameters
    ----------
    specs:
        Sequence of spectra to be grouped and averaged by replicate key.
    return_mapping:
        When ``True`` both the averaged spectra and a mapping keyed by
        replicate identifier are returned.
    outlier:
        Optional configuration enabling the rejection of replicate spectra
        deemed outliers. When provided, the configuration may specify
        ``method`` (currently only ``"mad"`` is supported), ``threshold``
        defining the score above which replicates are discarded, and an
        ``enabled`` flag. A simple boolean can be passed to toggle the
        default configuration.
    """

    if outlier is None or outlier is False:
        outlier_cfg: Dict[str, object] = {}
    elif isinstance(outlier, dict):
        outlier_cfg = dict(outlier)
    elif isinstance(outlier, bool):
        outlier_cfg = {"enabled": outlier}
    else:
        raise TypeError("'outlier' must be a mapping, boolean or None")

    outlier_enabled = bool(outlier_cfg.get("enabled", bool(outlier_cfg)))
    outlier_method = str(outlier_cfg.get("method", "mad")).lower()
    outlier_threshold = float(outlier_cfg.get("threshold", 3.5))

    groups: "OrderedDict[Tuple[str, str], List[Spectrum]]" = OrderedDict()
    for spec in specs:
        key = key_func(spec) if key_func is not None else replicate_key(spec)
        groups.setdefault(key, []).append(spec)

    outputs: List[Spectrum] = []
    mapping: Dict[Tuple[str, str], Spectrum] = {}

    for key, members in groups.items():
        sources = [
            member.meta.get("source_file")
            or member.meta.get("channel")
            or member.meta.get("sample_id")
            for member in members
        ]

        if len(members) == 1:
            member = members[0]
            meta = dict(member.meta)
            meta.setdefault("replicate_sources", sources)
            meta["replicate_total"] = 1
            meta["replicate_excluded"] = []
            meta["replicates"] = [
                {
                    "wavelength": np.asarray(member.wavelength, dtype=float).tolist(),
                    "intensity": np.asarray(member.intensity, dtype=float).tolist(),
                    "meta": dict(member.meta),
                    "excluded": False,
                    "score": None,
                }
            ]
            averaged = Spectrum(
                wavelength=np.asarray(member.wavelength, dtype=float).copy(),
                intensity=np.asarray(member.intensity, dtype=float).copy(),
                meta=meta,
            )
        else:
            ref_wl = members[0].wavelength
            for other in members[1:]:
                if not np.array_equal(other.wavelength, ref_wl):
                    raise ValueError("Replicates must share identical domains for averaging")

            stack = np.vstack([np.asarray(m.intensity, dtype=float) for m in members])

            method_aliases = {
                "median_absolute_deviation": "mad",
                "cook": "cooksd",
                "cookd": "cooksd",
                "cooksd": "cooksd",
            }
            resolved_method = method_aliases.get(outlier_method, outlier_method)
            scores = np.zeros(len(members), dtype=float)
            excluded_mask = np.zeros(len(members), dtype=bool)
            leverage = np.full(len(members), float("nan"))
            if outlier_enabled and outlier_threshold > 0 and len(members) >= 2:
                if resolved_method == "mad":
                    residual = np.abs(stack - np.nanmedian(stack, axis=0))
                    metrics = np.nanmedian(residual, axis=1)
                    median_metric = np.nanmedian(metrics)
                    mad_metric = np.nanmedian(np.abs(metrics - median_metric))

                    if np.isfinite(mad_metric) and mad_metric > 0:
                        scores = 0.6745 * np.abs(metrics - median_metric) / mad_metric
                        excluded_mask = scores > outlier_threshold
                elif resolved_method == "cooksd":
                    scores, leverage = _compute_cooksd_scores(ref_wl, stack)
                    excluded_mask = scores > outlier_threshold
                else:
                    raise ValueError(f"Unsupported outlier method: {outlier_method}")

            if excluded_mask.all():
                excluded_mask[:] = False

            included_mask = ~excluded_mask
            averaged_intensity = np.nanmean(stack[included_mask], axis=0)

            meta = dict(members[0].meta)
            meta["replicate_count"] = int(included_mask.sum())
            meta["replicate_total"] = len(members)
            meta["replicate_averaged"] = True
            meta["replicate_sources"] = sources
            meta["replicate_excluded"] = [
                src for src, flag in zip(sources, excluded_mask) if flag and src is not None
            ]
            if len(members) > 1:
                meta["replicate_outlier_enabled"] = bool(
                    outlier_enabled and outlier_threshold > 0
                )
                meta["replicate_outlier_method"] = resolved_method
                meta["replicate_outlier_threshold"] = outlier_threshold
                meta["replicate_outlier_scores"] = [
                    float(score) if np.isfinite(score) else float("nan")
                    for score in scores
                ]
                if np.any(np.isfinite(leverage)):
                    meta["replicate_outlier_leverage"] = [
                        float(val) if np.isfinite(val) else float("nan")
                        for val in leverage
                    ]
            meta["replicates"] = [
                {
                    "wavelength": np.asarray(member.wavelength, dtype=float).tolist(),
                    "intensity": stack[idx].tolist(),
                    "meta": dict(member.meta),
                    "excluded": bool(excluded_mask[idx]),
                    "score": float(scores[idx]) if np.isfinite(scores[idx]) else float("nan"),
                    "leverage": float(leverage[idx]) if np.isfinite(leverage[idx]) else None,
                }
                for idx, member in enumerate(members)
            ]
            averaged = Spectrum(wavelength=ref_wl.copy(), intensity=averaged_intensity, meta=meta)

        mapping[key] = averaged
        outputs.append(averaged)

    if return_mapping:
        return outputs, mapping
    return outputs


def blank_identifier(spec: Spectrum, strategy: str | None = None) -> str | None:
    role = str(spec.meta.get("role", "sample"))
    if role != "blank":
        return None
    return blank_match_identifier(spec, strategy)

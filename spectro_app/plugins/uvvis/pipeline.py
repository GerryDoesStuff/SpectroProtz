"""UV-Vis preprocessing helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

from spectro_app.engine.plugin_api import Spectrum

__all__ = [
    "coerce_domain",
    "subtract_blank",
    "apply_baseline",
    "detect_joins",
    "correct_joins",
    "normalise_exclusion_windows",
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


def _average_replicate_channels(
    included_members: Sequence[Spectrum], averaged_intensity: np.ndarray
) -> Dict[str, Any]:
    """Combine per-stage channels for averaged replicates.

    Parameters
    ----------
    included_members:
        Replicate spectra that contributed to the averaged result.
    averaged_intensity:
        Intensity trace representing the averaged spectrum.
    """

    averaged_channels: Dict[str, Any] = {}

    # Gather all channel names present on the contributing replicates.
    channel_names = set()
    for member in included_members:
        meta = member.meta if isinstance(member.meta, Mapping) else {}
        channels = meta.get("channels") if isinstance(meta, Mapping) else None
        if isinstance(channels, Mapping):
            channel_names.update(channels.keys())

    for name in sorted(channel_names):
        collected: List[np.ndarray] = []
        convertible = True
        shape = None

        for member in included_members:
            meta = member.meta if isinstance(member.meta, Mapping) else {}
            channels = meta.get("channels") if isinstance(meta, Mapping) else None
            if not isinstance(channels, Mapping) or name not in channels:
                continue

            value = channels[name]
            try:
                arr = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                convertible = False
                break

            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                convertible = False
                break

            collected.append(arr)

        if convertible and collected:
            stacked = np.stack(collected, axis=0)
            averaged = np.nanmean(stacked, axis=0)
            averaged_channels[name] = np.asarray(averaged, dtype=float).copy()
            continue

        fallback_value = None
        for member in included_members:
            meta = member.meta if isinstance(member.meta, Mapping) else {}
            channels = meta.get("channels") if isinstance(meta, Mapping) else None
            if isinstance(channels, Mapping) and name in channels:
                fallback_value = channels[name]
                break

        if fallback_value is not None:
            if isinstance(fallback_value, np.ndarray):
                averaged_channels[name] = fallback_value.copy()
            else:
                try:
                    averaged_channels[name] = fallback_value.copy()  # type: ignore[attr-defined]
                except AttributeError:
                    averaged_channels[name] = fallback_value

    averaged_channels["raw"] = np.asarray(averaged_intensity, dtype=float).copy()
    return averaged_channels


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


def normalise_join_windows(windows_cfg: object) -> List[Dict[str, float | int | None]]:
    """Normalise a join window configuration into explicit numeric bounds."""

    if windows_cfg is None:
        return []

    if isinstance(windows_cfg, Mapping):
        windows_iterable = windows_cfg.get("windows")
        if windows_iterable is None:
            return []
    else:
        windows_iterable = windows_cfg

    if not isinstance(windows_iterable, Sequence) or isinstance(windows_iterable, (str, bytes)):
        return []

    normalised: List[Dict[str, float | int | None]] = []
    for entry in windows_iterable:
        if entry is None:
            continue

        lower_val: float | None
        upper_val: float | None
        spikes_val: int

        if isinstance(entry, Mapping):
            lo = entry.get("min_nm")
            if lo is None:
                lo = entry.get("lower_nm")
            if lo is None:
                lo = entry.get("min")
            if lo is None:
                lo = entry.get("lower")
            hi = entry.get("max_nm")
            if hi is None:
                hi = entry.get("upper_nm")
            if hi is None:
                hi = entry.get("max")
            if hi is None:
                hi = entry.get("upper")

            spikes_raw = entry.get("spikes")
        elif isinstance(entry, Sequence):
            data = list(entry)
            if len(data) < 2:
                continue
            lo = data[0]
            hi = data[1]
            spikes_raw = data[3] if len(data) > 3 else None
        else:
            continue

        lower_val = float(lo) if lo is not None else None
        upper_val = float(hi) if hi is not None else None
        if lower_val is not None and upper_val is not None and lower_val > upper_val:
            lower_val, upper_val = upper_val, lower_val

        spikes_val = 1
        if spikes_raw is not None:
            try:
                spikes_val = int(spikes_raw)
            except (TypeError, ValueError):
                spikes_val = 1
        if spikes_val < 0:
            spikes_val = 0

        normalised.append(
            {
                "lower_nm": lower_val,
                "upper_nm": upper_val,
                "spikes": spikes_val,
            }
        )

    return normalised


def normalise_exclusion_windows(exclusions_cfg: object) -> List[Dict[str, float | None]]:
    """Normalise spike exclusion window configuration into numeric bounds."""

    if exclusions_cfg is None:
        return []

    windows_iterable: object
    if isinstance(exclusions_cfg, Mapping):
        if any(
            key in exclusions_cfg
            for key in ("min", "max", "min_nm", "max_nm", "lower", "upper", "lower_nm", "upper_nm")
        ):
            windows_iterable = [exclusions_cfg]
        else:
            windows_iterable = exclusions_cfg.get("windows")
            if windows_iterable is None:
                return []
    else:
        windows_iterable = exclusions_cfg

    if not isinstance(windows_iterable, Sequence) or isinstance(windows_iterable, (str, bytes)):
        return []

    normalised: List[Dict[str, float | None]] = []
    for entry in windows_iterable:
        if entry is None:
            continue

        lower_val: float | None
        upper_val: float | None

        if isinstance(entry, Mapping):
            lo = (
                entry.get("min_nm")
                if entry.get("min_nm") is not None
                else entry.get("lower_nm")
            )
            if lo is None:
                lo = entry.get("min")
            if lo is None:
                lo = entry.get("lower")
            hi = (
                entry.get("max_nm")
                if entry.get("max_nm") is not None
                else entry.get("upper_nm")
            )
            if hi is None:
                hi = entry.get("max")
            if hi is None:
                hi = entry.get("upper")
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            data = list(entry)
            lo = data[0] if data else None
            hi = data[1] if len(data) > 1 else None
        else:
            continue

        try:
            lower_val = float(lo) if lo is not None else None
        except (TypeError, ValueError):
            lower_val = None
        try:
            upper_val = float(hi) if hi is not None else None
        except (TypeError, ValueError):
            upper_val = None

        if lower_val is not None and upper_val is not None and lower_val > upper_val:
            lower_val, upper_val = upper_val, lower_val

        if lower_val is None and upper_val is None:
            continue

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

    explicit_threshold = float(threshold) if threshold is not None else None

    valid_positions = np.where(np.isfinite(scores))[0]
    window_masks: list[np.ndarray] = []
    window_spike_limits: list[int | None] = []
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
            spikes_val = entry.get("spikes")
            spikes_int: int | None
            if spikes_val is None:
                spikes_int = 1
            else:
                try:
                    spikes_int = int(spikes_val)
                except (TypeError, ValueError):
                    spikes_int = 1
            if spikes_int is not None and spikes_int < 0:
                spikes_int = 0
            lo = entry.get("lower_nm")
            hi = entry.get("upper_nm")
            if lo is None and hi is None:
                window_condition = np.isfinite(wl)
                mask |= window_condition
                window_masks.append(window_condition)
                window_spike_limits.append(spikes_int)
                continue
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
            window_masks.append(window_condition)
            window_spike_limits.append(spikes_int)
        mask &= np.isfinite(wl)
        if not np.any(mask):
            return []
        valid_positions = valid_positions[mask[valid_positions]]
    if valid_positions.size == 0:
        return []

    if explicit_threshold is not None:
        valid_scores = scores[valid_positions]
        if valid_scores.size == 0 or not np.any(valid_scores >= explicit_threshold):
            return []

    def _auto_threshold(span_positions: np.ndarray, span_scores: np.ndarray) -> float | None:
        if span_scores.size == 0:
            return None
        try:
            peak_index = int(np.nanargmax(span_scores))
        except ValueError:
            return None
        if peak_index < 0 or peak_index >= span_positions.size:
            return None
        peak_pos = int(span_positions[peak_index])
        median = float(np.nanmedian(span_scores))
        mad = float(np.nanmedian(np.abs(span_scores - median)))
        peak = float(np.nanmax(span_scores))
        if not np.isfinite(peak) or peak <= 0:
            return None
        exclusion_radius = max(raw_window, half_window)
        original_positive_scores = np.sort(
            span_scores[np.isfinite(span_scores) & (span_scores > 0)]
        )

        if not np.isfinite(mad) or mad == 0:
            positive_mask = np.isfinite(span_scores) & (span_scores > 0)
            proximity_mask = np.abs(span_positions - peak_pos) < exclusion_radius
            positive_mask &= ~proximity_mask
            positive_scores = np.sort(span_scores[positive_mask])
            if positive_scores.size == 0:
                threshold_value = peak * 0.5
            else:
                unique_scores = np.unique(positive_scores)
                if unique_scores.size == 1:
                    threshold_value = unique_scores[0]
                else:
                    candidate = unique_scores[-2]
                    if candidate <= 0:
                        candidate = unique_scores[-1] * 0.5
                    threshold_value = candidate
            if not np.isfinite(threshold_value) or threshold_value <= 0:
                threshold_value = peak
        else:
            scale = 1.4826 * mad
            computed = median + 6.0 * scale
            if not np.isfinite(computed) or computed <= 0:
                threshold_value = peak
            elif peak <= computed:
                original_below_peak = original_positive_scores[
                    original_positive_scores < peak
                ]
                if original_below_peak.size > 0:
                    original_max_below = float(np.nanmax(original_below_peak))
                    if (
                        np.isfinite(original_max_below)
                        and original_max_below > 0
                        and peak > 0
                        and original_max_below / peak > 0.999
                    ):
                        return None
                positive_mask = np.isfinite(span_scores) & (span_scores > 0)
                proximity_mask = np.abs(span_positions - peak_pos) < exclusion_radius
                positive_mask &= ~proximity_mask
                positive_scores = np.sort(span_scores[positive_mask])
                if positive_scores.size == 0:
                    threshold_value = peak * 0.5
                else:
                    below_peak = positive_scores[positive_scores < peak]
                    if below_peak.size == 0:
                        threshold_value = peak * 0.99
                    else:
                        max_below = float(np.nanmax(below_peak))
                        if not np.isfinite(max_below) or max_below <= 0:
                            max_below = peak * 0.5
                        separation_ratio = max_below / peak if peak else 1.0
                        if separation_ratio > 0.999:
                            return None
                        percentile = float(np.nanpercentile(below_peak, 97.5))
                        if not np.isfinite(percentile) or percentile <= 0:
                            percentile = max_below
                        threshold_candidate = min(percentile, max_below)
                        if not np.isfinite(threshold_candidate) or threshold_candidate <= 0:
                            threshold_candidate = max_below
                        threshold_value = min(threshold_candidate, peak * 0.99)
                        if not np.isfinite(threshold_value) or threshold_value <= 0:
                            threshold_value = max_below if max_below > 0 else peak * 0.99
            else:
                threshold_value = computed
        if not np.isfinite(threshold_value) or threshold_value <= 0:
            return None
        return float(threshold_value)

    candidate_info: Dict[int, Dict[str, Any]] = {}

    def _record_candidate(idx: int, span_key: tuple[int, int], local_threshold: float) -> None:
        if idx < 0 or idx >= n:
            return
        info = candidate_info.get(idx)
        score = float(scores[idx]) if 0 <= idx < scores.size else float("nan")
        if info is None:
            info = {
                "score": score,
                "threshold": float(local_threshold),
                "spans": {span_key},
            }
            candidate_info[idx] = info
            return
        if np.isfinite(score):
            prev_score = info.get("score")
            if prev_score is None or not np.isfinite(prev_score) or score > prev_score:
                info["score"] = score
        info_threshold = info.get("threshold")
        if info_threshold is None or not np.isfinite(info_threshold) or local_threshold < info_threshold:
            info["threshold"] = float(local_threshold)
        spans: set[tuple[int, int]] = info.setdefault("spans", set())
        spans.add(span_key)

    def _binary_segment_span(
        span_positions: np.ndarray,
        start: int,
        stop: int,
        local_threshold: float,
        span_key: tuple[int, int],
    ) -> None:
        if span_positions.size == 0:
            return

        def _binary_segment(start_idx: int, stop_idx: int) -> None:
            segment_mask = (span_positions >= start_idx) & (span_positions < stop_idx)
            segment_positions = span_positions[segment_mask]
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
            center = (start_idx + stop_idx) / 2.0
            representative_delta = float(deltas[int(maxima_positions[0])])
            if np.isfinite(representative_delta) and representative_delta != 0:
                median_idx = int(np.median(maxima_positions))
                best_idx = median_idx
            else:
                best_idx = int(
                    maxima_positions[int(np.argmin(np.abs(maxima_positions - center)))]
                )
            eligible_positions = [
                int(pos)
                for pos in maxima_positions
                if pos - start_idx >= half_window and stop_idx - pos >= half_window
            ]
            if (
                best_idx - start_idx < half_window or stop_idx - best_idx < half_window
            ) and eligible_positions:
                best_idx = int(np.median(eligible_positions))
            if not np.isfinite(best_score) or best_score < local_threshold:
                return
            if best_idx - start_idx < half_window or stop_idx - best_idx < half_window:
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
                left_idx - 1 >= start_idx
                and left_idx - 1 >= best_idx - (half_window - 1)
                and scores[left_idx - 1] >= local_threshold
                and (
                    sign == 0
                    or not np.isfinite(deltas[left_idx - 1])
                    or np.sign(deltas[left_idx - 1]) == sign
                )
            ):
                left_idx -= 1
            right_idx = best_idx
            max_right = best_idx + (half_window - 1)
            while (
                right_idx + 1 < stop_idx
                and right_idx + 1 <= max_right
                and scores[right_idx + 1] >= local_threshold
                and (
                    sign == 0
                    or not np.isfinite(deltas[right_idx + 1])
                    or np.sign(deltas[right_idx + 1]) == sign
                )
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
            _record_candidate(chosen, span_key, local_threshold)
            _binary_segment(start_idx, chosen)
            _binary_segment(chosen + 1, stop_idx)

        _binary_segment(start, stop)

    if windows is None:
        span_positions = valid_positions
        if span_positions.size == 0:
            return []
        span_scores = scores[span_positions]
        if explicit_threshold is None:
            threshold_value = _auto_threshold(span_positions, span_scores)
            if threshold_value is None:
                return []
        else:
            threshold_value = explicit_threshold
        span_start = half_window
        span_stop = n - half_window + 1
        if span_start < span_stop:
            _binary_segment_span(
                span_positions,
                span_start,
                span_stop,
                threshold_value,
                (0, 0),
            )
    else:
        wl_valid_positions = valid_positions
        for window_index, window_mask in enumerate(window_masks):
            if window_mask.size != n:
                continue
            window_positions = wl_valid_positions[window_mask[wl_valid_positions]]
            if window_positions.size == 0:
                continue
            splits = np.where(np.diff(window_positions) > 1)[0]
            segment_starts = np.concatenate(([0], splits + 1))
            segment_stops = np.concatenate((splits + 1, [window_positions.size]))
            for segment_index, (seg_start, seg_stop) in enumerate(zip(segment_starts, segment_stops)):
                span_positions = window_positions[seg_start:seg_stop]
                if span_positions.size == 0:
                    continue
                span_scores = scores[span_positions]
                if explicit_threshold is None:
                    threshold_value = _auto_threshold(span_positions, span_scores)
                    if threshold_value is None:
                        continue
                else:
                    threshold_value = explicit_threshold
                span_start = half_window
                span_stop = n - half_window + 1
                if span_start >= span_stop:
                    continue
                _binary_segment_span(
                    span_positions,
                    span_start,
                    span_stop,
                    threshold_value,
                    (window_index, segment_index),
                )

    if not candidate_info:
        return []

    candidate_indices = sorted(
        idx for idx in candidate_info.keys() if half_window <= idx < n
    )
    candidate_signs: Dict[int, int] = {}
    for idx in candidate_indices:
        delta_value = deltas[idx] if 0 <= idx < deltas.size else float("nan")
        sign = 0
        if np.isfinite(delta_value):
            if delta_value > 0:
                sign = 1
            elif delta_value < 0:
                sign = -1
        candidate_signs[idx] = sign
    filtered: list[int] = []
    min_spacing = max(1, raw_window)
    for idx in candidate_indices:
        if filtered and idx - filtered[-1] < min_spacing:
            prev = filtered[-1]
            prev_sign = candidate_signs.get(prev, 0)
            curr_sign = candidate_signs.get(idx, 0)
            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                filtered.append(idx)
                continue
            current_spans = candidate_info[idx]["spans"]
            previous_spans = candidate_info[prev]["spans"]
            if current_spans.isdisjoint(previous_spans):
                filtered.append(idx)
                continue
            if scores[idx] > scores[prev]:
                filtered[-1] = idx
            continue
        filtered.append(idx)

    if not filtered:
        return []

    diffs = np.abs(np.diff(y))
    refined: list[int] = []
    refined_source_scores: Dict[int, float] = {}
    refined_thresholds: Dict[int, float] = {}
    refined_span_map: Dict[int, set[tuple[int, int]]] = {}
    refined_signs: Dict[int, set[int]] = {}

    def _update_refined_score(index: int, score: float) -> None:
        existing = refined_source_scores.get(index)
        if existing is None:
            refined_source_scores[index] = score
            return
        if np.isfinite(existing):
            if np.isfinite(score) and score > existing:
                refined_source_scores[index] = score
            return
        refined_source_scores[index] = score

    for idx in filtered:
        info = candidate_info[idx]
        local_threshold = float(info.get("threshold", float("nan")))
        span_ids = set(info.get("spans", set()))
        candidate_score = float(info.get("score", float("nan")))
        lo = max(1, idx - half_window)
        hi = min(n - 1, idx + half_window)
        window_diffs = diffs[lo - 1 : hi]
        sign_value = candidate_signs.get(idx, 0)
        if window_diffs.size == 0:
            refined.append(idx)
            if 0 <= idx < scores.size:
                _update_refined_score(idx, candidate_score)
                refined_thresholds[idx] = (
                    local_threshold
                    if np.isfinite(local_threshold)
                    else refined_thresholds.get(idx, local_threshold)
                )
                existing_spans = refined_span_map.setdefault(idx, set())
                existing_spans.update(span_ids)
                refined_signs.setdefault(idx, set()).add(sign_value)
            continue
        rel = int(np.nanargmax(window_diffs))
        refined_idx = lo - 1 + rel + 1
        delta_value = deltas[idx] if 0 <= idx < deltas.size else float("nan")
        if np.isfinite(delta_value):
            if delta_value > 0 and refined_idx < idx:
                refined_idx = idx
            elif delta_value < 0 and refined_idx > idx:
                refined_idx = idx
            if delta_value > 0:
                sign_value = 1
            elif delta_value < 0:
                sign_value = -1
        existing_signs = refined_signs.get(refined_idx)
        if (
            existing_signs
            and sign_value != 0
            and ((sign_value > 0 and -1 in existing_signs) or (sign_value < 0 and 1 in existing_signs))
        ):
            refined_idx = idx
        refined.append(refined_idx)
        if 0 <= idx < scores.size:
            _update_refined_score(refined_idx, candidate_score)
            prev_threshold = refined_thresholds.get(refined_idx)
            if prev_threshold is None or not np.isfinite(prev_threshold) or (
                np.isfinite(local_threshold) and local_threshold < prev_threshold
            ):
                refined_thresholds[refined_idx] = local_threshold
            existing_spans = refined_span_map.setdefault(refined_idx, set())
            existing_spans.update(span_ids)
            refined_signs.setdefault(refined_idx, set()).add(sign_value)

    refined_sorted: list[int] = []
    final_refined_scores: Dict[int, float] = {}
    final_refined_thresholds: Dict[int, float] = {}
    final_refined_spans: Dict[int, set[tuple[int, int]]] = {}

    def _combine_scores(first: float | None, second: float | None) -> float | None:
        if first is None or not np.isfinite(first):
            return second
        if second is None or not np.isfinite(second):
            return first
        return first if first >= second else second

    def _has_opposite_nonzero(first: set[int] | None, second: set[int] | None) -> bool:
        if not first or not second:
            return False
        return (1 in first and -1 in second) or (-1 in first and 1 in second)

    for idx in sorted(set(refined)):
        if 0 < idx < n:
            if refined_sorted and idx - refined_sorted[-1] < min_spacing:
                prev_idx = refined_sorted[-1]
                prev_signs = refined_signs.get(prev_idx, set())
                curr_signs = refined_signs.get(idx, set())
                if _has_opposite_nonzero(prev_signs, curr_signs):
                    refined_sorted.append(idx)
                    final_refined_scores[idx] = refined_source_scores.get(idx)
                    final_refined_thresholds[idx] = refined_thresholds.get(idx)
                    final_refined_spans[idx] = set(refined_span_map.get(idx, set()))
                    continue
                prev_spans = final_refined_spans.get(prev_idx, set())
                curr_spans = refined_span_map.get(idx, set())
                if prev_spans and curr_spans and prev_spans.isdisjoint(curr_spans):
                    refined_sorted.append(idx)
                    final_refined_scores[idx] = refined_source_scores.get(idx)
                    final_refined_thresholds[idx] = refined_thresholds.get(idx)
                    final_refined_spans[idx] = set(curr_spans)
                    continue
                prev_diff = diffs[prev_idx - 1] if 0 < prev_idx <= diffs.size else float("nan")
                curr_diff = diffs[idx - 1] if 0 < idx <= diffs.size else float("nan")
                prev_diff_val = np.nan_to_num(prev_diff, nan=float("-inf"))
                curr_diff_val = np.nan_to_num(curr_diff, nan=float("-inf"))
                if curr_diff_val > prev_diff_val:
                    previous_score = final_refined_scores.pop(prev_idx, refined_source_scores.get(prev_idx))
                    refined_sorted[-1] = idx
                    combined_signs = set(prev_signs)
                    combined_signs.update(curr_signs)
                    refined_signs[idx] = combined_signs
                    final_refined_scores[idx] = _combine_scores(previous_score, refined_source_scores.get(idx))
                    previous_threshold = final_refined_thresholds.pop(prev_idx, refined_thresholds.get(prev_idx))
                    curr_threshold = refined_thresholds.get(idx)
                    thresholds = [t for t in (previous_threshold, curr_threshold) if t is not None]
                    if thresholds:
                        final_refined_thresholds[idx] = min(
                            float(t) for t in thresholds if np.isfinite(float(t))
                        )
                    else:
                        final_refined_thresholds[idx] = curr_threshold
                    combined_spans = refined_span_map.get(idx, set()).union(
                        final_refined_spans.get(prev_idx, set())
                    )
                    final_refined_spans[idx] = combined_spans
                else:
                    previous_score = final_refined_scores.get(prev_idx, refined_source_scores.get(prev_idx))
                    final_refined_scores[prev_idx] = _combine_scores(previous_score, refined_source_scores.get(idx))
                    prev_threshold = final_refined_thresholds.get(prev_idx)
                    candidate_threshold = refined_thresholds.get(idx)
                    if candidate_threshold is not None and np.isfinite(candidate_threshold):
                        if prev_threshold is None or not np.isfinite(prev_threshold) or candidate_threshold < prev_threshold:
                            final_refined_thresholds[prev_idx] = candidate_threshold
                    final_refined_spans.setdefault(prev_idx, set()).update(
                        refined_span_map.get(idx, set())
                    )
                    refined_signs.setdefault(prev_idx, set()).update(curr_signs)
                continue
            refined_sorted.append(idx)
            final_refined_scores[idx] = refined_source_scores.get(idx)
            final_refined_thresholds[idx] = refined_thresholds.get(idx)
            final_refined_spans[idx] = set(refined_span_map.get(idx, set()))

    if refined_sorted:
        filtered_candidates: list[int] = []
        for idx in refined_sorted:
            if idx >= scores.size:
                continue
            stored_score = final_refined_scores.get(idx)
            score = float(scores[idx])
            if stored_score is not None and np.isfinite(stored_score):
                score = float(stored_score)
            local_threshold = final_refined_thresholds.get(idx)
            if local_threshold is None or not np.isfinite(local_threshold):
                local_threshold = explicit_threshold
            if local_threshold is None or not np.isfinite(local_threshold):
                local_threshold = 0.0
            if not np.isfinite(score) or score >= local_threshold:
                filtered_candidates.append(idx)
        refined_sorted = filtered_candidates

    if refined_sorted and windows is not None and window_spike_limits:
        keep: set[int] = set(refined_sorted)
        per_window: Dict[int, list[tuple[float, int]]] = {}
        for idx in refined_sorted:
            spans = final_refined_spans.get(idx, set())
            if not spans:
                continue
            stored_score = final_refined_scores.get(idx)
            if stored_score is None or not np.isfinite(stored_score):
                raw_score = float(scores[idx]) if 0 <= idx < scores.size else float("nan")
                score_value = raw_score
            else:
                score_value = float(stored_score)
            score_for_sort = score_value if np.isfinite(score_value) else float("-inf")
            for window_idx, _segment_idx in spans:
                if window_idx < 0 or window_idx >= len(window_spike_limits):
                    continue
                per_window.setdefault(window_idx, []).append((score_for_sort, idx))

        for window_idx, candidates in per_window.items():
            limit_value = window_spike_limits[window_idx]
            if limit_value is None:
                continue
            limit_int = int(limit_value)
            if limit_int < 0:
                limit_int = 0
            if limit_int == 0:
                for _, idx in candidates:
                    keep.discard(idx)
                continue
            ordered = sorted(
                candidates,
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            allowed = {idx for _, idx in ordered[:limit_int]}
            for _, idx in candidates:
                if idx not in allowed:
                    keep.discard(idx)

        refined_sorted = [idx for idx in refined_sorted if idx in keep]

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
    baseline_window: int | None = None,
    spread_window: int | None = None,
    spread_method: str = "mad",
    spread_epsilon: float = 1e-6,
    residual_floor: float = 2e-2,
    isolation_ratio: float = 20.0,
    tail_decay_ratio: float = 0.85,
    max_passes: int = 10,
    join_indices: Sequence[int] | None = None,
    leading_padding: int = 0,
    trailing_padding: int = 0,
    noise_scale_multiplier: float = 1.0,
    rng: np.random.Generator | None = None,
    rng_seed: int | np.random.SeedSequence | None = None,
    exclusion_windows: Sequence[Mapping[str, float | None]] | Sequence[Sequence[float | None]] | None = None,
) -> Spectrum:
    """Replace impulsive spikes using adaptive rolling baselines.

    The entire intensity trace is processed in a single pass and the
    previously-used ``join_indices`` argument is retained only for backwards
    compatibility. The intensity trace is iteratively compared against a
    rolling baseline and local spread estimate; samples exceeding
    ``baseline ± zscore * spread`` are replaced with the contemporaneous
    baseline and detection repeats until no spikes remain or ``max_passes`` is
    reached. Degenerate windows that collapse the spread fall back to global
    residual statistics so isolated spikes are still removed while gentle
    curvature and real peaks are preserved. Replaced samples receive a small
    random perturbation (zero-mean Gaussian) with a scale derived from the same
    spread estimate used for detection so the corrected points blend into the
    surrounding noise floor. ``noise_scale_multiplier`` can expand or shrink
    that perturbation and callers may provide ``rng`` or ``rng_seed`` to control
    reproducibility (both default to fresh ``default_rng`` draws when omitted).
    Optional ``exclusion_windows`` (expressed as wavelength bounds) fence off
    spectral regions that should be ignored by spike detection and replacement.
    ``tail_decay_ratio`` tunes how aggressively spike tails must decay before
    they are absorbed into the correction cluster; raise it toward ``1`` to
    chase broader shoulders or lower it to demand steeper fall-offs.
    """

    if noise_scale_multiplier < 0:
        raise ValueError("noise_scale_multiplier must be non-negative")

    if rng is not None and rng_seed is not None:
        raise ValueError("Provide either rng or rng_seed, not both")

    if not (0.0 < float(tail_decay_ratio) <= 1.0):
        raise ValueError("tail_decay_ratio must be in the interval (0, 1]")
    tail_decay_ratio = float(tail_decay_ratio)

    if rng is None:
        noise_rng = np.random.default_rng(rng_seed)
    else:
        noise_rng = rng

    x = np.asarray(spec.wavelength, dtype=float)
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

    def _normalise_window(length: int, requested: int | None, fallback: int) -> int | None:
        if length < 3:
            return None
        base = fallback if requested is None else int(requested)
        if base <= 1:
            base = 3
        eff = min(int(base), length)
        if eff < 3:
            return None
        if eff % 2 == 0:
            eff = max(3, eff - 1)
            if eff > length:
                eff = length if length % 2 == 1 else max(3, length - 1)
        if eff < 3:
            return None
        return eff

    base_window = int(window)
    if base_window <= 1:
        base_window = 3

    n = y.size

    corrected = y.copy()
    overall_mask = np.zeros(n, dtype=bool)

    exclusion_mask = np.zeros(n, dtype=bool)
    if exclusion_windows:
        finite_wavelength = np.isfinite(x)
        for entry in exclusion_windows:
            lower_val: float | None = None
            upper_val: float | None = None
            if isinstance(entry, Mapping):
                lo = entry.get("lower_nm")
                if lo is None:
                    lo = entry.get("min_nm")
                hi = entry.get("upper_nm")
                if hi is None:
                    hi = entry.get("max_nm")
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                data = list(entry)
                lo = data[0] if data else None
                hi = data[1] if len(data) > 1 else None
            else:
                continue
            try:
                lower_val = float(lo) if lo is not None else None
            except (TypeError, ValueError):
                lower_val = None
            try:
                upper_val = float(hi) if hi is not None else None
            except (TypeError, ValueError):
                upper_val = None
            if lower_val is not None and upper_val is not None and lower_val > upper_val:
                lower_val, upper_val = upper_val, lower_val
            mask = finite_wavelength.copy()
            if lower_val is not None:
                mask &= x >= lower_val
            if upper_val is not None:
                mask &= x <= upper_val
            exclusion_mask |= mask

    def _rolling_baseline(values: np.ndarray, size: int) -> np.ndarray:
        if size <= 1:
            return values
        return median_filter(values, size=size, mode="reflect")

    def _rolling_spread(residual: np.ndarray, size: int, *, method: str) -> np.ndarray:
        if size <= 1:
            spread = np.abs(residual)
        else:
            method_lower = method.lower()
            if method_lower == "mad":
                spread = median_filter(np.abs(residual), size=size, mode="reflect")
                spread = spread * 1.4826
            elif method_lower in {"std", "variance", "var"}:
                mean = uniform_filter1d(residual, size=size, mode="reflect")
                mean_sq = uniform_filter1d(residual * residual, size=size, mode="reflect")
                variance = np.maximum(mean_sq - mean * mean, 0.0)
                if method_lower == "std":
                    spread = np.sqrt(variance)
                else:
                    spread = np.sqrt(variance)
            else:
                raise ValueError(f"Unsupported spread_method '{method}'")
        return spread

    def _global_spread(residual: np.ndarray, *, method: str) -> float:
        method_lower = method.lower()
        finite_residual = residual[np.isfinite(residual)]
        if finite_residual.size == 0:
            return float("nan")
        if method_lower == "mad":
            med = np.nanmedian(np.abs(finite_residual))
            return float(med * 1.4826)
        return float(np.nanstd(finite_residual))

    pad_leading = max(0, int(leading_padding))
    pad_trailing = max(0, int(trailing_padding))

    leading = min(pad_leading, n)
    trailing = min(pad_trailing, max(0, n - leading))
    work_start = leading
    work_stop = n - trailing
    work_len = work_stop - work_start

    if work_len < 3:
        if leading:
            corrected[:leading] = y[:leading]
        if trailing:
            corrected[n - trailing :] = y[n - trailing :]
        return _final_spectrum(corrected, mask=overall_mask)

    seg_baseline_window = _normalise_window(work_len, baseline_window, base_window)
    if seg_baseline_window is None:
        if leading:
            corrected[:leading] = y[:leading]
        if trailing:
            corrected[n - trailing :] = y[n - trailing :]
        return _final_spectrum(corrected, mask=overall_mask)

    seg_spread_window = _normalise_window(
        work_len,
        spread_window,
        seg_baseline_window if baseline_window is not None else base_window,
    )
    if seg_spread_window is None:
        seg_spread_window = seg_baseline_window

    working = corrected[work_start:work_stop]
    working_mask = overall_mask[work_start:work_stop]
    working_exclusion = exclusion_mask[work_start:work_stop]

    for _ in range(int(max_passes)):
        baseline = _rolling_baseline(working, seg_baseline_window)
        residual = working - baseline
        spread = _rolling_spread(residual, seg_spread_window, method=spread_method)
        spread = np.asarray(spread, dtype=float)

        valid_spread = spread > spread_epsilon
        candidate_mask = np.abs(residual) > float(zscore) * np.maximum(spread, spread_epsilon)
        if working_exclusion.size:
            candidate_mask &= ~working_exclusion

        if np.any(candidate_mask):
            newly_detected = candidate_mask & ~working_mask
            if np.any(newly_detected):
                new_indices = np.flatnonzero(newly_detected)
                split_points = np.where(np.diff(new_indices) > 1)[0] + 1
                clusters = np.split(new_indices, split_points)
                for cluster in clusters:
                    if cluster.size == 0:
                        continue
                    cluster_abs_residual = np.abs(residual[cluster])
                    peak_local = int(cluster[np.argmax(cluster_abs_residual)])
                    peak_residual = residual[peak_local]
                    if peak_residual == 0:
                        continue
                    sign = 1.0 if peak_residual > 0 else -1.0
                    peak_magnitude = float(abs(peak_residual))
                    if peak_magnitude <= spread_epsilon:
                        continue

                    tail_start_floor = max(residual_floor, spread_epsilon)

                    def _extend(start: int, stop: int, step: int) -> list[int]:
                        collected: list[int] = []
                        previous = peak_magnitude
                        for idx in range(start, stop, step):
                            value = residual[idx]
                            if value * sign <= 0:
                                break
                            magnitude = float(abs(value))
                            if magnitude <= spread_epsilon:
                                break
                            if magnitude < tail_start_floor and not collected:
                                break
                            if magnitude > peak_magnitude * tail_decay_ratio:
                                break
                            if magnitude > previous * tail_decay_ratio:
                                break
                            collected.append(idx)
                            previous = magnitude if magnitude > spread_epsilon else spread_epsilon
                        return collected

                    if sign > 0:
                        right_indices = _extend(peak_local + 1, work_len, 1)
                        if right_indices:
                            candidate_mask[right_indices] = True
                    else:
                        left_indices = _extend(peak_local - 1, -1, -1)
                        if left_indices:
                            candidate_mask[left_indices] = True

        if not np.any(valid_spread):
            global_scale = _global_spread(residual, method=spread_method)
            if np.isfinite(global_scale) and global_scale > spread_epsilon:
                candidate_mask |= np.abs(residual) > float(zscore) * global_scale
            else:
                abs_residual = np.abs(residual)
                max_residual = float(abs_residual.max(initial=0.0))
                if max_residual > max(residual_floor, spread_epsilon):
                    if abs_residual.size > 1:
                        second = float(np.partition(abs_residual, -2)[-2])
                    else:
                        second = 0.0
                    if second <= 0 or max_residual >= isolation_ratio * second:
                        candidate_mask = abs_residual >= max_residual - 1e-12
                    else:
                        candidate_mask = np.zeros(work_len, dtype=bool)
                else:
                    candidate_mask = np.zeros(work_len, dtype=bool)

        if working_exclusion.size:
            candidate_mask &= ~working_exclusion

        newly_flagged = candidate_mask & ~working_mask
        if not np.any(newly_flagged):
            break

        working_mask |= candidate_mask

        if np.any(newly_flagged) and noise_scale_multiplier > 0:
            local_scale = np.asarray(spread[newly_flagged], dtype=float)

            def _background_scale() -> float:
                """Resolve a representative noise scale for fallback jitter."""

                candidate_sets: list[np.ndarray] = [residual[~candidate_mask]]
                # Values untouched in previous passes still reflect the ambient
                # noise, so consider them before falling back to the full residual.
                candidate_sets.append(residual[~working_mask])
                candidate_sets.append(residual)

                for sample in candidate_sets:
                    if sample.size == 0:
                        continue
                    scale = _global_spread(sample, method=spread_method)
                    if np.isfinite(scale) and scale > spread_epsilon:
                        return float(scale)

                # Fall back to the spread estimates from the unflagged region if
                # they contain usable information.
                spread_masks = [~candidate_mask, np.ones_like(spread, dtype=bool)]
                for mask in spread_masks:
                    sample = spread[mask & np.isfinite(spread)]
                    if sample.size == 0:
                        continue
                    scale = float(np.nanmedian(sample))
                    if np.isfinite(scale) and scale > spread_epsilon:
                        return scale

                return float(spread_epsilon)

            background_scale = _background_scale()

            if local_scale.size:
                invalid = ~np.isfinite(local_scale) | (local_scale <= spread_epsilon)
                if np.any(invalid):
                    local_scale = local_scale.copy()
                    if np.any(~invalid):
                        # Borrow the typical magnitude from the valid members of
                        # this cluster; fall back to the global background if
                        # every entry collapsed.
                        fill_value = float(np.nanmedian(local_scale[~invalid]))
                        if not np.isfinite(fill_value) or fill_value <= spread_epsilon:
                            fill_value = background_scale
                    else:
                        fill_value = background_scale
                    local_scale[invalid] = fill_value
            else:
                local_scale = np.array([], dtype=float)

            min_scale = background_scale if np.isfinite(background_scale) and background_scale > spread_epsilon else spread_epsilon
            if local_scale.size:
                local_scale = np.maximum(local_scale, min_scale)
                if np.isfinite(background_scale) and background_scale > 0:
                    cap = max(background_scale, spread_epsilon) * 3.0
                    local_scale = np.minimum(local_scale, cap)

            noise = noise_rng.normal(
                loc=0.0,
                scale=np.asarray(local_scale, dtype=float) * float(noise_scale_multiplier),
            )
            working[newly_flagged] = baseline[newly_flagged] + noise
        else:
            working[newly_flagged] = baseline[newly_flagged]

    if leading:
        corrected[:leading] = y[:leading]
        overall_mask[:leading] = False
    if trailing:
        corrected[n - trailing :] = y[n - trailing :]
        overall_mask[n - trailing :] = False

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
            intensity = np.asarray(member.intensity, dtype=float).copy()
            meta["channels"] = _average_replicate_channels([member], intensity)
            averaged = Spectrum(
                wavelength=np.asarray(member.wavelength, dtype=float).copy(),
                intensity=intensity,
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
            included_members = [member for flag, member in zip(included_mask, members) if flag]
            meta["channels"] = _average_replicate_channels(included_members, averaged_intensity)
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

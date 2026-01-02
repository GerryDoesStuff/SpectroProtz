"""Shared peak detection helpers derived from the index builder.

This module provides the detection/refinement logic that used to live in
``scripts/jdxIndexBuilder.py`` so plugins can share a single implementation.
"""

from __future__ import annotations

import logging
import math
import signal
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import (
    find_peaks,
    find_peaks_cwt,
    peak_prominences,
    peak_widths,
    savgol_filter,
)
from scipy.sparse.linalg import spsolve
from scipy.special import wofz

try:
    from scipy.signal import PeakPropertyWarning
except ImportError:  # pragma: no cover - handled by fallback imports
    try:
        from scipy.signal._peak_finding import PeakPropertyWarning
    except ImportError:  # pragma: no cover - fallback definition
        class PeakPropertyWarning(UserWarning):
            pass

logger = logging.getLogger(__name__)

DEFAULT_PEAK_CONFIG: Dict[str, object] = {
    "enabled": True,
    "prominence": 0.01,
    "min_distance": 5.0,
    "max_peaks": 5,
    "height": None,
    "min_absorbance_threshold": 0.05,
    "min_distance_mode": "fixed",
    "min_distance_fwhm_fraction": 0.5,
    "noise_sigma_multiplier": 1.5,
    "noise_window_cm": 0.0,
    "min_prominence_by_region": None,
    "peak_width_min": 0.0,
    "peak_width_max": 0.0,
    "max_peak_candidates": 0,
    "cwt_enabled": False,
    "cwt_width_min": 0.0,
    "cwt_width_max": 0.0,
    "cwt_width_step": 0.0,
    "cwt_cluster_tolerance": 8.0,
    "plateau_min_points": 3,
    "plateau_prominence_factor": 1.0,
    "merge_tolerance": 8.0,
    "close_peak_tolerance_cm": 1.5,
    "shoulder_merge_tolerance_cm": 1.5,
    "shoulder_curvature_prominence_factor": 1.2,
    "sg_win": 7,
    "sg_poly": 3,
    "sg_window_cm": 0.0,
    "als_lam": 0.0,
    "als_p": 0.01,
    "baseline_method": "rubberband",
    "baseline_niter": 20,
    "baseline_piecewise": False,
    "baseline_ranges": None,
    "model": "Gaussian",
    "min_r2": 0.75,
    "fit_window_pts": 70,
    "shoulder_fit_window_pts": None,
    "shoulder_min_r2": 0.65,
    "multi_fit_min_span_cm": 10.0,
    "fit_maxfev": 10000,
    "fit_timeout_sec": 0.0,
    "fit_progress_every": 0,
    "flat_max_amplitude": None,
    "flat_variance_threshold": None,
    "detect_negative_peaks": False,
}


class FitTimeoutError(TimeoutError):
    """Raised when a peak fit exceeds the configured timeout."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"Fit timed out after {seconds:.1f}s")


class _FitTimeoutGuard:
    def __init__(self, seconds: float):
        self.seconds = float(seconds or 0.0)
        self._enabled = bool(self.seconds and self.seconds > 0 and hasattr(signal, "SIGALRM"))
        self._previous_handler = None
        self._previous_timer = None

    def __enter__(self):
        if not self._enabled:
            return self
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        self._previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def _handle_alarm(_signum, _frame):
            raise FitTimeoutError(self.seconds)

        signal.signal(signal.SIGALRM, _handle_alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and hasattr(signal, "SIGALRM"):
            if self._previous_timer is not None:
                signal.setitimer(signal.ITIMER_REAL, *self._previous_timer)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)
        return False


def log_line(msg: str, stream=sys.stdout, flush: bool = False) -> None:
    stream.write(f"{msg}\n")
    if flush:
        stream.flush()


def resolve_peak_config(peak_cfg: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Return merged peak configuration using shared defaults."""

    resolved = dict(DEFAULT_PEAK_CONFIG)
    if peak_cfg:
        resolved.update({k: v for k, v in peak_cfg.items() if v is not None})
    if "distance" in resolved and "min_distance" not in resolved:
        resolved["min_distance"] = resolved["distance"]
    if "num_peaks" in resolved and "max_peaks" not in resolved:
        resolved["max_peaks"] = resolved["num_peaks"]
    if resolved.get("shoulder_fit_window_pts") is None:
        fit_window = int(resolved.get("fit_window_pts", 70) or 70)
        resolved["shoulder_fit_window_pts"] = max(5, int(fit_window * 0.6))
    if resolved.get("flat_max_amplitude") is None:
        min_abs = float(resolved.get("min_absorbance_threshold", 0.05))
        resolved["flat_max_amplitude"] = max(min_abs * 0.5, 0.0)
    if resolved.get("flat_variance_threshold") is None:
        min_abs = float(resolved.get("min_absorbance_threshold", 0.05))
        resolved["flat_variance_threshold"] = (min_abs * 0.1) ** 2
    return resolved


def _nan_safe(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        return values
    idx = np.arange(values.size)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.zeros_like(values)
    filled = values.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], values[mask])
    return filled


def rubberband_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    y_arr = _nan_safe(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.zeros_like(y_arr)
    mask = np.isfinite(x_arr)
    if not np.any(mask):
        return np.zeros_like(y_arr)
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if x_valid.size == 1:
        baseline = np.zeros_like(y_arr)
        baseline[mask] = y_valid
        return baseline
    order = np.argsort(x_valid)
    xs = x_valid[order]
    ys = y_valid[order]
    points = list(zip(xs, ys))

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for pt in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    hull_x = np.array([pt[0] for pt in lower], dtype=float)
    hull_y = np.array([pt[1] for pt in lower], dtype=float)
    baseline_sorted = np.interp(xs, hull_x, hull_y)
    baseline_valid = np.empty_like(xs)
    baseline_valid[order.argsort()] = baseline_sorted
    baseline = np.zeros_like(y_arr)
    baseline[mask] = baseline_valid
    if not np.all(mask):
        idx = np.arange(y_arr.size)
        baseline[~mask] = np.interp(idx[~mask], idx[mask], baseline[mask])
    return baseline


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = _nan_safe(y)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L), format="csc")
    laplacian = (D.T @ D).tocsc()
    w = np.ones(L)
    for _ in range(max(1, int(niter))):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + lam * laplacian
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def airpls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    niter: int = 20,
    tol: float = 1e-6,
    weight_floor: float = 1e-3,
) -> np.ndarray:
    y = _nan_safe(y)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L), format="csc")
    H = lam * (D.T @ D).tocsc()
    w = np.ones(L)
    for idx in range(1, max(1, int(niter)) + 1):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        neg = d[d < 0]
        if neg.size == 0:
            break
        mean = float(np.mean(neg))
        std = float(np.std(neg))
        if std <= 1e-12:
            break
        if abs(mean) / std < tol:
            break
        scaled = np.clip((idx * d) / (abs(mean) + 1e-12), -50.0, 50.0)
        weights = np.exp(scaled)
        w = np.where(d >= 0, weight_floor, weights)
        w[0] = weight_floor
        w[-1] = weight_floor
    return z


def arpls_baseline(y: np.ndarray, lam: float = 1e5, niter: int = 20, tol: float = 1e-6) -> np.ndarray:
    return airpls_baseline(y, lam=lam, niter=niter, tol=tol)


def _parse_piecewise_ranges(ranges: str | None) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    parsed: List[Tuple[float, float]] = []
    for block in ranges.split(","):
        parts = block.replace("–", "-").split("-")
        if len(parts) != 2:
            continue
        try:
            start = float(parts[0])
            stop = float(parts[1])
        except ValueError:
            continue
        parsed.append((start, stop))
    return parsed


def _apply_baseline(
    y: np.ndarray,
    method: str,
    lam: float,
    p: float,
    niter: int,
    x: np.ndarray | None = None,
) -> np.ndarray:
    method = (method or "").lower()
    if method == "asls":
        return als_baseline(y, lam=lam, p=p, niter=niter)
    if method == "arpls":
        return arpls_baseline(y, lam=lam, niter=niter)
    if method == "airpls":
        return airpls_baseline(y, lam=lam, niter=niter)
    if method == "rubberband":
        if x is None:
            x = np.arange(len(y), dtype=float)
        return rubberband_baseline(x, y)
    raise ValueError(f"Unsupported baseline method: {method}")


def _baseline_piecewise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    lam: float,
    p: float,
    niter: int,
    ranges: List[Tuple[float, float]],
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.zeros_like(y)
    covered = np.zeros_like(y, dtype=bool)
    for start, stop in ranges:
        low = min(start, stop)
        high = max(start, stop)
        mask = (x >= low) & (x <= high)
        if not np.any(mask):
            continue
        baseline[mask] = _apply_baseline(y[mask], method, lam, p, niter, x=x[mask])
        covered |= mask
    if not np.all(covered):
        fallback = _apply_baseline(y, method, lam, p, niter, x=x)
        baseline[~covered] = fallback[~covered]
    return baseline


MAD_SCALE = 1.4826


def _estimate_spacing(x: np.ndarray) -> float:
    diffs = np.diff(np.asarray(x, dtype=float))
    if diffs.size == 0:
        return float("nan")
    spacing = float(np.nanmedian(np.abs(diffs)))
    return spacing if np.isfinite(spacing) and spacing > 0 else float("nan")


def _resolve_sg_window_points(
    x: np.ndarray,
    sg_win: int | None,
    sg_window_cm: float | None,
    n: int,
) -> int | None:
    window: int | None = None
    if sg_window_cm is not None and sg_window_cm > 0:
        spacing = _estimate_spacing(x)
        if np.isfinite(spacing) and spacing > 0:
            points = int(np.round(float(sg_window_cm) / spacing)) + 1
            if points < 3:
                points = 3
            if points % 2 == 0:
                points += 1
            window = points
    if window is None and sg_win:
        window = int(sg_win)
        if window % 2 == 0:
            window += 1
    if window is None:
        return None
    if n < 3:
        return None
    max_window = n if n % 2 == 1 else n - 1
    return min(window, max_window) if max_window >= 3 else None


def _coerce_sg_polyorder(value: int) -> int:
    poly = int(value)
    if poly < 2:
        return 2
    if poly > 3:
        return 3
    return poly


def _apply_sg_smoothing(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    sg_window_cm: float | None,
) -> np.ndarray:
    y2 = np.asarray(y, dtype=float).copy()
    n = y2.size
    if n < 3:
        return y2
    window = _resolve_sg_window_points(x, sg_win, sg_window_cm, n)
    if window is None or window < 3:
        return y2
    poly = _coerce_sg_polyorder(sg_poly)
    if window <= poly:
        poly = max(1, window - 1)
    if window <= poly or poly < 1:
        return y2
    return savgol_filter(y2, window, poly)


def estimate_noise_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    sg_window_cm: float | None = None,
) -> float:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size < 3:
        return float("nan")
    smooth = _apply_sg_smoothing(x, y_arr, sg_win, sg_poly, sg_window_cm)
    residual = y_arr - smooth
    median = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - median)))
    if not np.isfinite(mad):
        return float("nan")
    return MAD_SCALE * mad


def _estimate_segment_noise_sigma(y: np.ndarray, fallback: float) -> float:
    values = np.asarray(y, dtype=float)
    if values.size < 3:
        return float(fallback)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    if not np.isfinite(mad) or mad <= 0:
        return float(fallback)
    sigma = MAD_SCALE * mad
    if not np.isfinite(sigma) or sigma <= 0:
        return float(fallback)
    return float(sigma)


def _rolling_mad_sigma(y: np.ndarray, window_pts: int, fallback: float) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    n = values.size
    if n == 0:
        return np.asarray([], dtype=float)
    if window_pts < 3 or window_pts >= n:
        sigma = _estimate_segment_noise_sigma(values, fallback)
        return np.full(n, sigma, dtype=float)
    if window_pts % 2 == 0:
        window_pts += 1
    pad = window_pts // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_pts)
    medians = np.nanmedian(windows, axis=1)
    mads = np.nanmedian(np.abs(windows - medians[:, None]), axis=1)
    sigma = MAD_SCALE * mads
    fallback_value = float(fallback) if np.isfinite(fallback) and fallback > 0 else float("nan")
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_value)
    return sigma.astype(float)


def _parse_min_prominence_by_region(spec: Optional[str]) -> List[Tuple[float, float, float]]:
    if not spec:
        return []
    regions: List[Tuple[float, float, float]] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            logger.warning("Invalid region prominence entry '%s' (missing ':')", chunk)
            continue
        range_part, prom_part = chunk.split(":", 1)
        range_part = range_part.strip()
        prom_part = prom_part.strip()
        try:
            prominence = float(prom_part)
        except ValueError:
            logger.warning("Invalid prominence value '%s' in '%s'", prom_part, chunk)
            continue
        if "-" not in range_part:
            logger.warning("Invalid range '%s' in '%s'", range_part, chunk)
            continue
        lo_str, hi_str = range_part.split("-", 1)
        try:
            low = float(lo_str.strip())
            high = float(hi_str.strip())
        except ValueError:
            logger.warning("Invalid range bounds '%s' in '%s'", range_part, chunk)
            continue
        if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(prominence):
            logger.warning("Invalid region prominence entry '%s'", chunk)
            continue
        low, high = (low, high) if low <= high else (high, low)
        regions.append((low, high, float(prominence)))
    return regions


def _build_prominence_segments(
    x: np.ndarray,
    window_cm: float,
    regions: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, bool]]:
    if x.size == 0:
        return []
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return []
    if x_min == x_max:
        return [(x_min, x_max, True)]
    edges = {x_min, x_max}
    if window_cm and window_cm > 0:
        step = float(window_cm)
        if step > 0:
            current = x_min + step
            while current < x_max:
                edges.add(current)
                current += step
    for low, high, _ in regions:
        edges.add(float(low))
        edges.add(float(high))
    ordered = sorted(edges)
    segments: List[Tuple[float, float, bool]] = []
    for idx in range(len(ordered) - 1):
        low = ordered[idx]
        high = ordered[idx + 1]
        if low == high:
            continue
        is_last = idx == len(ordered) - 2
        segments.append((low, high, is_last))
    return segments


def _region_prominence_floor(
    low: float,
    high: float,
    regions: List[Tuple[float, float, float]],
) -> float:
    floor = 0.0
    for region_low, region_high, region_floor in regions:
        if high < region_low or low > region_high:
            continue
        floor = max(floor, region_floor)
    return float(floor)


def preprocess_with_noise(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    als_lam: float,
    als_p: float,
    *,
    sg_window_cm: float | None = None,
    baseline_method: str = "rubberband",
    baseline_niter: int = 20,
    baseline_piecewise: bool = False,
    baseline_ranges: str | None = None,
) -> tuple[np.ndarray, float]:
    y2 = np.asarray(y, dtype=float).copy()
    n = len(y2)
    noise_sigma = estimate_noise_sigma(x, y2, sg_win, sg_poly, sg_window_cm)
    y2 = _apply_sg_smoothing(x, y2, sg_win, sg_poly, sg_window_cm)
    baseline = np.zeros_like(y2)
    if als_lam > 0:
        if n >= 3:
            ranges = _parse_piecewise_ranges(baseline_ranges)
            if baseline_piecewise and ranges:
                baseline = _baseline_piecewise(
                    x,
                    y2,
                    method=baseline_method,
                    lam=als_lam,
                    p=als_p,
                    niter=baseline_niter,
                    ranges=ranges,
                )
            else:
                baseline = _apply_baseline(
                    y2,
                    baseline_method,
                    als_lam,
                    als_p,
                    baseline_niter,
                    x=x,
                )
            y2 = y2 - baseline
        elif n:
            baseline = np.full_like(y2, float(np.mean(y2)))
            y2 = y2 - baseline
    m = np.max(np.abs(y2))
    if m > 0:
        y2 /= m
        if np.isfinite(noise_sigma):
            noise_sigma /= m
    return y2, noise_sigma


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    als_lam: float,
    als_p: float,
    *,
    sg_window_cm: float | None = None,
    baseline_method: str = "rubberband",
    baseline_niter: int = 20,
    baseline_piecewise: bool = False,
    baseline_ranges: str | None = None,
) -> np.ndarray:
    y2, _ = preprocess_with_noise(
        x,
        y,
        sg_win,
        sg_poly,
        als_lam,
        als_p,
        sg_window_cm=sg_window_cm,
        baseline_method=baseline_method,
        baseline_niter=baseline_niter,
        baseline_piecewise=baseline_piecewise,
        baseline_ranges=baseline_ranges,
    )
    return y2


def gaussian(x, A, x0, sigma, C):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + C


def lorentzian(x, A, x0, gamma, C):
    return A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2) + C


def voigt_profile(x, A, x0, sigma, gamma, C):
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return A * profile + C


def _ensure_odd_window(window: int, max_len: int) -> int:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > max_len:
        window = max_len if max_len % 2 == 1 else max_len - 1
    return max(window, 3)


def _resolve_step_from_x(xs: np.ndarray) -> float:
    step = np.median(np.abs(np.diff(xs))) if xs.size > 1 else 0.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    return float(step)


def _compute_second_derivative(xs: np.ndarray, ys: np.ndarray, distance_pts: int) -> np.ndarray:
    if ys.size < 5:
        return np.zeros_like(ys)
    window = _ensure_odd_window(max(7, distance_pts * 2 + 1), ys.size)
    poly = 3 if window >= 5 else 2
    step = _resolve_step_from_x(xs)
    try:
        return savgol_filter(ys, window_length=window, polyorder=poly, deriv=2, delta=step)
    except Exception:
        first = np.gradient(ys, step)
        return np.gradient(first, step)


def _fit_peak_spline(xs: np.ndarray, ys: np.ndarray, x0_guess: float) -> Optional[Dict[str, float]]:
    if xs.size < 5:
        return None
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    smoothing = max(0.0, 0.001 * np.nanvar(ys_sorted) * xs_sorted.size)
    spline = UnivariateSpline(xs_sorted, ys_sorted, k=3, s=smoothing)
    dense_x = np.linspace(xs_sorted[0], xs_sorted[-1], xs_sorted.size * 8)
    dense_y = spline(dense_x)
    local_mask = np.abs(dense_x - x0_guess) <= (xs_sorted[-1] - xs_sorted[0]) * 0.5
    if not np.any(local_mask):
        local_mask = np.ones_like(dense_x, dtype=bool)
    local_idx = int(np.argmax(dense_y[local_mask]))
    local_x = dense_x[local_mask][local_idx]
    peak_y = float(dense_y[local_mask][local_idx])
    baseline = float(np.median(ys_sorted))
    half_height = baseline + (peak_y - baseline) * 0.5
    left_candidates = np.where(dense_x <= local_x)[0]
    right_candidates = np.where(dense_x >= local_x)[0]
    left_idx = left_candidates[np.where(dense_y[left_candidates] <= half_height)[0]]
    right_idx = right_candidates[np.where(dense_y[right_candidates] <= half_height)[0]]
    if left_idx.size == 0 or right_idx.size == 0:
        return None
    left_x = float(dense_x[left_idx[-1]])
    right_x = float(dense_x[right_idx[0]])
    fwhm = float(abs(right_x - left_x))
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(np.maximum(dense_y - baseline, 0.0), dense_x))
    else:
        area = float(np.trapz(np.maximum(dense_y - baseline, 0.0), dense_x))
    return dict(center=float(local_x), fwhm=fwhm, amplitude=float(peak_y - baseline), area=area, r2=1.0)


def _fit_peak_single(
    xs: np.ndarray,
    ys: np.ndarray,
    x0_guess: float,
    model: str,
    *,
    center_bounds: tuple[float, float] | None = None,
    maxfev: int = 10000,
    fit_timeout_sec: float = 0.0,
) -> Optional[Dict[str, float]]:
    if xs.size < 5:
        return None
    model = model.lower()
    if model == "spline":
        return _fit_peak_spline(xs, ys, x0_guess)
    local_idx = int(np.argmin(np.abs(xs - x0_guess)))
    step = _resolve_step_from_x(xs)
    C = float(np.median(ys))
    A = max(float(ys[local_idx] - C), 1e-6)
    width_guess = max(step * 2.0, np.abs(xs[-1] - xs[0]) / 6.0)
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    x0 = float(x0_guess)
    x0_min = float(np.min(xs_sorted))
    x0_max = float(np.max(xs_sorted))
    if center_bounds is not None:
        bound_min = float(min(center_bounds))
        bound_max = float(max(center_bounds))
        x0_min = max(x0_min, bound_min)
        x0_max = min(x0_max, bound_max)
        if x0_min >= x0_max:
            x0_min = float(np.min(xs_sorted))
            x0_max = float(np.max(xs_sorted))
    if model == "lorentzian":
        bounds = (
            [0.0, x0_min, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf],
        )
        with _FitTimeoutGuard(fit_timeout_sec):
            popt, _ = curve_fit(
                lorentzian,
                xs_sorted,
                ys_sorted,
                p0=[A, x0, width_guess, C],
                maxfev=maxfev,
                bounds=bounds,
            )
        A, x0, gamma, C = popt
        yfit = lorentzian(xs_sorted, *popt)
        fwhm = 2.0 * abs(gamma)
        component = lorentzian(xs_sorted, A, x0, gamma, 0.0)
    elif model == "voigt":
        bounds = (
            [0.0, x0_min, step * 0.25, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf, np.inf],
        )
        with _FitTimeoutGuard(fit_timeout_sec):
            popt, _ = curve_fit(
                voigt_profile,
                xs_sorted,
                ys_sorted,
                p0=[A, x0, width_guess, width_guess, C],
                maxfev=maxfev,
                bounds=bounds,
            )
        A, x0, sigma, gamma, C = popt
        yfit = voigt_profile(xs_sorted, *popt)
        fwhm = 0.5346 * (2.0 * abs(gamma)) + math.sqrt(
            0.2166 * (2.0 * abs(gamma)) ** 2 + (2.3548 * abs(sigma)) ** 2
        )
        component = voigt_profile(xs_sorted, A, x0, sigma, gamma, 0.0)
    else:
        bounds = (
            [0.0, x0_min, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf],
        )
        with _FitTimeoutGuard(fit_timeout_sec):
            popt, _ = curve_fit(
                gaussian,
                xs_sorted,
                ys_sorted,
                p0=[A, x0, width_guess, C],
                maxfev=maxfev,
                bounds=bounds,
            )
        A, x0, sigma, C = popt
        yfit = gaussian(xs_sorted, *popt)
        fwhm = 2.3548 * abs(sigma)
        component = gaussian(xs_sorted, A, x0, sigma, 0.0)
    r2 = 1 - np.sum((ys_sorted - yfit) ** 2) / (np.sum((ys_sorted - np.mean(ys_sorted)) ** 2) + 1e-12)
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(component, xs_sorted))
    else:
        area = float(np.trapz(component, xs_sorted))
    return dict(center=float(x0), fwhm=float(fwhm), amplitude=float(A), area=area, r2=float(r2))


def _fit_multi_peak(
    xs: np.ndarray,
    ys: np.ndarray,
    centers: List[float],
    model: str,
    *,
    maxfev: int = 10000,
    fit_timeout_sec: float = 0.0,
) -> Optional[Tuple[List[Dict[str, float]], float]]:
    if xs.size < 5 or not centers:
        return None
    model = model.lower()
    if model not in {"gaussian", "lorentzian", "voigt"}:
        return None
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    step = _resolve_step_from_x(xs_sorted)
    C0 = float(np.median(ys_sorted))
    span = float(np.abs(xs_sorted[-1] - xs_sorted[0]))
    width_guess = max(step * 2.0, span / max(len(centers) * 3.0, 3.0))
    params = []
    lower = []
    upper = []
    for center in centers:
        local_idx = int(np.argmin(np.abs(xs_sorted - center)))
        A0 = max(float(ys_sorted[local_idx] - C0), 1e-6)
        params.append(A0)
        params.append(float(center))
        params.append(width_guess)
        if model == "voigt":
            params.append(width_guess)
            lower.extend([0.0, float(np.min(xs_sorted)), step * 0.25, step * 0.25])
            upper.extend([np.inf, float(np.max(xs_sorted)), np.inf, np.inf])
        else:
            lower.extend([0.0, float(np.min(xs_sorted)), step * 0.25])
            upper.extend([np.inf, float(np.max(xs_sorted)), np.inf])
    params.append(C0)
    lower.append(-np.inf)
    upper.append(np.inf)

    def _sum_gaussian(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += gaussian(x, p[idx], p[idx + 1], p[idx + 2], 0.0)
            idx += 3
        total += p[-1]
        return total

    def _sum_lorentzian(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += lorentzian(x, p[idx], p[idx + 1], p[idx + 2], 0.0)
            idx += 3
        total += p[-1]
        return total

    def _sum_voigt(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += voigt_profile(x, p[idx], p[idx + 1], p[idx + 2], p[idx + 3], 0.0)
            idx += 4
        total += p[-1]
        return total

    fit_fn = _sum_gaussian if model == "gaussian" else _sum_lorentzian if model == "lorentzian" else _sum_voigt
    with _FitTimeoutGuard(fit_timeout_sec):
        popt, _ = curve_fit(
            fit_fn,
            xs_sorted,
            ys_sorted,
            p0=params,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
    yfit = fit_fn(xs_sorted, *popt)
    r2 = 1 - np.sum((ys_sorted - yfit) ** 2) / (np.sum((ys_sorted - np.mean(ys_sorted)) ** 2) + 1e-12)
    results: List[Dict[str, float]] = []
    idx = 0
    for _ in centers:
        A = float(popt[idx])
        x0 = float(popt[idx + 1])
        if model == "voigt":
            sigma = float(popt[idx + 2])
            gamma = float(popt[idx + 3])
            fwhm = 0.5346 * (2.0 * abs(gamma)) + math.sqrt(
                0.2166 * (2.0 * abs(gamma)) ** 2 + (2.3548 * abs(sigma)) ** 2
            )
            component = voigt_profile(xs_sorted, A, x0, sigma, gamma, 0.0)
            idx += 4
        elif model == "lorentzian":
            gamma = float(popt[idx + 2])
            fwhm = 2.0 * abs(gamma)
            component = lorentzian(xs_sorted, A, x0, gamma, 0.0)
            idx += 3
        else:
            sigma = float(popt[idx + 2])
            fwhm = 2.3548 * abs(sigma)
            component = gaussian(xs_sorted, A, x0, sigma, 0.0)
            idx += 3
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(component, xs_sorted))
        else:
            area = float(np.trapz(component, xs_sorted))
        results.append(
            dict(center=x0, fwhm=float(fwhm), amplitude=A, area=area, r2=float(r2))
        )
    return results, float(r2)


def _fit_peak_window(x: np.ndarray, y: np.ndarray, idx: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    i0 = max(0, idx - window)
    i1 = min(n, idx + window)
    return x[i0:i1], y[i0:i1]


def fit_peak(
    x,
    y,
    idx,
    model,
    window,
    *,
    center_bounds: tuple[float, float] | None = None,
    x0_guess: float | None = None,
    maxfev: int = 10000,
    fit_timeout_sec: float = 0.0,
    raise_on_error: bool = False,
):
    xs, ys = _fit_peak_window(x, y, idx, window)
    if len(xs) < 5:
        return None
    if x0_guess is None:
        x0_guess = float(x[idx])
    step = _resolve_step_from_x(xs)
    max_delta = step * window * 0.5
    local_mask = np.abs(xs - x0_guess) <= max_delta
    if np.count_nonzero(local_mask) >= 5:
        xs = xs[local_mask]
        ys = ys[local_mask]
    if raise_on_error:
        return _fit_peak_single(
            xs,
            ys,
            x0_guess,
            model,
            center_bounds=center_bounds,
            maxfev=maxfev,
            fit_timeout_sec=fit_timeout_sec,
        )
    try:
        return _fit_peak_single(
            xs,
            ys,
            x0_guess,
            model,
            center_bounds=center_bounds,
            maxfev=maxfev,
            fit_timeout_sec=fit_timeout_sec,
        )
    except Exception:
        return None


def merge_peak_indices(y_proc: np.ndarray, pos_idxs: np.ndarray, neg_idxs: np.ndarray, min_distance_pts: int) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int, float]] = []
    if pos_idxs is not None:
        for idx in np.asarray(pos_idxs, dtype=int):
            if 0 <= idx < len(y_proc):
                candidates.append((int(idx), 1, float(abs(y_proc[int(idx)]))))
    if neg_idxs is not None:
        for idx in np.asarray(neg_idxs, dtype=int):
            if 0 <= idx < len(y_proc):
                candidates.append((int(idx), -1, float(abs(y_proc[int(idx)]))))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    min_distance_pts = max(int(min_distance_pts), 0)
    for idx, polarity, score in candidates:
        if any(abs(idx - prev_idx) <= min_distance_pts for prev_idx, _, _ in selected):
            continue
        selected.append((idx, polarity, score))
    selected.sort(key=lambda item: item[0])
    return [(idx, polarity) for idx, polarity, _ in selected]


def _get_param(args: object, name: str, default):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _resolve_step(x: np.ndarray, fallback: float) -> float:
    diffs = np.diff(x)
    if diffs.size:
        step = float(np.nanmedian(np.abs(diffs)))
    else:
        step = float("nan")
    if not np.isfinite(step) or step <= 0:
        span = float(np.nanmax(x) - np.nanmin(x)) if x.size else float("nan")
        if np.isfinite(span) and span > 0 and len(x) > 1:
            step = span / (len(x) - 1)
        else:
            step = fallback if fallback > 0 else 1.0
    return step


def _resolve_width_bounds_cm(
    width_min_cm: float,
    width_max_cm: float,
    *,
    step_cm: float,
    resolution_cm: float | None,
) -> tuple[float, float]:
    width_min_cm = float(width_min_cm)
    width_max_cm = float(width_max_cm)
    if width_min_cm <= 0:
        if resolution_cm is not None and resolution_cm > 0:
            width_min_cm = max(float(resolution_cm), step_cm)
        else:
            width_min_cm = step_cm
    if width_max_cm <= 0:
        width_max_cm = max(width_min_cm * 6.0, width_min_cm + step_cm * 4.0)
    if width_max_cm < width_min_cm:
        width_max_cm = width_min_cm
    return width_min_cm, width_max_cm


def _cluster_indices_by_tolerance(
    indices: np.ndarray,
    *,
    x: np.ndarray,
    scores: np.ndarray,
    tolerance_cm: float,
) -> List[int]:
    if indices.size == 0:
        return []
    order = np.argsort(x[indices])
    indices = indices[order]
    scores = scores[order]
    groups: List[List[int]] = []
    current: List[int] = [int(indices[0])]
    for idx in indices[1:]:
        if abs(float(x[int(idx)]) - float(x[current[-1]])) <= tolerance_cm:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)
    representatives: List[int] = []
    for group in groups:
        group_scores = [scores[np.where(indices == g)[0][0]] for g in group]
        best_idx = group[int(np.argmax(group_scores))]
        representatives.append(best_idx)
    return representatives


def _build_cwt_widths(
    *,
    width_min_cm: float,
    width_max_cm: float,
    width_step_cm: float,
    step_cm: float,
) -> np.ndarray:
    if width_step_cm <= 0:
        width_step_cm = max(step_cm, (width_max_cm - width_min_cm) / 8.0 if width_max_cm > width_min_cm else step_cm)
    widths_cm = np.arange(width_min_cm, width_max_cm + width_step_cm * 0.5, width_step_cm)
    widths_pts = np.unique(np.clip(np.round(widths_cm / step_cm), 1, None)).astype(int)
    return widths_pts


def detect_peak_candidates(
    x: np.ndarray,
    y_proc: np.ndarray,
    noise_sigma: float,
    args: object,
    *,
    resolution_cm: float | None = None,
    file_path: str | None = None,
    spectrum_id: int | None = None,
) -> List[Dict[str, object]]:
    min_distance_cm = float(_get_param(args, "min_distance", 0.8))
    x_clean = np.asarray(x, dtype=float)
    x_clean = x_clean[np.isfinite(x_clean)]
    if x_clean.size > 1:
        x_sorted = np.sort(x_clean)
        step_cm = _resolve_step_from_x(x_sorted)
    else:
        step_cm = _resolve_step(x, min_distance_cm)
    distance_pts = max(1, int(np.floor(min_distance_cm / max(step_cm, 1e-9))))
    min_distance_mode = str(_get_param(args, "min_distance_mode", "fixed")).lower()
    min_distance_fwhm_fraction = float(_get_param(args, "min_distance_fwhm_fraction", 0.5))
    if min_distance_mode not in {"fixed", "adaptive"}:
        logger.warning(
            "Unknown min_distance_mode=%s, falling back to fixed behavior.",
            min_distance_mode,
        )
        min_distance_mode = "fixed"

    prominence_floor = float(_get_param(args, "prominence", 0.003))
    min_absorbance_threshold = float(_get_param(args, "min_absorbance_threshold", 0.05))
    sigma_multiplier = float(_get_param(args, "noise_sigma_multiplier", 1.5))
    noise_window_cm = float(_get_param(args, "noise_window_cm", 0.0))
    region_prominence_spec = _get_param(args, "min_prominence_by_region", None)
    region_prominence = _parse_min_prominence_by_region(region_prominence_spec)
    segments = _build_prominence_segments(x, noise_window_cm, region_prominence)
    if not segments:
        segments = [(float(np.nanmin(x)), float(np.nanmax(x)), True)]

    width_min_arg_raw = float(_get_param(args, "peak_width_min", 0.0))
    width_max_arg = float(_get_param(args, "peak_width_max", 0.0))
    width_min_arg = 0.0
    width_min_cm, width_max_cm = _resolve_width_bounds_cm(
        width_min_arg,
        width_max_arg,
        step_cm=step_cm,
        resolution_cm=resolution_cm,
    )
    width_min_pts = max(1, int(np.ceil(width_min_cm / max(step_cm, 1e-9))))
    width_max_pts = max(width_min_pts, int(np.ceil(width_max_cm / max(step_cm, 1e-9))))
    width_filter_enabled = width_max_arg > 0
    width_bounds = (width_min_pts, width_max_pts) if width_filter_enabled else None

    detect_negative = bool(_get_param(args, "detect_negative_peaks", False))
    cwt_enabled = bool(_get_param(args, "cwt_enabled", False))
    cwt_width_min_cm = float(_get_param(args, "cwt_width_min", 0.0))
    cwt_width_max_cm = float(_get_param(args, "cwt_width_max", 0.0))
    cwt_width_step_cm = float(_get_param(args, "cwt_width_step", 0.0))
    cwt_cluster_tolerance_cm = float(_get_param(args, "cwt_cluster_tolerance", 8.0))
    merge_tolerance_cm = float(_get_param(args, "merge_tolerance", 8.0))
    close_peak_tolerance_cm = float(_get_param(args, "close_peak_tolerance_cm", 1.5))
    shoulder_min_distance_cm = float(
        _get_param(args, "shoulder_min_distance_cm", max(step_cm, min_distance_cm * 0.5))
    )
    shoulder_merge_tolerance_cm = float(_get_param(args, "shoulder_merge_tolerance_cm", 1.5))
    shoulder_curvature_prominence_factor = float(
        _get_param(args, "shoulder_curvature_prominence_factor", 1.2)
    )

    fallback_noise_sigma = _estimate_segment_noise_sigma(
        y_proc,
        noise_sigma if np.isfinite(noise_sigma) else float("nan"),
    )
    if noise_window_cm > 0 and np.isfinite(step_cm) and step_cm > 0:
        window_pts = int(max(3, round(noise_window_cm / step_cm)))
    else:
        window_pts = max(3, int(y_proc.size))
    local_noise_sigma = _rolling_mad_sigma(y_proc, window_pts, fallback_noise_sigma)
    local_noise_floor = sigma_multiplier * np.where(
        np.isfinite(local_noise_sigma) & (local_noise_sigma > 0),
        local_noise_sigma,
        0.0,
    )

    def _is_nonflat_peak(target: np.ndarray, idx: int) -> bool:
        if idx < 0 or idx >= target.size:
            return False
        height = float(target[idx])
        if not np.isfinite(height) or height <= 0:
            return False
        left = max(0, idx - 1)
        right = min(target.size, idx + 2)
        window = target[left:right]
        if window.size < 2:
            return False
        finite = window[np.isfinite(window)]
        if finite.size < 2:
            return False
        return float(np.nanmax(finite) - np.nanmin(finite)) > 0

    finite_local_floor = local_noise_floor[np.isfinite(local_noise_floor)]
    adaptive_fwhm_cm = float("nan")
    if min_distance_mode == "adaptive":
        adaptive_floor = prominence_floor
        if finite_local_floor.size:
            adaptive_floor = max(adaptive_floor, float(np.nanmedian(finite_local_floor)))

        def _estimate_median_fwhm_cm(prominence_value: float) -> float:
            if y_proc.size < 3:
                return 0.0
            target = np.abs(y_proc)
            peak_idxs, _ = find_peaks(target, prominence=prominence_value)
            if peak_idxs.size == 0:
                return 0.0
            valid_peak_idxs = [
                int(idx) for idx in np.asarray(peak_idxs, dtype=int) if _is_nonflat_peak(target, int(idx))
            ]
            if not valid_peak_idxs:
                return 0.0
            nonlocal suppressed_peak_property_warnings
            caught: List[warnings.WarningMessage] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.filterwarnings("always", category=PeakPropertyWarning)
                widths, _, left_ips, right_ips = peak_widths(
                    target,
                    valid_peak_idxs,
                    rel_height=0.5,
                )
            suppressed_peak_property_warnings += _count_peak_property_warnings(caught)
            xs = np.arange(len(x), dtype=float)
            fwhm_values: List[float] = []
            for left_ip, right_ip, width in zip(left_ips, right_ips, widths):
                if np.isfinite(left_ip) and np.isfinite(right_ip):
                    left_x = float(np.interp(left_ip, xs, x))
                    right_x = float(np.interp(right_ip, xs, x))
                    fwhm = abs(right_x - left_x)
                else:
                    fwhm = float(width) * step_cm
                if np.isfinite(fwhm) and fwhm > 0:
                    fwhm_values.append(fwhm)
            if not fwhm_values:
                return 0.0
            return float(np.nanmedian(fwhm_values))

        adaptive_fwhm_cm = _estimate_median_fwhm_cm(adaptive_floor)
        if adaptive_fwhm_cm > 0 and min_distance_fwhm_fraction > 0:
            adaptive_limit_cm = adaptive_fwhm_cm * min_distance_fwhm_fraction
            if np.isfinite(adaptive_limit_cm) and adaptive_limit_cm > 0:
                min_distance_cm = min(min_distance_cm, adaptive_limit_cm)
                distance_pts = max(1, int(np.floor(min_distance_cm / max(step_cm, 1e-9))))

    if min_distance_cm > 0:
        merge_tolerance_cm = min(merge_tolerance_cm, min_distance_cm)

    logger.info(
        "Peak thresholds path=%s spectrum_id=%s min_distance_cm=%.4g step_cm=%.4g "
        "distance_pts=%d min_distance_mode=%s min_distance_fwhm_fraction=%.3g "
        "adaptive_fwhm_cm=%.4g prominence_floor=%.4g noise_sigma=%.4g noise_sigma_multiplier=%.3g "
        "min_absorbance_threshold=%.4g "
        "width_min_cm=%.4g width_max_cm=%.4g width_min_pts=%d width_max_pts=%d width_filter=%s "
        "plateau_min_points=%d plateau_prominence_factor=%.3g",
        file_path or "unknown",
        spectrum_id if spectrum_id is not None else "unknown",
        min_distance_cm,
        step_cm,
        distance_pts,
        min_distance_mode,
        min_distance_fwhm_fraction,
        adaptive_fwhm_cm,
        prominence_floor,
        float(noise_sigma) if np.isfinite(noise_sigma) else float("nan"),
        sigma_multiplier,
        min_absorbance_threshold,
        width_min_cm,
        width_max_cm,
        width_min_pts,
        width_max_pts,
        width_filter_enabled,
        int(_get_param(args, "plateau_min_points", 3)),
        float(_get_param(args, "plateau_prominence_factor", 1.0)),
    )
    if width_min_arg_raw > 0:
        logger.info(
            "Peak width min override path=%s spectrum_id=%s requested_min=%.4g applied_min=0.0",
            file_path or "unknown",
            spectrum_id if spectrum_id is not None else "unknown",
            width_min_arg_raw,
        )
    if finite_local_floor.size:
        logger.info(
            "Local noise floor path=%s spectrum_id=%s median=%.4g min=%.4g max=%.4g",
            file_path or "unknown",
            spectrum_id if spectrum_id is not None else "unknown",
            float(np.nanmedian(finite_local_floor)),
            float(np.nanmin(finite_local_floor)),
            float(np.nanmax(finite_local_floor)),
        )
    else:
        logger.info(
            "Local noise floor path=%s spectrum_id=%s median=nan min=nan max=nan",
            file_path or "unknown",
            spectrum_id if spectrum_id is not None else "unknown",
        )

    candidates: List[Dict[str, object]] = []
    fwhm_cache: Dict[Tuple[int, int], float] = {}
    zero_width_peaks: set[Tuple[int, int]] = set()
    suppressed_peak_property_warnings = 0

    def _count_peak_property_warnings(caught: List[warnings.WarningMessage]) -> int:
        return sum(
            1 for warning in caught if issubclass(warning.category, PeakPropertyWarning)
        )

    def _mark_zero_width(index: int, polarity: int) -> None:
        zero_width_peaks.add((int(index), int(polarity)))

    def _add_candidate(
        index: int,
        polarity: int,
        source: str,
        pass_label: str,
        metadata: Dict[str, object],
    ):
        amplitude = float(abs(y_proc[int(index)])) if 0 <= int(index) < len(y_proc) else 0.0
        is_shoulder = pass_label == "shoulder"
        score = amplitude * (1.15 if is_shoulder else 1.0)
        candidates.append(
            {
                "index": int(index),
                "polarity": int(polarity),
                "score": score,
                "baseline_corrected_amplitude": amplitude,
                "detections": [metadata | {"source": source, "pass": pass_label}],
                "sources": {pass_label},
                "close_peak": False,
                "priority": 1 if is_shoulder else 0,
            }
        )

    def _estimate_fwhm_cm(index: int, polarity: int) -> float:
        nonlocal suppressed_peak_property_warnings
        cache_key = (int(index), int(polarity))
        if cache_key in fwhm_cache:
            return fwhm_cache[cache_key]
        y_target = y_proc if polarity >= 0 else -y_proc
        fwhm_cm = float("nan")
        if not _is_nonflat_peak(y_target, int(index)):
            fwhm_cm = 0.0
            _mark_zero_width(index, polarity)
            fwhm_cache[cache_key] = fwhm_cm
            return fwhm_cm
        caught: List[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.filterwarnings("always", category=PeakPropertyWarning)
                widths, _, left_ips, right_ips = peak_widths(
                    y_target,
                    [int(index)],
                    rel_height=0.5,
                )
            if widths.size:
                left_ip = float(left_ips[0])
                right_ip = float(right_ips[0])
                if np.isfinite(left_ip) and np.isfinite(right_ip):
                    xs = np.arange(len(x), dtype=float)
                    left_x = float(np.interp(left_ip, xs, x))
                    right_x = float(np.interp(right_ip, xs, x))
                    fwhm_cm = abs(right_x - left_x)
                else:
                    fwhm_cm = float(widths[0]) * step_cm
        except (ValueError, IndexError):
            fwhm_cm = float("nan")
        suppressed_peak_property_warnings += _count_peak_property_warnings(caught)
        if not np.isfinite(fwhm_cm):
            fwhm_cm = 0.0
        if fwhm_cm <= 0:
            _mark_zero_width(index, polarity)
        fwhm_cache[cache_key] = fwhm_cm
        return fwhm_cm

    def _merge_tolerance_cm(index_a: int, index_b: int, polarity: int) -> float:
        fwhm_a = _estimate_fwhm_cm(index_a, polarity)
        fwhm_b = _estimate_fwhm_cm(index_b, polarity)
        fwhm = max(fwhm_a, fwhm_b)
        half_fwhm = 0.5 * fwhm if fwhm > 0 else 0.0
        target_tolerance = max(2.0, min(4.0, half_fwhm if half_fwhm > 0 else 4.0))
        return min(merge_tolerance_cm, target_tolerance)

    plateau_min_points = max(2, int(_get_param(args, "plateau_min_points", 3)))
    plateau_prominence_factor = float(_get_param(args, "plateau_prominence_factor", 1.0))

    for seg_low, seg_high, seg_last in segments:
        if seg_low == seg_high:
            continue
        if seg_last:
            mask = (x >= seg_low) & (x <= seg_high)
        else:
            mask = (x >= seg_low) & (x < seg_high)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        start, end = int(idxs[0]), int(idxs[-1]) + 1
        y_slice = y_proc[start:end]
        region_floor = _region_prominence_floor(seg_low, seg_high, region_prominence)
        base_floor = max(prominence_floor, region_floor)
        pass1_prominence = np.maximum(base_floor, local_noise_floor[start:end])
        pass2_prominence = np.maximum(base_floor * 1.5, local_noise_floor[start:end])
        pass1_distance = distance_pts
        pass2_distance = max(1, int(distance_pts * 1.2))
        width_bounds_pass1 = width_bounds
        width_bounds_pass2 = width_bounds

        def _record_prominence_candidates(
            target: np.ndarray,
            polarity: int,
            source: str,
            pass_label: str,
            prominence_vals: np.ndarray,
            distance_value: int,
            width_bounds_value: tuple[int, int] | None,
        ) -> None:
            nonlocal suppressed_peak_property_warnings
            min_prominence = float(np.nanmin(prominence_vals)) if prominence_vals.size else 0.0
            if not np.isfinite(min_prominence) or min_prominence <= 0:
                return
            try:
                if width_bounds_value is None:
                    peak_idxs, props = find_peaks(target, prominence=min_prominence, distance=distance_value)
                else:
                    peak_idxs, props = find_peaks(
                        target,
                        prominence=min_prominence,
                        distance=distance_value,
                        width=width_bounds_value,
                    )
            except (ValueError, IndexError):
                peak_idxs = np.empty(0, dtype=int)
                props = {}
            prominences = np.asarray(props.get("prominences", []), dtype=float)
            if peak_idxs.size == 0:
                return
            if prominences.size != peak_idxs.size:
                prominences = np.full(peak_idxs.size, min_prominence, dtype=float)
            for idx, prominence_value in zip(np.asarray(peak_idxs, dtype=int), prominences):
                abs_idx = int(idx) + start
                if abs_idx < 0 or abs_idx >= len(y_proc):
                    continue
                if not np.isfinite(prominence_value):
                    continue
                pass_id = f"{pass_label}_{polarity}"
                local_floor = float(prominence_vals[int(idx)]) if int(idx) < len(prominence_vals) else 0.0
                required_floor = max(min_prominence, local_floor)
                if prominence_value < required_floor:
                    continue
                if 0 <= abs_idx < len(y_proc):
                    _add_candidate(
                        abs_idx,
                        polarity,
                        source,
                        pass_label,
                        {
                            "pass_id": pass_id,
                            "prominence": float(prominence_value),
                            "prominence_floor": float(required_floor),
                            "width_pts": props.get("widths", [np.nan])[0] if props else np.nan,
                            "segment_range": (float(seg_low), float(seg_high)),
                        },
                    )

        _record_prominence_candidates(
            y_slice,
            1,
            "primary",
            "pass1",
            pass1_prominence,
            pass1_distance,
            width_bounds_pass1,
        )
        _record_prominence_candidates(
            y_slice,
            1,
            "primary",
            "pass2",
            pass2_prominence,
            pass2_distance,
            width_bounds_pass2,
        )

        plateau_base = float(np.nanmedian(pass1_prominence)) if pass1_prominence.size else 0.0
        if not np.isfinite(plateau_base):
            plateau_base = 0.0
        plateau_prominence = max(plateau_base * plateau_prominence_factor, 0.0)
        if plateau_min_points >= 2:
            plateau_idxs, plateau_props = find_peaks(
                y_slice,
                prominence=plateau_prominence,
                distance=distance_pts,
                plateau_size=plateau_min_points,
            )
            if plateau_idxs.size:
                left_edges = plateau_props.get("left_edges", [])
                right_edges = plateau_props.get("right_edges", [])
                sizes = plateau_props.get("plateau_sizes", [])
                for idx, left, right, size in zip(plateau_idxs, left_edges, right_edges, sizes):
                    left = int(left)
                    right = int(right)
                    center = int(round((left + right) / 2))
                    abs_center = center + start
                    if 0 <= abs_center < len(y_proc):
                        _add_candidate(
                            abs_center,
                            1,
                            "plateau",
                            "primary",
                            {
                                "pass_id": "plateau",
                                "plateau_left": left + start,
                                "plateau_right": right + start,
                                "plateau_size": int(size),
                                "prominence": plateau_prominence,
                                "segment_range": (float(seg_low), float(seg_high)),
                            },
                        )

        if detect_negative:
            _record_prominence_candidates(
                y_slice,
                -1,
                "primary",
                "pass1",
                pass1_prominence,
                pass1_distance,
                width_bounds_pass1,
            )
            _record_prominence_candidates(
                y_slice,
                -1,
                "primary",
                "pass2",
                pass2_prominence,
                pass2_distance,
                width_bounds_pass2,
            )

        def _record_curvature_candidates(slice_data: np.ndarray, polarity: int) -> None:
            curvature = -_compute_second_derivative(
                x[start:end],
                slice_data,
                max(1, int(shoulder_min_distance_cm / max(step_cm, 1e-9))),
            )
            if curvature.size == 0:
                return
            median_curvature = float(np.nanmedian(np.abs(curvature)))
            if not np.isfinite(median_curvature) or median_curvature <= 0:
                median_curvature = float(np.nanmax(np.abs(curvature))) * 0.25 if curvature.size else 0.0
            prominence = max(median_curvature * shoulder_curvature_prominence_factor, 1e-9)
            peak_idxs, props = find_peaks(
                curvature,
                prominence=prominence,
                distance=max(1, int(shoulder_min_distance_cm / max(step_cm, 1e-9))),
            )
            prominences = np.asarray(props.get("prominences", []), dtype=float)
            for idx, prominence_value in zip(np.asarray(peak_idxs, dtype=int), prominences):
                abs_idx = int(idx) + start
                if not np.isfinite(prominence_value) or prominence_value < prominence:
                    continue
                if 0 <= abs_idx < len(y_proc):
                    _add_candidate(
                        abs_idx,
                        polarity,
                        "curvature",
                        "shoulder",
                        {
                            "pass_id": "curvature",
                            "curvature": float(curvature[int(idx)]),
                            "curvature_prominence": float(prominence_value),
                            "curvature_floor": float(prominence),
                            "segment_range": (float(seg_low), float(seg_high)),
                        },
                    )

        _record_curvature_candidates(y_slice, 1)
        if detect_negative:
            _record_curvature_candidates(-y_slice, -1)

    if cwt_enabled:
        cwt_width_min_cm, cwt_width_max_cm = _resolve_width_bounds_cm(
            cwt_width_min_cm or width_min_cm,
            cwt_width_max_cm or width_max_cm,
            step_cm=step_cm,
            resolution_cm=resolution_cm,
        )
        widths_pts = _build_cwt_widths(
            width_min_cm=cwt_width_min_cm,
            width_max_cm=cwt_width_max_cm,
            width_step_cm=cwt_width_step_cm,
            step_cm=step_cm,
        )
        cwt_pos = np.asarray(find_peaks_cwt(y_proc, widths_pts), dtype=int)
        if cwt_pos.size:
            scores = np.abs(y_proc[cwt_pos])
            clustered = _cluster_indices_by_tolerance(
                cwt_pos, x=x, scores=scores, tolerance_cm=cwt_cluster_tolerance_cm
            )
            for idx in clustered:
                _add_candidate(
                    idx,
                    1,
                    "cwt",
                    "primary",
                    {
                        "pass_id": "cwt",
                        "widths_pts": widths_pts.tolist(),
                        "cluster_tolerance": cwt_cluster_tolerance_cm,
                    },
                )
        if detect_negative:
            cwt_neg = np.asarray(find_peaks_cwt(-y_proc, widths_pts), dtype=int)
            if cwt_neg.size:
                scores = np.abs(y_proc[cwt_neg])
                clustered = _cluster_indices_by_tolerance(
                    cwt_neg, x=x, scores=scores, tolerance_cm=cwt_cluster_tolerance_cm
                )
                for idx in clustered:
                    _add_candidate(
                        idx,
                        -1,
                        "cwt",
                        "primary",
                        {
                            "pass_id": "cwt",
                            "widths_pts": widths_pts.tolist(),
                            "cluster_tolerance": cwt_cluster_tolerance_cm,
                        },
                    )

    if candidates and min_absorbance_threshold > 0:
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("baseline_corrected_amplitude", 0.0)) >= min_absorbance_threshold
        ]

    if not candidates:
        return []

    def _extract_plateau_bounds(detections: List[Dict[str, object]]):
        lefts = []
        rights = []
        for detection in detections:
            if detection.get("source") != "plateau":
                continue
            left = detection.get("plateau_left")
            right = detection.get("plateau_right")
            if left is None or right is None:
                continue
            lefts.append(int(left))
            rights.append(int(right))
        if not lefts or not rights:
            return None
        return min(lefts), max(rights)

    def _merge_candidates_by_tolerance(
        items: List[Dict[str, object]],
        *,
        polarity: int,
    ) -> List[Dict[str, object]]:
        if not items:
            return []
        items_sorted = sorted(items, key=lambda c: int(c["index"]))
        merged_items: List[Dict[str, object]] = []
        for candidate in items_sorted:
            idx = int(candidate["index"])
            if not merged_items:
                merged_items.append(candidate)
                continue
            prev = merged_items[-1]
            prev_idx = int(prev["index"])
            tolerance_cm = _merge_tolerance_cm(idx, prev_idx, polarity)
            if "shoulder" in candidate.get("sources", []) or "shoulder" in prev.get("sources", []):
                tolerance_cm = min(tolerance_cm, shoulder_merge_tolerance_cm)
            if abs(float(x[idx]) - float(x[prev_idx])) > tolerance_cm:
                merged_items.append(candidate)
                continue
            logger.info(
                "Peak merge by tolerance path=%s spectrum_id=%s idx=%d existing_idx=%d "
                "tolerance_cm=%.3g dropped_index=%d kept_index=%d",
                file_path or "unknown",
                spectrum_id if spectrum_id is not None else "unknown",
                idx,
                prev_idx,
                tolerance_cm,
                idx,
                prev_idx,
            )
            prev["sources"] = set(prev["sources"]) | set(candidate["sources"])
            prev["detections"] = list(prev["detections"]) + list(candidate["detections"])
            prev["close_peak"] = bool(prev.get("close_peak")) or bool(candidate.get("close_peak"))
            prev_priority = int(prev.get("priority", 0) or 0)
            candidate_priority = int(candidate.get("priority", 0) or 0)
            if candidate_priority > prev_priority or (
                candidate_priority == prev_priority and float(candidate["score"]) > float(prev["score"])
            ):
                prev["index"] = idx
                prev["score"] = float(candidate["score"])
                prev["priority"] = candidate_priority
            else:
                prev["priority"] = max(prev_priority, candidate_priority)
        return merged_items

    merged = (
        _merge_candidates_by_tolerance(
            [c for c in candidates if int(c["polarity"]) == 1],
            polarity=1,
        )
        + _merge_candidates_by_tolerance(
            [c for c in candidates if int(c["polarity"]) == -1],
            polarity=-1,
        )
    )

    def _cluster_close_candidates(
        items: List[Dict[str, object]],
        *,
        tolerance_cm: float,
        polarity: int,
    ) -> List[Dict[str, object]]:
        if len(items) <= 1:
            return items
        items_sorted = sorted(items, key=lambda c: float(x[int(c["index"])]))
        clusters: List[List[Dict[str, object]]] = []
        current = [items_sorted[0]]
        for candidate in items_sorted[1:]:
            prev = current[-1]
            if abs(float(x[int(candidate["index"])]) - float(x[int(prev["index"])]) ) <= tolerance_cm:
                current.append(candidate)
            else:
                clusters.append(current)
                current = [candidate]
        clusters.append(current)
        clustered: List[Dict[str, object]] = []
        for cluster in clusters:
            if len(cluster) == 1:
                clustered.append(cluster[0])
                continue
            best = max(cluster, key=lambda c: (int(c.get("priority", 0) or 0), float(c["score"])))
            merged_cluster = dict(best)
            merged_cluster["detections"] = []
            merged_cluster["sources"] = set()
            merged_cluster["close_peak"] = True
            for candidate in cluster:
                merged_cluster["detections"].extend(candidate.get("detections", []))
                merged_cluster["sources"] = merged_cluster["sources"] | set(candidate.get("sources", []))
                merged_cluster["priority"] = max(
                    int(merged_cluster.get("priority", 0) or 0),
                    int(candidate.get("priority", 0) or 0),
                )
            logger.debug(
                "Close-peak merge path=%s spectrum_id=%s polarity=%d indices=%s tolerance_cm=%.3g",
                file_path or "unknown",
                spectrum_id if spectrum_id is not None else "unknown",
                polarity,
                [int(c["index"]) for c in cluster],
                tolerance_cm,
            )
            logger.info(
                "Close-peak merge path=%s spectrum_id=%s polarity=%d indices=%s tolerance_cm=%.3g "
                "kept_index=%d dropped_indices=%s",
                file_path or "unknown",
                spectrum_id if spectrum_id is not None else "unknown",
                polarity,
                [int(c["index"]) for c in cluster],
                tolerance_cm,
                int(best["index"]),
                [int(c["index"]) for c in cluster if c is not best],
            )
            clustered.append(merged_cluster)
        return clustered

    merged = (
        _cluster_close_candidates(
            [c for c in merged if int(c["polarity"]) == 1],
            tolerance_cm=close_peak_tolerance_cm,
            polarity=1,
        )
        + _cluster_close_candidates(
            [c for c in merged if int(c["polarity"]) == -1],
            tolerance_cm=close_peak_tolerance_cm,
            polarity=-1,
        )
    )

    merged.sort(key=lambda c: c["index"])
    max_candidates = int(_get_param(args, "max_peak_candidates", 0) or 0)
    min_prominence = np.finfo(float).eps
    min_width_pts = np.finfo(float).eps

    def _annotate_and_filter_metrics(
        items: List[Dict[str, object]],
        *,
        polarity: int,
    ) -> List[Dict[str, object]]:
        nonlocal suppressed_peak_property_warnings
        if not items:
            return []
        candidate_indices = [int(item["index"]) for item in items]
        target = y_proc if polarity >= 0 else -y_proc
        valid_items: List[Dict[str, object]] = []
        valid_indices: List[int] = []
        for item, idx in zip(items, candidate_indices):
            if _is_nonflat_peak(target, int(idx)):
                valid_items.append(item)
                valid_indices.append(int(idx))
            else:
                _mark_zero_width(idx, polarity)
        if not valid_indices:
            return []
        indices = np.asarray(valid_indices, dtype=int)
        prominences = np.full(indices.shape, np.nan, dtype=float)
        widths = np.full(indices.shape, np.nan, dtype=float)
        caught_prominences: List[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught_prominences:
                warnings.filterwarnings("always", category=PeakPropertyWarning)
                prominences, _, _ = peak_prominences(target, indices)
        except (ValueError, IndexError):
            pass
        suppressed_peak_property_warnings += _count_peak_property_warnings(caught_prominences)
        caught_widths: List[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught_widths:
                warnings.filterwarnings("always", category=PeakPropertyWarning)
                widths, _, _, _ = peak_widths(target, indices, rel_height=0.5)
        except (ValueError, IndexError):
            pass
        suppressed_peak_property_warnings += _count_peak_property_warnings(caught_widths)
        filtered: List[Dict[str, object]] = []
        for item, prominence_value, width_value in zip(valid_items, prominences, widths):
            prominence_value = float(prominence_value) if np.isfinite(prominence_value) else 0.0
            width_value = float(width_value) if np.isfinite(width_value) else 0.0
            item["prominence"] = prominence_value
            item["width_pts"] = width_value
            if prominence_value <= min_prominence or width_value <= min_width_pts:
                if width_value <= min_width_pts:
                    _mark_zero_width(item.get("index", 0), polarity)
                continue
            filtered.append(item)
        return filtered

    merged = _annotate_and_filter_metrics(
        [c for c in merged if int(c["polarity"]) == 1],
        polarity=1,
    ) + _annotate_and_filter_metrics(
        [c for c in merged if int(c["polarity"]) == -1],
        polarity=-1,
    )
    if suppressed_peak_property_warnings:
        log_line(
            "Warning: zero-width peaks encountered "
            f"path={file_path or 'unknown'} spectrum_id={spectrum_id if spectrum_id is not None else 'unknown'} "
            f"count={suppressed_peak_property_warnings}",
            stream=sys.stderr,
        )

    if max_candidates > 0 and len(merged) > max_candidates:
        original_count = len(merged)
        merged = sorted(
            merged,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )[:max_candidates]
        logger.info(
            "Peak candidate cap applied path=%s spectrum_id=%s kept=%d dropped=%d",
            file_path or "unknown",
            spectrum_id if spectrum_id is not None else "unknown",
            len(merged),
            max(0, original_count - len(merged)),
        )
        merged.sort(key=lambda c: c["index"])
    for candidate_id, candidate in enumerate(merged):
        candidate["candidate_id"] = int(candidate_id)
        candidate["sources"] = sorted(candidate.get("sources", []))
        plateau_bounds = _extract_plateau_bounds(candidate.get("detections", []))
        if plateau_bounds is not None:
            candidate["plateau_bounds"] = plateau_bounds
    return merged


def _cluster_candidates_by_window(
    candidates: List[Dict[str, object]],
    *,
    window: int,
) -> List[List[Dict[str, object]]]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda c: int(c["index"]))
    clusters: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = [sorted_candidates[0]]
    for candidate in sorted_candidates[1:]:
        if int(candidate["index"]) - int(current[-1]["index"]) <= window:
            current.append(candidate)
        else:
            clusters.append(current)
            current = [candidate]
    clusters.append(current)
    return clusters


def _curvature_seed_centers(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    distance_pts: int,
    max_peaks: int,
) -> List[float]:
    curvature = -_compute_second_derivative(xs, ys, distance_pts)
    if curvature.size == 0:
        return []
    prominence = np.nanmedian(np.abs(curvature))
    if not np.isfinite(prominence) or prominence <= 0:
        prominence = np.nanmax(np.abs(curvature)) * 0.1 if curvature.size else 0.0
    prominence = max(prominence, 1e-9)
    idxs, _ = find_peaks(curvature, prominence=prominence, distance=max(1, distance_pts))
    if idxs.size == 0:
        return []
    scores = curvature[idxs]
    order = np.argsort(scores)[::-1]
    idxs = idxs[order][:max_peaks]
    return [float(xs[int(idx)]) for idx in idxs]


def refine_peak_candidates(
    x: np.ndarray,
    y_proc: np.ndarray,
    candidates: List[Dict[str, object]],
    args: object,
    *,
    y_abs: np.ndarray | None = None,
    file_path: str | None = None,
    spectrum_id: int | None = None,
    fit_errors: Optional[List[Dict[str, object]]] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Dict[str, object]], int]:
    if not candidates:
        return [], 0
    model = str(_get_param(args, "model", "Gaussian") or "Gaussian")
    fit_window = max(3, int(_get_param(args, "fit_window_pts", 70)))
    min_r2 = float(_get_param(args, "min_r2", 0.75))
    shoulder_fit_window = int(
        _get_param(args, "shoulder_fit_window_pts", max(5, int(fit_window * 0.6)))
    )
    shoulder_min_r2 = float(_get_param(args, "shoulder_min_r2", 0.65))
    min_absorbance_threshold = float(_get_param(args, "min_absorbance_threshold", 0.05))
    fit_maxfev = int(_get_param(args, "fit_maxfev", 10000) or 10000)
    fit_maxfev = max(1, min(fit_maxfev, 20000))
    fit_timeout_sec = float(_get_param(args, "fit_timeout_sec", 900.0) or 0.0)
    multi_fit_min_span_cm = float(_get_param(args, "multi_fit_min_span_cm", 10.0) or 0.0)
    flat_max_threshold = float(
        _get_param(args, "flat_max_amplitude", max(min_absorbance_threshold * 0.5, 0.0))
    )
    flat_variance_threshold = float(
        _get_param(args, "flat_variance_threshold", (min_absorbance_threshold * 0.1) ** 2)
    )
    max_abs = float(np.nanmax(np.abs(y_proc))) if y_proc.size else 0.0
    variance = float(np.nanvar(y_proc)) if y_proc.size else 0.0
    if (
        (flat_max_threshold > 0 and max_abs <= flat_max_threshold)
        or (flat_variance_threshold > 0 and variance <= flat_variance_threshold)
    ):
        log_line(
            "Skipping peak fitting for flat spectrum "
            f"path={file_path or 'unknown'} spectrum_id={spectrum_id if spectrum_id is not None else 'unknown'} "
            f"max_abs={max_abs:.4g} variance={variance:.4g} "
            f"thresholds=(max_abs={flat_max_threshold:.4g}, variance={flat_variance_threshold:.4g})",
            stream=sys.stderr,
        )
        return []
    results: List[Dict[str, object]] = []
    timeout_skips = 0
    spectrum_timeout_skips = 0
    total_candidates = len(candidates)
    progress_every = int(_get_param(args, "fit_progress_every", 50) or 0)
    progress_every = max(progress_every, 0)
    processed_candidates = 0
    next_progress = progress_every if progress_every else None
    spectrum_deadline = (
        time.monotonic() + fit_timeout_sec if fit_timeout_sec and fit_timeout_sec > 0 else None
    )

    def _remaining_spectrum_time() -> Optional[float]:
        if spectrum_deadline is None:
            return None
        return spectrum_deadline - time.monotonic()

    def _spectrum_timed_out() -> bool:
        remaining = _remaining_spectrum_time()
        return remaining is not None and remaining <= 0

    def mark_candidate_processed(count: int = 1) -> None:
        nonlocal processed_candidates, next_progress
        if count <= 0:
            return
        processed_candidates += count
        if next_progress is None:
            return
        if processed_candidates >= next_progress:
            log_line(
                "Peak fit progress "
                f"{processed_candidates}/{total_candidates} candidates "
                f"path={file_path or 'unknown'} spectrum_id={spectrum_id if spectrum_id is not None else 'unknown'}",
                stream=sys.stderr,
            )
            next_progress += progress_every
        if progress_callback is not None:
            progress_callback(
                {
                    "file": file_path,
                    "spectrum_id": spectrum_id,
                    "stage": "fit_progress",
                    "candidates": processed_candidates,
                    "refined": len(results),
                    "skipped_due_to_timeout": timeout_skips + spectrum_timeout_skips,
                }
            )

    def _shoulder_candidate(candidate: Dict[str, object]) -> bool:
        sources = candidate.get("sources", [])
        return "shoulder" in sources or bool(candidate.get("close_peak"))

    def record_fit_failure(candidate_id: int, model_name: str, exc: BaseException) -> None:
        nonlocal timeout_skips
        if isinstance(exc, FitTimeoutError):
            timeout_skips += 1
            if fit_errors is not None:
                fit_errors.append(
                    {
                        "file_path": file_path or "unknown",
                        "spectrum_id": spectrum_id if spectrum_id is not None else "unknown",
                        "candidate_id": candidate_id,
                        "model": model_name,
                        "error": "fit_timeout",
                    }
                )
            return
        path_label = file_path or "unknown"
        spectrum_label = spectrum_id if spectrum_id is not None else "unknown"
        logger.warning(
            "Peak fit failed path=%s spectrum_id=%s candidate_id=%s model=%s error=%s",
            path_label,
            spectrum_label,
            candidate_id,
            model_name,
            exc,
        )
        if fit_errors is not None:
            fit_errors.append(
                {
                    "file_path": path_label,
                    "spectrum_id": spectrum_label,
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "error": str(exc),
                }
            )

    def record_r2_rejection(
        candidate_id: int,
        r2: float,
        threshold: float,
        sources: List[str],
    ) -> None:
        path_label = file_path or "unknown"
        spectrum_label = spectrum_id if spectrum_id is not None else "unknown"
        logger.info(
            "Peak rejected by r2 path=%s spectrum_id=%s candidate_id=%s r2=%.4f threshold=%.4f sources=%s",
            path_label,
            spectrum_label,
            candidate_id,
            r2,
            threshold,
            sources,
        )

    spectrum_timed_out = False
    for polarity in (1, -1):
        if _spectrum_timed_out():
            spectrum_timed_out = True
            break
        group = [c for c in candidates if int(c["polarity"]) == polarity]
        if not group:
            continue
        fit_y = -y_proc if polarity < 0 else y_proc
        fit_y_abs = None
        if y_abs is not None and polarity > 0:
            fit_y_abs = np.asarray(y_abs, dtype=float)
        clusters = _cluster_candidates_by_window(group, window=fit_window)
        for cluster in clusters:
            if _spectrum_timed_out():
                spectrum_timed_out = True
                break
            has_plateau = any(c.get("plateau_bounds") is not None for c in cluster)
            cluster_indices = [int(c["index"]) for c in cluster]
            span_start = max(min(cluster_indices) - fit_window, 0)
            span_end = min(max(cluster_indices) + fit_window, len(x))
            xs_cluster = x[span_start:span_end]
            ys_cluster = fit_y[span_start:span_end]
            candidate_centers = [float(x[int(c["index"])]) for c in cluster]
            cluster_span_cm = 0.0
            if candidate_centers:
                cluster_span_cm = abs(max(candidate_centers) - min(candidate_centers))
            allow_multi_fit = (
                len(cluster) > 1
                and model.lower() in {"gaussian", "lorentzian", "voigt"}
                and not has_plateau
                and (multi_fit_min_span_cm <= 0.0 or cluster_span_cm >= multi_fit_min_span_cm)
            )
            if allow_multi_fit:
                remaining = _remaining_spectrum_time()
                if remaining is not None and remaining <= 0:
                    spectrum_timed_out = True
                    break
                fit_timeout_value = (
                    min(fit_timeout_sec, max(0.0, remaining)) if remaining is not None else fit_timeout_sec
                )
                curvature_centers = _curvature_seed_centers(
                    xs_cluster,
                    ys_cluster,
                    distance_pts=max(1, int(fit_window / max(len(cluster), 1))),
                    max_peaks=max(len(cluster), 2),
                )
                step = _resolve_step_from_x(xs_cluster)
                extra_centers = [
                    c
                    for c in curvature_centers
                    if all(abs(c - base) > step * 0.75 for base in candidate_centers)
                ]
                centers = candidate_centers + extra_centers
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        fitted = _fit_multi_peak(
                            xs_cluster,
                            ys_cluster,
                            centers,
                            model,
                            maxfev=fit_maxfev,
                            fit_timeout_sec=fit_timeout_value,
                        )
                except (RuntimeError, OptimizeWarning, ValueError, FitTimeoutError) as exc:
                    for candidate in cluster:
                        record_fit_failure(int(candidate["candidate_id"]), model, exc)
                    fitted = None
                if fitted is not None:
                    fitted_peaks, _ = fitted
                    for candidate, fit_result in zip(cluster, fitted_peaks[: len(candidate_centers)]):
                        if not fit_result:
                            logger.info(
                                "Peak candidate dropped path=%s spectrum_id=%s candidate_id=%s "
                                "index=%s reason=fit_missing model=%s",
                                file_path or "unknown",
                                spectrum_id if spectrum_id is not None else "unknown",
                                candidate.get("candidate_id", "unknown"),
                                candidate.get("index", "unknown"),
                                model,
                            )
                            mark_candidate_processed()
                            continue
                        is_shoulder = _shoulder_candidate(candidate)
                        candidate_sources = list(candidate.get("sources", []))
                        fit_r2 = float(fit_result.get("r2", 0.0))
                        fit_amplitude = abs(float(fit_result.get("amplitude", 0.0)))
                        if is_shoulder:
                            if fit_amplitude <= min_absorbance_threshold:
                                continue
                        else:
                            if fit_r2 < min_r2:
                                record_r2_rejection(
                                    int(candidate["candidate_id"]),
                                    fit_r2,
                                    min_r2,
                                    candidate_sources,
                                )
                                mark_candidate_processed()
                                continue
                        result = dict(fit_result)
                        result["candidate_id"] = int(candidate["candidate_id"])
                        result["index"] = int(candidate["index"])
                        result["polarity"] = int(polarity)
                        result["sources"] = candidate_sources
                        if polarity < 0:
                            result["amplitude"] = float(result.get("amplitude", 0.0)) * -1
                            result["area"] = float(result.get("area", 0.0)) * -1
                        results.append(result)
                        mark_candidate_processed()
                else:
                    mark_candidate_processed(len(cluster))
                    continue
            for candidate in cluster:
                if _spectrum_timed_out():
                    spectrum_timed_out = True
                    break
                idx = int(candidate["index"])
                plateau_bounds = candidate.get("plateau_bounds")
                x0_guess = float(x[idx])
                center_bounds = None
                use_absorbance = fit_y_abs is not None and plateau_bounds is not None
                is_shoulder = _shoulder_candidate(candidate)
                candidate_fit_window = shoulder_fit_window if is_shoulder else fit_window
                candidate_min_r2 = shoulder_min_r2 if is_shoulder else min_r2
                if plateau_bounds is not None:
                    left_idx, right_idx = plateau_bounds
                    left_x = float(x[left_idx])
                    right_x = float(x[right_idx])
                    center_bounds = (min(left_x, right_x), max(left_x, right_x))
                    x0_guess = (center_bounds[0] + center_bounds[1]) / 2.0
                fit_source = fit_y_abs if use_absorbance else fit_y
                fit = None
                fit_timed_out = False
                try:
                    remaining = _remaining_spectrum_time()
                    if remaining is not None and remaining <= 0:
                        spectrum_timed_out = True
                        break
                    fit_timeout_value = (
                        min(fit_timeout_sec, max(0.0, remaining)) if remaining is not None else fit_timeout_sec
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        fit = fit_peak(
                            x,
                            fit_source,
                            idx,
                            model,
                            candidate_fit_window,
                            center_bounds=center_bounds,
                            x0_guess=x0_guess,
                            maxfev=fit_maxfev,
                            fit_timeout_sec=fit_timeout_value,
                            raise_on_error=True,
                        )
                except (RuntimeError, OptimizeWarning, ValueError, FitTimeoutError) as exc:
                    if isinstance(exc, FitTimeoutError):
                        fit_timed_out = True
                    record_fit_failure(int(candidate["candidate_id"]), model, exc)
                if not fit:
                    if plateau_bounds is None:
                        if not fit_timed_out:
                            logger.info(
                                "Peak candidate dropped path=%s spectrum_id=%s candidate_id=%s "
                                "index=%s reason=fit_failed model=%s",
                                file_path or "unknown",
                                spectrum_id if spectrum_id is not None else "unknown",
                                candidate.get("candidate_id", "unknown"),
                                candidate.get("index", "unknown"),
                                model,
                            )
                        mark_candidate_processed()
                        continue
                    xs, ys = _fit_peak_window(x, fit_source, idx, candidate_fit_window)
                    if xs.size < 3:
                        mark_candidate_processed()
                        continue
                    local_idx = int(np.argmin(np.abs(xs - float(x[idx]))))
                    baseline = float(np.median(ys))
                    peak_y = float(ys[local_idx])
                    plateau_width = abs(float(x[plateau_bounds[1]]) - float(x[plateau_bounds[0]]))
                    fwhm = max(plateau_width, _resolve_step_from_x(xs) * 2.0)
                    if hasattr(np, "trapezoid"):
                        area = float(np.trapezoid(np.maximum(ys - baseline, 0.0), xs))
                    else:
                        area = float(np.trapz(np.maximum(ys - baseline, 0.0), xs))
                    fit = dict(
                        center=float(x0_guess),
                        fwhm=float(fwhm),
                        amplitude=float(max(peak_y - baseline, 0.0)),
                        area=area,
                        r2=float(candidate_min_r2),
                    )
                if plateau_bounds is None:
                    fit_r2 = float(fit.get("r2", 0.0))
                    fit_amplitude = abs(float(fit.get("amplitude", 0.0)))
                    if is_shoulder:
                        if fit_amplitude <= min_absorbance_threshold:
                            continue
                    else:
                        if fit_r2 < candidate_min_r2:
                            record_r2_rejection(
                                int(candidate["candidate_id"]),
                                fit_r2,
                                candidate_min_r2,
                                list(candidate.get("sources", [])),
                            )
                            mark_candidate_processed()
                            continue
                result = dict(fit)
                result["candidate_id"] = int(candidate["candidate_id"])
                result["index"] = int(candidate["index"])
                result["polarity"] = int(polarity)
                result["sources"] = list(candidate.get("sources", []))
                if polarity < 0:
                    result["amplitude"] = float(result.get("amplitude", 0.0)) * -1
                    result["area"] = float(result.get("area", 0.0)) * -1
                results.append(result)
                mark_candidate_processed()
            if spectrum_timed_out:
                break
        if spectrum_timed_out:
            break

    if spectrum_timed_out:
        remaining_candidates = max(0, total_candidates - processed_candidates)
        spectrum_timeout_skips += remaining_candidates
        log_line(
            "Peak fit spectrum timeout "
            f"skipped={spectrum_timeout_skips} candidates "
            f"path={file_path or 'unknown'} spectrum_id={spectrum_id if spectrum_id is not None else 'unknown'}",
            stream=sys.stderr,
        )

    results.sort(key=lambda r: (int(r.get("index", 0)), int(r.get("candidate_id", 0))))
    if timeout_skips:
        log_line(
            "Peak fit timeouts "
            f"skipped={timeout_skips} candidates "
            f"path={file_path or 'unknown'} spectrum_id={spectrum_id if spectrum_id is not None else 'unknown'}"
        )
    return results, timeout_skips + spectrum_timeout_skips


_PEAK_RETRY_RELAXATIONS = (
    {
        "prominence_factor": 0.6,
        "noise_sigma_multiplier_factor": 0.8,
        "min_absorbance_factor": 0.7,
        "min_distance_factor": 0.9,
        "fit_window_factor": 1.2,
        "max_peak_candidates_factor": 2,
        "max_peak_candidates_floor": 50,
    },
    {
        "prominence_factor": 0.4,
        "noise_sigma_multiplier_factor": 0.6,
        "min_absorbance_factor": 0.5,
        "min_distance_factor": 0.8,
        "fit_window_factor": 1.4,
        "max_peak_candidates_factor": 3,
        "max_peak_candidates_floor": 100,
    },
)


def _args_to_dict(args: object) -> Dict[str, object]:
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _apply_peak_retry_params(
    base_params: Dict[str, object],
    attempt: int,
) -> Dict[str, object]:
    params = dict(base_params)
    if attempt <= 0:
        return params
    relax = _PEAK_RETRY_RELAXATIONS[attempt - 1]

    def _scale(name: str, factor: float, default: float, minimum: float = 0.0) -> None:
        value = float(params.get(name, default))
        params[name] = max(value * factor, minimum)

    _scale("prominence", relax["prominence_factor"], 0.003, minimum=np.finfo(float).eps)
    _scale("noise_sigma_multiplier", relax["noise_sigma_multiplier_factor"], 1.5, minimum=0.1)
    _scale("min_absorbance_threshold", relax["min_absorbance_factor"], 0.05, minimum=0.0)
    _scale("min_distance", relax["min_distance_factor"], 0.8, minimum=0.0)
    fit_window = int(params.get("fit_window_pts", 70) or 70)
    params["fit_window_pts"] = max(3, int(math.ceil(fit_window * relax["fit_window_factor"])))

    base_max_candidates = int(base_params.get("max_peak_candidates", 0) or 0)
    if base_max_candidates <= 0:
        params["max_peak_candidates"] = int(relax["max_peak_candidates_floor"])
    else:
        params["max_peak_candidates"] = max(
            base_max_candidates,
            int(math.ceil(base_max_candidates * relax["max_peak_candidates_factor"])),
        )
    return params


def detect_and_refine_with_retries(
    x: np.ndarray,
    y_proc: np.ndarray,
    noise_sigma: float,
    args: object,
    *,
    y_abs: np.ndarray | None = None,
    resolution_cm: float | None = None,
    file_path: str | None = None,
    spectrum_id: int | None = None,
    fit_errors: Optional[List[Dict[str, object]]] = None,
    progress_callback: Optional[callable] = None,
    min_peaks: int = 2,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int, int]:
    base_params = _args_to_dict(args)
    attempts = 1 + len(_PEAK_RETRY_RELAXATIONS)
    peak_candidates: List[Dict[str, object]] = []
    refined: List[Dict[str, object]] = []
    skipped_due_to_timeout = 0
    attempt_used = 0
    for attempt in range(attempts):
        params = _apply_peak_retry_params(base_params, attempt)
        if attempt > 0:
            logger.info(
                "Retrying peak detection path=%s spectrum_id=%s attempt=%d prominence=%.4g "
                "noise_sigma_multiplier=%.3g min_absorbance_threshold=%.4g min_distance=%.4g "
                "fit_window_pts=%s max_peak_candidates=%s",
                file_path or "unknown",
                spectrum_id if spectrum_id is not None else "unknown",
                attempt,
                float(params.get("prominence", 0.0) or 0.0),
                float(params.get("noise_sigma_multiplier", 0.0) or 0.0),
                float(params.get("min_absorbance_threshold", 0.0) or 0.0),
                float(params.get("min_distance", 0.0) or 0.0),
                params.get("fit_window_pts"),
                params.get("max_peak_candidates"),
            )
        peak_candidates = detect_peak_candidates(
            x,
            y_proc,
            noise_sigma,
            params,
            resolution_cm=resolution_cm,
            file_path=file_path,
            spectrum_id=spectrum_id,
        )
        refined, skipped_due_to_timeout = refine_peak_candidates(
            x,
            y_proc,
            peak_candidates,
            params,
            y_abs=y_abs,
            file_path=file_path,
            spectrum_id=spectrum_id,
            fit_errors=fit_errors,
            progress_callback=progress_callback,
        )
        attempt_used = attempt + 1
        if len(refined) >= min_peaks:
            break
    return peak_candidates, refined, skipped_due_to_timeout, attempt_used


def _interp_intensity(x: np.ndarray, y: np.ndarray, center: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_mask):
        return float("nan")
    x_valid = x[finite_mask]
    y_valid = y[finite_mask]
    if x_valid[0] > x_valid[-1]:
        x_valid = x_valid[::-1]
        y_valid = y_valid[::-1]
    return float(np.interp(center, x_valid, y_valid, left=np.nan, right=np.nan))


def build_peak_features(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict[str, object]],
    *,
    axis_key: str = "wavelength",
    max_peaks: int = 0,
    min_height: float | None = None,
) -> List[Dict[str, object]]:
    features: List[Dict[str, object]] = []
    for peak in peaks:
        center = peak.get("center")
        if center is None:
            continue
        try:
            center_val = float(center)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(center_val):
            continue
        intensity = _interp_intensity(x, y, center_val)
        if min_height is not None and np.isfinite(intensity) and intensity < min_height:
            continue
        feature = {
            axis_key: center_val,
            "intensity": float(intensity),
            "center": center_val,
            "index": int(peak.get("index", -1)) if peak.get("index") is not None else None,
            "fwhm": float(peak.get("fwhm", 0.0)) if peak.get("fwhm") is not None else None,
            "area": float(peak.get("area", 0.0)) if peak.get("area") is not None else None,
            "amplitude": float(peak.get("amplitude", 0.0)) if peak.get("amplitude") is not None else None,
            "r2": float(peak.get("r2", 0.0)) if peak.get("r2") is not None else None,
            "polarity": int(peak.get("polarity", 1)) if peak.get("polarity") is not None else None,
            "sources": peak.get("sources"),
        }
        features.append(feature)
    if max_peaks and len(features) > max_peaks:
        features = sorted(
            features,
            key=lambda row: abs(row.get("intensity", 0.0)) if row.get("intensity") is not None else 0.0,
            reverse=True,
        )[:max_peaks]
    return features


def detect_peaks_for_features(
    x: np.ndarray,
    y: np.ndarray,
    peak_cfg: Optional[Dict[str, object]] = None,
    *,
    axis_key: str = "wavelength",
    resolution_cm: float | None = None,
) -> List[Dict[str, object]]:
    config = resolve_peak_config(peak_cfg)
    if not bool(config.get("enabled", True)):
        return []
    if x.size < 3 or y.size < 3:
        return []
    y_proc, noise_sigma = preprocess_with_noise(
        x,
        y,
        int(config.get("sg_win", 7) or 7),
        int(config.get("sg_poly", 3) or 3),
        float(config.get("als_lam", 0.0) or 0.0),
        float(config.get("als_p", 0.01) or 0.01),
        sg_window_cm=float(config.get("sg_window_cm", 0.0) or 0.0),
        baseline_method=str(config.get("baseline_method", "rubberband") or "rubberband"),
        baseline_niter=int(config.get("baseline_niter", 20) or 20),
        baseline_piecewise=bool(config.get("baseline_piecewise", False)),
        baseline_ranges=config.get("baseline_ranges"),
    )
    _, refined, _, _ = detect_and_refine_with_retries(
        x,
        y_proc,
        noise_sigma,
        config,
        y_abs=y,
        resolution_cm=resolution_cm,
        min_peaks=1,
    )
    return build_peak_features(
        x,
        y,
        refined,
        axis_key=axis_key,
        max_peaks=int(config.get("max_peaks", 0) or 0),
        min_height=float(config["height"]) if config.get("height") is not None else None,
    )

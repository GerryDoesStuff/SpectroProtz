"""UV-Vis preprocessing helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

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


def coerce_domain(spec: Spectrum, domain: Dict[str, float] | None) -> Spectrum:
    """Ensure wavelength axis is sorted and optionally interpolated."""

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

    if domain:
        start = float(domain.get("min", wl.min()))
        stop = float(domain.get("max", wl.max()))
        if start >= stop:
            raise ValueError("Domain minimum must be smaller than maximum")

        if "step" in domain:
            step = float(domain["step"])
            if step <= 0:
                raise ValueError("Domain step must be positive")
            count = int(np.floor((stop - start) / step + 0.5)) + 1
            axis = start + np.arange(count) * step
            if axis[-1] > stop + 1e-9:
                axis = axis[axis <= stop + 1e-9]
            if axis.size < 2:
                axis = np.linspace(start, stop, 2)
        else:
            num = int(domain.get("num", wl.size))
            if num < 2:
                raise ValueError("Domain must request at least two points")
            axis = np.linspace(start, stop, num)

        inten = np.interp(axis, wl, inten, left=np.nan, right=np.nan)
        wl = axis

    return Spectrum(wavelength=wl, intensity=inten, meta=dict(spec.meta))


def subtract_blank(sample: Spectrum, blank: Spectrum) -> Spectrum:
    """Subtract a blank from the sample after verifying domain overlap."""

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

    meta = dict(sample.meta)
    meta["blank_subtracted"] = True
    return Spectrum(wavelength=sample.wavelength.copy(), intensity=corrected, meta=meta)


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
    meta = dict(spec.meta)
    meta.setdefault("baseline", method)
    return Spectrum(wavelength=spec.wavelength.copy(), intensity=corrected, meta=meta)


def detect_joins(
    wavelength: Sequence[float],
    intensity: Sequence[float],
    *,
    threshold: float | None = None,
    window: int = 5,
) -> List[int]:
    """Detect detector joins based on large absolute first differences."""

    y = np.asarray(intensity, dtype=float)
    diffs = np.abs(np.diff(y))
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return []

    if threshold is None:
        median = np.median(finite)
        if median <= 0:
            threshold = np.max(finite) * 0.5
        else:
            threshold = median * 10
    else:
        threshold = float(threshold)

    if not np.isfinite(threshold) or threshold <= 0:
        return []

    candidate_indices = [idx + 1 for idx, diff in enumerate(diffs) if diff >= threshold]
    if not candidate_indices:
        return []

    filtered: List[int] = []
    min_spacing = max(1, int(window))
    for idx in candidate_indices:
        if filtered and idx - filtered[-1] < min_spacing:
            continue
        filtered.append(idx)
    return filtered


def correct_joins(spec: Spectrum, join_indices: Iterable[int], *, window: int = 10) -> Spectrum:
    """Apply offset corrections after joins using neighbouring medians."""

    y = np.asarray(spec.intensity, dtype=float).copy()
    n = y.size
    window = max(1, int(window))

    for join in sorted(set(int(idx) for idx in join_indices)):
        if join <= 0 or join >= n:
            continue
        left = y[max(0, join - window) : join]
        right = y[join : min(n, join + window)]
        if left.size == 0 or right.size == 0:
            continue
        offset = np.nanmedian(right) - np.nanmedian(left)
        if np.isfinite(offset):
            y[join:] -= offset

    meta = dict(spec.meta)
    if join_indices:
        meta.setdefault("join_corrected", True)
    return Spectrum(wavelength=spec.wavelength.copy(), intensity=y, meta=meta)


def despike_spectrum(
    spec: Spectrum,
    *,
    zscore: float = 5.0,
    window: int = 5,
) -> Spectrum:
    """Replace spikes using a moving median filter with robust detection."""

    y = np.asarray(spec.intensity, dtype=float)
    if y.size < 3:
        return Spectrum(wavelength=spec.wavelength.copy(), intensity=y.copy(), meta=dict(spec.meta))

    window = int(window)
    if window <= 1:
        window = 3
    if window % 2 == 0:
        window += 1
    if window > y.size:
        window = y.size if y.size % 2 else y.size - 1
    if window < 3:
        return Spectrum(wavelength=spec.wavelength.copy(), intensity=y.copy(), meta=dict(spec.meta))

    baseline = medfilt(y, kernel_size=window)
    residual = y - baseline
    mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
    if not np.isfinite(mad) or mad == 0:
        return Spectrum(wavelength=spec.wavelength.copy(), intensity=y.copy(), meta=dict(spec.meta))

    threshold = float(zscore) * 1.4826 * mad
    mask = np.abs(residual) > threshold
    corrected = y.copy()
    if np.any(mask):
        indices = np.arange(y.size)
        good = indices[~mask]
        if good.size >= 2:
            corrected[mask] = np.interp(indices[mask], good, corrected[~mask])
        else:
            corrected[mask] = baseline[mask]

    meta = dict(spec.meta)
    if np.any(mask):
        meta.setdefault("despiked", True)
    return Spectrum(wavelength=spec.wavelength.copy(), intensity=corrected, meta=meta)


def smooth_spectrum(spec: Spectrum, *, window: int, polyorder: int) -> Spectrum:
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

    smoothed = savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")
    meta = dict(spec.meta)
    meta.setdefault("smoothed", True)
    return Spectrum(wavelength=spec.wavelength.copy(), intensity=smoothed, meta=meta)


def replicate_key(spec: Spectrum) -> Tuple[str, str]:
    role = str(spec.meta.get("role", "sample"))
    if role == "blank":
        ident = spec.meta.get("blank_id") or spec.meta.get("sample_id") or spec.meta.get("channel")
    else:
        ident = spec.meta.get("sample_id") or spec.meta.get("channel") or spec.meta.get("id")
    if ident is None:
        ident = f"{role}_{id(spec)}"
    return role, str(ident)


def average_replicates(
    specs: Sequence[Spectrum],
    *,
    return_mapping: bool = False,
) -> Tuple[List[Spectrum], Dict[Tuple[str, str], Spectrum]] | List[Spectrum]:
    groups: "OrderedDict[Tuple[str, str], List[Spectrum]]" = OrderedDict()
    for spec in specs:
        key = replicate_key(spec)
        groups.setdefault(key, []).append(spec)

    outputs: List[Spectrum] = []
    mapping: Dict[Tuple[str, str], Spectrum] = {}

    for key, members in groups.items():
        if len(members) == 1:
            averaged = members[0]
        else:
            ref_wl = members[0].wavelength
            for other in members[1:]:
                if not np.array_equal(other.wavelength, ref_wl):
                    raise ValueError("Replicates must share identical domains for averaging")
            stack = np.vstack([np.asarray(m.intensity, dtype=float) for m in members])
            averaged_intensity = np.nanmean(stack, axis=0)
            meta = dict(members[0].meta)
            meta["replicate_count"] = len(members)
            meta["replicate_averaged"] = True
            meta["replicate_sources"] = [
                member.meta.get("source_file") or member.meta.get("channel") or member.meta.get("sample_id")
                for member in members
            ]
            averaged = Spectrum(wavelength=ref_wl.copy(), intensity=averaged_intensity, meta=meta)

        mapping[key] = averaged
        outputs.append(averaged)

    if return_mapping:
        return outputs, mapping
    return outputs


def blank_identifier(spec: Spectrum) -> str | None:
    role = spec.meta.get("role")
    if role != "blank":
        return None
    ident = spec.meta.get("blank_id") or spec.meta.get("sample_id") or spec.meta.get("channel")
    return str(ident) if ident is not None else None

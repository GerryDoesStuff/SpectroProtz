"""Shared interpolation helpers for processing pipelines."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from spectro_app.engine.plugin_api import Spectrum

__all__ = ["interpolate_series", "interpolate_spectrum"]


def _build_upsampled_grid(x: np.ndarray, factor: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size < 2 or factor <= 1:
        return x_arr.copy()
    segments = [
        np.linspace(x_arr[idx], x_arr[idx + 1], num=factor, endpoint=False, dtype=float)
        for idx in range(x_arr.size - 1)
    ]
    return np.concatenate([*segments, x_arr[-1:]])


def _akima_resample(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_new_arr = np.asarray(x_new, dtype=float)
    if x_new_arr.size == 0:
        return np.array([], dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return np.full_like(x_new_arr, y_arr[0] if y_arr.size else 0.0, dtype=float)
    diffs = np.diff(x_arr)
    ascending = np.all(diffs > 0)
    descending = np.all(diffs < 0)
    if descending:
        x_work = x_arr[::-1]
        y_work = y_arr[::-1]
        x_new_work = x_new_arr[::-1]
        interpolator = Akima1DInterpolator(x_work, y_work)
        return interpolator(x_new_work)[::-1]
    if ascending:
        interpolator = Akima1DInterpolator(x_arr, y_arr)
        return interpolator(x_new_arr)
    order = np.argsort(x_arr)
    interpolator = Akima1DInterpolator(x_arr[order], y_arr[order])
    return interpolator(x_new_arr)


def interpolate_series(
    x: Iterable[float] | np.ndarray,
    y: Iterable[float] | np.ndarray,
    *,
    method: str = "akima",
    factor: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ``y`` onto a denser grid derived from ``x``."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return x_arr.copy(), y_arr.copy()
    if method.lower() != "akima":
        raise ValueError(f"Unsupported interpolation method: {method}")
    x_new = _build_upsampled_grid(x_arr, factor)
    y_new = _akima_resample(x_arr, y_arr, x_new)
    return x_new, y_new


def interpolate_spectrum(
    spec: Spectrum,
    *,
    method: str = "akima",
    factor: int = 8,
) -> Spectrum:
    """Return a new spectrum interpolated onto a denser grid."""

    x_arr = np.asarray(spec.wavelength, dtype=float)
    y_arr = np.asarray(spec.intensity, dtype=float)
    x_new, y_new = interpolate_series(x_arr, y_arr, method=method, factor=factor)

    meta = dict(spec.meta or {})
    channels = dict(meta.get("channels") or {})
    if channels:
        for name, values in list(channels.items()):
            try:
                arr = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.shape != x_arr.shape:
                continue
            channels[name] = _akima_resample(x_arr, arr, x_new)
    channels["interpolated"] = np.asarray(y_new, dtype=float).copy()
    meta["channels"] = channels
    meta["interpolation"] = {"enabled": True, "method": method, "factor": int(factor)}

    return Spectrum(
        wavelength=np.asarray(x_new, dtype=float).copy(),
        intensity=np.asarray(y_new, dtype=float).copy(),
        meta=meta,
    )

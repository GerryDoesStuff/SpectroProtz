"""Helpers for harmonising UV-Vis intensity channels."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["normalise_mode", "convert_intensity_to_absorbance"]


def normalise_mode(value: object | None) -> Optional[str]:
    """Return a canonical representation of a UV-Vis acquisition mode."""

    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text.startswith("abs") or "absorb" in text:
        return "absorbance"
    if text.startswith("%t") or "trans" in text:
        return "transmittance"
    if text.startswith("%r") or "reflect" in text:
        return "reflectance"
    return text


def convert_intensity_to_absorbance(
    intensity: np.ndarray,
    mode: object | None,
) -> Tuple[np.ndarray, Optional[str], Optional[str], Optional[str]]:
    """Convert ``intensity`` to absorbance if ``mode`` warrants it.

    Parameters
    ----------
    intensity:
        Raw intensity values as recorded in the source export.
    mode:
        Acquisition mode hint extracted from metadata or headers.

    Returns
    -------
    converted, channel_key, updated_mode, original_mode
        ``converted`` contains absorbance data when a conversion occurred and
        the original data otherwise. ``channel_key`` is the metadata channel
        name that should receive the raw trace (``None`` when no conversion is
        necessary). ``updated_mode`` is the canonical mode for the converted
        data, while ``original_mode`` preserves the detected mode prior to any
        conversion.
    """

    arr = np.asarray(intensity, dtype=float)
    original_mode = normalise_mode(mode)
    updated_mode = original_mode
    channel_key: Optional[str] = None

    if original_mode in {"transmittance", "reflectance"}:
        finite = arr[np.isfinite(arr)]
        scale = 100.0 if finite.size and float(np.nanmax(np.abs(finite))) > 1.5 else 1.0
        fraction = np.divide(arr, scale, dtype=float)
        converted = np.full_like(fraction, np.nan, dtype=float)
        valid = fraction > 0
        converted[valid] = -np.log10(fraction[valid])
        channel_key = "raw_transmittance" if original_mode == "transmittance" else "raw_reflectance"
        updated_mode = "absorbance"
        return converted, channel_key, updated_mode, original_mode

    return arr, None, updated_mode, original_mode

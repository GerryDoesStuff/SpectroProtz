from datetime import datetime, timedelta

import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def test_uvvis_analyze_produces_qc_metrics():
    wl = np.linspace(400, 500, 21)
    intensity = np.linspace(0, 100, wl.size)
    intensity[:5] += np.array([0.0, 0.5, -0.3, 0.4, -0.2])
    intensity[10:] += 50.0
    intensity[12] += 75.0  # spike

    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": "S1", "role": "sample", "mode": "transmittance"},
    )

    recipe = {
        "qc": {
            "quiet_window": {"min": 400.0, "max": 420.0},
            "noise_rsd_limit": 1.0,
            "spike_limit": 0,
            "join_offset_limit": 100.0,
        },
        "join": {"enabled": True, "window": 2, "threshold": 20.0},
        "despike": {"window": 5, "zscore": 3.0},
        "smoothing": {"enabled": True, "window": 7, "polyorder": 2},
    }

    _, qc_rows = UvVisPlugin().analyze([spec], recipe)
    assert len(qc_rows) == 1
    row = qc_rows[0]

    assert row["saturation_flag"] is True
    assert row["spike_count"] >= 1
    assert row["join_count"] >= 1
    assert np.isfinite(row["noise_rsd"]) and row["noise_rsd"] >= 0
    assert "saturation" in row["flags"]
    assert "Spikes" in row["summary"]
    assert row["noise_window"] == (400.0, 420.0)


def _make_synthetic_spec(
    wl: np.ndarray,
    baseline: float,
    timestamp: datetime,
    sample_id: str,
) -> Spectrum:
    intensity = np.full_like(wl, baseline, dtype=float)
    return Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": sample_id, "role": "sample", "acquired_datetime": timestamp.isoformat()},
    )


def _drift_recipe() -> dict:
    return {
        "qc": {
            "drift": {
                "enabled": True,
                "window": {"min": 210.0, "max": 290.0},
                "max_slope_per_hour": 0.5,
                "max_delta": 0.6,
                "max_residual": 0.4,
            }
        }
    }


def test_uvvis_qc_drift_stable_batch_not_flagged():
    wl = np.linspace(200, 300, 51)
    start = datetime(2024, 1, 1, 8, 0)
    baselines = [0.5, 0.55, 0.48, 0.52]
    specs = [
        _make_synthetic_spec(wl, base, start + timedelta(minutes=idx * 15), f"S{idx}")
        for idx, base in enumerate(baselines)
    ]
    _, qc_rows = UvVisPlugin().analyze(specs, _drift_recipe())

    assert qc_rows
    assert all(row["drift_available"] for row in qc_rows)
    assert all(not row["drift_flag"] for row in qc_rows)
    assert all("drift" not in row["flags"] for row in qc_rows)


def test_uvvis_qc_drift_flags_increasing_trend():
    wl = np.linspace(200, 300, 51)
    start = datetime(2024, 1, 1, 9, 0)
    baselines = [0.5, 0.9, 1.3, 1.7]
    specs = [
        _make_synthetic_spec(wl, base, start + timedelta(minutes=idx * 15), f"D{idx}")
        for idx, base in enumerate(baselines)
    ]
    _, qc_rows = UvVisPlugin().analyze(specs, _drift_recipe())

    assert qc_rows
    assert any(row["drift_flag"] for row in qc_rows)
    assert any("drift" in row["flags"] for row in qc_rows)
    flagged = [row for row in qc_rows if row["drift_flag"]]
    assert flagged
    for row in flagged:
        assert "slope" in row["drift_reasons"]
        assert row["drift_slope_per_hour"] is not None
        assert np.isfinite(row["drift_slope_per_hour"])
    assert qc_rows[-1]["drift_span_minutes"] == pytest.approx(45.0)

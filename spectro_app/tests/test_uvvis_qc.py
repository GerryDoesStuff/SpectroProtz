from datetime import datetime, timedelta

import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis import pipeline
from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def _mean_abs_first_diff(values: np.ndarray) -> float:
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
    assert "roughness" in row
    assert np.isfinite(row["roughness"]["processed"])
    assert row["roughness"]["channels"] == {}
    assert "roughness_delta" in row
    assert row["roughness_delta"] == {}


def test_uvvis_qc_reports_roughness_for_stage_channels():
    wl = np.linspace(200.0, 260.0, 121)
    raw_intensity = np.sin(wl / 9.0) + 0.25 * np.sin(wl * 0.9)
    spec = Spectrum(
        wavelength=wl,
        intensity=raw_intensity,
        meta={"sample_id": "rough", "role": "sample"},
    )
    coerced = pipeline.coerce_domain(spec, None)
    smoothed = pipeline.smooth_spectrum(coerced, window=9, polyorder=3)

    _, qc_rows = UvVisPlugin().analyze([smoothed], {})
    assert len(qc_rows) == 1
    row = qc_rows[0]

    processed_expected = _mean_abs_first_diff(smoothed.intensity)
    raw_expected = _mean_abs_first_diff(smoothed.meta["channels"]["raw"])
    smoothed_expected = _mean_abs_first_diff(smoothed.meta["channels"]["smoothed"])

    roughness = row["roughness"]
    assert roughness["processed"] == pytest.approx(processed_expected)
    assert roughness["channels"]["raw"] == pytest.approx(raw_expected)
    assert roughness["channels"]["smoothed"] == pytest.approx(smoothed_expected)
    assert roughness["channels"]["smoothed"] <= roughness["channels"]["raw"]

    delta = row["roughness_delta"]
    assert delta["raw"] == pytest.approx(raw_expected - processed_expected)
    assert delta["smoothed"] == pytest.approx(smoothed_expected - processed_expected, abs=1e-9)
    assert delta["raw"] > 0.0


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


def test_uvvis_isosbestic_checks_capture_crossing():
    wl = np.linspace(240.0, 260.0, 201)
    line = np.linspace(0.2, 0.3, wl.size)
    modulation = 0.01 * np.sin(((wl - wl.min()) / (wl.max() - wl.min())) * 2.0 * np.pi)
    intensity = line + modulation
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": "iso", "role": "sample"},
    )
    recipe = {
        "features": {
            "isosbestic": [
                {"name": "Iso_245_255", "wavelengths": [245.0, 255.0], "tolerance": 5e-3}
            ]
        }
    }

    processed, qc_rows = UvVisPlugin().analyze([spec], recipe)

    assert qc_rows[0]["isosbestic"]
    check = qc_rows[0]["isosbestic"][0]
    assert check["name"] == "Iso_245_255"
    assert check["crossing_detected"] is True
    assert check["within_tolerance"] is True
    assert check["min_abs_deviation"] == pytest.approx(0.0, abs=5e-4)
    assert 245.0 <= check["crossing_wavelength"] <= 255.0

    feature_checks = processed[0].meta["features"]["isosbestic"]
    assert feature_checks == qc_rows[0]["isosbestic"]


def test_uvvis_negative_intensity_flagging():
    wl = np.linspace(400.0, 410.0, 6)
    positive = Spectrum(
        wavelength=wl,
        intensity=np.linspace(0.1, 0.2, wl.size),
        meta={"sample_id": "pos", "role": "sample"},
    )

    negative_trace = np.linspace(0.1, 0.2, wl.size)
    negative_trace[2] = -0.05
    negative = Spectrum(
        wavelength=wl,
        intensity=negative_trace,
        meta={
            "sample_id": "neg",
            "role": "sample",
            "channels": {"raw": negative_trace.copy(), "processed": negative_trace.copy()},
        },
    )

    recipe = {"qc": {"negative_intensity_tolerance": 0.0}}

    _, qc_rows = UvVisPlugin().analyze([positive, negative], recipe)

    by_id = {row["sample_id"]: row for row in qc_rows}
    pos_row = by_id["pos"]
    neg_row = by_id["neg"]

    assert pos_row["negative_intensity_flag"] is False
    assert pos_row["negative_intensity_fraction"] == pytest.approx(0.0)
    assert "negative_intensity" not in pos_row["flags"]

    assert neg_row["negative_intensity_flag"] is True
    assert neg_row["negative_intensity_fraction"] == pytest.approx(1 / wl.size)
    assert "negative_intensity" in neg_row["flags"]
    channels = neg_row["negative_intensity_channels"]
    assert channels["raw"]["count"] >= 1
    assert channels["raw"]["fraction"] == pytest.approx(1 / wl.size)


def test_uvvis_kinetics_summary_tracks_delta_a():
    wl = np.linspace(200.0, 260.0, 121)
    start = datetime(2024, 1, 1, 10, 0)
    minutes = [0.0, 5.0, 10.0]
    base_curve = np.linspace(0.1, 0.15, wl.size)

    specs = []
    for offset in minutes:
        intensity = base_curve + 0.005 * offset
        specs.append(
            Spectrum(
                wavelength=wl,
                intensity=intensity,
                meta={
                    "sample_id": "kinetic",
                    "role": "sample",
                    "acquired_datetime": (start + timedelta(minutes=offset)).isoformat(),
                },
            )
        )

    recipe = {
        "features": {
            "kinetics": {
                "targets": [
                    {"name": "A250", "wavelength": 250.0, "bandwidth": 2.0},
                ]
            }
        }
    }

    processed, qc_rows = UvVisPlugin().analyze(specs, recipe)

    assert len(qc_rows) == 3
    relative_times = sorted(row["kinetics"]["relative_minutes"] for row in qc_rows)
    assert relative_times == pytest.approx(minutes)

    summaries = {row["sample_id"]: row["kinetics_summary"] for row in qc_rows}
    summary = summaries["kinetic"]
    assert summary["n_points"] == 3
    assert summary["elapsed_minutes"] == pytest.approx(10.0)
    target_metrics = summary["targets"]["A250"]
    assert target_metrics["delta"] == pytest.approx(0.05)
    assert target_metrics["slope_per_min"] == pytest.approx(0.005)
    assert target_metrics["points"] == 3
    assert len(summary["series"]) == 3

    feature_summary = processed[0].meta["features"]["kinetics"]["summary"]
    assert feature_summary == summary

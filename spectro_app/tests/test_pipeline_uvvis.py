from datetime import datetime, timedelta
from typing import Dict, Sequence

import math
import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.engine.run_controller import _flatten_recipe
from spectro_app.engine import pipeline
from spectro_app.plugins.uvvis.plugin import UvVisPlugin, disable_blank_requirement


def test_coerce_domain_interpolates_and_sorts():
    spec = Spectrum(
        wavelength=np.array([500, 400, 450], dtype=float),
        intensity=np.array([5.0, 1.0, 3.0], dtype=float),
        meta={},
    )
    domain = {"min": 400.0, "max": 500.0, "step": 25.0}
    coerced = pipeline.coerce_domain(spec, domain)
    expected_wl = np.array([400.0, 425.0, 450.0, 475.0, 500.0])
    sorted_wl = np.array([400.0, 450.0, 500.0])
    sorted_intensity = np.array([1.0, 3.0, 5.0])
    expected_intensity = np.interp(expected_wl, sorted_wl, sorted_intensity)
    assert np.allclose(coerced.wavelength, expected_wl)
    assert np.allclose(coerced.intensity, expected_intensity)
    domain_meta = coerced.meta.get("wavelength_domain_nm")
    assert domain_meta is not None
    assert domain_meta["original_min_nm"] == pytest.approx(400.0)
    assert domain_meta["original_max_nm"] == pytest.approx(500.0)
    assert domain_meta["requested_min_nm"] == pytest.approx(400.0)
    assert domain_meta["requested_max_nm"] == pytest.approx(500.0)
    assert domain_meta["output_min_nm"] == pytest.approx(400.0)
    assert domain_meta["output_max_nm"] == pytest.approx(500.0)
    channels = coerced.meta.get("channels")
    assert channels is not None
    assert np.allclose(channels["original_wavelength"], sorted_wl)
    assert np.allclose(channels["original_intensity"], sorted_intensity)
    assert np.allclose(channels["raw"], expected_intensity)


def test_coerce_domain_resample_num_preserves_original_grid_metadata():
    spec = Spectrum(
        wavelength=np.array([410.0, 430.0, 470.0, 450.0]),
        intensity=np.array([2.0, 4.0, 8.0, 6.0]),
        meta={"channels": {"some_existing": np.array([1, 2, 3])}},
    )
    domain = {"min": 410.0, "max": 470.0, "num": 7}
    coerced = pipeline.coerce_domain(spec, domain)
    assert coerced.wavelength.size == 7
    assert np.all(coerced.wavelength[:-1] < coerced.wavelength[1:])

    sorted_wl = np.array([410.0, 430.0, 450.0, 470.0])
    sorted_intensity = np.array([2.0, 4.0, 6.0, 8.0])
    channels = coerced.meta.get("channels")
    assert channels is not None
    assert "some_existing" in channels
    assert np.allclose(channels["original_wavelength"], sorted_wl)
    assert np.allclose(channels["original_intensity"], sorted_intensity)
    assert np.allclose(channels["raw"], coerced.intensity)


def test_coerce_domain_averages_duplicate_wavelengths_before_clipping():
    spec = Spectrum(
        wavelength=np.array([500.0, 400.0, 500.0, 450.0, 400.0]),
        intensity=np.array([10.0, 1.0, 30.0, 5.0, 5.0]),
        meta={},
    )
    domain = {"min": 395.0, "max": 505.0}
    coerced = pipeline.coerce_domain(spec, domain)

    assert np.allclose(coerced.wavelength, np.array([400.0, 450.0, 500.0]))
    assert np.allclose(coerced.intensity, np.array([3.0, 5.0, 20.0]))


def test_coerce_domain_interpolates_from_merged_duplicates():
    spec = Spectrum(
        wavelength=np.array([450.0, 450.0, 475.0, 400.0, 400.0]),
        intensity=np.array([2.0, 4.0, 6.0, 0.0, 1.0]),
        meta={},
    )
    domain = {"min": 400.0, "max": 475.0, "step": 25.0}
    coerced = pipeline.coerce_domain(spec, domain)

    expected_axis = np.array([400.0, 425.0, 450.0, 475.0])
    base_wl = np.array([400.0, 450.0, 475.0])
    base_intensity = np.array([0.5, 3.0, 6.0])
    expected_intensity = np.interp(expected_axis, base_wl, base_intensity)

    assert np.allclose(coerced.wavelength, expected_axis)
    assert np.allclose(coerced.intensity, expected_intensity)
    assert np.allclose(coerced.meta["channels"]["raw"], expected_intensity)


def test_normalise_exclusion_windows_accepts_multiple_forms():
    config = {
        "windows": [
            {"min_nm": 400, "max_nm": 410},
            {"lower": 420, "upper": 430},
            [440, 450],
        ]
    }
    result = pipeline.normalise_exclusion_windows(config)
    assert result == [
        {"lower_nm": 400.0, "upper_nm": 410.0},
        {"lower_nm": 420.0, "upper_nm": 430.0},
        {"lower_nm": 440.0, "upper_nm": 450.0},
    ]


def test_core_pipeline_accepts_wavenumber_windows():
    wn = np.linspace(1200.0, 800.0, 81)
    intensity = np.sin(np.linspace(0.0, 2.0, wn.size))
    spec = Spectrum(
        wavelength=wn,
        intensity=intensity,
        meta={"axis_key": "wavenumber", "axis_unit": "cm^-1", "role": "sample"},
    )
    recipe = {
        "despike": {
            "enabled": True,
            "window": 5,
            "exclusions": [{"lower_cm": 1050.0, "upper_cm": 1100.0}],
            "noise_scale_multiplier": 0.0,
        }
    }

    processed, qc_rows = pipeline.run_pipeline([spec], recipe)

    assert len(processed) == 1
    assert qc_rows
    assert processed[0].meta["axis_key"] == "wavenumber"


def test_subtract_blank_records_channel():
    wl = np.array([400.0, 410.0, 420.0])
    sample_intensity = np.array([1.5, 1.7, 2.0])
    sample = Spectrum(
        wavelength=wl,
        intensity=sample_intensity,
        meta={"channels": {"raw": sample_intensity.copy()}},
    )
    blank = Spectrum(
        wavelength=wl,
        intensity=np.array([0.5, 0.5, 0.5]),
        meta={},
    )

    corrected = pipeline.subtract_blank(sample, blank)
    channels = corrected.meta.get("channels") or {}
    assert np.allclose(channels["blanked"], corrected.intensity)
    assert np.allclose(channels["raw"], sample_intensity)


def test_apply_baseline_records_channel():
    wl = np.linspace(400, 460, 7)
    intensity = np.sin(wl / 40.0) + 0.1 * wl
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"channels": {"blanked": intensity.copy()}},
    )

    corrected = pipeline.apply_baseline(spec, "asls", lam=1e3, p=0.01, niter=10)
    channels = corrected.meta.get("channels") or {}
    assert "blanked" in channels
    assert np.allclose(channels["baseline_corrected"], corrected.intensity)


def test_apply_baseline_anchor_windows_zeroed():
    wl = np.linspace(400, 520, 25)
    baseline = 0.015 * (wl - 400)
    peak = np.exp(-0.5 * ((wl - 460) / 6.0) ** 2)
    intensity = baseline + peak + 0.2
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    anchor_cfg = {
        "enabled": True,
        "windows": [
            {"min_nm": 400.0, "max_nm": 420.0, "target": 0.0, "label": "low"},
            {"min_nm": 500.0, "max_nm": 520.0, "target": 0.05},
        ],
    }

    corrected = pipeline.apply_baseline(
        spec,
        "asls",
        lam=5e4,
        p=0.01,
        niter=15,
        anchor=anchor_cfg,
    )

    left_mask = (wl >= 400.0) & (wl <= 420.0)
    right_mask = (wl >= 500.0) & (wl <= 520.0)
    assert np.allclose(corrected.intensity[left_mask], 0.0, atol=1e-8)
    assert np.allclose(corrected.intensity[right_mask], 0.05, atol=1e-8)

    anchor_meta = corrected.meta.get("baseline_anchor")
    assert anchor_meta is not None
    assert anchor_meta.get("applied") is True
    windows_meta = anchor_meta.get("windows") or []
    assert len(windows_meta) == 2
    assert windows_meta[0]["label"] == "low"
    assert windows_meta[0]["points"] > 0
    assert windows_meta[1]["target_level"] == pytest.approx(0.05)


def test_correct_joins_records_channel():
    wl = np.linspace(400, 450, 6)
    intensity = np.array([0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"channels": {"blanked": intensity.copy()}},
    )

    corrected = pipeline.correct_joins(spec, [3], window=1)
    channels = corrected.meta.get("channels") or {}
    assert "blanked" in channels
    assert np.allclose(channels["joined"], corrected.intensity)


def test_stitch_regions_interpolates_window_and_audits_shoulders():
    wl = np.linspace(400.0, 410.0, 6)
    intensity = np.array([0.2, 0.3, np.nan, np.nan, 0.8, 0.9])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"sample_id": "S1"})

    with pytest.warns(RuntimeWarning):
        stitched = pipeline.stitch_regions(
            spec,
            windows=[{"lower_nm": 404.0, "upper_nm": 406.0}],
            shoulder_points=2,
            method="linear",
        )

    window_mask = (wl >= 404.0) & (wl <= 406.0)
    support_x = wl[[0, 1, 4, 5]]
    support_y = intensity[[0, 1, 4, 5]]
    coeff = np.polyfit(support_x, support_y, deg=1)
    expected_segment = np.polyval(coeff, wl[window_mask])

    assert np.allclose(stitched.intensity[window_mask], expected_segment, equal_nan=False)
    assert stitched.meta.get("stitched") is True

    channels = stitched.meta.get("channels") or {}
    assert "stitched" in channels
    assert np.allclose(channels["stitched"], stitched.intensity)

    regions = stitched.meta.get("stitched_regions") or []
    assert len(regions) == 1
    region = regions[0]
    assert region.get("applied") is True
    assert region.get("fallback_reason") is None
    assert region.get("shoulder_counts") == {"left": 2, "right": 2}
    assert region.get("method") == "linear"


def test_stitch_regions_records_fallback_when_shoulders_missing():
    wl = np.array([400.0, 402.0, 404.0, 406.0])
    intensity = np.array([0.2, 0.25, 0.7, 0.75])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"sample_id": "S1"})

    stitched = pipeline.stitch_regions(
        spec,
        windows=[{"lower_nm": 402.0, "upper_nm": 404.0, "method": "cubic", "shoulder_points": 1}],
        shoulder_points=0,
        method="linear",
    )

    assert np.allclose(stitched.intensity, intensity)

    channels = stitched.meta.get("channels") or {}
    assert "stitched" in channels
    assert np.allclose(channels["stitched"], intensity)

    regions = stitched.meta.get("stitched_regions") or []
    assert len(regions) == 1
    region = regions[0]
    assert region.get("applied") is False
    assert region.get("fallback_reason") == "insufficient shoulder samples for degree"
    assert region.get("shoulder_counts") == {"left": 1, "right": 1}


def test_smooth_spectrum_respects_join_segments():
    wl = np.linspace(400.0, 450.0, 11)
    step_intensity = np.concatenate([np.zeros(5), np.full(6, 5.0)])
    spec = Spectrum(wavelength=wl, intensity=step_intensity, meta={})

    blended = pipeline.smooth_spectrum(spec, window=5, polyorder=2)
    segmented = pipeline.smooth_spectrum(spec, window=5, polyorder=2, join_indices=[5])

    assert blended.intensity[4] > 0.0
    assert blended.intensity[5] < 5.0

    assert segmented.intensity[4] == pytest.approx(0.0)
    assert segmented.intensity[5] == pytest.approx(5.0)
    assert segmented.meta.get("smoothed_segmented") is True


def test_despike_removes_spikes_straddling_join_window():
    wl = np.linspace(200.0, 400.0, 120)
    baseline = 0.35 + 0.02 * np.sin(wl / 30.0)
    intensity = baseline.copy()
    join_idx = 60
    spike_indices = np.array([join_idx - 1, join_idx, join_idx + 1])
    intensity[spike_indices] += np.array([3.5, 4.5, 3.0])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=11,
        join_indices=[join_idx],
        noise_scale_multiplier=0.0,
    )

    assert np.allclose(despiked.intensity[spike_indices], baseline[spike_indices], atol=5e-2)
    mask = np.ones_like(wl, dtype=bool)
    mask[spike_indices] = False
    assert np.allclose(despiked.intensity[mask], baseline[mask], atol=3e-2)


def test_despike_respects_exclusion_windows():
    wl = np.linspace(400.0, 700.0, 301)
    baseline = 0.02 * np.sin(wl / 40.0)
    intensity = baseline.copy()
    excluded_idx = 150
    neighbor_idx = excluded_idx + 5
    intensity[excluded_idx] += 4.0
    intensity[neighbor_idx] += 3.5
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    exclusions = [
        {
            "lower_nm": float(wl[excluded_idx]) - 0.5,
            "upper_nm": float(wl[excluded_idx]) + 0.5,
        }
    ]

    corrected = pipeline.despike_spectrum(
        spec,
        window=7,
        zscore=2.5,
        noise_scale_multiplier=0.0,
        exclusion_windows=exclusions,
    )

    assert corrected.intensity[excluded_idx] == pytest.approx(intensity[excluded_idx])
    assert corrected.intensity[neighbor_idx] == pytest.approx(baseline[neighbor_idx], abs=5e-3)
    assert corrected.meta.get("despiked") is True


def test_despike_iteratively_handles_clustered_spikes():
    wl = np.linspace(300.0, 500.0, 80)
    baseline = np.linspace(0.1, 0.5, wl.size)
    intensity = baseline.copy()
    spike_idx = np.array([20, 21, 22, 23])
    intensity[spike_idx] += np.array([3.0, 6.0, 5.5, 4.0])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(spec, window=9, noise_scale_multiplier=0.0)

    assert np.allclose(despiked.intensity[spike_idx], baseline[spike_idx], atol=3e-2)
    mask = np.ones_like(wl, dtype=bool)
    mask[spike_idx] = False
    assert np.allclose(despiked.intensity[mask], baseline[mask], atol=2e-2)
    assert despiked.meta.get("despiked") is True


def test_despike_handles_tail_spike():
    wl = np.linspace(260.0, 340.0, 161)
    baseline = 0.15 + 0.0015 * (wl - wl.min())
    intensity = baseline.copy()
    spike_idx = 80
    intensity[spike_idx] += 2.5
    tail = 0.12 * np.exp(-np.arange(1, 7) / 1.5)
    intensity[spike_idx + 1 : spike_idx + 1 + tail.size] += tail
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(spec, window=7, noise_scale_multiplier=0.0)

    tail_slice = slice(spike_idx, spike_idx + 1 + tail.size)
    assert np.allclose(despiked.intensity[tail_slice], baseline[tail_slice], atol=2e-2)

    left_guard = slice(spike_idx - 4, spike_idx)
    right_guard = slice(spike_idx + 1 + tail.size, spike_idx + 1 + tail.size + 4)
    assert np.allclose(despiked.intensity[left_guard], baseline[left_guard], atol=5e-3)
    assert np.allclose(despiked.intensity[right_guard], baseline[right_guard], atol=5e-3)


def test_despike_single_spike_tracks_curved_baseline():
    wl = np.linspace(320.0, 720.0, 161)
    curvature = 0.25 + 0.06 * np.sin(wl / 35.0) + 0.0004 * (wl - 520.0) ** 2
    intensity = curvature.copy()
    spike_idx = 85
    intensity[spike_idx] += 0.75
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=9,
        baseline_window=13,
        spread_window=13,
        zscore=4.0,
        noise_scale_multiplier=0.0,
    )

    mask = np.ones_like(wl, dtype=bool)
    mask[spike_idx] = False
    edge_guard = np.zeros_like(wl, dtype=bool)
    edge_guard[:6] = True
    edge_guard[-6:] = True
    mask &= ~edge_guard
    assert np.allclose(despiked.intensity[mask], curvature[mask], atol=2e-2)
    assert np.isclose(despiked.intensity[spike_idx], curvature[spike_idx], atol=0.08)
    local_window = curvature[spike_idx - 4 : spike_idx + 5]
    assert local_window.min() <= despiked.intensity[spike_idx] <= local_window.max()
    changed = np.where(np.abs(despiked.intensity - curvature) > 1e-3)[0]
    assert spike_idx in changed


def test_despike_noise_injection_matches_baseline_statistics():
    wl = np.linspace(300.0, 500.0, 200)
    noise_sigma = 0.01
    rng = np.random.default_rng(1234)
    clean = rng.normal(0.0, noise_sigma, size=wl.size)

    intensity = clean.copy()
    spike_idx = np.arange(20, 180, 5)
    intensity[spike_idx] += 0.5

    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=11,
        zscore=3.5,
        noise_scale_multiplier=1.0,
        rng_seed=202405,
    )

    corrected = despiked.intensity
    spike_values = corrected[spike_idx]
    non_spike_mask = np.ones_like(wl, dtype=bool)
    non_spike_mask[spike_idx] = False

    assert np.allclose(
        despiked.intensity[non_spike_mask],
        clean[non_spike_mask],
        atol=3 * noise_sigma,
    )

    mean_tolerance = 3 * noise_sigma / np.sqrt(spike_idx.size)
    assert spike_values.mean() == pytest.approx(0.0, abs=mean_tolerance)

    baseline_variance = clean[spike_idx].var(ddof=1)
    sample_variance = spike_values.var(ddof=1)
    variance_ratio = sample_variance / baseline_variance if baseline_variance > 0 else 1.0
    assert 0.25 <= variance_ratio <= 4.0

    assert np.all(np.abs(spike_values) <= 6 * noise_sigma)


def test_despike_multipass_detects_nested_spikes():
    wl = np.linspace(280.0, 640.0, 181)
    baseline = 0.15 + 0.0008 * (wl - 360.0)
    intensity = baseline.copy()
    primary_idx, secondary_idx = 90, 91
    intensity[primary_idx] += 5.0
    intensity[secondary_idx] += 1.6
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=9,
        spread_method="std",
        zscore=2.5,
        max_passes=6,
        noise_scale_multiplier=0.0,
    )

    assert np.isclose(despiked.intensity[primary_idx], baseline[primary_idx], atol=5e-3)
    assert np.isclose(despiked.intensity[secondary_idx], baseline[secondary_idx], atol=5e-3)
    untouched = np.ones_like(wl, dtype=bool)
    untouched[[primary_idx, secondary_idx]] = False
    assert np.allclose(despiked.intensity[untouched], baseline[untouched], atol=1e-8)


def test_despike_leading_padding_preserves_prefix():
    wl = np.linspace(240.0, 420.0, 181)
    baseline = np.exp(0.02 * (wl - wl.min()))
    intensity = baseline.copy()
    spike_idx = np.array([40, 90, 130])
    intensity[spike_idx] += np.array([4.0, 5.0, 3.0])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked_no_pad = pipeline.despike_spectrum(spec, window=9, noise_scale_multiplier=0.0)
    despiked_with_pad = pipeline.despike_spectrum(
        spec,
        window=9,
        leading_padding=12,
        noise_scale_multiplier=0.0,
    )

    assert np.allclose(despiked_with_pad.intensity[:12], baseline[:12])
    assert np.allclose(
        despiked_with_pad.intensity[spike_idx],
        baseline[spike_idx],
        rtol=2.5e-2,
        atol=0.0,
    )
    assert np.allclose(
        despiked_with_pad.intensity[12:],
        despiked_no_pad.intensity[12:],
        atol=0.15,
    )
    assert np.allclose(
        despiked_no_pad.intensity,
        pipeline.despike_spectrum(spec, window=9, leading_padding=0).intensity,
    )


def test_despike_padding_only_applies_at_global_edges():
    wl = np.linspace(200.0, 260.0, 30)
    baseline = 0.2 + 0.01 * (wl - wl[0])
    intensity = baseline.copy()
    interior_spikes = {11: baseline[11] + 3.0, 18: baseline[18] - 2.5}
    for idx, value in interior_spikes.items():
        intensity[idx] = value

    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=7,
        join_indices=[10, 20],
        leading_padding=2,
        trailing_padding=2,
        noise_scale_multiplier=0.0,
    )

    assert np.allclose(despiked.intensity[:2], baseline[:2])
    assert np.allclose(despiked.intensity[-2:], baseline[-2:])
    assert np.isclose(despiked.intensity[11], baseline[11], atol=5e-2)
    assert np.isclose(despiked.intensity[18], baseline[18], atol=5e-2)
    assert despiked.meta.get("despiked") is True


def test_despike_regression_join_metadata_does_not_block_spikes():
    wl = np.linspace(220.0, 420.0, 161)
    left = 0.25 + 0.0025 * (wl - wl.min())
    right = 0.45 + 0.0018 * (wl - 320.0)
    baseline = np.where(wl < 320.0, left, right)
    intensity = baseline.copy()
    spike_indices = np.array([45, 80, 120])
    intensity[spike_indices] += np.array([1.6, -1.8, 2.1])
    joins = [30, 60, 90, 130]
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(
        spec,
        window=9,
        baseline_window=11,
        spread_window=11,
        join_indices=joins,
        noise_scale_multiplier=0.0,
    )

    assert np.allclose(despiked.intensity[spike_indices], baseline[spike_indices], atol=1.5e-2)
    preserved = np.ones_like(wl, dtype=bool)
    guard = np.zeros_like(wl, dtype=bool)
    for idx in spike_indices:
        preserved[idx] = False
        guard[idx - 4 : idx + 5] = True
    guard[:4] = True
    guard[-4:] = True
    preserved &= ~guard
    assert np.allclose(despiked.intensity[preserved], baseline[preserved], atol=1e-2)
    assert despiked.meta.get("despiked") is True


def test_despike_handles_isolated_spike_with_zero_mad():
    wl = np.linspace(400.0, 420.0, 11)
    baseline = np.zeros_like(wl)
    intensity = baseline.copy()
    intensity[5] = 5.0
    intensity[2] = 0.03
    intensity[8] = -0.025
    original = intensity.copy()
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    despiked = pipeline.despike_spectrum(spec, window=5, noise_scale_multiplier=0.0)

    assert despiked.meta.get("despiked") is True
    assert abs(despiked.intensity[5]) <= 1e-6
    assert abs(despiked.intensity[5] - original[5]) >= 4.0
    assert abs(despiked.intensity[2]) <= abs(original[2]) + 1e-9
    assert abs(despiked.intensity[8]) <= abs(original[8]) + 1e-9


def test_preprocess_join_correction_emits_channel_without_detected_joins(monkeypatch):
    plugin = UvVisPlugin()
    wl = np.linspace(400, 420, 11)
    intensity = np.linspace(0.2, 0.8, wl.size)
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"channels": {"raw": intensity.copy()}},
    )

    called: Dict[str, bool] = {"detect": False}

    def fake_detect_joins(wavelength, intensity, *, threshold=None, window=5, windows=None):
        called["detect"] = True
        return []

    monkeypatch.setattr(pipeline, "detect_joins", fake_detect_joins)

    recipe = {
        "join": {"enabled": True, "window": 5},
        "blank": {"subtract": False, "require": False},
    }
    processed = plugin.preprocess([spec], recipe)

    assert called["detect"] is True
    assert processed and len(processed) == 1
    result = processed[0]
    channels = result.meta.get("channels") or {}
    assert "joined" in channels
    assert np.allclose(channels["joined"], result.intensity)
    stats = result.meta.get("join_statistics")
    assert stats is not None
    assert stats.get("indices") == []


def test_preprocess_stitching_runs_before_join_detection(monkeypatch):
    plugin = UvVisPlugin()
    wl = np.linspace(400, 420, 5)
    base_intensity = np.linspace(0.0, 1.0, wl.size)
    spec = Spectrum(
        wavelength=wl,
        intensity=base_intensity.copy(),
        meta={"role": "sample"},
    )

    stitched_intensity = base_intensity + 10.0
    stitched_regions = [
        {
            "lower_nm": 405.0,
            "upper_nm": 415.0,
            "applied": True,
            "shoulder_count": 5,
        }
    ]

    recorded: Dict[str, object] = {}

    def fake_stitch_regions(
        input_spec: Spectrum,
        windows,
        *,
        shoulder_points=None,
        method=None,
    ) -> Spectrum:
        meta = dict(input_spec.meta or {})
        channels = dict(meta.get("channels") or {})
        channels["stitched"] = stitched_intensity.copy()
        meta["channels"] = channels
        meta["stitched"] = True
        meta["stitched_regions"] = stitched_regions
        return Spectrum(
            wavelength=np.asarray(input_spec.wavelength, dtype=float).copy(),
            intensity=stitched_intensity.copy(),
            meta=meta,
        )

    def fake_detect_joins(wavelength, intensity, **kwargs):
        recorded["detect_intensity"] = np.asarray(intensity, dtype=float).copy()
        recorded["detect_kwargs"] = kwargs
        return [2]

    def fake_correct_joins(spec: Spectrum, indices, **kwargs) -> Spectrum:
        recorded["pre_correction_meta"] = dict(spec.meta)
        channels = dict(spec.meta.get("channels") or {})
        channels["joined"] = np.asarray(spec.intensity, dtype=float).copy()
        spec.meta["channels"] = channels
        return spec

    monkeypatch.setattr(pipeline, "stitch_regions", fake_stitch_regions)
    monkeypatch.setattr(pipeline, "detect_joins", fake_detect_joins)
    monkeypatch.setattr(pipeline, "correct_joins", fake_correct_joins)

    recipe = {
        "blank": {"subtract": False, "require": False},
        "stitch": {
            "enabled": True,
            "windows": [{"min_nm": 405.0, "max_nm": 415.0}],
        },
        "join": {"enabled": True, "window": 3},
    }

    processed = plugin.preprocess([spec], recipe)
    assert recorded["detect_intensity"].shape == stitched_intensity.shape
    assert np.allclose(recorded["detect_intensity"], stitched_intensity)

    pre_meta = recorded["pre_correction_meta"]
    assert pre_meta.get("stitch_method") == plugin.DEFAULT_STITCH_METHOD
    assert pre_meta.get("stitch_shoulders") == plugin.DEFAULT_STITCH_SHOULDER_POINTS
    assert pre_meta.get("stitch_fallback_policy") == plugin.DEFAULT_STITCH_FALLBACK_POLICY

    assert processed and len(processed) == 1
    result = processed[0]
    assert result.meta.get("join_indices") == (2,)
    channels = result.meta.get("channels") or {}
    assert np.allclose(channels.get("stitched"), stitched_intensity)
    assert np.allclose(channels.get("joined"), result.intensity)


def test_despike_records_channel():
    wl = np.linspace(400, 440, 5)
    intensity = np.array([1.0, 1.05, 5.0, 1.1, 1.15])
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"channels": {"joined": intensity.copy()}},
    )

    corrected = pipeline.despike_spectrum(spec, zscore=2.5, window=3)
    channels = corrected.meta.get("channels") or {}
    assert "joined" in channels
    assert np.allclose(channels["despiked"], corrected.intensity)


def test_smooth_records_channel():
    wl = np.linspace(400, 460, 7)
    intensity = np.array([1.0, 1.2, 1.1, 1.3, 1.25, 1.27, 1.26])
    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"channels": {"baseline_corrected": intensity.copy()}},
    )

    smoothed = pipeline.smooth_spectrum(spec, window=5, polyorder=2)
    channels = smoothed.meta.get("channels") or {}
    assert "baseline_corrected" in channels
    assert np.allclose(channels["smoothed"], smoothed.intensity)


def test_subtract_blank_validates_overlap_and_subtracts():
    sample = Spectrum(
        wavelength=np.array([400.0, 450.0, 500.0]),
        intensity=np.array([2.0, 3.0, 4.0]),
        meta={"role": "sample", "blank_id": "B1"},
    )
    blank = Spectrum(
        wavelength=np.array([390.0, 450.0, 510.0]),
        intensity=np.array([1.0, 1.5, 1.0]),
        meta={"role": "blank", "blank_id": "B1"},
    )
    corrected = pipeline.subtract_blank(sample, blank)
    expected = sample.intensity - np.interp(sample.wavelength, blank.wavelength, blank.intensity)
    assert np.allclose(corrected.intensity[1:], expected[1:])
    assert corrected.meta["blank_subtracted"] is True
    audit = corrected.meta.get("blank_audit")
    expected_interp = np.interp(sample.wavelength, blank.wavelength, blank.intensity)
    assert audit["overlap_points"] == sample.wavelength.size
    assert audit["overlap_fraction"] == pytest.approx(1.0)
    assert audit["blank_mean_intensity"] == pytest.approx(np.mean(expected_interp))

    far_blank = Spectrum(
        wavelength=np.array([600.0, 610.0]),
        intensity=np.array([0.1, 0.2]),
        meta={"role": "blank", "blank_id": "B2"},
    )
    with pytest.raises(ValueError):
        pipeline.subtract_blank(sample, far_blank)


def _build_simple_spectrum(role: str, wl: np.ndarray, value: float, meta: Dict[str, object]) -> Spectrum:
    return Spectrum(wavelength=wl, intensity=np.full_like(wl, value, dtype=float), meta={"role": role, **meta})


def test_preprocess_blank_pairing_records_audit_metadata():
    wl = np.linspace(400, 410, 3)
    blank_time = datetime(2024, 1, 1, 10, 0)
    sample_time = blank_time + timedelta(minutes=30)
    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.5,
        {"blank_id": "B1", "sample_id": "B1", "acquired_datetime": blank_time.isoformat(), "pathlength_cm": 1.0},
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {"sample_id": "S1", "blank_id": "B1", "acquired_datetime": sample_time.isoformat(), "pathlength_cm": 1.0},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": True, "max_time_delta_minutes": 120}}
    with pytest.warns(DeprecationWarning):
        processed = plugin.preprocess([blank, sample], recipe)
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")
    audit = processed_sample.meta.get("blank_audit")
    assert audit["timestamp_delta_minutes"] == pytest.approx(30.0)
    assert audit["pathlength_delta_cm"] == pytest.approx(0.0)
    assert audit["blank_id"] == "B1"
    assert audit["sample_id"] == "S1"
    assert audit["validation_ran"] is True
    assert audit["validation_enforced"] is True
    assert audit["validation_violations"] == []


def test_preprocess_blank_pairing_rejects_time_gap():
    wl = np.linspace(400, 410, 3)
    blank_time = datetime(2024, 1, 1, 8, 0)
    sample_time = blank_time + timedelta(hours=5)
    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.5,
        {"blank_id": "B1", "sample_id": "B1", "acquired_datetime": blank_time.isoformat(), "pathlength_cm": 1.0},
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {"sample_id": "S1", "blank_id": "B1", "acquired_datetime": sample_time.isoformat(), "pathlength_cm": 1.0},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": True, "max_time_delta_minutes": 60}}
    with pytest.warns(DeprecationWarning) as warning_info:
        processed = plugin.preprocess([blank, sample], recipe)

    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")
    audit = processed_sample.meta.get("blank_audit")

    assert processed_sample.meta.get("blank_subtracted") is True
    assert audit["validation_ran"] is True
    assert audit["validation_enforced"] is False
    assert audit["timestamp_delta_minutes"] == pytest.approx(300.0)
    violations = audit["validation_violations"]
    gap_violation = next(v for v in violations if v["type"] == "timestamp_gap")
    assert gap_violation["limit_minutes"] == pytest.approx(60.0)
    assert gap_violation["observed_minutes"] == pytest.approx(300.0)
    assert any("blank.max_time_delta_minutes" in str(w.message) for w in warning_info)


def test_preprocess_blank_pairing_rejects_pathlength_mismatch():
    wl = np.linspace(400, 410, 3)
    timestamp = datetime(2024, 1, 1, 9, 0).isoformat()
    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.5,
        {"blank_id": "B1", "sample_id": "B1", "acquired_datetime": timestamp, "pathlength_cm": 0.5},
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {"sample_id": "S1", "blank_id": "B1", "acquired_datetime": timestamp, "pathlength_cm": 1.0},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": True, "pathlength_tolerance_cm": 0.1}}
    with pytest.raises(ValueError):
        plugin.preprocess([blank, sample], recipe)


def test_preprocess_blank_pairing_opt_out_logs_violations():
    wl = np.linspace(400, 410, 3)
    blank_time = datetime(2024, 1, 1, 8, 0)
    sample_time = blank_time + timedelta(hours=5)
    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.5,
        {
            "blank_id": "B1",
            "sample_id": "B1",
            "acquired_datetime": blank_time.isoformat(),
            "pathlength_cm": 0.5,
        },
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {
            "sample_id": "S1",
            "blank_id": "B1",
            "acquired_datetime": sample_time.isoformat(),
            "pathlength_cm": 1.0,
        },
    )

    plugin = UvVisPlugin()
    recipe = {
        "blank": {
            "subtract": True,
            "max_time_delta_minutes": 60,
            "pathlength_tolerance_cm": 0.1,
            "validate_metadata": False,
        }
    }
    with pytest.warns(DeprecationWarning):
        processed = plugin.preprocess([blank, sample], recipe)
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")
    audit = processed_sample.meta.get("blank_audit")

    assert processed_sample.meta.get("blank_subtracted") is True
    assert audit["validation_ran"] is True
    assert audit["validation_enforced"] is False
    violations = audit["validation_violations"]
    assert {v["type"] for v in violations} == {"timestamp_gap", "pathlength_mismatch"}
    assert any(v["observed_minutes"] > 60 for v in violations if v["type"] == "timestamp_gap")
    assert any(v["observed_delta_cm"] > 0.1 for v in violations if v["type"] == "pathlength_mismatch")


def test_disable_blank_requirement_allows_missing_blank():
    wl = np.linspace(400, 410, 3)
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {
            "sample_id": "S1",
            "blank_id": "B_missing",
            "acquired_datetime": datetime(2024, 1, 1, 9, 0).isoformat(),
        },
    )

    plugin = UvVisPlugin()
    base_recipe: Dict[str, object] = {"blank": {"subtract": True}}
    recipe = disable_blank_requirement(base_recipe)

    processed = plugin.preprocess([sample], recipe)
    assert len(processed) == 1
    processed_sample = processed[0]

    assert processed_sample.meta.get("blank_subtracted") is False
    audit = processed_sample.meta.get("blank_audit")
    assert audit["blank_id"] == "B_missing"
    assert audit["validation_ran"] is False
    violations = audit["validation_violations"]
    assert any(v.get("type") == "blank_missing" for v in violations)


def test_ui_shaped_recipe_allows_missing_blank_without_error():
    wl = np.linspace(400, 410, 3)
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {
            "sample_id": "S1",
            "blank_id": "B_missing",
            "acquired_datetime": datetime(2024, 1, 1, 9, 0).isoformat(),
        },
    )

    plugin = UvVisPlugin()
    recipe = {"params": {"blank": {"subtract": True, "require": False}}}
    flattened = _flatten_recipe(recipe)
    assert "blank" in flattened and "params" not in flattened

    processed = plugin.preprocess([sample], flattened)
    assert len(processed) == 1
    processed_sample = processed[0]

    assert processed_sample.meta.get("blank_subtracted") is False
    audit = processed_sample.meta.get("blank_audit")
    assert audit["blank_id"] == "B_missing"
    assert audit["validation_ran"] is False
    violations = audit["validation_violations"]
    assert any(v.get("type") == "blank_missing" for v in violations)


def test_disable_blank_requirement_captures_audit_without_subtraction():
    wl = np.linspace(400, 410, 3)
    timestamp = datetime(2024, 1, 1, 9, 0)
    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.5,
        {
            "blank_id": "B1",
            "sample_id": "B1",
            "acquired_datetime": timestamp.isoformat(),
            "pathlength_cm": 1.0,
        },
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {
            "sample_id": "S1",
            "blank_id": "B1",
            "acquired_datetime": (timestamp + timedelta(minutes=30)).isoformat(),
            "pathlength_cm": 1.0,
        },
    )

    plugin = UvVisPlugin()
    base_recipe: Dict[str, object] = {"blank": {"subtract": False, "max_time_delta_minutes": 60}}
    recipe = disable_blank_requirement(base_recipe)

    with pytest.warns(DeprecationWarning):
        processed = plugin.preprocess([blank, sample], recipe)
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    assert processed_sample.meta.get("blank_subtracted") is False
    audit = processed_sample.meta.get("blank_audit")
    assert audit["blank_id"] == "B1"
    assert audit["validation_ran"] is True
    assert audit["validation_enforced"] is False
    assert audit["validation_violations"] == []
    assert audit["timestamp_delta_minutes"] == pytest.approx(30.0)


def test_baseline_methods_reduce_background():
    wl = np.linspace(400, 600, 201)
    baseline = 0.01 * (wl - 400)
    peak = np.exp(-0.5 * ((wl - 520) / 5) ** 2)
    spec = Spectrum(wavelength=wl, intensity=baseline + peak, meta={})

    asls = pipeline.apply_baseline(spec, "asls", lam=1e5, p=0.01, niter=10)
    rubber = pipeline.apply_baseline(spec, "rubberband")
    snip = pipeline.apply_baseline(spec, "snip", iterations=40)

    for corrected in (asls, rubber, snip):
        edges_mean = np.mean(corrected.intensity[:20])
        assert abs(edges_mean) < 0.1
        assert corrected.intensity.max() > 0.8


def test_despike_handles_step_change_without_join_segmentation():
    wl = np.linspace(400, 500, 11)
    baseline = np.concatenate([np.zeros(5), np.ones(6) * 10])
    spike_idx = 4
    intensity = baseline.copy()
    intensity[spike_idx] = 25.0
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    corrected = pipeline.despike_spectrum(
        spec,
        window=5,
        join_indices=[5],
        noise_scale_multiplier=0.0,
    )

    assert corrected.meta.get("despiked") is True
    assert corrected.intensity[spike_idx] == pytest.approx(baseline[spike_idx - 1]) or corrected.intensity[
        spike_idx
    ] == pytest.approx(baseline[spike_idx + 1])
    mask = np.ones_like(wl, dtype=bool)
    mask[spike_idx] = False
    assert np.allclose(corrected.intensity[mask], baseline[mask])


def test_preprocess_no_longer_threads_join_indices_to_despike(monkeypatch):
    wl = np.linspace(400, 500, 11)
    intensity = np.concatenate([np.zeros(5), np.ones(6) * 12])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"role": "sample"})

    captured: dict[str, object] = {}

    def fake_despike(spectrum: Spectrum, **kwargs) -> Spectrum:
        captured["join_indices"] = tuple(kwargs.get("join_indices") or [])
        captured["window"] = kwargs.get("window")
        captured["zscore"] = kwargs.get("zscore")
        return spectrum

    monkeypatch.setattr(pipeline, "despike_spectrum", fake_despike)

    plugin = UvVisPlugin()
    recipe = {
        "join": {"enabled": True, "window": 3},
        "despike": {"enabled": True},
        "blank": {"subtract": False},
    }
    plugin.preprocess([spec], recipe)

    expected_joins = tuple(pipeline.detect_joins(wl, intensity, window=3))
    assert expected_joins == (5,)
    assert captured.get("join_indices") == ()


def test_preprocess_defaults_limit_join_windows_for_helios():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 330.0] += 3.0
    intensity[wl >= 350.0] += 3.0

    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"role": "sample", "instrument": "Helios Gamma"},
    )

    plugin = UvVisPlugin()
    recipe = {
        "join": {"enabled": True, "threshold": 0.5, "window": 5},
        "blank": {"subtract": False},
    }
    processed = plugin.preprocess([spec], recipe)
    assert processed
    helios_result = processed[0]
    helios_indices = tuple(helios_result.meta.get("join_indices", ()))
    assert helios_indices
    helios_wavelengths = helios_result.wavelength[np.asarray(helios_indices, dtype=int)]
    assert all(340.0 <= float(val) <= 360.0 for val in helios_wavelengths)

    override_recipe = {
        "join": {
            "enabled": True,
            "threshold": 0.5,
            "window": 5,
            "windows": [{"min_nm": 320.0, "max_nm": 335.0}],
        },
        "blank": {"subtract": False},
    }
    override_processed = plugin.preprocess([spec], override_recipe)
    assert override_processed
    override_indices = tuple(override_processed[0].meta.get("join_indices", ()))
    assert override_indices
    override_wavelengths = override_processed[0].wavelength[np.asarray(override_indices, dtype=int)]
    assert all(320.0 <= float(val) <= 335.0 for val in override_wavelengths)


def test_join_detection_and_correction():
    wl = np.linspace(400, 700, 31)
    intensity = np.zeros_like(wl)
    intensity[16:] += 5.0
    joins = pipeline.detect_joins(wl, intensity)
    assert joins == [16]
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})
    corrected = pipeline.correct_joins(spec, joins, window=3)
    left_mean = np.mean(corrected.intensity[:15])
    right_mean = np.mean(corrected.intensity[16:])
    assert abs(left_mean - right_mean) < 1e-6


def test_detect_joins_handles_short_spike_pair_and_correction_restores_baseline():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    start = 85
    width = 3
    intensity[start : start + width] += 5.0

    joins = pipeline.detect_joins(wl, intensity, window=5)

    assert len(joins) == 2
    first, second = sorted(joins)
    assert first < second

    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})
    corrected = pipeline.correct_joins(spec, joins, window=5)

    plateau_after = corrected.intensity[start : start + width]
    assert np.allclose(plateau_after, 0.0, atol=1e-6)

    tail_region = corrected.intensity[second + 5 :]
    if tail_region.size:
        assert np.allclose(tail_region, 0.0, atol=1e-6)


def test_detect_joins_keeps_all_peaks_above_threshold():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 320.0] += 0.6
    intensity[wl >= 350.0] += 3.0

    joins = pipeline.detect_joins(wl, intensity, window=3, threshold=0.5)

    assert len(joins) == 2
    join_wavelengths = np.sort(wl[np.asarray(joins, dtype=int)])
    expected_wavelengths = np.array([320.0, 350.0])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=0.5)


def test_detect_joins_short_circuits_when_threshold_unmet(monkeypatch):
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.linspace(0.0, 1.0, wl.size)

    def fail_nanargmax(*args, **kwargs):
        raise AssertionError("nanargmax should not be called when threshold is unattainable")

    monkeypatch.setattr(pipeline.np, "nanargmax", fail_nanargmax)

    joins = pipeline.detect_joins(wl, intensity, window=5, threshold=10.0)

    assert joins == []


def test_detect_joins_respects_window_spike_limits():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 330.0] += 5.0
    intensity[wl >= 360.0] += 9.0

    limited = pipeline.detect_joins(
        wl,
        intensity,
        window=3,
        threshold=0.5,
        windows=[{"min_nm": 330.0, "max_nm": 365.0, "spikes": 1}],
    )

    assert limited == [120]

    expanded = pipeline.detect_joins(
        wl,
        intensity,
        window=3,
        threshold=0.5,
        windows=[{"min_nm": 330.0, "max_nm": 365.0, "spikes": 2}],
    )

    assert sorted(expanded) == [60, 120]


def test_detect_joins_auto_threshold_retains_multiple_join_candidates():
    wl = np.linspace(300.0, 500.0, 401)
    intensity = np.zeros_like(wl)

    first_join = 160
    second_join = 260
    ramp_width = 11
    ramp1 = np.linspace(0.0, 6.0, ramp_width)
    ramp1_start = first_join - ramp_width // 2
    intensity[ramp1_start : ramp1_start + ramp_width] += ramp1
    intensity[ramp1_start + ramp_width :] += 6.0
    intensity[second_join:] += 3.0

    joins = pipeline.detect_joins(wl, intensity, window=5)

    assert len(joins) == 2
    assert any(abs(idx - first_join) <= 3 for idx in joins)
    assert any(abs(idx - second_join) <= 1 for idx in joins)
    join_wavelengths = wl[np.asarray(sorted(joins), dtype=int)]
    expected_wavelengths = np.array([wl[first_join], wl[second_join]])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=3 * (wl[1] - wl[0]))


def test_detect_joins_auto_threshold_handles_windowed_mixed_magnitudes():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 320.0] += 0.6
    intensity[wl >= 350.0] += 3.0

    windows = [
        {"min_nm": 318.0, "max_nm": 322.0},
        {"min_nm": 350.5, "max_nm": 352.0},
    ]

    joins = pipeline.detect_joins(wl, intensity, window=5, windows=windows)

    assert len(joins) == 2
    join_wavelengths = np.sort(wl[np.asarray(joins, dtype=int)])
    expected_wavelengths = np.array([320.0, 350.0])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=0.5)


def test_detect_joins_auto_threshold_balances_overlap_magnitudes():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 332.0] += 6.0
    intensity[wl >= 362.0] += 1.5

    windows = [
        {"min_nm": 332.0, "max_nm": 332.0},
        {"min_nm": 362.0, "max_nm": 362.0},
    ]

    joins = pipeline.detect_joins(wl, intensity, window=5, windows=windows)

    assert len(joins) == 2
    join_wavelengths = np.sort(wl[np.asarray(joins, dtype=int)])
    expected_wavelengths = np.array([332.0, 362.0])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=0.5)


def test_detect_joins_window_magnitude_disparity_respects_threshold():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 330.0] += 6.0
    intensity[wl >= 360.0] += 0.4

    windows = [
        {"min_nm": 329.0, "max_nm": 331.0},
        {"min_nm": 359.0, "max_nm": 361.0},
    ]

    joins = pipeline.detect_joins(wl, intensity, window=5, threshold=0.2, windows=windows)

    assert len(joins) == 2
    join_wavelengths = np.sort(wl[np.asarray(joins, dtype=int)])
    expected_wavelengths = np.array([330.0, 360.0])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=0.5)


def test_detect_joins_visits_all_windows():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 320.0] += 0.6
    intensity[wl >= 350.0] += 3.0

    windows = [
        {"min_nm": 318.0, "max_nm": 322.0},
        {"min_nm": 348.0, "max_nm": 352.0},
    ]

    joins = pipeline.detect_joins(wl, intensity, window=3, threshold=0.5, windows=windows)

    assert len(joins) == 2
    join_wavelengths = np.sort(wl[np.asarray(joins, dtype=int)])
    expected_wavelengths = np.array([320.0, 350.0])
    assert np.allclose(join_wavelengths, expected_wavelengths, atol=0.5)


def test_detect_joins_respects_wavelength_windows():
    wl = np.linspace(300.0, 400.0, 201)
    intensity = np.zeros_like(wl)
    intensity[wl >= 330.0] += 3.0
    intensity[wl >= 350.0] += 3.0

    unrestricted = pipeline.detect_joins(wl, intensity, window=5, threshold=0.5)
    assert len(unrestricted) >= 2

    restricted = pipeline.detect_joins(
        wl,
        intensity,
        window=5,
        threshold=0.5,
        windows=[{"min_nm": 345.0, "max_nm": 355.0}],
    )
    assert restricted
    restricted_wavelengths = wl[np.asarray(restricted, dtype=int)]
    assert restricted_wavelengths.size == 1
    assert 345.0 <= float(restricted_wavelengths[0]) <= 355.0

    blocked = pipeline.detect_joins(
        wl,
        intensity,
        window=5,
        threshold=0.5,
        windows=[{"min_nm": 300.0, "max_nm": 320.0}],
    )
    assert blocked == []


def test_detect_joins_handles_gradual_spectra():
    wl = np.linspace(400, 600, 51)
    intensity = np.sin(wl / 40.0) * 0.01 + np.linspace(0, 0.2, wl.size)

    joins = pipeline.detect_joins(wl, intensity, window=5)

    assert joins == []


def test_detect_joins_change_point_localises_join():
    wl = np.linspace(350, 650, 61)
    baseline = 0.002 * (wl - 350)
    intensity = baseline.copy()
    intensity[30:] += 4.5

    joins = pipeline.detect_joins(wl, intensity, window=5)

    assert joins == [30]


def test_detect_joins_retains_strong_source_score_after_refinement():
    wl = np.linspace(400.0, 700.0, 61)
    intensity = np.zeros_like(wl)
    intensity[28:31] = 6.0
    intensity[32:] = 10.0

    joins = pipeline.detect_joins(wl, intensity, window=5, threshold=6.0)

    assert joins == [32]

    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})
    corrected = pipeline.correct_joins(spec, joins, window=5)

    original_tail = intensity[32:]
    corrected_tail = corrected.intensity[32:]
    assert not np.allclose(corrected_tail, original_tail)
    assert corrected_tail.mean() == pytest.approx(6.0)


def test_correct_joins_clamps_offsets_with_bounds():
    wl = np.linspace(400, 500, 21)
    intensity = np.concatenate([np.zeros(10), np.ones(11) * 10.0])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    bounded = pipeline.correct_joins(spec, [10], window=3, offset_bounds=(-1.0, 1.0))

    assert np.allclose(bounded.intensity[:10], 0.0, atol=1e-8)
    assert np.allclose(bounded.intensity[10:], 9.0, atol=1e-8)

    stats = bounded.meta.get("join_statistics") or {}
    assert stats.get("offsets") and stats.get("offsets")[0] == pytest.approx(1.0)
    assert stats.get("raw_offsets") and stats.get("raw_offsets")[0] == pytest.approx(10.0)
    assert stats.get("offset_bounds") == {"min": -1.0, "max": 1.0}


def test_despike_and_smooth_operations():
    wl = np.linspace(400, 500, 51)
    baseline = np.sin(wl / 20.0)
    intensity = baseline.copy()
    intensity[25] += 10
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})
    despiked = pipeline.despike_spectrum(spec, zscore=3.0, window=7)
    assert despiked.intensity[25] == pytest.approx(baseline[25], abs=1e-1)

    noisy = Spectrum(wavelength=wl, intensity=np.sin(wl / 18.0) + 0.2 * np.random.RandomState(0).normal(size=wl.size), meta={})
    smoothed = pipeline.smooth_spectrum(noisy, window=7, polyorder=3)
    assert smoothed.meta["smoothed"] is True


def test_average_replicates():
    wl = np.linspace(400, 500, 11)
    spec_a1 = Spectrum(wavelength=wl, intensity=np.ones_like(wl), meta={"sample_id": "A"})
    spec_a2 = Spectrum(wavelength=wl, intensity=2 * np.ones_like(wl), meta={"sample_id": "A"})
    spec_b = Spectrum(wavelength=wl, intensity=np.zeros_like(wl), meta={"sample_id": "B"})
    averaged, mapping = pipeline.average_replicates(
        [spec_a1, spec_a2, spec_b], return_mapping=True, outlier={"enabled": True}
    )
    assert len(averaged) == 2
    key_a = pipeline.replicate_key(spec_a1)
    assert np.allclose(mapping[key_a].intensity, np.ones_like(wl) * 1.5)
    meta = mapping[key_a].meta
    assert meta["replicate_count"] == 2
    assert meta["replicate_total"] == 2
    assert meta.get("replicate_excluded") == []
    assert meta.get("replicate_outlier_method") == "mad"
    assert meta.get("replicate_outlier_enabled") is True
    assert meta.get("replicate_outlier_threshold") == 3.5
    stored_scores = meta.get("replicate_outlier_scores")
    assert stored_scores and len(stored_scores) == 2
    assert all(score <= 3.5 for score in stored_scores)
    stored = meta.get("replicates")
    assert stored and len(stored) == 2
    assert all(not entry["excluded"] for entry in stored)
    assert np.allclose(stored[0]["intensity"], spec_a1.intensity)
    assert np.allclose(stored[1]["intensity"], spec_a2.intensity)
    assert all(entry.get("leverage") is None for entry in stored)


def test_average_replicates_channel_metadata_is_averaged():
    wl = np.linspace(450.0, 455.0, 6)
    raw_a = np.linspace(0.0, 1.0, wl.size)
    raw_b = np.linspace(0.5, 1.5, wl.size)
    processed_a = raw_a + 0.1
    processed_b = raw_b + 0.2
    channels_a = {
        "raw": raw_a.copy(),
        "smoothed": raw_a + 0.2,
        "reference_labels": ["a", "b", "c", "d", "e", "f"],
    }
    channels_b = {
        "raw": raw_b.copy(),
        "smoothed": raw_b + 0.4,
        "reference_labels": ["x", "y", "z", "u", "v", "w"],
    }
    spec_a1 = Spectrum(wavelength=wl, intensity=processed_a, meta={"sample_id": "A", "channels": channels_a})
    spec_a2 = Spectrum(wavelength=wl, intensity=processed_b, meta={"sample_id": "A", "channels": channels_b})

    averaged, mapping = pipeline.average_replicates([spec_a1, spec_a2], return_mapping=True)

    assert len(averaged) == 1
    averaged_spec = mapping[pipeline.replicate_key(spec_a1)]
    channels = averaged_spec.meta.get("channels") or {}

    expected_intensity = np.mean([processed_a, processed_b], axis=0)
    expected_raw = np.mean([raw_a, raw_b], axis=0)
    expected_smoothed = np.mean([channels_a["smoothed"], channels_b["smoothed"]], axis=0)

    assert np.allclose(averaged_spec.intensity, expected_intensity)
    assert np.allclose(channels.get("raw"), expected_raw)
    assert not np.allclose(expected_intensity, expected_raw)
    assert np.allclose(channels.get("smoothed"), expected_smoothed)
    assert channels.get("reference_labels") == channels_a["reference_labels"]


def test_average_replicates_preserves_stitch_channels_and_audits():
    wl = np.linspace(400.0, 420.0, 9)
    base = np.linspace(0.2, 0.8, wl.size)

    sample_meta = {"sample_id": "S1", "role": "sample"}
    stitched_success = pipeline.stitch_regions(
        Spectrum(wavelength=wl, intensity=base, meta=dict(sample_meta)),
        windows=[{"lower_nm": 406.0, "upper_nm": 410.0}],
        shoulder_points=2,
        method="linear",
    )

    stitched_failure = pipeline.stitch_regions(
        Spectrum(wavelength=wl, intensity=base, meta=dict(sample_meta)),
        windows=[{"lower_nm": 406.0, "upper_nm": 410.0, "method": "poly4", "shoulder_points": 0}],
        shoulder_points=0,
        method="linear",
    )

    window_mask = (wl >= 406.0) & (wl <= 410.0)
    assert np.allclose(stitched_failure.intensity[window_mask], base[window_mask])

    averaged, mapping = pipeline.average_replicates(
        [stitched_success, stitched_failure],
        return_mapping=True,
    )

    assert len(averaged) == 1
    key = pipeline.replicate_key(stitched_success)
    averaged_spec = mapping[key]

    channels = averaged_spec.meta.get("channels") or {}
    assert "stitched" in channels
    expected_stitched = np.nanmean(
        [
            stitched_success.meta["channels"]["stitched"],
            stitched_failure.meta["channels"]["stitched"],
        ],
        axis=0,
    )
    assert np.allclose(channels["stitched"], expected_stitched)

    regions = averaged_spec.meta.get("stitched_regions") or []
    assert len(regions) == 2
    assert any(entry.get("applied") for entry in regions)
    fallback_entries = [entry for entry in regions if not entry.get("applied")]
    assert fallback_entries
    assert fallback_entries[0].get("fallback_reason") in {
        "no shoulder samples",
        "insufficient shoulder samples for degree",
    }
    assert averaged_spec.meta.get("stitched") is True


def test_average_replicates_outlier_rejection():
    wl = np.linspace(400, 500, 11)
    baseline = np.sin(wl / 60.0)
    spec_a1 = Spectrum(wavelength=wl, intensity=baseline + 0.02, meta={"sample_id": "A", "channel": "rep1"})
    spec_a2 = Spectrum(wavelength=wl, intensity=baseline - 0.01, meta={"sample_id": "A", "channel": "rep2"})
    spec_outlier = Spectrum(
        wavelength=wl,
        intensity=baseline + 5.0,
        meta={"sample_id": "A", "channel": "rep3"},
    )

    averaged, mapping = pipeline.average_replicates(
        [spec_a1, spec_a2, spec_outlier],
        return_mapping=True,
        outlier={"enabled": True, "threshold": 3.5},
    )

    assert len(averaged) == 1
    averaged_spec = mapping[pipeline.replicate_key(spec_a1)]
    meta = averaged_spec.meta
    assert meta["replicate_count"] == 2
    assert meta["replicate_total"] == 3
    assert meta["replicate_excluded"] == ["rep3"]
    assert meta.get("replicate_outlier_method") == "mad"
    assert meta.get("replicate_outlier_enabled") is True
    assert meta.get("replicate_outlier_threshold") == 3.5
    assert meta.get("replicate_outlier_scores") and len(meta["replicate_outlier_scores"]) == 3
    stored = meta["replicates"]
    assert len(stored) == 3
    excluded_entries = [entry for entry in stored if entry["excluded"]]
    assert len(excluded_entries) == 1
    included_entries = [entry for entry in stored if not entry["excluded"]]
    assert len(included_entries) == 2
    assert np.allclose(averaged_spec.intensity, np.mean([spec_a1.intensity, spec_a2.intensity], axis=0))
    assert excluded_entries[0]["score"] > 3.5
    assert excluded_entries[0]["leverage"] is None


def test_average_replicates_cooksd_outlier():
    wl = np.linspace(400, 500, 41)
    baseline = 0.05 + 0.01 * np.sin(wl / 12.0)
    spec1 = Spectrum(
        wavelength=wl,
        intensity=baseline + 0.002 * np.cos(wl / 8.0),
        meta={"sample_id": "A", "channel": "rep1"},
    )
    spec2 = Spectrum(
        wavelength=wl,
        intensity=baseline - 0.002 * np.sin(wl / 9.0),
        meta={"sample_id": "A", "channel": "rep2"},
    )
    heavy = baseline + 0.12 * np.exp(-0.5 * ((wl - 455) / 4.0) ** 2) + 0.05 * (wl - wl.mean())
    spec_heavy = Spectrum(
        wavelength=wl,
        intensity=heavy,
        meta={"sample_id": "A", "channel": "rep3"},
    )
    spec4 = Spectrum(
        wavelength=wl,
        intensity=baseline + 0.001 * np.random.RandomState(1).normal(size=wl.size),
        meta={"sample_id": "A", "channel": "rep4"},
    )

    threshold = 150.0
    averaged, mapping = pipeline.average_replicates(
        [spec1, spec2, spec_heavy, spec4],
        return_mapping=True,
        outlier={"enabled": True, "method": "cooksd", "threshold": threshold},
    )

    assert len(averaged) == 1
    averaged_spec = mapping[pipeline.replicate_key(spec1)]
    meta = averaged_spec.meta
    assert meta["replicate_outlier_method"] == "cooksd"
    assert meta["replicate_outlier_enabled"] is True
    assert meta["replicate_outlier_threshold"] == threshold
    assert meta.get("replicate_outlier_leverage") and len(meta["replicate_outlier_leverage"]) == 4
    assert meta.get("replicate_outlier_scores") and len(meta["replicate_outlier_scores"]) == 4
    assert meta["replicate_excluded"] == ["rep3"]

    replicates_meta = meta["replicates"]
    assert len(replicates_meta) == 4
    heavy_entry = next(entry for entry in replicates_meta if entry["meta"]["channel"] == "rep3")
    assert heavy_entry["excluded"] is True
    heavy_score = heavy_entry["score"]
    assert math.isnan(heavy_score) or heavy_score > threshold
    assert heavy_entry["leverage"] and heavy_entry["leverage"] > 0.4
    retained = [entry for entry in replicates_meta if not entry["excluded"]]
    assert all(entry["score"] <= threshold for entry in retained)
    assert all((entry.get("leverage") or 0.0) < heavy_entry["leverage"] for entry in retained)


def test_plugin_preprocess_full_pipeline():
    wl = np.linspace(400, 700, 16)
    blank_intensity = 0.05 + 0.0005 * (wl - 400)
    blank = Spectrum(
        wavelength=wl,
        intensity=blank_intensity,
        meta={"role": "blank", "blank_id": "BLK", "sample_id": "BLK"},
    )

    peak = np.exp(-0.5 * ((wl - 550) / 15.0) ** 2)
    sample_base = blank_intensity + peak

    sample1_int = sample_base.copy()
    sample2_int = sample_base * 1.05 + 0.02
    join_mask = wl >= 550
    sample1_int[join_mask] += 0.5
    sample2_int[join_mask] += 0.5
    sample1_int[5] += 3
    sample2_int[8] -= 3

    sample1 = Spectrum(
        wavelength=wl,
        intensity=sample1_int,
        meta={"role": "sample", "sample_id": "S1", "blank_id": "BLK"},
    )
    sample2 = Spectrum(
        wavelength=wl[::-1],
        intensity=sample2_int[::-1],
        meta={"role": "sample", "sample_id": "S1", "blank_id": "BLK"},
    )

    plugin = UvVisPlugin()
    recipe = {
        "domain": {"min": 400.0, "max": 700.0, "step": 20.0},
        "join": {"enabled": True, "threshold": 0.2, "window": 2},
        "despike": {"enabled": True, "window": 5, "zscore": 3.0},
        "blank": {"subtract": True},
        "baseline": {"method": "asls", "lam": 1e4, "p": 0.01, "niter": 8},
        "smoothing": {"enabled": True, "window": 5, "polyorder": 2},
        "replicates": {"average": True},
    }

    processed = plugin.preprocess([blank, sample1, sample2], recipe)
    assert len(processed) == 2

    processed_blank = next(spec for spec in processed if spec.meta.get("role") == "blank")
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    assert processed_sample.meta.get("replicate_count") == 2
    assert processed_sample.meta.get("replicate_total") == 2
    replicates_meta = processed_sample.meta.get("replicates")
    assert replicates_meta and len(replicates_meta) == 2
    assert all(not entry.get("excluded") for entry in replicates_meta)
    assert np.nanmax(processed_sample.intensity) > 0.01


def test_replicate_averaging_corrects_mismatched_joins():
    wl = np.arange(400.0, 520.0, 6.0)
    base = np.linspace(0.05, 0.95, wl.size)

    sample1_int = base.copy()
    sample2_int = base.copy()
    join_a = 5
    join_b = 6
    sample1_int[join_a:] += 0.6
    sample2_int[join_b:] += 0.6

    sample1 = Spectrum(
        wavelength=wl,
        intensity=sample1_int,
        meta={"role": "sample", "sample_id": "S1", "channel": "rep1"},
    )
    sample2 = Spectrum(
        wavelength=wl,
        intensity=sample2_int,
        meta={"role": "sample", "sample_id": "S1", "channel": "rep2"},
    )

    recipe = disable_blank_requirement(
        {
            "join": {"enabled": True, "threshold": 0.1, "window": 3},
            "replicates": {"average": True},
        }
    )
    plugin = UvVisPlugin()
    processed = plugin.preprocess([sample1, sample2], recipe)

    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    join_indices = processed_sample.meta.get("join_indices")
    assert join_indices == (join_a, join_b)

    channels = processed_sample.meta.get("channels") or {}
    join_corrected = channels.get("join_corrected")
    assert join_corrected is not None
    assert np.allclose(join_corrected, processed_sample.intensity)

    join_stats = processed_sample.meta.get("join_statistics") or {}
    assert join_stats.get("indices") == [join_a, join_b]
    post_deltas = join_stats.get("post_deltas") or []
    assert all(abs(delta) < 0.05 for delta in post_deltas if np.isfinite(delta))

def test_blank_matching_strategy_default_blank_id():
    wl = np.linspace(400, 420, 5)
    blank = Spectrum(
        wavelength=wl,
        intensity=np.linspace(0.1, 0.2, wl.size),
        meta={"role": "blank", "blank_id": "BLK"},
    )
    sample = Spectrum(
        wavelength=wl,
        intensity=np.linspace(1.0, 1.4, wl.size),
        meta={"role": "sample", "sample_id": "S1", "blank_id": "BLK"},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": True, "require": True}}

    processed = plugin.preprocess([blank, sample], recipe)
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    assert processed_sample.meta.get("blank_subtracted") is True
    expected = sample.intensity - blank.intensity
    assert np.allclose(processed_sample.intensity, expected)


def test_blank_match_identifiers_include_indref_base_token():
    blank = Spectrum(
        wavelength=np.array([400.0, 405.0, 410.0]),
        intensity=np.array([0.1, 0.1, 0.1]),
        meta={"role": "blank", "blank_id": "Sample01_indRef"},
    )

    tokens = pipeline.blank_match_identifiers(blank)

    assert tokens == ["Sample01_indRef", "Sample01"]


def test_preprocess_blank_selection_supports_indref_suffix():
    wl = np.linspace(400, 410, 3)
    recipe = {"blank": {"subtract": True, "require": True}}

    blank = _build_simple_spectrum(
        "blank",
        wl,
        0.25,
        {"blank_id": "Sample01_indRef"},
    )
    sample = _build_simple_spectrum(
        "sample",
        wl,
        1.0,
        {"sample_id": "Sample01"},
    )

    plugin = UvVisPlugin()
    processed = plugin.preprocess([blank, sample], recipe)
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    assert processed_sample.meta.get("blank_subtracted") is True
    assert np.allclose(processed_sample.intensity, sample.intensity - blank.intensity)
    audit = processed_sample.meta.get("blank_audit") or {}
    assert audit.get("blank_id") == "Sample01_indRef"

    channel_blank = _build_simple_spectrum(
        "blank",
        wl,
        0.1,
        {"blank_id": "ChannelA_indRef"},
    )
    channel_sample = _build_simple_spectrum(
        "sample",
        wl,
        0.8,
        {"channel": "ChannelA"},
    )

    plugin_channel = UvVisPlugin()
    processed_channel = plugin_channel.preprocess([channel_blank, channel_sample], recipe)
    processed_channel_sample = next(
        spec for spec in processed_channel if spec.meta.get("role") != "blank"
    )

    assert processed_channel_sample.meta.get("blank_subtracted") is True
    assert np.allclose(
        processed_channel_sample.intensity,
        channel_sample.intensity - channel_blank.intensity,
    )
    channel_audit = processed_channel_sample.meta.get("blank_audit") or {}
    assert channel_audit.get("blank_id") == "ChannelA_indRef"


def test_blank_matching_strategy_cuvette_slot_pairs_by_slot():
    wl = np.linspace(400, 410, 4)
    blank_a = Spectrum(
        wavelength=wl,
        intensity=np.full(wl.size, 0.1),
        meta={"role": "blank", "blank_id": "BLK_A", "cuvette_slot": "SLOT1"},
    )
    blank_b = Spectrum(
        wavelength=wl,
        intensity=np.full(wl.size, 0.2),
        meta={"role": "blank", "blank_id": "BLK_B", "cuvette_slot": "SLOT1"},
    )
    sample = Spectrum(
        wavelength=wl,
        intensity=np.linspace(1.0, 1.3, wl.size),
        meta={
            "role": "sample",
            "sample_id": "S2",
            "blank_id": "UNRELATED",
            "cuvette_slot": "SLOT1",
        },
    )

    plugin = UvVisPlugin()
    recipe = {
        "blank": {"subtract": True, "require": True, "match_strategy": "cuvette_slot"},
        "replicates": {"average": True},
    }

    processed = plugin.preprocess([blank_a, blank_b, sample], recipe)

    processed_blank = next(spec for spec in processed if spec.meta.get("role") == "blank")
    processed_sample = next(spec for spec in processed if spec.meta.get("role") != "blank")

    assert processed_blank.meta.get("replicate_total") == 2
    assert processed_sample.meta.get("blank_subtracted") is True
    averaged_blank = np.mean([blank_a.intensity, blank_b.intensity], axis=0)
    expected = sample.intensity - averaged_blank
    assert np.allclose(processed_sample.intensity, expected)


def test_plugin_preprocess_threads_cooksd_configuration():
    wl = np.linspace(380, 520, 57)
    baseline = 0.04 + 0.008 * np.sin(wl / 15.0)
    sample1 = Spectrum(
        wavelength=wl,
        intensity=baseline + 0.001 * np.cos(wl / 7.0),
        meta={"role": "sample", "sample_id": "S2", "channel": "rep1"},
    )
    sample2 = Spectrum(
        wavelength=wl,
        intensity=baseline - 0.0015 * np.sin(wl / 9.0),
        meta={"role": "sample", "sample_id": "S2", "channel": "rep2"},
    )
    levered = baseline + 0.09 * np.exp(-0.5 * ((wl - 450) / 5.0) ** 2)
    sample3 = Spectrum(
        wavelength=wl,
        intensity=levered,
        meta={"role": "sample", "sample_id": "S2", "channel": "rep3"},
    )

    plugin = UvVisPlugin()
    recipe = {
        "replicates": {
            "average": True,
            "outlier": {"enabled": True, "method": "cooksd", "threshold": 150.0},
        },
        "blank": {"subtract": False},
    }

    processed = plugin.preprocess([sample1, sample2, sample3], recipe)
    assert len(processed) == 1
    processed_sample = processed[0]
    meta = processed_sample.meta
    assert meta["replicate_outlier_method"] == "cooksd"
    assert meta["replicate_outlier_enabled"] is True
    assert meta["replicate_outlier_threshold"] == 150.0
    assert meta["replicate_excluded"] == ["rep3"]
    replicates_meta = meta["replicates"]
    heavy_entry = next(entry for entry in replicates_meta if entry["meta"]["channel"] == "rep3")
    assert heavy_entry["excluded"] is True
    heavy_score = heavy_entry["score"]
    assert math.isnan(heavy_score) or heavy_score > 150.0
    assert heavy_entry["leverage"] and heavy_entry["leverage"] > 0.3


def test_preprocess_threads_baseline_anchor_configuration():
    wl = np.linspace(400.0, 600.0, 21)
    baseline = 0.01 * (wl - 400.0)
    intensity = baseline + 0.2
    sample = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"role": "sample", "sample_id": "S1"},
    )

    recipe = disable_blank_requirement(
        {
            "baseline": {
                "method": "asls",
                "lam": 2e4,
                "p": 0.01,
                "anchor": {
                    "windows": [
                        {"min_nm": 400.0, "max_nm": 420.0, "target": 0.0},
                        {"min_nm": 580.0, "max_nm": 600.0, "target": 0.02},
                    ]
                },
            }
        }
    )

    plugin = UvVisPlugin()
    processed = plugin.preprocess([sample], recipe)
    assert len(processed) == 1
    processed_sample = processed[0]

    left_mask = (wl >= 400.0) & (wl <= 420.0)
    right_mask = (wl >= 580.0) & (wl <= 600.0)
    assert np.allclose(processed_sample.intensity[left_mask], 0.0, atol=1e-8)
    assert np.allclose(processed_sample.intensity[right_mask], 0.02, atol=1e-8)

    anchor_meta = processed_sample.meta.get("baseline_anchor")
    assert anchor_meta is not None
    assert anchor_meta.get("applied") is True
    assert len(anchor_meta.get("windows") or []) == 2


def test_preprocess_guardrails_and_missing_blank():
    wl = np.linspace(400, 500, 5)
    sample = Spectrum(
        wavelength=wl,
        intensity=np.zeros_like(wl),
        meta={"role": "sample", "sample_id": "S1", "blank_id": "BLK"},
    )
    plugin = UvVisPlugin()

    bad_recipe = {"smoothing": {"enabled": True, "window": 4, "polyorder": 2}}
    with pytest.raises(ValueError):
        plugin.preprocess([sample], bad_recipe)

    recipe_missing_blank = {"blank": {"subtract": True}}
    with pytest.raises(ValueError):
        plugin.preprocess([sample], recipe_missing_blank)


def test_smooth_spectrum_respects_join_boundaries():
    wl = np.linspace(400, 500, 20)
    left = np.linspace(0.0, 1.0, 10)
    right = np.linspace(10.0, 11.0, 10)
    intensity = np.concatenate([left, right])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    global_smoothed = pipeline.smooth_spectrum(spec, window=5, polyorder=2)
    segmented_smoothed = pipeline.smooth_spectrum(
        spec, window=5, polyorder=2, join_indices=[10]
    )

    original_gap = intensity[10] - intensity[9]
    segmented_gap = segmented_smoothed.intensity[10] - segmented_smoothed.intensity[9]
    global_gap = global_smoothed.intensity[10] - global_smoothed.intensity[9]

    assert segmented_smoothed.meta.get("smoothed") is True
    assert segmented_smoothed.meta.get("smoothed_segmented") is True
    assert abs(segmented_gap - original_gap) < abs(global_gap - original_gap)
    assert abs(segmented_gap) > abs(global_gap)


def test_preprocess_smoothing_uses_join_segments():
    wl = np.linspace(400, 500, 20)
    left = np.linspace(0.0, 1.0, 10)
    right = np.linspace(0.0, 1.0, 10)
    right[0] = 5.0
    intensity = np.concatenate([left, right])
    sample = Spectrum(wavelength=wl, intensity=intensity, meta={"role": "sample", "sample_id": "S1"})

    plugin = UvVisPlugin()
    base_recipe = disable_blank_requirement({"join": {"enabled": True}})
    unsmoothed = plugin.preprocess([sample], base_recipe)
    assert len(unsmoothed) == 1
    unsmoothed_sample = unsmoothed[0]
    join_indices = unsmoothed_sample.meta.get("join_indices")

    expected = pipeline.smooth_spectrum(
        unsmoothed_sample, window=5, polyorder=2, join_indices=join_indices
    )

    recipe = disable_blank_requirement(
        {
            "join": {"enabled": True},
            "smoothing": {"enabled": True, "window": 5, "polyorder": 2},
        }
    )

    processed = plugin.preprocess([sample], recipe)
    assert len(processed) == 1
    processed_sample = processed[0]

    assert processed_sample.meta.get("smoothed_segmented") is True
    assert processed_sample.meta.get("join_indices") == tuple(join_indices)

    assert np.allclose(processed_sample.intensity, expected.intensity)


def test_preprocess_uses_default_join_detection():
    wl = np.linspace(300.0, 800.0, 101)
    baseline = 0.002 * (wl - wl.min())
    join_idx = 10
    intensity = baseline.copy()
    intensity[join_idx:] += 0.4
    sample = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"role": "sample", "instrument": "Helios Gamma"},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": False}}

    processed = plugin.preprocess([sample], recipe)
    assert len(processed) == 1
    processed_sample = processed[0]

    assert processed_sample.meta.get("join_indices") == (join_idx,)
    assert processed_sample.meta.get("join_corrected") is True

    stats = processed_sample.meta.get("join_statistics")
    assert isinstance(stats, dict)
    assert stats["indices"] == [join_idx]
    assert stats["offsets"][0] == pytest.approx(stats["pre_deltas"][0], rel=1e-6)
    assert stats["offsets"][0] > 0.3
    assert stats["post_deltas"][0] == pytest.approx(0.0, abs=1e-6)

    segments = processed_sample.meta.get("join_segments")
    assert isinstance(segments, dict)
    assert segments["indices"] == [join_idx]
    assert len(segments["raw"]) == 2
    assert len(segments["corrected"]) == 2

    channels = processed_sample.meta.get("channels")
    assert "join_raw" in channels
    assert "join_corrected" in channels

    original_delta = np.max(np.abs(intensity - baseline))
    corrected_delta = np.max(np.abs(processed_sample.intensity - baseline))
    assert original_delta > 0.3
    assert corrected_delta < 0.05


def test_preprocess_applies_default_domain_limits():
    wavelengths = np.array([150.0, 190.0, 250.0, 1090.0, 1120.0])
    intensities = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sample = Spectrum(
        wavelength=wavelengths,
        intensity=intensities,
        meta={"role": "sample", "instrument": "Helios Gamma"},
    )

    plugin = UvVisPlugin()
    recipe = {"blank": {"subtract": False}}

    processed = plugin.preprocess([sample], recipe)
    assert len(processed) == 1
    trimmed = processed[0]
    assert np.all(trimmed.wavelength >= 190.0)
    assert np.all(trimmed.wavelength <= 1100.0)
    assert trimmed.wavelength[0] == pytest.approx(190.0)
    assert trimmed.wavelength[-1] == pytest.approx(1090.0)
    domain_meta = trimmed.meta.get("wavelength_domain_nm")
    assert domain_meta["original_min_nm"] == pytest.approx(150.0)
    assert domain_meta["original_max_nm"] == pytest.approx(1120.0)
    assert domain_meta["requested_min_nm"] == pytest.approx(190.0)
    assert domain_meta["requested_max_nm"] == pytest.approx(1100.0)
    assert domain_meta["output_min_nm"] == pytest.approx(190.0)
    assert domain_meta["output_max_nm"] == pytest.approx(1090.0)


def test_compute_peak_metrics_detects_prominent_peak():
    wl = np.linspace(400.0, 420.0, 41)
    true_center = 410.35
    baseline = 2.0
    amplitude = 5.0
    intensity = baseline + amplitude * np.exp(-0.5 * ((wl - true_center) / 1.2) ** 2)

    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"role": "sample"})
    recipe = {"blank": {"subtract": False}, "features": {"peaks": {"max_peaks": 1, "prominence": 0.1}}}
    processed, _ = pipeline.run_pipeline([spec], recipe)
    metrics = processed[0].meta.get("features", {}).get("peaks")

    assert metrics and len(metrics) == 1
    peak = metrics[0]
    assert peak["wavelength"] == pytest.approx(true_center, abs=0.6)
    assert peak["intensity"] == pytest.approx(float(np.max(intensity)), rel=0, abs=0.1)


def test_analyze_skips_disabled_peaks():
    wl = np.linspace(400.0, 500.0, 201)
    peak_center = 450.0
    intensity = np.exp(-0.5 * ((wl - peak_center) / 2.5) ** 2)
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"role": "sample"})

    recipe_enabled = {"blank": {"subtract": False}, "features": {"peaks": {"enabled": True}}}
    processed_enabled, qc_enabled = pipeline.run_pipeline([spec], recipe_enabled)
    assert qc_enabled, "Expected QC rows when enabled"
    assert processed_enabled and processed_enabled[0].meta.get("features", {}).get("peaks")

    recipe_disabled = {"blank": {"subtract": False}, "features": {"peaks": {"enabled": False}}}
    processed_disabled, qc_disabled = pipeline.run_pipeline([spec], recipe_disabled)
    assert qc_disabled, "Expected QC rows when disabled"
    assert processed_disabled and processed_disabled[0].meta.get("features", {}).get("peaks") == []

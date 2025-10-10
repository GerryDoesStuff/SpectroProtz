from datetime import datetime, timedelta
from typing import Dict, Sequence

import math
import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.engine.run_controller import _flatten_recipe
from spectro_app.plugins.uvvis import pipeline
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


def test_despike_respects_join_boundaries():
    wl = np.linspace(400, 500, 11)
    baseline = np.concatenate([np.zeros(5), np.ones(6) * 10])
    spike_idx = 4
    intensity = baseline.copy()
    intensity[spike_idx] = 25.0
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})

    corrected = pipeline.despike_spectrum(spec, window=5, join_indices=[5])

    assert corrected.meta.get("despiked") is True
    assert corrected.intensity[spike_idx] == pytest.approx(baseline[spike_idx])
    assert corrected.intensity[spike_idx + 1] == pytest.approx(baseline[spike_idx + 1])
    assert corrected.intensity[spike_idx + 1] - corrected.intensity[spike_idx] == pytest.approx(
        baseline[spike_idx + 1] - baseline[spike_idx]
    )


def test_preprocess_threads_join_indices_to_despike(monkeypatch):
    wl = np.linspace(400, 500, 11)
    intensity = np.concatenate([np.zeros(5), np.ones(6) * 12])
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={"role": "sample"})

    captured: dict[str, object] = {}

    def fake_despike(
        spectrum: Spectrum,
        *,
        zscore: float,
        window: int,
        join_indices: Sequence[int] | None = None,
    ) -> Spectrum:
        captured["join_indices"] = tuple(join_indices or [])
        captured["window"] = window
        captured["zscore"] = zscore
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
    assert captured.get("join_indices") == expected_joins


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
    intensity = np.sin(wl / 20.0)
    intensity[25] += 10
    spec = Spectrum(wavelength=wl, intensity=intensity, meta={})
    despiked = pipeline.despike_spectrum(spec, zscore=3.0, window=7)
    assert abs(despiked.intensity[25] - intensity[25]) < 5

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
    assert processed_sample.meta.get("join_indices") == (10,)

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


def test_compute_peak_metrics_uses_quadratic_refinement():
    wl = np.linspace(400.0, 420.0, 41)
    true_center = 410.35
    baseline = 2.0
    amplitude = 5.0
    curvature = -0.05
    intensity = baseline + amplitude + curvature * (wl - true_center) ** 2

    plugin = UvVisPlugin()
    metrics = plugin._compute_peak_metrics(wl, intensity, {"max_peaks": 1, "prominence": 0.1})

    assert len(metrics) == 1
    peak = metrics[0]

    assert peak["quadratic_refined"] is True
    # The raw wavelength should remain on the grid whilst the refined value
    # approaches the true centre between samples.
    assert peak["raw_wavelength"] in wl
    assert peak["raw_wavelength"] != pytest.approx(true_center, abs=1e-6)
    assert peak["wavelength"] == pytest.approx(true_center, rel=0, abs=0.05)
    assert peak["height"] == pytest.approx(baseline + amplitude, rel=0, abs=0.05)

    assert peak["quadratic_half_height"] is not None
    coeffs = peak["quadratic_coefficients"]
    assert coeffs and len(coeffs) == 3
    assert peak["quadratic_window_indices"]
    assert peak["quadratic_window_wavelengths"]

    # The stored quadratic parameters should reproduce the measured FWHM.
    a = coeffs[0]
    delta_height = peak["height"] - peak["quadratic_half_height"]
    assert delta_height > 0
    expected_width = 2.0 * math.sqrt(delta_height / -a)
    assert peak["fwhm"] == pytest.approx(expected_width, rel=1e-6)

    # Quadratic refinement should adjust the result away from the discrete grid.
    assert abs(peak["wavelength"] - peak["raw_wavelength"]) > 0
    assert abs(peak["fwhm"] - peak["raw_fwhm"]) > 0

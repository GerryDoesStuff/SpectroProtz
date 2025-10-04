from datetime import datetime, timedelta
from typing import Dict, Sequence

import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
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
    with pytest.raises(ValueError):
        plugin.preprocess([blank, sample], recipe)


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
    assert mapping[key_a].meta["replicate_count"] == 2
    assert mapping[key_a].meta["replicate_total"] == 2
    assert mapping[key_a].meta.get("replicate_excluded") == []
    stored = mapping[key_a].meta.get("replicates")
    assert stored and len(stored) == 2
    assert all(not entry["excluded"] for entry in stored)
    assert np.allclose(stored[0]["intensity"], spec_a1.intensity)
    assert np.allclose(stored[1]["intensity"], spec_a2.intensity)


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
    assert averaged_spec.meta["replicate_count"] == 2
    assert averaged_spec.meta["replicate_total"] == 3
    assert averaged_spec.meta["replicate_excluded"] == ["rep3"]
    stored = averaged_spec.meta["replicates"]
    assert len(stored) == 3
    excluded_entries = [entry for entry in stored if entry["excluded"]]
    assert len(excluded_entries) == 1
    included_entries = [entry for entry in stored if not entry["excluded"]]
    assert len(included_entries) == 2
    assert np.allclose(averaged_spec.intensity, np.mean([spec_a1.intensity, spec_a2.intensity], axis=0))


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
    left_mean = np.nanmean(processed_sample.intensity[:5])
    right_mean = np.nanmean(processed_sample.intensity[-5:])
    assert abs(left_mean - right_mean) < 0.1
    assert processed_blank.meta.get("baseline") == "asls"


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

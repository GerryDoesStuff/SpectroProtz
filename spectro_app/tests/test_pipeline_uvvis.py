import numpy as np
import pytest

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis import pipeline
from spectro_app.plugins.uvvis.plugin import UvVisPlugin


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

    far_blank = Spectrum(
        wavelength=np.array([600.0, 610.0]),
        intensity=np.array([0.1, 0.2]),
        meta={"role": "blank", "blank_id": "B2"},
    )
    with pytest.raises(ValueError):
        pipeline.subtract_blank(sample, far_blank)


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
    averaged, mapping = pipeline.average_replicates([spec_a1, spec_a2, spec_b], return_mapping=True)
    assert len(averaged) == 2
    key_a = pipeline.replicate_key(spec_a1)
    assert np.allclose(mapping[key_a].intensity, np.ones_like(wl) * 1.5)
    assert mapping[key_a].meta["replicate_count"] == 2


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

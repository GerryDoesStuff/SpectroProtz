from types import SimpleNamespace

import numpy as np

from spectro_app.tests.jdx_index_builder_test_utils import install_duckdb_stub, load_index_builder


def _load_module(monkeypatch):
    install_duckdb_stub(monkeypatch)
    return load_index_builder()


def _build_args(**overrides):
    base = {
        "min_distance": 5.0,
        "prominence": 0.01,
        "noise_sigma_multiplier": 4.0,
        "peak_width_min": 0.0,
        "peak_width_max": 0.0,
        "cwt_enabled": False,
        "cwt_width_min": 0.0,
        "cwt_width_max": 0.0,
        "cwt_width_step": 0.0,
        "cwt_cluster_tolerance": 8.0,
        "merge_tolerance": 8.0,
        "detect_negative_peaks": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _peak_positions(candidates, x):
    return np.asarray([float(x[int(cand["index"])]) for cand in candidates], dtype=float)


def test_detects_broad_and_narrow_peaks(monkeypatch):
    mod = _load_module(monkeypatch)
    x = np.linspace(4000.0, 3600.0, 801)
    rng = np.random.default_rng(13)
    narrow = np.exp(-0.5 * ((x - 3900.0) / 2.0) ** 2)
    broad = 0.6 * np.exp(-0.5 * ((x - 3725.0) / 18.0) ** 2)
    y = narrow + broad + rng.normal(0.0, 0.01, size=x.size)

    y_proc, noise_sigma = mod.preprocess_with_noise(
        x,
        y,
        sg_win=0,
        sg_poly=3,
        als_lam=0.0,
        als_p=0.01,
    )

    args = _build_args(
        cwt_enabled=True,
        cwt_width_min=6.0,
        cwt_width_max=70.0,
        cwt_width_step=6.0,
    )
    candidates = mod.detect_peak_candidates(x, y_proc, noise_sigma, args, resolution_cm=4.0)
    positions = _peak_positions(candidates, x)

    assert np.any(np.abs(positions - 3900.0) <= 5.0)
    assert np.any(np.abs(positions - 3725.0) <= 8.0)


def test_distance_width_limits_reduce_ripple(monkeypatch):
    mod = _load_module(monkeypatch)
    x = np.linspace(4000.0, 3600.0, 1601)
    rng = np.random.default_rng(7)
    ripple = 0.12 * np.sin(2 * np.pi * (x - 3600.0) / 6.0)
    peak = np.exp(-0.5 * ((x - 3800.0) / 8.0) ** 2)
    y = peak + ripple + rng.normal(0.0, 0.015, size=x.size)

    y_proc, noise_sigma = mod.preprocess_with_noise(
        x,
        y,
        sg_win=0,
        sg_poly=3,
        als_lam=0.0,
        als_p=0.01,
    )

    loose_args = _build_args(
        min_distance=2.0,
        peak_width_min=1.0,
        peak_width_max=8.0,
        noise_sigma_multiplier=0.5,
    )
    strict_args = _build_args(
        min_distance=12.0,
        peak_width_min=12.0,
        peak_width_max=40.0,
        noise_sigma_multiplier=0.5,
    )

    loose = mod.detect_peak_candidates(x, y_proc, noise_sigma, loose_args, resolution_cm=4.0)
    strict = mod.detect_peak_candidates(x, y_proc, noise_sigma, strict_args, resolution_cm=4.0)

    assert len(strict) < len(loose)
    strict_positions = _peak_positions(strict, x)
    assert np.any(np.abs(strict_positions - 3800.0) <= 6.0)

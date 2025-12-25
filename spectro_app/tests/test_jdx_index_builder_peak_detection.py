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


def test_refines_crowded_mid_ir_peaks(monkeypatch):
    mod = _load_module(monkeypatch)
    x = np.linspace(1800.0, 1500.0, 1201)
    peak_a = 1.2 * np.exp(-0.5 * ((x - 1605.0) / 6.0) ** 2)
    peak_b = 0.9 * np.exp(-0.5 * ((x - 1622.0) / 8.0) ** 2)
    rng = np.random.default_rng(42)
    y = peak_a + peak_b + rng.normal(0.0, 0.01, size=x.size)

    y_proc, noise_sigma = mod.preprocess_with_noise(
        x,
        y,
        sg_win=0,
        sg_poly=3,
        als_lam=0.0,
        als_p=0.01,
    )

    args = _build_args(
        min_distance=4.0,
        peak_width_min=3.0,
        peak_width_max=30.0,
        noise_sigma_multiplier=2.0,
        cwt_enabled=True,
        cwt_width_min=4.0,
        cwt_width_max=30.0,
        cwt_width_step=4.0,
    )
    args.model = "Voigt"
    args.fit_window_pts = 60
    args.min_r2 = 0.7

    candidates = mod.detect_peak_candidates(x, y_proc, noise_sigma, args, resolution_cm=4.0)
    refined = mod.refine_peak_candidates(x, y_proc, candidates, args, y_abs=y)

    centers = np.asarray([float(p["center"]) for p in refined], dtype=float)
    assert np.any(np.abs(centers - 1605.0) <= 4.0)
    assert np.any(np.abs(centers - 1622.0) <= 5.0)

    fwhm = np.asarray([float(p["fwhm"]) for p in refined], dtype=float)
    assert np.all(fwhm > 4.0)


def test_plateau_midpoint_center(monkeypatch):
    mod = _load_module(monkeypatch)
    x = np.linspace(1200.0, 1100.0, 201)
    peak = 2.0 * np.exp(-0.5 * ((x - 1150.0) / 4.0) ** 2)
    y = np.minimum(peak, 1.0)

    y_proc, noise_sigma = mod.preprocess_with_noise(
        x,
        y,
        sg_win=0,
        sg_poly=3,
        als_lam=0.0,
        als_p=0.01,
    )

    args = _build_args(min_distance=2.0, noise_sigma_multiplier=0.5)
    args.fit_window_pts = 8
    args.min_r2 = 0.0

    candidates = mod.detect_peak_candidates(x, y_proc, noise_sigma, args, resolution_cm=4.0)
    refined = mod.refine_peak_candidates(x, y_proc, candidates, args, y_abs=y)

    assert refined
    centers = np.asarray([float(p["center"]) for p in refined], dtype=float)

    plateau_mask = y >= (np.max(y) - 1e-6)
    plateau_indices = np.flatnonzero(plateau_mask)
    expected_center = float((x[plateau_indices[0]] + x[plateau_indices[-1]]) / 2.0)

    closest = centers[np.argmin(np.abs(centers - expected_center))]
    step = float(np.median(np.abs(np.diff(x))))
    assert abs(closest - expected_center) <= step * 1.5

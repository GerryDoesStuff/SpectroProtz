import numpy as np
import pytest

from spectro_app.tests.jdx_index_builder_test_utils import install_duckdb_stub, load_index_builder


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    peak_idx = int(np.nanargmax(y))
    half_max = float(y[peak_idx]) / 2.0
    left_candidates = np.where(y[:peak_idx] <= half_max)[0]
    right_candidates = np.where(y[peak_idx:] <= half_max)[0]
    if left_candidates.size == 0 or right_candidates.size == 0:
        return float("nan")
    left_idx = int(left_candidates[-1])
    right_idx = int(peak_idx + right_candidates[0])
    return float(abs(x[right_idx] - x[left_idx]))


def _load_module(monkeypatch):
    install_duckdb_stub(monkeypatch)
    return load_index_builder()


def test_savgol_spacing_window_preserves_narrow_peak(monkeypatch):
    mod = _load_module(monkeypatch)
    x = np.linspace(4000.0, 3800.0, 201)
    sigma = 2.0
    peak_center = 3900.0
    y = np.exp(-0.5 * ((x - peak_center) / sigma) ** 2)

    original_fwhm = _fwhm(x, y)
    smoothed, _ = mod.preprocess_with_noise(
        x,
        y,
        sg_win=0,
        sg_poly=3,
        als_lam=0.0,
        als_p=0.01,
        sg_window_cm=9.0,
    )
    smoothed_fwhm = _fwhm(x, smoothed)

    assert np.isfinite(original_fwhm)
    assert np.isfinite(smoothed_fwhm)
    assert smoothed_fwhm <= original_fwhm * 1.4


def test_noise_sigma_is_stable_between_clean_and_noisy(monkeypatch):
    mod = _load_module(monkeypatch)
    rng = np.random.default_rng(42)
    x = np.linspace(4000.0, 3800.0, 401)
    baseline = 0.0001 * (x - x.mean()) ** 2 + 0.5
    sigma = 0.01

    sigma_clean = mod.estimate_noise_sigma(
        x, baseline, sg_win=0, sg_poly=3, sg_window_cm=15.0
    )
    noisy = baseline + rng.normal(0.0, sigma, size=baseline.size)
    sigma_noisy = mod.estimate_noise_sigma(
        x, noisy, sg_win=0, sg_poly=3, sg_window_cm=15.0
    )

    assert sigma_clean < sigma * 0.3
    assert sigma_noisy == pytest.approx(sigma, rel=0.4)

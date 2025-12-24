import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_jdx_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "jdxIndexBuilder.py"
    if "duckdb" not in sys.modules:
        duckdb_stub = types.ModuleType("duckdb")
        sys.modules["duckdb"] = duckdb_stub
    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.__path__ = []  # mark as package
        cluster_stub = types.ModuleType("sklearn.cluster")

        class _DBSCAN:  # pragma: no cover - not used directly
            def __init__(self, *args, **kwargs):
                raise RuntimeError("DBSCAN stub")

            def fit_predict(self, *args, **kwargs):
                raise RuntimeError("DBSCAN stub")

        cluster_stub.DBSCAN = _DBSCAN
        sklearn_stub.cluster = cluster_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.cluster"] = cluster_stub
    spec = importlib.util.spec_from_file_location("jdxIndexBuilder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preprocess_handles_very_short_spectrum():
    mod = _load_jdx_module()
    spectrum = np.array([1.0, 0.5], dtype=float)
    x_axis = np.array([4000.0, 3990.0], dtype=float)
    result = mod.preprocess(x_axis, spectrum, sg_win=9, sg_poly=3, als_lam=1e5, als_p=0.01)
    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))


def test_preprocess_handles_single_point_spectrum():
    mod = _load_jdx_module()
    spectrum = np.array([2.0], dtype=float)
    x_axis = np.array([4000.0], dtype=float)
    result = mod.preprocess(x_axis, spectrum, sg_win=9, sg_poly=3, als_lam=1e5, als_p=0.01)
    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))


def test_airpls_baseline_preserves_strong_band():
    mod = _load_jdx_module()
    x_axis = np.linspace(4000.0, 450.0, 1200)
    baseline = 0.0004 * (x_axis - x_axis.mean())
    peak_center = 1700.0
    peak = 1.8 * np.exp(-0.5 * ((x_axis - peak_center) / 18.0) ** 2)
    intensity = baseline + peak
    baseline_fit = mod.airpls_baseline(intensity, lam=8e4, niter=25)
    corrected = intensity - baseline_fit
    peak_idx = int(np.argmin(np.abs(x_axis - peak_center)))
    assert corrected[peak_idx] > 1.4


def test_piecewise_baseline_correction_handles_regions():
    mod = _load_jdx_module()
    x_axis = np.linspace(4000.0, 450.0, 1400)
    baseline = 0.0002 * (x_axis - 4000.0) + 0.03 * np.sin(x_axis / 600.0)
    peak = 0.8 * np.exp(-0.5 * ((x_axis - 2900.0) / 25.0) ** 2)
    intensity = baseline + peak
    corrected = mod.preprocess(
        x_axis,
        intensity,
        sg_win=7,
        sg_poly=2,
        als_lam=1e5,
        als_p=0.01,
        baseline_method="airpls",
        baseline_niter=20,
        baseline_piecewise=True,
        baseline_ranges="4000-2500,2500-1800,1800-900,900-450",
    )
    assert corrected.shape == intensity.shape
    assert np.all(np.isfinite(corrected))
    assert np.max(corrected) > 0.7

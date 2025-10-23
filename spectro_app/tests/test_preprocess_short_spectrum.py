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
    result = mod.preprocess(spectrum, sg_win=9, sg_poly=3, als_lam=1e5, als_p=0.01)
    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))


def test_preprocess_handles_single_point_spectrum():
    mod = _load_jdx_module()
    spectrum = np.array([2.0], dtype=float)
    result = mod.preprocess(spectrum, sg_win=9, sg_poly=3, als_lam=1e5, als_p=0.01)
    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))

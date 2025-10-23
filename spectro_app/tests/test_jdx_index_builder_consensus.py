import importlib.util
import sys
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import sqlite3
import types


def _install_duckdb_stub(monkeypatch):
    class _Result:
        def __init__(self, cursor):
            self._rows = cursor.fetchall()
            self._cols = [col[0] for col in cursor.description or []]

        def fetch_df(self):
            return pd.DataFrame(self._rows, columns=self._cols)

        def fetchall(self):
            return list(self._rows)

    class _Connection:
        def __init__(self, path: str):
            self._conn = sqlite3.connect(path)

        def execute(self, sql: str, params=None):
            cursor = self._conn.cursor()
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            self._conn.commit()
            return _Result(cursor)

        def executemany(self, sql: str, seq_of_parameters):
            cursor = self._conn.cursor()
            cursor.executemany(sql, list(seq_of_parameters))
            self._conn.commit()
            return _Result(cursor)

        def close(self):
            self._conn.close()

    duckdb_stub = types.ModuleType("duckdb")
    duckdb_stub.connect = lambda path: _Connection(path)
    monkeypatch.setitem(sys.modules, "duckdb", duckdb_stub)


def _load_index_builder():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "jdxIndexBuilder.py"
    spec = importlib.util.spec_from_file_location("jdxIndexBuilder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _StubDBSCAN:
    def __init__(self, eps: float, min_samples: int, **_):
        self.eps = float(eps)
        self.min_samples = int(min_samples)

    def fit_predict(self, X):
        values = np.asarray(X, dtype=float)[:, 0]
        if values.size == 0:
            return np.empty(0, dtype=int)
        order = np.argsort(values)
        labels = np.full(values.size, -1, dtype=int)
        cluster = []
        cluster_id = 0
        last_val = None
        for idx in order:
            val = values[idx]
            if not cluster:
                cluster = [idx]
                last_val = val
                continue
            if last_val is not None and abs(val - last_val) <= self.eps:
                cluster.append(idx)
                last_val = val
                continue
            if len(cluster) >= self.min_samples:
                for c_idx in cluster:
                    labels[c_idx] = cluster_id
                cluster_id += 1
            cluster = [idx]
            last_val = val
        if len(cluster) >= self.min_samples:
            for c_idx in cluster:
                labels[c_idx] = cluster_id
        return labels


def test_indexer_persists_consensus(tmp_path, monkeypatch):
    _install_duckdb_stub(monkeypatch)
    module = _load_index_builder()
    monkeypatch.setattr(module, "DBSCAN", _StubDBSCAN)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"

    x = np.linspace(1000.0, 1048.0, 33)
    peak = np.exp(-0.5 * ((x - 1024.0) / 2.5) ** 2)
    y = 0.05 + peak

    lines = [
        "##TITLE=Fixture",
        "##JCAMP-DX=5.00",
        "##DATA TYPE=INFRARED SPECTRUM",
        "##XUNITS=1/CM",
        "##YUNITS=ABSORBANCE",
        "##XYDATA=(X,Y)",
    ]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.2f} {yi:.6f}")
    lines.append("##END=")

    spectrum_path = data_dir / "sample.jdx"
    spectrum_path.write_text(textwrap.dedent("\n".join(lines)))

    argv = [
        "jdxIndexBuilder",
        str(data_dir),
        str(index_dir),
        "--file-min-samples",
        "1",
        "--global-min-samples",
        "1",
        "--prominence",
        "0.05",
        "--min-distance",
        "1.0",
        "--sg-win",
        "5",
        "--sg-poly",
        "2",
    ]

    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    db_path = index_dir / "peaks.duckdb"
    assert db_path.exists()

    duckdb = sys.modules["duckdb"]
    con = duckdb.connect(str(db_path))
    file_rows = con.execute(
        "SELECT file_id, cluster_id, center, fwhm, support FROM file_consensus"
    ).fetchall()
    global_rows = con.execute(
        "SELECT cluster_id, center, support FROM global_consensus"
    ).fetchall()
    con.close()

    assert file_rows, "per-file consensus should be persisted"
    assert global_rows, "global consensus should be persisted"

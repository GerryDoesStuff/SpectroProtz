import importlib.util
import re
import sys
from pathlib import Path
import sqlite3
import types

import numpy as np
import pandas as pd


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
        upper_sql = sql.upper()
        if "ADD COLUMN IF NOT EXISTS" in upper_sql:
            return self._execute_add_column_if_not_exists(sql)
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

    def _execute_add_column_if_not_exists(self, sql: str):
        match = re.search(
            r"ALTER\s+TABLE\s+(\S+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\S+)\s+(.+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise sqlite3.OperationalError(sql)
        table = self._strip_identifier(match.group(1))
        column = self._strip_identifier(match.group(2))
        column_def = match.group(3)
        cursor = self._conn.cursor()
        if not self._has_column(table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
            self._conn.commit()
            return _Result(cursor)
        cursor.execute("SELECT 1 WHERE 0")
        return _Result(cursor)

    def _has_column(self, table: str, column: str) -> bool:
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())

    @staticmethod
    def _strip_identifier(identifier: str) -> str:
        return identifier.strip().strip('"').strip("'").strip('`')


def install_duckdb_stub(monkeypatch):
    duckdb_stub = types.ModuleType("duckdb")
    duckdb_stub.connect = lambda path: _Connection(path)
    monkeypatch.setitem(sys.modules, "duckdb", duckdb_stub)


def load_index_builder():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "jdxIndexBuilder.py"
    spec = importlib.util.spec_from_file_location("jdxIndexBuilder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StubDBSCAN:
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

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import duckdb

SPECTRA_CORE_COLUMNS = [
    "file_id",
    "path",
    "n_points",
    "n_spectra",
    "meta_json",
]
SPECTRA_PROMOTED_COLUMNS = [
    "title",
    "data_type",
    "jcamp_ver",
    "npoints_hdr",
    "x_units_raw",
    "y_units_raw",
    "x_factor",
    "y_factor",
    "deltax_hdr",
    "firstx",
    "lastx",
    "firsty",
    "maxx",
    "minx",
    "maxy",
    "miny",
    "resolution",
    "state",
    "class",
    "origin",
    "owner",
    "date",
    "names",
    "cas",
    "molform",
    "nist_source",
]

SPECTRA_COLUMNS = SPECTRA_CORE_COLUMNS + SPECTRA_PROMOTED_COLUMNS

REQUIRED_TABLE_COLUMNS: Dict[str, List[str]] = {
    "spectra": SPECTRA_COLUMNS,
    "peaks": [
        "file_id",
        "spectrum_id",
        "peak_id",
        "polarity",
        "center",
        "fwhm",
        "amplitude",
        "area",
        "r2",
    ],
    "file_consensus": [
        "file_id",
        "cluster_id",
        "polarity",
        "center",
        "fwhm",
        "support",
    ],
    "global_consensus": [
        "cluster_id",
        "polarity",
        "center",
        "support",
    ],
}

LOOKUP_QUERIES = {
    "spectra": f"SELECT {', '.join(SPECTRA_COLUMNS)} FROM spectra",
    "peaks_by_file": (
        "SELECT file_id, spectrum_id, peak_id, polarity, center, fwhm, "
        "amplitude, area, r2 FROM peaks WHERE file_id = ?"
    ),
    "file_consensus_by_file": (
        "SELECT file_id, cluster_id, polarity, center, fwhm, support "
        "FROM file_consensus WHERE file_id = ?"
    ),
    "global_consensus": (
        "SELECT cluster_id, polarity, center, support FROM global_consensus"
    ),
}


def validate_ftir_index_schema(db_path: str | Path) -> List[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return [f"Index database not found: {path}"]
    if not path.is_file():
        return [f"Index database path is not a file: {path}"]

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = _fetch_tables(con)
        errors: List[str] = []
        for table, required_columns in REQUIRED_TABLE_COLUMNS.items():
            if table not in tables:
                errors.append(f"Missing table '{table}'.")
                continue
            missing = _missing_columns(con, table, required_columns)
            if missing:
                errors.append(
                    f"Missing columns in '{table}': {', '.join(sorted(missing))}."
                )
        return errors
    finally:
        con.close()


def _fetch_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SHOW TABLES").fetchall()
    return {str(row[0]) for row in rows}


def _missing_columns(
    con: duckdb.DuckDBPyConnection,
    table: str,
    required_columns: Iterable[str],
) -> set[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    existing = {str(row[1]) for row in rows}
    return {col for col in required_columns if col not in existing}


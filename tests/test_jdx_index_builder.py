import importlib.util
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest


def _load_index_builder():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "jdxIndexBuilder.py"
    spec = importlib.util.spec_from_file_location("jdxIndexBuilder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


idx = _load_index_builder()


def _build_args(prominence: float) -> SimpleNamespace:
    return SimpleNamespace(
        prominence=prominence,
        min_distance=1.0,
        sg_win=0,
        sg_poly=0,
        als_lam=0.0,
        als_p=0.01,
        model="Gaussian",
        fit_window_pts=5,
        min_r2=0.0,
        strict=False,
    )


def _write_test_jdx(path):
    lines = [
        "##TITLE=Two Peaks",
        "##JCAMP-DX=5.00",
        "##DATA TYPE=INFRARED SPECTRUM",
        "##XUNITS=1/CM",
        "##YUNITS=ABSORBANCE",
        "##NPOINTS=10",
        "##XYDATA=(X,Y)",
        "1000 0.0",
        "1001 0.2",
        "1002 0.2",
        "1003 0.2",
        "1004 0.0",
        "1005 0.0",
        "1006 0.1",
        "1007 1.0",
        "1008 0.1",
        "1009 0.0",
        "##END=",
    ]
    path.write_text("\n".join(lines))


def _write_descending_test_jdx(path):
    lines = [
        "##TITLE=Two Peaks Descending",
        "##JCAMP-DX=5.00",
        "##DATA TYPE=INFRARED SPECTRUM",
        "##XUNITS=1/CM",
        "##YUNITS=ABSORBANCE",
        "##NPOINTS=10",
        "##XYDATA=(X,Y)",
        "1009 0.0",
        "1008 0.1",
        "1007 1.0",
        "1006 0.1",
        "1005 0.0",
        "1004 0.0",
        "1003 0.2",
        "1002 0.2",
        "1001 0.2",
        "1000 0.0",
        "##END=",
    ]
    path.write_text("\n".join(lines))


def _write_malformed_jdx(path):
    path.write_text(
        "\n".join(
            [
                "##TITLE=Bad Spectrum",
                "##JCAMP-DX=5.00",
                "##DATA TYPE=INFRARED SPECTRUM",
                "##XYDATA=(X,Y)",
                "##END=",
            ]
        )
    )


def test_index_file_overwrites_existing_peaks(tmp_path):
    data_path = tmp_path / "sample.jdx"
    _write_test_jdx(data_path)

    index_dir = tmp_path / "index"
    con = idx.init_db(str(index_dir))

    try:
        _, first_total = idx.index_file(str(data_path), con, _build_args(prominence=0.05))
        assert first_total == 2

        _, second_total = idx.index_file(str(data_path), con, _build_args(prominence=0.6))
        assert second_total == 1

        file_id = idx.file_sha1(str(data_path))
        stored_total = con.execute(
            "SELECT COUNT(*) FROM peaks WHERE file_id = ?", [file_id]
        ).fetchone()[0]

        assert stored_total == second_total
    finally:
        con.close()


def test_index_file_logs_and_records_malformed_input(tmp_path, caplog):
    data_path = tmp_path / "broken.jdx"
    _write_malformed_jdx(data_path)

    index_dir = tmp_path / "index"
    con = idx.init_db(str(index_dir))

    args = _build_args(prominence=0.05)

    with caplog.at_level("ERROR"):
        processed, peaks = idx.index_file(str(data_path), con, args)

    try:
        assert processed == 0
        assert peaks == 0

        assert any("Failed to parse JCAMP file" in record.getMessage() for record in caplog.records)

        error_rows = con.execute(
            "SELECT file_path, error FROM ingest_errors"
        ).fetchall()

        assert error_rows and error_rows[0][0] == str(data_path)
        assert "No spectra parsed" in error_rows[0][1]

        file_id = idx.file_sha1(str(data_path))
        peak_count = con.execute(
            "SELECT COUNT(*) FROM peaks WHERE file_id = ?",
            [file_id],
        ).fetchone()[0]
        assert peak_count == 0
    finally:
        con.close()


def test_index_file_strict_mode_raises(tmp_path):
    data_path = tmp_path / "broken.jdx"
    _write_malformed_jdx(data_path)

    index_dir = tmp_path / "index"
    con = idx.init_db(str(index_dir))

    args = _build_args(prominence=0.05)
    args.strict = True

    try:
        with pytest.raises(ValueError):
            idx.index_file(str(data_path), con, args)

        error_rows = con.execute(
            "SELECT file_path FROM ingest_errors"
        ).fetchall()
        assert error_rows == [(str(data_path),)]
    finally:
        con.close()


def test_descending_axis_consensus_matches(tmp_path):
    asc_path = tmp_path / "ascending.jdx"
    desc_path = tmp_path / "descending.jdx"
    _write_test_jdx(asc_path)
    _write_descending_test_jdx(desc_path)

    index_dir = tmp_path / "index"
    con = idx.init_db(str(index_dir))

    args = _build_args(prominence=0.05)
    args.file_min_samples = 1
    args.file_eps_factor = 0.5
    args.file_eps_min = 0.1

    try:
        idx.index_file(str(asc_path), con, args)
        idx.index_file(str(desc_path), con, args)

        fc = idx.build_file_consensus(con, args)

        asc_id = idx.file_sha1(str(asc_path))
        desc_id = idx.file_sha1(str(desc_path))

        asc_centers = sorted(fc.loc[fc["file_id"] == asc_id, "center"].tolist())
        desc_centers = sorted(fc.loc[fc["file_id"] == desc_id, "center"].tolist())

        assert asc_centers
        assert desc_centers
        assert asc_centers == pytest.approx(desc_centers)
    finally:
        con.close()

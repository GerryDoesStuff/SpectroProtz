import importlib.util
from pathlib import Path
from types import SimpleNamespace

import duckdb


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

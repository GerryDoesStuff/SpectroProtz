import sys
from pathlib import Path

import numpy as np

from .jdx_index_builder_test_utils import (
    StubDBSCAN,
    install_duckdb_stub,
    load_index_builder,
)


def _write_fixture(path: Path):
    x = np.linspace(1000.0, 1008.0, 5)
    y = np.linspace(0.05, 0.15, 5)
    lines = [
        "##TITLE=Promoted Fixture",
        "##DATA TYPE=INFRARED SPECTRUM",
        "##JCAMP-DX=5.00",
        "##XUNITS=1/CM",
        "##YUNITS=ABSORBANCE",
        "##NPOINTS=5",
        "##XFACTOR=1",
        "##YFACTOR=1",
        "##FIRSTX=1000",
        "##LASTX=1008",
        "##FIRSTY=0.05",
        "##DELTAX=2",
        "##MAXX=1008",
        "##MINX=1000",
        "##MAXY=0.15",
        "##MINY=0.05",
        "##RESOLUTION=4",
        "##STATE=Liquid",
        "##CLASS=Reference",
        "##ORIGIN=Spectro Lab",
        "##OWNER=Analyst",
        "##DATE=2024-05-01",
        "##NAMES=Sample A; Sample B",
        "##CAS REGISTRY NO=50-00-0",
        "##MOLFORM=CH2O",
        "##$NIST SOURCE=Instrument",
        "##XYDATA=(X,Y)",
    ]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.2f} {yi:.5f}")
    lines.append("##END=")
    path.write_text("\n".join(lines))


def test_promoted_columns_populated(tmp_path, monkeypatch):
    install_duckdb_stub(monkeypatch)
    module = load_index_builder()
    monkeypatch.setattr(module, "DBSCAN", StubDBSCAN)

    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    data_dir.mkdir()

    fixture_path = data_dir / "fixture.jdx"
    _write_fixture(fixture_path)

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
    result = con.execute(
        "SELECT title, npoints_hdr, firstx, owner, cas FROM spectra"
    ).fetchall()
    con.close()

    assert result == [
        ("Promoted Fixture", 5.0, 1000.0, "Analyst", "50-00-0"),
    ]

import sys
import textwrap

import numpy as np

from .jdx_index_builder_test_utils import (
    StubDBSCAN,
    install_duckdb_stub,
    load_index_builder,
)


def test_indexer_persists_consensus(tmp_path, monkeypatch):
    install_duckdb_stub(monkeypatch)
    module = load_index_builder()
    monkeypatch.setattr(module, "DBSCAN", StubDBSCAN)

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

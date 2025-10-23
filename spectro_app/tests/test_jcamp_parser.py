import sys
import types

import numpy as np

if 'duckdb' not in sys.modules:
    sys.modules['duckdb'] = types.SimpleNamespace(connect=lambda *args, **kwargs: None)

if 'sklearn' not in sys.modules:
    sklearn_module = types.ModuleType('sklearn')
    cluster_module = types.ModuleType('sklearn.cluster')
    cluster_module.DBSCAN = object()
    sklearn_module.cluster = cluster_module
    sys.modules['sklearn'] = sklearn_module
    sys.modules['sklearn.cluster'] = cluster_module

from scripts.jdxIndexBuilder import parse_jcamp_multispec


def test_parse_jcamp_multispec_expands_compressed_single_spectrum(tmp_path):
    sample = """##TITLE=Compressed Spectrum
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=TRANSMITTANCE
##XFACTOR=1
##YFACTOR=1
##FIRSTX=4000
##DELTAX=-1
##NPOINTS=11
##XYDATA=(X++(Y..Y))
4000 0.10 0.20 0.30 0.40 0.50
3995 0.60 0.70 0.80 0.90 1.00
3990 1.10
##END="""
    path = tmp_path / "compressed.jdx"
    path.write_text(sample)

    x, spectra, _ = parse_jcamp_multispec(str(path))

    assert len(spectra) == 1
    y = spectra[0]
    assert len(x) == len(y) == 11

    expected_x = np.linspace(4000, 3990, 11)
    expected_y = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10])

    np.testing.assert_allclose(x, expected_x)
    np.testing.assert_allclose(y, expected_y)


def test_parse_jcamp_multispec_preserves_multiple_blocks(tmp_path):
    sample = """##TITLE=Replicated Spectrum
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=TRANSMITTANCE
##XFACTOR=1
##YFACTOR=1
##FIRSTX=4000
##DELTAX=-1
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000 0.1 0.2
3998 0.3 0.4
##XYDATA=(X++(Y..Y))
4000 0.5 0.6
3998 0.7 0.8
##END="""
    path = tmp_path / "replicates.jdx"
    path.write_text(sample)

    x, spectra, _ = parse_jcamp_multispec(str(path))

    assert len(spectra) == 2
    np.testing.assert_allclose(x, np.array([4000., 3999., 3998., 3997.]))
    np.testing.assert_allclose(spectra[0], np.array([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_allclose(spectra[1], np.array([0.5, 0.6, 0.7, 0.8]))

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

from scripts.jdxIndexBuilder import (
    convert_y_for_processing,
    parse_jcamp_multispec,
    transmittance_to_abs,
)


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


def test_parse_jcamp_multispec_trims_short_replicate_columns(tmp_path):
    sample = """##TITLE=Uneven Replicates
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=TRANSMITTANCE
##XFACTOR=1
##YFACTOR=1
##XYDATA=(XY..Y)
4000 0.10 0.20
3995 0.30 0.40
3990 0.50
3985 0.60
##END"""

    path = tmp_path / "uneven_replicates.jdx"
    path.write_text(sample)

    x, spectra, _ = parse_jcamp_multispec(str(path))

    assert len(spectra) == 2
    np.testing.assert_allclose(x, np.array([4000.0, 3995.0, 3990.0, 3985.0]))
    np.testing.assert_allclose(spectra[0], np.array([0.10, 0.30, 0.50, 0.60]))
    assert len(spectra[1]) == 2
    np.testing.assert_allclose(spectra[1], np.array([0.20, 0.40]))
    assert not np.isnan(spectra[1]).any()


def test_convert_y_for_processing_handles_fraction_transmittance():
    y = np.array([0.5, 0.25, 1.0])

    converted = convert_y_for_processing(y, "TRANSMITTANCE")

    expected = transmittance_to_abs(y)
    np.testing.assert_allclose(converted, expected)
    # original array should remain untouched
    np.testing.assert_allclose(y, np.array([0.5, 0.25, 1.0]))


def test_convert_y_for_processing_handles_percent_transmittance(tmp_path):
    sample = """##TITLE=Percent Transmittance
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=%T
##XFACTOR=1
##YFACTOR=1
##FIRSTX=1000
##DELTAX=-1
##NPOINTS=3
##XYDATA=(X++(Y..Y))
1000 50 75 100
##END"""
    path = tmp_path / "percent_transmittance.jdx"
    path.write_text(sample)

    _, spectra, headers = parse_jcamp_multispec(str(path))

    converted = convert_y_for_processing(spectra[0], headers.get("YUNITS", ""))
    expected = transmittance_to_abs(np.array([50.0, 75.0, 100.0]), percent=True)

    np.testing.assert_allclose(converted, expected)


def test_convert_y_for_processing_leaves_absorbance_untouched():
    y = np.array([0.1, 0.2, 0.3])

    converted = convert_y_for_processing(y, "ABSORBANCE")

    np.testing.assert_allclose(converted, y)

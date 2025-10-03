import numpy as np
import pandas as pd

from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def test_helios_csv_parsing(tmp_path):
    helios_csv = """Instrument: Helios Gamma
Mode: Absorbance
Pathlength: 1 cm
Sample ID: Sample-001
Reference ID: Blank-A

Wavelength (nm);Absorbance
200;0,123
201;0,456
"""
    path = tmp_path / "sample_helios.csv"
    path.write_text(helios_csv, encoding="utf-8")

    plugin = UvVisPlugin()
    spectra = plugin.load([str(path)])

    assert len(spectra) == 1
    spec = spectra[0]
    assert np.allclose(spec.wavelength, [200.0, 201.0])
    assert np.allclose(spec.intensity, [0.123, 0.456])
    assert spec.meta["instrument"] == "Helios Gamma"
    assert spec.meta["mode"] == "absorbance"
    assert spec.meta["pathlength_cm"] == 1.0
    assert spec.meta["sample_id"] == "Sample-001"
    assert spec.meta["blank_id"] == "Blank-A"
    assert spec.meta["role"] == "sample"


def test_helios_excel_parsing(tmp_path):
    rows = [
        ["Instrument", "Helios Gamma"],
        ["Mode", "%T"],
        ["Pathlength", "0.5 cm"],
        ["Sample ID", "Protein-01"],
        ["Reference ID", "Buffer"],
        ["", ""],
        ["Wavelength (nm)", "Protein-01"],
        [240, 85.0],
        [260, 83.0],
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "helios.xlsx"
    df.to_excel(path, index=False, header=False)

    plugin = UvVisPlugin()
    spectra = plugin.load([str(path)])

    assert len(spectra) == 1
    spec = spectra[0]
    assert np.allclose(spec.wavelength, [240.0, 260.0])
    assert np.allclose(spec.intensity, [85.0, 83.0])
    assert spec.meta["mode"] == "transmittance"
    assert spec.meta["pathlength_cm"] == 0.5
    assert spec.meta["blank_id"] == "Buffer"
    assert spec.meta["sample_id"] == "Protein-01"
    assert spec.meta["role"] == "sample"


def test_generic_csv_with_metadata(tmp_path):
    generic_csv = """Instrument: CustomSpec 2000
Mode: Absorbance

wavelength,Sample A,Blank Control
200,0.100,0.010
205,0.105,0.011
"""
    path = tmp_path / "generic.csv"
    path.write_text(generic_csv, encoding="utf-8")

    plugin = UvVisPlugin()
    spectra = plugin.load([str(path)])

    assert len(spectra) == 2
    samples = {spec.meta["sample_id"]: spec for spec in spectra}
    sample_spec = samples["Sample A"]
    blank_spec = samples["Blank Control"]

    assert np.allclose(sample_spec.wavelength, [200.0, 205.0])
    assert np.allclose(sample_spec.intensity, [0.1, 0.105])
    assert sample_spec.meta["instrument"] == "CustomSpec 2000"
    assert sample_spec.meta["mode"] == "absorbance"
    assert sample_spec.meta["blank_id"] == "Blank Control"
    assert sample_spec.meta["role"] == "sample"

    assert blank_spec.meta["role"] == "blank"
    assert blank_spec.meta["blank_id"] == "Blank Control"

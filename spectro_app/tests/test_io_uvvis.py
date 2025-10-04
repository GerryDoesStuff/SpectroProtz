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
    expected_absorbance = -np.log10(np.array([0.85, 0.83]))
    assert np.allclose(spec.intensity, expected_absorbance)
    assert spec.meta["mode"] == "absorbance"
    assert spec.meta.get("original_mode") == "transmittance"
    channels = spec.meta.get("channels") or {}
    assert "raw_transmittance" in channels
    assert np.allclose(np.asarray(channels["raw_transmittance"], dtype=float), [85.0, 83.0])
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


def test_generic_reflectance_conversion(tmp_path):
    generic_csv = """Instrument: CustomSpec 2000
Mode: Reflectance

wavelength,Sample
400,45
450,30
"""

    path = tmp_path / "reflectance.csv"
    path.write_text(generic_csv, encoding="utf-8")

    plugin = UvVisPlugin()
    spectra = plugin.load([str(path)])

    assert len(spectra) == 1
    spec = spectra[0]
    assert np.allclose(spec.wavelength, [400.0, 450.0])
    expected_absorbance = -np.log10(np.array([0.45, 0.30]))
    assert np.allclose(spec.intensity, expected_absorbance)
    assert spec.meta["mode"] == "absorbance"
    assert spec.meta.get("original_mode") == "reflectance"
    channels = spec.meta.get("channels") or {}
    assert "raw_reflectance" in channels
    assert np.allclose(np.asarray(channels["raw_reflectance"], dtype=float), [45.0, 30.0])


def test_manifest_metadata_enrichment(tmp_path):
    generic_csv = """wavelength,Sample A,Sample B,Blank Control
200,0.100,0.200,0.010
205,0.105,0.205,0.011
"""
    data_path = tmp_path / "generic.csv"
    data_path.write_text(generic_csv, encoding="utf-8")

    manifest_csv = """file,channel,sample_id,blank_id,replicate,group,role
generic.csv,Sample A,Treated-A,Blank-01,rep-1,Dose-Low,
generic.csv,Sample B,Treated-B,Blank-01,rep-2,Dose-Low,
generic.csv,Blank Control,Blank-01,,ref-blank,QC,blank
"""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(manifest_csv, encoding="utf-8")

    plugin = UvVisPlugin()
    spectra = plugin.load([str(data_path), str(manifest_path)])

    assert len(spectra) == 3

    by_id = {spec.meta["sample_id"]: spec for spec in spectra}
    sample_a = by_id["Treated-A"]
    sample_b = by_id["Treated-B"]
    blank_spec = by_id["Blank-01"]

    assert sample_a.meta["blank_id"] == "Blank-01"
    assert sample_a.meta["replicate_id"] == "rep-1"
    assert sample_a.meta["group_id"] == "Dose-Low"
    assert sample_a.meta["role"] == "sample"

    assert sample_b.meta["blank_id"] == "Blank-01"
    assert sample_b.meta["replicate_id"] == "rep-2"
    assert sample_b.meta["group_id"] == "Dose-Low"

    assert blank_spec.meta["role"] == "blank"
    assert blank_spec.meta["blank_id"] == "Blank-01"
    assert blank_spec.meta["replicate_id"] == "ref-blank"
    assert blank_spec.meta["group_id"] == "QC"
def test_manifest_named_csv_when_disabled(tmp_path):
    content = """wavelength,Sample
200,0.100
205,0.110
"""
    path = tmp_path / "sample_manifest.csv"
    path.write_text(content, encoding="utf-8")

    plugin = UvVisPlugin(enable_manifest=False)
    spectra = plugin.load([str(path)])

    assert len(spectra) == 1
    spec = spectra[0]
    assert np.allclose(spec.wavelength, [200.0, 205.0])
    assert np.allclose(spec.intensity, [0.1, 0.11])
    assert spec.meta["instrument"] == "Generic UV-Vis"
    assert spec.meta["sample_id"] == "Sample"

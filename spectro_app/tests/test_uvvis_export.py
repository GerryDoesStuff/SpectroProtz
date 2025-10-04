import json
from pathlib import Path

import numpy as np

from openpyxl import load_workbook

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def _mock_spectrum():
    wl = np.linspace(220.0, 320.0, 401)
    peak1 = 1.2 * np.exp(-0.5 * ((wl - 260.0) / 4.0) ** 2)
    peak2 = 0.8 * np.exp(-0.5 * ((wl - 280.0) / 5.0) ** 2)
    baseline = 0.05 * (wl - wl.min()) / (wl.max() - wl.min())
    intensity = peak1 + peak2 + baseline
    return Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": "Sample-1", "role": "sample", "mode": "absorbance"},
    )


def test_uvvis_analyze_extracts_features():
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    recipe = {
        "features": {
            "integrals": [
                {"name": "Area_250_280", "min": 250.0, "max": 280.0},
            ],
        }
    }

    processed, qc_rows = plugin.analyze([spec], recipe)

    assert len(processed) == 1
    enriched = processed[0]
    features = enriched.meta.get("features")
    assert features, "features should be captured in spectrum metadata"

    ratios = features.get("band_ratios", {})
    assert "A260_A280" in ratios
    assert np.isfinite(ratios["A260_A280"]["value"])

    integrals = features.get("integrals", {})
    assert "Area_250_280" in integrals
    assert integrals["Area_250_280"] > 0

    peaks = features.get("peaks", [])
    assert peaks and peaks[0]["wavelength"]

    channels = enriched.meta.get("channels", {})
    assert "first_derivative" in channels
    assert channels["first_derivative"].shape == enriched.wavelength.shape

    assert len(qc_rows) == 1
    qc_entry = qc_rows[0]
    assert "band_ratios" in qc_entry
    assert "integrals" in qc_entry


def test_uvvis_export_creates_workbook_with_derivatives(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    recipe = {"export": {"path": str(tmp_path / "uvvis_batch.xlsx")}}

    processed, qc_rows = plugin.analyze([spec], recipe)
    result = plugin.export(processed, qc_rows, recipe)

    workbook_path = Path(recipe["export"]["path"])
    assert workbook_path.exists(), "Workbook should be written to disk"

    wb = load_workbook(workbook_path)
    expected_sheets = {"Processed_Spectra", "Metadata", "QC_Flags", "Calibration", "Audit_Log"}
    assert expected_sheets.issubset(set(wb.sheetnames))

    ws_processed = wb["Processed_Spectra"]
    header = [cell.value for cell in next(ws_processed.iter_rows(min_row=1, max_row=1))]
    assert "channel" in header
    channel_idx = header.index("channel")
    channels = {row[channel_idx] for row in ws_processed.iter_rows(min_row=2, values_only=True)}
    assert "first_derivative" in channels
    assert "second_derivative" in channels

    ws_metadata = wb["Metadata"]
    metadata_header = [cell.value for cell in next(ws_metadata.iter_rows(min_row=1, max_row=1))]
    assert any(str(value).startswith("features.band_ratios") for value in metadata_header)

    ws_qc = wb["QC_Flags"]
    qc_header = [cell.value for cell in next(ws_qc.iter_rows(min_row=1, max_row=1))]
    assert any("band_ratios" in str(value) for value in qc_header)

    assert result.figures, "Export should include generated plots"
    assert any("Workbook written" in entry for entry in result.audit)


def test_uvvis_export_writes_sidecar_and_pdf(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    workbook = tmp_path / "uvvis_batch.xlsx"
    recipe_path = tmp_path / "uvvis_batch.recipe.json"
    pdf_path = tmp_path / "uvvis_batch.pdf"
    recipe = {
        "export": {
            "path": str(workbook),
            "recipe_path": str(recipe_path),
            "pdf_path": str(pdf_path),
        },
        "features": {"integrals": [{"name": "Area", "min": 250.0, "max": 270.0}]},
    }

    processed, qc_rows = plugin.analyze([spec], recipe)
    result = plugin.export(processed, qc_rows, recipe)

    assert workbook.exists(), "Workbook should be generated"
    assert recipe_path.exists(), "Recipe sidecar should be generated"
    assert pdf_path.exists(), "PDF report should be generated"
    assert pdf_path.stat().st_size > 0

    exported_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert "features" in exported_recipe
    assert any("Recipe sidecar written" in entry for entry in result.audit)
    assert any("PDF report written" in entry for entry in result.audit)


def test_clean_value_sanitises_formula_strings(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    spec.meta["sample_id"] = "=2+2"
    recipe = {"export": {"path": str(tmp_path / "uvvis_batch.xlsx")}}

    processed, qc_rows = plugin.analyze([spec], recipe)
    plugin.export(processed, qc_rows, recipe)

    wb = load_workbook(recipe["export"]["path"])
    ws_processed = wb["Processed_Spectra"]
    header = [cell.value for cell in next(ws_processed.iter_rows(min_row=1, max_row=1))]
    sample_idx = header.index("sample_id") + 1
    sample_cell = ws_processed.cell(row=2, column=sample_idx)
    assert sample_cell.value.startswith("'")
    assert sample_cell.value[1:] == "=2+2"
    assert sample_cell.data_type == "s"  # Written as literal string rather than formula

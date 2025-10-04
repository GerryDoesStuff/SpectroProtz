import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from importlib import metadata
from openpyxl import load_workbook

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis.plugin import UvVisPlugin
from spectro_app.plugins.uvvis import pipeline


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


def _calibration_spectrum(sample_id: str, absorbance: float, *, role: str = "sample") -> Spectrum:
    wl = np.linspace(240.0, 280.0, 201)
    intensity = np.full_like(wl, absorbance, dtype=float)
    return Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": sample_id, "role": role, "mode": "absorbance", "pathlength_cm": 1.0},
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
    assert "qc_summary_noise.png" in result.figures
    assert any("Workbook written" in entry for entry in result.audit)


def test_uvvis_export_includes_pipeline_stage_channels(tmp_path):
    plugin = UvVisPlugin()
    wl = np.linspace(400.0, 460.0, 7)
    raw_intensity = np.array([1.0, 1.2, 1.1, 3.5, 3.6, 3.7, 3.8])
    sample = Spectrum(
        wavelength=wl,
        intensity=raw_intensity,
        meta={"sample_id": "StageSample", "role": "sample"},
    )
    blank = Spectrum(
        wavelength=wl,
        intensity=np.full_like(wl, 0.2, dtype=float),
        meta={"role": "blank"},
    )

    coerced = pipeline.coerce_domain(sample, {"min": 400.0, "max": 460.0, "num": 7})
    joined = pipeline.correct_joins(coerced, [3], window=1)
    joined.meta["join_indices"] = (3,)
    despiked = pipeline.despike_spectrum(joined, zscore=2.0, window=3)
    blanked = pipeline.subtract_blank(despiked, blank)
    baseline = pipeline.apply_baseline(blanked, "asls", lam=1e2, p=0.01, niter=10)
    smoothed = pipeline.smooth_spectrum(baseline, window=5, polyorder=2)

    recipe = {"export": {"path": str(tmp_path / "stage_channels.xlsx")}}
    result = plugin.export([smoothed], [{}], recipe)
    processed, qc_rows = plugin.analyze([smoothed], recipe)
    assert qc_rows
    qc_row = qc_rows[0]
    assert "roughness" in qc_row
    assert "raw" in qc_row["roughness"]["channels"]
    assert "smoothed" in qc_row["roughness"]["channels"]
    assert qc_row["roughness"]["channels"]["smoothed"] <= qc_row["roughness"]["channels"]["raw"]
    assert qc_row["roughness_delta"]["raw"] > 0.0

    plugin.export(processed, qc_rows, recipe)

    workbook_path = Path(recipe["export"]["path"])
    assert workbook_path.exists()
    wb = load_workbook(workbook_path)
    ws_processed = wb["Processed_Spectra"]
    header = [cell.value for cell in next(ws_processed.iter_rows(min_row=1, max_row=1))]
    channel_idx = header.index("channel")
    channels = {
        row[channel_idx]
        for row in ws_processed.iter_rows(min_row=2, values_only=True)
        if row and row[1] == "StageSample"
    }
    expected = {"raw", "blanked", "baseline_corrected", "joined", "despiked", "smoothed"}
    assert expected.issubset(channels)
    assert any(name.endswith("join_1.png") for name in result.figures)
    ws_qc = wb["QC_Flags"]
    qc_header = [cell.value for cell in next(ws_qc.iter_rows(min_row=1, max_row=1))]
    assert "roughness.processed" in qc_header
    assert "roughness.channels.raw" in qc_header
    assert "roughness.channels.smoothed" in qc_header
    assert "roughness_delta.raw" in qc_header
    data_row = next(ws_qc.iter_rows(min_row=2, max_row=2, values_only=True))
    processed_idx = qc_header.index("roughness.processed")
    raw_idx = qc_header.index("roughness.channels.raw")
    smoothed_idx = qc_header.index("roughness.channels.smoothed")
    raw_delta_idx = qc_header.index("roughness_delta.raw")
    assert data_row[processed_idx] is not None
    assert data_row[raw_idx] is not None
    assert data_row[smoothed_idx] is not None
    assert data_row[raw_idx] > data_row[smoothed_idx]
    assert data_row[raw_delta_idx] > 0.0


def test_uvvis_export_emits_svg_when_requested(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    recipe = {
        "export": {
            "path": str(tmp_path / "uvvis_svg.xlsx"),
            "formats": ["png", "svg"],
        }
    }

    processed, qc_rows = plugin.analyze([spec], recipe)
    result = plugin.export(processed, qc_rows, recipe)

    expected_base = Path("Sample-1_processed")
    png_name = expected_base.with_suffix(".png").name
    svg_name = expected_base.with_suffix(".svg").name
    assert png_name in result.figures
    assert svg_name in result.figures
    svg_stems = {Path(name).stem for name in result.figures if name.endswith(".svg")}
    png_stems = {Path(name).stem for name in result.figures if name.endswith(".png")}
    assert svg_stems <= png_stems
def test_uvvis_export_audit_includes_runtime_and_input_hash(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    source_path = tmp_path / "sample_source.csv"
    source_path.write_text("wavelength,intensity\n")
    spec.meta["source_file"] = str(source_path)
    recipe = {"export": {"path": str(tmp_path / "audit.xlsx")}}

    processed, qc_rows = plugin.analyze([spec], recipe)
    audit_entries = plugin._build_audit_entries(processed, qc_rows, recipe, figures={})

    package_label = "spectro-app"
    version = "unknown"
    for candidate in ("spectro-app", "spectro_app"):
        try:
            version = metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
        else:
            package_label = candidate
            break
    runtime_token = f"{package_label}=={version}"
    assert any(runtime_token in entry for entry in audit_entries)

    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    assert any(digest in entry for entry in audit_entries)
def test_uvvis_export_with_workbook_path_creates_sidecar_and_pdf(tmp_path):
    plugin = UvVisPlugin()
    spec = _mock_spectrum()
    recipe = {
        "export": {
            "path": str(tmp_path / "uvvis_batch.xlsx"),
            "recipe_sidecar": True,
            "pdf_report": True,
        }
    }

    processed, qc_rows = plugin.analyze([spec], recipe)

    result = plugin.export(processed, qc_rows, recipe)

    workbook_path = tmp_path / "uvvis_batch.xlsx"
    recipe_sidecar = tmp_path / "uvvis_batch.recipe.json"
    pdf_report = tmp_path / "uvvis_batch.pdf"

    assert workbook_path.exists()
    assert recipe_sidecar.exists()
    assert pdf_report.exists()

    audit_text = "\n".join(result.audit)
    assert "Workbook written" in audit_text
    assert "Recipe sidecar written" in audit_text
    assert "PDF report written" in audit_text


def test_uvvis_calibration_success(tmp_path):
    plugin = UvVisPlugin()
    slope = 0.12
    intercept = 0.02
    standards = [("Std-0", 0.0), ("Std-1", 5.0), ("Std-2", 10.0)]
    unknowns = [("Sample-1", 7.5)]

    spectra = [
        _calibration_spectrum(sample_id, intercept + slope * conc, role="standard")
        for sample_id, conc in standards
    ]
    spectra.extend(
        _calibration_spectrum(sample_id, intercept + slope * conc, role="sample")
        for sample_id, conc in unknowns
    )

    recipe = {
        "calibration": {
            "targets": [
                {
                    "name": "Analyte",
                    "wavelength": 260.0,
                    "standards": [
                        {"sample_id": sample_id, "concentration": conc}
                        for sample_id, conc in standards
                    ],
                    "unknowns": [{"sample_id": sample_id} for sample_id, _ in unknowns],
                }
            ],
        },
        "export": {"path": str(tmp_path / "calibration_ok.xlsx")},
    }

    processed, qc_rows = plugin.analyze(spectra, recipe)
    results = plugin.last_calibration_results

    assert results and results["status"] == "ok"
    target = results["targets"][0]
    assert pytest.approx(target["fit"]["slope"], rel=1e-6) == slope
    unknown_entry = next(entry for entry in target["unknowns"] if entry["sample_id"] == "Sample-1")
    assert unknown_entry["status"] == "ok"
    assert unknown_entry["predicted_concentration"] == pytest.approx(7.5, rel=1e-4)

    qc_unknown = next(row for row in qc_rows if row["sample_id"] == "Sample-1")
    qc_calibration = qc_unknown.get("calibration", {}).get("Analyte")
    assert qc_calibration
    assert qc_calibration["predicted_concentration"] == pytest.approx(7.5, rel=1e-4)

    result = plugin.export(processed, qc_rows, recipe)
    workbook_path = Path(recipe["export"]["path"])
    assert workbook_path.exists()
    wb = load_workbook(workbook_path)
    ws_calibration = wb["Calibration"]
    rows = list(ws_calibration.iter_rows(values_only=True))
    summary_header = rows[0]
    assert summary_header[:3] == ("Target", "Status", "Slope")
    analyte_summary = next(row for row in rows if row and row[0] == "Analyte")
    assert analyte_summary[1] == "ok"
    assert analyte_summary[2] == pytest.approx(slope, rel=1e-6)

    standards_header = next(row for row in rows if row and "Response (A/cm)" in row)
    assert standards_header[:3] == ("Sample ID", "Concentration", "Response (A/cm)")
    std_row = next(row for row in rows if row and row[0] == "Std-1")
    assert std_row[2] == pytest.approx(intercept + slope * 5.0, rel=1e-6)

    unknowns_header = next(row for row in rows if row and "Predicted Concentration" in row)
    assert unknowns_header[0] == "Sample ID"
    sample_row = next(row for row in rows if row and row[0] == "Sample-1")
    assert sample_row[3] == pytest.approx(7.5, rel=1e-4)
    assert any("Calibration Analyte" in entry for entry in result.audit)
    assert any(name.startswith("calibration_Analyte") for name in result.figures)


def test_uvvis_calibration_insufficient_standards():
    plugin = UvVisPlugin()
    spectra = [
        _calibration_spectrum("Std-0", 0.05, role="standard"),
        _calibration_spectrum("Sample-1", 0.18, role="sample"),
    ]

    recipe = {
        "calibration": {
            "targets": [
                {
                    "name": "Analyte",
                    "wavelength": 260.0,
                    "standards": [{"sample_id": "Std-0", "concentration": 0.0}],
                    "unknowns": [{"sample_id": "Sample-1"}],
                }
            ]
        }
    }

    processed, qc_rows = plugin.analyze(spectra, recipe)
    results = plugin.last_calibration_results
    assert results and results["targets"][0]["status"] == "failed"
    errors = " ".join(results["targets"][0]["errors"])
    assert "At least two" in errors

    qc_unknown = next(row for row in qc_rows if row["sample_id"] == "Sample-1")
    calibration_meta = qc_unknown.get("calibration", {}).get("Analyte")
    assert calibration_meta["status"] == "no_model"


def test_uvvis_calibration_poor_regression():
    plugin = UvVisPlugin()
    standards = [
        ("Std-0", 0.0, 0.02),
        ("Std-1", 5.0, 0.6),
        ("Std-2", 10.0, 0.25),
    ]
    unknown = ("Sample-1", 7.0, 0.31)

    spectra = [
        _calibration_spectrum(sample_id, absorbance, role="standard")
        for sample_id, _, absorbance in standards
    ]
    spectra.append(_calibration_spectrum(unknown[0], unknown[2], role="sample"))

    recipe = {
        "calibration": {
            "r2_threshold": 0.995,
            "targets": [
                {
                    "name": "Analyte",
                    "wavelength": 260.0,
                    "standards": [
                        {"sample_id": sample_id, "concentration": conc}
                        for sample_id, conc, _ in standards
                    ],
                    "unknowns": [{"sample_id": unknown[0]}],
                }
            ],
        }
    }

    _, qc_rows = plugin.analyze(spectra, recipe)
    results = plugin.last_calibration_results
    assert results and results["targets"][0]["status"] == "failed"
    assert any("R^2" in error for error in results["targets"][0]["errors"])

    qc_unknown = next(row for row in qc_rows if row["sample_id"] == unknown[0])
    calibration_meta = qc_unknown.get("calibration", {}).get("Analyte")
    assert calibration_meta["status"] == "no_model"
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

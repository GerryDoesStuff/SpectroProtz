from pathlib import Path

from spectro_app.engine import pipeline as core_pipeline
from spectro_app.engine.excel_writer import write_workbook
from spectro_app.engine.plugin_api import SpectroscopyPlugin, BatchResult
from spectro_app.io.opus import is_opus_path, load_opus_spectra

class FtirPlugin(SpectroscopyPlugin):
    id = "ftir"
    label = "FTIR"
    xlabel = "Wavenumber (cm⁻¹)"

    def detect(self, paths):
        return any(
            str(p).lower().endswith((".csv", ".txt")) or is_opus_path(p)
            for p in paths
        )

    def load(self, paths, cancelled=None):
        spectra = []
        for path in paths:
            if cancelled is not None and cancelled():
                raise RuntimeError("Cancelled")
            if is_opus_path(path):
                # Allow load_opus_spectra errors to propagate so callers can
                # surface the real parsing failure.
                spectra.extend(load_opus_spectra(path, technique="ftir"))
            if cancelled is not None and cancelled():
                raise RuntimeError("Cancelled")
        if not spectra:
            raise ValueError("No OPUS spectra were loaded from the provided paths.")
        return spectra

    def analyze(self, specs, recipe):
        return core_pipeline.run_pipeline(specs, recipe)

    def export(self, specs, qc, recipe):
        specs = list(specs or [])
        qc = list(qc or [])
        export_cfg = dict(recipe.get("export", {})) if recipe else {}
        workbook_value = export_cfg.get("path") or export_cfg.get("workbook")
        workbook_target = None
        if workbook_value not in (None, "", False):
            workbook_target = Path(str(workbook_value)).expanduser()
        audit_entries = []
        if workbook_target:
            resolved_path = str(workbook_target)
            audit_entries.append(f"Workbook written to {resolved_path}")
            write_workbook(
                resolved_path,
                specs,
                qc,
                audit_entries,
                figures={},
                calibration=None,
            )
        else:
            audit_entries.append("No workbook path provided; workbook not written.")
        report_text = "\n".join(audit_entries) if audit_entries else None
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures={},
            audit=audit_entries,
            report_text=report_text,
        )

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
        qc = _normalize_qc_rows(qc or [])
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


def _normalize_qc_rows(qc_rows: Iterable[Any]) -> list[dict[str, Any]]:
    normalized = [_normalize_qc_row(row) for row in qc_rows]
    if normalized:
        _validate_qc_row(normalized[0])
    return normalized


def _normalize_qc_row(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        row = asdict(row)
    elif isinstance(row, dict):
        row = dict(row)
    else:
        row = dict(row)
    return {str(key): _normalize_qc_value(value) for key, value in row.items()}


def _normalize_qc_value(value: Any) -> Any:
    if is_dataclass(value):
        return {str(key): _normalize_qc_value(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_qc_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_qc_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _validate_qc_row(row: dict[str, Any]) -> None:
    if _contains_dataclass(row):
        raise TypeError("QC rows must not contain dataclass instances after normalization.")


def _contains_dataclass(value: Any) -> bool:
    if is_dataclass(value):
        return True
    if isinstance(value, dict):
        return any(_contains_dataclass(val) for val in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_dataclass(item) for item in value)
    return False

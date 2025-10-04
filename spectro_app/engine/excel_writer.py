from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from openpyxl import Workbook

from spectro_app.engine.plugin_api import Spectrum


def _ensure_parent(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _clean_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple, set)):
        return json.dumps([_clean_value(v) for v in value])
    if isinstance(value, dict):
        return json.dumps({str(k): _clean_value(v) for k, v in value.items()})
    if isinstance(value, (Path, os.PathLike)):
        value = os.fspath(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        if value and value[0] in "=+-@":
            if not value.startswith("'"):
                return "'" + value
        return value
    return value


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        safe_key = str(key).replace(" ", "_").replace("/", "_")
        new_key = f"{prefix}{safe_key}" if not prefix else f"{prefix}.{safe_key}"
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        elif isinstance(value, np.ndarray):
            flat[new_key] = json.dumps(_clean_value(value.tolist()))
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, dict) for item in value):
                flat[new_key] = json.dumps([_clean_value(item) for item in value])
            else:
                flat[new_key] = json.dumps([_clean_value(item) for item in value])
        else:
            flat[new_key] = _clean_value(value)
    return flat


def _processed_rows(processed: Sequence[Spectrum]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for idx, spec in enumerate(processed):
        wl = np.asarray(spec.wavelength, dtype=float)
        intensity = np.asarray(spec.intensity, dtype=float)
        meta = spec.meta or {}
        sample_id = _clean_value(
            meta.get("sample_id")
            or meta.get("channel")
            or meta.get("blank_id")
            or f"spec_{idx}"
        )
        role = _clean_value(meta.get("role", "sample"))
        mode = _clean_value(meta.get("mode"))
        base_channel_name = _clean_value(meta.get("channel_label", "processed"))
        for wl_val, inten_val in zip(wl, intensity):
            rows.append([
                idx,
                sample_id,
                role,
                mode,
                base_channel_name,
                _clean_value(wl_val),
                _clean_value(inten_val),
            ])
        extra_channels = meta.get("channels") or {}
        for name, channel in extra_channels.items():
            arr = np.asarray(channel, dtype=float)
            if arr.shape != wl.shape:
                continue
            for wl_val, inten_val in zip(wl, arr):
                rows.append([
                    idx,
                    sample_id,
                    role,
                    mode,
                    _clean_value(name),
                    _clean_value(wl_val),
                    _clean_value(inten_val),
                ])
    return rows


def _metadata_rows(processed: Sequence[Spectrum]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(processed):
        meta = dict(spec.meta or {})
        meta.pop("channels", None)
        flat = _flatten_dict(meta)
        flat["spectrum_index"] = idx
        rows.append(flat)
    return rows


def _qc_rows(qc_table: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in qc_table:
        output.append(_flatten_dict(dict(row)))
    return output


def _write_dict_rows(ws, rows: List[Dict[str, Any]]):
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()})
    ws.append(headers)
    for row in rows:
        ws.append([_clean_value(row.get(header)) for header in headers])


def _write_calibration_sheet(ws, calibration: Dict[str, Any] | None) -> None:
    targets = calibration.get("targets", []) if calibration else []
    if not targets:
        ws.append(["Section", "Details"])
        ws.append(["Status", "No calibration performed"])
        return

    summary_header = [
        "Target",
        "Status",
        "Slope",
        "Intercept",
        "R^2",
        "Residual Std",
        "LOD",
        "LOQ",
        "Points",
    ]
    ws.append(summary_header)
    for target in targets:
        fit = target.get("fit") or {}
        ws.append(
            [
                _clean_value(target.get("name")),
                _clean_value(target.get("status")),
                _clean_value(fit.get("slope")),
                _clean_value(fit.get("intercept")),
                _clean_value(fit.get("r_squared")),
                _clean_value(fit.get("residual_std")),
                _clean_value(fit.get("lod")),
                _clean_value(fit.get("loq")),
                _clean_value(fit.get("points")),
            ]
        )

    for target in targets:
        ws.append([])
        ws.append([f"Target: {target.get('name')}"])
        for error in target.get("errors", []):
            ws.append(["Error", _clean_value(error)])
        for warning in target.get("warnings", []):
            ws.append(["Warning", _clean_value(warning)])

        standards = target.get("standards") or []
        if standards:
            ws.append(["Standards"])
            std_header = [
                "Sample ID",
                "Concentration",
                "Response (A/cm)",
                "Raw Absorbance",
                "Predicted Response",
                "Residual",
                "Included",
                "Replicates",
                "Pathlength (cm)",
            ]
            ws.append(std_header)
            for entry in standards:
                ws.append(
                    [
                        _clean_value(entry.get("sample_id")),
                        _clean_value(entry.get("concentration")),
                        _clean_value(entry.get("response")),
                        _clean_value(entry.get("raw_absorbance")),
                        _clean_value(entry.get("predicted_response")),
                        _clean_value(entry.get("residual")),
                        _clean_value(entry.get("included")),
                        _clean_value(entry.get("replicates")),
                        _clean_value(entry.get("pathlength_cm")),
                    ]
                )
        else:
            ws.append(["Standards", "None"])

        unknowns = target.get("unknowns") or []
        if unknowns:
            ws.append(["Unknowns"])
            unk_header = [
                "Sample ID",
                "Response (A/cm)",
                "Raw Absorbance",
                "Predicted Concentration",
                "Status",
                "Within Range",
                "LOD/LOQ Flag",
                "Replicates",
                "Pathlength (cm)",
                "Dilution Factor",
            ]
            ws.append(unk_header)
            for entry in unknowns:
                ws.append(
                    [
                        _clean_value(entry.get("sample_id")),
                        _clean_value(entry.get("response")),
                        _clean_value(entry.get("raw_absorbance")),
                        _clean_value(entry.get("predicted_concentration")),
                        _clean_value(entry.get("status")),
                        _clean_value(entry.get("within_range")),
                        _clean_value(entry.get("lod_flag")),
                        _clean_value(entry.get("replicates")),
                        _clean_value(entry.get("pathlength_cm")),
                        _clean_value(entry.get("dilution_factor")),
                    ]
                )
        else:
            ws.append(["Unknowns", "None"])


def write_workbook(out_path: str, processed, qc_table, audit, figures, calibration=None):
    workbook_path = Path(out_path)
    _ensure_parent(workbook_path)

    wb = Workbook()
    ws_processed = wb.active
    ws_processed.title = "Processed_Spectra"
    ws_processed.append([
        "spectrum_index",
        "sample_id",
        "role",
        "mode",
        "channel",
        "wavelength",
        "value",
    ])
    for row in _processed_rows(processed):
        ws_processed.append(row)

    ws_metadata = wb.create_sheet("Metadata")
    _write_dict_rows(ws_metadata, _metadata_rows(processed))

    ws_qc = wb.create_sheet("QC_Flags")
    _write_dict_rows(ws_qc, _qc_rows(qc_table))

    ws_calibration = wb.create_sheet("Calibration")
    _write_calibration_sheet(ws_calibration, calibration)

    ws_audit = wb.create_sheet("Audit_Log")
    ws_audit.append(["Index", "Entry"])
    for idx, entry in enumerate(audit or [], start=1):
        ws_audit.append([idx, _clean_value(entry)])

    wb.save(workbook_path)
    return str(workbook_path)

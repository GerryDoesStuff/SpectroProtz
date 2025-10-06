from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from spectro_app.engine.io_common import sniff_locale
from .conversions import convert_intensity_to_absorbance, normalise_mode


KEY_ALIASES = {
    "instrument": "instrument",
    "instrument name": "instrument",
    "model": "instrument_model",
    "model name": "instrument_model",
    "serial": "instrument_serial",
    "serial number": "instrument_serial",
    "mode": "mode_raw",
    "measurement mode": "mode_raw",
    "method": "mode_raw",
    "pathlength": "pathlength_raw",
    "path length": "pathlength_raw",
    "cell length": "pathlength_raw",
    "cell pathlength": "pathlength_raw",
    "sample id": "sample_id",
    "sample": "sample_id",
    "sample name": "sample_id",
    "reference id": "blank_id",
    "blank": "blank_id",
    "blank id": "blank_id",
    "operator": "operator",
    "user": "operator",
    "acquired by": "operator",
    "scan speed": "scan_speed",
    "scan rate": "scan_speed",
    "averages": "averages",
    "# averages": "averages",
    "averaging": "averages",
    "bandwidth": "bandwidth_raw",
    "slit width": "bandwidth_raw",
    "data": "datetime_raw",
    "date": "date_raw",
    "time": "time_raw",
    "timestamp": "datetime_raw",
    "comment": "comment",
}


HELIOS_KEYWORDS = (
    "helios",
    "gamma",
    "sample id",
    "reference id",
    "scan speed",
    "pathlength",
    "measurement mode",
)


def is_helios_file(path: Path | str, sample: Optional[str] = None) -> bool:
    """Best-effort heuristic to identify Helios Gamma exports."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dsp":
        return True

    if sample is None:
        try:
            if suffix in {".csv", ".txt"}:
                sample = path.read_text(encoding="utf-8", errors="ignore")[:4000]
            elif suffix in {".xls", ".xlsx"}:
                peek = pd.read_excel(path, nrows=6, header=None)
                sample = "\n".join(
                    " ".join(str(v).lower() for v in row if not pd.isna(v))
                    for row in peek.to_numpy()
                )
            else:
                sample = ""
        except Exception:
            sample = ""

    lowered = (sample or "").lower()
    return any(keyword in lowered for keyword in HELIOS_KEYWORDS)


def read_helios(path: Path | str) -> List[Dict[str, Any]]:
    """Parse Helios Gamma CSV/Excel exports into tabular spectra."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".dsp"}:
        records = _read_helios_text(path)
    elif suffix in {".xls", ".xlsx"}:
        records = _read_helios_excel(path)
    else:
        raise ValueError(f"Unsupported Helios file type: {suffix}")

    for rec in records:
        meta = rec.setdefault("meta", {})
        meta.setdefault("source_type", "helios_gamma")
        meta.setdefault("instrument", "Helios Gamma")
        meta.setdefault("technique", "uvvis")
        meta.setdefault("source_file", str(path))
    return records


def parse_metadata_lines(lines: Iterable[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    kv_pairs: Dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip().strip("#")
        if not line:
            continue
        for sep in (":", "=", ";", "\t"):
            if sep in line:
                left, right = line.split(sep, 1)
                key = left.strip().lower()
                value = right.strip()
                if key:
                    kv_pairs[key] = value
                break

    for key, value in kv_pairs.items():
        mapped = KEY_ALIASES.get(key, key)
        if mapped == "mode_raw":
            meta["mode"] = _normalise_mode(value)
        elif mapped == "pathlength_raw":
            meta["pathlength_cm"] = _parse_pathlength(value)
        elif mapped == "bandwidth_raw":
            meta["bandwidth_nm"] = _parse_float(value)
        elif mapped == "averages":
            parsed = _parse_float(value)
            if parsed is not None:
                meta["averages"] = int(round(parsed))
        elif mapped == "datetime_raw":
            parsed = _parse_datetime(value)
            if parsed:
                meta["acquired_datetime"] = parsed.isoformat()
        elif mapped == "date_raw":
            parsed = _parse_datetime(value)
            if parsed:
                existing = meta.get("acquired_datetime")
                if existing:
                    # Merge only if time missing; otherwise prefer original.
                    dt = datetime.fromisoformat(existing)
                    meta["acquired_datetime"] = datetime.combine(
                        parsed.date(), dt.time()
                    ).isoformat()
                else:
                    meta["acquired_datetime"] = parsed.date().isoformat()
        elif mapped == "time_raw":
            parsed = _parse_datetime(value)
            if parsed:
                existing = meta.get("acquired_datetime")
                time_str = parsed.time().isoformat()
                if existing and len(existing) > 10:
                    dt = datetime.fromisoformat(existing)
                    meta["acquired_datetime"] = datetime.combine(
                        dt.date(), parsed.time()
                    ).isoformat()
                else:
                    meta["acquired_time"] = time_str
        else:
            meta[mapped] = value

    if "acquired_datetime" not in meta:
        # Compose from separate date/time entries if both were provided.
        date_part = kv_pairs.get("date") or kv_pairs.get("data")
        time_part = kv_pairs.get("time")
        if date_part or time_part:
            parsed = _compose_datetime(date_part, time_part)
            if parsed:
                meta["acquired_datetime"] = parsed.isoformat()

    return meta


def _read_helios_text(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".dsp":
        parsed = _parse_visionlite_dsp(path, text)
        if parsed is not None:
            return parsed

    locale = sniff_locale(text)
    lines = text.splitlines()
    header_idx, has_header, meta_lines = _locate_table(
        lines, locale["delimiter"], locale["decimal"]
    )
    data_str = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(data_str),
        sep=locale["delimiter"] if locale["delimiter"] else None,
        decimal=locale["decimal"],
        engine="python",
        header=0 if has_header else None,
    )
    if not has_header:
        df_columns = [f"col_{idx}" for idx in range(df.shape[1])]
        df.columns = df_columns

    meta = parse_metadata_lines(meta_lines)
    return _dataframe_to_records(df, meta)


def _parse_visionlite_dsp(path: Path, text: str) -> Optional[List[Dict[str, Any]]]:
    if "#DATA" not in text:
        return None

    try:
        header, data_section = text.split("#DATA", 1)
    except ValueError:
        return None

    data_values: List[float] = []
    for raw in data_section.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed = _parse_float(stripped)
        if parsed is None:
            continue
        data_values.append(parsed)

    if not data_values:
        return None

    intensity = np.asarray(data_values, dtype=float)

    metadata_blocks: List[List[str]] = []
    current_block: List[str] = []
    for raw_line in header.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#SPECIFIC"):
            break
        if stripped:
            current_block.append(stripped)
        elif current_block:
            metadata_blocks.append(current_block)
            current_block = []
    if current_block:
        metadata_blocks.append(current_block)

    if not metadata_blocks:
        return None

    primary = metadata_blocks[0]
    sample_token = primary[3] if len(primary) > 3 else path.name
    sample_name = Path(sample_token).stem or path.stem

    start_nm = _parse_float(primary[5]) if len(primary) > 5 else None
    end_nm = _parse_float(primary[6]) if len(primary) > 6 else None
    step_nm = _parse_float(primary[7]) if len(primary) > 7 else None
    point_count_float = _parse_float(primary[8]) if len(primary) > 8 else None
    point_count = int(round(point_count_float)) if point_count_float else None
    if point_count is None or point_count <= 0:
        point_count = intensity.size
    if point_count != intensity.size:
        point_count = intensity.size

    if step_nm is None and start_nm is not None and end_nm is not None and point_count > 1:
        step_nm = (end_nm - start_nm) / float(point_count - 1)
    if step_nm is None or step_nm == 0:
        step_nm = 1.0
    if start_nm is None:
        start_nm = 0.0
    if end_nm is None:
        end_nm = start_nm + step_nm * float(max(point_count - 1, 0))

    wavelength = start_nm + np.arange(intensity.size, dtype=float) * step_nm

    mode_token = primary[9] if len(primary) > 9 else None
    mapped_mode = _parse_visionlite_mode(mode_token)

    meta: Dict[str, Any] = {
        "channel": sample_name,
        "sample_id": sample_name,
        "role": "sample",
        "wavelength_start_nm": start_nm,
        "wavelength_end_nm": end_nm,
        "wavelength_step_nm": step_nm,
        "points": intensity.size,
        "source_type": "visionlite_dsp",
        "instrument": "Helios Gamma",
    }

    if mapped_mode:
        normalised = normalise_mode(mapped_mode)
        if normalised:
            meta["mode"] = normalised
        else:
            meta["mode_raw"] = mapped_mode
    elif mode_token:
        meta["mode_raw"] = mode_token

    if len(primary) > 13 and primary[13]:
        meta["operator"] = primary[13]

    if len(primary) > 4 and primary[4]:
        meta["wavelength_units"] = primary[4]

    if len(metadata_blocks) > 1:
        timeline = metadata_blocks[1]
        labels = ("acquired_datetime", "saved_datetime", "created_datetime")
        for label, value in zip(labels, timeline):
            parsed_dt = _parse_visionlite_datetime(value) or _parse_datetime(value)
            if parsed_dt:
                meta[label] = parsed_dt.isoformat()
        for extra in timeline[len(labels):]:
            lowered = extra.lower()
            if lowered.startswith("version"):
                meta["file_version"] = extra.split(",", 1)[0].strip()
            if "method" in lowered:
                method_part = extra.split("Method:", 1)[-1].strip().strip("|")
                if method_part:
                    meta["method"] = method_part

    if len(metadata_blocks) > 3:
        instrument_block = metadata_blocks[3]
        if instrument_block:
            meta.setdefault("instrument_model", instrument_block[0])
        if len(instrument_block) > 1 and instrument_block[1]:
            meta["instrument_firmware"] = instrument_block[1]
        if len(instrument_block) > 2 and instrument_block[2]:
            meta["instrument_serial"] = instrument_block[2]

    if len(metadata_blocks) > 4 and metadata_blocks[4]:
        software_block = metadata_blocks[4]
        slot_value = _parse_float(software_block[0]) if software_block else None
        if slot_value is not None:
            rounded = int(round(slot_value))
            meta["cuvette_slot"] = rounded if abs(slot_value - rounded) < 1e-6 else slot_value
        meta["software"] = software_block[-1]

    raw_trace = intensity.copy()
    converted, channel_key, updated_mode, original_mode = convert_intensity_to_absorbance(
        intensity, meta.get("mode")
    )
    intensity = converted
    if channel_key:
        channels = {channel_key: raw_trace}
        meta["channels"] = channels
        if original_mode:
            meta["original_mode"] = original_mode
    if updated_mode:
        meta["mode"] = updated_mode

    record = {
        "wavelength": wavelength,
        "intensity": intensity,
        "meta": meta,
    }
    return [record]


def _read_helios_excel(path: Path) -> List[Dict[str, Any]]:
    raw = pd.read_excel(path, header=None)
    rows_as_text = [
        "\t".join(str(v) for v in row if not pd.isna(v))
        for row in raw.itertuples(index=False, name=None)
    ]
    header_idx, has_header, meta_lines = _locate_table(rows_as_text, "\t", ".")
    if has_header:
        df = pd.read_excel(path, header=header_idx)
    else:
        df = pd.read_excel(path, header=None, skiprows=header_idx)
        df.columns = [f"col_{idx}" for idx in range(df.shape[1])]

    df = df.dropna(axis=1, how="all")
    meta = parse_metadata_lines(meta_lines)
    return _dataframe_to_records(df, meta)


def _dataframe_to_records(df: pd.DataFrame, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = df.dropna(how="all")
    df.columns = [str(col).strip() for col in df.columns]
    if df.shape[1] < 2:
        raise ValueError("Helios export must contain wavelength and at least one intensity column")

    base_meta = dict(base_meta)
    mode = normalise_mode(base_meta.get("mode"))
    if mode:
        base_meta["mode"] = mode

    wavelength = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    records: List[Dict[str, Any]] = []
    for idx, column in enumerate(df.columns[1:], start=1):
        intensity = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        mask = wavelength.notna() & intensity.notna()
        wl = wavelength[mask].to_numpy(dtype=float)
        inten = intensity[mask].to_numpy(dtype=float)
        if wl.size == 0:
            continue
        column_meta = dict(base_meta)
        column_meta.setdefault("channel", column)
        # Sample/blank hints fall back to column naming when explicit IDs absent.
        if "sample_id" not in column_meta or not column_meta["sample_id"]:
            column_meta["sample_id"] = column
        role = _infer_role(column, column_meta)
        column_meta.setdefault("role", role)
        if role == "blank":
            column_meta.setdefault("blank_id", column_meta.get("sample_id", column))
        raw_trace = inten.copy()
        converted, channel_key, updated_mode, original_mode = convert_intensity_to_absorbance(
            inten, column_meta.get("mode")
        )
        inten = converted
        if channel_key:
            channels = dict(column_meta.get("channels") or {})
            channels[channel_key] = raw_trace
            column_meta["channels"] = channels
            if original_mode:
                column_meta.setdefault("original_mode", original_mode)
        if updated_mode:
            column_meta["mode"] = updated_mode
        records.append(
            {
                "wavelength": wl,
                "intensity": inten,
                "meta": column_meta,
            }
        )
    return records


def _locate_table(lines: Sequence[str], delimiter: Optional[str], decimal: str) -> tuple[int, bool, List[str]]:
    numeric_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = _tokenise(stripped, delimiter)
        numeric_tokens = sum(1 for tok in tokens if _is_number(tok, decimal))
        if numeric_tokens >= 2:
            numeric_idx = idx
            break
    if numeric_idx is None:
        raise ValueError("Unable to locate numeric data rows in Helios export")

    header_idx = numeric_idx
    if numeric_idx > 0:
        prev_tokens = _tokenise(lines[numeric_idx - 1], delimiter)
        if prev_tokens and not all(_is_number(tok, decimal) for tok in prev_tokens):
            header_idx = numeric_idx - 1

    meta_lines = [line for line in lines[:header_idx] if line and line.strip()]
    has_header = header_idx != numeric_idx
    return header_idx, has_header, meta_lines


def _tokenise(line: str, delimiter: Optional[str]) -> List[str]:
    if delimiter and delimiter.strip():
        return [token.strip() for token in line.split(delimiter)]
    return line.split()


def _is_number(token: str, decimal: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if decimal != ".":
        token = token.replace(decimal, ".")
    token = token.replace(" ", "")
    try:
        float(token)
        return True
    except ValueError:
        return False


def _normalise_mode(value: str | None) -> Optional[str]:
    return normalise_mode(value)


def _parse_pathlength(value: str | None) -> Optional[float]:
    if not value:
        return None
    match = re.search(r"([\d.,]+)\s*(mm|cm|m|µm|um)?", value, re.IGNORECASE)
    if not match:
        return _parse_float(value)
    magnitude = _parse_float(match.group(1))
    if magnitude is None:
        return None
    unit = (match.group(2) or "cm").lower()
    if unit == "cm":
        return magnitude
    if unit == "mm":
        return magnitude / 10.0
    if unit in {"m"}:
        return magnitude * 100.0
    if unit in {"µm", "um"}:
        return magnitude / 10000.0
    return magnitude


def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("µ", "u").replace(" ", "")
    if cleaned.count(",") > cleaned.count("."):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_visionlite_mode(value: str | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    mapping = {
        "A": "absorbance",
        "ABS": "absorbance",
        "ABSORBANCE": "absorbance",
        "%T": "transmittance",
        "T": "transmittance",
        "TRANS": "transmittance",
        "TRANSMITTANCE": "transmittance",
        "%R": "reflectance",
        "R": "reflectance",
        "REFLECTANCE": "reflectance",
        "F": "fluorescence",
        "FLUORESCENCE": "fluorescence",
        "E": "emission",
        "EMISSION": "emission",
    }
    return mapping.get(upper)


def _parse_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for dayfirst in (True, False):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    return None


def _compose_datetime(date_value: Optional[str], time_value: Optional[str]) -> Optional[datetime]:
    date_part = _parse_datetime(date_value)
    time_part = _parse_datetime(time_value)
    if date_part and time_part:
        return datetime.combine(date_part.date(), time_part.time())
    return date_part or time_part


def _parse_visionlite_datetime(value: str | None) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    patterns = (
        "%Y.%m.%d. %H:%M:%S",
        "%Y.%m.%d.%H:%M:%S",
        "%Y.%m.%d.",
    )
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _infer_role(column: str, meta: Dict[str, Any]) -> str:
    column_lower = column.lower()
    blank_id = str(meta.get("blank_id", "")).lower()
    if "blank" in column_lower or "reference" in column_lower:
        return "blank"
    if blank_id and blank_id in column_lower:
        return "blank"
    return "sample"

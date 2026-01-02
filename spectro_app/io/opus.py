from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from struct import error as struct_error
from struct import unpack
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from spectro_app.engine.plugin_api import Spectrum

HEADER_LEN = 504
FIRST_CURSOR_POSITION = 24
META_BLOCK_SIZE = 12

UNSIGNED_INT = "<I"
UNSIGNED_CHAR = "<B"
UNSIGNED_SHORT = "<H"
INT = "<i"
DOUBLE = "<d"

ENCODING_LATIN = "latin-1"
ENCODING_UTF = "utf-8"
NULL_BYTE = b"\x00"
NULL_STR = "\x00"

PARAM_TYPES = {0: "int", 1: "float", 2: "str", 3: "str", 4: "str"}

SERIES_BLOCKS = ("AB", "ScSm", "IgSm", "PhSm", "ScRf", "IgRf")


class UnknownBlockType(Exception):
    pass


def opus_optional_reader_status() -> Dict[str, bool]:
    return {
        "spectrochempy": find_spec("spectrochempy") is not None,
        "brukeropusreader": find_spec("brukeropusreader") is not None,
    }


@dataclass(frozen=True)
class OpusBlockMeta:
    data_type: int
    channel_type: int
    text_type: int
    chunk_size: int
    offset: int


def _read_data_type(header: bytes, cursor: int) -> int:
    return unpack(UNSIGNED_CHAR, header[cursor : cursor + 1])[0]


def _read_channel_type(header: bytes, cursor: int) -> int:
    return unpack(UNSIGNED_CHAR, header[cursor + 1 : cursor + 2])[0]


def _read_text_type(header: bytes, cursor: int) -> int:
    return unpack(UNSIGNED_CHAR, header[cursor + 2 : cursor + 3])[0]


def _read_chunk_size(header: bytes, cursor: int) -> int:
    return unpack(UNSIGNED_INT, header[cursor + 4 : cursor + 8])[0]


def _read_offset(header: bytes, cursor: int) -> int:
    return unpack(UNSIGNED_INT, header[cursor + 8 : cursor + 12])[0]


def _read_chunk(data: bytes, block_meta: OpusBlockMeta) -> bytes:
    start = block_meta.offset
    end = start + 4 * block_meta.chunk_size
    return data[start:end]


def _parse_param(data: bytes, block_meta: OpusBlockMeta) -> Dict[str, object]:
    chunk = _read_chunk(data, block_meta)
    cursor = 0
    params: Dict[str, object] = {}
    while cursor + 3 <= len(chunk):
        param_name = chunk[cursor : cursor + 3].decode(ENCODING_UTF, errors="ignore")
        if param_name == "END":
            break
        if cursor + 8 > len(chunk):
            break
        type_index = unpack(UNSIGNED_SHORT, chunk[cursor + 4 : cursor + 6])[0]
        param_type = PARAM_TYPES.get(type_index, "raw")
        param_size = unpack(UNSIGNED_SHORT, chunk[cursor + 6 : cursor + 8])[0]
        param_bytes = chunk[cursor + 8 : cursor + 8 + 2 * param_size]
        if param_type == "int":
            try:
                param_val = unpack(INT, param_bytes)[0]
            except struct_error:
                break
        elif param_type == "float":
            param_val = unpack(DOUBLE, param_bytes)[0]
        elif param_type == "str":
            p_end = param_bytes.find(NULL_BYTE)
            if p_end == -1:
                p_end = len(param_bytes)
            param_val = param_bytes[:p_end].decode(ENCODING_LATIN, errors="ignore")
        else:
            param_val = param_bytes
        params[param_name] = param_val
        cursor = cursor + 8 + 2 * param_size
    return params


def _parse_text(data: bytes, block_meta: OpusBlockMeta) -> str:
    chunk = _read_chunk(data, block_meta)
    return chunk.decode(ENCODING_LATIN, errors="ignore").strip(NULL_STR)


def _parse_series(data: bytes, block_meta: OpusBlockMeta) -> np.ndarray:
    chunk = _read_chunk(data, block_meta)
    fmt = f"<{block_meta.chunk_size}f"
    values = unpack(fmt, chunk)
    return np.array(values, dtype=float)


BLOCK_0 = {
    8: ("Info Block", _parse_param),
    104: ("History", _parse_text),
    152: ("Curve Fit", _parse_text),
    168: ("Signature", _parse_text),
    240: ("Integration Method", _parse_text),
}

BLOCK_7 = {4: "ScSm", 8: "IgSm", 12: "PhSm"}
BLOCK_11 = {4: "ScRf", 8: "IgRf"}
BLOCK_23 = {4: "ScSm Data Parameter", 8: "IgSm Data Parameter", 12: "PhSm Data Parameter"}
BLOCK_27 = {4: "ScRf Data Parameter", 8: "IgRf Data Parameter"}

DIFFERENT_BLOCKS = {
    31: "AB Data Parameter",
    32: "Instrument",
    40: "Instrument (Rf)",
    48: "Acquisition",
    56: "Acquisition (Rf)",
    64: "Fourier Transformation",
    72: "Fourier Transformation (Rf)",
    96: "Optik",
    104: "Optik (Rf)",
    160: "Sample",
}


def _get_block_parser(block_meta: OpusBlockMeta) -> Tuple[str, Callable[[bytes, OpusBlockMeta], object]]:
    if block_meta.data_type == 0:
        return BLOCK_0.get(block_meta.text_type, ("Text Information", _parse_text))
    if block_meta.data_type == 7:
        return BLOCK_7[block_meta.channel_type], _parse_series
    if block_meta.data_type == 11:
        return BLOCK_11[block_meta.channel_type], _parse_series
    if block_meta.data_type == 15:
        return "AB", _parse_series
    if block_meta.data_type == 23:
        return BLOCK_23[block_meta.channel_type], _parse_param
    if block_meta.data_type == 27:
        return BLOCK_27[block_meta.channel_type], _parse_param
    if block_meta.data_type in DIFFERENT_BLOCKS:
        return DIFFERENT_BLOCKS[block_meta.data_type], _parse_param
    raise UnknownBlockType()


def _parse_meta(data: bytes) -> List[OpusBlockMeta]:
    header = data[:HEADER_LEN]
    blocks: List[OpusBlockMeta] = []
    cursor = FIRST_CURSOR_POSITION
    while cursor + META_BLOCK_SIZE <= HEADER_LEN:
        data_type = _read_data_type(header, cursor)
        channel_type = _read_channel_type(header, cursor)
        text_type = _read_text_type(header, cursor)
        chunk_size = _read_chunk_size(header, cursor)
        offset = _read_offset(header, cursor)
        if offset <= 0:
            break
        blocks.append(
            OpusBlockMeta(
                data_type=data_type,
                channel_type=channel_type,
                text_type=text_type,
                chunk_size=chunk_size,
                offset=offset,
            )
        )
        next_offset = offset + 4 * chunk_size
        if next_offset >= len(data):
            break
        cursor += META_BLOCK_SIZE
    return blocks


def _parse_data(data: bytes, blocks: Iterable[OpusBlockMeta]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for block_meta in blocks:
        try:
            name, parser = _get_block_parser(block_meta)
        except UnknownBlockType:
            continue
        parsed[name] = parser(data, block_meta)
    return parsed


def read_opus(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    data = path.read_bytes()
    blocks = _parse_meta(data)
    return _parse_data(data, blocks)


def _axis_from_params(params: Dict[str, object], count: int) -> np.ndarray:
    fxv = params.get("FXV")
    lxv = params.get("LXV")
    if isinstance(fxv, (int, float)) and isinstance(lxv, (int, float)):
        return np.linspace(float(fxv), float(lxv), count)
    return np.arange(count, dtype=float)


def read_opus_records(path: str | Path) -> List[Dict[str, object]]:
    data = read_opus(path)
    records: List[Dict[str, object]] = []
    sample_meta = data.get("Sample")
    for series_name in SERIES_BLOCKS:
        if series_name not in data:
            continue
        intensity = np.asarray(data[series_name], dtype=float)
        params = data.get(f"{series_name} Data Parameter", {})
        axis = _axis_from_params(params if isinstance(params, dict) else {}, intensity.size)
        meta: Dict[str, object] = {
            "source": "opus",
            "source_file": str(path),
            "axis_key": "wavenumber",
            "axis_unit": "cm^-1",
            "opus_series": series_name,
        }
        if isinstance(sample_meta, dict) and sample_meta:
            meta["sample"] = dict(sample_meta)
        records.append({"wavelength": axis, "intensity": intensity, "meta": meta})
    if not records:
        raise ValueError("No spectrum blocks found in OPUS file.")
    return records


def _normalize_axis_unit(axis_unit: object | None) -> str | None:
    if axis_unit is None:
        return None
    if isinstance(axis_unit, str):
        return axis_unit
    try:
        return str(axis_unit)
    except Exception:
        return None


def _records_from_spectrochempy(dataset: object, path: str | Path) -> List[Dict[str, object]]:
    def iter_datasets(value: object) -> Iterable[object]:
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    records: List[Dict[str, object]] = []
    for entry in iter_datasets(dataset):
        if entry is None:
            continue
        intensity = getattr(entry, "data", entry)
        intensity = np.asarray(intensity, dtype=float)

        axis = None
        axis_unit = None
        axis_source = getattr(entry, "x", None)
        if axis_source is None:
            axes = getattr(entry, "axes", None)
            if axes:
                axis_source = axes[0]
        if axis_source is None:
            axis_source = getattr(entry, "axis", None)
        if axis_source is not None:
            axis = getattr(axis_source, "data", axis_source)
            axis_unit = _normalize_axis_unit(getattr(axis_source, "units", None))

        if axis is None:
            axis = np.arange(intensity.size, dtype=float)

        meta: Dict[str, object] = {
            "source": "opus",
            "source_file": str(path),
            "axis_key": "wavenumber",
            "axis_unit": axis_unit or "cm^-1",
            "external_reader": "spectrochempy",
        }
        extra_meta = getattr(entry, "meta", None)
        if extra_meta is None:
            extra_meta = getattr(entry, "metadata", None)
        if isinstance(extra_meta, dict) and extra_meta:
            meta["external_meta"] = dict(extra_meta)
        records.append({"wavelength": np.asarray(axis, dtype=float), "intensity": intensity, "meta": meta})
    return records


def _records_from_brukeropusreader(data: object, path: str | Path) -> List[Dict[str, object]]:
    def record_from_mapping(mapping: Dict[str, object]) -> Dict[str, object] | None:
        intensity = (
            mapping.get("y")
            or mapping.get("intensity")
            or mapping.get("spectrum")
            or mapping.get("spec")
            or mapping.get("data")
        )
        if intensity is None:
            return None
        axis = mapping.get("x") or mapping.get("wavenumber") or mapping.get("wavelength") or mapping.get("axis")
        if axis is None:
            axis = np.arange(len(intensity), dtype=float)
        axis_unit = _normalize_axis_unit(
            mapping.get("axis_unit") or mapping.get("x_unit") or mapping.get("xunit")
        )
        meta: Dict[str, object] = {
            "source": "opus",
            "source_file": str(path),
            "axis_key": "wavenumber",
            "axis_unit": axis_unit or "cm^-1",
            "external_reader": "brukeropusreader",
        }
        extra_meta = mapping.get("meta") or mapping.get("metadata") or mapping.get("params")
        if isinstance(extra_meta, dict) and extra_meta:
            meta["external_meta"] = dict(extra_meta)
        return {
            "wavelength": np.asarray(axis, dtype=float),
            "intensity": np.asarray(intensity, dtype=float),
            "meta": meta,
        }

    records: List[Dict[str, object]] = []
    if isinstance(data, dict):
        spectra = data.get("spectra")
        if isinstance(spectra, list):
            for entry in spectra:
                if isinstance(entry, dict):
                    record = record_from_mapping(entry)
                    if record:
                        records.append(record)
        record = record_from_mapping(data)
        if record:
            records.append(record)
    elif isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, dict):
                record = record_from_mapping(entry)
                if record:
                    records.append(record)
    return records


def read_opus_records_external(path: str | Path) -> List[Dict[str, object]]:
    errors: List[str] = []
    try:
        from spectrochempy import read_opus as scp_read_opus
    except Exception as exc:
        errors.append(f"spectrochempy.read_opus not available: {exc}")
    else:
        try:
            records = _records_from_spectrochempy(scp_read_opus(str(path)), path)
            if records:
                return records
        except Exception as exc:
            errors.append(f"spectrochempy.read_opus failed: {exc}")

    try:
        from brukeropusreader import read_file as bruker_read_file
    except Exception as exc:
        errors.append(f"brukeropusreader.read_file not available: {exc}")
    else:
        try:
            records = _records_from_brukeropusreader(bruker_read_file(str(path)), path)
            if records:
                return records
        except Exception as exc:
            errors.append(f"brukeropusreader.read_file failed: {exc}")

    if errors:
        error_message = "External OPUS readers unavailable or failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        raise ValueError(error_message)
    return []


def load_opus_spectra(path: str | Path, *, technique: str | None = None) -> List[Spectrum]:
    try:
        records = read_opus_records_external(path)
    except ValueError:
        records = []
    if not records:
        records = read_opus_records(path)
    spectra: List[Spectrum] = []
    for record in records:
        meta = dict(record.get("meta", {}))
        if technique:
            meta.setdefault("technique", technique)
        spectra.append(
            Spectrum(
                wavelength=np.asarray(record["wavelength"], dtype=float),
                intensity=np.asarray(record["intensity"], dtype=float),
                meta=meta,
            )
        )
    return spectra

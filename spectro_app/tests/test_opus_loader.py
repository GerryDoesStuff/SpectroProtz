from __future__ import annotations

import struct

import numpy as np

from spectro_app.io import opus as opus_reader


def _pack_param(name: str, type_index: int, param_bytes: bytes) -> bytes:
    param_size = len(param_bytes) // 2
    return (
        name.encode("ascii")
        + b"\x00"
        + struct.pack("<H", type_index)
        + struct.pack("<H", param_size)
        + param_bytes
    )


def _param_float(name: str, value: float) -> bytes:
    return _pack_param(name, 1, struct.pack("<d", value))


def _param_str(name: str, value: str) -> bytes:
    raw = value.encode("latin-1") + b"\x00"
    if len(raw) % 2:
        raw += b"\x00"
    return _pack_param(name, 2, raw)


def _param_end() -> bytes:
    return b"END" + b"\x00" + struct.pack("<H", 0) + struct.pack("<H", 0)


def _write_opus_fixture(path) -> None:
    header = bytearray(opus_reader.HEADER_LEN)
    blocks = []
    offset = opus_reader.HEADER_LEN

    def add_block(data_type: int, channel_type: int, text_type: int, payload: bytes) -> None:
        nonlocal offset
        chunk_size = len(payload) // 4
        blocks.append((data_type, channel_type, text_type, chunk_size, offset, payload))
        offset += len(payload)

    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="<f4").tobytes()
    add_block(15, 0, 0, series)

    params = _param_float("FXV", 4000.0) + _param_float("LXV", 400.0) + _param_end()
    add_block(31, 0, 0, params)

    sample = _param_str("SNA", "Test Sample") + _param_end()
    add_block(160, 0, 0, sample)

    cursor = opus_reader.FIRST_CURSOR_POSITION
    for data_type, channel_type, text_type, chunk_size, block_offset, _payload in blocks:
        header[cursor] = data_type
        header[cursor + 1] = channel_type
        header[cursor + 2] = text_type
        header[cursor + 4 : cursor + 8] = struct.pack("<I", chunk_size)
        header[cursor + 8 : cursor + 12] = struct.pack("<I", block_offset)
        cursor += opus_reader.META_BLOCK_SIZE

    path.write_bytes(header + b"".join(block[-1] for block in blocks))


def test_read_opus_records_axis_units_and_sample_meta(tmp_path) -> None:
    opus_path = tmp_path / "fixture.opus"
    _write_opus_fixture(opus_path)

    records = opus_reader.read_opus_records(opus_path)

    assert len(records) == 1
    record = records[0]
    np.testing.assert_allclose(record["wavelength"], np.linspace(4000.0, 400.0, 5))
    np.testing.assert_allclose(record["intensity"], np.array([1, 2, 3, 4, 5], dtype=float))
    meta = record["meta"]
    assert meta["axis_key"] == "wavenumber"
    assert meta["axis_unit"] == "cm^-1"
    assert meta["sample"]["SNA"] == "Test Sample"

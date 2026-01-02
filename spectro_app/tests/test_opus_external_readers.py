from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from spectro_app.io import opus as opus_reader


class _DummyAxis:
    def __init__(self, data: np.ndarray, units: str) -> None:
        self.data = data
        self.units = units


class _DummyDataset:
    def __init__(self) -> None:
        self.data = np.array([1.0, 2.0], dtype=float)
        self.x = _DummyAxis(np.array([100.0, 200.0], dtype=float), "cm^-1")
        self.meta = {"name": "fixture"}


def test_load_opus_spectra_prefers_spectrochempy(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    spectrochempy = ModuleType("spectrochempy")

    def read_opus(path: str) -> _DummyDataset:
        calls.append(("spectrochempy", path))
        return _DummyDataset()

    spectrochempy.read_opus = read_opus
    monkeypatch.setitem(__import__("sys").modules, "spectrochempy", spectrochempy)

    brukeropusreader = ModuleType("brukeropusreader")

    def read_file(path: str) -> dict[str, object]:
        calls.append(("brukeropusreader", path))
        return {}

    brukeropusreader.read_file = read_file
    monkeypatch.setitem(__import__("sys").modules, "brukeropusreader", brukeropusreader)

    def fail_internal(_: str) -> list[dict[str, object]]:
        raise AssertionError("internal parser should not be used when spectrochempy succeeds")

    monkeypatch.setattr(opus_reader, "read_opus_records", fail_internal)

    opus_path = tmp_path / "fixture.opus"
    opus_path.write_bytes(b"unused")

    spectra = opus_reader.load_opus_spectra(opus_path)

    assert calls == [("spectrochempy", str(opus_path))]
    assert len(spectra) == 1
    assert spectra[0].meta["external_reader"] == "spectrochempy"


def test_load_opus_spectra_error_when_all_readers_fail(monkeypatch, tmp_path) -> None:
    spectrochempy = ModuleType("spectrochempy")

    def read_opus(_: str) -> None:
        raise RuntimeError("spectrochempy boom")

    spectrochempy.read_opus = read_opus
    monkeypatch.setitem(__import__("sys").modules, "spectrochempy", spectrochempy)

    brukeropusreader = ModuleType("brukeropusreader")

    def read_file(_: str) -> None:
        raise RuntimeError("brukeropusreader boom")

    brukeropusreader.read_file = read_file
    monkeypatch.setitem(__import__("sys").modules, "brukeropusreader", brukeropusreader)

    opus_path = tmp_path / "broken.opus"
    opus_path.write_bytes(b"unused")

    with pytest.raises(ValueError) as excinfo:
        opus_reader.load_opus_spectra(opus_path)

    message = str(excinfo.value)
    assert "External OPUS readers unavailable or failed" in message


def test_load_opus_spectra_ftir_stops_on_spectrochempy_error(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    spectrochempy = ModuleType("spectrochempy")

    def read_opus(_: str) -> None:
        calls.append(("spectrochempy", "read_opus"))
        raise RuntimeError("spectrochempy boom")

    spectrochempy.read_opus = read_opus
    monkeypatch.setitem(__import__("sys").modules, "spectrochempy", spectrochempy)

    brukeropusreader = ModuleType("brukeropusreader")

    def read_file(_: str) -> None:
        calls.append(("brukeropusreader", "read_file"))
        return None

    brukeropusreader.read_file = read_file
    monkeypatch.setitem(__import__("sys").modules, "brukeropusreader", brukeropusreader)

    opus_path = tmp_path / "ftir-broken.opus"
    opus_path.write_bytes(b"unused")

    with pytest.raises(ValueError) as excinfo:
        opus_reader.load_opus_spectra(opus_path, technique="ftir")

    assert calls == [("spectrochempy", "read_opus")]
    assert "spectrochempy.read_opus failed: spectrochempy boom" in str(excinfo.value)

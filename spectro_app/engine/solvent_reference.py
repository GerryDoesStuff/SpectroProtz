from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import uuid4

import numpy as np

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.io.opus import load_opus_spectra

DEFAULT_REFERENCE_PATH = Path.home() / "SpectroApp" / "solvent_references.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tags(tags: Iterable[object]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for tag in tags:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _serialize_spectrum(spec: Spectrum) -> Dict[str, Any]:
    return {
        "wavelength": np.asarray(spec.wavelength, dtype=float).tolist(),
        "intensity": np.asarray(spec.intensity, dtype=float).tolist(),
        "meta": dict(spec.meta or {}),
    }


def _deserialize_spectrum(data: Mapping[str, Any]) -> Spectrum:
    return Spectrum(
        wavelength=np.asarray(data.get("wavelength") or [], dtype=float),
        intensity=np.asarray(data.get("intensity") or [], dtype=float),
        meta=dict(data.get("meta") or {}),
    )


def load_reference_spectrum(path: str | Path) -> Spectrum:
    suffix = Path(path).suffix.lower()
    if suffix != ".opus":
        raise ValueError("Solvent reference files must be OPUS (.opus) format.")
    spectra = load_opus_spectra(path, technique="ftir")
    if not spectra:
        raise ValueError("No spectra found in reference file.")
    spectrum = spectra[0]
    meta = dict(spectrum.meta or {})
    meta.setdefault("source_path", str(path))
    return Spectrum(
        wavelength=np.asarray(spectrum.wavelength, dtype=float),
        intensity=np.asarray(spectrum.intensity, dtype=float),
        meta=meta,
    )


@dataclass
class SolventReferenceEntry:
    id: str
    name: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spectrum: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None
    defaults: bool = False
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "spectrum": self.spectrum,
            "source_path": self.source_path,
            "defaults": bool(self.defaults),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolventReferenceEntry":
        return cls(
            id=str(data.get("id") or uuid4()),
            name=str(data.get("name") or ""),
            tags=_normalize_tags(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
            spectrum=dict(data.get("spectrum") or {}) or None,
            source_path=str(data.get("source_path") or "") or None,
            defaults=bool(data.get("defaults") or False),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


def build_reference_entry(
    spectrum: Spectrum,
    *,
    name: str,
    tags: Iterable[object] = (),
    metadata: Mapping[str, Any] | None = None,
    reference_id: str | None = None,
    source_path: str | None = None,
    defaults: bool = False,
) -> SolventReferenceEntry:
    entry_id = reference_id or str(uuid4())
    meta = dict(spectrum.meta or {})
    meta.setdefault("reference_id", entry_id)
    meta.setdefault("reference_name", name)
    meta.setdefault("reference_tags", _normalize_tags(tags))
    meta.setdefault("reference_metadata", dict(metadata or {}))
    meta.setdefault("role", "solvent_reference")
    if source_path:
        meta.setdefault("source_path", source_path)
    serialized = _serialize_spectrum(
        Spectrum(
            wavelength=np.asarray(spectrum.wavelength, dtype=float),
            intensity=np.asarray(spectrum.intensity, dtype=float),
            meta=meta,
        )
    )
    return SolventReferenceEntry(
        id=entry_id,
        name=name,
        tags=_normalize_tags(tags),
        metadata=dict(metadata or {}),
        spectrum=serialized,
        source_path=source_path,
        defaults=defaults,
    )


def build_reference_spectrum(entry: Mapping[str, Any]) -> Optional[Spectrum]:
    spectrum_payload = entry.get("spectrum")
    if not isinstance(spectrum_payload, Mapping):
        return None
    spectrum = _deserialize_spectrum(spectrum_payload)
    meta = dict(spectrum.meta or {})
    reference_id = entry.get("id")
    if reference_id:
        meta.setdefault("reference_id", str(reference_id))
    name = entry.get("name")
    if name:
        meta.setdefault("reference_name", str(name))
    tags = entry.get("tags")
    if tags:
        meta.setdefault("reference_tags", list(tags))
    metadata = entry.get("metadata")
    if metadata:
        meta.setdefault("reference_metadata", dict(metadata))
    source_path = entry.get("source_path")
    if source_path:
        meta.setdefault("source_path", str(source_path))
    meta.setdefault("role", "solvent_reference")
    return Spectrum(
        wavelength=np.asarray(spectrum.wavelength, dtype=float),
        intensity=np.asarray(spectrum.intensity, dtype=float),
        meta=meta,
    )


class SolventReferenceStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_REFERENCE_PATH

    def load(self) -> List[SolventReferenceEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        return [SolventReferenceEntry.from_dict(item) for item in payload]

    def save(self, entries: Iterable[SolventReferenceEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.to_dict() for entry in entries]
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def upsert(self, entry: SolventReferenceEntry) -> None:
        entries = self.load()
        updated = False
        for idx, existing in enumerate(entries):
            if existing.id == entry.id:
                entry.updated_at = _now_iso()
                entries[idx] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        self.save(entries)

    def get(self, entry_id: str) -> Optional[SolventReferenceEntry]:
        for entry in self.load():
            if entry.id == entry_id:
                return entry
        return None

    def set_default(self, entry_id: str, is_default: bool) -> None:
        entries = self.load()
        changed = False
        for entry in entries:
            if entry.id == entry_id:
                entry.defaults = bool(is_default)
                entry.updated_at = _now_iso()
                changed = True
        if changed:
            self.save(entries)

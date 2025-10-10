from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

__all__ = ["QueueEntry", "order_queue_entries"]


@dataclass
class QueueEntry:
    path: str
    display_name: str
    plugin_id: Optional[str] = None
    technique: Optional[str] = None
    mode: Optional[str] = None
    role: Optional[str] = None
    manifest_status: Optional[str] = None
    is_manifest: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)
    overrides: Dict[str, object] = field(default_factory=dict)


def _normalize_role_value(value: object) -> Optional[str]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return None


def _entry_role(entry: QueueEntry) -> Optional[str]:
    candidates = [entry.role]
    if isinstance(entry.metadata, dict):
        for key in ("role", "sample_role", "sample_type", "type"):
            candidates.append(entry.metadata.get(key))
    for candidate in candidates:
        normalized = _normalize_role_value(candidate)
        if normalized:
            return normalized
    return None


def _is_blank_like(entry: QueueEntry) -> bool:
    role = _entry_role(entry)
    if not role:
        return False
    if role.startswith("blank"):
        return True
    if role.startswith("ref") or "reference" in role:
        return True
    return False


def order_queue_entries(entries: Iterable[QueueEntry]) -> List[QueueEntry]:
    blank_like: List[QueueEntry] = []
    samples: List[QueueEntry] = []
    for entry in entries:
        if _is_blank_like(entry):
            blank_like.append(entry)
        else:
            samples.append(entry)
    return [*blank_like, *samples]

"""Shared helpers for FTIR metadata normalization."""

from __future__ import annotations

import re
from typing import Optional


def _normalize_molform(value: object | None) -> Optional[str]:
    """Return a compact molform string with whitespace removed."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    return re.sub(r"\s+", "", text)

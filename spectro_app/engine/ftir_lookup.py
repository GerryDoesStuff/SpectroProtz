"""Helpers for parsing FTIR lookup search text into DuckDB queries."""

from __future__ import annotations

import dataclasses
import re
import shlex
from typing import Dict, List, Tuple

from spectro_app.engine.ftir_index_schema import SPECTRA_COLUMNS

DEFAULT_TOLERANCE_CM = 5.0

_PEAK_RE = re.compile(
    r"^\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?:±|\+/-|\+-)\s*(?P<tol>\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"\s*(?:cm-1|cm\^-1|cm⁻¹)?\s*$"
)

_FILTER_RE = re.compile(r"^(?P<key>[A-Za-z][\w_-]*)\s*[:=]\s*(?P<value>.+)$")

_ALIASES: Dict[str, str] = {
    "id": "file_id",
    "file": "file_id",
    "path": "path",
    "name": "title",
    "title": "title",
    "origin": "origin",
    "owner": "owner",
    "cas": "cas",
    "names": "names",
    "formula": "molform",
    "molform": "molform",
    "source": "nist_source",
    "nist": "nist_source",
    "class": "class",
    "state": "state",
    "datatype": "data_type",
    "data_type": "data_type",
}

_VALID_COLUMNS = {col.lower(): col for col in SPECTRA_COLUMNS}


@dataclasses.dataclass(frozen=True)
class PeakCriterion:
    center: float
    tolerance: float

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.center - self.tolerance, self.center + self.tolerance)


@dataclasses.dataclass
class LookupCriteria:
    peaks: List[PeakCriterion]
    filters: Dict[str, List[str]]
    errors: List[str]

    @property
    def is_empty(self) -> bool:
        return not self.peaks and not self.filters


@dataclasses.dataclass(frozen=True)
class LookupQueries:
    spectra_sql: str
    spectra_params: List[object]
    peaks_sql: str
    peaks_params: List[object]


def parse_lookup_text(
    text: str,
    *,
    default_tolerance: float = DEFAULT_TOLERANCE_CM,
) -> LookupCriteria:
    peaks: List[PeakCriterion] = []
    filters: Dict[str, List[str]] = {}
    errors: List[str] = []

    raw = (text or "").strip()
    if not raw:
        return LookupCriteria(peaks=peaks, filters=filters, errors=errors)

    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return LookupCriteria(peaks=peaks, filters=filters, errors=[str(exc)])

    for token in tokens:
        if not token:
            continue
        filter_match = _FILTER_RE.match(token)
        if filter_match:
            key = _normalize_filter_key(filter_match.group("key"))
            value = filter_match.group("value").strip()
            if not value:
                errors.append(f"Filter '{token}' has no value.")
                continue
            if not key:
                errors.append(f"Unknown filter key '{filter_match.group('key')}'.")
                continue
            filters.setdefault(key, []).append(value)
            continue

        peak = _parse_peak_token(token, default_tolerance, errors)
        if peak is not None:
            peaks.append(peak)
        else:
            errors.append(f"Unrecognized token '{token}'.")

    return LookupCriteria(peaks=peaks, filters=filters, errors=errors)


def build_lookup_queries(criteria: LookupCriteria) -> LookupQueries:
    spectra_columns = ", ".join(_quote_ident(col) for col in SPECTRA_COLUMNS)
    spectra_sql = f"SELECT {spectra_columns} FROM spectra"
    peaks_sql = (
        "SELECT p.file_id, p.spectrum_id, p.peak_id, p.polarity, p.center, "
        "p.fwhm, p.amplitude, p.area, p.r2 "
        "FROM peaks p JOIN spectra s ON s.file_id = p.file_id"
    )

    if criteria.is_empty:
        empty_clause = " WHERE 1 = 0"
        return LookupQueries(
            spectra_sql=f"{spectra_sql}{empty_clause}",
            spectra_params=[],
            peaks_sql=f"{peaks_sql}{empty_clause}",
            peaks_params=[],
        )

    spectra_conditions: List[str] = []
    spectra_params: List[object] = []
    peaks_metadata_conditions: List[str] = []
    peaks_metadata_params: List[object] = []
    peaks_peak_conditions: List[str] = []
    peaks_peak_params: List[object] = []

    _apply_metadata_filters(
        criteria.filters,
        spectra_conditions,
        spectra_params,
        table_alias="spectra",
    )
    _apply_metadata_filters(
        criteria.filters,
        peaks_metadata_conditions,
        peaks_metadata_params,
        table_alias="s",
    )

    if criteria.peaks:
        for peak in criteria.peaks:
            lower, upper = peak.bounds
            spectra_conditions.append(
                "EXISTS (SELECT 1 FROM peaks p WHERE p.file_id = spectra.file_id "
                "AND p.center BETWEEN ? AND ?)"
            )
            spectra_params.extend([lower, upper])
            peaks_peak_conditions.append("p.center BETWEEN ? AND ?")
            peaks_peak_params.extend([lower, upper])

    if spectra_conditions:
        spectra_sql = f"{spectra_sql} WHERE " + " AND ".join(spectra_conditions)
    peaks_conditions: List[str] = []
    if peaks_metadata_conditions:
        peaks_conditions.extend(peaks_metadata_conditions)
    if peaks_peak_conditions:
        peaks_conditions.append("(" + " OR ".join(peaks_peak_conditions) + ")")
    if peaks_conditions:
        peaks_sql = f"{peaks_sql} WHERE " + " AND ".join(peaks_conditions)
    peaks_params = peaks_metadata_params + peaks_peak_params

    return LookupQueries(
        spectra_sql=spectra_sql,
        spectra_params=spectra_params,
        peaks_sql=peaks_sql,
        peaks_params=peaks_params,
    )


def _parse_peak_token(
    token: str,
    default_tolerance: float,
    errors: List[str],
) -> PeakCriterion | None:
    match = _PEAK_RE.match(token)
    if not match:
        return None
    try:
        value = float(match.group("value"))
    except (TypeError, ValueError):
        errors.append(f"Invalid peak value '{token}'.")
        return None
    tol_text = match.group("tol")
    if tol_text is None:
        tolerance = float(default_tolerance)
    else:
        try:
            tolerance = float(tol_text)
        except (TypeError, ValueError):
            errors.append(f"Invalid tolerance in '{token}'.")
            return None
    if tolerance < 0:
        errors.append(f"Negative tolerance in '{token}'.")
        return None
    return PeakCriterion(center=value, tolerance=tolerance)


def _normalize_filter_key(key: str) -> str | None:
    key_lower = key.strip().lower()
    mapped = _ALIASES.get(key_lower, key_lower)
    return _VALID_COLUMNS.get(mapped)


def _apply_metadata_filters(
    filters: Dict[str, List[str]],
    conditions: List[str],
    params: List[object],
    *,
    table_alias: str,
) -> None:
    for column, values in filters.items():
        if not values:
            continue
        qualified = f"{table_alias}.{_quote_ident(column)}"
        clauses = [f"CAST({qualified} AS VARCHAR) ILIKE ?" for _ in values]
        conditions.append("(" + " OR ".join(clauses) + ")")
        params.extend([f"%{value}%" for value in values])


def _quote_ident(identifier: str) -> str:
    return f'"{identifier.replace("\"", "\"\"")}"'

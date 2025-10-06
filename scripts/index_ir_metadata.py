#!/usr/bin/env python3
"""CLI utility for indexing JCAMP-DX metadata."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, MutableMapping, Sequence, Tuple

DATA_SECTION_PREFIXES = {
    "XYDATA",
    "XYPOINTS",
    "PEAK TABLE",
    "PEAKTABLE",
    "PEAK LIST",
    "XYPOINT",
}


def normalize_key(label: str) -> str:
    """Normalize a JCAMP-DX header label to snake_case."""
    cleaned = label.strip().lower()
    for prefix in ("$", "@", "%"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    sanitized = []
    prev_underscore = False
    for char in cleaned:
        if char.isalnum():
            sanitized.append(char)
            prev_underscore = False
        else:
            if not prev_underscore:
                sanitized.append("_")
                prev_underscore = True
    normalized = "".join(sanitized).strip("_")
    return normalized or "field"


def _should_stop(label: str) -> bool:
    upper = label.upper()
    return any(upper.startswith(prefix) for prefix in DATA_SECTION_PREFIXES)


def _store_header(
    normalized: MutableMapping[str, str],
    raw_entries: List[Tuple[str, str]],
    label: str,
    value: str,
) -> None:
    raw_entries.append((label, value))
    key = normalize_key(label)
    candidate = key
    index = 2
    while candidate in normalized:
        if normalized[candidate] == value:
            return
        candidate = f"{key}_{index}"
        index += 1
    normalized[candidate] = value


def parse_jdx_headers(path: Path) -> Tuple[OrderedDict[str, str], List[Tuple[str, str]]]:
    """Parse JCAMP-DX headers until the first data section."""
    normalized: OrderedDict[str, str] = OrderedDict()
    raw_entries: List[Tuple[str, str]] = []
    current_label: str | None = None
    current_parts: List[str] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # pragma: no cover - surfaced via CLI usage
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    for line in text:
        stripped = line.rstrip()
        if stripped.startswith("##"):
            if current_label is not None:
                value = " ".join(part for part in current_parts if part).strip()
                _store_header(normalized, raw_entries, current_label, value)
            content = stripped[2:]
            if "=" in content:
                label, value = content.split("=", 1)
            else:
                label, value = content, ""
            label = label.strip()
            if _should_stop(label):
                current_label = None
                current_parts = []
                break
            current_label = label
            value = value.strip()
            current_parts = [value] if value else []
        else:
            continuation = stripped.strip()
            if continuation and current_label is not None:
                current_parts.append(continuation)
    if current_label is not None:
        value = " ".join(part for part in current_parts if part).strip()
        _store_header(normalized, raw_entries, current_label, value)
    return normalized, raw_entries


def iter_jdx_files(root: Path) -> Iterable[Path]:
    for file_path in sorted(root.rglob("*.jdx")):
        if file_path.is_file():
            yield file_path


def build_records(
    root: Path,
    relative_to: Path | None = None,
    include_raw_headers: bool = False,
) -> List[OrderedDict[str, object]]:
    records: List[OrderedDict[str, object]] = []
    for file_path in iter_jdx_files(root):
        normalized, raw_entries = parse_jdx_headers(file_path)
        record: OrderedDict[str, object] = OrderedDict()
        rel_path: Path
        if relative_to is not None:
            try:
                rel_path = file_path.resolve().relative_to(relative_to)
            except ValueError:
                rel_path = Path(os.path.relpath(file_path.resolve(), relative_to))
        else:
            rel_path = file_path.relative_to(root)
        record["path"] = rel_path.as_posix()
        for key, value in normalized.items():
            record[key] = value
        if include_raw_headers:
            record["_raw_headers"] = [
                {"label": label, "value": value} for label, value in raw_entries
            ]
        records.append(record)
    return records


def determine_fields(
    records: Sequence[MutableMapping[str, object]],
    requested_fields: Sequence[str] | None,
    include_raw_headers: bool,
) -> List[str]:
    if requested_fields:
        ordered: List[str] = []
        seen = set()
        for field in requested_fields:
            normalized = normalize_key(field) if field != "_raw_headers" else field
            if normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        fields = [field for field in ordered if field != "path"]
        final_fields = ["path"] + fields
    else:
        collected = set()
        for record in records:
            for key in record.keys():
                if key in {"path", "_raw_headers"}:
                    continue
                collected.add(key)
        final_fields = ["path"] + sorted(collected)
    if include_raw_headers and "_raw_headers" not in final_fields:
        final_fields.append("_raw_headers")
    return final_fields


def prepare_records_for_output(
    records: Sequence[MutableMapping[str, object]],
    fields: Sequence[str],
) -> List[OrderedDict[str, object]]:
    prepared: List[OrderedDict[str, object]] = []
    for record in records:
        item: OrderedDict[str, object] = OrderedDict()
        for field in fields:
            value = record.get(field)
            item[field] = value
        prepared.append(item)
    return prepared


def output_json(records: Sequence[MutableMapping[str, object]]) -> None:
    json.dump(records, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def output_csv(records: Sequence[MutableMapping[str, object]], fields: Sequence[str]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = {}
        for field in fields:
            value = record.get(field)
            if isinstance(value, (dict, list)):
                row[field] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                row[field] = ""
            else:
                row[field] = value
        writer.writerow(row)


def list_fields(records: Sequence[MutableMapping[str, object]]) -> None:
    collected = set()
    for record in records:
        for key in record.keys():
            if key != "path":
                collected.add(key)
    for field in sorted(collected):
        sys.stdout.write(f"{field}\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="IR_referenceDatabase",
        help="Root directory containing JCAMP-DX files.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        help="Subset of normalized metadata fields to include in the output.",
    )
    parser.add_argument(
        "--relative-to",
        dest="relative_to",
        help="Base directory for emitted paths.",
    )
    parser.add_argument(
        "--include-raw-headers",
        action="store_true",
        help="Include a `_raw_headers` field with unnormalized label/value pairs.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available normalized metadata fields and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")
    relative_to = Path(args.relative_to).resolve() if args.relative_to else None
    records = build_records(root, relative_to=relative_to, include_raw_headers=args.include_raw_headers)

    if args.list_fields:
        list_fields(records)
        return 0

    fields = determine_fields(records, args.fields, args.include_raw_headers)
    prepared = prepare_records_for_output(records, fields)

    try:
        if args.format == "json":
            output_json(prepared)
        else:
            output_csv(prepared, fields)
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

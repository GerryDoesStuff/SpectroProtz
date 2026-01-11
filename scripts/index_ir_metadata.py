#!/usr/bin/env python3
"""Export JCAMP-DX header metadata from the IR reference database."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectro_app.engine.ftir_metadata import _normalize_molform


KEY_RE = re.compile(r"^##\s*([^=]+?)\s*=\s*(.*)$")
COMMENT_RE = re.compile(r"^\s*\$\$")


def _parse_headers(path: Path) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    current = None
    buffer: List[str] = []
    with path.open("r", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if COMMENT_RE.match(line):
                continue
            match = KEY_RE.match(line)
            if match:
                if current is not None:
                    headers[current] = "\n".join(buffer).strip()
                    buffer = []
                current = match.group(1).strip()
                buffer = [match.group(2).strip()]
                continue
            if line.upper().startswith("##XYDATA="):
                break
            if current is not None:
                buffer.append(line)
    if current is not None and current not in headers:
        headers[current] = "\n".join(buffer).strip()
    return headers


def _normalize_key(raw: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", raw.strip())
    return normalized.strip("_").lower()


def _iter_jdx_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.jdx"))


def _build_record(path: Path, root: Path) -> Dict[str, str]:
    headers = _parse_headers(path)
    record: Dict[str, str] = {"path": str(path.relative_to(root))}
    for key, value in headers.items():
        normalized_key = _normalize_key(key)
        if normalized_key == "molform":
            value = _normalize_molform(value) or ""
        record[normalized_key] = value
    return record


def _emit_json(records: List[Dict[str, str]]) -> None:
    print(json.dumps(records, ensure_ascii=False))


def _emit_csv(records: List[Dict[str, str]], fields: List[str]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "IR_referenceDatabase",
        help="Root folder containing .jdx files.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Optional list of normalized metadata fields to include.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List the available normalized fields and exit.",
    )
    args = parser.parse_args()

    root = args.root
    records = [_build_record(path, root) for path in _iter_jdx_files(root)]

    all_fields = sorted({key for record in records for key in record.keys()})
    if args.list_fields:
        for field in all_fields:
            print(field)
        return

    selected_fields = [field.lower() for field in (args.fields or [])]
    if selected_fields:
        fields = ["path"] + [field for field in selected_fields if field != "path"]
    else:
        fields = all_fields

    if args.format == "csv":
        _emit_csv(records, fields)
    else:
        if selected_fields:
            filtered = [
                {field: record.get(field, "") for field in fields}
                for record in records
            ]
            _emit_json(filtered)
        else:
            _emit_json(records)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import re
from typing import Dict


def sniff_locale(sample: str) -> Dict[str, str]:
    """Infer delimiter and decimal separator from a text sample.

    The heuristics intentionally favour robustness over strict correctness –
    Helios exports as well as ad-hoc CSV files often mix locale specific
    decimal separators and delimiters.  We prioritise recognising decimal
    commas so downstream parsing can substitute the delimiter accordingly.
    """

    if not sample:
        return {"decimal": ".", "delimiter": ","}

    # Normalise newlines and remove obviously empty lines to improve sniffing.
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    trimmed = "\n".join(lines)

    # Detect decimal separator by counting dot/comma occurrences between digits.
    dot_matches = re.findall(r"\d\.\d", trimmed)
    comma_matches = re.findall(r"\d,\d", trimmed)
    decimal = "," if len(comma_matches) > len(dot_matches) else "."

    # Try csv.Sniffer first; fall back to manual counting if it fails.
    delimiter = None
    try:
        dialect = csv.Sniffer().sniff(trimmed, delimiters=",;\t")
        delimiter = dialect.delimiter
    except (csv.Error, ValueError):
        pass

    if not delimiter:
        counts = {sep: trimmed.count(sep) for sep in (";", "\t", ",")}
        # If comma is the decimal separator, discount those occurrences.
        if decimal == ",":
            counts[","] = max(0, counts[","] - len(comma_matches))
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            delimiter = ","

    # Avoid conflicting comma-as-decimal with comma-as-delimiter when possible.
    if delimiter == "," and decimal == ",":
        delimiter = ";" if ";" in trimmed else "\t"
        if delimiter in {None, "\t"} and "\t" not in trimmed:
            delimiter = ","

    return {"decimal": decimal, "delimiter": delimiter or ","}

"""UV-Vis plugin primitives and integration helpers.

This module exposes :func:`disable_blank_requirement`, a convenience helper for
integrators that need to run the UV-Vis pipeline when blank spectra are
unavailable. The helper creates a recipe copy where blank enforcement is
disabled while preserving the caller's subtraction and default blank
configuration, allowing downstream validation to accept the relaxed policy.
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

from spectro_app.engine.io_common import sniff_locale
from spectro_app.engine.plugin_api import BatchResult, SpectroscopyPlugin, Spectrum
from spectro_app.engine.excel_writer import write_workbook
from .io_helios import is_helios_file, parse_metadata_lines, read_helios
from . import pipeline


def disable_blank_requirement(recipe: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of ``recipe`` with blank enforcement disabled.

    The helper is intended for programmatic integrations that need to process
    samples when blank spectra are not guaranteed. Existing subtraction flags
    and default/fallback blank settings are preserved so the caller can decide
    whether subtraction should remain enabled.
    """

    base_recipe: Dict[str, object]
    if recipe is None:
        base_recipe = {}
    elif isinstance(recipe, dict):
        base_recipe = recipe
    else:
        raise TypeError("disable_blank_requirement expects a dict recipe")

    updated = copy.deepcopy(base_recipe)
    blank_cfg_raw = updated.get("blank")
    if isinstance(blank_cfg_raw, dict):
        blank_cfg: Dict[str, object] = dict(blank_cfg_raw)
    elif blank_cfg_raw is None:
        blank_cfg = {}
    else:
        blank_cfg = dict(blank_cfg_raw)  # type: ignore[arg-type]

    blank_cfg["require"] = False
    updated["blank"] = blank_cfg
    return updated

class UvVisPlugin(SpectroscopyPlugin):
    id = "uvvis"
    label = "UV-Vis"
    xlabel = "Wavelength (nm)"
    DEFAULT_BLANK_TIME_WINDOW_MINUTES = 240.0
    DEFAULT_PATHLENGTH_TOLERANCE_CM = 0.01

    def __init__(self, *, enable_manifest: bool = True) -> None:
        self.enable_manifest = bool(enable_manifest)

    def detect(self, paths):
        return any(str(p).lower().endswith((".dsp", ".csv", ".xlsx", ".txt")) for p in paths)

    def load(self, paths: Iterable[str]) -> List[Spectrum]:
        spectra: List[Spectrum] = []
        path_objects = [Path(p) for p in paths]
        manifest_index, manifest_files = self._build_manifest_index(path_objects)

        for path in path_objects:
            if path.resolve() in manifest_files:
                continue
        manifest_entries: List[Dict[str, object]] = []
        data_paths: List[Path] = []

        for path_str in paths:
            path = Path(path_str)
            if self.enable_manifest and self._is_manifest_file(path):
                manifest_entries.extend(self._parse_manifest_file(path))
            else:
                data_paths.append(path)

        manifest_lookup = self._build_manifest_lookup(manifest_entries) if self.enable_manifest else {}

        for path in data_paths:
            suffix = path.suffix.lower()
            file_records: List[Dict[str, object]]

            if suffix in {".dsp"}:
                file_records = read_helios(path)
            elif suffix in {".csv", ".txt"}:
                text_sample = path.read_text(encoding="utf-8", errors="ignore")[:4000]
                if is_helios_file(path, text_sample):
                    file_records = read_helios(path)
                else:
                    file_records = self._read_generic_text(path, text_sample)
            elif suffix in {".xls", ".xlsx"}:
                if is_helios_file(path):
                    file_records = read_helios(path)
                else:
                    file_records = self._read_generic_excel(path)
            else:
                raise ValueError(f"Unsupported UV-Vis file type: {suffix}")

            for record in file_records:
                wl = np.asarray(record["wavelength"], dtype=float)
                inten = np.asarray(record["intensity"], dtype=float)
                meta = dict(record.get("meta", {}))
                meta.setdefault("technique", "uvvis")
                meta.setdefault("source_file", str(path))
                if manifest_index:
                    updates = self._lookup_manifest_index(manifest_index, path, meta)
                    if updates:
                        for key, value in updates.items():
                            if key == "role" and isinstance(value, str):
                                meta[key] = value.lower()
                            else:
                                meta[key] = value
                        if meta.get("role") == "blank":
                            if not meta.get("blank_id"):
                                meta["blank_id"] = meta.get("sample_id") or meta.get("channel")
                            elif (
                                meta.get("sample_id")
                                and meta.get("channel")
                                and meta.get("blank_id") == meta.get("channel")
                            ):
                                meta["blank_id"] = meta.get("sample_id")
                if self.enable_manifest:
                    manifest_meta = self._lookup_manifest_entries(manifest_lookup, path, meta)
                    if manifest_meta:
                        meta.update(manifest_meta)
                spectra.append(Spectrum(wavelength=wl, intensity=inten, meta=meta))

        return spectra

    @staticmethod
    def _is_manifest_file(path: Path) -> bool:
        name = path.name.lower()
        if "manifest" not in name:
            return False
        return path.suffix.lower() in {".csv", ".txt", ".tsv", ".xls", ".xlsx"}

    @staticmethod
    def _normalize_manifest_key(key: object) -> str:
        key_str = str(key).strip().lower()
        key_str = key_str.replace("-", "_")
        key_str = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key_str)
        while "__" in key_str:
            key_str = key_str.replace("__", "_")
        return key_str.strip("_")

    @staticmethod
    def _normalize_manifest_token(value: object) -> str:
        text = str(value).strip().replace("\\", "/")
        return text.lower()

    @staticmethod
    def _first_manifest_field(entry: Dict[str, object], keys: Iterable[str]) -> object | None:
        for key in keys:
            value = entry.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    def _parse_manifest_file(self, path: Path) -> List[Dict[str, object]]:
        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".txt", ".tsv"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if not text.strip():
                    return []
                locale = sniff_locale(text)
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=locale["delimiter"] if locale.get("delimiter") else None,
                    decimal=locale.get("decimal", "."),
                    engine="python",
                )
            elif suffix in {".xls", ".xlsx"}:
                df = pd.read_excel(path)
            else:
                return []
        except Exception:
            return []

        if df.empty:
            return []

        df = df.dropna(how="all")
        df.columns = [self._normalize_manifest_key(col) for col in df.columns]

        entries: List[Dict[str, object]] = []
        for row in df.to_dict(orient="records"):
            entry: Dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    continue
                if isinstance(value, float) and np.isnan(value):
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                entry[key] = value
            if entry:
                entries.append(entry)
        return entries

    def _build_manifest_lookup(self, entries: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
        lookup: Dict[str, Dict[str, Dict[str, object]]] = {
            "by_source": {},
            "by_sample": {},
            "by_channel": {},
        }
        for entry in entries:
            file_value = self._first_manifest_field(
                entry,
                ("file", "filename", "file_name", "source_file", "data_file", "path"),
            )
            sample_value = self._first_manifest_field(
                entry,
                ("sample_id", "sample", "sample_name"),
            )
            channel_value = self._first_manifest_field(
                entry,
                ("channel", "channel_id", "channel_name"),
            )

            if file_value:
                norm_full = self._normalize_manifest_token(file_value)
                lookup["by_source"][norm_full] = entry
                try:
                    norm_name = self._normalize_manifest_token(Path(str(file_value)).name)
                    lookup["by_source"].setdefault(norm_name, entry)
                except Exception:
                    pass
            if sample_value:
                lookup["by_sample"][self._normalize_manifest_token(sample_value)] = entry
            if channel_value:
                lookup["by_channel"][self._normalize_manifest_token(channel_value)] = entry

        return lookup

    def _lookup_manifest_entries(
        self,
        lookup: Dict[str, Dict[str, Dict[str, object]]],
        path: Path,
        meta: Dict[str, object],
    ) -> Dict[str, object]:
        if not lookup:
            return {}

        matches: List[Dict[str, object]] = []
        seen: set[int] = set()

        def add_entry(entry: Dict[str, object] | None) -> None:
            if not entry:
                return
            entry_id = id(entry)
            if entry_id in seen:
                return
            seen.add(entry_id)
            matches.append(entry)

        sample_tokens: List[str] = []
        sample_id = meta.get("sample_id")
        if sample_id is not None:
            token = self._normalize_manifest_token(sample_id)
            if token:
                sample_tokens.append(token)

        blank_id = meta.get("blank_id")
        if blank_id is not None:
            token = self._normalize_manifest_token(blank_id)
            if token and token not in sample_tokens:
                role = str(meta.get("role", "")).lower()
                if role == "blank" or not sample_tokens:
                    sample_tokens.append(token)

        channel_tokens: List[str] = []
        channel = meta.get("channel")
        if channel is not None:
            token = self._normalize_manifest_token(channel)
            if token:
                channel_tokens.append(token)

        file_tokens: List[str] = []
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        for candidate in (str(resolved), resolved.name):
            token = self._normalize_manifest_token(candidate)
            if token:
                file_tokens.append(token)

        for token in sample_tokens:
            add_entry(lookup.get("by_sample", {}).get(token))

        for token in channel_tokens:
            add_entry(lookup.get("by_channel", {}).get(token))

        if not matches:
            for token in file_tokens:
                add_entry(lookup.get("by_source", {}).get(token))

        if not matches:
            return {}

        skip_keys = {"file", "filename", "file_name", "data_file", "source_file", "path"}
        merged: Dict[str, object] = {}
        for entry in matches:
            for key, value in entry.items():
                if key in skip_keys:
                    continue
                merged[key] = value
        return merged

    # ------------------------------------------------------------------
    # Manifest ingestion helpers

    @dataclass
    class _ManifestIndex:
        by_file_channel: Dict[Tuple[Optional[str], str], Dict[str, object]]
        by_file_sample: Dict[Tuple[Optional[str], str], Dict[str, object]]

        def __bool__(self) -> bool:  # pragma: no cover - trivial
            return bool(self.by_file_channel or self.by_file_sample)

    def _build_manifest_index(
        self, paths: Iterable[Path]
    ) -> Tuple["UvVisPlugin._ManifestIndex", Set[Path]]:
        manifest_files: Set[Path] = set()
        index = self._ManifestIndex(by_file_channel={}, by_file_sample={})

        for path in paths:
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                resolved = path
            if path.suffix.lower() == ".csv" and "manifest" in path.stem.lower():
                manifest_files.add(resolved)

        if not manifest_files:
            return index, manifest_files

        for manifest_path in manifest_files:
            try:
                df = pd.read_csv(manifest_path, sep=None, engine="python")
            except Exception:
                continue
            if df.empty:
                continue

            df = df.rename(columns=lambda col: str(col).strip())
            for raw_row in df.to_dict(orient="records"):
                normalized: Dict[str, str] = {}
                for key, value in raw_row.items():
                    if key is None:
                        continue
                    cleaned_value = self._clean_manifest_value(value)
                    if cleaned_value is None:
                        continue
                    normalized[str(key).strip().lower()] = cleaned_value
                if not normalized:
                    continue

                file_key = self._first_nonempty(
                    normalized,
                    ["file", "filename", "source", "source_file", "path"],
                )
                norm_file = self._normalize_manifest_token(file_key) if file_key else None

                channel_key = self._first_nonempty(
                    normalized,
                    ["channel", "column", "signal", "trace"],
                )
                norm_channel = (
                    self._normalize_manifest_token(channel_key) if channel_key else None
                )

                sample_match = self._first_nonempty(
                    normalized,
                    ["sample_id", "sample", "id"],
                )
                norm_sample = (
                    self._normalize_manifest_token(sample_match) if sample_match else None
                )

                assignments = self._extract_manifest_assignments(normalized)
                if norm_channel:
                    index.by_file_channel[(norm_file, norm_channel)] = dict(assignments)
                if norm_sample:
                    index.by_file_sample[(norm_file, norm_sample)] = dict(assignments)

        return index, manifest_files

    @staticmethod
    def _clean_manifest_value(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        if pd.isna(value):
            return None
        return str(value).strip()

    @staticmethod
    def _normalize_manifest_token(token: Optional[str]) -> Optional[str]:
        if token is None:
            return None
        lowered = str(token).strip().lower()
        return lowered or None

    @staticmethod
    def _first_nonempty(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            value = row.get(key)
            if value:
                return value
        return None

    @staticmethod
    def _extract_manifest_assignments(row: Dict[str, str]) -> Dict[str, object]:
        alias_map: Dict[str, Tuple[str, ...]] = {
            "sample_id": ("sample_id", "sample", "name"),
            "blank_id": ("blank_id", "blank", "reference", "ref"),
            "role": ("role",),
            "replicate_id": ("replicate", "replicate_id", "rep"),
            "group_id": ("group", "group_id", "set", "cohort"),
            "notes": ("notes", "note", "comment"),
        }
        assignments: Dict[str, object] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                value = row.get(alias)
                if value:
                    if target == "role":
                        assignments[target] = str(value).strip().lower()
                    else:
                        assignments[target] = value
                    break
        return assignments

    def _lookup_manifest_index(
        self,
        index: "UvVisPlugin._ManifestIndex",
        source_path: Path,
        meta: Dict[str, object],
    ) -> Dict[str, object]:
        candidates_channel: List[str] = []
        candidates_sample: List[str] = []
        channel = meta.get("channel")
        if channel:
            norm = self._normalize_manifest_token(channel)
            if norm:
                candidates_channel.append(norm)
        sample_id = meta.get("sample_id")
        if sample_id:
            norm = self._normalize_manifest_token(sample_id)
            if norm:
                candidates_sample.append(norm)
        blank_id = meta.get("blank_id")
        if blank_id:
            norm = self._normalize_manifest_token(blank_id)
            if norm and norm not in candidates_sample:
                role = str(meta.get("role", "")).lower()
                if role == "blank" or not candidates_sample:
                    candidates_sample.append(norm)

        if not candidates_channel and not candidates_sample:
            return {}

        file_candidates: List[Optional[str]] = []
        try:
            resolved = source_path.resolve()
        except FileNotFoundError:
            resolved = source_path
        file_candidates.extend(
            [
                self._normalize_manifest_token(str(resolved)),
                self._normalize_manifest_token(resolved.name),
            ]
        )
        file_candidates.append(None)

        for file_key in file_candidates:
            if file_key is None:
                continue
            for channel_key in candidates_channel:
                match = index.by_file_channel.get((file_key, channel_key))
                if match:
                    return match
        for file_key in file_candidates:
            if file_key is None:
                continue
            for sample_key in candidates_sample:
                match = index.by_file_sample.get((file_key, sample_key))
                if match:
                    return match

        for channel_key in candidates_channel:
            match = index.by_file_channel.get((None, channel_key))
            if match:
                return match
        for sample_key in candidates_sample:
            match = index.by_file_sample.get((None, sample_key))
            if match:
                return match

        return {}

    # ------------------------------------------------------------------
    # Generic CSV/Excel ingestion helpers
    def _read_generic_text(self, path: Path, sample: str) -> List[Dict[str, object]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        locale = sniff_locale(text or sample)
        lines = text.splitlines()
        header_idx, has_header, meta_lines = self._locate_table(
            lines, locale["delimiter"], locale["decimal"]
        )
        data_str = "\n".join(lines[header_idx:])
        df = pd.read_csv(
            io.StringIO(data_str),
            sep=locale["delimiter"] if locale["delimiter"] else None,
            decimal=locale["decimal"],
            engine="python",
            header=0 if has_header else None,
        )
        if not has_header:
            df.columns = [f"col_{idx}" for idx in range(df.shape[1])]

        return self._dataframe_to_records(df, meta_lines, {
            "source_type": "generic_csv",
            "instrument": "Generic UV-Vis",
        })

    def _read_generic_excel(self, path: Path) -> List[Dict[str, object]]:
        raw = pd.read_excel(path, header=None)
        rows_as_text = [
            "\t".join(str(v) for v in row if not pd.isna(v))
            for row in raw.itertuples(index=False, name=None)
        ]
        header_idx, has_header, meta_lines = self._locate_table(rows_as_text, "\t", ".")
        if has_header:
            df = pd.read_excel(path, header=header_idx)
        else:
            df = pd.read_excel(path, header=None, skiprows=header_idx)
            df.columns = [f"col_{idx}" for idx in range(df.shape[1])]

        df = df.dropna(axis=1, how="all")
        return self._dataframe_to_records(df, meta_lines, {
            "source_type": "generic_excel",
            "instrument": "Generic UV-Vis",
        })

    def _dataframe_to_records(
        self,
        df: pd.DataFrame,
        meta_lines: Iterable[str],
        base_meta: Dict[str, object],
    ) -> List[Dict[str, object]]:
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]
        if df.shape[1] < 2:
            raise ValueError("Tabular UV-Vis file must contain wavelength and one intensity column")

        parsed_meta = parse_metadata_lines(meta_lines)
        meta: Dict[str, object] = {**base_meta, **parsed_meta}
        meta.setdefault("technique", "uvvis")
        mode = meta.get("mode")
        if not mode:
            mode = self._infer_mode_from_headers(df.columns[1:])
            if mode:
                meta["mode"] = mode

        wavelength = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        intensity_columns = df.columns[1:]
        blank_candidates = [col for col in intensity_columns if self._infer_role(col) == "blank"]
        records: List[Dict[str, object]] = []
        for idx, column in enumerate(intensity_columns, start=1):
            intensities = pd.to_numeric(df.iloc[:, idx], errors="coerce")
            mask = wavelength.notna() & intensities.notna()
            wl = wavelength[mask].to_numpy(dtype=float)
            inten = intensities[mask].to_numpy(dtype=float)
            if wl.size == 0:
                continue
            column_meta = dict(meta)
            column_meta.setdefault("channel", column)
            column_meta.setdefault("sample_id", column)
            role = self._infer_role(column)
            column_meta.setdefault("role", role)
            if role == "sample" and blank_candidates:
                column_meta.setdefault("blank_id", blank_candidates[0])
            if role == "blank":
                column_meta.setdefault("blank_id", column)
            records.append({
                "wavelength": wl,
                "intensity": inten,
                "meta": column_meta,
            })
        return records

    def _locate_table(self, lines: Iterable[str], delimiter: str | None, decimal: str):
        lines = list(lines)
        numeric_idx = None
        for idx, line in enumerate(lines):
            stripped = str(line).strip()
            if not stripped:
                continue
            tokens = self._tokenise(stripped, delimiter)
            numeric_tokens = sum(1 for token in tokens if self._is_number(token, decimal))
            if numeric_tokens >= 2:
                numeric_idx = idx
                break
        if numeric_idx is None:
            raise ValueError("Unable to locate numeric data rows in UV-Vis file")

        header_idx = numeric_idx
        if numeric_idx > 0:
            prev_tokens = self._tokenise(lines[numeric_idx - 1], delimiter)
            if prev_tokens and not all(self._is_number(tok, decimal) for tok in prev_tokens):
                header_idx = numeric_idx - 1

        meta_lines = [line for line in lines[:header_idx] if str(line).strip()]
        has_header = header_idx != numeric_idx
        return header_idx, has_header, meta_lines

    @staticmethod
    def _tokenise(line: str, delimiter: str | None):
        if delimiter and delimiter.strip():
            return [token.strip() for token in line.split(delimiter)]
        return line.split()

    @staticmethod
    def _is_number(token: str, decimal: str) -> bool:
        token = token.strip()
        if not token:
            return False
        if decimal != ".":
            token = token.replace(decimal, ".")
        token = token.replace(" ", "")
        try:
            float(token)
            return True
        except ValueError:
            return False

    @staticmethod
    def _infer_role(label: str) -> str:
        lower = label.lower()
        if "blank" in lower or "reference" in lower or lower.startswith("ref"):
            return "blank"
        return "sample"

    @staticmethod
    def _infer_mode_from_headers(headers: Iterable[str]) -> str | None:
        for header in headers:
            lower = str(header).lower()
            if "abs" in lower:
                return "absorbance"
            if "%t" in lower or "trans" in lower:
                return "transmittance"
            if "%r" in lower or "reflect" in lower:
                return "reflectance"
        return None

    def preprocess(self, specs, recipe):
        if not specs:
            return []

        domain_cfg = recipe.get("domain")
        blank_cfg = recipe.get("blank", {})
        baseline_cfg = recipe.get("baseline", {})
        join_cfg = recipe.get("join", {})
        despike_cfg = recipe.get("despike", {})
        smoothing_cfg = recipe.get("smoothing", {})
        replicate_cfg = recipe.get("replicates", {})

        if domain_cfg and domain_cfg.get("min") is not None and domain_cfg.get("max") is not None:
            if float(domain_cfg["min"]) >= float(domain_cfg["max"]):
                raise ValueError("Domain minimum must be smaller than maximum")

        if join_cfg.get("enabled"):
            window = int(join_cfg.get("window", 10))
            if window < 1:
                raise ValueError("Join correction window must be positive")
            threshold = join_cfg.get("threshold")
            if threshold is not None and float(threshold) <= 0:
                raise ValueError("Join detection threshold must be positive")

        if smoothing_cfg.get("enabled"):
            window = int(smoothing_cfg.get("window", 5))
            poly = int(smoothing_cfg.get("polyorder", 2))
            if window % 2 == 0:
                raise ValueError("Savitzky-Golay window must be odd")
            if window <= poly:
                raise ValueError("Savitzky-Golay window must exceed polynomial order")

        average_replicates = bool(replicate_cfg.get("average", True))
        outlier_cfg = replicate_cfg.get("outlier")

        stage_one: list[Spectrum] = []
        for spec in specs:
            coerced = pipeline.coerce_domain(spec, domain_cfg)
            processed = coerced
            joins: list[int] = []
            if join_cfg.get("enabled"):
                joins = pipeline.detect_joins(
                    processed.wavelength,
                    processed.intensity,
                    threshold=join_cfg.get("threshold"),
                    window=join_cfg.get("window", 10),
                )
                if joins:
                    processed = pipeline.correct_joins(processed, joins, window=join_cfg.get("window", 10))
            if despike_cfg.get("enabled"):
                processed = pipeline.despike_spectrum(
                    processed,
                    zscore=despike_cfg.get("zscore", 5.0),
                    window=despike_cfg.get("window", 5),
                    join_indices=joins,
                )
            stage_one.append(processed)

        blanks_stage = [spec for spec in stage_one if spec.meta.get("role") == "blank"]
        samples_stage = [spec for spec in stage_one if spec.meta.get("role") != "blank"]

        if average_replicates and blanks_stage:
            blanks_avg, blank_map_by_key = pipeline.average_replicates(
                blanks_stage,
                return_mapping=True,
                outlier=outlier_cfg,
            )
        else:
            blanks_avg = blanks_stage
            blank_map_by_key = {pipeline.replicate_key(b): b for b in blanks_stage}

        blank_lookup: dict[str, Spectrum] = {}
        for blank_spec in blanks_avg:
            ident = pipeline.blank_identifier(blank_spec)
            if ident:
                blank_lookup[ident] = blank_spec

        subtract_blank = blank_cfg.get("subtract", blank_cfg.get("enabled", True))
        require_blank = blank_cfg.get("require", subtract_blank)
        validate_flag = blank_cfg.get("validate_metadata")
        if validate_flag is None:
            validate_flag = True
        else:
            validate_flag = bool(validate_flag)
        fallback_blank = blank_cfg.get("default") or blank_cfg.get("fallback")

        processed_samples: list[Spectrum] = []
        for spec in samples_stage:
            working = spec
            blank_identifier = spec.meta.get("blank_id") or fallback_blank
            blank_spec = None
            if blank_identifier:
                blank_spec = blank_lookup.get(str(blank_identifier))

            if subtract_blank:
                if blank_spec is not None:
                    audit = self._validate_blank_pairing(
                        working,
                        blank_spec,
                        blank_cfg,
                        enforce=validate_flag,
                    )
                    working = pipeline.subtract_blank(working, blank_spec, audit=audit)
                elif blank_identifier:
                    if require_blank:
                        raise ValueError(
                            f"No blank spectrum available for blank_id '{blank_identifier}'"
                        )
                    audit = self._build_missing_blank_audit(working, blank_identifier)
                    working = self._attach_blank_audit(working, audit)
                elif require_blank:
                    sample_id = spec.meta.get("sample_id") or spec.meta.get("channel")
                    raise ValueError(
                        f"Sample '{sample_id}' does not have a blank for subtraction"
                    )
            else:
                if blank_spec is not None:
                    audit = self._validate_blank_pairing(
                        working,
                        blank_spec,
                        blank_cfg,
                        enforce=False,
                    )
                    working = self._attach_blank_audit(working, audit)
                elif blank_identifier:
                    audit = self._build_missing_blank_audit(working, blank_identifier)
                    working = self._attach_blank_audit(working, audit)

            if baseline_cfg.get("method"):
                method = baseline_cfg.get("method")
                params = {
                    "lam": baseline_cfg.get("lam", baseline_cfg.get("lambda", 1e5)),
                    "p": baseline_cfg.get("p", 0.01),
                    "niter": baseline_cfg.get("niter", 10),
                    "iterations": baseline_cfg.get("iterations", 24),
                }
                working = pipeline.apply_baseline(working, method, **params)

            if smoothing_cfg.get("enabled"):
                working = pipeline.smooth_spectrum(
                    working,
                    window=int(smoothing_cfg.get("window", 5)),
                    polyorder=int(smoothing_cfg.get("polyorder", 2)),
                )

            processed_samples.append(working)

        if average_replicates and processed_samples:
            processed_samples, sample_map = pipeline.average_replicates(
                processed_samples,
                return_mapping=True,
                outlier=outlier_cfg,
            )
        else:
            sample_map = {pipeline.replicate_key(s): s for s in processed_samples}

        blanks_final: list[Spectrum] = []
        if blanks_avg:
            blanks_working = list(blanks_avg)
            if baseline_cfg.get("method"):
                blanks_working = [
                    pipeline.apply_baseline(
                        blank,
                        baseline_cfg.get("method"),
                        lam=baseline_cfg.get("lam", baseline_cfg.get("lambda", 1e5)),
                        p=baseline_cfg.get("p", 0.01),
                        niter=baseline_cfg.get("niter", 10),
                        iterations=baseline_cfg.get("iterations", 24),
                    )
                    for blank in blanks_working
                ]
            if smoothing_cfg.get("enabled"):
                blanks_working = [
                    pipeline.smooth_spectrum(
                        blank,
                        window=int(smoothing_cfg.get("window", 5)),
                        polyorder=int(smoothing_cfg.get("polyorder", 2)),
                    )
                    for blank in blanks_working
                ]
            blanks_final = blanks_working
            blank_map_by_key = {
                pipeline.replicate_key(blank): blank for blank in blanks_final
            }

        order_keys = []
        for spec in stage_one:
            key = pipeline.replicate_key(spec)
            if key not in order_keys:
                order_keys.append(key)

        processed_map = {**blank_map_by_key, **sample_map}
        final_specs = [processed_map[key] for key in order_keys if key in processed_map]
        return final_specs

    def _validate_blank_pairing(
        self,
        sample: Spectrum,
        blank: Spectrum,
        blank_cfg: Dict[str, object],
        *,
        enforce: bool = True,
    ) -> Dict[str, object]:
        """Validate metadata compatibility between ``sample`` and ``blank``.

        Returns a dictionary with audit metadata that should be recorded when
        the subtraction succeeds. Raises ``ValueError`` when validation fails.
        """

        sample_time = self._parse_timestamp(sample.meta)
        blank_time = self._parse_timestamp(blank.meta)

        max_minutes_cfg = blank_cfg.get("max_time_delta_minutes")
        max_minutes = (
            float(max_minutes_cfg)
            if max_minutes_cfg is not None
            else self.DEFAULT_BLANK_TIME_WINDOW_MINUTES
        )
        time_delta_minutes: float | None = None
        violations: List[Dict[str, object]] = []
        if sample_time and blank_time:
            delta = abs(sample_time - blank_time)
            time_delta_minutes = delta.total_seconds() / 60.0
            if max_minutes is not None and time_delta_minutes > max_minutes:
                sample_id = self._safe_sample_id(sample, "sample")
                blank_id = blank.meta.get("blank_id") or blank.meta.get("sample_id") or "blank"
                violations.append(
                    {
                        "type": "timestamp_gap",
                        "sample_id": sample_id,
                        "blank_id": blank_id,
                        "limit_minutes": max_minutes,
                        "observed_minutes": time_delta_minutes,
                    }
                )
                if enforce:
                    raise ValueError(
                        "Blank/sample timestamp gap exceeds allowed window: "
                        f"sample '{sample_id}' vs blank '{blank_id}'"
                    )

        tolerance_cfg = blank_cfg.get("pathlength_tolerance_cm")
        tolerance = (
            float(tolerance_cfg)
            if tolerance_cfg is not None
            else self.DEFAULT_PATHLENGTH_TOLERANCE_CM
        )
        sample_path = sample.meta.get("pathlength_cm")
        blank_path = blank.meta.get("pathlength_cm")
        sample_path_val = self._coerce_float(sample_path)
        blank_path_val = self._coerce_float(blank_path)
        path_delta: float | None = None
        if sample_path_val is not None and blank_path_val is not None:
            path_delta = abs(sample_path_val - blank_path_val)
            if tolerance is not None and path_delta > tolerance:
                sample_id = self._safe_sample_id(sample, "sample")
                blank_id = blank.meta.get("blank_id") or blank.meta.get("sample_id") or "blank"
                violations.append(
                    {
                        "type": "pathlength_mismatch",
                        "sample_id": sample_id,
                        "blank_id": blank_id,
                        "tolerance_cm": tolerance,
                        "observed_delta_cm": path_delta,
                        "sample_pathlength_cm": sample_path_val,
                        "blank_pathlength_cm": blank_path_val,
                    }
                )
                if enforce:
                    raise ValueError(
                        "Sample and blank pathlengths are incompatible for subtraction: "
                        f"sample '{sample_id}' vs blank '{blank_id}'"
                    )

        sample_identifier = self._safe_sample_id(sample, "sample")
        blank_identifier = blank.meta.get("blank_id") or blank.meta.get("sample_id")

        audit: Dict[str, object] = {
            "sample_id": sample_identifier,
            "blank_id": blank_identifier,
            "blank_source_file": blank.meta.get("source_file"),
            "sample_timestamp": sample_time.isoformat() if sample_time else None,
            "blank_timestamp": blank_time.isoformat() if blank_time else None,
            "timestamp_delta_minutes": time_delta_minutes,
            "sample_pathlength_cm": sample_path_val,
            "blank_pathlength_cm": blank_path_val,
            "pathlength_delta_cm": path_delta,
            "validation_ran": True,
            "validation_enforced": bool(enforce),
            "validation_violations": violations,
        }
        return audit

    @staticmethod
    def _parse_timestamp(meta: Dict[str, object]) -> datetime | None:
        for key in ("acquired_datetime", "timestamp", "datetime"):
            value = meta.get(key)
            if isinstance(value, datetime):
                return value
            if value:
                try:
                    return datetime.fromisoformat(str(value))
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _parse_specific_timestamp(meta: Dict[str, object], key: str) -> datetime | None:
        if not key:
            return None
        value = meta.get(key)
        if isinstance(value, datetime):
            return value
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    def _build_missing_blank_audit(self, sample: Spectrum, blank_id: object) -> Dict[str, object]:
        sample_time = self._parse_timestamp(sample.meta)
        sample_identifier = self._safe_sample_id(sample, "sample")
        sample_path_val = self._coerce_float(sample.meta.get("pathlength_cm"))
        violation = {
            "type": "blank_missing",
            "sample_id": sample_identifier,
            "blank_id": str(blank_id),
        }
        return {
            "sample_id": sample_identifier,
            "blank_id": str(blank_id),
            "blank_source_file": None,
            "sample_timestamp": sample_time.isoformat() if sample_time else None,
            "blank_timestamp": None,
            "timestamp_delta_minutes": None,
            "sample_pathlength_cm": sample_path_val,
            "blank_pathlength_cm": None,
            "pathlength_delta_cm": None,
            "validation_ran": False,
            "validation_enforced": False,
            "validation_violations": [violation],
        }

    @staticmethod
    def _attach_blank_audit(sample: Spectrum, audit: Dict[str, object]) -> Spectrum:
        meta = dict(sample.meta or {})
        existing_audit = meta.get("blank_audit")
        if isinstance(existing_audit, dict):
            merged = dict(existing_audit)
            for key, value in audit.items():
                merged.setdefault(key, value)
        else:
            merged = dict(audit)
        meta["blank_audit"] = merged
        meta.setdefault("blank_subtracted", False)
        return Spectrum(wavelength=sample.wavelength, intensity=sample.intensity, meta=meta)

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_sample_id(spec: Spectrum, fallback: str) -> str:
        meta = spec.meta or {}
        return str(
            meta.get("sample_id")
            or meta.get("channel")
            or meta.get("blank_id")
            or fallback
        )

    @staticmethod
    def _fill_nan(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
        x = np.arange(arr.size)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return np.zeros_like(arr)
        arr = arr.copy()
        arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])
        return arr

    def _compute_band_value(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        center: float,
        width: float,
    ) -> float:
        if wl.size == 0:
            return float("nan")
        half = max(float(width) / 2.0, 0.0)
        mask = (wl >= center - half) & (wl <= center + half)
        if not np.any(mask):
            return float(np.interp(center, wl, intensity, left=np.nan, right=np.nan))
        return float(np.nanmean(intensity[mask]))

    def _compute_band_ratios(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        configs: List[Dict[str, object]],
    ) -> Dict[str, Dict[str, float]]:
        ratios: Dict[str, Dict[str, float]] = {}
        for cfg in configs:
            name = str(cfg.get("name") or "ratio")
            numerator = float(cfg.get("numerator", cfg.get("num", 0.0)))
            denominator = float(cfg.get("denominator", cfg.get("den", 1.0)))
            num_width = float(cfg.get("numerator_width", cfg.get("bandwidth", 2.0)))
            den_width = float(cfg.get("denominator_width", cfg.get("bandwidth", 2.0)))
            num_val = self._compute_band_value(wl, intensity, numerator, num_width)
            den_val = self._compute_band_value(wl, intensity, denominator, den_width)
            if not np.isfinite(num_val) or not np.isfinite(den_val) or abs(den_val) < 1e-12:
                ratio_val = float("nan")
            else:
                ratio_val = float(num_val) / float(den_val)
            ratios[name] = {
                "value": ratio_val,
                "numerator_value": num_val,
                "denominator_value": den_val,
                "numerator_center": numerator,
                "denominator_center": denominator,
            }
        return ratios

    @staticmethod
    def _integrate_band(wl: np.ndarray, intensity: np.ndarray, start: float, stop: float) -> float:
        if wl.size == 0:
            return float("nan")
        lo, hi = sorted((float(start), float(stop)))
        mask = (wl >= lo) & (wl <= hi)
        if mask.sum() < 2:
            return float("nan")
        return float(np.trapezoid(intensity[mask], wl[mask]))

    def _compute_integrals(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        configs: List[Dict[str, object]],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for cfg in configs:
            name = str(cfg.get("name") or f"area_{cfg.get('min')}_{cfg.get('max')}")
            start = cfg.get("min") if cfg.get("min") is not None else cfg.get("start")
            stop = cfg.get("max") if cfg.get("max") is not None else cfg.get("stop")
            if start is None or stop is None:
                continue
            results[name] = self._integrate_band(wl, intensity, float(start), float(stop))
        return results

    def _compute_derivatives(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        enabled: bool = True,
    ) -> Dict[str, np.ndarray]:
        if not enabled or intensity.size < 3:
            return {}
        safe_y = self._fill_nan(intensity)
        safe_wl = self._fill_nan(wl)
        first = np.gradient(safe_y, safe_wl)
        second = np.gradient(first, safe_wl)
        return {
            "first_derivative": first,
            "second_derivative": second,
        }

    def _compute_peak_metrics(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        peak_cfg: Dict[str, object],
    ) -> List[Dict[str, float]]:
        if intensity.size < 3:
            return []
        prominence = float(peak_cfg.get("prominence", 0.01))
        min_distance = max(1, int(peak_cfg.get("min_distance", peak_cfg.get("distance", 5))))
        height = peak_cfg.get("height")
        if height is not None:
            height = float(height)
        peaks, properties = find_peaks(
            intensity,
            prominence=prominence if prominence > 0 else None,
            distance=min_distance,
            height=height,
        )
        if peaks.size == 0:
            return []
        max_peaks = int(peak_cfg.get("max_peaks", peak_cfg.get("num_peaks", 5)))
        prominences = properties.get("prominences")
        if prominences is not None:
            order = np.argsort(prominences)[::-1]
            prominences = prominences[order]
            peaks = peaks[order]
        if max_peaks > 0:
            peaks = peaks[:max_peaks]
            if prominences is not None:
                prominences = prominences[:max_peaks]
        _, _, left_ips, right_ips = peak_widths(intensity, peaks, rel_height=0.5)
        x_indices = np.arange(wl.size)
        peak_rows: List[Dict[str, float]] = []
        for idx, peak in enumerate(peaks):
            peak_wl = float(wl[peak])
            peak_height = float(intensity[peak])
            left_wl = float(np.interp(left_ips[idx], x_indices, wl))
            right_wl = float(np.interp(right_ips[idx], x_indices, wl))
            peak_rows.append({
                "wavelength": peak_wl,
                "height": peak_height,
                "fwhm": max(right_wl - left_wl, 0.0),
                "prominence": float(prominences[idx]) if prominences is not None else float("nan"),
            })
        return peak_rows

    def _compute_isosbestic_checks(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        configs: Iterable[Dict[str, object]] | Iterable[object],
    ) -> List[Dict[str, object]]:
        wl = np.asarray(wl, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        results: List[Dict[str, object]] = []
        if wl.size < 2 or intensity.size < 2:
            return results
        for idx, cfg in enumerate(configs or []):
            if isinstance(cfg, dict):
                name = str(cfg.get("name") or f"isosbestic_{idx + 1}")
                wavelengths = cfg.get("wavelengths") or cfg.get("pair") or cfg.get("window")
                tolerance = cfg.get("tolerance")
            else:
                name = f"isosbestic_{idx + 1}"
                wavelengths = cfg
                tolerance = None
            wl_pair: Tuple[float, float] | None = None
            if isinstance(wavelengths, (list, tuple, np.ndarray)) and len(wavelengths) >= 2:
                lo, hi = float(wavelengths[0]), float(wavelengths[1])
                if lo == hi:
                    hi = lo + 1e-6
                wl_pair = (min(lo, hi), max(lo, hi))
            if wl_pair is None:
                results.append({
                    "name": name,
                    "wavelengths": None,
                    "crossing_detected": False,
                    "min_abs_deviation": float("nan"),
                    "crossing_wavelength": None,
                    "tolerance": float(tolerance) if tolerance is not None else None,
                    "within_tolerance": None,
                })
                continue
            lo, hi = wl_pair
            interp_lo = float(np.interp(lo, wl, intensity, left=np.nan, right=np.nan))
            interp_hi = float(np.interp(hi, wl, intensity, left=np.nan, right=np.nan))
            mask = (wl >= lo) & (wl <= hi)
            if mask.sum() < 2:
                results.append({
                    "name": name,
                    "wavelengths": [lo, hi],
                    "crossing_detected": False,
                    "min_abs_deviation": float("nan"),
                    "crossing_wavelength": None,
                    "tolerance": float(tolerance) if tolerance is not None else None,
                    "within_tolerance": None,
                    "endpoints": [interp_lo, interp_hi],
                })
                continue
            segment_wl = wl[mask]
            segment_intensity = intensity[mask]
            line_baseline = np.interp(segment_wl, [lo, hi], [interp_lo, interp_hi])
            deviation = np.asarray(segment_intensity - line_baseline, dtype=float)
            finite_mask = np.isfinite(deviation)
            crossing_detected = False
            crossing_wavelength: float | None = None
            min_abs_deviation = float("nan")
            within_tolerance: bool | None = None
            if np.any(finite_mask):
                finite_dev = deviation[finite_mask]
                min_abs_deviation = float(np.nanmin(np.abs(finite_dev)))
                if finite_dev.size:
                    # Determine if the deviation crosses zero within the window.
                    crossing_detected = bool(
                        np.any(np.diff(np.signbit(finite_dev)) != 0)
                        or np.any(finite_dev == 0.0)
                    )
                if crossing_detected:
                    # Find the first zero crossing for reporting.
                    finite_wl = segment_wl[finite_mask]
                    sign_changes = np.where(np.signbit(finite_dev[:-1]) != np.signbit(finite_dev[1:]))[0]
                    if sign_changes.size:
                        idx_change = sign_changes[0]
                        x0 = float(finite_wl[idx_change])
                        x1 = float(finite_wl[idx_change + 1])
                        y0 = float(finite_dev[idx_change])
                        y1 = float(finite_dev[idx_change + 1])
                        if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
                            crossing_wavelength = x0 + (x1 - x0) * (-y0) / (y1 - y0)
                        else:
                            crossing_wavelength = x0
                    else:
                        zero_indices = np.where(finite_dev == 0.0)[0]
                        if zero_indices.size:
                            crossing_wavelength = float(finite_wl[zero_indices[0]])
                if tolerance is not None and np.isfinite(min_abs_deviation):
                    within_tolerance = bool(min_abs_deviation <= float(tolerance))
            result = {
                "name": name,
                "wavelengths": [lo, hi],
                "crossing_detected": crossing_detected,
                "min_abs_deviation": min_abs_deviation,
                "crossing_wavelength": crossing_wavelength,
                "tolerance": float(tolerance) if tolerance is not None else None,
                "within_tolerance": within_tolerance,
                "endpoints": [interp_lo, interp_hi],
            }
            results.append(result)
        return results

    @staticmethod
    def _normalize_kinetics_targets(config: object) -> List[Dict[str, float]]:
        if not config:
            return []
        if isinstance(config, dict):
            raw_targets = config.get("targets") or config.get("wavelengths") or []
        else:
            raw_targets = config
        targets: List[Dict[str, float]] = []
        for idx, entry in enumerate(raw_targets or []):
            if isinstance(entry, dict):
                name = str(entry.get("name") or f"target_{idx + 1}")
                wavelength = entry.get("wavelength")
                if wavelength is None:
                    wavelength = entry.get("lambda") or entry.get("center")
                if wavelength is None:
                    continue
                bandwidth = float(entry.get("bandwidth", entry.get("width", 2.0)))
                targets.append(
                    {
                        "name": name,
                        "wavelength": float(wavelength),
                        "bandwidth": max(float(bandwidth), 0.0),
                    }
                )
            else:
                try:
                    wavelength = float(entry)
                except (TypeError, ValueError):
                    continue
                targets.append(
                    {
                        "name": f"target_{idx + 1}",
                        "wavelength": float(wavelength),
                        "bandwidth": 2.0,
                    }
                )
        return targets

    def _compute_kinetics_values(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        targets: List[Dict[str, float]],
    ) -> Dict[str, float]:
        if not targets:
            return {}
        values: Dict[str, float] = {}
        for target in targets:
            values[target["name"]] = self._compute_band_value(
                wl,
                intensity,
                target["wavelength"],
                target.get("bandwidth", 0.0),
            )
        return values

    def _summarize_kinetics(
        self,
        records: List[Dict[str, object]],
        targets: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, object]]:
        summaries: Dict[str, Dict[str, object]] = {}
        if not records:
            return summaries
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for record in records:
            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            grouped.setdefault(str(sample_id), []).append(record)
        for sample_id, group in grouped.items():
            valid = [rec for rec in group if isinstance(rec.get("timestamp"), datetime)]
            valid.sort(key=lambda rec: rec.get("timestamp"))
            if valid:
                anchor = valid[0]["timestamp"]
            else:
                anchor = None
            for rec in valid:
                timestamp = rec.get("timestamp")
                assert isinstance(timestamp, datetime)
                if anchor:
                    delta = timestamp - anchor
                    relative = delta.total_seconds() / 60.0
                else:
                    relative = 0.0
                rec["relative_minutes"] = relative
                row_entry = rec.get("row_entry")
                feature_entry = rec.get("feature_entry")
                if isinstance(row_entry, dict):
                    row_entry["relative_minutes"] = float(relative)
                if isinstance(feature_entry, dict):
                    feature_entry["relative_minutes"] = float(relative)
            # For entries without timestamps, explicitly set relative minutes to None.
            for rec in group:
                if "relative_minutes" not in rec:
                    row_entry = rec.get("row_entry")
                    feature_entry = rec.get("feature_entry")
                    if isinstance(row_entry, dict):
                        row_entry["relative_minutes"] = None
                    if isinstance(feature_entry, dict):
                        feature_entry["relative_minutes"] = None
            if len(valid) < 2:
                for rec in group:
                    feature_entry = rec.get("feature_entry")
                    if isinstance(feature_entry, dict):
                        feature_entry["summary"] = None
                continue
            times = np.array([rec["relative_minutes"] for rec in valid], dtype=float)
            series: List[Dict[str, object]] = []
            group_sorted = sorted(
                group,
                key=lambda rec: rec.get("timestamp") or datetime.min,
            )
            for rec in group_sorted:
                timestamp = rec.get("timestamp")
                iso = timestamp.isoformat() if isinstance(timestamp, datetime) else None
                relative = rec.get("relative_minutes")
                series.append(
                    {
                        "timestamp": iso,
                        "relative_minutes": float(relative) if relative is not None else None,
                        "values": dict(rec.get("values", {})),
                    }
                )
            target_metrics: Dict[str, Dict[str, object]] = {}
            for target in targets:
                name = target["name"]
                values_arr = np.array(
                    [rec.get("values", {}).get(name, float("nan")) for rec in valid],
                    dtype=float,
                )
                mask = np.isfinite(times) & np.isfinite(values_arr)
                n_points = int(mask.sum())
                first_val = float(values_arr[mask][0]) if n_points else float("nan")
                last_val = float(values_arr[mask][-1]) if n_points else float("nan")
                delta_val = float(last_val - first_val) if n_points else float("nan")
                if n_points >= 2:
                    slope, _ = np.polyfit(times[mask], values_arr[mask], 1)
                    slope_val = float(slope)
                    corr = np.corrcoef(times[mask], values_arr[mask])
                    r_value = float(corr[0, 1]) if corr.size >= 4 else float("nan")
                else:
                    slope_val = float("nan")
                    r_value = float("nan")
                target_metrics[name] = {
                    "wavelength": float(target["wavelength"]),
                    "first": first_val,
                    "last": last_val,
                    "delta": delta_val,
                    "slope_per_min": slope_val,
                    "r_value": r_value,
                    "points": n_points,
                }
            elapsed = float(times[-1]) if times.size else float("nan")
            summary = {
                "sample_id": sample_id,
                "n_points": len(valid),
                "elapsed_minutes": elapsed,
                "targets": target_metrics,
                "series": series,
            }
            for rec in group:
                feature_entry = rec.get("feature_entry")
                if isinstance(feature_entry, dict):
                    feature_entry["summary"] = summary
                row = rec.get("row")
                if isinstance(row, dict):
                    row["kinetics_summary"] = summary
            summaries[sample_id] = summary
        return summaries

    def analyze(self, specs, recipe):
        from spectro_app.engine.qc import compute_uvvis_drift_map, compute_uvvis_qc

        feature_cfg = dict(recipe.get("features", {})) if recipe else {}
        peak_cfg = dict(feature_cfg.get("peaks", {}))
        ratio_cfg = feature_cfg.get("band_ratios")
        if not ratio_cfg:
            ratio_cfg = [
                {"name": "A260_A280", "numerator": 260.0, "denominator": 280.0, "bandwidth": 4.0},
                {"name": "A260_A230", "numerator": 260.0, "denominator": 230.0, "bandwidth": 4.0},
            ]
        integral_cfg = feature_cfg.get("integrals")
        if not integral_cfg:
            integral_cfg = [
                {"name": "Area_240_260", "min": 240.0, "max": 260.0},
                {"name": "Area_260_280", "min": 260.0, "max": 280.0},
            ]
        derivative_enabled = feature_cfg.get("derivatives", True)
        isosbestic_cfg = feature_cfg.get("isosbestic") or feature_cfg.get("isosbestic_checks") or []
        kinetics_cfg_raw = feature_cfg.get("kinetics") or {}
        kinetics_targets = self._normalize_kinetics_targets(kinetics_cfg_raw)
        kinetics_time_field: str | None = None
        if isinstance(kinetics_cfg_raw, dict):
            raw_field = kinetics_cfg_raw.get("time_key") or kinetics_cfg_raw.get("timestamp_field")
            if raw_field:
                kinetics_time_field = str(raw_field)

        qc_rows = []
        drift_map = compute_uvvis_drift_map(specs, recipe)
        processed_with_features: List[Spectrum] = []
        kinetics_records: List[Dict[str, object]] = []
        for idx, spec in enumerate(specs):
            metrics = compute_uvvis_qc(spec, recipe, drift_map.get(id(spec)))
            saturation = metrics["saturation"]
            join = metrics["join"]
            noise = metrics["noise"]
            spikes = metrics["spikes"]
            smoothing = metrics["smoothing"]
            drift = metrics["drift"]
            wl = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            derivatives = self._compute_derivatives(wl, intensity, derivative_enabled)
            band_ratios = self._compute_band_ratios(wl, intensity, ratio_cfg)
            integrals = self._compute_integrals(wl, intensity, integral_cfg)
            peaks = self._compute_peak_metrics(wl, intensity, peak_cfg)
            isosbestic_checks = self._compute_isosbestic_checks(wl, intensity, isosbestic_cfg)
            timestamp = self._parse_timestamp(spec.meta)
            if kinetics_time_field:
                specific = self._parse_specific_timestamp(spec.meta, kinetics_time_field)
                if specific:
                    timestamp = specific
            kinetics_values = self._compute_kinetics_values(wl, intensity, kinetics_targets)
            derivative_stats = {
                name: {
                    "min": float(np.nanmin(values)) if values.size else float("nan"),
                    "max": float(np.nanmax(values)) if values.size else float("nan"),
                    "mean": float(np.nanmean(values)) if values.size else float("nan"),
                    "rms": float(np.sqrt(np.nanmean(values ** 2))) if values.size else float("nan"),
                }
                for name, values in derivatives.items()
            }
            row = {
                "id": idx,
                "sample_id": spec.meta.get("sample_id") or spec.meta.get("channel") or spec.meta.get("blank_id"),
                "role": spec.meta.get("role", "sample"),
                "mode": metrics["mode"],
                "saturation_flag": saturation.flag,
                "saturation_min": saturation.minimum,
                "saturation_max": saturation.maximum,
                "noise_rsd": noise.rsd,
                "noise_window": noise.window,
                "noise_points": noise.used_points,
                "join_count": join.count,
                "join_max_offset": join.max_offset,
                "join_mean_offset": join.mean_offset,
                "join_max_overlap_error": join.max_overlap_error,
                "join_indices": join.indices,
                "spike_count": spikes.count,
                "spike_threshold": spikes.threshold,
                "smoothing_guard": smoothing.flag,
                "drift_available": drift.available,
                "drift_flag": drift.flag,
                "drift_reasons": drift.reasons,
                "drift_batch_reasons": drift.batch_reasons,
                "drift_baseline": drift.baseline,
                "drift_baseline_delta": drift.baseline_delta,
                "drift_slope_per_hour": drift.slope_per_hour,
                "drift_residual": drift.residual,
                "drift_predicted": drift.predicted,
                "drift_span_minutes": drift.span_minutes,
                "drift_timestamp": drift.timestamp.isoformat() if drift.timestamp else None,
                "drift_window": drift.window,
                "flags": metrics["flags"],
                "summary": metrics["summary"],
                "band_ratios": band_ratios,
                "integrals": integrals,
                "peak_metrics": peaks,
                "derivative_stats": derivative_stats,
                "isosbestic": isosbestic_checks,
            }
            qc_rows.append(row)
            meta = dict(spec.meta)
            meta.setdefault("features", {})
            meta["features"].update({
                "band_ratios": band_ratios,
                "integrals": integrals,
                "peaks": peaks,
                "derivative_stats": derivative_stats,
                "isosbestic": isosbestic_checks,
            })
            existing_channels = {
                key: np.asarray(val, dtype=float)
                for key, val in (meta.get("channels") or {}).items()
                if isinstance(val, (list, tuple, np.ndarray))
            }
            for name, arr in derivatives.items():
                existing_channels[name] = np.asarray(arr, dtype=float)
            if existing_channels:
                meta["channels"] = existing_channels
            sample_id_value = row["sample_id"]
            sample_label = str(sample_id_value) if sample_id_value is not None else None
            if sample_label is not None:
                row["sample_id"] = sample_label
            kinetics_entry_row = {
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
                "relative_minutes": None,
                "values": {name: float(val) for name, val in kinetics_values.items()},
            }
            row["kinetics"] = kinetics_entry_row
            row["kinetics_summary"] = None
            processed_with_features.append(
                Spectrum(
                    wavelength=spec.wavelength.copy(),
                    intensity=spec.intensity.copy(),
                    meta=meta,
                )
            )
            feature_kinetics = {
                "timestamp": kinetics_entry_row["timestamp"],
                "relative_minutes": None,
                "values": dict(kinetics_entry_row["values"]),
                "summary": None,
            }
            meta["features"]["kinetics"] = feature_kinetics
            kinetics_records.append(
                {
                    "sample_id": sample_label,
                    "timestamp": timestamp,
                    "values": dict(kinetics_entry_row["values"]),
                    "row": row,
                    "row_entry": kinetics_entry_row,
                    "feature_entry": feature_kinetics,
                }
            )
        if kinetics_records:
            self._summarize_kinetics(kinetics_records, kinetics_targets)
        return processed_with_features, qc_rows

    def _build_audit_entries(
        self,
        specs: List[Spectrum],
        qc: List[Dict[str, object]],
        recipe: Dict[str, object],
        figures: Dict[str, bytes],
    ) -> List[str]:
        entries = [
            "UV-Vis export initiated.",
            f"Spectra processed: {len(specs)}",
            f"QC rows: {len(qc)}",
        ]
        if recipe:
            entries.append(f"Recipe keys: {sorted(recipe.keys())}")
        for spec, qc_row in zip(specs, qc):
            sample_id = self._safe_sample_id(spec, f"spec_{id(spec)}")
            features = spec.meta.get("features", {})
            ratios = features.get("band_ratios", {})
            ratio_summary = ", ".join(
                f"{name}={vals.get('value'):.3f}" if np.isfinite(vals.get("value", np.nan)) else f"{name}=nan"
                for name, vals in ratios.items()
            )
            peak_info = features.get("peaks", [])
            if peak_info:
                primary_peak = peak_info[0]
                peak_summary = (
                    f"peak@{primary_peak.get('wavelength', float('nan')):.1f}nm"
                    if np.isfinite(primary_peak.get("wavelength", float("nan")))
                    else "peak@nan"
                )
            else:
                peak_summary = "no_peaks"
            entries.append(f"Spectrum {sample_id}: ratios[{ratio_summary}] peaks[{peak_summary}]")
        if figures:
            entries.append(f"Generated plots: {', '.join(sorted(figures))}")
        return entries

    def _generate_figures(self, specs: List[Spectrum]) -> Dict[str, bytes]:
        figures: Dict[str, bytes] = {}
        for spec in specs:
            wl = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            if wl.size == 0 or intensity.size == 0:
                continue
            channels = spec.meta.get("channels") or {}
            sample_id = self._safe_sample_id(spec, f"spec_{id(spec)}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(wl, intensity, label="Processed", linewidth=1.5)
            for name, channel in channels.items():
                channel_arr = np.asarray(channel, dtype=float)
                if channel_arr.shape != wl.shape:
                    continue
                ax.plot(wl, channel_arr, label=name.replace("_", " "))
            features = spec.meta.get("features", {})
            for peak in features.get("peaks", [])[:5]:
                wavelength = peak.get("wavelength")
                if wavelength is None or not np.isfinite(wavelength):
                    continue
                ax.axvline(wavelength, color="tab:orange", linestyle="--", alpha=0.3)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("Intensity")
            ax.set_title(f"Spectrum {sample_id}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.2)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            figures[f"{sample_id}_processed.png"] = buf.read()
        return figures

    def export(self, specs, qc, recipe):
        export_cfg = dict(recipe.get("export", {})) if recipe else {}
        output_path = export_cfg.get("path") or export_cfg.get("workbook")
        figures = self._generate_figures(specs)
        audit_entries = self._build_audit_entries(specs, qc, recipe, figures)
        workbook_audit = list(audit_entries)
        if output_path:
            resolved_path = str(output_path)
            workbook_audit.append(f"Workbook written to {resolved_path}")
            write_workbook(resolved_path, specs, qc, workbook_audit, figures)
            audit_entries = workbook_audit
        else:
            audit_entries.append("No workbook path provided; workbook not written.")
        return BatchResult(processed=specs, qc_table=qc, figures=figures, audit=audit_entries)

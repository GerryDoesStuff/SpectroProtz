"""UV-Vis plugin primitives and integration helpers.

This module exposes :func:`disable_blank_requirement`, a convenience helper for
integrators that need to run the UV-Vis pipeline when blank spectra are
unavailable. The helper creates a recipe copy where blank enforcement is
disabled while preserving the caller's subtraction and default blank
configuration, allowing downstream validation to accept the relaxed policy.
"""

from __future__ import annotations

import copy
import hashlib
import os
import io
import math
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set, Mapping

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.signal import find_peaks, peak_widths, savgol_filter

from spectro_app.engine.io_common import sniff_locale
from spectro_app.engine.plugin_api import BatchResult, SpectroscopyPlugin, Spectrum
from spectro_app.engine.excel_writer import write_workbook
from .conversions import convert_intensity_to_absorbance, normalise_mode
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
    HELIOS_GAMMA_WAVELENGTH_LIMITS_NM = (190.0, 1100.0)
    GENERIC_WAVELENGTH_LIMITS_NM = (190.0, 1100.0)
    DEFAULT_JOIN_ENABLED = True
    DEFAULT_JOIN_WINDOW_POINTS = 3
    DEFAULT_JOIN_THRESHOLD_ABS = 0.2
    DEFAULT_JOIN_WINDOWS_BY_INSTRUMENT: Mapping[
        str, Sequence[Mapping[str, float | int | None]]
    ] = {
        "helios": (
            {"min_nm": 340.0, "max_nm": 360.0},
        ),
    }
    DEFAULT_QC_QUIET_WINDOW_NM = (850.0, 900.0)

    UI_CAPABILITY_MANIFEST = "manifest_enrichment"

    def __init__(
        self,
        *,
        enable_manifest: bool = False,
        ui_capabilities: Optional[Mapping[str, object]] = None,
    ) -> None:
        if enable_manifest:
            warnings.warn(
                "enable_manifest=True is deprecated; configure UI capabilities instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self.enable_manifest = bool(enable_manifest)
        self._last_calibration_results: Dict[str, Any] | None = None
        self._report_context: Dict[str, Dict[str, Any]] = {}
        self._queue_overrides: Dict[str, Dict[str, object]] = {}
        self._ui_capabilities: Dict[str, bool] = {}
        self._reset_report_context()
        if ui_capabilities is None and self.enable_manifest:
            effective_caps: Dict[str, bool] | None = {
                self.UI_CAPABILITY_MANIFEST: True
            }
        else:
            effective_caps = dict(ui_capabilities) if ui_capabilities else None
        self.set_ui_capabilities(effective_caps)

    def supports_manifest_ui(self) -> bool:
        """Return whether the plugin should expose manifest-related UI."""

        return bool(self.enable_manifest)

    def _reset_report_context(self) -> None:
        self._report_context = {
            "ingestion": {
                "requested_paths": [],
                "data_files": [],
                "manifest_files": [],
                "spectra_count": 0,
                "blank_count": 0,
                "manifest_entry_count": 0,
                "manifest_index_hits": 0,
                "manifest_lookup_hits": 0,
                "manifest_enabled": bool(self._manifest_pipeline_enabled()),
            },
            "preprocessing": {},
            "analysis": {},
            "results": {},
        }

    def set_ui_capabilities(self, capabilities: Mapping[str, object] | None) -> None:
        values: Dict[str, bool] = {}
        if capabilities:
            for key, value in capabilities.items():
                if value is None:
                    continue
                key_str = str(key).strip().lower()
                if not key_str:
                    continue
                try:
                    enabled = bool(value)
                except Exception:
                    enabled = False
                values[key_str] = enabled
        self._ui_capabilities = values
        ingestion = self._report_context.get("ingestion")
        if isinstance(ingestion, dict):
            ingestion["manifest_enabled"] = bool(self._manifest_pipeline_enabled())

    @property
    def manifest_ui_capability_enabled(self) -> bool:
        return bool(self._ui_capabilities.get(self.UI_CAPABILITY_MANIFEST))

    def _manifest_pipeline_enabled(self) -> bool:
        return bool(self.enable_manifest and self.manifest_ui_capability_enabled)

    def set_queue_overrides(self, overrides: Mapping[str, Mapping[str, object]] | None) -> None:
        normalized: Dict[str, Dict[str, object]] = {}
        if overrides:
            for path, values in overrides.items():
                sanitized: Dict[str, object] = {}
                if isinstance(values, Mapping):
                    for key, value in values.items():
                        if value is None:
                            continue
                        key_str = str(key)
                        if not key_str:
                            continue
                        if isinstance(value, str):
                            trimmed = value.strip()
                            if not trimmed:
                                continue
                            if key_str == "role":
                                sanitized[key_str] = trimmed.lower()
                            else:
                                sanitized[key_str] = trimmed
                        else:
                            sanitized[key_str] = value
                if not sanitized:
                    continue
                for candidate in self._queue_override_keys(path):
                    normalized[candidate] = dict(sanitized)
        self._queue_overrides = normalized

    def _queue_override_keys(self, path: object) -> List[str]:
        keys: List[str] = []
        try:
            path_obj = Path(path)
        except Exception:
            text = str(path)
            if text:
                keys.append(text)
            return keys
        text = str(path_obj)
        if text:
            keys.append(text)
        try:
            resolved = path_obj.resolve()
        except FileNotFoundError:
            resolved = path_obj
        except RuntimeError:
            resolved = path_obj
        resolved_text = str(resolved)
        if resolved_text and resolved_text not in keys:
            keys.append(resolved_text)
        return keys

    def _lookup_queue_override(self, path: Path) -> Optional[Dict[str, object]]:
        if not self._queue_overrides:
            return None
        for key in self._queue_override_keys(path):
            override = self._queue_overrides.get(key)
            if override:
                return dict(override)
        return None

    @staticmethod
    def _summarise_sequence(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return {}
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {}
        return {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
        }

    @property
    def last_calibration_results(self) -> Dict[str, Any] | None:
        """Return results from the most recent calibration run."""

        return self._last_calibration_results

    def _apply_recipe_defaults(self, recipe: Dict[str, object] | None) -> Dict[str, object]:
        if recipe is None:
            base: Dict[str, object] = {}
        elif isinstance(recipe, dict):
            base = dict(recipe)
        else:
            raise TypeError("Recipe configuration must be a mapping or None")

        join_cfg_raw = base.get("join")
        if join_cfg_raw is None:
            join_cfg: Dict[str, object] = {}
        elif isinstance(join_cfg_raw, dict):
            join_cfg = dict(join_cfg_raw)
        else:
            raise TypeError("Join configuration must be a mapping")

        join_defaults: Dict[str, object] = {"enabled": self.DEFAULT_JOIN_ENABLED, "window": self.DEFAULT_JOIN_WINDOW_POINTS}
        if self.DEFAULT_JOIN_THRESHOLD_ABS is not None:
            join_defaults["threshold"] = self.DEFAULT_JOIN_THRESHOLD_ABS

        join_defaults.update(join_cfg)
        if join_defaults.get("windows") is None:
            join_defaults["windows"] = copy.deepcopy(self.DEFAULT_JOIN_WINDOWS_BY_INSTRUMENT)
        base["join"] = join_defaults

        qc_cfg_raw = base.get("qc")
        if qc_cfg_raw is None:
            qc_cfg: Dict[str, object] = {}
        elif isinstance(qc_cfg_raw, dict):
            qc_cfg = dict(qc_cfg_raw)
        else:
            raise TypeError("QC configuration must be a mapping")

        quiet_cfg = qc_cfg.get("quiet_window")
        if quiet_cfg is None:
            qc_cfg["quiet_window"] = {
                "min": float(self.DEFAULT_QC_QUIET_WINDOW_NM[0]),
                "max": float(self.DEFAULT_QC_QUIET_WINDOW_NM[1]),
            }
        elif not isinstance(quiet_cfg, Mapping):
            raise TypeError("Quiet window configuration must be a mapping")

        base["qc"] = qc_cfg
        return base

    def _resolve_join_windows(
        self, windows_cfg: object, spec: Spectrum
    ) -> Sequence[Mapping[str, float | int | None]] | None:
        if windows_cfg is None:
            return None

        if isinstance(windows_cfg, Mapping):
            instrument = str(spec.meta.get("instrument") or "").lower()
            default_windows = None
            for key, value in windows_cfg.items():
                if value is None:
                    continue
                key_lower = str(key).strip().lower()
                if not key_lower:
                    continue
                if key_lower == "default":
                    default_windows = value
                    continue
                if key_lower in instrument:
                    return copy.deepcopy(value)  # type: ignore[return-value]
            if default_windows is None:
                return None
            return copy.deepcopy(default_windows)  # type: ignore[return-value]

        if isinstance(windows_cfg, Sequence) and not isinstance(windows_cfg, (str, bytes)):
            return copy.deepcopy(windows_cfg)  # type: ignore[return-value]

        raise TypeError("Join windows configuration must be a mapping or sequence")

    def _default_wavelength_limits(self, spec: Spectrum) -> tuple[float, float]:
        instrument = str(spec.meta.get("instrument") or "").lower()
        if "helios gamma" in instrument:
            return self.HELIOS_GAMMA_WAVELENGTH_LIMITS_NM
        return self.GENERIC_WAVELENGTH_LIMITS_NM

    def _effective_domain(
        self, base_domain: dict[str, object] | None, spec: Spectrum
    ) -> dict[str, object] | None:
        limits_min, limits_max = self._default_wavelength_limits(spec)
        domain: dict[str, object]
        if base_domain:
            domain = dict(base_domain)
        else:
            domain = {}

        base_min = domain.get("min")
        base_max = domain.get("max")

        effective_min = limits_min if base_min is None else max(limits_min, float(base_min))
        effective_max = limits_max if base_max is None else min(limits_max, float(base_max))

        if effective_min >= effective_max:
            raise ValueError("Domain minimum must be smaller than maximum")

        domain["min"] = effective_min
        domain["max"] = effective_max
        return domain

    def detect(self, paths):
        return any(
            str(p).lower().endswith((".dsp", ".csv", ".xlsx", ".xls", ".txt"))
            for p in paths
        )

    def load(self, paths: Iterable[str]) -> List[Spectrum]:
        self._reset_report_context()
        spectra: List[Spectrum] = []
        path_objects = [Path(p) for p in paths]
        ingestion_ctx = self._report_context["ingestion"]
        ingestion_ctx["requested_paths"] = [str(path) for path in path_objects]
        manifest_files: Set[Path] = set()
        manifest_index: UvVisPlugin._ManifestIndex | None = None
        manifest_allowed = self._manifest_pipeline_enabled()

        if manifest_allowed:
            manifest_index, manifest_files = self._build_manifest_index(path_objects)
        ingestion_ctx["manifest_files"] = (
            sorted(str(path) for path in manifest_files) if manifest_allowed else []
        )
        ingestion_ctx["manifest_enabled"] = bool(manifest_allowed)

        manifest_entries: List[Dict[str, object]] = []
        data_paths: List[Path] = []
        has_non_manifest_candidate = any(
            not self._is_manifest_file(path) for path in path_objects
        )

        for path_str in paths:
            path = Path(path_str)
            is_manifest = self._is_manifest_file(path)
            if is_manifest and manifest_allowed:
                manifest_entries.extend(self._parse_manifest_file(path))
                continue
            if (
                not manifest_allowed
                and is_manifest
                and has_non_manifest_candidate
            ):
                continue
            data_paths.append(path)

        ingestion_ctx["data_files"] = [str(path) for path in data_paths]
        manifest_lookup = (
            self._build_manifest_lookup(manifest_entries)
            if manifest_allowed
            else {}
        )

        manifest_index_hits = 0
        manifest_lookup_hits = 0
        blank_count = 0
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
                override_meta = self._lookup_queue_override(path)
                if override_meta:
                    for key, value in override_meta.items():
                        if key == "role" and isinstance(value, str):
                            meta[key] = value.strip().lower()
                        elif value is not None:
                            meta[key] = value
                index_hit = False
                lookup_hit = False
                if manifest_allowed and manifest_index:
                    updates = self._lookup_manifest_index(manifest_index, path, meta)
                    if updates:
                        index_hit = True
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
                if manifest_allowed and manifest_lookup:
                    manifest_meta = self._lookup_manifest_entries(
                        manifest_lookup, path, meta
                    )
                    if manifest_meta:
                        lookup_hit = True
                        meta.update(manifest_meta)
                spectra.append(Spectrum(wavelength=wl, intensity=inten, meta=meta))
                if meta.get("role") == "blank":
                    if not meta.get("blank_id"):
                        meta["blank_id"] = (
                            meta.get("sample_id")
                            or meta.get("channel")
                            or path.stem
                        )
                    blank_count += 1
                if index_hit:
                    manifest_index_hits += 1
                if lookup_hit:
                    manifest_lookup_hits += 1

        ingestion_ctx["manifest_entry_count"] = len(manifest_entries) if manifest_allowed else 0
        ingestion_ctx["manifest_index_hits"] = (
            manifest_index_hits if manifest_allowed else 0
        )
        ingestion_ctx["manifest_lookup_hits"] = (
            manifest_lookup_hits if manifest_allowed else 0
        )
        ingestion_ctx["spectra_count"] = len(spectra)
        ingestion_ctx["blank_count"] = blank_count
        ingestion_ctx["sample_count"] = max(len(spectra) - blank_count, 0)
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

        # Manifest precedence (lowest to highest):
        #   global → sample-only → channel-only → file+sample → file+channel.
        # Later merges override earlier ones so the most specific assignment wins.

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

        file_token_set = set(file_tokens)
        sample_token_set = set(sample_tokens)
        channel_token_set = set(channel_tokens)

        candidate_entries: Dict[int, Dict[str, object]] = {}

        for token in file_tokens:
            entry = lookup.get("by_source", {}).get(token)
            if entry is not None:
                candidate_entries.setdefault(id(entry), entry)

        for token in channel_tokens:
            entry = lookup.get("by_channel", {}).get(token)
            if entry is not None:
                candidate_entries.setdefault(id(entry), entry)

        for token in sample_tokens:
            entry = lookup.get("by_sample", {}).get(token)
            if entry is not None:
                candidate_entries.setdefault(id(entry), entry)

        if not candidate_entries:
            return {}

        file_keys = ("file", "filename", "file_name", "source_file", "data_file", "path")
        sample_keys = ("sample_id", "sample", "sample_name")
        channel_keys = ("channel", "channel_id", "channel_name")

        buckets: Dict[str, List[Dict[str, object]]] = {
            "global": [],
            "sample": [],
            "channel": [],
            "file_sample": [],
            "file_channel": [],
        }

        for entry in candidate_entries.values():
            file_value = self._first_manifest_field(entry, file_keys)
            sample_value = self._first_manifest_field(entry, sample_keys)
            channel_value = self._first_manifest_field(entry, channel_keys)

            norm_file = (
                self._normalize_manifest_token(file_value)
                if file_value is not None
                else None
            )
            norm_sample = (
                self._normalize_manifest_token(sample_value)
                if sample_value is not None
                else None
            )
            norm_channel = (
                self._normalize_manifest_token(channel_value)
                if channel_value is not None
                else None
            )

            has_file = norm_file is not None and norm_file in file_token_set
            has_sample = norm_sample is not None and norm_sample in sample_token_set
            has_channel = norm_channel is not None and norm_channel in channel_token_set

            if has_file and has_channel:
                buckets["file_channel"].append(entry)
            elif has_file and has_sample:
                buckets["file_sample"].append(entry)
            elif has_channel:
                buckets["channel"].append(entry)
            elif has_sample:
                buckets["sample"].append(entry)
            else:
                if has_file:
                    if not sample_token_set and not channel_token_set:
                        buckets["global"].append(entry)
                else:
                    buckets["global"].append(entry)

        ordered_entries: List[Dict[str, object]] = []
        for key in ("global", "sample", "channel", "file_sample", "file_channel"):
            for entry in buckets[key]:
                ordered_entries.append(entry)

        if not ordered_entries:
            return {}

        skip_keys = {"file", "filename", "file_name", "data_file", "source_file", "path"}
        merged: Dict[str, object] = {}
        for entry in ordered_entries:
            normalized_entry: Dict[str, object] = {}
            for key, value in entry.items():
                norm_key = self._normalize_manifest_key(key)
                normalized_entry[norm_key] = value

            assignments = self._extract_manifest_assignments(normalized_entry)
            for key, value in assignments.items():
                merged[key] = value

            for key, value in normalized_entry.items():
                if key in skip_keys or key in assignments:
                    continue
                if key == "role" and isinstance(value, str):
                    merged[key] = value.strip().lower()
                else:
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
        mode = normalise_mode(meta.get("mode"))
        if mode:
            meta["mode"] = mode
        else:
            inferred_mode = self._infer_mode_from_headers(df.columns[1:])
            if inferred_mode:
                meta["mode"] = inferred_mode

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
            raw_trace = inten.copy()
            converted, channel_key, updated_mode, original_mode = convert_intensity_to_absorbance(
                inten, column_meta.get("mode")
            )
            inten = converted
            if channel_key:
                channels = dict(column_meta.get("channels") or {})
                channels[channel_key] = raw_trace
                column_meta["channels"] = channels
                if original_mode:
                    column_meta.setdefault("original_mode", original_mode)
            if updated_mode:
                column_meta["mode"] = updated_mode
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
            self._report_context["preprocessing"] = {}
            return []

        recipe = self._apply_recipe_defaults(recipe)
        context = self._report_context
        ingestion_ctx = context.setdefault("ingestion", {})
        ingestion_ctx.setdefault("spectra_count", len(specs))
        blank_inputs = sum(1 for spec in specs if spec.meta.get("role") == "blank")
        ingestion_ctx.setdefault("blank_count", blank_inputs)
        ingestion_ctx.setdefault("sample_count", max(len(specs) - blank_inputs, 0))
        pre_ctx = context.setdefault("preprocessing", {})

        domain_cfg_raw = recipe.get("domain")
        if domain_cfg_raw is None:
            domain_cfg: dict[str, object] | None = None
        elif isinstance(domain_cfg_raw, dict):
            domain_cfg = dict(domain_cfg_raw)
        else:
            raise TypeError("Domain configuration must be a mapping")
        blank_cfg_raw = recipe.get("blank", {})
        if isinstance(blank_cfg_raw, Mapping):
            blank_cfg = dict(blank_cfg_raw)
        else:
            blank_cfg = {}
        match_strategy = pipeline.normalize_blank_match_strategy(
            blank_cfg.get("match_strategy")
        )
        baseline_cfg = recipe.get("baseline", {})
        join_cfg = dict(recipe.get("join", {}))
        despike_cfg = recipe.get("despike", {})
        smoothing_cfg = recipe.get("smoothing", {})
        replicate_cfg = recipe.get("replicates", {})
        qc_cfg = dict(recipe.get("qc", {})) if recipe else {}

        if domain_cfg:
            pre_ctx["domain"] = {
                "min": float(domain_cfg.get("min")) if domain_cfg.get("min") is not None else None,
                "max": float(domain_cfg.get("max")) if domain_cfg.get("max") is not None else None,
            }
        else:
            pre_ctx["domain"] = None
        pre_ctx["blank"] = {
            "subtract": bool(blank_cfg.get("subtract", blank_cfg.get("enabled", True))),
            "require": bool(blank_cfg.get("require", blank_cfg.get("subtract", blank_cfg.get("enabled", True)))),
            "default_blank": blank_cfg.get("default") or blank_cfg.get("fallback"),
            "validate_metadata": bool(blank_cfg.get("validate_metadata", True)),
            "match_strategy": match_strategy,
        }
        pre_ctx["baseline"] = {
            "method": baseline_cfg.get("method"),
            "parameters": {
                "lambda": baseline_cfg.get("lam", baseline_cfg.get("lambda")),
                "p": baseline_cfg.get("p"),
                "iterations": baseline_cfg.get("iterations", baseline_cfg.get("niter")),
            },
        }
        pre_ctx["join"] = {
            "enabled": bool(join_cfg.get("enabled")),
            "window": int(join_cfg.get("window", 10)),
            "threshold": self._coerce_float(join_cfg.get("threshold")),
        }
        pre_ctx["despike"] = {
            "enabled": bool(despike_cfg.get("enabled")),
            "window": despike_cfg.get("window"),
            "zscore": self._coerce_float(despike_cfg.get("zscore")),
        }
        pre_ctx["smoothing"] = {
            "enabled": bool(smoothing_cfg.get("enabled")),
            "window": int(smoothing_cfg.get("window", 5)) if smoothing_cfg.get("window") is not None else None,
            "polyorder": int(smoothing_cfg.get("polyorder", 2)) if smoothing_cfg.get("polyorder") is not None else None,
        }
        pre_ctx["replicates"] = {
            "average": bool(replicate_cfg.get("average", True)),
            "outlier": replicate_cfg.get("outlier"),
        }
        quiet_window_cfg = qc_cfg.get("quiet_window") if isinstance(qc_cfg.get("quiet_window"), dict) else None
        if isinstance(quiet_window_cfg, dict):
            pre_ctx["quiet_window"] = {
                "min": self._coerce_float(quiet_window_cfg.get("min")),
                "max": self._coerce_float(quiet_window_cfg.get("max")),
            }

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

        def _normalize_identifier(value: object | None) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        stage_one: list[Spectrum] = []
        for spec in specs:
            effective_domain = self._effective_domain(domain_cfg, spec)
            coerced = pipeline.coerce_domain(spec, effective_domain)
            processed = coerced
            joins: list[int] = []
            if join_cfg.get("enabled"):
                join_window = int(join_cfg.get("window", 10))
                join_windows = self._resolve_join_windows(join_cfg.get("windows"), processed)
                joins = pipeline.detect_joins(
                    processed.wavelength,
                    processed.intensity,
                    threshold=join_cfg.get("threshold"),
                    window=join_window,
                    windows=join_windows,
                )
                bounds = join_cfg.get("offset_bounds")
                if bounds is None:
                    bounds = (
                        join_cfg.get("min_offset"),
                        join_cfg.get("max_offset"),
                    )
                processed = pipeline.correct_joins(
                    processed,
                    joins,
                    window=join_window,
                    offset_bounds=bounds if bounds is not None else None,
                )
            if despike_cfg.get("enabled"):
                processed = pipeline.despike_spectrum(
                    processed,
                    zscore=despike_cfg.get("zscore", 5.0),
                    window=despike_cfg.get("window", 5),
                    join_indices=joins,
                )
            processed.meta["join_indices"] = tuple(joins)
            stage_one.append(processed)

        blanks_stage = [spec for spec in stage_one if spec.meta.get("role") == "blank"]
        samples_stage = [spec for spec in stage_one if spec.meta.get("role") != "blank"]

        if average_replicates and blanks_stage:
            blanks_avg, blank_map_by_key = pipeline.average_replicates(
                blanks_stage,
                return_mapping=True,
                outlier=outlier_cfg,
                key_func=lambda spec, strategy=match_strategy: pipeline.blank_replicate_key(
                    spec, strategy
                ),
            )
        else:
            blanks_avg = blanks_stage
            blank_map_by_key = {
                pipeline.blank_replicate_key(b, match_strategy): b for b in blanks_stage
            }

        blank_lookup: dict[str, Spectrum] = {}
        for blank_spec in blanks_avg:
            identifiers = pipeline.blank_match_identifiers(blank_spec, match_strategy)
            if not identifiers:
                continue
            for ident in identifiers:
                if ident not in blank_lookup:
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
        fallback_identifier = _normalize_identifier(fallback_blank)
        for spec in samples_stage:
            working = spec
            candidates: list[str] = []
            if match_strategy == "cuvette_slot":
                slot_value = _normalize_identifier(spec.meta.get("cuvette_slot"))
                if slot_value:
                    candidates.append(slot_value)
            blank_id_value = _normalize_identifier(spec.meta.get("blank_id"))
            if blank_id_value and blank_id_value not in candidates:
                candidates.append(blank_id_value)
            else:
                derived_sources: list[str] = []
                for key in ("sample_id", "channel"):
                    derived = _normalize_identifier(spec.meta.get(key))
                    if derived and derived not in derived_sources:
                        derived_sources.append(derived)
                suffix = "_indref"
                for derived in derived_sources:
                    if derived not in candidates:
                        candidates.append(derived)
                    if not derived.lower().endswith(suffix):
                        indref_candidate = f"{derived}_indRef"
                        if indref_candidate not in candidates:
                            candidates.append(indref_candidate)
            if fallback_identifier and fallback_identifier not in candidates:
                candidates.append(fallback_identifier)

            selected_identifier = candidates[0] if candidates else None
            blank_spec = None
            for candidate in candidates:
                lookup = blank_lookup.get(candidate)
                if lookup is not None:
                    blank_spec = lookup
                    selected_identifier = candidate
                    break

            if subtract_blank:
                if blank_spec is not None:
                    audit = self._validate_blank_pairing(
                        working,
                        blank_spec,
                        blank_cfg,
                        enforce=validate_flag,
                    )
                    working = pipeline.subtract_blank(working, blank_spec, audit=audit)
                elif selected_identifier:
                    if require_blank:
                        raise ValueError(
                            f"No blank spectrum available for identifier '{selected_identifier}'"
                        )
                    audit = self._build_missing_blank_audit(working, selected_identifier)
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
                elif selected_identifier:
                    audit = self._build_missing_blank_audit(working, selected_identifier)
                    working = self._attach_blank_audit(working, audit)

            if baseline_cfg.get("method"):
                method = baseline_cfg.get("method")
                params = {
                    "lam": baseline_cfg.get("lam", baseline_cfg.get("lambda", 1e5)),
                    "p": baseline_cfg.get("p", 0.01),
                    "niter": baseline_cfg.get("niter", 10),
                    "iterations": baseline_cfg.get("iterations", 24),
                }
                anchor_cfg = (
                    baseline_cfg.get("anchor")
                    or baseline_cfg.get("anchor_windows")
                    or baseline_cfg.get("anchors")
                )
                if anchor_cfg is None:
                    anchor_cfg = baseline_cfg.get("zeroing")
                params["anchor"] = anchor_cfg
                working = pipeline.apply_baseline(working, method, **params)

            if smoothing_cfg.get("enabled"):
                join_indices = working.meta.get("join_indices")
                working = pipeline.smooth_spectrum(
                    working,
                    window=int(smoothing_cfg.get("window", 5)),
                    polyorder=int(smoothing_cfg.get("polyorder", 2)),
                    join_indices=join_indices,
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
                        anchor=(
                            baseline_cfg.get("anchor")
                            or baseline_cfg.get("anchor_windows")
                            or baseline_cfg.get("anchors")
                            or baseline_cfg.get("zeroing")
                        ),
                    )
                    for blank in blanks_working
                ]
            if smoothing_cfg.get("enabled"):
                blanks_working = [
                    pipeline.smooth_spectrum(
                        blank,
                        window=int(smoothing_cfg.get("window", 5)),
                        polyorder=int(smoothing_cfg.get("polyorder", 2)),
                        join_indices=blank.meta.get("join_indices"),
                    )
                    for blank in blanks_working
                ]
            blanks_final = blanks_working
            blank_map_by_key = {
                pipeline.blank_replicate_key(blank, match_strategy): blank
                for blank in blanks_final
            }

        order_keys: list[tuple[str, str]] = []
        for spec in stage_one:
            if spec.meta.get("role") == "blank":
                key = pipeline.blank_replicate_key(spec, match_strategy)
            else:
                key = pipeline.replicate_key(spec)
            if key not in order_keys:
                order_keys.append(key)

        processed_map = {**blank_map_by_key, **sample_map}
        final_specs = [processed_map[key] for key in order_keys if key in processed_map]
        processed_blanks = [spec for spec in final_specs if spec.meta.get("role") == "blank"]
        processed_samples_only = [spec for spec in final_specs if spec.meta.get("role") != "blank"]
        pre_ctx["counts"] = {
            "input_samples": len(samples_stage),
            "input_blanks": len(blanks_stage),
            "processed_samples": len(processed_samples_only),
            "processed_blanks": len(processed_blanks),
            "replicate_groups": len(sample_map),
        }
        blank_identifiers: Set[str] = set()
        for spec in processed_blanks:
            ident = pipeline.blank_identifier(spec, match_strategy)
            if ident:
                blank_identifiers.add(str(ident))
        pre_ctx["blank_ids"] = sorted(blank_identifiers)
        context["ingestion"]["spectra_count"] = len(final_specs)
        context["ingestion"]["blank_count"] = len(processed_blanks)
        context["ingestion"]["sample_count"] = len(processed_samples_only)
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

    @staticmethod
    def _calibration_signal_from_spec(
        spec: Spectrum,
        *,
        wavelength: float,
        bandwidth: float | None,
    ) -> float:
        wl = np.asarray(spec.wavelength, dtype=float)
        intensity = np.asarray(spec.intensity, dtype=float)
        mask = np.isfinite(wl) & np.isfinite(intensity)
        if mask.sum() < 2:
            return float("nan")
        wl = wl[mask]
        intensity = intensity[mask]
        if wavelength < float(np.min(wl)) or wavelength > float(np.max(wl)):
            return float("nan")
        if bandwidth is not None:
            width = float(bandwidth)
        else:
            width = 0.0
        if width > 0:
            half = width / 2.0
            region = (wl >= wavelength - half) & (wl <= wavelength + half)
            if np.count_nonzero(region) >= 1:
                window_values = intensity[region]
                if window_values.size == 0:
                    return float("nan")
                return float(np.nanmean(window_values))
        return float(np.interp(wavelength, wl, intensity))

    def _collect_calibration_measurements(
        self,
        sample_specs: Sequence[Spectrum],
        *,
        wavelength: float,
        bandwidth: float | None,
        default_pathlength: float | None,
    ) -> Dict[str, Any]:
        raw_values: List[float] = []
        norm_values: List[float] = []
        pathlengths: List[float] = []
        for spec in sample_specs:
            raw = self._calibration_signal_from_spec(
                spec,
                wavelength=wavelength,
                bandwidth=bandwidth,
            )
            if not math.isfinite(raw):
                continue
            path = self._coerce_float(spec.meta.get("pathlength_cm")) if spec.meta else None
            if path is None or not math.isfinite(path) or path <= 0:
                path = default_pathlength
            if path is None or not math.isfinite(path) or path <= 0:
                path = 1.0
            raw_values.append(float(raw))
            norm_values.append(float(raw) / float(path))
            pathlengths.append(float(path))
        if not raw_values:
            return {
                "raw_absorbance": float("nan"),
                "response": float("nan"),
                "responses": [],
                "raw_values": [],
                "pathlengths": [],
                "pathlength_cm": float(default_pathlength) if default_pathlength is not None else float("nan"),
                "replicates": 0,
            }
        return {
            "raw_absorbance": float(np.nanmean(raw_values)),
            "response": float(np.nanmean(norm_values)),
            "responses": [float(val) for val in norm_values],
            "raw_values": [float(val) for val in raw_values],
            "pathlengths": [float(val) for val in pathlengths],
            "pathlength_cm": float(np.nanmean(pathlengths)) if pathlengths else (
                float(default_pathlength) if default_pathlength is not None else float("nan")
            ),
            "replicates": len(raw_values),
        }

    @staticmethod
    def _normalize_standard_entries(entries: Any) -> List[Dict[str, Any]]:
        if entries is None:
            return []
        normalized: List[Dict[str, Any]] = []
        if isinstance(entries, dict):
            items: Sequence[Any] = [
                {"sample_id": key, "concentration": value} for key, value in entries.items()
            ]
        elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            items = entries
        else:
            items = [entries]
        for item in items:
            if isinstance(item, dict):
                sample_id = item.get("sample_id") or item.get("id") or item.get("name")
                if sample_id is None:
                    continue
                normalized.append({"sample_id": str(sample_id), **{k: v for k, v in item.items() if k not in {"sample_id", "id", "name"}}})
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                normalized.append({"sample_id": str(item[0]), "concentration": item[1]})
        return normalized

    @staticmethod
    def _normalize_unknown_entries(entries: Any) -> List[Dict[str, Any]]:
        if entries is None:
            return []
        normalized: List[Dict[str, Any]] = []
        if isinstance(entries, dict):
            items: Sequence[Any] = []
            for key, value in entries.items():
                if isinstance(value, dict):
                    payload = dict(value)
                else:
                    payload = {"dilution_factor": value} if isinstance(value, (int, float)) else {}
                payload["sample_id"] = key
                items.append(payload)
        elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            items = entries
        else:
            items = [entries]
        for item in items:
            if isinstance(item, dict):
                sample_id = item.get("sample_id") or item.get("id") or item.get("name")
                if sample_id is None:
                    continue
                payload = dict(item)
                payload["sample_id"] = str(sample_id)
                normalized.append(payload)
            else:
                normalized.append({"sample_id": str(item)})
        return normalized

    @staticmethod
    def _assign_calibration_payload(
        sample_specs: Sequence[Spectrum] | None,
        qc_rows: Sequence[Dict[str, Any]] | None,
        target_name: str,
        payload: Dict[str, Any],
    ) -> None:
        for spec in sample_specs or []:
            if spec.meta is None:
                spec.meta = {}
            calibration_meta = spec.meta.setdefault("calibration", {})
            calibration_meta[target_name] = dict(payload)
        for row in qc_rows or []:
            row.setdefault("calibration", {})
            row["calibration"][target_name] = dict(payload)

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
        *,
        window: int | None = None,
        polyorder: int | None = None,
    ) -> Dict[str, np.ndarray]:
        if not enabled or intensity.size < 3:
            return {}

        safe_y = self._fill_nan(intensity)
        safe_wl = self._fill_nan(wl)

        n = safe_y.size
        requested_window = int(window) if window is not None else None
        if requested_window is not None and requested_window <= 0:
            requested_window = None

        if requested_window is None:
            default_window = 5
            if n < default_window:
                default_window = n if n % 2 else n - 1
            if default_window < 3:
                default_window = 3
            eff_window = default_window
        else:
            eff_window = requested_window

        if eff_window % 2 == 0:
            # Prefer shrinking to preserve the requested neighbourhood when possible.
            eff_window -= 1

        if eff_window > n:
            eff_window = n if n % 2 else n - 1

        while eff_window > n and eff_window >= 3:
            eff_window -= 2

        if eff_window < 3:
            eff_window = 3 if n >= 3 else eff_window
        if eff_window < 3 or eff_window > n:
            return {}

        base_poly = int(polyorder) if polyorder is not None else 2
        if base_poly < 0:
            base_poly = 0

        min_poly_for_second = 2
        if base_poly < min_poly_for_second:
            base_poly = min_poly_for_second

        if base_poly >= eff_window:
            base_poly = eff_window - 1

        if base_poly < min_poly_for_second:
            return {}

        diffs = np.diff(safe_wl)
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size:
            delta = float(np.nanmedian(finite_diffs))
            if not np.isfinite(delta) or delta == 0:
                delta = float(np.nanmedian(np.abs(finite_diffs)))
        else:
            delta = 1.0

        if not np.isfinite(delta) or delta == 0:
            delta = 1.0

        delta = abs(delta)
        if delta == 0:
            delta = 1.0

        first = savgol_filter(
            safe_y,
            window_length=eff_window,
            polyorder=base_poly,
            deriv=1,
            delta=delta,
            mode="interp",
        )
        second = savgol_filter(
            safe_y,
            window_length=eff_window,
            polyorder=base_poly,
            deriv=2,
            delta=delta,
            mode="interp",
        )
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
        widths, width_heights, left_ips, right_ips = peak_widths(intensity, peaks, rel_height=0.5)
        x_indices = np.arange(wl.size)
        peak_rows: List[Dict[str, float]] = []
        for idx, peak in enumerate(peaks):
            raw_peak_wl = float(wl[peak])
            raw_peak_height = float(intensity[peak])
            left_wl = float(np.interp(left_ips[idx], x_indices, wl))
            right_wl = float(np.interp(right_ips[idx], x_indices, wl))
            raw_fwhm = max(right_wl - left_wl, 0.0)

            refinement = self._fit_peak_quadratic(
                wl,
                intensity,
                peak,
                float(width_heights[idx]) if width_heights is not None else None,
            )

            if refinement is not None:
                refined_wl = refinement["wavelength"]
                refined_height = refinement["height"]
                left_half = refinement["left_half_max"]
                right_half = refinement["right_half_max"]
                refined_fwhm = max(right_half - left_half, 0.0) if left_half is not None and right_half is not None else raw_fwhm
            else:
                refined_wl = raw_peak_wl
                refined_height = raw_peak_height
                refined_fwhm = raw_fwhm
                left_half = None
                right_half = None

            peak_rows.append({
                "wavelength": float(refined_wl),
                "height": float(refined_height),
                "fwhm": float(refined_fwhm),
                "prominence": float(prominences[idx]) if prominences is not None else float("nan"),
                "raw_wavelength": raw_peak_wl,
                "raw_height": raw_peak_height,
                "raw_fwhm": raw_fwhm,
                "quadratic_refined": refinement is not None,
                "quadratic_coefficients": refinement.get("coefficients") if refinement is not None else None,
                "quadratic_window_indices": refinement.get("window_indices") if refinement is not None else None,
                "quadratic_window_wavelengths": refinement.get("window_wavelengths") if refinement is not None else None,
                "quadratic_half_height": refinement.get("half_height") if refinement is not None else None,
                "quadratic_left_half_max": left_half,
                "quadratic_right_half_max": right_half,
            })
        return peak_rows

    def _fit_peak_quadratic(
        self,
        wl: np.ndarray,
        intensity: np.ndarray,
        peak_idx: int,
        half_height: float | None,
    ) -> Optional[Dict[str, object]]:
        """Refine peak characteristics using a local quadratic fit.

        The method considers a neighbourhood around ``peak_idx`` and fits a
        parabola to derive sub-sample peak metrics. If a reliable quadratic
        cannot be obtained the function returns ``None``.
        """

        if wl.size < 3 or intensity.size < 3:
            return None

        # Select a symmetric window (up to +/-2 neighbours) while remaining in
        # bounds to capture the local curvature of the peak.
        start = max(peak_idx - 2, 0)
        stop = min(peak_idx + 3, wl.size)
        if stop - start < 3:
            # Expand window if near the boundaries.
            deficit = 3 - (stop - start)
            start = max(start - deficit, 0)
            stop = min(start + 3, wl.size)
        indices = np.arange(start, stop)
        if indices.size < 3:
            return None

        x = wl[indices].astype(float)
        y = intensity[indices].astype(float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return None

        try:
            coeffs = np.polyfit(x, y, 2)
        except Exception:  # pragma: no cover - guard against unexpected polyfit errors
            return None

        a, b, c = coeffs
        if abs(a) < 1e-12:
            return None

        vertex_x = -b / (2.0 * a)
        # Require the vertex to fall within the chosen window to avoid
        # extrapolation that could degrade the estimate.
        if vertex_x < x[0] - 1e-9 or vertex_x > x[-1] + 1e-9:
            return None

        vertex_y = np.polyval(coeffs, vertex_x)
        if not np.isfinite(vertex_y):
            return None

        left_half: Optional[float] = None
        right_half: Optional[float] = None
        if half_height is not None and np.isfinite(half_height):
            # Solve the quadratic for the half-height contour. The returned
            # roots are used as refined half-maximum crossings when they fall
            # on either side of the apex.
            roots = np.roots([a, b, c - half_height])
            real_roots = roots[np.isreal(roots)].real
            if real_roots.size:
                left_candidates = real_roots[real_roots <= vertex_x]
                right_candidates = real_roots[real_roots >= vertex_x]
                if left_candidates.size:
                    left_half = float(np.max(left_candidates))
                if right_candidates.size:
                    right_half = float(np.min(right_candidates))

        return {
            "wavelength": float(vertex_x),
            "height": float(vertex_y),
            "left_half_max": left_half,
            "right_half_max": right_half,
            "coefficients": [float(val) for val in coeffs],
            "window_indices": [int(i) for i in indices],
            "window_wavelengths": [float(val) for val in x],
            "half_height": float(half_height) if half_height is not None and np.isfinite(half_height) else None,
        }

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

    def _perform_calibration(
        self,
        specs: List[Spectrum],
        qc_rows: List[Dict[str, Any]],
        recipe: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        calibration_cfg = dict(recipe.get("calibration", {})) if recipe else {}
        enabled = bool(calibration_cfg.get("enabled", True))
        if not calibration_cfg or not enabled:
            status = "disabled" if calibration_cfg and not enabled else "not_configured"
            return {"enabled": False, "status": status, "targets": []}

        lod_multiplier = float(calibration_cfg.get("lod_multiplier", 3.0))
        loq_multiplier = float(calibration_cfg.get("loq_multiplier", 10.0))
        default_bandwidth = self._coerce_float(calibration_cfg.get("bandwidth"))

        sample_lookup: Dict[str, List[Spectrum]] = {}
        sample_roles: Dict[str, str] = {}
        for spec in specs:
            sample_id = self._safe_sample_id(spec, f"spec_{id(spec)}")
            sample_lookup.setdefault(sample_id, []).append(spec)
            role = str(spec.meta.get("role", "")) if spec.meta else ""
            if sample_id not in sample_roles and role:
                sample_roles[sample_id] = role.lower()

        qc_lookup: Dict[str, List[Dict[str, Any]]] = {}
        for row in qc_rows:
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            qc_lookup.setdefault(str(sample_id), []).append(row)

        raw_targets = calibration_cfg.get("targets")
        if raw_targets is None:
            raw_targets = []
        elif not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
            raw_targets = [raw_targets]

        target_results: List[Dict[str, Any]] = []
        for index, target_cfg_raw in enumerate(raw_targets):
            if isinstance(target_cfg_raw, dict):
                target_cfg = dict(target_cfg_raw)
            else:
                target_cfg = {"name": target_cfg_raw}

            name = str(target_cfg.get("name") or f"target_{index + 1}")
            wavelength_val = self._coerce_float(target_cfg.get("wavelength"))
            if wavelength_val is None:
                target_results.append(
                    {
                        "name": name,
                        "status": "failed",
                        "errors": ["Target wavelength was not provided."],
                        "warnings": [],
                        "standards": [],
                        "unknowns": [],
                        "fit": None,
                        "wavelength": None,
                        "bandwidth": None,
                        "pathlength_cm": None,
                    }
                )
                continue

            bandwidth = self._coerce_float(target_cfg.get("bandwidth"))
            if bandwidth is None:
                bandwidth = default_bandwidth

            pathlength = self._coerce_float(target_cfg.get("pathlength_cm"))
            if pathlength is None:
                pathlength = self._coerce_float(calibration_cfg.get("pathlength_cm"))
            if pathlength is None or not math.isfinite(pathlength) or pathlength <= 0:
                pathlength = 1.0

            fit_intercept = bool(target_cfg.get("fit_intercept", True))
            r2_threshold = self._coerce_float(target_cfg.get("r2_threshold"))
            if r2_threshold is None:
                r2_threshold = self._coerce_float(calibration_cfg.get("r2_threshold"))

            standards_entries = self._normalize_standard_entries(target_cfg.get("standards"))
            if not standards_entries and calibration_cfg.get("standards") is not None:
                standards_entries = self._normalize_standard_entries(calibration_cfg.get("standards"))

            detected_standards: Dict[str, float] = {}
            if not standards_entries:
                for sample_id, sample_specs in sample_lookup.items():
                    role = sample_roles.get(sample_id, "").lower()
                    if "standard" not in role:
                        continue
                    concentration_val: float | None = None
                    for spec in sample_specs:
                        meta = spec.meta or {}
                        concentration_val = (
                            self._coerce_float(meta.get("concentration"))
                            or self._coerce_float(meta.get("target_concentration"))
                            or self._coerce_float(meta.get("standard_concentration"))
                        )
                        if concentration_val is not None:
                            break
                    if concentration_val is None:
                        continue
                    if sample_id not in detected_standards:
                        detected_standards[sample_id] = concentration_val
                standards_entries = [
                    {"sample_id": sample_id, "concentration": conc}
                    for sample_id, conc in detected_standards.items()
                ]

            target_result: Dict[str, Any] = {
                "name": name,
                "status": "not_computed",
                "errors": [],
                "warnings": [],
                "standards": [],
                "unknowns": [],
                "fit": None,
                "wavelength": float(wavelength_val),
                "bandwidth": float(bandwidth) if bandwidth is not None else None,
                "pathlength_cm": float(pathlength),
            }

            if not standards_entries:
                target_result["status"] = "failed"
                target_result["errors"].append("No calibration standards were provided.")
                target_results.append(target_result)
                continue

            standard_points: List[Dict[str, Any]] = []
            for entry in standards_entries:
                sample_id = str(entry.get("sample_id")) if entry.get("sample_id") is not None else None
                if not sample_id:
                    continue
                concentration_val = self._coerce_float(entry.get("concentration"))
                if concentration_val is None or not math.isfinite(concentration_val):
                    meta_specs = sample_lookup.get(sample_id, [])
                    for spec in meta_specs:
                        meta = spec.meta or {}
                        concentration_val = (
                            self._coerce_float(meta.get("concentration"))
                            or self._coerce_float(meta.get("target_concentration"))
                            or self._coerce_float(meta.get("standard_concentration"))
                        )
                        if concentration_val is not None and math.isfinite(concentration_val):
                            break
                sample_specs = sample_lookup.get(sample_id, [])
                measurement = self._collect_calibration_measurements(
                    sample_specs,
                    wavelength=float(wavelength_val),
                    bandwidth=bandwidth,
                    default_pathlength=pathlength,
                )
                point = {
                    "sample_id": sample_id,
                    "concentration": float(concentration_val) if concentration_val is not None else float("nan"),
                    "response": measurement["response"],
                    "raw_absorbance": measurement["raw_absorbance"],
                    "pathlength_cm": measurement["pathlength_cm"],
                    "replicates": measurement["replicates"],
                    "responses": measurement["responses"],
                    "included": bool(entry.get("include", True)),
                }
                weight_val = self._coerce_float(entry.get("weight"))
                if weight_val is not None and math.isfinite(weight_val):
                    point["weight"] = float(weight_val)
                if weight_val is not None and math.isfinite(weight_val) and weight_val > 0:
                    point["applied_weight"] = float(weight_val)
                else:
                    point["applied_weight"] = 1.0
                if not sample_specs:
                    target_result["errors"].append(f"Standard {sample_id} was not found in the processed spectra.")
                if concentration_val is None or not math.isfinite(point["concentration"]):
                    target_result["errors"].append(f"Standard {sample_id} is missing a concentration value.")
                if not math.isfinite(point["response"]):
                    target_result["warnings"].append(f"Standard {sample_id} did not yield a valid absorbance measurement.")
                standard_points.append(point)

            target_result["standards"] = standard_points

            unknown_entries = self._normalize_unknown_entries(target_cfg.get("unknowns"))
            if not unknown_entries and calibration_cfg.get("unknowns") is not None:
                unknown_entries = self._normalize_unknown_entries(calibration_cfg.get("unknowns"))

            if not unknown_entries:
                for sample_id in sample_lookup.keys():
                    if sample_id in {point["sample_id"] for point in standard_points}:
                        continue
                    role = sample_roles.get(sample_id, "").lower()
                    if role == "blank":
                        continue
                    unknown_entries.append({"sample_id": sample_id})

            unknown_measurements: List[Dict[str, Any]] = []
            for entry in unknown_entries:
                sample_id = str(entry.get("sample_id")) if entry.get("sample_id") is not None else None
                if not sample_id:
                    continue
                sample_specs = sample_lookup.get(sample_id, [])
                measurement = self._collect_calibration_measurements(
                    sample_specs,
                    wavelength=float(wavelength_val),
                    bandwidth=bandwidth,
                    default_pathlength=pathlength,
                )
                payload_base = {
                    "role": "unknown",
                    "target": name,
                    "response": measurement["response"],
                    "raw_absorbance": measurement["raw_absorbance"],
                    "pathlength_cm": measurement["pathlength_cm"],
                    "replicates": measurement["replicates"],
                }
                dilution_val = self._coerce_float(entry.get("dilution_factor") or entry.get("dilution"))
                if dilution_val is not None and math.isfinite(dilution_val):
                    payload_base["dilution_factor"] = float(dilution_val)
                unknown_measurements.append(
                    {
                        "sample_id": sample_id,
                        "entry": entry,
                        "measurement": measurement,
                        "payload": payload_base,
                    }
                )

            valid_points = [
                point
                for point in standard_points
                if point.get("included", True)
                and math.isfinite(point.get("response", float("nan")))
                and math.isfinite(point.get("concentration", float("nan")))
            ]

            if len(valid_points) < 2:
                target_result["status"] = "failed"
                target_result["errors"].append("At least two calibration standards with valid responses are required.")
                target_results.append(target_result)
                for point in standard_points:
                    payload = {
                        "role": "standard",
                        "status": "no_model",
                        "target": name,
                        "concentration": point.get("concentration"),
                        "response": point.get("response"),
                        "raw_absorbance": point.get("raw_absorbance"),
                        "pathlength_cm": point.get("pathlength_cm"),
                        "replicates": point.get("replicates"),
                        "included": point.get("included", True),
                    }
                    self._assign_calibration_payload(sample_lookup.get(point["sample_id"]), qc_lookup.get(point["sample_id"]), name, payload)
                target_result["unknowns"] = []
                for unknown in unknown_measurements:
                    payload = dict(unknown["payload"])
                    payload.update(
                        {
                            "status": "no_model",
                            "predicted_concentration": float("nan"),
                            "predicted_concentration_std_err": float("nan"),
                            "within_range": False,
                            "lod_flag": None,
                        }
                    )
                    self._assign_calibration_payload(
                        sample_lookup.get(unknown["sample_id"]),
                        qc_lookup.get(unknown["sample_id"]),
                        name,
                        payload,
                    )
                    target_result["unknowns"].append(payload | {"sample_id": unknown["sample_id"]})
                continue

            x = np.array([point["concentration"] for point in valid_points], dtype=float)
            y = np.array([point["response"] for point in valid_points], dtype=float)
            weight_values: List[float] = []
            weighting_applied = False
            for point in valid_points:
                raw_weight = point.get("weight")
                if raw_weight is not None and math.isfinite(raw_weight) and raw_weight > 0:
                    weight = float(raw_weight)
                    weighting_applied = True
                else:
                    weight = 1.0
                point["applied_weight"] = float(weight)
                weight_values.append(weight)
            for point in standard_points:
                if "applied_weight" not in point or not math.isfinite(point.get("applied_weight", float("nan"))):
                    raw_weight = point.get("weight")
                    if raw_weight is not None and math.isfinite(raw_weight) and raw_weight > 0:
                        point["applied_weight"] = float(raw_weight)
                    else:
                        point["applied_weight"] = 1.0
            weights_arr = np.array(weight_values, dtype=float) if weighting_applied else None

            model_valid = False
            slope = float("nan")
            intercept = float("nan")
            if not fit_intercept:
                if weights_arr is not None:
                    numerator = float(np.dot(weights_arr, x * y))
                    denominator = float(np.dot(weights_arr, x * x))
                else:
                    numerator = float(np.dot(x, y))
                    denominator = float(np.dot(x, x))
                if math.isfinite(denominator) and denominator > 0:
                    slope = numerator / denominator
                    intercept = 0.0
                    model_valid = math.isfinite(slope)
            else:
                try:
                    if weights_arr is not None:
                        slope, intercept = np.polyfit(x, y, 1, w=weights_arr)
                    else:
                        slope, intercept = np.polyfit(x, y, 1)
                    model_valid = math.isfinite(slope) and math.isfinite(intercept)
                except Exception as exc:  # pragma: no cover - numpy failure unlikely
                    target_result["errors"].append(f"Regression failed: {exc}")

            if not model_valid or not math.isfinite(slope):
                target_result["status"] = "failed"
                target_result["errors"].append("Calibration regression could not be computed.")
            elif slope <= 0:
                target_result["status"] = "failed"
                target_result["errors"].append("Calibration slope must be positive for Beer-Lambert model.")

            y_pred = slope * x + intercept if math.isfinite(slope) else np.full_like(y, np.nan)
            residuals = y - y_pred
            if weights_arr is not None:
                ss_res = float(np.nansum(weights_arr * (residuals**2)))
                try:
                    mean_y = float(np.average(y, weights=weights_arr))
                except ZeroDivisionError:
                    mean_y = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
                ss_tot = float(np.nansum(weights_arr * (y - mean_y) ** 2))
            else:
                ss_res = float(np.nansum((residuals) ** 2))
                mean_y = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
                ss_tot = float(np.nansum((y - mean_y) ** 2))
            if ss_tot <= 0:
                r_squared = 1.0 if ss_res <= 1e-12 else 0.0
            else:
                r_squared = max(0.0, 1.0 - ss_res / ss_tot)

            if weights_arr is not None:
                try:
                    weight_sum = float(np.nansum(weights_arr))
                except Exception:  # pragma: no cover - defensive
                    weight_sum = float(len(valid_points))
                dof_raw = weight_sum - (1.0 if not fit_intercept else 2.0)
            else:
                dof_raw = float(len(valid_points) - (1 if not fit_intercept else 2))
            if not math.isfinite(dof_raw):
                dof_raw = float(len(valid_points) - (1 if not fit_intercept else 2))
            dof = max(float(dof_raw), 1.0)
            residual_variance = ss_res / dof if dof > 0 else float("nan")
            residual_std = math.sqrt(residual_variance) if math.isfinite(residual_variance) else float("nan")
            lod = (lod_multiplier * residual_std / slope) if model_valid and slope > 0 else float("nan")
            loq = (loq_multiplier * residual_std / slope) if model_valid and slope > 0 else float("nan")
            min_conc = float(np.nanmin(x)) if x.size else float("nan")
            max_conc = float(np.nanmax(x)) if x.size else float("nan")

            if r2_threshold is not None and math.isfinite(r2_threshold) and r_squared < r2_threshold:
                target_result["errors"].append(
                    f"Calibration R^2 {r_squared:.4f} is below threshold {r2_threshold:.4f}."
                )
                target_result["status"] = "failed"

            for point in standard_points:
                conc_val = point.get("concentration")
                if math.isfinite(conc_val) and math.isfinite(slope):
                    predicted = slope * conc_val + intercept
                else:
                    predicted = float("nan")
                point["predicted_response"] = float(predicted)
                if math.isfinite(point.get("response", float("nan"))) and math.isfinite(predicted):
                    point["residual"] = float(point["response"] - predicted)
                else:
                    point["residual"] = float("nan")
                payload = {
                    "role": "standard",
                    "status": "ok" if target_result.get("status") not in {"failed"} else "no_model",
                    "target": name,
                    "concentration": point.get("concentration"),
                    "response": point.get("response"),
                    "raw_absorbance": point.get("raw_absorbance"),
                    "pathlength_cm": point.get("pathlength_cm"),
                    "replicates": point.get("replicates"),
                    "included": point.get("included", True),
                    "predicted_response": point.get("predicted_response"),
                    "residual": point.get("residual"),
                    "weight": point.get("weight"),
                    "applied_weight": point.get("applied_weight"),
                }
                if target_result.get("status") == "failed":
                    payload["status"] = "no_model"
                self._assign_calibration_payload(
                    sample_lookup.get(point["sample_id"]),
                    qc_lookup.get(point["sample_id"]),
                    name,
                    payload,
                )

            fit_summary = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "residual_std": float(residual_std),
                "residual_dof": float(dof),
                "lod": float(lod),
                "loq": float(loq),
                "points": len(valid_points),
                "min_concentration": min_conc,
                "max_concentration": max_conc,
                "weights": {
                    str(point.get("sample_id")): float(point.get("applied_weight", 1.0))
                    for point in valid_points
                },
                "weighting_applied": bool(weighting_applied),
            }
            concentration_std = float("nan")
            if math.isfinite(residual_std) and math.isfinite(slope) and slope != 0:
                concentration_std = abs(residual_std / slope)
            fit_summary["concentration_std_per_response"] = float(concentration_std)
            target_result["fit"] = fit_summary

            unknown_results: List[Dict[str, Any]] = []
            for unknown in unknown_measurements:
                sample_id = unknown["sample_id"]
                measurement = unknown["measurement"]
                payload = dict(unknown["payload"])
                response_val = measurement["response"]
                model_available = target_result.get("status") not in {"failed"} and math.isfinite(slope) and slope != 0
                predicted_conc = float("nan")
                predicted_uncertainty = float("nan")
                status = "no_model"
                if model_available:
                    predicted_conc = (response_val - intercept) / slope
                    dilution_val = payload.get("dilution_factor")
                    if dilution_val is not None and math.isfinite(dilution_val):
                        predicted_conc *= float(dilution_val)
                    status = "ok"
                    if not math.isfinite(predicted_conc):
                        status = "invalid"
                    if math.isfinite(concentration_std):
                        predicted_uncertainty = float(concentration_std)
                within_range = (
                    math.isfinite(predicted_conc)
                    and math.isfinite(fit_summary["min_concentration"])
                    and math.isfinite(fit_summary["max_concentration"])
                    and fit_summary["min_concentration"] <= predicted_conc <= fit_summary["max_concentration"]
                )
                if status == "ok" and not within_range:
                    status = "extrapolated"
                lod_flag = None
                if status != "no_model" and math.isfinite(predicted_conc) and math.isfinite(lod) and predicted_conc < lod:
                    lod_flag = "below_lod"
                elif status != "no_model" and math.isfinite(predicted_conc) and math.isfinite(loq) and predicted_conc < loq:
                    lod_flag = "below_loq"
                payload.update(
                    {
                        "status": status if model_available else "no_model",
                        "predicted_concentration": float(predicted_conc),
                        "predicted_concentration_std_err": float(predicted_uncertainty),
                        "within_range": within_range if status not in {"no_model", "invalid"} else False,
                        "lod_flag": lod_flag if status not in {"no_model", "invalid"} else None,
                    }
                )
                self._assign_calibration_payload(
                    sample_lookup.get(sample_id),
                    qc_lookup.get(sample_id),
                    name,
                    payload,
                )
                unknown_results.append(payload | {"sample_id": sample_id})

            target_result["unknowns"] = unknown_results

            if target_result.get("status") == "failed" and not target_result["errors"]:
                target_result["errors"].append("Calibration failed for an unspecified reason.")
            elif target_result.get("status") != "failed":
                target_result["status"] = "ok"
                if target_result["warnings"]:
                    target_result["status"] = "warning"

            target_results.append(target_result)

        overall_status = "ok"
        if any(target.get("status") == "failed" for target in target_results):
            overall_status = "failed"
        elif any(target.get("status") == "warning" for target in target_results):
            overall_status = "warning"

        return {"enabled": True, "status": overall_status, "targets": target_results}

    def analyze(self, specs, recipe):
        from spectro_app.engine.qc import compute_uvvis_drift_map, compute_uvvis_qc

        recipe = self._apply_recipe_defaults(recipe)
        context = self._report_context
        analysis_ctx = context.setdefault("analysis", {})
        ingestion_ctx = context.setdefault("ingestion", {})
        ingestion_ctx.setdefault("spectra_count", len(specs))
        ingestion_ctx.setdefault(
            "blank_count", sum(1 for spec in specs if spec.meta.get("role") == "blank")
        )
        ingestion_ctx.setdefault(
            "sample_count", max(len(specs) - ingestion_ctx.get("blank_count", 0), 0)
        )

        feature_cfg = dict(recipe.get("features", {})) if recipe else {}
        smoothing_cfg = dict(recipe.get("smoothing", {})) if recipe else {}
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
        derivative_cfg_raw = feature_cfg.get("derivatives", True)
        derivative_enabled = True
        derivative_window: int | None = None
        derivative_polyorder: int | None = None
        if smoothing_cfg:
            if smoothing_cfg.get("window") is not None:
                derivative_window = int(smoothing_cfg.get("window"))
            if smoothing_cfg.get("polyorder") is not None:
                derivative_polyorder = int(smoothing_cfg.get("polyorder"))
        if isinstance(derivative_cfg_raw, Mapping):
            derivative_enabled = bool(derivative_cfg_raw.get("enabled", True))
            if derivative_cfg_raw.get("window") is not None:
                derivative_window = int(derivative_cfg_raw.get("window"))
            if derivative_cfg_raw.get("polyorder") is not None:
                derivative_polyorder = int(derivative_cfg_raw.get("polyorder"))
        else:
            derivative_enabled = bool(derivative_cfg_raw)
        isosbestic_cfg = feature_cfg.get("isosbestic") or feature_cfg.get("isosbestic_checks") or []
        kinetics_cfg_raw = feature_cfg.get("kinetics") or {}
        kinetics_targets = self._normalize_kinetics_targets(kinetics_cfg_raw)
        kinetics_time_field: str | None = None
        if isinstance(kinetics_cfg_raw, dict):
            raw_field = kinetics_cfg_raw.get("time_key") or kinetics_cfg_raw.get("timestamp_field")
            if raw_field:
                kinetics_time_field = str(raw_field)

        qc_rows = []
        flag_counts: Dict[str, int] = {}
        flagged_samples = 0
        ratio_aggregates: Dict[str, List[float]] = {}
        integral_aggregates: Dict[str, List[float]] = {}
        drift_map = compute_uvvis_drift_map(specs, recipe)
        processed_with_features: List[Spectrum] = []
        kinetics_records: List[Dict[str, object]] = []
        for idx, spec in enumerate(specs):
            metrics = compute_uvvis_qc(spec, recipe, drift_map.get(id(spec)))
            saturation = metrics["saturation"]
            join = metrics["join"]
            noise = metrics["noise"]
            rolling_noise = metrics["rolling_noise"]
            spikes = metrics["spikes"]
            smoothing = metrics["smoothing"]
            drift = metrics["drift"]
            roughness_metrics = metrics.get("roughness", {})
            processed_roughness_raw = roughness_metrics.get("processed")
            try:
                processed_roughness = float(processed_roughness_raw)
            except (TypeError, ValueError):
                processed_roughness = float("nan")
            channel_roughness_raw = roughness_metrics.get("channels")
            channel_roughness: Dict[str, float] = {}
            if isinstance(channel_roughness_raw, dict):
                for name, value in channel_roughness_raw.items():
                    try:
                        channel_roughness[name] = float(value)
                    except (TypeError, ValueError):
                        channel_roughness[name] = float("nan")
            roughness_delta: Dict[str, float] = {}
            if np.isfinite(processed_roughness):
                for name, value in channel_roughness.items():
                    if np.isfinite(value):
                        roughness_delta[name] = value - processed_roughness
                    else:
                        roughness_delta[name] = float("nan")
            else:
                roughness_delta = {name: float("nan") for name in channel_roughness}
            negative = metrics["negative_intensity"]
            wl = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            derivatives = self._compute_derivatives(
                wl,
                intensity,
                derivative_enabled,
                window=derivative_window,
                polyorder=derivative_polyorder,
            )
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
            def _finite_max(values, *, absolute: bool = False) -> float:
                arr = np.asarray(values, dtype=float)
                if arr.size == 0:
                    return float("nan")
                if absolute:
                    arr = np.abs(arr)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return float("nan")
                return float(np.nanmax(finite))
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
                "rolling_noise_enabled": rolling_noise.enabled,
                "rolling_noise_window_points": rolling_noise.window_points,
                "rolling_noise_step_points": rolling_noise.step_points,
                "rolling_noise_windows": rolling_noise.windows,
                "rolling_noise_max_rsd": rolling_noise.max_rsd,
                "rolling_noise_median_rsd": rolling_noise.median_rsd,
                "negative_intensity_flag": negative.flag,
                "negative_intensity_fraction": negative.processed_fraction,
                "negative_intensity_count": negative.processed_count,
                "negative_intensity_total": negative.processed_total,
                "negative_intensity_channels": negative.channels,
                "negative_intensity_tolerance": negative.tolerance,
                "join_count": join.count,
                "join_max_offset": join.max_offset,
                "join_mean_offset": join.mean_offset,
                "join_max_overlap_error": join.max_overlap_error,
                "join_indices": join.indices,
                "join_offsets": join.offsets,
                "join_pre_deltas": join.pre_deltas,
                "join_post_deltas": join.post_deltas,
                "join_pre_overlap_errors": join.pre_overlap_errors,
                "join_post_overlap_errors": join.post_overlap_errors,
                "join_pre_delta_abs_max": _finite_max(join.pre_deltas, absolute=True),
                "join_post_delta_abs_max": _finite_max(join.post_deltas, absolute=True),
                "join_pre_overlap_max": _finite_max(join.pre_overlap_errors),
                "join_post_overlap_max": _finite_max(join.post_overlap_errors),
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
                "roughness": {"processed": processed_roughness, "channels": channel_roughness},
                "roughness_delta": roughness_delta,
                "flags": metrics["flags"],
                "summary": metrics["summary"],
                "band_ratios": band_ratios,
                "integrals": integrals,
                "peak_metrics": peaks,
                "derivative_stats": derivative_stats,
                "isosbestic": isosbestic_checks,
            }
            if metrics["flags"]:
                flagged_samples += 1
            for flag in metrics["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            for name, payload in band_ratios.items():
                value = payload.get("value")
                if isinstance(value, (int, float)) and math.isfinite(value):
                    ratio_aggregates.setdefault(name, []).append(float(value))
            for name, value in integrals.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    integral_aggregates.setdefault(name, []).append(float(value))
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
        calibration_results = self._perform_calibration(processed_with_features, qc_rows, recipe)
        self._last_calibration_results = calibration_results
        analysis_ctx["qc_total"] = len(qc_rows)
        analysis_ctx["flag_counts"] = dict(sorted(flag_counts.items()))
        analysis_ctx["flagged_samples"] = flagged_samples
        feature_summary: Dict[str, Dict[str, Dict[str, float]]] = {
            "band_ratios": {},
            "integrals": {},
        }
        for name, values in ratio_aggregates.items():
            summary = self._summarise_sequence(values)
            if summary:
                feature_summary["band_ratios"][name] = summary
        for name, values in integral_aggregates.items():
            summary = self._summarise_sequence(values)
            if summary:
                feature_summary["integrals"][name] = summary
        analysis_ctx["feature_summary"] = feature_summary
        analysis_ctx["calibration"] = calibration_results
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
        entries.extend(self._runtime_audit_tokens())
        entries.extend(self._input_source_hash_tokens(specs))
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
        calibration = getattr(self, "_last_calibration_results", None)
        if calibration and calibration.get("targets"):
            for target in calibration.get("targets", []):
                fit = target.get("fit") or {}
                slope = fit.get("slope")
                r_squared = fit.get("r_squared")
                slope_str = f"{slope:.4f}" if isinstance(slope, (int, float)) and math.isfinite(slope) else "nan"
                r2_str = f"{r_squared:.4f}" if isinstance(r_squared, (int, float)) and math.isfinite(r_squared) else "nan"
                conc_sigma = fit.get("concentration_std_per_response")
                conc_sigma_str = (
                    f"{conc_sigma:.4f}"
                    if isinstance(conc_sigma, (int, float)) and math.isfinite(conc_sigma)
                    else "nan"
                )
                weighting_flag = "weighted" if fit.get("weighting_applied") else "unweighted"
                entries.append(
                    f"Calibration {target.get('name', 'target')}: status={target.get('status', 'unknown')} "
                    f"slope={slope_str} r2={r2_str} weighting={weighting_flag} conc_se={conc_sigma_str}"
                )
        else:
            entries.append("Calibration: not performed.")
        return entries

    def _runtime_audit_tokens(self) -> List[str]:
        tokens: List[str] = []
        package_label = "spectro-app"
        version = "unknown"
        for candidate in ("spectro-app", "spectro_app"):
            try:
                version = metadata.version(candidate)
            except metadata.PackageNotFoundError:
                continue
            else:
                package_label = candidate
                break
        tokens.append(f"Runtime library {package_label}=={version}")
        return tokens

    def _input_source_hash_tokens(self, specs: List[Spectrum]) -> List[str]:
        tokens: List[str] = []
        hash_cache: Dict[str, str] = {}
        failure_cache: Dict[str, str] = {}
        for spec in specs:
            source = spec.meta.get("source_file")
            if not source:
                continue
            try:
                source_path_str = os.fspath(source)
            except TypeError:
                source_path_str = str(source)
            sample_id = self._safe_sample_id(spec, source_path_str)

            if source_path_str in hash_cache:
                digest = hash_cache[source_path_str]
                tokens.append(f"Input {sample_id} source_hash=sha256:{digest}")
                continue
            if source_path_str in failure_cache:
                reason = failure_cache[source_path_str]
                tokens.append(f"Input {sample_id} source_hash_error={reason}")
                continue

            path = Path(source_path_str)
            try:
                payload = path.read_bytes()
            except FileNotFoundError:
                reason = "file_not_found"
            except PermissionError:
                reason = "permission_denied"
            except OSError as exc:  # pragma: no cover - fallback for unexpected errors
                reason = f"read_failed:{exc.__class__.__name__}"
            else:
                digest = hashlib.sha256(payload).hexdigest()
                hash_cache[source_path_str] = digest
                tokens.append(f"Input {sample_id} source_hash=sha256:{digest}")
                continue

            failure_cache[source_path_str] = reason
            tokens.append(f"Input {sample_id} source_hash_error={reason}")
        return tokens

    @staticmethod
    def _sanitise_figure_name(*parts: str, ext: str = "png") -> str:
        tokens: List[str] = []
        for part in parts:
            if part is None:
                continue
            text = str(part)
            cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
            cleaned = cleaned.strip("_")
            if cleaned:
                tokens.append(cleaned)
        if not tokens:
            tokens.append("figure")
        base = "_".join(tokens)
        ext_str = str(ext or "").strip()
        if ext_str.startswith("."):
            ext_str = ext_str[1:]
        if not ext_str:
            return base
        return f"{base}.{ext_str}"

    def _render_processed_figure(self, spec: Spectrum):
        wl = np.asarray(spec.wavelength, dtype=float)
        intensity = np.asarray(spec.intensity, dtype=float)
        if wl.size == 0 or intensity.size == 0:
            return None
        channels = spec.meta.get("channels") or {}
        sample_id = self._safe_sample_id(spec, f"spec_{id(spec)}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(wl, intensity, label="Processed", linewidth=1.5)
        stage_order = ["raw", "blanked", "baseline_corrected", "joined", "despiked", "smoothed"]
        ordered_names = [name for name in stage_order if name in channels]
        ordered_names.extend(name for name in channels.keys() if name not in ordered_names)
        for name in ordered_names:
            channel = channels[name]
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
        fig.tight_layout()
        return sample_id, fig

    def _render_join_overlap_figures(
        self,
        spec: Spectrum,
        qc_row: Dict[str, object] | None = None,
    ) -> List[Tuple[Tuple[str, ...], Figure]]:
        join_indices: Tuple[int, ...] = tuple()
        if qc_row and isinstance(qc_row.get("join_indices"), (list, tuple)):
            join_indices = tuple(
                int(idx)
                for idx in qc_row.get("join_indices", [])
                if isinstance(idx, (int, float))
            )
        elif isinstance(spec.meta.get("join_indices"), (list, tuple)):
            join_indices = tuple(int(idx) for idx in spec.meta.get("join_indices", []))

        if not join_indices:
            join_count = qc_row.get("join_count") if qc_row else None
            if not join_count:
                return []

        channels = spec.meta.get("channels") or {}
        raw_channel = channels.get("raw")
        joined_channel = channels.get("joined")
        if raw_channel is None or joined_channel is None:
            return []

        wl = np.asarray(spec.wavelength, dtype=float)
        raw_vals = np.asarray(raw_channel, dtype=float)
        joined_vals = np.asarray(joined_channel, dtype=float)

        if wl.size == 0 or raw_vals.shape != wl.shape or joined_vals.shape != wl.shape:
            return []

        if not join_indices:
            diff = np.abs(raw_vals - joined_vals)
            if not np.any(np.isfinite(diff)):
                return []
            candidate = int(np.nanargmax(diff)) if diff.size else None
            join_indices = (candidate,) if candidate is not None else tuple()
        if not join_indices:
            return []

        figures: List[Tuple[Tuple[str, ...], Figure]] = []
        sample_id = self._safe_sample_id(spec, f"spec_{id(spec)}")
        n = wl.size
        for order, join_idx in enumerate(join_indices, start=1):
            idx = int(join_idx)
            if idx <= 0 or idx >= n:
                continue
            window = max(10, min(80, n // 10))
            start = max(0, idx - window)
            stop = min(n, idx + window)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(wl[start:stop], raw_vals[start:stop], label="Before join", color="tab:gray", linewidth=1.2)
            ax.plot(
                wl[start:stop],
                joined_vals[start:stop],
                label="After correction",
                color="tab:blue",
                linewidth=1.2,
            )
            ax.axvline(wl[min(idx, n - 1)], color="tab:red", linestyle="--", alpha=0.5)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("Intensity")
            ax.set_title(f"Join correction around {wl[min(idx, n - 1)]:.1f} nm")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            figures.append(((sample_id, f"join_{order}"), fig))
        return figures

    def _render_calibration_figures(self) -> List[Tuple[Tuple[str, ...], Figure]]:
        calibration = getattr(self, "_last_calibration_results", None)
        targets = calibration.get("targets", []) if calibration else []
        figures: List[Tuple[Tuple[str, ...], Figure]] = []
        for target in targets:
            standards = target.get("standards") or []
            if not standards:
                continue
            concentrations = np.array(
                [point.get("concentration", float("nan")) for point in standards],
                dtype=float,
            )
            responses = np.array([point.get("response", float("nan")) for point in standards], dtype=float)
            included_mask = np.array([bool(point.get("included", True)) for point in standards], dtype=bool)
            residuals = np.array([point.get("residual", float("nan")) for point in standards], dtype=float)
            if not np.any(np.isfinite(concentrations)) or not np.any(np.isfinite(responses)):
                continue

            fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            inc_x = concentrations[included_mask]
            inc_y = responses[included_mask]
            exc_x = concentrations[~included_mask]
            exc_y = responses[~included_mask]
            if inc_x.size:
                ax_fit.scatter(inc_x, inc_y, label="Included", color="tab:blue")
            if exc_x.size:
                ax_fit.scatter(exc_x, exc_y, label="Excluded", color="tab:orange", marker="x")

            fit = target.get("fit") or {}
            slope = fit.get("slope")
            intercept = fit.get("intercept")
            if (
                isinstance(slope, (int, float))
                and isinstance(intercept, (int, float))
                and math.isfinite(slope)
                and math.isfinite(intercept)
            ):
                x_min = float(np.nanmin(concentrations))
                x_max = float(np.nanmax(concentrations))
                x_span = np.linspace(x_min, x_max, 100)
                ax_fit.plot(x_span, slope * x_span + intercept, color="tab:green", label="Fit")
                r_squared = fit.get("r_squared", float("nan"))
                ax_fit.text(
                    0.02,
                    0.95,
                    f"Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR$^2$: {r_squared:.4f}",
                    transform=ax_fit.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6},
                )
            ax_fit.set_ylabel("Response (A/cm)")
            ax_fit.set_title(f"Calibration fit: {target.get('name', 'target')}")
            ax_fit.grid(True, alpha=0.2)
            ax_fit.legend(loc="best")

            if np.any(np.isfinite(residuals)):
                ax_resid.axhline(0.0, color="tab:gray", linestyle="--", linewidth=1.0)
                ax_resid.scatter(concentrations[included_mask], residuals[included_mask], color="tab:blue")
                if exc_x.size:
                    ax_resid.scatter(concentrations[~included_mask], residuals[~included_mask], color="tab:orange", marker="x")
            ax_resid.set_xlabel("Concentration")
            ax_resid.set_ylabel("Residual (A/cm)")
            ax_resid.grid(True, alpha=0.2)

            fig.tight_layout()
            figures.append((("calibration", str(target.get("name", "target"))), fig))
        return figures

    def _render_qc_summary_figures(
        self, qc_rows: Sequence[Dict[str, object]] | None
    ) -> List[Tuple[Tuple[str, ...], Figure]]:
        if not qc_rows:
            return []

        figures: List[Tuple[Tuple[str, ...], Figure]] = []
        noise_values = np.array(
            [row.get("noise_rsd", float("nan")) for row in qc_rows],
            dtype=float,
        )
        finite_mask = np.isfinite(noise_values)
        if not np.any(finite_mask):
            return []

        sample_labels = [
            str(row.get("sample_id") or row.get("id") or idx)
            for idx, row in enumerate(qc_rows)
        ]
        join_counts = np.array([row.get("join_count", 0) or 0 for row in qc_rows], dtype=float)

        scatter_fig = self._plot_noise_scatter(sample_labels, noise_values, join_counts, finite_mask)
        figures.append((("qc", "summary", "noise"), scatter_fig))

        hist_fig = self._plot_noise_histogram(noise_values[finite_mask])
        if hist_fig is not None:
            figures.append((("qc", "summary", "noise", "hist"), hist_fig))

        timestamp_fig = self._plot_noise_trend(sample_labels, noise_values, qc_rows, finite_mask)
        if timestamp_fig is not None:
            figures.append((("qc", "summary", "noise", "trend"), timestamp_fig))

        return figures

    @staticmethod
    def _plot_noise_scatter(
        sample_labels: Sequence[str],
        noise_values: np.ndarray,
        join_counts: np.ndarray,
        finite_mask: np.ndarray,
    ) -> Figure:
        x = np.arange(len(sample_labels), dtype=float)
        width = max(6.0, len(sample_labels) * 0.6)
        fig, ax = plt.subplots(figsize=(width, 4.0))
        scatter = ax.scatter(
            x[finite_mask],
            noise_values[finite_mask],
            c=join_counts[finite_mask],
            cmap="viridis",
            s=60,
            edgecolors="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sample_labels, rotation=45, ha="right")
        ax.set_ylabel("Noise RSD (%)")
        ax.set_xlabel("Sample")
        ax.set_title("QC summary: noise by sample")
        ax.grid(True, alpha=0.2)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Join count")
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_noise_histogram(noise_values: np.ndarray) -> Figure | None:
        if noise_values.size < 3:
            return None
        finite_noise = noise_values[np.isfinite(noise_values)]
        if finite_noise.size < 3:
            return None
        bins = min(30, max(5, int(np.ceil(np.sqrt(finite_noise.size)))))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.hist(
            finite_noise,
            bins=bins,
            color="tab:blue",
            alpha=0.75,
            edgecolor="black",
        )
        mean_val = float(np.nanmean(finite_noise))
        median_val = float(np.nanmedian(finite_noise))
        ax.axvline(mean_val, color="tab:orange", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
        ax.axvline(median_val, color="tab:green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.3f}")
        ax.set_xlabel("Noise RSD (%)")
        ax.set_ylabel("Count")
        ax.set_title("QC summary: noise distribution")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_noise_trend(
        self,
        sample_labels: Sequence[str],
        noise_values: np.ndarray,
        qc_rows: Sequence[Dict[str, object]],
        finite_mask: np.ndarray,
    ) -> Figure | None:
        timestamp_strings: List[Optional[str]] = []
        for row in qc_rows:
            ts: Optional[str] = None
            kinetics = row.get("kinetics") if isinstance(row, dict) else None
            if isinstance(kinetics, dict):
                candidate = kinetics.get("timestamp")
                if isinstance(candidate, str):
                    ts = candidate
            if not ts:
                for key in ("timestamp", "acquired_datetime", "drift_timestamp"):
                    candidate = row.get(key) if isinstance(row, dict) else None
                    if isinstance(candidate, str):
                        ts = candidate
                        break
            timestamp_strings.append(ts)

        if not any(timestamp_strings):
            return None

        timestamps = pd.to_datetime(timestamp_strings, errors="coerce", utc=True)
        try:
            timestamps = timestamps.tz_convert(None)  # type: ignore[call-arg]
        except AttributeError:
            pass

        timestamp_mask = ~pd.isna(timestamps)
        if not np.any(timestamp_mask):
            return None

        combined_mask = np.asarray(finite_mask) & np.asarray(timestamp_mask)
        if np.count_nonzero(combined_mask) < 2:
            return None

        valid_times = timestamps[combined_mask]
        valid_noise = noise_values[combined_mask]
        valid_labels = np.asarray(sample_labels, dtype=object)[combined_mask]

        order = np.argsort(valid_times.astype("datetime64[ns]"))
        ordered_times = valid_times[order]
        ordered_noise = valid_noise[order]
        ordered_labels = valid_labels[order]

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ordered_python_times = ordered_times.to_pydatetime()
        ax.plot(ordered_python_times, ordered_noise, linestyle="-", marker="o", color="tab:blue")
        for time_val, noise_val, label in zip(ordered_python_times, ordered_noise, ordered_labels):
            ax.annotate(
                str(label),
                xy=(time_val, noise_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_ylabel("Noise RSD (%)")
        ax.set_xlabel("Timestamp")
        ax.set_title("QC summary: noise trend")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def _generate_figures(
        self,
        specs: List[Spectrum],
        qc_rows: Sequence[Dict[str, object]] | None = None,
        formats: object = None,
    ) -> Tuple[Dict[str, bytes], List[Tuple[str, Figure]]]:
        figure_formats = self._normalise_figure_formats(formats)
        figures: Dict[str, bytes] = {}
        figure_objs: List[Tuple[str, Figure]] = []
        qc_lookup: Dict[int, Dict[str, object]] = {}
        if qc_rows:
            for spec, qc_row in zip(specs, qc_rows):
                if isinstance(qc_row, dict):
                    qc_lookup[id(spec)] = qc_row

        def capture(parts: Sequence[str], fig: Figure) -> None:
            base_name = self._sanitise_figure_name(*parts, ext=figure_formats[0])
            for fmt in figure_formats:
                filename = self._sanitise_figure_name(*parts, ext=fmt)
                buf = io.BytesIO()
                save_kwargs = {"format": fmt}
                if fmt == "png":
                    save_kwargs["dpi"] = 150
                fig.savefig(buf, **save_kwargs)
                buf.seek(0)
                figures[filename] = buf.read()
            figure_objs.append((base_name, fig))

        for spec in specs:
            rendered = self._render_processed_figure(spec)
            if not rendered:
                continue
            sample_id, fig = rendered
            capture((sample_id, "processed"), fig)

            qc_row = qc_lookup.get(id(spec))
            for name_parts, join_fig in self._render_join_overlap_figures(spec, qc_row):
                capture(name_parts, join_fig)

        for name_parts, cal_fig in self._render_calibration_figures():
            capture(name_parts, cal_fig)

        for name_parts, qc_fig in self._render_qc_summary_figures(qc_rows):
            capture(name_parts, qc_fig)

        return figures, figure_objs

    @staticmethod
    def _normalise_figure_formats(formats: object = None) -> Tuple[str, ...]:
        default = ("png",)
        if formats is None:
            return default
        candidates: List[str]
        if isinstance(formats, Mapping):
            candidates = [str(key) for key, value in formats.items() if value]
        elif isinstance(formats, (str, bytes)):
            candidates = [formats]
        else:
            try:
                candidates = list(formats)  # type: ignore[arg-type]
            except TypeError:
                candidates = [str(formats)]
        normalised: List[str] = []
        for raw in candidates:
            if raw is None:
                continue
            token = str(raw).strip().lower()
            if not token:
                continue
            if token.startswith("."):
                token = token[1:]
            if token not in {"png", "svg"}:
                continue
            if token not in normalised:
                normalised.append(token)
        return tuple(normalised) if normalised else default

    @staticmethod
    def _coerce_export_path(value, default: Optional[Path] = None) -> Optional[Path]:
        if value in (None, False):
            return None
        if isinstance(value, dict):
            value = value.get("path") or value.get("file")
        if isinstance(value, (str, Path)):
            return Path(value)
        if value is True and default is not None:
            return default
        return None

    @staticmethod
    def _json_sanitise(value):
        if isinstance(value, dict):
            return {str(k): UvVisPlugin._json_sanitise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [UvVisPlugin._json_sanitise(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()  # type: ignore[return-value]
            except Exception:  # pragma: no cover - best effort serialisation
                return str(value)
        return value

    def _write_recipe_sidecar(self, recipe_path: Path, recipe: Dict[str, object]) -> None:
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = self._json_sanitise(recipe or {})
        with recipe_path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2, sort_keys=True)

    def _build_text_report(self, audit_entries: Sequence[str]) -> Optional[str]:
        context = self._report_context
        ingestion = context.get("ingestion", {})
        preprocessing = context.get("preprocessing", {})
        analysis = context.get("analysis", {})
        results = context.get("results", {})

        def _fmt_float(value: object, precision: int = 2) -> Optional[str]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            return f"{number:.{precision}f}"

        ingestion_lines: List[str] = []
        spectra_count = ingestion.get("spectra_count") or analysis.get("qc_total") or 0
        data_files = ingestion.get("data_files") or []
        if data_files:
            file_names = ", ".join(sorted(Path(path).name for path in data_files))
            ingestion_lines.append(
                f"Processed {spectra_count} spectrum(s) from {len(data_files)} data file(s): {file_names}."
            )
        else:
            ingestion_lines.append(f"Processed {spectra_count} spectrum(s).")
        blank_count = ingestion.get("blank_count")
        sample_count = ingestion.get("sample_count")
        if blank_count is not None and sample_count is not None:
            ingestion_lines.append(
                f"Identified {sample_count} sample spectrum(s) and {blank_count} blank(s)."
            )
        manifest_enabled = bool(ingestion.get("manifest_enabled"))
        if manifest_enabled:
            manifest_files = ingestion.get("manifest_files") or []
            manifest_entry_count = ingestion.get("manifest_entry_count", 0) or 0
            manifest_hits = (ingestion.get("manifest_index_hits", 0) or 0) + (
                ingestion.get("manifest_lookup_hits", 0) or 0
            )
            if manifest_files or manifest_entry_count:
                if manifest_files:
                    manifest_names = ", ".join(
                        sorted(Path(path).name for path in manifest_files)
                    )
                else:
                    manifest_names = "None"
                if manifest_entry_count:
                    ingestion_lines.append(
                        f"Manifest files: {manifest_names} (matched {manifest_hits} of {manifest_entry_count} entries)."
                    )
                else:
                    ingestion_lines.append(f"Manifest files: {manifest_names}.")

        preprocessing_lines: List[str] = []
        domain_cfg = preprocessing.get("domain") or {}
        domain_min = _fmt_float(domain_cfg.get("min"), 1)
        domain_max = _fmt_float(domain_cfg.get("max"), 1)
        if domain_min is not None and domain_max is not None:
            preprocessing_lines.append(f"Domain restricted to {domain_min}–{domain_max} nm.")
        else:
            preprocessing_lines.append("Domain spans the full available wavelength range.")
        blank_cfg = preprocessing.get("blank") or {}
        subtract_enabled = bool(blank_cfg.get("subtract", True))
        require_blank = bool(blank_cfg.get("require", subtract_enabled))
        blank_line = "Blank subtraction " + ("enabled" if subtract_enabled else "disabled")
        blank_line += " (blank required)" if require_blank else " (blank optional)"
        default_blank = blank_cfg.get("default_blank")
        if default_blank:
            blank_line += f"; default blank {default_blank}"
        match_strategy_summary = str(blank_cfg.get("match_strategy") or "blank_id")
        match_strategy_summary = match_strategy_summary.replace("-", "_")
        if match_strategy_summary == "cuvette_slot":
            blank_line += "; match via cuvette slot"
        else:
            blank_line += "; match via blank identifier"
        blank_line += "."
        preprocessing_lines.append(blank_line)
        if not blank_cfg.get("validate_metadata", True):
            preprocessing_lines.append("Blank pairing validation ran in advisory mode.")
        baseline_cfg = preprocessing.get("baseline") or {}
        baseline_method = baseline_cfg.get("method")
        if baseline_method:
            params = baseline_cfg.get("parameters") or {}
            pieces: List[str] = []
            lam_val = _fmt_float(params.get("lambda"))
            if lam_val is not None:
                pieces.append(f"λ={lam_val}")
            p_val = _fmt_float(params.get("p"))
            if p_val is not None:
                pieces.append(f"p={p_val}")
            iter_val = params.get("iterations")
            if iter_val is not None:
                pieces.append(f"iterations={iter_val}")
            summary = ", ".join(pieces)
            if summary:
                preprocessing_lines.append(
                    f"Baseline correction applied using {baseline_method} ({summary})."
                )
            else:
                preprocessing_lines.append(
                    f"Baseline correction applied using {baseline_method}."
                )
        else:
            preprocessing_lines.append("Baseline correction was not applied.")
        join_cfg = preprocessing.get("join") or {}
        if join_cfg.get("enabled"):
            join_window = join_cfg.get("window")
            join_threshold = _fmt_float(join_cfg.get("threshold"))
            detail_parts: List[str] = []
            if join_window:
                detail_parts.append(f"window={join_window}")
            if join_threshold is not None:
                detail_parts.append(f"threshold={join_threshold}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            preprocessing_lines.append(f"Join correction enabled{detail}.")
        else:
            preprocessing_lines.append("Join correction disabled.")
        despike_cfg = preprocessing.get("despike") or {}
        if despike_cfg.get("enabled"):
            detail_parts = []
            if despike_cfg.get("window") is not None:
                detail_parts.append(f"window={despike_cfg.get('window')}")
            zscore_val = _fmt_float(despike_cfg.get("zscore"))
            if zscore_val is not None:
                detail_parts.append(f"z-score={zscore_val}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            preprocessing_lines.append(f"Despiking enabled{detail}.")
        smoothing_cfg = preprocessing.get("smoothing") or {}
        if smoothing_cfg.get("enabled"):
            detail_parts = []
            if smoothing_cfg.get("window"):
                detail_parts.append(f"window={smoothing_cfg.get('window')}")
            if smoothing_cfg.get("polyorder") is not None:
                detail_parts.append(f"polyorder={smoothing_cfg.get('polyorder')}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            preprocessing_lines.append(f"Savitzky-Golay smoothing enabled{detail}.")
        else:
            preprocessing_lines.append("Smoothing disabled.")
        replicates_cfg = preprocessing.get("replicates") or {}
        if replicates_cfg.get("average", True):
            replicate_line = "Replicate spectra averaged"
        else:
            replicate_line = "Replicate spectra kept separate"
        if replicates_cfg.get("outlier"):
            replicate_line += f" with outlier policy {replicates_cfg.get('outlier')}"
        replicate_line += "."
        preprocessing_lines.append(replicate_line)
        quiet_window = preprocessing.get("quiet_window") or {}
        quiet_min = _fmt_float(quiet_window.get("min"), 1)
        quiet_max = _fmt_float(quiet_window.get("max"), 1)
        if quiet_min is not None and quiet_max is not None:
            preprocessing_lines.append(
                f"Noise assessed in quiet window {quiet_min}–{quiet_max} nm."
            )
        counts = preprocessing.get("counts") or {}
        if counts:
            preprocessing_lines.append(
                "Pre-processing yielded "
                f"{counts.get('processed_samples', 0)} sample(s) and {counts.get('processed_blanks', 0)} blank(s)."
            )
        blank_ids = preprocessing.get("blank_ids") or []
        if blank_ids:
            preprocessing_lines.append("Blank identifiers observed: " + ", ".join(blank_ids) + ".")

        analysis_lines: List[str] = []
        qc_total = analysis.get("qc_total", 0)
        analysis_lines.append(f"QC metrics computed for {qc_total} spectrum(s).")
        flagged_samples = analysis.get("flagged_samples", 0)
        if flagged_samples:
            analysis_lines.append(f"{flagged_samples} spectrum(s) triggered one or more QC flags.")
        flag_counts = analysis.get("flag_counts") or {}
        if flag_counts:
            ordered = ", ".join(
                f"{flag} ({count})" for flag, count in sorted(flag_counts.items())
            )
            analysis_lines.append(f"QC flags observed: {ordered}.")
        else:
            analysis_lines.append("No QC flags were raised.")
        feature_summary = analysis.get("feature_summary") or {}
        ratio_summary = feature_summary.get("band_ratios") or {}
        ratio_highlights: List[str] = []
        for name, stats in sorted(ratio_summary.items())[:3]:
            mean_val = _fmt_float(stats.get("mean"))
            if mean_val is not None:
                ratio_highlights.append(f"{name} mean {mean_val}")
        if ratio_highlights:
            analysis_lines.append("Key band ratios: " + ", ".join(ratio_highlights) + ".")
        integral_summary = feature_summary.get("integrals") or {}
        integral_highlights: List[str] = []
        for name, stats in sorted(integral_summary.items())[:3]:
            mean_val = _fmt_float(stats.get("mean"))
            if mean_val is not None:
                integral_highlights.append(f"{name} mean {mean_val}")
        if integral_highlights:
            analysis_lines.append("Representative integrals: " + ", ".join(integral_highlights) + ".")
        calibration = analysis.get("calibration") or {}
        if calibration:
            if not calibration.get("enabled", True):
                analysis_lines.append(
                    f"Calibration skipped ({calibration.get('status', 'disabled')})."
                )
            else:
                targets = calibration.get("targets") or []
                if targets:
                    status_counts: Dict[str, int] = {}
                    for target in targets:
                        status = str(target.get("status") or "unknown")
                        status_counts[status] = status_counts.get(status, 0) + 1
                    ordered_status = ", ".join(
                        f"{status} ({count})" for status, count in sorted(status_counts.items())
                    )
                    analysis_lines.append(
                        f"Calibration targets evaluated: {ordered_status}."
                    )
                else:
                    analysis_lines.append("Calibration configured without defined targets.")

        results_lines: List[str] = []
        figure_count = results.get("figure_count", 0)
        results_lines.append(
            f"Generated {figure_count} figure(s) and {len(audit_entries)} audit message(s)."
        )
        outputs = results.get("export_targets") or {}
        workbook_path = outputs.get("workbook")
        if workbook_path:
            results_lines.append(f"Workbook location: {workbook_path}.")
        pdf_path = outputs.get("pdf")
        if pdf_path:
            results_lines.append(f"PDF report: {pdf_path}.")
        recipe_path = outputs.get("recipe")
        if recipe_path:
            results_lines.append(f"Recipe sidecar: {recipe_path}.")
        if audit_entries:
            highlights = "; ".join(audit_entries[:3])
            results_lines.append(f"Audit highlights: {highlights}.")

        sections = [
            ("Ingestion", ingestion_lines),
            ("Pre-processing", preprocessing_lines),
            ("Analysis", analysis_lines),
            ("Results", results_lines),
        ]

        lines: List[str] = []
        for title, body in sections:
            lines.append(title)
            lines.append("-" * len(title))
            if body:
                lines.extend(body)
            else:
                lines.append("No details available.")
            lines.append("")

        text = "\n".join(lines).strip()
        return text or None

    def _write_pdf_report(
        self,
        pdf_path: Path,
        figures: List[Tuple[str, Figure]],
        audit_entries: List[str],
        specs: List[Spectrum],
        qc: List[Dict[str, object]],
        report_text: Optional[str] = None,
    ) -> None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            summary_fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            title = "UV-Vis Export Report"
            ax.text(0.5, 0.95, title, ha="center", va="center", fontsize=16, fontweight="bold")
            timestamp = datetime.now(UTC).isoformat()
            ax.text(0.02, 0.9, f"Generated: {timestamp}", fontsize=10, va="top")
            ax.text(0.02, 0.85, f"Spectra processed: {len(specs)}", fontsize=10, va="top")
            ax.text(0.02, 0.82, f"QC rows: {len(qc)}", fontsize=10, va="top")
            narrative_lines: List[str] = []
            if report_text:
                narrative_lines.append("Summary")
                narrative_lines.append("~~~~~~~")
                narrative_lines.append(report_text)
                narrative_lines.append("")
            narrative_lines.append("Audit log")
            narrative_lines.append("~~~~~~~~~")
            if audit_entries:
                narrative_lines.extend(audit_entries)
            else:
                narrative_lines.append("No audit entries available.")
            ax.text(0.02, 0.75, "\n".join(narrative_lines), fontsize=9, va="top", wrap=True)
            pdf.savefig(summary_fig)
            plt.close(summary_fig)
            for _, figure in figures:
                pdf.savefig(figure)

    def export(self, specs, qc, recipe):
        export_cfg = dict(recipe.get("export", {})) if recipe else {}
        workbook_target = self._coerce_export_path(export_cfg.get("path") or export_cfg.get("workbook"))
        processed_layout = export_cfg.get("processed_layout", "tidy")
        if isinstance(processed_layout, str):
            processed_layout = processed_layout.lower()
        else:
            processed_layout = "tidy"
        if processed_layout not in {"tidy", "wide"}:
            processed_layout = "tidy"
        figures, figure_objs = self._generate_figures(specs, qc, formats=export_cfg.get("formats"))
        audit_entries = self._build_audit_entries(specs, qc, recipe, figures)
        workbook_audit = list(audit_entries)
        calibration_results = getattr(self, "_last_calibration_results", None)
        workbook_default = workbook_target if workbook_target else None
        recipe_target = self._coerce_export_path(
            export_cfg.get("recipe_path") or export_cfg.get("recipe_sidecar") or export_cfg.get("recipe"),
            workbook_default.with_suffix(".recipe.json") if workbook_default else None,
        )
        pdf_target = self._coerce_export_path(
            export_cfg.get("pdf_path")
            or export_cfg.get("pdf_report")
            or export_cfg.get("pdf")
            or export_cfg.get("report"),
            workbook_default.with_suffix(".pdf") if workbook_default else None,
        )
        results_ctx = self._report_context.setdefault("results", {})
        results_ctx["figure_count"] = len(figures)
        results_ctx["export_targets"] = {
            "workbook": str(workbook_target) if workbook_target else None,
            "pdf": str(pdf_target) if pdf_target else None,
            "recipe": str(recipe_target) if recipe_target else None,
        }

        try:
            if workbook_target:
                resolved_path = str(workbook_target)
                workbook_audit.append(f"Workbook written to {resolved_path}")
                write_workbook(
                    resolved_path,
                    specs,
                    qc,
                    workbook_audit,
                    figures,
                    calibration_results,
                    processed_layout=processed_layout,
                )
            else:
                workbook_audit.append("No workbook path provided; workbook not written.")

            if recipe_target and recipe:
                self._write_recipe_sidecar(recipe_target, recipe)
                workbook_audit.append(f"Recipe sidecar written to {recipe_target}")
            elif export_cfg.get("recipe") or export_cfg.get("recipe_path") or export_cfg.get("recipe_sidecar"):
                workbook_audit.append("Recipe sidecar requested but no path resolved.")

            if pdf_target:
                audit_for_pdf = list(workbook_audit)
                report_for_pdf = self._build_text_report(audit_for_pdf)
                self._write_pdf_report(
                    pdf_target,
                    figure_objs,
                    audit_for_pdf,
                    specs,
                    qc,
                    report_for_pdf,
                )
                workbook_audit.append(f"PDF report written to {pdf_target}")
        finally:
            for _, fig in figure_objs:
                plt.close(fig)

        results_ctx["audit_messages"] = list(workbook_audit)
        report_text = self._build_text_report(workbook_audit)
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures=figures,
            audit=workbook_audit,
            report_text=report_text,
        )

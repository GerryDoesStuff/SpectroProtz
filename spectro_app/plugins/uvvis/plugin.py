from __future__ import annotations

import io
from dataclasses import dataclass
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

class UvVisPlugin(SpectroscopyPlugin):
    id = "uvvis"
    label = "UV-Vis"
    xlabel = "Wavelength (nm)"

    def detect(self, paths):
        return any(str(p).lower().endswith((".dsp", ".csv", ".xlsx", ".txt")) for p in paths)

    def load(self, paths: Iterable[str]) -> List[Spectrum]:
        spectra: List[Spectrum] = []
        path_objects = [Path(p) for p in paths]
        manifest_index, manifest_files = self._parse_manifest(path_objects)

        for path in path_objects:
            if path.resolve() in manifest_files:
                continue
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
                    updates = self._lookup_manifest(manifest_index, path, meta)
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
                spectra.append(Spectrum(wavelength=wl, intensity=inten, meta=meta))

        return spectra

    # ------------------------------------------------------------------
    # Manifest ingestion helpers

    @dataclass
    class _ManifestIndex:
        by_file_channel: Dict[Tuple[Optional[str], str], Dict[str, object]]
        by_file_sample: Dict[Tuple[Optional[str], str], Dict[str, object]]

        def __bool__(self) -> bool:  # pragma: no cover - trivial
            return bool(self.by_file_channel or self.by_file_sample)

    def _parse_manifest(
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

    def _lookup_manifest(
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

        stage_one: list[Spectrum] = []
        for spec in specs:
            coerced = pipeline.coerce_domain(spec, domain_cfg)
            processed = coerced
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
                )
            stage_one.append(processed)

        blanks_stage = [spec for spec in stage_one if spec.meta.get("role") == "blank"]
        samples_stage = [spec for spec in stage_one if spec.meta.get("role") != "blank"]

        if average_replicates and blanks_stage:
            blanks_avg, blank_map_by_key = pipeline.average_replicates(blanks_stage, return_mapping=True)
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
        fallback_blank = blank_cfg.get("default") or blank_cfg.get("fallback")

        processed_samples: list[Spectrum] = []
        for spec in samples_stage:
            working = spec
            if subtract_blank:
                blank_id = spec.meta.get("blank_id") or fallback_blank
                if blank_id:
                    blank_spec = blank_lookup.get(str(blank_id))
                    if blank_spec is None:
                        raise ValueError(f"No blank spectrum available for blank_id '{blank_id}'")
                    working = pipeline.subtract_blank(working, blank_spec)
                elif require_blank:
                    sample_id = spec.meta.get("sample_id") or spec.meta.get("channel")
                    raise ValueError(f"Sample '{sample_id}' does not have a blank for subtraction")

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
            processed_samples, sample_map = pipeline.average_replicates(processed_samples, return_mapping=True)
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

    def analyze(self, specs, recipe):
        from spectro_app.engine.qc import compute_uvvis_qc

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

        qc_rows = []
        processed_with_features: List[Spectrum] = []
        for idx, spec in enumerate(specs):
            metrics = compute_uvvis_qc(spec, recipe)
            saturation = metrics["saturation"]
            join = metrics["join"]
            noise = metrics["noise"]
            spikes = metrics["spikes"]
            smoothing = metrics["smoothing"]
            wl = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            derivatives = self._compute_derivatives(wl, intensity, derivative_enabled)
            band_ratios = self._compute_band_ratios(wl, intensity, ratio_cfg)
            integrals = self._compute_integrals(wl, intensity, integral_cfg)
            peaks = self._compute_peak_metrics(wl, intensity, peak_cfg)
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
                "flags": metrics["flags"],
                "summary": metrics["summary"],
                "band_ratios": band_ratios,
                "integrals": integrals,
                "peak_metrics": peaks,
                "derivative_stats": derivative_stats,
            }
            qc_rows.append(row)
            meta = dict(spec.meta)
            meta.setdefault("features", {})
            meta["features"].update({
                "band_ratios": band_ratios,
                "integrals": integrals,
                "peaks": peaks,
                "derivative_stats": derivative_stats,
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
            processed_with_features.append(
                Spectrum(
                    wavelength=spec.wavelength.copy(),
                    intensity=spec.intensity.copy(),
                    meta=meta,
                )
            )
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

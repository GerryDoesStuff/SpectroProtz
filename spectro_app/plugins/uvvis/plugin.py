from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from spectro_app.engine.io_common import sniff_locale
from spectro_app.engine.plugin_api import BatchResult, SpectroscopyPlugin, Spectrum
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
        for path_str in paths:
            path = Path(path_str)
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
                spectra.append(Spectrum(wavelength=wl, intensity=inten, meta=meta))

        return spectra

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

    def analyze(self, specs, recipe):
        qc = [{"id": i, "flags": []} for i,_ in enumerate(specs)]
        return specs, qc

    def export(self, specs, qc, recipe):
        return BatchResult(processed=specs, qc_table=qc, figures={}, audit=["UV-Vis export stub"])

from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
from spectro_app.engine.peak_detection import detect_peaks_for_features, resolve_peak_config
import numpy as np

class FtirPlugin(SpectroscopyPlugin):
    id = "ftir"
    label = "FTIR"
    xlabel = "Wavenumber (cm⁻¹)"

    def detect(self, paths):
        return any(str(p).lower().endswith((".csv", ".txt")) for p in paths)

    def load(self, paths):
        wn = np.linspace(4000, 400, 1801)
        inten = np.zeros_like(wn)
        return [
            Spectrum(
                wavelength=wn,
                intensity=inten,
                meta={"source": "stub", "axis_key": "wavenumber", "axis_unit": "cm^-1"},
            )
        ]

    def preprocess(self, specs, recipe):
        # TODO: baseline, rubberband, atmospheric compensation
        return specs

    def _detect_peak_features(self, wn: np.ndarray, intensity: np.ndarray, peak_cfg: dict) -> list[dict]:
        config = resolve_peak_config(peak_cfg)
        return detect_peaks_for_features(wn, intensity, config, axis_key="wavenumber")

    def analyze(self, specs, recipe):
        feature_cfg = dict(recipe.get("features", {})) if recipe else {}
        peak_cfg = dict(feature_cfg.get("peaks", {}))
        processed = []
        qc_rows = []
        for spec in specs:
            wn = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            peaks = self._detect_peak_features(wn, intensity, peak_cfg)
            meta = dict(spec.meta)
            meta.setdefault("features", {})
            meta["features"]["peaks"] = peaks
            processed.append(
                Spectrum(
                    wavelength=spec.wavelength.copy(),
                    intensity=spec.intensity.copy(),
                    meta=meta,
                )
            )
            qc_rows.append({"peaks": len(peaks)})
        return processed, qc_rows

    def export(self, specs, qc, recipe):
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures={},
            audit=["FTIR export stub"],
            report_text=None,
        )

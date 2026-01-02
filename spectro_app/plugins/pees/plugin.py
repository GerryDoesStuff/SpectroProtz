from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
from spectro_app.engine.peak_detection import detect_peaks_for_features, resolve_peak_config
import numpy as np

class PeesPlugin(SpectroscopyPlugin):
    id = "pees"
    label = "PEES"
    xlabel = "Axis"

    def detect(self, paths):
        return any(str(p).lower().endswith((".csv", ".txt")) for p in paths)

    def load(self, paths):
        x = np.linspace(0, 1, 1001)
        y = np.zeros_like(x)
        return [Spectrum(wavelength=x, intensity=y, meta={"source": "stub"})]

    def preprocess(self, specs, recipe):
        # TODO: method-specific steps
        return specs

    def _detect_peak_features(self, x_axis: np.ndarray, intensity: np.ndarray, peak_cfg: dict) -> list[dict]:
        config = resolve_peak_config(peak_cfg)
        return detect_peaks_for_features(x_axis, intensity, config, axis_key="wavelength")

    def analyze(self, specs, recipe):
        feature_cfg = dict(recipe.get("features", {})) if recipe else {}
        peak_cfg = dict(feature_cfg.get("peaks", {}))
        processed = []
        qc_rows = []
        for spec in specs:
            x_axis = np.asarray(spec.wavelength, dtype=float)
            intensity = np.asarray(spec.intensity, dtype=float)
            peaks = self._detect_peak_features(x_axis, intensity, peak_cfg)
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
            audit=["PEES export stub"],
            report_text=None,
        )

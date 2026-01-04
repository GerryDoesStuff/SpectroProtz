import numpy as np

from spectro_app.engine import pipeline as core_pipeline
from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
from spectro_app.io.opus import is_opus_path, load_opus_spectra

class RamanPlugin(SpectroscopyPlugin):
    id = "raman"
    label = "Raman"
    xlabel = "Raman shift (cm⁻¹)"

    def detect(self, paths):
        return any(
            str(p).lower().endswith((".csv", ".txt")) or is_opus_path(p)
            for p in paths
        )

    def load(self, paths, cancelled=None):
        spectra = []
        for path in paths:
            if cancelled is not None and cancelled():
                raise RuntimeError("Cancelled")
            if is_opus_path(path):
                spectra.extend(load_opus_spectra(path, technique="raman"))
            if cancelled is not None and cancelled():
                raise RuntimeError("Cancelled")
        if spectra:
            return spectra
        wn = np.linspace(100, 3500, 3401)
        inten = np.zeros_like(wn)
        return [
            Spectrum(
                wavelength=wn,
                intensity=inten,
                meta={"source": "stub", "axis_key": "wavenumber", "axis_unit": "cm^-1"},
            )
        ]

    def analyze(self, specs, recipe):
        return core_pipeline.run_pipeline(specs, recipe)

    def export(self, specs, qc, recipe):
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures={},
            audit=["Raman export stub"],
            report_text=None,
        )

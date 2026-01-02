import numpy as np

from spectro_app.engine import pipeline as core_pipeline
from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
from spectro_app.io.opus import load_opus_spectra

class FtirPlugin(SpectroscopyPlugin):
    id = "ftir"
    label = "FTIR"
    xlabel = "Wavenumber (cm⁻¹)"

    def detect(self, paths):
        return any(str(p).lower().endswith((".csv", ".txt", ".opus")) for p in paths)

    def load(self, paths):
        spectra = []
        for path in paths:
            if str(path).lower().endswith(".opus"):
                spectra.extend(load_opus_spectra(path, technique="ftir"))
        if spectra:
            return spectra
        wn = np.linspace(4000, 400, 1801)
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
            audit=["FTIR export stub"],
            report_text=None,
        )

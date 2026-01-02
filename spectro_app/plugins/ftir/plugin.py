from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
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
        return [Spectrum(wavelength=wn, intensity=inten, meta={"source": "stub"})]

    def export(self, specs, qc, recipe):
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures={},
            audit=["FTIR export stub"],
            report_text=None,
        )

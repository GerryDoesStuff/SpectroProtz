from spectro_app.engine.plugin_api import SpectroscopyPlugin, Spectrum, BatchResult
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

    def export(self, specs, qc, recipe):
        return BatchResult(
            processed=specs,
            qc_table=qc,
            figures={},
            audit=["PEES export stub"],
            report_text=None,
        )

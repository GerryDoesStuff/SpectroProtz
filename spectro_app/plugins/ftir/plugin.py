from spectro_app.engine import pipeline as core_pipeline
from spectro_app.engine.plugin_api import SpectroscopyPlugin, BatchResult
from spectro_app.io.opus import is_opus_path, load_opus_spectra

class FtirPlugin(SpectroscopyPlugin):
    id = "ftir"
    label = "FTIR"
    xlabel = "Wavenumber (cm⁻¹)"

    def detect(self, paths):
        return any(
            str(p).lower().endswith((".csv", ".txt")) or is_opus_path(p)
            for p in paths
        )

    def load(self, paths):
        spectra = []
        for path in paths:
            if is_opus_path(path):
                # Allow load_opus_spectra errors to propagate so callers can
                # surface the real parsing failure.
                spectra.extend(load_opus_spectra(path, technique="ftir"))
        if not spectra:
            raise ValueError("No OPUS spectra were loaded from the provided paths.")
        return spectra

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

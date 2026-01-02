import numpy as np

from spectro_app.engine import pipeline as core_pipeline
from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.raman.plugin import RamanPlugin


def test_core_pipeline_detects_peaks_for_non_uvvis_plugin():
    plugin = RamanPlugin()
    x_axis = np.linspace(100.0, 500.0, 801)
    intensity = np.exp(-0.5 * ((x_axis - 300.0) / 8.0) ** 2)
    spectrum = Spectrum(wavelength=x_axis, intensity=intensity, meta={"role": "sample"})

    processed, qc_rows = core_pipeline.run_processing(
        plugin,
        [spectrum],
        {"features": {"peaks": {"enabled": True}}},
    )

    assert processed
    assert qc_rows
    peaks = processed[0].meta.get("features", {}).get("peaks", [])
    assert peaks, "Expected core pipeline to detect at least one peak."
    assert qc_rows[0]["peaks"] == len(peaks)

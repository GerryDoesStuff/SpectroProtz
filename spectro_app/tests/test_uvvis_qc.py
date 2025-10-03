import numpy as np

from spectro_app.engine.plugin_api import Spectrum
from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def test_uvvis_analyze_produces_qc_metrics():
    wl = np.linspace(400, 500, 21)
    intensity = np.linspace(0, 100, wl.size)
    intensity[:5] += np.array([0.0, 0.5, -0.3, 0.4, -0.2])
    intensity[10:] += 50.0
    intensity[12] += 75.0  # spike

    spec = Spectrum(
        wavelength=wl,
        intensity=intensity,
        meta={"sample_id": "S1", "role": "sample", "mode": "transmittance"},
    )

    recipe = {
        "qc": {
            "quiet_window": {"min": 400.0, "max": 420.0},
            "noise_rsd_limit": 1.0,
            "spike_limit": 0,
            "join_offset_limit": 100.0,
        },
        "join": {"enabled": True, "window": 2, "threshold": 20.0},
        "despike": {"window": 5, "zscore": 3.0},
        "smoothing": {"enabled": True, "window": 7, "polyorder": 2},
    }

    _, qc_rows = UvVisPlugin().analyze([spec], recipe)
    assert len(qc_rows) == 1
    row = qc_rows[0]

    assert row["saturation_flag"] is True
    assert row["spike_count"] >= 1
    assert row["join_count"] >= 1
    assert np.isfinite(row["noise_rsd"]) and row["noise_rsd"] >= 0
    assert "saturation" in row["flags"]
    assert "Spikes" in row["summary"]
    assert row["noise_window"] == (400.0, 420.0)

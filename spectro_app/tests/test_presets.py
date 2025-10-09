from __future__ import annotations

from pathlib import Path

import yaml


def test_uvvis_default_preset_includes_expected_params() -> None:
    preset_path = Path(__file__).resolve().parent.parent / "config" / "presets" / "uvvis_default.yaml"
    with preset_path.open("r", encoding="utf-8") as handle:
        preset = yaml.safe_load(handle)

    params = preset.get("params", {})

    assert params.get("baseline") == {
        "method": "asls",
        "lam": 100000.0,
        "p": 0.01,
        "niter": 10,
    }

    assert params.get("join") == {
        "enabled": True,
        "window": 3,
        "threshold": 0.2,
        "windows": {
            "helios": [
                {
                    "min_nm": 340.0,
                    "max_nm": 360.0,
                }
            ]
        },
    }

    assert params.get("qc") == {
        "quiet_window": {
            "min": 850.0,
            "max": 900.0,
        }
    }

    assert params.get("smoothing") == {
        "enabled": False,
        "window": 15,
        "polyorder": 3,
    }

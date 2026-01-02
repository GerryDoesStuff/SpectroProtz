import numpy as np
import pytest

from spectro_app.engine.pipeline import apply_solvent_subtraction
from spectro_app.engine.plugin_api import Spectrum


def _gaussian(x: np.ndarray, center: float, width: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def _spectrum(x: np.ndarray, y: np.ndarray, meta: dict | None = None) -> Spectrum:
    return Spectrum(wavelength=x, intensity=y, meta=meta or {})


def test_solvent_subtraction_recovers_scale_single_reference():
    x = np.linspace(1000.0, 1100.0, 201)
    ref = _gaussian(x, 1035.0, 4.5, amp=1.3) + _gaussian(x, 1075.0, 6.0, amp=0.6)
    solute = _gaussian(x, 1090.0, 3.0, amp=0.3)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.002, size=x.size)

    scale = 0.75
    target = scale * ref + solute + noise
    spec = _spectrum(x, target, meta={"role": "sample"})
    ref_spec = _spectrum(x, ref, meta={"reference_id": "solvent-a"})

    corrected = apply_solvent_subtraction(spec, ref_spec, fit_scale=True)
    solvent_meta = corrected.meta["solvent_subtraction"]

    assert solvent_meta["scale"] == pytest.approx(scale, rel=1e-2, abs=2e-3)
    assert solvent_meta["reference_id"] == "solvent-a"


def test_solvent_subtraction_recovers_coefficients_multi_reference():
    x = np.linspace(1000.0, 1100.0, 201)
    ref_a = _gaussian(x, 1020.0, 4.0, amp=1.0)
    ref_b = _gaussian(x, 1085.0, 5.5, amp=0.9)
    solute = _gaussian(x, 1055.0, 3.0, amp=0.25)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.001, size=x.size)

    coef_a, coef_b = 0.55, 1.15
    target = coef_a * ref_a + coef_b * ref_b + solute + noise
    spec = _spectrum(x, target, meta={"role": "sample"})
    ref_specs = [
        _spectrum(x, ref_a, meta={"reference_id": "ref-a"}),
        _spectrum(x, ref_b, meta={"reference_id": "ref-b"}),
    ]

    corrected = apply_solvent_subtraction(spec, ref_specs, fit_scale=True)
    solvent_meta = corrected.meta["solvent_subtraction"]
    assert solvent_meta["coefficients"] == pytest.approx([coef_a, coef_b], rel=2e-2, abs=2e-3)
    assert solvent_meta["reference_count"] == 2


def test_solvent_subtraction_recovers_shift():
    x = np.linspace(1000.0, 1100.0, 201)
    ref = _gaussian(x, 1045.0, 4.0, amp=1.0) + _gaussian(x, 1070.0, 6.0, amp=0.8)
    solute = _gaussian(x, 1090.0, 3.5, amp=0.2)
    shift = 0.3
    shifted_ref = np.interp(x - shift, x, ref)

    target = shifted_ref + solute
    spec = _spectrum(x, target, meta={"role": "sample"})
    ref_spec = _spectrum(x, ref, meta={"reference_id": "solvent-shift"})

    corrected = apply_solvent_subtraction(
        spec,
        ref_spec,
        fit_scale=True,
        shift_compensation={"enabled": True, "min": -1.0, "max": 1.0, "step": 0.05},
    )
    shift_meta = corrected.meta["solvent_subtraction"]["shift_compensation"]
    assert shift_meta["selected_shift"] == pytest.approx(shift, abs=0.1)


def test_solvent_subtraction_handles_partial_overlap():
    x = np.linspace(1000.0, 1100.0, 201)
    ref_x = np.linspace(1030.0, 1070.0, 81)
    ref = _gaussian(ref_x, 1050.0, 4.0, amp=1.2)
    solute = _gaussian(x, 1090.0, 3.5, amp=0.25)

    ref_interp = np.interp(x, ref_x, ref, left=np.nan, right=np.nan)
    solvent_signal = np.where(np.isfinite(ref_interp), ref_interp, 0.0)
    target = 0.8 * solvent_signal + solute
    spec = _spectrum(x, target, meta={"role": "sample"})
    ref_spec = _spectrum(ref_x, ref, meta={"reference_id": "partial"})

    corrected = apply_solvent_subtraction(spec, ref_spec, fit_scale=True)

    overlap_mask = np.isfinite(ref_interp)
    assert np.allclose(corrected.intensity[~overlap_mask], target[~overlap_mask])
    assert corrected.meta["solvent_subtraction"]["overlap_points"] == int(np.count_nonzero(overlap_mask))


def test_solvent_subtraction_returns_original_when_no_overlap():
    x = np.linspace(1000.0, 1100.0, 201)
    ref_x = np.linspace(2000.0, 2100.0, 50)
    ref = _gaussian(ref_x, 2050.0, 6.0, amp=1.0)
    target = _gaussian(x, 1060.0, 5.0, amp=0.4)

    spec = _spectrum(x, target, meta={"marker": "unchanged"})
    ref_spec = _spectrum(ref_x, ref, meta={"reference_id": "mismatch"})

    corrected = apply_solvent_subtraction(spec, ref_spec, fit_scale=True)

    assert corrected is spec
    assert corrected.meta["marker"] == "unchanged"


def test_solvent_subtraction_warnings_for_oversubtraction_and_derivative_residuals():
    x = np.linspace(1000.0, 1100.0, 201)
    ref = _gaussian(x, 1045.0, 4.5, amp=1.2)

    oversub_target = 0.2 * ref
    oversub_spec = _spectrum(x, oversub_target, meta={"role": "sample"})
    ref_spec = _spectrum(x, ref, meta={"reference_id": "over"})

    oversub = apply_solvent_subtraction(oversub_spec, ref_spec, scale=1.0, fit_scale=False)
    oversub_warnings = oversub.meta["solvent_subtraction"]["warnings"]
    assert any("Potential oversubtraction" in warning for warning in oversub_warnings)

    derivative = np.gradient(ref, x)
    derivative_target = ref + 0.8 * derivative
    derivative_spec = _spectrum(x, derivative_target, meta={"role": "sample"})

    derivative_corrected = apply_solvent_subtraction(
        derivative_spec,
        ref_spec,
        scale=1.0,
        fit_scale=False,
        shift_compensation={"enabled": False},
    )
    derivative_warnings = derivative_corrected.meta["solvent_subtraction"]["warnings"]
    assert any("Residuals correlated with reference derivative" in warning for warning in derivative_warnings)


def test_solvent_subtraction_is_deterministic():
    x = np.linspace(1000.0, 1100.0, 201)
    ref = _gaussian(x, 1035.0, 5.0, amp=1.0)
    solute = _gaussian(x, 1080.0, 4.0, amp=0.2)
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.001, size=x.size)

    target = 0.9 * ref + solute + noise
    spec = _spectrum(x, target, meta={"role": "sample"})
    ref_spec = _spectrum(x, ref, meta={"reference_id": "det"})

    first = apply_solvent_subtraction(spec, ref_spec, fit_scale=True)
    second = apply_solvent_subtraction(spec, ref_spec, fit_scale=True)

    assert np.allclose(first.intensity, second.intensity)
    first_meta = first.meta["solvent_subtraction"]
    second_meta = second.meta["solvent_subtraction"]
    assert first_meta["scale"] == pytest.approx(second_meta["scale"], rel=0.0, abs=0.0)
    assert first_meta["warnings"] == second_meta["warnings"]

# Spectroscopy Workflow Overview

This document summarises how batch jobs move through the Spectro processing
pipeline and where integrators can adjust behaviour through recipes and global
configuration files. Use it as the canonical entry point when wiring the
UV-Vis plugin into custom tooling; other modality plugins (FTIR, Raman, etc.)
should add comparable sections here as they expose similar controls.

## Batch lifecycle

Batches started through `RunController` fan out work onto the global Qt thread
pool. Each run executes the following stages in order:

1. **Load** – the plugin's `load` hook reads spectra from the supplied paths and
   normalises metadata. The stage is responsible for optional manifest joins
   and raw file decoding.【F:spectro_app/engine/run_controller.py†L10-L38】【F:spectro_app/plugins/uvvis/plugin.py†L171-L313】
   - *Normalises metadata* means that each spectrum's `meta` dict is cleaned so
     keys/values follow a predictable format (e.g. lower-casing roles,
     back-filling `blank_id`, defaulting the technique and source file). This
     prevents downstream code from juggling per-file quirks.【F:spectro_app/plugins/uvvis/plugin.py†L232-L309】
   - *Manifest join* refers to the merge between parsed manifest rows and the
     instrument metadata. The loader builds lookups by file, sample and channel
     so that the most specific manifest entry wins before copying its fields
     into each spectrum.【F:spectro_app/plugins/uvvis/plugin.py†L214-L355】
   - *Raw file decoding* covers the family of readers that convert vendor CSV,
     Excel or Helios `.dsp` exports into wavelength/intensity arrays while
     extracting header metadata and inferring blanks/modes.【F:spectro_app/plugins/uvvis/plugin.py†L248-L309】【F:spectro_app/plugins/uvvis/plugin.py†L758-L868】
2. **Validate** – `validate` checks recipe compatibility and returns any fatal
   issues. When errors are reported the run aborts without touching the
   remaining steps.【F:spectro_app/engine/run_controller.py†L16-L24】
3. **Preprocess** – `preprocess` applies wavelength domain coercion, join
   detection/correction, blank handling, baseline removal, smoothing, despiking
   and replicate aggregation before yielding spectra ready for analysis.【F:spectro_app/engine/run_controller.py†L21-L22】【F:spectro_app/plugins/uvvis/plugin.py†L927-L1186】
4. **Analyse** – feature extraction, QC metric computation and optional
   calibration are performed here. The stage enriches spectra metadata and
   returns QC rows for reporting.【F:spectro_app/engine/run_controller.py†L22-L23】【F:spectro_app/plugins/uvvis/plugin.py†L2598-L2869】
5. **Export** – final results, plots and audit logs are written according to
   the recipe's export configuration. Recipe sidecars can also be emitted for
   provenance.【F:spectro_app/engine/run_controller.py†L23-L24】【F:spectro_app/plugins/uvvis/plugin.py†L3457-L3503】

Integrations that display progress should listen to `RunController.job_started`
and `job_finished` signals to mirror this lifecycle in the UI.【F:spectro_app/engine/run_controller.py†L28-L38】

## Global processing toggles

Default processing switches live in `spectro_app/config/defaults.yaml` and are
merged into user recipes unless explicitly overridden. Key toggles include:

- `processing.common_grid.enabled` — snap spectra onto the shared wavelength
  grid defined by `start`, `stop` and `step`.
- `processing.smoothing.enabled` — apply global Savitzky–Golay smoothing using
  the configured window and polynomial order.
- `processing.baseline.method` — select the baseline removal algorithm; omitting
  `method` disables baseline correction. `lambda`, `p`, `iterations` and
  `niter` tune the solver.
- `processing.join_correction.enabled` — enable automatic join detection around
  detector overlap windows defined by `overlap_nm`.
- `processing.qc.saturation_abs` and `processing.qc.quiet_window_nm` — control
  QC thresholds used to flag saturated spectra and quiet-region noise checks.

Recipes can override these values directly (for example, setting
`processing.smoothing.enabled: true` in YAML) to enable/disable functionality on
per-batch bases.【F:spectro_app/config/defaults.yaml†L7-L24】

## UV-Vis plugin recipe controls

### Recipe defaults and overrides

`UvVisPlugin._apply_recipe_defaults` injects sensible defaults—baseline
configuration that matches expected instrument behaviour—when a recipe omits
them. Defaults include enabling join correction with a three-point window,
providing Helios overlap windows, and seeding QC with an 850–900 nm quiet
range. These values are considered “sensible” because they mirror the plugin's
internal constants; the helper only fills blanks so any values you set in the
recipe win.【F:spectro_app/plugins/uvvis/plugin.py†L90-L151】

Baseline-oriented parameters appear in several recipe blocks:

- `baseline.lambda` / `baseline.lam` – smoothness penalty for the AsLS solver;
  larger numbers enforce flatter baselines. Start from the default `1e5` and
  explore roughly `1e3–1e7` depending on how aggressive the background drift
  is.【F:spectro_app/config/defaults.yaml†L17-L22】【F:spectro_app/tests/test_pipeline_uvvis.py†L118-L151】
- `baseline.p` – asymmetry weight that decides how strongly positive residuals
  are penalised. Typical UV-Vis data works between `0.001` and `0.05`; lower
  values cling to peaks while higher values suppress them faster.【F:spectro_app/config/defaults.yaml†L17-L22】【F:spectro_app/tests/test_pipeline_uvvis.py†L118-L151】
- `baseline.niter` – number of AsLS refinement passes. Keep this in the
  `10–30` range to balance speed and convergence.【F:spectro_app/plugins/uvvis/pipeline.py†L282-L318】【F:spectro_app/tests/test_pipeline_uvvis.py†L118-L151】
- `baseline.iterations` – iteration count used by SNIP baselines; defaults to
  `24`, and values between `20` and `60` cover most fluorescence-contaminated
  spectra.【F:spectro_app/plugins/uvvis/pipeline.py†L282-L318】
- Anchor helpers (`baseline.anchor`, `baseline.anchor_windows`,
  `baseline.anchors`, `baseline.zeroing`) let you pin corrected spectra to
  specific windows/targets; provide ranges as `min`/`max` pairs with optional
  `target` intensity to zero or level-shift outputs.【F:spectro_app/plugins/uvvis/plugin.py†L1057-L1126】【F:spectro_app/plugins/uvvis/pipeline.py†L323-L420】

### Relaxing blank enforcement

`disable_blank_requirement(recipe)` returns a deep copy with `blank.require`
set to `false`, allowing integrations to skip hard failures when blanks are
absent while preserving subtraction defaults. Use this helper before submitting
recipes through automation when instrument workflows cannot guarantee blank
captures.【F:spectro_app/plugins/uvvis/plugin.py†L12-L74】

### Preprocessing switches

The preprocessing stage honours the following recipe keys:

- **Domain limiting** – `domain.min` / `domain.max` trim spectra to the desired
  wavelength span while respecting instrument limits (190–1100 nm for Helios
  and generic drivers). Keep requested bounds inside those envelopes to avoid
  `ValueError`s.【F:spectro_app/plugins/uvvis/plugin.py†L80-L188】【F:spectro_app/plugins/uvvis/plugin.py†L927-L978】
- **Blank handling** – `blank.subtract`, `blank.require`,
  `blank.validate_metadata`, `blank.default`/`blank.fallback`,
  `blank.match_strategy`, `blank.max_time_delta_minutes`
  (default 240 minutes) and `blank.pathlength_tolerance_cm` (default 0.01 cm)
  govern subtraction and guard-rails. Time windows should reflect instrument
  drift; stick to sub-day spans unless long acquisitions demand otherwise.
  Set `blank.match_strategy` to `cuvette_slot` when blanks should be grouped
  and matched via `meta["cuvette_slot"]`; the pipeline automatically falls
  back to legacy blank identifiers when slot metadata is unavailable so
  existing recipes continue to work.【F:spectro_app/plugins/uvvis/plugin.py†L60-L78】【F:spectro_app/plugins/uvvis/plugin.py†L978-L1268】
- **Join detection/correction** – `join.enabled`, `join.window`,
  `join.threshold`, `join.windows`, `join.offset_bounds`, `join.min_offset` and
  `join.max_offset` manage detector stitching. Windows are provided per
  instrument; detection defaults to a 10-point neighbourhood, so keep custom
  values above `3` to ensure enough data for offset estimation.【F:spectro_app/plugins/uvvis/plugin.py†L60-L151】【F:spectro_app/plugins/uvvis/plugin.py†L951-L1015】【F:spectro_app/plugins/uvvis/pipeline.py†L700-L884】
- **Despiking** – `despike.enabled`, `despike.zscore` and `despike.window`
  remove impulsive noise. `zscore` is typically `2.5–6.0`; the default five
  point window balances spike suppression with peak preservation.【F:spectro_app/plugins/uvvis/plugin.py†L1015-L1037】【F:spectro_app/tests/test_pipeline_uvvis.py†L170-L206】
- **Baseline correction** – `baseline.method` selects `asls`, `rubberband` or
  `snip`. Combine with the parameters listed above; choose `asls` when gradual
  drift dominates, `rubberband` for broad curvature and `snip` for fluorescence
  streaks requiring many iterations.【F:spectro_app/plugins/uvvis/pipeline.py†L282-L318】
- **Smoothing** – `smoothing.enabled`, `smoothing.window` and
  `smoothing.polyorder` drive Savitzky–Golay smoothing (window must be odd and
  larger than the polynomial order). Defaults of window `15`/polyorder `3`
  suit 1 nm sampling; shrink for sparse data.【F:spectro_app/config/defaults.yaml†L12-L19】【F:spectro_app/plugins/uvvis/plugin.py†L986-L1111】
- **Replicate handling** – `replicates.average` toggles averaging, while the
  nested `replicates.outlier` block is forwarded to the replicate pipeline for
  robust aggregation (supply thresholds such as Z-score cut-offs as needed).【F:spectro_app/plugins/uvvis/plugin.py†L990-L1054】

### Additional feature gates

- **Manifest ingestion** – construct `UvVisPlugin(enable_manifest=False)` to
  bypass manifest parsing. Leave it `True` (the default) to merge CSV/Excel
  descriptors using the join precedence described above.【F:spectro_app/plugins/uvvis/plugin.py†L90-L355】
- **Feature extraction** – populate the `features` section to override peak
  detection (`prominence`, `min_distance`, `max_peaks`, `height`), band ratios
  (`numerator`, `denominator`, `bandwidth`), integrals (`min`/`max` windows),
  derivatives (toggle or provide `window`/`polyorder`), isosbestic checks
  (wavelength pairs plus optional `tolerance`) and kinetics (`targets` with
  `wavelength`/`bandwidth` and an optional `time_key`). Omit the block to fall
  back to bundled defaults for nucleic-acid QC.【F:spectro_app/plugins/uvvis/plugin.py†L2600-L2811】【F:spectro_app/plugins/uvvis/plugin.py†L1669-L1934】
- **Calibration** – enable by setting `calibration.enabled: true` and providing
  target definitions (`calibration.targets`), standards and unknown IDs along
  with instrument specifics (`bandwidth`, `pathlength_cm`, detection limits,
  optional `r2_threshold`). Leaving `enabled` false skips quantitation entirely
  but still records spectra-level QC.【F:spectro_app/plugins/uvvis/plugin.py†L2106-L2368】

## Presets and recipe authoring

Shared UV-Vis defaults live in `spectro_app/plugins/uvvis/presets.yaml`. The
`defaults` block mirrors the recipe structure (for example, the same `join` and
`smoothing` keys) and is merged before user recipes are applied. Integrators can
copy snippets from this file into YAML/JSON recipes to toggle behaviour, e.g.:

```yaml
join:
  enabled: false
smoothing:
  enabled: true
  window: 11
  polyorder: 3
```

Coupling these recipe overrides with the global switches in
`spectro_app/config/defaults.yaml` gives fine-grained control over batch
processing without code changes.【F:spectro_app/plugins/uvvis/presets.yaml†L1-L17】【F:spectro_app/config/defaults.yaml†L7-L24】

## Future extensions

When adding new modality plugins (FTIR, Raman, etc.), extend this document with
matching sections that describe their preprocessing toggles, analysis features
and preset files. Keeping the overview centralised helps developers discover
relevant recipe keys and prevents configuration drift across techniques.

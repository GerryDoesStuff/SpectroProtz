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

`UvVisPlugin._apply_recipe_defaults` injects sensible defaults for join and QC
configuration when a recipe omits them. Join correction is enabled by default
with a three-point window and instrument-specific overlap windows. QC gets a
quiet-window fallback spanning 850–900 nm. Recipes can override any of these
values by supplying `join.enabled`, `join.window`, `join.threshold`, custom
`join.windows`, or alternate `qc.quiet_window` ranges.【F:spectro_app/plugins/uvvis/plugin.py†L103-L151】

### Relaxing blank enforcement

`disable_blank_requirement(recipe)` returns a deep copy with `blank.require`
set to `false`, allowing integrations to skip hard failures when blanks are
absent while preserving subtraction defaults. Use this helper before submitting
recipes through automation when instrument workflows cannot guarantee blank
captures.【F:spectro_app/plugins/uvvis/plugin.py†L12-L74】

### Preprocessing switches

The preprocessing stage honours the following recipe keys:

- **Domain limiting** – `domain.min` / `domain.max` trim spectra to the desired
  wavelength span while respecting instrument limits.【F:spectro_app/plugins/uvvis/plugin.py†L927-L978】
- **Blank handling** – `blank.subtract`, `blank.require`, `blank.validate_metadata`,
  `blank.default`/`blank.fallback`, `blank.max_time_delta_minutes` and
  `blank.pathlength_tolerance_cm` govern subtraction, enforcement windows and
  metadata validation.【F:spectro_app/plugins/uvvis/plugin.py†L978-L1153】
- **Join detection/correction** – `join.enabled`, `join.window`, `join.threshold`,
  `join.windows`, `join.offset_bounds`, `join.min_offset` and `join.max_offset`
  manage detector stitching across channel joins.【F:spectro_app/plugins/uvvis/plugin.py†L951-L1015】
- **Despiking** – `despike.enabled`, `despike.zscore` and `despike.window` remove
  impulsive noise, optionally respecting join indices.【F:spectro_app/plugins/uvvis/plugin.py†L1015-L1037】
- **Baseline correction** – `baseline.method` toggles the algorithm; supporting
  parameters like `baseline.lambda`, `baseline.p`, `baseline.niter`,
  `baseline.iterations` and anchor/zeroing options adjust behaviour.【F:spectro_app/plugins/uvvis/plugin.py†L1057-L1091】
- **Smoothing** – `smoothing.enabled`, `smoothing.window` and
  `smoothing.polyorder` drive Savitzky–Golay smoothing for samples and blanks.
  Window length must be odd and exceed the polynomial order.【F:spectro_app/plugins/uvvis/plugin.py†L986-L1111】
- **Replicate handling** – `replicates.average` toggles averaging, while the
  nested `replicates.outlier` block is forwarded to the replicate pipeline for
  robust aggregation.【F:spectro_app/plugins/uvvis/plugin.py†L990-L1054】

### Additional feature gates

- **Manifest ingestion** – constructing `UvVisPlugin(enable_manifest=False)`
  disables CSV/Excel manifest parsing for environments without curated sample
  descriptors.【F:spectro_app/plugins/uvvis/plugin.py†L90-L263】
- **Feature extraction** – `features` sections control peaks, band ratios,
  integrals, derivatives, isosbestic checks and kinetics. Omitting them falls
  back to built-in defaults for common UV-Vis reports.【F:spectro_app/plugins/uvvis/plugin.py†L2602-L2709】
- **Calibration** – `calibration.enabled`, `calibration.targets`, `calibration.standards`,
  `calibration.unknowns`, `calibration.bandwidth`, `calibration.pathlength_cm`,
  `calibration.lod_multiplier`, `calibration.loq_multiplier`, and optional
  `calibration.r2_threshold` tune the quantitation workflow and reporting
  payloads.【F:spectro_app/plugins/uvvis/plugin.py†L2106-L2368】

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

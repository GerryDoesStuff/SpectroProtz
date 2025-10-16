# UV-Vis Signal Processing Reference

## Utility Infrastructure

### `_nan_safe`
- Interpolates over NaNs (falling back to zeros when every value is missing) so that downstream calculations always receive finite arrays.

### `_update_channel`
- Copies processed traces into the spectrum metadata under a named channel, leaving existing values intact unless overwriting is requested.

## Domain and Blank Handling

### `coerce_domain`
- Sorts and de-duplicates the wavelength axis, aggregates duplicate intensities, optionally clips or resamples to requested bounds, records domain metadata, and stores the processed trace in the `raw` channel.

### `subtract_blank`
- Interpolates a blank spectrum onto the sample grid, subtracts the overlapping region, records overlap statistics, and writes the corrected trace to the `blanked` channel while preserving prior audit data.

## Baseline Correction

### `_baseline_asls`
- Runs the asymmetric least-squares smoother with iteratively reweighted penalties to obtain a smooth, asymmetric baseline estimate.

### `_lower_hull`
- Builds the lower convex hull of (x, y) points, providing the geometric foundation for the rubber-band baseline.

### `_baseline_rubberband`
- Fits the lower-hull “rubber band” baseline by interpolating between hull vertices across the original wavelength axis.

### `_baseline_snip`
- Applies the SNIP iterative smoothing procedure, shrinking toward neighbourhood averages while protecting the spectrum ends.

### `apply_baseline`
- Dispatches to the requested baseline method (ASLS, rubber-band, or SNIP), subtracts it, applies optional anchor corrections, and records the corrected channel and method metadata.

### `_apply_baseline_anchor`
- Normalises anchor configuration, measures per-window offsets, interpolates a smooth correction, hard-sets anchor regions to their targets, and reports detailed metadata about the adjustments.

### `_normalise_anchor_windows`
- Canonicalises user-provided anchor window definitions (dicts or sequences) into consistent `{lower_nm, upper_nm, target}` records, fixing inverted bounds and preserving labels.

## Join Detection and Correction

### `normalise_join_windows`
- Canonicalises join windows into explicit wavelength bounds while preserving per-window `spikes` limits (defaulting to a single join per window) so plateau-style detector discontinuities can be targeted independently of narrow spikes.

### `detect_joins`
- Computes rolling-median change scores, constrains candidates to configured windows, short-circuits when an explicit threshold cannot be met, derives an adaptive MAD-based threshold, recursively segments to isolate peaks, refines indices by local gradients and spacing rules, and trims results to each window’s spike allowance so only plateau discontinuities at detector overlaps are corrected.

### `correct_joins`
- Measures pre/post window means, overlap errors, and robust median offsets around each detected join, clamps corrections to optional bounds, adjusts downstream samples, and logs raw/corrected segments plus rich diagnostics focused on stitching detectors that diverge by a step change.

## Spike Removal and Smoothing

### `despike_spectrum`
- Runs the adaptive spike remover across the full spectrum (optionally respecting explicit `join_indices` when supplied), builds a rolling baseline and spread (median/MAD by default, with optional rolling standard deviation), replaces samples exceeding ``baseline ± zscore * spread`` with the local baseline, and iterates until a pass produces no new spikes or ``max_passes`` is reached. ``baseline_window``, ``spread_window``, ``spread_method``, ``spread_epsilon``, ``residual_floor`` and ``max_passes`` remain configurable so recipes can tighten or relax the adaptive rejection behaviour. ``leading_padding`` and ``trailing_padding`` can now reserve untouched samples at either end of the trace (default ``0``) so steep slopes or guard bands survive despiking intact, while configurable spike exclusion regions let recipes fence off transient features that should remain untouched. When a spike is replaced the corrected point is jittered by a zero-mean Gaussian draw scaled to the same local spread estimate, keeping the repaired sample inside the ambient noise floor. ``noise_scale_multiplier`` expands or shrinks that perturbation (set it to ``0`` to suppress the jitter entirely), while ``rng``/``rng_seed`` enable deterministic playback by passing an explicit Generator or seed to ``numpy.random.default_rng``.
- ``tail_decay_ratio`` tunes how quickly spike shoulders must fall off before they are absorbed into the replacement window, letting recipes loosen the decay requirement when broader artefacts need to be flattened without splitting them into multiple spikes.
- Supply exclusion bands via ``despike.exclusions.windows`` entries with ``min_nm``/``max_nm`` bounds—parsed by :func:`normalise_exclusion_windows`—to keep narrow spectral features untouched while leaving adjacent spikes removable.【F:spectro_app/plugins/uvvis/pipeline.py†L548-L620】【F:spectro_app/plugins/uvvis/plugin.py†L1259-L1285】【F:spectro_app/tests/test_pipeline_uvvis.py†L202-L227】

### `smooth_spectrum`
- Runs Savitzky–Golay smoothing across the full trace or per segment between joins, automatically shrinking the window to fit each segment and tracking whether segmentation was necessary.

## Replicate Management

### `replicate_key`
- Builds a `(role, identifier)` tuple so blanks and samples group independently even when metadata is sparse, defaulting to blank identifiers unless a slot-based strategy is supplied via the higher-level helpers.

### `_replicate_feature_matrix`
- Generates robust summary features (bias, mean, spread, slope, energy, area) for each replicate, normalising feature columns to stabilise downstream scoring.

### `_compute_cooksd_scores`
- Fills missing intensity values, derives the replicate feature matrix, computes leverage, fits a multivariate least squares model, and produces Cook’s distance scores for outlier screening.

### `average_replicates`
- Groups spectra by replicate key, applies optional MAD- or Cook’s-distance outlier rejection, averages the retained traces, and records provenance, outlier metrics, and segment metadata for each group. Callers may supply a custom key function so blanks can be grouped using cuvette slots instead of legacy identifiers.

### `blank_match_identifier`
- Returns the primary identifier for a blank spectrum according to the configured match strategy, preferring cuvette slots when requested and gracefully falling back to legacy identifiers when slot metadata is absent.

### `blank_match_identifiers`
- Enumerates all viable identifiers for matching a blank (slot, explicit blank ID, sample ID, channel), enabling downstream lookups to support both slot-based and legacy association in a single pass.

### `blank_replicate_key`
- Builds replicate keys for blanks using the active match strategy so that averaging collapses slot-based replicates even when their `blank_id` values differ.

### `normalize_blank_match_strategy`
- Canonicalises user-provided strategy strings (e.g. `"slot"`, `"cuvette-slot"`) into the supported tokens (`"blank_id"` or `"cuvette_slot"`).

## Blank Identification

### `blank_identifier`
- Emits a stable identifier for blank spectra based on the selected match strategy—preferring cuvette slots when enabled and otherwise using blank/sample/channel metadata—returning `None` for non-blank roles.

## Intensity Mode Harmonisation

### `normalise_mode`
- Maps free-form acquisition mode text into canonical labels (`absorbance`, `transmittance`, `reflectance`) to standardise processing pipelines.

### `convert_intensity_to_absorbance`
- Converts transmission/reflectance intensities into absorbance (detecting whether data are percentages), records the original trace channel, and returns both the updated and original acquisition modes for auditing.

---

### Testing
Covered by automated test suite.

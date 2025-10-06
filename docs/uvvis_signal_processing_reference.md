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
- Reuses the anchor normaliser to produce explicit wavelength bounds that focus join detection on known overlap regions.

### `detect_joins`
- Computes rolling-median change scores, constrains candidates to configured windows, derives an adaptive MAD-based threshold, recursively segments to isolate peaks, and refines indices by local gradients and spacing rules.

### `correct_joins`
- Measures pre/post window means, overlap errors, and robust median offsets around each detected join, clamps corrections to optional bounds, adjusts downstream samples, and logs raw/corrected segments plus rich diagnostics.

## Spike Removal and Smoothing

### `despike_spectrum`
- Applies a moving-median baseline globally or per join-segment, flags spikes using a MAD-scaled z-score, replaces outliers via interpolation (or baseline values), and reports which samples were modified.

### `smooth_spectrum`
- Runs Savitzky–Golay smoothing across the full trace or per segment between joins, automatically shrinking the window to fit each segment and tracking whether segmentation was necessary.

## Replicate Management

### `replicate_key`
- Builds a `(role, identifier)` tuple so blanks and samples group independently even when metadata is sparse.

### `_replicate_feature_matrix`
- Generates robust summary features (bias, mean, spread, slope, energy, area) for each replicate, normalising feature columns to stabilise downstream scoring.

### `_compute_cooksd_scores`
- Fills missing intensity values, derives the replicate feature matrix, computes leverage, fits a multivariate least squares model, and produces Cook’s distance scores for outlier screening.

### `average_replicates`
- Groups spectra by replicate key, applies optional MAD- or Cook’s-distance outlier rejection, averages the retained traces, and records provenance, outlier metrics, and segment metadata for each group.

## Blank Identification

### `blank_identifier`
- Emits a stable identifier for blank spectra based on blank, sample, or channel metadata when the role is “blank,” returning `None` otherwise.

## Intensity Mode Harmonisation

### `normalise_mode`
- Maps free-form acquisition mode text into canonical labels (`absorbance`, `transmittance`, `reflectance`) to standardise processing pipelines.

### `convert_intensity_to_absorbance`
- Converts transmission/reflectance intensities into absorbance (detecting whether data are percentages), records the original trace channel, and returns both the updated and original acquisition modes for auditing.

---

### Testing
⚠️ Not run (documentation update only).

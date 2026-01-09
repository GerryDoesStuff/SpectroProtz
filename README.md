# SpectroProtz

## Overview
SpectroProtz is a PyQt-based desktop application for managing spectroscopy
batches, visualising spectra, and exporting analysis artefacts. The GUI bundles
processing helpers, batch orchestration, and recipe management behind a single
entry point so analysts can focus on domain workflows instead of wiring
individual scripts. Internally, the launcher hands control to
[`spectro_app/main.py`](spectro_app/main.py), which wires up the Qt event loop
and the shared application context.
During batch runs, the main window keeps the status bar and log panel updated
with per-spectrum progress messages (including the spectrum ID when available)
so operators can see each spectrum complete in real time alongside the overall
progress bar, which also surfaces the running spectrum counts.
Operators can cancel a running job from the toolbar or **Process → Cancel**; the
UI switches the action to a “Cancelling…” state while the engine checks
cancellation signals during file ingestion and pipeline processing, then reports
“Job cancelled” once the background work has halted.
When closing the application, SpectroProtz warns if a processing job is still
running, cancels the job on request, and waits for background thread-pool work
to finish before the UI is destroyed to avoid leaving silent work in flight.

## Prerequisites
Ensure your environment matches the expectations declared in
[`pyproject.toml`](pyproject.toml):

- Python **3.10 or newer** (`requires-python = ">=3.10"`).
- Qt bindings via **PyQt6 >= 6.6**.
- Scientific stack: **numpy >= 1.26**, **scipy >= 1.12**, **pandas >= 2.0**,
  **matplotlib >= 3.8**, **pyqtgraph >= 0.13**.
- File/utility helpers: **openpyxl >= 3.1**, **pillow >= 10.0**, **PyYAML >= 6.0**.
- FTIR indexer backend: **duckdb >= 0.8**.

A typical workflow uses a virtual environment so the above packages can be
installed without affecting system Python.

### FTIR OPUS support (SpectroChemPy)
SpectroProtz reads FTIR OPUS files through **SpectroChemPy**. Install it in the
same environment as the app:

```bash
pip install spectrochempy
```

The OPUS importer expects SpectroChemPy datasets to provide `values`/`data` for
the Y axis and `x`/`coordset.x` for the X axis. If SpectroChemPy cannot parse an
OPUS file or those accessors are missing, the application surfaces the raw
exception (type + message) in the UI and records the full traceback in the
log folder. Logs are accessible via **Tools → Open Log Folder**.
Accessor selection is done with explicit `None` checks so Pint-backed arrays
never trigger boolean evaluation while loading spectra.

OPUS ingestion accepts both `.opus` files and Bruker-style numeric suffixes
such as `.0`, `.1`, or `.000`, routing them through the same OPUS reader for
FTIR and Raman workflows. Solvent reference imports use the same acceptance
rules so curated solvent libraries can be sourced from `.opus` or numeric OPUS
extensions without renaming.

## Supported techniques and file formats
SpectroProtz focuses on FTIR, Raman, and UV-Vis processing. The current import
coverage includes:

- **FTIR:** JCAMP-DX (`.jdx`) and OPUS (`.opus`, Bruker numeric suffixes) when
  SpectroChemPy is available.
- **Raman:** CSV/TXT tabular exports and OPUS (`.opus`, Bruker numeric suffixes)
  when SpectroChemPy is available.
- **UV-Vis:** Thermo Helios `.dsp` files, generic CSV/TXT delimited text, and
  Excel (`.xls`, `.xlsx`) workbooks.

## Setup
From a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install -e .
```

The editable install exposes the `spectro_app` package to your interpreter
while keeping the source tree editable for rapid iteration.

## Launching the GUI
Once dependencies are installed and the virtual environment is active, start the
application by running:

```bash
python -m spectro_app.main
```

On Windows you can launch the GUI directly with the bundled
[`launch_spectro_app.cmd`](launch_spectro_app.cmd) helper, which takes care of
activating `.venv` when available before delegating to the same module entry
point. This command bootstraps the Qt event loop and shows the main window
immediately, allowing users to load recipes, configure processing options, and
monitor batch runs.

## Solvent reference management
FTIR solvent subtraction relies on curated solvent references that can be saved,
reviewed, and edited from within the GUI. From the file queue, choose **Save as
FTIR Solvent Reference…** to store the currently selected spectrum. The metadata
dialog captures a reference name, optional tags, and contextual fields such as
solvent identity, measurement mode, temperature, instrument, and date. Saved
entries are stored in `~/SpectroApp/solvent_references.json` by default and are
available in the **Reference** selector inside the recipe editor’s solvent
subtraction panel.

The solvent subtraction panel now includes selection controls for multi-
reference fits: best-reference selection runs per spectrum, you can choose the
selection metric (RMSE or pattern correlation), and optionally apply region
weighting with a global coverage requirement plus a focused window to emphasize
diagnostic bands. These settings are stored in the recipe payload so preset
exports and subsequent runs reuse the same selection criteria.

When you open **Select Solvent Reference**, the dialog lists stored entries,
shows their metadata, and lets you browse a new file. Use **Edit Metadata…** to
update the name, tags, metadata fields, or default status of an existing
reference; changes are saved back to the solvent reference store immediately so
future selections reflect the updated information. The edit action stays
disabled until a list entry is selected, and selecting a reference refreshes the
detail pane immediately.

## Peak detection configuration
The recipe editor’s **Peak detection** panel focuses on tuning how peaks are
found, filtered, and classified. Enable peak detection to record peak
wavelength, height, and width statistics in analysis outputs, then adjust the
prominence floor, minimum absorbance threshold, noise sigma multiplier, and
noise window (global or regional) to control which candidates are kept. Use the
minimum-distance controls to enforce spacing between peaks, optionally switching
to adaptive spacing based on median FWHM and tightening or loosening the
distance fraction. Width bounds, a maximum candidate count, and a negative-peak
toggle help shape the candidate set further. Additional controls cover merge and
close-peak tolerances, plateau detection sensitivity, shoulder detection
thresholds, and optional CWT-based detection with configurable width ranges and
cluster tolerances so the feature set stays consistent across UV-Vis and FTIR
recipes.

## FTIR reference lookup parsing
The FTIR indexer’s DuckDB outputs can be queried with search-bar style input
using the helper in `spectro_app/engine/ftir_lookup.py`. The parser accepts
numeric peak positions with optional tolerances (for example `1720±5` or
`1720 +/- 5`) and metadata filters written as `key:value` or `key=value`
tokens. Filters map to promoted FTIR metadata columns such as `title`, `origin`,
`cas`, `names`, `molform`, `state`, and `nist_source`, with aliases like `name`
or `formula` automatically normalized. The parser returns structured criteria
with error messages for invalid tokens, then maps them to parameterized DuckDB
SQL: one query for matching reference spectra and another for the peak rows
that satisfy the same metadata filters plus any requested peak ranges. Empty or
whitespace-only searches are handled defensively by returning queries that
produce no rows, ensuring the lookup flow never issues an unbounded query by
default.

## FTIR reference lookup window
SpectroProtz now includes a dedicated **FTIR Reference Lookup** window
accessible under **Tools → FTIR Reference Lookup...**. The dialog lets you
point at a `peaks.duckdb` index, run manual searches from a query bar, and view
matching reference spectra in the left-hand results sidebar. Manual search text
is debounced, so the lookup only runs after you pause typing instead of on
every keystroke, keeping the UI responsive on large indexes. The results list
surfaces the spectrum name, molecular formula, and summary match statistics
(including matched peak counts) so you can spot likely candidates quickly while
keeping the original manual query text intact. The list supports multi-selection
and paginates through large result sets with a results cap so the sidebar
remains responsive when a search returns many references. Use the **Add to plot**
button or the left
sidebar context menu to move reference spectra into the right-hand plotting
sidebar, which preserves the order you add items.
The lookup dialog validates the selected DuckDB file before it queries, and it
reports clear errors when the database is missing, unreadable, or cannot be
opened. Searches that return no matches show an empty-state message in the
results list so you know the query executed successfully but yielded no
references.

The right sidebar keeps a stable list of selected references with remove controls
via a **Remove selected** button or right-click menu. Every add, remove, or
selection change batches re-plotting of the top comparison chart in the lookup
window, which draws normalized reference spectrum traces and overlays vertical
peak sticks for each selected reference. The chart also overlays the currently
selected preview spectrum as a normalized trace, scaling each spectrum’s
intensities independently from the reference peak heights so all plotted data
fits in a 0–1 range without changing internal relative amplitudes. Selecting a
subset of references in the right sidebar filters the plot to just those
entries, while leaving the list unselected shows the full selected-reference
set. Search actions do not change the plot until you explicitly add or remove
references, keeping comparisons stable while you browse new matches.
Right-clicking any reference in the left or right sidebar previews that
reference’s normalized spectrum and peak sticks in the bottom plot immediately,
and the metadata panel beside the chart summarizes the selected reference’s
name, formula, and related header fields for quick inspection without leaving
the lookup dialog.
Reference spectrum traces and peak sticks are cached within the lookup window
session so repeated plot updates avoid redundant database or file reads when
revisiting the same references.
If the lookup database contains malformed peak rows (for example missing or
non-numeric peak data), the plot skips those rows and reports a warning in the
dialog status area instead of halting the preview.
When you are ready to compare candidates against live preview data, use
**Send plot to main preview** to push the currently plotted reference spectra
into the main preview plot as an overlay. The preview dock now includes an
**Identified** checkbox alongside the stage toggles to show or hide those
reference overlays on demand, keeping the main plot uncluttered while you work
through match candidates.

Preview peak identification can also feed the lookup workflow without
overwriting manual search state. From the preview plot context menu, choose
**Use selected peaks for reference lookup** to capture the currently selected
spectrum’s FTIR peaks. The lookup window stores those peaks as a pending
auto-search request, shows how many were captured, and waits for you to press
**Apply preview peaks** (with a configurable tolerance) before it refreshes the
left sidebar with auto-search matches. Manual search inputs remain unchanged
until you explicitly trigger the preview-driven search.

### FTIR lookup search syntax
Manual searches accept peak tokens, optional tolerances, and metadata filters in
a single query. Peak tokens are interpreted as wavenumbers (cm⁻¹). You can
combine multiple peaks and filters to narrow results.

- **Single peak:** `1720` uses the default ±5 cm⁻¹ tolerance.
- **Multiple peaks:** `1720±2 1600±3` requires both peak ranges to match.
- **Tolerance syntax:** `±`, `+/-`, or `+-` are accepted (for example
  `1720 +/- 5`). Optional unit suffixes include `cm-1`, `cm^-1`, or `cm⁻¹`.
- **Metadata filters:** `key:value` or `key=value`, e.g.
  `name:acetone origin:"NIST" 1720±5`. Use quotes for values with spaces.

Lookup results are not explicitly re-ranked by a weighted score; the sidebar
shows a **Matched peaks** count per result so you can prioritize candidates
with higher peak coverage when multiple peaks are requested.

## Verifying processed spectra
SpectroProtz keeps a complete audit trail for every spectrum so you can confirm
that processed traces remain representative of their raw counterparts:

- **Stage-by-stage channels.** Each processing step writes its result into a
  dedicated metadata channel (for example `raw`, `blanked`,
  `baseline_corrected`, `solvent_subtracted` for FTIR, `joined`, `despiked`,
  `smoothed`). The original signal stays available alongside every intermediate,
  enabling numerical and visual comparisons at any point in the pipeline.
- **Interactive stage toggles.** The preview dock in the UI lets you toggle the
  visibility of stored stages so you can overlay the processed curve on top of
  the raw trace or inspect any intermediate discrepancies. When the preview is
  showing FTIR data (identified by `technique == "ftir"` or a wavenumber axis),
  an extra **Solvent Subtracted** stage appears so you can compare solvent-
  corrected traces; non-FTIR previews hide this toggle.
- **Peak markers.** When peak detection metadata is available, the preview
  plot overlays vertical peak lines at detected positions and provides a
  **Peaks** toggle to show or hide the markers alongside the spectra.
- **Identified overlays.** Reference spectra sent from the FTIR lookup window
  appear as dashed overlays in the main preview plot and can be toggled with
  the **Identified** checkbox to quickly compare current samples to matched
  references without leaving the preview.
- **Quantitative QC metrics.** The QC engine computes diagnostics such as noise
  levels, join offsets, spike counts, smoothing guards, drift, and per-stage
  roughness deltas that compare processed spectra against their retained
  channels. These metrics roll up into QC flags that highlight when processing
  diverges too far from the source data.
- **Solvent subtraction diagnostics.** When FTIR solvent subtraction is enabled,
  the pipeline records overlap metrics (RMSE, normalized RMSE, residual/derivative
  correlations, condition numbers) and emits warnings if overlap quality drops
  below configured thresholds. If the reference is identical to the sample and
  variance collapses to zero, the diagnostics are skipped and a warning notes
  that the correlation and normalization checks were suppressed to avoid divide-by-zero
  noise. Solvent subtraction uses nearest-value edge extrapolation so the
  reference remains finite at the sample boundaries, allowing subtraction across
  the full sample range while keeping fit diagnostics scoped to the true overlap.
  The metadata channel records the edge handling strategy under
  `meta["solvent_subtraction"]["edge_strategy"]` for traceability. When multiple
  solvent references are selected (or the multi-reference option is enabled),
  the pipeline evaluates each reference individually and chooses the best fit
  per spectrum once a minimum overlap coverage requirement is satisfied;
  candidates that fall below the overlap threshold are excluded from selection
  so the chosen reference remains representative of the sample. The selection
  metric can be RMSE or pattern correlation, and scoring can optionally blend
  global overlap coverage with a weighted window so emphasized bands influence
  the final rank. Shift compensation, scale, and offset fitting are applied to
  every candidate before the metric comparison. The per-reference fit details
  (reference identifier, RMSE, pattern correlation, selection metric/score,
  shift, scale, offset, overlap points, and any region weighting settings) are
  captured in `meta["solvent_subtraction"]["candidate_scores"]` so debugging and
  audits can review every candidate considered, matching the recipe editor
  option to select the best reference by RMSE or pattern correlation. When all
  candidates are rejected due to insufficient overlap, the solvent subtraction
  metadata records a warning explaining that the overlap requirement was not
  met and the subtraction was skipped.
- **Parallel per-spectrum FTIR processing.** FTIR batches can fan out the full
  per-spectrum pipeline in multiprocessing when multiple spectra are queued and
  more than one worker is available. The
  pipeline builds explicit per-spectrum tasks (recipe snapshot + axis metadata +
  sample data) and executes them in a spawn-safe process pool so coerce-domain,
  stitching, join correction, despiking, blank subtraction, baseline correction,
  solvent subtraction, smoothing, and peak detection all run in subprocesses.
  Results are reassembled in input order for deterministic outputs (matching the
  input list regardless of completion order), and Windows uses the spawn start
  method so each worker starts with a clean interpreter state. Any failed
  spectrum tasks are omitted from the final result set so partial success is
  still returned. Worker tasks capture exceptions (type/message/stack trace) and
  the main process logs each failure alongside the spectrum ID/path in the same
  log folder exposed by **Tools → Open Log Folder**, then continues processing
  the remaining spectra. Solvent-subtraction exceptions are logged similarly and
  fall back to the original pre-solvent spectrum for that entry, with a warning
  summarizing how many spectra failed solvent subtraction at the end of the run.
  FTIR multiprocessing defaults to a worker count of `min(4, os.cpu_count())`,
  while `chunk_size` batches multiple spectra per worker submission and
  `max_tasks_per_child` can recycle processes to limit long-lived memory growth.
  The per-spectrum progress callback fires for every completed spectrum even
  when multiprocessing is active, ensuring the UI continues to log each
  finished item. Peak-fit retries and failures now include the originating
  spectrum ID/label and `source_path`/`source_file` metadata in the log entries
  so you can trace troublesome fits back to the precise input.
- **Peak detection opt-in.** Peak detection runs only when the recipe includes a
  `features.peaks` configuration with `enabled: true`. If the recipe omits the
  peak section or explicitly disables it, the pipeline skips peak detection and
  leaves the spectrum metadata unchanged, preventing defaults from silently
  inserting peak annotations.
- **Peak overlays follow spectrum axes.** When peak markers are available in
  `meta["features"]["peaks"]`, the preview plot reads the spectrum axis key
  from `meta["axis_key"]` (or `meta["axis_type"]`) to decide which coordinate
  to plot. Wavelength spectra continue to use the `wavelength` peak field,
  while wavenumber spectra use `wavenumber`, keeping peak markers aligned with
  whichever axis the spectrum declares.
- **Workbook exports for auditing.** Exported workbooks bundle processed
  spectra, metadata, QC flags, and an audit log so you can review the exact
  sequence of operations and verify whether any QC thresholds were exceeded.
- **Replicate-level scoring.** When averaging replicates, the pipeline can apply
  MAD or Cook’s-distance screening to discard obvious outliers before
  aggregation, keeping the representative trace faithful to the cluster of raw
  measurements.

Together, the retained channels, UI overlays, QC metrics and flags, exported
logs, and replicate outlier scores provide both qualitative and quantitative
checks that processed spectra remain valid stand-ins for their raw measurements.

## Troubleshooting settings and configuration
SpectroProtz persists user preferences with
[`QSettings("SpectroLab", "SpectroApp")`](spectro_app/app_context.py). On most
systems the store resolves to:

- **Linux**: `~/.config/SpectroLab/SpectroApp.conf`
- **macOS**: `~/Library/Preferences/com.SpectroLab.SpectroApp.plist`
- **Windows**: Registry under `HKEY_CURRENT_USER\\Software\\SpectroLab\\SpectroApp`

If widgets appear with stale values or the application fails to remember recent
files, clear or rename the corresponding store before relaunching the GUI. After
resetting settings, review the default configuration in
[`spectro_app/config/defaults.yaml`](spectro_app/config/defaults.yaml) and the
workflow reference in [`docs/workflow_overview.md`](docs/workflow_overview.md)
for guidance on expected processing behaviour.

Preview failures now surface the raw exception text (including the exception
type and message) in the dialog so you can see the precise parsing issue (for
example missing axes or unexpected dataset shapes). The application also
captures a timestamped entry with the file path(s), exception details, and full
stack trace, writing it to `preview_errors.log` inside the log folder (open via
**Tools → Open Log Folder**) and echoing the same diagnostics in the on-screen
logger panel for rapid triage and sharing with support.

## Utilities
Analysts can generate a searchable index of the JCAMP-DX headers bundled in
`IR_referenceDatabase/` with the `index_ir_metadata.py` helper. The script
walks every `.jdx` file, normalises header names to snake case, merges
continuation lines, and emits a deterministic record set that is convenient for
diffing or downstream processing.

The higher-level peak indexer, [`scripts/jdxIndexBuilder.py`](scripts/jdxIndexBuilder.py),
parses JCAMP-DX spectra, normalises axes, and fits peaks after converting any
transmittance (%T or fractional) signals into absorbance with `A = -log10(T)`.
This ensures downstream preprocessing and peak finding work with a consistent
representation regardless of how the original instrument exported the Y axis.
At startup the CLI prompts to export per-step XLSX workbooks unless you pass
`--no-prompt-export` (or explicitly set `--export-step-plots`).
Detailed CLI usage and output descriptions are available in
[`docs/ftir-indexer.md`](docs/ftir-indexer.md).
Results are persisted in `peaks.duckdb` under `index_dir` with four tables:

- `spectra` – one row per JCAMP source file including normalised metadata.
- `peaks` – fitted peak parameters for every detected spectrum peak.
- `file_consensus` – per-file consensus clusters with representative centre,
  width, and supporting peak count.
- `global_consensus` – global consensus clusters aggregating across all files.

When SpectroApp starts, it validates the last-used FTIR index database (if
configured) against the expected DuckDB schema so missing tables or columns are
flagged early in the logger panel before lookup workflows proceed.
Lookup query projections are kept in sync with the index builder schema to
ensure downstream peak searches request the correct columns.

The `spectra` table includes both foundational bookkeeping fields and promoted
JCAMP headers for convenient filtering and projection:

- Core columns: `file_id` (TEXT), `path` (TEXT), `n_points` (INT),
  `n_spectra` (INT), and `meta_json` (TEXT).
- Promoted columns:

  | Column          | JCAMP header        | Type   |
  | -------------- | ------------------- | ------ |
  | `title`        | `TITLE`             | TEXT   |
  | `data_type`    | `DATA TYPE`         | TEXT   |
  | `jcamp_ver`    | `JCAMP-DX`          | TEXT   |
  | `npoints_hdr`  | `NPOINTS`           | DOUBLE |
  | `x_units_raw`  | `XUNITS`            | TEXT   |
  | `y_units_raw`  | `YUNITS`            | TEXT   |
  | `x_factor`     | `XFACTOR`           | DOUBLE |
  | `y_factor`     | `YFACTOR`           | DOUBLE |
  | `deltax_hdr`   | `DELTAX`            | DOUBLE |
  | `firstx`       | `FIRSTX`            | DOUBLE |
  | `lastx`        | `LASTX`             | DOUBLE |
  | `firsty`       | `FIRSTY`            | DOUBLE |
  | `maxx`         | `MAXX`              | DOUBLE |
  | `minx`         | `MINX`              | DOUBLE |
  | `maxy`         | `MAXY`              | DOUBLE |
  | `miny`         | `MINY`              | DOUBLE |
  | `resolution`   | `RESOLUTION`        | DOUBLE |
  | `state`        | `STATE`             | TEXT   |
  | `class`        | `CLASS`             | TEXT   |
  | `origin`       | `ORIGIN`            | TEXT   |
  | `owner`        | `OWNER`             | TEXT   |
  | `date`         | `DATE`              | TEXT   |
  | `names`        | `NAMES`             | TEXT   |
  | `cas`          | `CAS REGISTRY NO`   | TEXT   |
  | `molform`      | `MOLFORM`           | TEXT   |
  | `nist_source`  | `$NIST SOURCE`      | TEXT   |

List the available metadata fields that can be projected into the report:

```bash
python scripts/index_ir_metadata.py --list-fields
```

Write the complete index as JSON (default format) with paths relative to the
database root:

```bash
python scripts/index_ir_metadata.py > ir_metadata.json
```

Emit a compact CSV that only keeps a handful of high-value metadata fields:

```bash
python scripts/index_ir_metadata.py --format csv --fields title cas_registry_no molform owner > ir_metadata.csv
```

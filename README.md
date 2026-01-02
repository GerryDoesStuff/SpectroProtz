# SpectroProtz

## Overview
SpectroProtz is a PyQt-based desktop application for managing spectroscopy
batches, visualising spectra, and exporting analysis artefacts. The GUI bundles
processing helpers, batch orchestration, and recipe management behind a single
entry point so analysts can focus on domain workflows instead of wiring
individual scripts. Internally, the launcher hands control to
[`spectro_app/main.py`](spectro_app/main.py), which wires up the Qt event loop
and the shared application context.

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

### Optional OPUS readers
SpectroProtz loads OPUS spectra through external readers to ensure the axis and
intensity arrays are populated with SpectroChemPy’s default dataset accessors.
Install one (or both) of the optional readers below to enable OPUS imports:

- **spectrochempy** (adds `spectrochempy.read_opus`)
- **brukeropusreader** (adds `brukeropusreader.read_file`)

Install either package (or both) in the same environment as the app:

```bash
pip install spectrochempy
```

```bash
pip install brukeropusreader
```

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

## Verifying processed spectra
SpectroProtz keeps a complete audit trail for every spectrum so you can confirm
that processed traces remain representative of their raw counterparts:

- **Stage-by-stage channels.** Each processing step writes its result into a
  dedicated metadata channel (for example `raw`, `blanked`,
  `baseline_corrected`, `joined`, `despiked`, `smoothed`). The original signal
  stays available alongside every intermediate, enabling numerical and visual
  comparisons at any point in the pipeline.
- **Interactive stage toggles.** The preview dock in the UI lets you toggle the
  visibility of stored stages so you can overlay the processed curve on top of
  the raw trace or inspect any intermediate discrepancies.
- **Quantitative QC metrics.** The QC engine computes diagnostics such as noise
  levels, join offsets, spike counts, smoothing guards, drift, and per-stage
  roughness deltas that compare processed spectra against their retained
  channels. These metrics roll up into QC flags that highlight when processing
  diverges too far from the source data.
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

Preview failures now surface the raw exception text in the dialog so you can
see the precise parsing issue (for example missing axes or unexpected dataset
shapes). The same stack trace is appended to `preview_errors.log` inside the
log folder (open via **Help → Open Log Folder**) for deeper debugging and
sharing with support.

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

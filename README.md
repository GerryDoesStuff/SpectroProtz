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

A typical workflow uses a virtual environment so the above packages can be
installed without affecting system Python.

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

This command bootstraps the Qt event loop and shows the main window immediately,
allowing users to load recipes, configure processing options, and monitor batch
runs.

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

## Utilities
Analysts can generate a searchable index of the JCAMP-DX headers bundled in
`IR_referenceDatabase/` with the `index_ir_metadata.py` helper. The script
walks every `.jdx` file, normalises header names to snake case, merges
continuation lines, and emits a deterministic record set that is convenient for
diffing or downstream processing.

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

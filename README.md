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
The batch export flow writes Excel workbooks that include processed spectra
tables, metadata, and QC summaries; QC rows are normalized into plain
serializable dictionaries (including ISO-formatted timestamps and primitive
lists) before the workbook writer flattens them, ensuring the export pipeline
never emits raw dataclass instances into Excel output.
Wide-layout exports derive per-spectrum column labels from available metadata
in a consistent fallback order (for example, explicit display labels, sample
IDs, channel names, or source filenames) and sanitize the chosen identifier to
remove Excel-invalid characters while preserving the human-readable name.

## FTIR indexer pipeline
The FTIR indexer script (`scripts/jdxIndexBuilder.py`) is a standalone CLI that
ingests JCAMP-DX files, normalises spectra for peak detection, and writes peak
fits plus header metadata into DuckDB/Parquet outputs that back the reference
lookup tools. It does not depend on the main GUI package; it runs from the repo
root as long as the scientific Python stack and DuckDB are installed. The
pipeline handles single-spectrum and multi-spectrum JCAMPs (multiple Y columns
in `XYDATA`) and emits per-spectrum rows keyed by file, spectrum index, and
source metadata.

Preprocessing includes Savitzky-Golay smoothing, optional baseline correction,
and a normalisation pass that scales the working spectrum by its maximum
absolute value to stabilise fitting. Peak model fits always run on the
normalised data, which is then upsampled with Akima interpolation (8×) so peak
detection and fitting operate on a denser, smoothly interpolated grid while
retaining the original points in order. The indexer separately computes raw
absorbance metrics for storage after converting the original JCAMP Y-units into
absorbance (for %T or fractional transmittance, `A = -log10(T)`). The persisted
`peaks.amplitude` is sampled directly from that raw absorbance series at the
fitted center, and `peaks.area` is integrated from the raw absorbance window
without applying normalisation or baseline correction. Fit-space
amplitude/area values are kept alongside each fit (under `fit_amplitude` and
`fit_area`) for QA without mixing normalised units into the stored peak
columns.

Index outputs include:
- Per-spectrum peak fit tables (center, width, model parameters, fit metrics).
- Header metadata promoted to typed columns plus a key/value header store.
- Consensus peak tables (per-file and global) for quick lookup prioritisation.
These outputs can be queried directly with DuckDB or loaded into downstream
tools (including the GUI) without requiring the main application runtime.
### Timeout enforcement
Peak fit and file-level timeout controls take different paths depending on
platform support. On POSIX platforms with `SIGALRM`, the indexer arms an alarm
inside the current process so the guard remains low-overhead while still
raising the same timeout error types that callers expect. When `SIGALRM` is not
available (notably Windows), the indexer and shared peak-detection helpers
start the guarded operation in a separate process, enforce the wall-clock limit
via `join(timeout)`, and terminate the child if it exceeds the budget. This
multiprocessing fallback preserves the same timeout semantics but adds process
startup overhead, so throughput can dip when many short fits are executed.

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

When peak detection is enabled, the preview dock renders peak picks as vertical
marker lines (one per peak) that match the spectrum color, and the **Peaks**
checkbox lets you show or hide those markers alongside the stage traces. Peaks
are grouped under the same sample label used in the legend for the spectral
lines, but the peak markers themselves are not annotated with individual text
labels inside the plot; use the sample label and hover/cursor readout to
interpret positions. Peak filters are driven by the minimum absorbance/intensity
threshold and the maximum peak candidate count in the recipe editor, so you can
drop low-intensity features and cap the number of peaks retained per spectrum.
Axis units come from the spectrum metadata (`axis_key` + `axis_unit`) and drive
both the plot labels and which peak coordinate is read (wavelength/nm for UV-Vis
or wavenumber/cm⁻¹ for FTIR/Raman). No numeric unit conversion is applied during
peak rendering; when the pipeline sees a non-nm axis it remaps window keys (for
example `min_cm`/`max_cm`) into the nm-based config fields while preserving the
original axis metadata so downstream stages still report the original units.

## FTIR reference lookup parsing
The FTIR indexer’s DuckDB outputs can be queried with search-bar style input
using the helper in `spectro_app/engine/ftir_lookup.py`. The parser accepts
numeric peak positions with optional tolerances (for example `1720±5` or
`1720 +/- 5`), defaulting to ±2 cm⁻¹ when no explicit tolerance is supplied.
Metadata filters are written as `key:value` or `key=value` tokens. Filters map
to promoted FTIR metadata columns such as `title`, `origin`, `cas`, `names`,
`molform`, `state`, and `nist_source`, with aliases like `name` or `formula`
automatically normalized. Manual search input can list multiple peaks in one
query, and the parser returns structured criteria with error messages for
invalid tokens before mapping them to parameterized DuckDB SQL: one query for
matching reference spectra and another for the peak rows that satisfy the same
metadata filters plus any requested peak ranges. Empty or whitespace-only
searches are handled defensively by returning queries that produce no rows,
ensuring the lookup flow never issues an unbounded query by default.
Tokens that are not peaks or explicit filters are routed either to the
`molform`, `cas`, or `title` metadata filters. Unkeyed formula tokens that match
standard chemical formulas (element symbols with optional counts, such as
`C6H6`, `NaCl`, or `CH3COOH`) are treated as `molform` filters so a query like
`C6H6 1720±5` is interpreted as `molform:C6H6` plus the peak constraint. Any
unkeyed CAS Registry Number tokens (for example `64-17-5`) are treated as
`cas` filters so a query like `64-17-5 1720±5` is interpreted as `cas:64-17-5`
plus the peak constraint. Any remaining unkeyed tokens are treated as free-text
`title` filters, so a query like `acetone 1720±5` is interpreted as
`title:acetone` plus the peak constraint.

## FTIR reference lookup window
SpectroProtz now includes a dedicated **FTIR Reference Lookup** window
accessible under **Tools → FTIR Reference Lookup...**. The dialog lets you
pick from recent `peaks.duckdb` indexes (or browse for a new one), run manual
searches from a query bar, and view matching reference spectra in the left-hand
results sidebar. The sidebar now includes a compact **Auto-search from preview
peaks** panel alongside the results list, keeping the preview-driven search
controls and status together with the lookup matches. The metadata panel now
sits in the former auto-search area above the plots so selected-reference
details remain visible even before a match is chosen, and the lookup window
instantiates the metadata widget alongside the top controls so the same panel
is shared for both layout placement and live metadata updates. The index selector is
backed by app-level settings, so recent index paths and the last-used index are
shared across sessions and other FTIR tools (such as the indexer) that update
the same settings store. Selecting an index immediately validates the DuckDB
schema so the status line confirms when the index is ready to query. The index
selector combines a history list with a dedicated **Browse** button: picking a
recent entry sets it as the active index immediately, and browsing for a new
file adds it to the history list while making it the active source for
subsequent searches and plotting.
Manual searches can be run with the **Search** button or by pausing after
typing (debounced), which keeps the UI responsive on large indexes. Reference
spectra are ranked by the total matched peaks weighted by peak intensity/area,
using the sum of `abs(amplitude) + abs(area)` across all matched peak rows from
the `peaks` table (i.e. `score = Σ(|A| + |area|)` for matched peaks). Higher
scores indicate both more matches and stronger peak intensity/area. Results are
sorted by this score (descending), then by matched peak count, so manual and
preview-driven searches rank references consistently.
The reference preview plot mirrors the selected spectra context when available:
if the selection screen supplies spectra, the preview x-axis range is taken
from the selected spectrum with the widest wavenumber span (ties break by using
the spectrum with more data points). When no selected spectra are available,
the preview range falls back to the full span of the previewed reference
spectrum, and only uses peak centers as a last resort if no trace is loaded.
The list supports multi-selection, selection-driven previewing of the bottom
reference plot, and paginates through large result sets with a results cap so
the sidebar remains responsive when a search returns many references. The top
comparison plot
stays focused on the selected spectra plus any references that you move into
the right-hand plotting sidebar, so you can compare overlays without losing
the single-reference preview context below. The layout is a three-column split
with the left results sidebar, the dual-plot center column (comparison on top,
single-reference preview below), and the right selected-references sidebar
flanking the plots, while the reference metadata panel lives above the plots in
the former auto-search area. Use the add/remove buttons anchored below each
sidebar list, the left sidebar context menu, or a double-click on a match to
move reference spectra into the right-hand plotting sidebar, which preserves
the order shown in the results list for predictable comparisons.
The single-reference preview plot uses fixed scaling: it always clamps
normalized intensity to 0–1, sets the X range to the current spectrum (or
peak bounds when no spectrum trace is plotted), and disables zooming/panning so
the preview stays consistent as you browse entries.
The comparison plot is likewise non-interactive (no zoom/pan and no ViewBox
context menu) to keep the overlays fixed, and it disables auto-range so axes
remain locked to the selected spectra span (or the reference peak range) with
intensity clamped to the 0–1 normalized scale.
Both lookup plots show a live cursor readout beneath the plots; hovering inside
either ViewBox updates the wavenumber and normalized intensity readouts (with
explicit cm⁻¹ and a.u. units), and the values clear when the pointer leaves the
plot area.
While the lookup window is open, selecting FTIR files in the main file queue
automatically loads the chosen spectra and overlays them in the lookup
comparison plot as “selected spectra.” Multi-selecting queue entries plots
each loaded spectrum at once, and switching the queue selection to non-FTIR
data clears the live overlay so the reference traces remain uncluttered. There
are no extra buttons required for this behavior—simply highlight queue entries
to refresh the lookup plot. Use the **Auto-plot selected queue spectrum**
checkbox in the lookup window’s control area to disable or re-enable this live
queue overlay; the toggle is enabled by default so the lookup plot stays in
sync with queue selections until you opt out.
When preview peak detection is active, the lookup window captures the identified
peak centers and converts them into auto-search criteria using a fixed ±2 cm⁻¹
tolerance. Auto-search only runs when there is no active manual query, so manual
results remain in the sidebar unless you explicitly click **Apply preview peaks**
to re-run the lookup from the captured peaks.
Lookup results can be exported to CSV from the **Export CSV** menu, either for
all matches or only the right-hand selected references, and the export includes
every available column from the `peaks` table plus all non-`file_id` columns
from the `spectra` table (core IDs, peak metrics, polarity, file paths, and all
promoted metadata fields).
The lookup dialog validates the selected DuckDB file before it queries, and it
reports clear errors when the database is missing, unreadable, cannot be
opened, or lacks required tables/columns. The results list switches to a
guidance empty state when the index is invalid or when searches return no
matches, so you can tell the query ran successfully but yielded no references.

The right sidebar keeps a stable list of selected references with remove controls
via the transfer buttons, double-click removal, or the right-click menu. Every
add, remove, or selection change batches re-plotting of the top comparison chart
in the lookup window, which draws normalized reference spectrum traces and overlays vertical peak
sticks for each selected reference. The chart also overlays any currently
selected preview spectra as normalized traces, scaling each selected spectrum
independently into a 0–1 range while rendering each reference’s peak sticks
with their own 0–1 normalization based on reference amplitudes. If the preview
is in single-spectrum view with multiple spectra added to the window, every
visible selection is included in the lookup plot so you can compare several
live traces at once. Selecting a
subset of references in the right sidebar filters the plot to just those
entries, while leaving the list unselected shows the full selected-reference
set. Search actions do not change the plot until you explicitly add or remove
references, keeping comparisons stable while you browse new matches.
Right-clicking any reference in the left or right sidebar previews that
reference’s normalized spectrum and peak sticks in the bottom plot immediately,
showing a single reference trace with its normalized peak markers and metadata
fields. The preview plot can also draw the full reference spectrum stored in the
`meta_json` payload as a smooth overlay line; the indexer stores a single
`XYDATA` entry in that payload using absorbance-converted values (saved as
simple `(X,Y)` lines) derived from the first spectrum in the JCAMP file so the
preview overlay matches the same spectrum used for lookup previews. The
**Show reference spectrum overlay** checkbox lets you toggle that line on or off
so you can focus on peak sticks when needed.
Selection changes in either sidebar update the same preview, and the
metadata panel beside the chart summarizes the selected reference’s title,
molform (formula), CAS, origin, owner, date, data type, state, class, and related
header fields for quick inspection without leaving the lookup dialog.
The preview refresh stays scoped to the bottom plot when you right-click a
reference, so browsing entries does not force the top comparison overlay to
redraw.
Reference spectrum traces and peak sticks are cached within the lookup window
session so repeated plot updates avoid redundant database or file reads when
revisiting the same references.
If the lookup database contains malformed spectra or peak rows (for example
missing file IDs or non-numeric peak data), the lookup skips those rows and
reports a warning in the dialog status area instead of halting the preview.
When you are ready to compare candidates against live preview data, use
**Send overlay to main preview** to push the currently plotted reference
spectra into the main preview plot as an overlay. The preview dock includes an
**Identified** checkbox alongside the stage toggles to show or hide those
reference overlays on demand, keeping the main plot uncluttered while you work
through match candidates. The Identified toggle is persisted with the rest of
the UI session state, so the visibility preference is restored the next time
you reopen SpectroProtz.

Preview peak identification can also feed the lookup workflow without
overwriting manual search state. From the preview plot context menu, choose
**Use selected peaks for reference lookup** to capture the currently selected
spectrum’s FTIR peaks. The lookup window stores those peaks as a pending
auto-search request, shows how many were captured, and converts them into a
lookup query using a fixed ±2 cm⁻¹ tolerance. When the manual search box is
empty, the lookup window automatically runs that preview-driven query to
populate the left sidebar with matching references. If you are already running
a manual search, the preview matches wait in the background until you either
clear the manual query or explicitly click **Apply preview peaks** to override
the current results. Manual search inputs remain unchanged unless you trigger
the preview-driven search yourself.

### Right-click actions, selection shortcuts, and keyboard accelerators
SpectroProtz surfaces several context menus and keyboard accelerators to keep
navigation quick. The behaviors below reflect the current UI actions and
tooltips.

**Right-click behaviors**
- **File queue entries**: right-click a queued file to inspect data, preview the
  spectrum, set the role (auto-detect, sample, blank, standard), edit the blank
  ID (only when a blank is selected), save an FTIR solvent reference, remove
  selected items from the queue, or reveal the file in the system file manager.
- **FTIR lookup results (left sidebar)**: right-click a match to preview its
  reference spectrum and add it to the plotting sidebar.
- **FTIR lookup selected references (right sidebar)**: right-click a selected
  reference to preview it and remove it from the plot list.
- **Preview plot**: right-click to copy the data at the cursor, export the
  current figure, reset the view, send selected FTIR peaks to the lookup
  window, hide the selected spectrum, show hidden spectra, or zoom in/out on
  the cursor position.

**Selection shortcuts**
- The FTIR lookup results and selected-reference lists both use extended
  selection, so you can use **Ctrl/Cmd-click** to add/remove items and
  **Shift-click** to select a contiguous range before adding to plot or
  removing.
- The preview plot selection follows the currently selected spectrum label;
  hiding and lookup actions apply to the active selection.

**Keyboard accelerators**
- **File menu**: Open (**Ctrl+O**), Save Recipe (**Ctrl+S**), Exit (**Ctrl+Q**).
- **Edit menu**: Undo (**Ctrl+Z**), Redo (**Ctrl+Y**), Cut (**Ctrl+X**), Copy
  (**Ctrl+C**), Paste (**Ctrl+V**), Delete (**Del**), Select All (**Ctrl+A**).
- **Process menu**: Run (**F5**), Cancel (**Esc**).
- There are currently no dedicated keyboard accelerators for moving lookup
  results to the right sidebar, removing selected lookup references, or
  toggling the **Identified** overlay; use the buttons, context menus, or
  checkbox instead.

**Tooltips**
- FTIR lookup controls surface tooltips for the index picker, search input,
  tolerance spinner, paging buttons, add/remove actions, and **Send plot to
  main preview**.
- The **Identified** checkbox in the preview dock shows a tooltip describing
  the overlay behavior and retains its most recent state between sessions.
- Context menu items supply tooltips when an action is unavailable (for example
  export disabled if PyQtGraph exporters are missing, or lookup disabled when
  the selection lacks FTIR peaks).

### FTIR lookup search syntax
Manual searches accept peak tokens, optional tolerances, and metadata filters in
a single query. Peak tokens are interpreted as wavenumbers (cm⁻¹). You can
combine multiple peaks and filters to narrow results.

- **Single peak:** `1720` uses the default ±2 cm⁻¹ tolerance.
- **Multiple peaks:** `1720±2 1600±3` requires both peak ranges to match.
- **Tolerance syntax:** `±`, `+/-`, or `+-` are accepted (for example
  `1720 +/- 5`). Optional unit suffixes include `cm-1`, `cm^-1`, or `cm⁻¹`.
- **Metadata filters:** `key:value` or `key=value`, e.g.
  `name:acetone origin:"NIST" 1720±5`. Use quotes for values with spaces.
- **CAS tokens:** bare CAS Registry Numbers like `64-17-5` map to the `cas`
  filter, so `64-17-5 1720` is equivalent to `cas:64-17-5 1720`.
- **Free-text tokens:** bare words (no `:` or `=`) that are not formulas or
  CAS numbers are applied as `title` filters, so `acetone 1720` is equivalent
  to `title:acetone 1720`.

Lookup results are re-ranked by the weighted match score (sum of
`abs(amplitude) + abs(area)` across matched peaks), and each sidebar row shows
both **Matched peaks** and **Score** values so you can compare candidates that
share the same peak coverage but differ in peak intensity or area.

### Manual peak search by position
Manual peak search accepts one or more peak positions as plain numbers or
tokens with explicit tolerances. Each peak entry is interpreted as a wavenumber
and matched against the index using a default ±2 cm⁻¹ tolerance when no
explicit tolerance is supplied. Enter multiple peak positions in a single query
to require matches for every listed peak before results are returned.

Example inputs:
- `1720` (single peak, default ±2 cm⁻¹)
- `1720 1600 1510` (multiple peaks with default tolerance)
- `1720±5 1600±3` (multiple peaks with explicit tolerances)
- `name:acetone 1720 1365` (metadata filter plus peaks)

### Exporting lookup matches and sharing data
FTIR lookup results can be exported directly from the **FTIR Reference Lookup**
window by clicking **Export CSV** and choosing whether to export **All matches**
or **Selected references** (the highlighted items in the right sidebar). The
scope controls whether the CSV includes every match currently listed in the
left sidebar or only the references you have moved into the right-hand selected
list. If your lookup result list is capped, the export includes only the
matches currently loaded in the sidebar.

**CSV export fields** include every column from the `peaks` table plus every
column from the `spectra` table except `file_id` (along with core bookkeeping
fields like `path`, `n_points`, `n_spectra`, and `meta_json`). This means peak
rows always carry the full peak metrics (`file_id`, `spectrum_id`, `peak_id`,
`polarity`, `center`, `fwhm`, `amplitude`, `area`, `r2`, etc.) alongside all
available reference metadata (`title`, `origin`, `owner`, `cas`, `names`,
`molform`, `state`, `data_type`, `nist_source`, and any other promoted JCAMP
headers). Use the CSV output to pivot peak matches in spreadsheets or share the
complete reference metadata with collaborators.

Lookup data is still persisted in the `peaks.duckdb` database produced by the
indexer (stored in the chosen `index_dir`), so you can also export matched
peaks and reference metadata directly from DuckDB as CSV or JSON. If you
provide a relative output path in a `COPY` statement, the export file is
written under your current working directory; absolute paths land wherever you
specify.

**Matched peaks exports (CSV/JSON)** use the `peaks` table with the schema
fields `file_id`, `spectrum_id`, `peak_id`, `polarity`, `center`, `fwhm`,
`amplitude`, `area`, and `r2`. To export only peaks that satisfy your lookup
filters, join `peaks` to `spectra` and constrain both metadata and peak
positions. For example, to capture matches near 1720 cm⁻¹ for acetone-like
titles:

```bash
duckdb /path/to/index/peaks.duckdb -c "COPY (
  SELECT s.file_id,
         s.title,
         s.molform,
         s.cas,
         p.spectrum_id,
         p.peak_id,
         p.polarity,
         p.center,
         p.fwhm,
         p.amplitude,
         p.area,
         p.r2
  FROM peaks p
  JOIN spectra s ON s.file_id = p.file_id
  WHERE s.title ILIKE '%acetone%'
    AND p.center BETWEEN 1715 AND 1725
) TO 'matched_peaks.csv' (HEADER, DELIMITER ',');"
```

Swap the `COPY` format to JSON when needed:

```bash
duckdb /path/to/index/peaks.duckdb -c "COPY (
  SELECT s.file_id, s.title, s.molform, s.cas, p.center, p.fwhm, p.amplitude
  FROM peaks p
  JOIN spectra s ON s.file_id = p.file_id
  WHERE p.center BETWEEN 1715 AND 1725
) TO 'matched_peaks.json' (FORMAT JSON);"
```

**Reference metadata exports (CSV/JSON)** come from the `spectra` table, which
includes core columns (`file_id`, `path`, `n_points`, `n_spectra`, `meta_json`)
and promoted JCAMP headers (for example `title`, `origin`, `owner`, `cas`,
`names`, `molform`, `state`, `data_type`, and `nist_source`). Use the same
`COPY` pattern to export just the fields you need:

```bash
duckdb /path/to/index/peaks.duckdb -c "COPY (
  SELECT file_id, title, molform, cas, origin, owner, path
  FROM spectra
) TO 'reference_metadata.csv' (HEADER, DELIMITER ',');"
```

For quick sharing from the UI, right-click the preview plot and choose
**Copy data at cursor** to send the currently highlighted point to the
clipboard. The copied text is a tab-separated row containing the X value,
Y value, spectrum label, and processing stage label, so you can paste it
directly into chat messages, spreadsheets, or issue reports.

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
  references without leaving the preview. The toggle state persists with the
  saved UI session, so you can keep your preferred overlay visibility between
  launches.
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
  FTIR exports follow the same path: when a recipe provides an export
  `path`/`workbook`, the batch run writes a workbook and logs the resolved
  target; when no export path is provided, the run still completes and records
  that the workbook was skipped.
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
The PDF digitizer in [`scripts/pdfSpectraDigitizerFTIR.py`](scripts/pdfSpectraDigitizerFTIR.py)
extracts FTIR spectra from figure scans and writes a master XLSX plus optional
per-spectrum XLSX workbooks for each `entry_id`. It is fully standalone and does
not depend on `jdxIndexBuilder.py`, so the digitizer can emit JCAMP-DX payloads
without importing the FTIR indexer. The same per-spectrum loop can emit JCAMP-DX
outputs: enable `--per-spectrum-jdx` (or leave it on) to write one JDX file
into the per-spectrum JDX directory (default: `<out>_spectra_jdx`, override with
`--spectrum-jdx-outdir`), and use `--out-jdx` to collect all spectra into a
single multi-spectrum JDX when their X axes are consistent. JDX output uses
the digitized curve data, converts transmittance to absorbance, and normalizes
absorbance to a 0–1 range before writing `##XYDATA=(X++(Y..Y))` payloads. The
digitizer populates JCAMP headers for `JCAMP-DX`, `DATA TYPE`, `XUNITS`,
`YUNITS`, `NPOINTS`, `FIRSTX`, and `DELTAX`, plus optional metadata fields when
they are detected from OCR or source metadata. Optional JCAMP headers are
emitted only when non-empty values are available, with the following metadata
mapping:

Spectra are extracted even when plot labels are missing. The digitizer first
tries OCR labels on the same page, then optionally OCRs the next page (or more)
to recover label text; configure this behavior with
`--label-lookahead-pages` (default: 1). When lookahead labels are applied, the
Entries sheet records the number of pages ahead that supplied the labels in
`label_page_offset` (0 means the label was on the same page as the spectra).
If no labels are detected after lookahead, the digitizer scans the full plot
interior, assigns placeholder labels like `Unknown_01`, and records
`label_missing = true` in the Entries sheet so unlabeled spectra are easy to
review later; the Entries sheet keeps `label_missing` as a boolean flag so
downstream reviewers can filter and audit unlabeled extractions quickly. If labels are found but the label bands yield no curve components, the
digitizer falls back to scanning the full plot interior for curve components,
then assigns the closest component(s) to each label based on vertical
proximity between the label’s y position and the component bounding-box
center. The logs note when this fallback path is used so reviewers can trace
why a full scan was required.
Each processed page emits a summary log line with counts for labels found,
curve components detected, spectra digitized, and spectra rejected so you can
spot problematic pages quickly. After the run, the CLI prints totals for
spectra written/rejected plus a breakdown of rejection reasons (for example
missing labels, missing curve components, calibration failures, digitization
failures, or near-flat traces), and it also logs a QC summary with the number
of QC-flagged spectra and their reasons. QC failures never block output:
spectra are still written, the Entries sheet marks them with `qc_flag = true`,
the QC sheet records the failed stage plus summary notes, and the per-spectrum
JDX `##NOTES` field appends `QC: ...` for rapid review. The run also writes
`<out>_qc_failures.csv` and `<out>_qc_failures.json` with `entry_id`, page,
label, and QC reasons so reviewers can filter failures without opening Excel.
Alongside the QC report, the digitizer writes per-spectrum diagnostic traces
to `<out>_diagnostics.json` and `<out>_diagnostics.csv`. Each entry includes
spectrum identifiers plus a `debug_trace` (JSON) or `debug_trace_json` (CSV)
array of stage objects. Example entry:

```json
{
  "entry_id": "p0003_img00_spec01",
  "page_number_1based": 3,
  "image_index": 0,
  "spectrum_index": 1,
  "label_ocr": "Quartz",
  "status": "failed",
  "failed_stage": "x_ticks",
  "failed_reason": "ocr_count=1",
  "debug_trace": [
    {"stage": "image_detected", "status": "ok"},
    {"stage": "axes_detected", "status": "ok"},
    {"stage": "x_ticks_found", "status": "fail", "reason": "ocr_count=1"},
    {"stage": "y_ticks_found", "status": "ok"},
    {"stage": "labels_found", "status": "ok"},
    {"stage": "components_found", "status": "ok"},
    {"stage": "digitize_success", "status": "ok"},
    {"stage": "qc_status", "status": "fail", "reason": "x calibration failed; using pixel x"}
  ]
}
```

Use the diagnostic trace to interpret failures: `image_detected` covers PDF
image extraction, `axes_detected` validates the plot crop, `x_ticks_found` and
`y_ticks_found` reflect OCR tick counts (two or more values are required for
calibration), `labels_found` reports whether OCR labels were found or a
placeholder label was used, `components_found` confirms curve components were
detected, `digitize_success` reports whether any curve points survived
digitization, and `qc_status` captures whether QC flagged the spectrum. The
CLI logs also emit a one-line summary per spectrum in the form
`entry_id=... status=failed stage=x_ticks reason=ocr_count=1` so batch runs can
be triaged without opening the reports.
QC failures no longer block output; QC details are reported separately. When
digitization yields too few points, no curve components, or no curve points
after filtering and post-processing, the digitizer still writes per-spectrum
XLSX and JDX artifacts with metadata and QC notes, but the CurvePoints sheet is
empty and the per-spectrum JDX contains only headers (no `XYDATA` block) along
with a `digitize_failed_no_curve` or `too_few_points` QC reason. These empty
outputs are intended for audit trails and should not be fed into the FTIR
indexer, which requires finite `XYDATA` values for indexing.

To stabilize digitized traces that include repeated or jittered x positions,
the digitizer concatenates all curve components for a spectrum and then
de-jitters the raw points by collapsing them into small wavenumber bins. The
bin tolerance is tied to the data spacing by using the larger of 0.2 cm⁻¹ or
half the median positive Δx. Within each bin it keeps one representative
transmittance (median by default, or the top-most pixel when
`--bin-representative top` is selected) and retains the median wavenumber as
the bin center before proceeding to gap imputation and output. Before binning,
an optional axis/border filter can discard curve points within a few pixels of
the plot axes or borders (`--axis-filter-px`, `--border-filter-px`) to avoid
digitizing axis lines. When the trace shows large oscillations, a small-window
rolling median filter (configured by `--rolling-median-window`) suppresses
short spikes without flattening real bands. The per-spectrum entry metadata
records whether points were collapsed (`collapsed_points`), the bin width
(`bin_width_cm1`), and which filters ran (`axis_filter_applied`,
`rolling_median_applied`) so reviewers can trace the post-processing step, and
the run logs capture per-spectrum QC stats for axis/border removals and bin
collapses.
During component selection, the digitizer evaluates the raw pixel y-range for
each connected curve component before any y normalization. Components whose
pixel y-span is below a small threshold (currently 8 pixels) are tagged as
axis/border candidates and excluded from digitization so axis lines or plot
borders do not yield flat normalized traces. When multiple components remain in
the plot interior, the selection step prefers components with the largest
pixel y-range to emphasize full-height spectra over stray fragments. If every
component is filtered out as an axis/border candidate, the digitizer records a
diagnostic failure reason and leaves the curve output empty instead of emitting
a flat-line spectrum.

When extracting axis calibration, the digitizer detects X-axis breaks by
digitizing the full curve in pixel space, sorting the x positions, and finding
the largest spacing that is significantly larger than the median (typically
10–20×). That pixel-space midpoint is then used to split the OCR’d ticks into
left/right calibrations so the right-hand segment is mapped to the correct
portion of the axis. The digitizer imputes the missing region across the
discontinuity so downstream curve outputs remain continuous.
If too few X-axis ticks are detected to build a calibration, the digitizer can
fall back to a user-supplied range by passing `--x-range` (for example
`4000-400`) or setting the `FTIR_PDF_X_RANGE` environment variable; the leftmost
plot pixel is mapped to the first value and the rightmost plot pixel to the
second value so digitization can still proceed. Entries record
`x_mode = fallback` along with the `x_range_used` string when this fallback is
applied, while runs without a supplied range revert to pixel-space x values
(`x_mode = pixel`) and retain QC notes about the missing ticks.

If no Y-axis tick labels are detected, the digitizer concatenates all curve
components for a spectrum, computes a single global 0–1 normalization from the
full pixel range, and applies that normalization once before any imputation
steps. The QC sheet records whether Y normalization was `global` or
`calibrated` so downstream review can distinguish the fallback path.

- `TITLE`: spectrum name parsed from text above the graph (preferably the line
  above the molecular formula); falls back to OCR label or entry ID when
  missing.
- `NOTES`: the remainder of a `Description:` line found below the graph (or
  subsequent wrapped line text if the label is on its own line).
- `ORIGIN`: source title / origin metadata.
- `OWNER`: source author / owner metadata.
- `DATE`: run timestamp or metadata date.
- `NAMES`: parsed mineral name or OCR label.
- `CAS REGISTRY NO`: parsed CAS registry value.
- `MOLFORM`: parsed chemical formula.

To build those fields, the digitizer extracts text in a rectangular band around
each graph using PyMuPDF `page.get_text(...)` clips, then runs targeted passes
on the text above the image (to capture the spectrum name near the molecular
formula) and the text below the image (to capture `Description:` lines). The
full text block around the graph is still used for peak list parsing, mineral
name detection, and formula extraction.

The FTIR indexer expects at least the following JCAMP headers to be present so
it can compute a uniform axis and parse spectra reliably: `JCAMP-DX`,
`DATA TYPE`, `XUNITS`, `YUNITS`, `NPOINTS`, `FIRSTX`, `DELTAX`, and an
`XYDATA=(X++(Y..Y))` block with finite values. Keep those headers intact when
preparing JDX files for indexing, and skip any empty per-spectrum JDX outputs
that were written for QC visibility.

Analysts can generate a searchable index of the JCAMP-DX headers bundled in
`IR_referenceDatabase/` with the `index_ir_metadata.py` helper. The script
walks every `.jdx` file, normalises header names to snake case, merges
continuation lines, and emits a deterministic record set that is convenient for
diffing or downstream processing. Molecular formula values (`MOLFORM`) are
normalized by removing whitespace so downstream overlays and exports receive a
compact formula string.

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

Core purpose

Single mission: reproducible batch processing of spectra with scientifically safe defaults and an auditable trail.

Supported techniques (modular)

UV-Vis (Helios Gamma & generic CSV/Excel).

FTIR (ASCII/CSV; optional bridges for proprietary).

Raman (ASCII/CSV from common vendors).

PEES (placeholders + method-specific steps).

Plugin system to add future modules (fluorescence, PL, XAS, etc.).

Input & ingestion

Drag-and-drop files/folders; file queue with badges (technique, A/%T, blank/sample).

Manifest CSV for sample↔blank mapping, groupings (dose/site), replicate IDs (optional enrichment; disable via `UvVisPlugin(enable_manifest=False)` when manifests should be ignored).
  - Columns (case-insensitive): `file` (optional; basename or path), `channel`/`column` (optional; raw trace label),
    `sample_id` (final label), `blank_id`, `replicate`/`replicate_id`, `group`/`group_id`, `role`, `notes`.
  - Rows without `file` act as defaults; matching prefers file+channel, then file+sample, then global fallbacks.
  - Channel match enables renaming (e.g., column `A1` → manifest `sample_id` "Dose 1"); blanks can be tagged via `role=blank`.
  - Parsed fields are merged into spectrum metadata before `Spectrum` creation so downstream blanking/replicate logic sees them.

Tolerant parsers (decimal comma/dot, delimiters, header variants).

Structured metadata extraction:

Instrument/model, mode, date/time, operator, pathlength, spectral resolution, slit/bandwidth, scan speed/averages, laser λ/power (Raman), accumulations (FTIR), known blank file.

Ordinate coercion to canonical domain (e.g., %T→Absorbance once; Raman counts→intensity with units noted).

Wavelength/wavenumber checks: monotonicity, duplicate collapse, missing-value handling.

Instrument-edge trimming rules per module.

Optional common grid interpolation (configurable start/stop/step), with original grid preserved.

Processing pipeline (shared skeleton)

Domain coercion (e.g., %T→A; orientation of FTIR/Raman axis).

Blank subtraction (where applicable) with timestamp gating and pathlength checks.

Baseline correction: AsLS, rubberband (convex hull), SNIP (optional); anchor windows for zeroing.

Discontinuity/segment join correction (UV-Vis; others as applicable): overlap detection via change-point; apply bounded piecewise offsets; record offsets.

Despiking: targeted median/Hampel for spikes/cosmic rays; never across joins.

Optional Savitzky–Golay smoothing (window/poly validated) with join guards.

Replicate handling: average in corrected domain; outlier detection (MAD/Cook’s D); retain per-replicate traces.

Technique-specific processing

UV-Vis

%T extremes → saturation flags; A>3 flag.

Quiet-window noise RSD (e.g., 850–900 nm).

Join windows defaults (e.g., 340–360 nm) + data-driven detection.

FTIR

Atmospheric compensation (H₂O/CO₂ regions).

Baseline strategies suited to broad features.

Axis orientation (high→low cm⁻¹) normalization and verification.

Raman

Cosmic-ray removal (narrow spikes) with minimal line distortion.

Laser wavelength/power metadata; optional vector normalization (shape-only).

Wavenumber calibration hooks.

PEES

Module-defined pipeline slots; metadata validation; export schema parity.

Quality control (QC)

Saturation/underflow flags; negative intensity after corrections flagged (not deleted).

Noise metrics: rolling RSD / quiet-window RSD (technique-specific windows).

Spike counts; roughness index pre/post smoothing; smoothing aggressiveness guard.

Join-offset magnitude & overlap delta before/after.

Time-drift checks (baseline drift vs timestamp).

Validation of recipe assumptions; per-spectrum QC summary.

Batch QC dashboards (histograms, scatter, time series).

Feature extraction & analysis

Peak picking with prominence/min-distance; λ/ν max, peak height, FWHM via local quadratic fit.

Band ratios (e.g., UV-Vis A260/A280), integrals over windows, isosbestic checks (UV-Vis).

Derivative spectra (SG 1st/2nd) as additional channels.

Kinetics hooks: if multiple timestamps, compute ΔA(t) at targets, rates from linear windows.

Calibration (optional): Beer–Lambert linear/weighted fits; residual diagnostics, LOD/LOQ (3σ/10σ), back-calc unknowns with uncertainty.

Visualization

Central interactive preview: Raw, Blanked, Baseline-corrected, Joined, Despiked, Smoothed (toggle overlays).

Crosshair readouts (x, y, and Δ between states); zoom/pan; reset view.

Small-multiples per sample; batch overlays with decimation for performance.

Join-overlap before/after plots; residuals plots for calibrations; QC summary charts.

Export plots (PNG/SVG) with sanitized filenames; optional PDF summary.

Export & provenance

Single workbook per batch (Excel):

Processed_Spectra (wide or tidy; common grid + per-step channels).

Metadata (typed fields per spectrum).

QC_Flags + metrics.

Calibration (if used): standards table, fit stats, residuals.

Audit_Log: ordered operations + parameters + library versions + input hashes.
  - Runtime entries include tokens such as ``Runtime library spectro-app==<version>`` so batches capture the
    executing build.
  - Each spectrum with a ``meta.source_file`` records ``Input <sample_id> source_hash=sha256:<digest>`` for
    provenance of the originating files.

Sidecar recipe JSON/YAML snapshot stored with outputs.

Optional PDF report (key plots + QC summaries).

Formula-injection safe writing (apostrophe prefix where needed).

UI/UX

Main window with remembered geometry; dockable panes: File Queue, Recipe Editor, Preview, QC, Logger.

Menus: File / Edit / View / Process / Tools / Help; standard shortcuts (Ctrl/Cmd).

Toolbars for quick actions: Open, Save Recipe, Run, Stop, Export Workbook/Plots.

Context menus in File Queue (inspect header, preview, locate on disk) and plots (copy data at cursor, export).

Preset recipes per module (e.g., “UV-Vis default”, “Raman narrow-line”).

Inline validation with tooltips; non-blocking warnings/toasts.

Dark/light themes; high-DPI aware.

Configuration & persistence

Recipe model with schema/version; save/load YAML/JSON; preset manager.

App settings (QSettings): theme, recent files, last export dir, window/dock state.

Unsaved recipe and running-job close prompts.

Performance & threading

QThreadPool / multiprocessing for CPU-heavy steps (AsLS, change-point) to dodge the GIL.

Streamed parsers; chunked Excel writer; lazy/decimated plotting.

Progress bars (overall & per stage), ETA, cancellable jobs; no zombie processes on cancel.

Security & privacy baseline

No execution of user content; sanitize paths; secure temp handling.

HTTPS for optional update checks; host pinning (configurable).

No telemetry by default; optional crash reports are opt-in and redacted.

Accessibility

Accessible names/labels; logical tab order; keyboard access to all actions.

High-contrast compatibility; respects OS DPI scaling and font size.

Cross-platform

Windows/macOS/Linux; platform-correct menus and shortcuts (Ctrl vs Cmd, macOS global menu bar).

Native file dialogs; Unicode paths robustly supported.

Updates & maintenance

About dialog: name, version, license, publisher.

“Check for Updates” opens release notes (manual update flow).

Semantic versioning; recipe schema migration notes.

Logging & diagnostics

Rotating app logs with levels; per-run Audit Log (human-readable).

“Open log folder” action; error dialogs include actionable recovery tips.

Redaction rules for sensitive info in logs and exports.

Testing & CI hooks

Unit tests: parsers, baselines, join offsets, despiking, peak fits, recipe validation.

Golden files / regression tests for pipelines.

Headless smoke test (launch app, load tiny batch, run stub).

Lint/type checks (ruff/mypy optional).

Sample datasets + expected outputs for dev fixtures.

Extensibility (plugin framework)

Clear Plugin API (detect, load, validate, preprocess, analyze, export).

Module-specific io_*.py, pipeline.py, and presets.yaml.

Shared helpers (io_common, qc, plots, excel_writer) to avoid copy-paste.

Discovery/registration mechanism (entry points or import at startup).

Error handling & recovery

Recipe pre-run validation; blocks unsafe orderings (e.g., smoothing before blank).

File-level isolation: bad files quarantined; batch continues; end-of-run summary of failures.

Clear, non-cryptic error messages with likely fixes.

Auto-guard rails: SG window vs polyorder, join-guard enforcement, numeric range checks.

Developer conveniences

Command-line entry point for batch runs (headless) sharing the same engine.

Dev “verbose audit” mode per run.

Structured timing metrics per stage for profiling.
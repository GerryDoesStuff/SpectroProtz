0) Prep: fixtures & logging

 Enable DEBUG logging and rotate logs per run; verify timestamps and timezone.

 Prepare tiny known-good fixtures per module (≥1 sample, 1 blank, 3 replicates).

 Prepare synthetic fixtures to provoke edge cases:

UV-Vis: join offset spike at 345–355 nm; %T near 0%/100%; random single-point spikes.

FTIR: sloped baseline; strong H₂O/CO₂ bands; reversed axis files (low→high cm⁻¹).

Raman: narrow cosmic rays; slight baseline; known peak positions; wrong laser λ in metadata.

PEES: minimal CSV with missing metadata field.

 Note expected outputs: λ/ν ranges, number of rows, peak positions, flags.

1) App startup & UI skeleton

 Launch time < ~2–3 s; no console tracebacks.

 Window geometry/state persist across restarts (including multi-monitor).

 Menus & shortcuts: Ctrl/Cmd variants work; macOS menu bar behaves.

 Tab order: can traverse all controls without mouse.

 High-DPI scaling: text readable; icons not blurry; dark/light theme legible.

2) Settings & persistence

 QSettings writes/reads: recent files, last export dir, theme, dock visibility.

 “Unsaved recipe” prompts on close; cancel truly cancels.

3) File queue & input handling

 Drag-drop folders and files; recursion finds spectra; unsupported files rejected with clear reason.

 Context menu: “Inspect header”, “Preview raw”, “Locate on disk” work.

 Manifest CSV loads (when opt-in flag enabled); bad mappings warn but don’t crash; missing blank → fallback to nearest-in-time with warning.

4) Plugin detection & switching

 Mixed folder (UV-Vis + Raman) triggers correct module selection per file or prompts.

 Switching module after queue loaded → forces re-validation; stale state cleared.

 Wrong module → helpful validation error (e.g., “Looks like FTIR; axis is cm⁻¹”).

5) Parsers & metadata (per module)

UV-Vis

 .dsp / CSV / Excel parse under both decimal dot/comma; delimiter variance OK.

 Ordinate coercion: %T→A once; raw A preserved; no normalization in T ever.

 Metadata fields populated: datetime, operator, pathlength, slit/bandwidth, scan speed, averaging, blank id.

FTIR

 ASCII/CSV import; orientation normalized to high→low cm⁻¹ (or consistently documented).

 Resolution/accumulations parsed; bad headers handled gracefully.

Raman

 Laser λ/power parsed; axis is cm⁻¹; counts/intensity semantics consistent.

PEES

 Minimal CSV accepted; unknown fields logged, not fatal.

6) Wavelength/axis hygiene

 Monotonicity enforced; duplicates collapsed correctly (mean or first-seen documented).

 Instrument edge trimming applied; common grid interpolation works; original grid retained for provenance.

 Out-of-range values clipped/flagged (not silently dropped).

7) Pipeline ordering (guard rails)

 Attempt to run “smoothing before blanking” → blocked with actionable message.

 Attempt to normalize T after reading A → blocked; message references best practice.

 SG window ≤ 2*polyorder → validation error.

8) Blank & baseline

 Blank mapping by manifest when the opt-in feature is enabled; otherwise nearest-in-time within window; pathlength mismatch warns.

 After blank subtraction: negative A allowed but flagged; values preserved.

 Baseline AsLS: adjusting λ and p has predictable effect; extremes don’t invert peaks.

 Rubberband: convex hull uses sensible anchor density; not over-tight.

 Anchor zero windows: applying/removing zeroing shows expected level shift only.

9) Discontinuity / join correction (UV-Vis)

 Change-point in overlap detects synthetic offset at ~345–355 nm.

 Piecewise offset applied to noisier side only; magnitude recorded in audit/QC.

 Out-of-bounds offset (too large) → flagged and capped or rejected with message.

 Before/after overlap delta plot shows flattening post-correction.

10) Despiking & smoothing

 Single-point spikes removed without damaging peaks (verify on synthetic).

 Despike never crosses join boundaries (check window straddling).

 SG smoothing off by default; when enabled, peaks’ FWHM only slightly changes; roughness index decreases modestly.

 Raman cosmic-ray removal eliminates narrow spikes; true spectral lines remain.

11) Replicates & averaging

 Replicate average performed after corrections in absorbance/intensity domain.

 Outlier replicates detected (MAD/Cook’s D) and can be excluded/included; audit notes decision.

 Averaged spectrum equals manual computation within tolerance.

12) Features & calibration

 Peak picking returns correct λ/ν_max within tolerance; FWHM from quadratic fit stable under small noise.

 Band ratios/integrals over user windows match manual integrals.

 Derivative spectra present as additional channels, not replacements.

 Calibration (if used): linear/weighted fits produce expected slope/intercept; residual plot sane; LOD/LOQ computed as 3σ/10σ of blank/low-conc residuals.

13) QC metrics & flags

 Saturation: UV-Vis A>3 or %T <1% / >99% flagged; FTIR/Raman over/underflow flagged analogously.

 Noise RSD (quiet windows) computed and logged.

 Roughness index drop (pre→post smoothing) within configured bounds; excessive smoothing flagged.

 Join-offset magnitude distribution appears in QC panel; outliers highlighted.

 Time-drift visual (baseline vs timestamp) behaves on synthetic drift set.

14) Preview & plotting

 Raw vs processed overlays toggle; Δ readout (difference between steps) displays correct values.

 Zoom/pan smooth; reset view restores full range; export PNG/SVG works and filenames sanitized.

 Large overlays (hundreds of spectra) auto-decimate; UI stays responsive.

15) Export & provenance

 Workbook contains all sheets: Processed_Spectra, Metadata, QC_Flags, Calibration (if used), Audit_Log.

 Tidy and wide variants correct; numeric dtypes preserved (no strings of numbers).

 Formula injection prevented (cells starting with = are prefixed).

 Sidecar recipe YAML/JSON snapshot written; audit includes library versions and input file hashes.

16) Performance & cancellation

 Progress bars advance per stage; ETA roughly stable.

 Cancel mid-AsLS and mid-export → stops quickly (< a few seconds), leaves partial outputs coherent or cleans them.

 No zombie threads/processes after cancel or exit; memory returns to baseline.

 1k-spectrum batch finishes without OOM; CPU scales across cores.

17) Accessibility & internationalization

 Screen reader labels on core controls; focus indicators visible.

 High-contrast mode readable; respects OS font scaling.

 Locale: decimal comma inputs parse; exports still use a stable numeric format (documented).

18) Cross-platform

 Windows/macOS/Linux: file dialogs native; paths with spaces/Unicode fine.

 Shortcuts: Ctrl on Win/Linux, Cmd on macOS; Exit/Quit behavior correct; app quits cleanly.

19) Updates, About, Help

 About shows version from one source of truth.

 “Check for updates” opens release notes; offline/HTTP errors handled gracefully.

 Help viewer loads; tooltips present for non-obvious controls.

20) Errors & recovery

 Bad file produces per-file error; batch continues; summary dialog lists failures with reasons and suggested fixes.

 Recipe schema mismatch (older/newer) → migration prompt or clear error, no crash.

 Network/update failures don’t block processing.

21) Tests & CI sanity

 Parsers/baseline/join/despike/peaks unit tests pass; golden files stable.

 Headless smoke test (load tiny UV-Vis set, run, export) passes in CI on all OS targets.
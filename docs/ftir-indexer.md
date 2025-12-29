# FTIR Indexer (jdxIndexBuilder)

## Purpose
The FTIR indexer (`scripts/jdxIndexBuilder.py`) builds a searchable peak index from
JCAMP-DX FTIR spectra. It scans a directory tree for `.jdx`/`.dx` files, normalises
spectral axes (X → cm⁻¹ and Y → absorbance), detects and fits peaks, and stores
results plus headers in a DuckDB database for downstream lookup and analysis.

## Inputs and outputs
**Inputs**
- **Data directory**: A directory containing JCAMP-DX files (`.jdx` / `.dx`). The
  script scans recursively.
- **Index directory**: A target directory where output artifacts are written.
- **Existing index database (for skip detection)**: If `peaks.duckdb` exists in
  the same directory as `scripts/jdxIndexBuilder.py`, the indexer loads it to
  detect files that were already indexed and skips them automatically.

**Outputs**
- **DuckDB database**: `peaks.duckdb` in `index_dir`, containing tables for spectra,
  peaks, and consensus clusters (see [Output artifacts](#output-artifacts)).
- **Step-plot exports** (optional): Per-spectrum Excel workbooks (`.xlsx`) written
  to `index_dir/debug_plots` or `--export-step-plots-dir` when export is enabled.
- **Parquet exports** (optional): The indexer does not automatically write Parquet
  files, but the DuckDB tables are designed to be exported directly to Parquet if
  needed (for example via `COPY peaks TO 'peaks.parquet' (FORMAT PARQUET);`).

## Workflow overview
1. **Discover files**: Recursively find `.jdx`/`.dx` files under `data_dir`.
2. **Parse headers**: Load JCAMP-DX headers, promote standard keys to columns,
   and validate that the file is an FTIR spectrum.
3. **Normalise axes**: Convert X to cm⁻¹ and Y to absorbance (`A = -log10(T)`
   for transmittance inputs).
4. **Preprocess**: Apply smoothing/baseline correction as configured.
5. **Detect peaks**: Find candidate peaks (optionally including negative peaks).
6. **Fit peaks**: Fit Gaussian/Lorentzian/Voigt/Spline models and filter on fit
   quality (R²).
7. **Persist results**: Write spectra, peaks, and consensus clusters to DuckDB.
8. **Optional exports**: If enabled, export step-by-step spectra to `.xlsx`.
9. **Operational defaults**: CLI logs are timestamped and indexing runs with up
   to four worker processes by default.

## Behavior changes
- **Timestamped CLI output**: Log messages include timestamps for easier
  run-to-run comparison.
- **Parallel indexing**: The indexer uses up to four worker processes to
  accelerate ingestion when multiple cores are available.

## Skip logic
The indexer checks for `peaks.duckdb` in the same directory as
`scripts/jdxIndexBuilder.py`. If present, it loads existing file identifiers
(for example, SHA-1 hashes stored in the database) and automatically skips any
input file that has already been indexed. This happens without prompting, which
keeps repeat runs fast and prevents reprocessing known files.

## CLI arguments
The following arguments are defined in `main()` of `scripts/jdxIndexBuilder.py`.
Defaults are shown in parentheses.

### Positional arguments
- `data_dir` (default: script directory)
  - Directory containing JCAMP-DX files.
- `index_dir` (default: script directory)
  - Output directory for the DuckDB database and optional exports.

### Peak detection and preprocessing
- `--prominence` (0.003)
- `--min-distance` (0.8) — minimum peak separation (cm⁻¹, converted to points using median X spacing).
- `--min-distance-mode` (`fixed`) — `fixed` uses `--min-distance` as-is; `adaptive` clamps it to a fraction
  of the estimated median FWHM so narrow peaks remain eligible.
- `--min-distance-fwhm-fraction` (0.5) — fraction of the median FWHM used to clamp `--min-distance` in adaptive mode.
- `--noise-sigma-multiplier` (1.5)
- `--noise-window-cm` (400.0) — window size for local noise estimation (set to 0 to use a global noise floor).
- `--min-prominence-by-region` (None) — comma-separated ranges with prominence floors
  (e.g. `400-1500:0.03,1500-1800:0.02`).
- `--peak-width-min` (0.0)
- `--peak-width-max` (0.0)
- `--cwt-enabled` (false)
- `--cwt-width-min` (0.0)
- `--cwt-width-max` (0.0)
- `--cwt-width-step` (0.0)
- `--cwt-cluster-tolerance` (8.0)
- `--plateau-min-points` (3)
- `--plateau-prominence-factor` (1.0)
- `--merge-tolerance` (8.0)
- `--sg-win` (7)
- `--sg-poly` (3)
- `--sg-window-cm` (0.0)
- `--als-lam` (5e4)
- `--als-p` (0.01)
- `--baseline-method` (`airpls`) — choices: `arpls`, `asls`, `airpls`.
- `--baseline-niter` (20)
- `--baseline-piecewise` (false)
- `--baseline-ranges` (`4000-2500,2500-1800,1800-900,900-450`)

### Peak fitting
- `--model` (`Gaussian`) — choices: `Gaussian`, `Lorentzian`, `Voigt`, `Spline`.
- `--min-r2` (0.75)
- `--fit-window-pts` (70)
- `--fit-maxfev` (10000)
- `--fit-timeout-sec` (0.0)

### Consensus clustering
- `--file-min-samples` (2)
- `--file-eps-factor` (0.5)
- `--file-eps-min` (2.0)
- `--global-min-samples` (2)
- `--global-eps-abs` (4.0)

### Behavior toggles and safety guards
- `--detect-negative-peaks` (false) — also detect negative peaks by inverting spectra.
- `--strict` (false) — raise exceptions instead of skipping unsupported files.
- `--file-timeout-seconds` (0.0) — abort indexing a single file after this many seconds.

### Step-plot export controls
- `--export-step-plots` (false) — export per-step spectra workbooks.
- `--export-step-plots-dir` (None) — defaults to `{index_dir}/debug_plots`.
- `--prompt-export` (false) — deprecated; forces an interactive prompt.
- `--no-prompt-export` (false) — disables the interactive prompt.
- `--plot-max-points` (0) — downsample exported plots (0 disables downsampling).

## Common usage examples
**Tuning guidance**
- The prominence threshold is computed as the max of `--prominence`, the local noise estimate
  (`--noise-sigma-multiplier` × noise sigma), and any `--min-prominence-by-region` floor.
- When `--min-distance-mode adaptive` is enabled, the minimum separation is clamped to
  `--min-distance-fwhm-fraction × median(FWHM)` so narrower peaks can still be detected.
- Increase `--noise-sigma-multiplier` or `--prominence` if weak artifacts still appear.
- Use `--min-prominence-by-region` to raise floors in the 400–1500 cm⁻¹ fingerprint region
  without suppressing stronger peaks elsewhere.
- Set `--peak-width-min 0 --peak-width-max 0` to disable width filtering if narrow peaks are missed.

**Basic run**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index
```

**Strict mode (fail fast on unsupported spectra)**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index --strict
```

**Peak tuning**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --prominence 0.01 --min-distance 5 --sg-win 11 --sg-poly 3 \
  --als-lam 1e5 --als-p 0.01 --min-r2 0.9
```

**Suppress low-value artifacts in the fingerprint region**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --min-prominence-by-region "400-1500:0.03" --noise-window-cm 400 \
  --noise-sigma-multiplier 3
```

**Recommended presets**

Sensitive (favor weak or narrow peaks):
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --prominence 0.005 --noise-sigma-multiplier 1.5 --min-distance 2 \
  --peak-width-min 0 --peak-width-max 0
```

Conservative (favor cleaner, well-separated peaks):
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --prominence 0.02 --noise-sigma-multiplier 3 --min-distance 5 \
  --peak-width-min 4 --peak-width-max 40
```

**Export per-step plots (Excel workbooks)**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --export-step-plots --no-prompt-export
```

**Export step plots to a custom output directory**
```bash
python scripts/jdxIndexBuilder.py /path/to/jdx /path/to/index \
  --export-step-plots --export-step-plots-dir /tmp/ftir_debug_plots \
  --no-prompt-export
```

## Dependencies
The indexer expects a Python environment with the following packages installed:
- `duckdb`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `pyarrow`

Python 3.9+ is recommended to match the project’s supported runtime.

## Output artifacts
- **`peaks.duckdb`**: Primary database stored under `index_dir`.
  - `spectra`: one row per JCAMP source file including promoted header fields.
  - `peaks`: fitted peak parameters for each detected peak.
  - `file_consensus`: per-file consensus peak clusters.
  - `global_consensus`: cross-file consensus clusters.
  - `ingest_errors`: errors encountered while ingesting a file. Each row includes
    the file path, the error message, and a timestamp. If a file succeeds on a
    subsequent run, its error row is removed.
- **Parquet files (optional)**: Not emitted by default, but you can export any
  DuckDB table to Parquet using DuckDB’s `COPY` command (for example,
  `COPY peaks TO 'peaks.parquet' (FORMAT PARQUET);`).
- **`failed_files.txt`**: When the GUI-driven FTIR indexer is used, it writes a
  tab-delimited list of files that failed to process, along with the reason, to
  `index_dir/failed_files.txt`. Use this to triage file-level errors.
- **Step plot exports**: When enabled, `.xlsx` files are written under
  `index_dir/debug_plots` (or `--export-step-plots-dir`) containing per-step
  spectra for each spectrum ID.
- **Skip detection source**: The script also reads `peaks.duckdb` from the
  script directory (alongside `scripts/jdxIndexBuilder.py`) to detect and skip
  already-indexed files on subsequent runs.

## Interpreting results
- **Consensus tables** (`file_consensus`, `global_consensus`) are designed for
  prioritising peaks during lookup: higher-support clusters represent more
  frequently observed features.
- **`ingest_errors` / `failed_files.txt`** highlight files that were skipped or
  failed. Re-run the indexer after fixing input issues to clear entries.

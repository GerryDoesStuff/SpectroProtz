Peak look up
Separate window that can be accessed through the Tools menu.
Goal: connects to a .duckdb file with spectral information produced by the index builder script and uses it to look for peak data, let's the user plot the peaks in a window that also has spectra selected in the preview window menu plotted.
The lookup window has a top central window for plotting spectra and peaks; a left sidebar that is used for ennumerating reference spectra found via search either by the user submitting a serach request in a search bar positioned above the sidebar, or by letting the app search for reference spectra with relavant peaks automatically if peak identification was performed on the selected spectra; a right sidebar that stores the selected reference spectra the peaks of which are plotted in the top central window along with the selected spectra (the peaks are plotted as vertical lines with heights representing the peak intensity value in the reference; both the selected spectra and the peak heights need to be separately normalized to fit a range from 0 to 1); a bottom central window that plots a single reference spectrum selected/deselected in the right or left sidebar via right mouse click (both the reference spectrum and it's peaks need to be normalized and plotted), the window should also have basic information about the reference spectrum (name, chemical formula, etc.; stuff that is the metadata). The window should also have arrow buttons that move a reference spectrum from the left "lookup sidebar" to the right "plotting" sidebar, and a button thatwould remove it from the right sidebar. The plotting should be performed as soon as a change was made (sans search and left sidebar update). There should be a button that plots the resulting plot onto the main preview window plot; the main plot should have a tick box that would enable or disable the plotting of an "identified" plot.

## Supported search formats
The search bar accepts peak positions, optional tolerances, and metadata filters in a single query. Peaks are interpreted as wavenumbers (cm⁻¹) and can be combined with metadata filters to narrow results.

### Single peak
Enter a single peak as a number to find spectra that contain a peak in the default tolerance band.

- **Example:** `1720`
- **Behavior:** matches spectra that have a peak center between **1715–1725 cm⁻¹** (default tolerance ±5 cm⁻¹).

### Multiple peaks
Space-separate multiple peak tokens to require all of them. Each peak token can carry its own tolerance.

- **Example:** `1720±2 1600±3`
- **Behavior:** matches spectra that contain a peak between **1718–1722 cm⁻¹** and another peak between **1597–1603 cm⁻¹**.

### Tolerance syntax
Tolerance can be written with the plus/minus sign or text equivalents.

- **Accepted formats:** `1720±5`, `1720 +/- 5`, `1720 +- 5`
- **Units:** `cm-1`, `cm^-1`, or `cm⁻¹` suffixes are optional (for example `1720±5 cm-1`).

### Metadata filters
Filters use `key:value` or `key=value` tokens and can be mixed with peak tokens. Use quotes for values containing spaces.

- **Example:** `name:acetone origin:"NIST" 1720±5`
- **Behavior:** limits matches to spectra whose metadata contains “acetone” in the title/name, “NIST” in the origin field, and includes a peak in the tolerance band around 1720 cm⁻¹.

### Default tolerance
If you omit the tolerance in a peak token, the lookup uses **±5 cm⁻¹**.

### Match ranking
Lookup results are not re-ordered by a weighted score; they appear in the order returned by the database. The results list shows **Matched peaks** counts for each entry so you can manually prioritize candidates (more matched peaks generally indicates a closer fit when multiple peaks are provided).

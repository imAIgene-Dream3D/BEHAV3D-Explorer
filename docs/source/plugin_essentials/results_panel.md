# 📄 Results panel

The **📄 Results** panel sits at the bottom of the **🧪 Feature Extraction**, **🧹 Filtering** and **📊 Analysis** tabs. It is the same panel in all three: a tree of every result file under the `analysis/` folder of your output directory, so you can open any plot or table without browsing folders.

## The tree

Files are grouped by the step that produced them, then by population (the folder name under `analysis/`), then by sub-folder, mirroring the folders on disk:

| Top level | Contains |
|---|---|
| **Filtering** | Quality-control plots from track filtering (filter counts, contact and dead-dye distributions) and track-count tables. |
| **Feature Extraction** | Feature tables (raw, filtered, summarised). |
| **Analysis → Population Dynamics** | Death dynamics, interaction, invasiveness and multi-organoid comparisons. |
| **Analysis → Single Cell** | Behavioural-state and trajectory classification outputs. |

The large `per_sample` folders of Death Dynamics start collapsed. Right-click a folder for **Expand all** / **Collapse all**.

Selecting a file shows a one-line description of what it contains under the tree (hover over it for the same text). Non-viewable files (tables, HTML, Zarr) are shown in grey.

The panel refreshes itself after a run, when metadata is loaded, and when you switch to its tab if anything under `analysis/` changed. If no output directory is set it shows **⚠ No output directory configured**; if nothing has been produced yet, **(No outputs found yet)**.

## Controls

| Control | Default | What it does |
|---|---|---|
| **▴ Show** / **▾ Hide** | collapsed | Expands or collapses the panel. Collapsed, only the header row is visible. |
| **DPI:** | `150` | Resolution (72–300) used to render a PDF when you open it in napari. Higher is sharper but slower. |
| **Show non-viewable** | on | Also list tables (CSV), HTML, JSON and Zarr files. Untick to list only PDFs and images. |
| **👁 Open in napari** | — | Opens the selected PDF or image (also: double-click it). Only for PDFs and images. |
| **Open externally** | — | Opens the selected file with your system's default program (PDF reader, Excel, browser…). Double-clicking a non-viewable file does the same. |
| **Reveal in folder** | — | Shows the file in your file explorer. |

The same three actions are available from the right-click menu of a file.

## Viewing results in napari

**👁 Open in napari** opens a second napari window titled **BEHAV3D – Results**, so your main viewer is not touched. A multi-page PDF becomes one image with a **page** slider. Every file you open is added as a new layer, so you can compare plots side by side by toggling layers; close the window when done. A small bar at the bottom of that window offers **Open externally** and **Reveal in folder** for the file shown last.

## See also

- [Output Directory & File Layout](output_layout.md): where the result files live on disk.
- [Running Steps](running_steps.md): how steps run and what happens with existing results.

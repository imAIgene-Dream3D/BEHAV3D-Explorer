# 📥 Import existing tracks

The **Import tracking** option brings tracking results produced outside BEHAV3D EXPLORER into the standard output layout, so Feature Extraction and the rest of the pipeline can use them.

```{important}
"Import" is **not** a tracking method: it *validates, converts and copies* an existing tracked label image into the expected zarr + CSV outputs. It works like the [segmentation importer](../segmentation/import.md).
```

## The Import page

Pick **Import tracking** in the **Method:** dropdown of a cell-type sub-tab. The **Run … Tracking** button is disabled for this method; everything is done with the buttons on the page:

- **➕ Add a new sample or cell type…** — jumps to the Data Preparation tab with the Metadata Builder already in edit mode, for samples or cell types not yet in the metadata.
- One row per sample (**📁 *sample name***) with an editable path field (pre-filled from the metadata column `<prefix>_<cell_type>_tracks_image_path`, e.g. `im_tcell_tracks_image_path`), a **Browse…** button (it first asks whether you want to pick a **File** or a **Zarr / Folder**), and a status. A new path is only written to `metadata.csv` once that row has been processed. Changing a path that was already set asks **Replace existing path?** first.

| Status shown | Meaning | Button offered |
|---|---|---|
| *No source path set* | The field is empty | — (type or browse a path) |
| ⚠️ *File not found* | The path does not exist (hover to see the path tried) | — (fix the path) |
| — | A `.tif` / `.tiff` source, not yet imported | **🔄 Convert TIFF → zarr** |
| — | A `.zarr` source (or folder), not yet imported | **📄 Import zarr** |
| ✅ *Already processed* | The tracked zarr and tracks CSV already exist at the standard paths | **🔄 Re-process** (overwrites) |
| ⚠️ *Unsupported format (…)* | Neither TIFF nor zarr | — |

**⚡ Process All Samples** appears when at least one sample still needs importing; it processes every row with a valid source path. Whenever a tracked zarr already exists at the destination, you are asked **Overwrite existing tracking?** before it is replaced.

## What happens when you process a row

1. **TIFF source:** a **Confirm Axis Order** dialog shows the image shape and the detected axis order; confirm or pick the right one (e.g. `TZYX` = time, z, y, x). The TIFF is then converted to zarr.
   **Zarr source:** a valid zarr (see below) is copied to the standard location; any other zarr is rewritten in the right layout.
2. The labels are converted to the standard 16-bit label format (32-bit when IDs exceed 65 535); negative labels are given new, unused IDs, the same in every frame. If the source already *is* the destination file and has to be converted, the original is kept as a backup copy next to it.
3. The tracks CSV is computed from the labels (centroid of every TrackID at every timepoint, in pixels and physical units — see [CSV column reference](index.md#csv-column-reference)).
4. The metadata columns `<prefix>_<cell_type>_tracks_image_path` and `<prefix>_<cell_type>_tracks_csv_path` are set to the new files and `metadata.csv` is saved.

Outputs: `<output_dir>/images/<sample>/<sample>_<cell_type>_tracked.zarr` and `<output_dir>/trackdata/<sample>/<cell_type>/<sample>_<cell_type>_tracks.csv`.

## What "valid" means

The source must be a **4-D `(T, Z, Y, X)` integer label image** in which the pixel value is the **TrackID** — the same physical cell keeps the same ID at every timepoint. A zarr is used as-is when it is a **single array** (not a group of arrays) stored **one timepoint per chunk**; otherwise it is rewritten.

```{important}
The importer only checks the file **structure** — it cannot verify that the IDs are really consistent over time. Always inspect a converted sample in the Visualization tab before running Feature Extraction.
```

```{warning}
Batch tracking and the processing queue do not import. When **Run Batch Tracking (All Cell Types)** or the queue asks about existing data, keep the imported cell types (**Skip Existing**, or **Skip (keep existing)** in **Customize…**) so they are not re-tracked.
```

## Importing from common sources

### Imaris

Export the tracked labels as a TIFF time series (same integer ID per track across timepoints), point the sample's path to it, and click **🔄 Convert TIFF → zarr**.

### btrack or another tracker run outside BEHAV3D EXPLORER

Point the path to the labelled zarr or TIFF it wrote; the importer validates, converts and copies it.

### A previous BEHAV3D EXPLORER run in a different output directory

Point the path to the previous run's `*_tracked.zarr`; it is copied into the new output directory.

## Tips

- Always **inspect the converted result** in the Visualization tab before running Feature Extraction.
- **Absolute paths are the most robust choice.** Relative paths work only if they resolve from the working directory or from the folder containing the metadata CSV.

## See also

- [Data Preparation](../../data_preparation.md) — where the `*_tracks_image_path` metadata columns live.
- [Output Directory & File Layout](../../plugin_essentials/output_layout.md) — the standard paths.
- [Import existing segmentation](../segmentation/import.md) — the analogous workflow at the segmentation stage.

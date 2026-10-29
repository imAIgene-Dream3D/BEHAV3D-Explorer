# 📥 Import existing segmentation

The **Import segmentation** option in the Segmentation tab lets you bring in segmentation masks produced elsewhere — typically by Imaris, Ilastik or an external script — and integrate them into the BEHAV3D EXPLORER output layout so downstream steps (tracking, feature extraction) can use them. Dead masks made elsewhere can be imported too.

```{important}
"Import" is **not** a method of *creating* segmentations; it *checks and converts* existing ones. The actual segmentation must have been done elsewhere.
```

## How it works

The page lists every sample (under a `📁 <sample>` header) with one row per population and, when the metadata declares a dead channel, one **dead mask:** row. Each row has a path field, a **Browse…** button and a status (label or action button) on the right.

1. **Enter the file**: click **Browse…** (choose **File** for a TIFF, **Zarr / Folder** for a `.zarr` folder) or type / paste the path. Paths already in metadata.csv are pre-filled. Replacing a path that is already set asks for confirmation (**Replace existing path?**).
2. **Read the status** and click the action button it offers (table below), or click **⚡ Convert / Save all for *sample*** (one sample) or **⚡ Convert / Save All Samples** (everything). These buttons only appear when something needs doing; otherwise the page shows **✨ All available segmentation files are in the correct format.**
3. When done, a **Metadata Updated** message confirms that metadata.csv was saved.

```{note}
Paths you type or browse are only written to metadata.csv when the row's Convert / Save action runs (or a **⚡ Convert / Save …** button).
```

| Status shown | Meaning | Action |
|---|---|---|
| **No segmentation available** (**No dead mask available**) | The path field is empty. | Browse to or type the file, if you have one. |
| **⚠️ File not found** | The path does not exist (hover to see where it was looked for). | Correct the path. |
| **✅ Ready for tracking** (**✅ Ready** for the dead mask) | A valid `.zarr` whose size matches the raw image, already saved in metadata.csv. | Nothing to do. |
| **💾 Save path to metadata** *(button)* | A valid `.zarr` that metadata.csv does not point to yet. | Click to save the path. The file stays where it is — nothing is copied. |
| **⚠️ Dimension mismatch** | A valid `.zarr` whose Z, Y, X or number of timepoints differs from the raw image (hover for details). | Re-export the segmentation with matching dimensions. |
| **🔄 Fix zarr format** *(button)* | A `.zarr` with the wrong internal layout (not stored one timepoint per chunk, or a group of arrays instead of a single array; hover for the reason). | Click to rewrite it in the expected layout. |
| **🔄 Convert TIFF → zarr** *(button)* | A `.tif` / `.tiff` file. | Click to convert it to Zarr (see below). |
| **⚠️ Format not supported (…)** | Any other file type (e.g. `.h5`, `.ims`). | Export the segmentation as TIFF first. |

**Fix zarr format** and **Convert TIFF → zarr** write a new file at the standard location and point metadata.csv to it:

```
<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr
<output_dir>/images/<sample>/<sample>_mask_dead.zarr        (dead mask)
```

If that file already exists, you are asked whether to **overwrite** it.

**Converting a TIFF** first opens **Confirm Axis Order**, showing the file's shape and the detected axis order. Pick the order that describes your file (e.g. `TZYX` = time, Z, Y, X); only orders with the right number of axes are offered. A missing time or Z axis is added (a single 3-D volume becomes one timepoint). Label images must have **one channel**; files with several channels are refused.

**➕ Add a new sample or cell type…** (top of the page) jumps to the Data Preparation **Metadata Builder** in edit mode, to add samples or populations that are not in the metadata yet.

```{important}
The checks cover the **file structure** and, for segmentations, that Z, Y, X and the number of timepoints match the raw image. They do **not** check that the pixel values are sensible labels (`0` = background, one integer per cell). Inspect the imported segmentation in the [Visualization tab](../../plugin_essentials/visualization.md) before tracking it.
```

## What "valid" looks like

BEHAV3D EXPLORER expects each segmentation zarr to be:

- a **single array** (not a zarr group containing sub-arrays);
- **4-D `(T, Z, Y, X)` integer labels**, with `0` = background and positive integers = individual cells within each timepoint;
- stored **one timepoint per chunk**.

A dead mask follows the same layout, with `0` = background and non-zero = dead signal.

## Importing from common sources

### Imaris

Imaris keeps segmentations inside the `.ims` file, which cannot be imported directly.

1. Export the segmentation (label image) from Imaris as a `.tif` / `.tiff` time series.
2. In the Import page, **Browse…** to the exported file in the row of that sample and population.
3. Click **🔄 Convert TIFF → zarr** (or **⚡ Convert / Save All Samples**) and confirm the axis order.

### Ilastik / scikit-image / external scripts

Export the result as a TIFF (or a zarr) label image and import it the same way. For any other format (`.h5`, …) export to TIFF first. Label values are kept as they are.

## After importing

Once each row shows **✅ Ready for tracking**, the Visualization tab can display these segmentations and the [Tracking](../tracking/index.md) tab can use them like segmentations produced inside BEHAV3D EXPLORER.

## Tips

- **Check one converted sample in the Visualization tab before converting the rest.** A wrong axis order (e.g. T and Z swapped) converts without error but gives nonsense downstream.
- **Prefer absolute paths.** A relative path is also looked up relative to the folder that contains metadata.csv.

## See also

- [Segmentation overview](./index.md) — method comparison and shared concepts.
- [Data Preparation](../../data_preparation.md) — the metadata.csv columns that store the segmentation and dead-mask paths.
- [Output Directory & File Layout](../../plugin_essentials/output_layout.md) — where files are written.
- [Import existing tracks](../tracking/import.md) — the same workflow for tracking data.

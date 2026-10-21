# 👁 Visualization tab

The Visualization tab is tab 2 of the BEHAV3D EXPLORER dock widget. Despite its position you come back to it after every step of the pipeline: it loads any sample's raw image, segmentation masks, tracked-segment labels and tracks into napari layers so you can inspect what was just produced.

![Visualization tab](../_static/screenshots/visualization_tab.png)

## What this tab does

1. Receives the metadata loaded in the Data Preparation tab (until then it only shows *Load metadata in the Data Preparation tab to see visualization options.*).
2. Lists every sample in the **Sample:** drop-down.
3. On **Load Dataset into Napari**, opens the selected sample as napari layers:
   - **Raw channels**: one image layer per channel;
   - **Segments**: one labels layer per population, plus the dead mask if there is one;
   - **Tracked Segments**: labels layers in which each cell keeps the same ID (and colour) over time;
   - **Tracks**: one napari Tracks layer per population.
4. Provides **Visibility Toggles** to show or hide each group of layers, and a **Manual Edition** panel to correct tracked segments.

## Where the data comes from

All paths are read from the loaded metadata.csv, which the pipeline steps keep up to date:

| Layer group | Source |
|---|---|
| **Raw** | `raw_image_path` (after conversion, the `.zarr` in `images/<sample>/`). |
| **Segments** | `{prefix}_{ct}_segments_image_path` for each population; dead mask from `dead_mask_path`. |
| **Tracked Segments** | `{prefix}_{ct}_tracks_image_path` for each population (including `{base}_merged` for multicolor populations). |
| **Tracks** | `{prefix}_{ct}_tracks_csv_path` for each population. |

Tracked segments and tracks of cell-type groups created in the Analysis tab are loaded too, once their tracked segments have been built. Anything that does not exist yet (for example, you have not tracked yet) is skipped, and the **Log** box at the bottom says why (`Path not defined in metadata` or `File not found`).

## Layer naming and colours

| Layer | Name | Colour |
|---|---|---|
| Raw channel | `{sample} – Ch0`, `{sample} – Ch1`, … | cyan, yellow, green, red, blue, magenta (repeats after six channels) |
| Segments | `{sample} – {cell type} segments` | napari's default label colours |
| Dead mask | `{sample} – dead mask` | napari's default label colours; hidden on load |
| Tracked segments | `{sample} – {cell type} tracked segments` | napari's default label colours |
| Tracks | `{sample} – {cell type} tracks` | same colour as the cell in the tracked-segments layer |

Colours and contrast can be changed in napari's layer controls. Changes to the **raw channels** are remembered per channel number for the whole project, so the next sample you load opens with the same colours and contrast.

## Workflow

### 1. Load metadata first

Load or build the metadata in the [Data Preparation](../data_preparation.md) tab. The info line then reads *N sample(s) available. Select one and click Load.*

### 2. Pick a sample and click Load Dataset into Napari

The tab:

- removes all existing layers from the viewer (so you start fresh);
- resets the toggles to **Raw** ✓, **Segments** ✓, **Tracked Segments** ✓, **Tracks** ✗;
- loads the layers. Converted (Zarr) images are read lazily, so even large movies open within seconds.

When a population already has tracked segments, its plain segments layer is hidden and the **Segments** toggle is switched off, because the tracked layer shows the same masks with stable IDs. For multicolor populations only the merged result is shown; the per-colour layers are loaded hidden and can be switched on individually in napari's layer list.

### 3. Toggle layer groups

| Toggle | Shows | Typical use |
|---|---|---|
| **Raw:** | All raw channels | Signal quality, dim or bright cells |
| **Segments:** | Segmentation labels of every population | Under- or over-segmentation, background bleed |
| **Tracked Segments:** | Tracking output (each cell keeps its colour) | ID swaps, broken tracks, identity jumps |
| **Tracks:** | The Tracks layers (lines through time) | Trajectories, especially with the time slider |

Switching a group off hides every layer of that group, including per-colour multicolor layers; switching it back on does not re-show the per-colour layers.

### 4. Manual Edition (optional)

The **Manual Edition** group appears once the loaded sample has at least one tracked-segments layer.

- **Edit target:** (only for multicolor populations): **Merged (combined)** edits the merged result directly; **Independent channels** edits each colour separately, after which **🔁 Rebuild merged** regenerates the merged result from the corrected colours.
- **Tracked segments:** picks the layer to correct.
- **✏️ Edit tracked segments** opens the editor below the button (the button then reads **⏹ Stop editing**).

The editor itself is documented in [Manual Editing](../processing/tracking/manual_editing.md).

```{warning}
If you try to switch tabs while the editor has unsaved changes, BEHAV3D EXPLORER asks whether to save, discard or cancel. Read the dialog before clicking, or you may lose your corrections.
```

## What to inspect after each step

| After this step… | …check that |
|---|---|
| Data Preparation (Zarr conversion) | the image opens, channels are in the right order, Z and time are not swapped |
| Segmentation | masks are tight to the cell boundaries, without background bleed or cells split into pieces |
| Tracking | cells keep the same colour over time, without jumps or merges |
| Filtering | the remaining tracks look sensible |

## Tips

- The Tracks layers are **off by default** because they can be visually noisy. Turn them on to inspect motility and displacement.
- The **Raw:** toggle hides all raw channels at once, convenient when you only want to look at masks.
- The **Log** box lists every layer that was added or skipped; use it to answer "why don't I see my segments?".

## See also

- [Output Directory & File Layout](output_layout.md): where the files shown here live on disk.
- [Manual Editing](../processing/tracking/manual_editing.md): correcting tracked segments.

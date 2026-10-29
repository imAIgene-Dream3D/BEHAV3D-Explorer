# 📍 Tracking

Tab 4 of the BEHAV3D EXPLORER dock widget (**📍 Tracking**). Tracking turns per-timepoint segmentation labels into **trajectories** — each physical cell keeps the same integer ID across time — and writes two outputs per (sample, cell type):

```
<output_dir>/images/<sample>/<sample>_<cell_type>_tracked.zarr
<output_dir>/trackdata/<sample>/<cell_type>/<sample>_<cell_type>_tracks.csv
```

The tracked zarr is a 4-D `(T, Z, Y, X)` label image in which the pixel value is the **TrackID** (stable across timepoints). The CSV has one row per `(TrackID, timepoint)` with physical-unit and pixel coordinates.

## Per-cell-type sub-tabs

The Tracking tab shows one **sub-tab per cell type** declared in the metadata (listed down the left side), so each population gets its own method and parameters — what works for objects that overlap themselves between frames rarely works for objects that do not. Until metadata is loaded the tab only shows *"Load metadata in the Data Preparation tab to see tracking options."*

| Icon | Category | Method selected in a fresh project |
|---|---|---|
| 🟣 | Organoid (`or_` columns) | Fragmentation Propagation (in the combined **🟣 All Organoids** sub-tab, see below) |
| 🔵 | Immune (`im_` columns) | btrack (Bayesian) |
| 🟡 | Other (`ot_` columns) | LapTrack |

These are starting points only — choose the method from your data (see the note below).

Special sub-tabs:

- **🟣 All Organoids** — shown by default whenever the metadata has organoid cell types: every organoid type is tracked together in one Fragmentation Propagation run. Untick **Track all organoids together** to get one independent sub-tab per organoid type. Details: [Fragmentation Propagation → Track all organoids together](fragmentation_tracking.md).
- **… (multicolor)** — per-channel multicolor cell types (e.g. `tcell_1_multicolor`, `tcell_2_multicolor`) share one sub-tab, e.g. **🔵 Tcell (multicolor)**. See [Multicolor cell types](#multicolor-cell-types).

## The tracking methods

Pick the method from the **Method:** dropdown of each sub-tab. The **?** button next to it opens an illustrated summary of all methods (with schematics for the three propagation methods). Methods fall into two families — **centroid-based** (link by how far a detection moved) and **overlap-based / propagation** (link by how much a detection overlaps its previous position).

### Centroid-based tracking methods

Link detections frame-to-frame by comparing centroid positions, not overlap. Use one of
these when an object moves enough between frames that it may no longer overlap itself —
roughly one cell diameter or more.

| Method | What it does |
|---|---|
| **btrack (Bayesian)** | Bayesian tracker whose Kalman-filter motion model predicts where each object will be next, with an optional global hypothesis optimiser. The routine choice for objects that lose overlap between frames. See [btrack](btrack.md). |
| **LapTrack** | Linear Assignment Problem tracking: links detections between consecutive frames by optimal assignment on centroid distances, then closes gaps and optionally detects merges and splits. No motion prediction. An alternative to btrack. See [LapTrack](lap.md). |
| **TrackPy** | Crocker–Grier linker: links each frame to the next with the smallest total displacement, with memory for short gaps and an adaptive search range. No motion prediction, no merge/split handling. An alternative to btrack, reliable only when cells are well separated relative to how far they move per frame. See [TrackPy](trackpy.md). |

### Overlap-based tracking methods (propagation)

Propagate the previous timepoint's labels onto the current timepoint's mask by spatial
overlap (watershed-based), not centroid distance. Use one of these when an object stays
largely in place so consecutive frames still overlap — typical for organoids and other
near-static structures.

| Method | What it does |
|---|---|
| **Fragmentation Propagation** | Propagates labels by overlap; no tunable parameters. Every piece of an object keeps the object's ID, so it suits objects that mainly change by **fragmenting** (e.g. organoids breaking up on death) in a closed set: IDs come only from the first timepoint, so objects that appear mid-movie get **no ID** and are not tracked. See [Fragmentation Propagation](fragmentation_tracking.md). |
| **Bounded Propagation** | Same overlap-based propagation, but one track ID can never span disconnected regions, and objects that appear mid-movie as separate regions get **new IDs**. Use it when masks can **touch or join** but should keep separate identities. See [Bounded Propagation](bounded_propagation.md). |
| **Reporter Propagation** | Pools segments across the whole movie, groups spatially overlapping ones regardless of time, and stamps each group's largest detection onto every timepoint — so an object first detected mid-movie also gets an ID and is present at **every** timepoint. For near-static objects whose segmentation **flickers on and off** (e.g. an intermittent reporter). See [Reporter Propagation](reporter_propagation.md). |

### Import tracking

| Method | What it does |
|---|---|
| **Import tracking** | Brings in label images tracked outside BEHAV3D EXPLORER (TIFF or zarr) and converts them into the standard outputs. See [Import tracking](import.md). |

```{note}
**Pick the family from what you measure, not from the cell type.** Between two
consecutive frames, does an object still overlap where it was?

- **Large overlap** → an overlap-based/propagation method.
- **Little or no overlap** (the object moves about one cell diameter or more between
  frames) → a centroid-based method, normally **btrack**.

Whether a population falls on one side or the other depends on your **frame
interval**, not on what the cells are: the same cell is "slow" at a short interval and
"fast" at a long one. Check the overlap in the Visualization tab before committing, and
verify every method visually on at least one sample before a batch run.
```

## Generic workflow per sub-tab

1. **Method:** — pick the method in the **Tracking Method** box.
2. **Set the parameters** in the panel that appears below it (each control has a **?** help button). Fragmentation Propagation has no parameters; btrack has a two-step layout; Import tracking shows a per-sample table instead.
3. **Apply to all *Category*** / **Apply to all** (optional) — copy this sub-tab's method and all its parameters to the other sub-tabs of the same category (e.g. **Apply to all Immunes**) or to every cell type. Only ordinary per-cell-type sub-tabs receive the settings, not the All Organoids or multicolor sub-tabs.
4. **Run *Cell type* Tracking** — the green button at the bottom of the sub-tab (e.g. **Run Tcell Tracking**). It tracks this one cell type in **every sample** of the metadata. It is disabled when **Import tracking** is selected (imports use their own buttons).

Tracking runs [in the background](../../plugin_essentials/running_steps.md#background-execution), with progress in the bar under the sub-tabs and in the **Log** box. If some outputs already exist you are first asked what to do with them (see [Existing tracking data](#existing-tracking-data)). When the run finishes, a dialog offers to switch to the Visualization tab, load the first sample and show the tracks.

The current settings are saved to `behav3d_parameters.yml` each time you run, so they are restored when you reopen the project.

```{tip}
**How to confirm tracking worked.** In the Visualization tab, look at the tracked-segments layer over time: **untracked** segments get a new colour/ID every timepoint, whereas **correctly tracked** segments keep the **same colour/ID across frames**. Hover over a label and check that its ID stays constant as you scrub through time.
```

### Tab-level controls (below the sub-tabs)

Two controls under the sub-tab area act on **all** cell types at once:

- **Run Batch Tracking (All Cell Types)** — the blue button. Runs every sub-tab with its current method and parameters, one after the other (the All Organoids and multicolor sub-tabs included).
- **+🛒** — adds a **📍 Batch Tracking** step to the [Processing Queue](../../plugin_essentials/processing_queue.md). The step is global: when it runs it tracks every cell type with the settings of its sub-tab at that moment.

### Existing tracking data

When tracked outputs already exist for some samples, any run first shows an **Existing Tracking Data** dialog with the usual choices (see [When results already exist](../../plugin_essentials/running_steps.md#when-results-already-exist)). With **Skip & Resume**, an incomplete output is completed without re-tracking where possible (e.g. a missing CSV is rebuilt from the tracked image).

### Distance units

LapTrack, TrackPy and btrack have a **Distance units:** switch (**px** ⇄ **µm**) at the top of their parameters. It is on **µm** by default, because these trackers link on physical coordinates (pixel size from `pixel_distance_xy` / `pixel_distance_z` in the metadata). Switching to **px** only changes how the distances are displayed and entered; the values are converted for you.

## Multicolor cell types

A multicolor population is tracked **channel by channel** with one shared set of parameters, then the channels are combined into a single cell type named `<base>_merged` (e.g. `tcell_merged`). The **(multicolor)** sub-tab shows an orange banner, a **Channels to track (same parameters applied to all)** list, the usual method and parameter panel, and the button **▶ Run *Base* Multicolor Tracking (*N* channels + merge)**.

When merging, track IDs are renumbered so they stay unique across channels, and the merged CSV gains a `source_cell_type` column naming each track's channel. If two channels claim the same voxel at the same timepoint, the merge stops with a "Voxel overlap detected" error — fix the overlapping segmentations first.

## CSV column reference

Every tracking method writes a CSV with the same columns:

| Column | Description |
|---|---|
| `TrackID` | Track identifier, stable across timepoints; equals the pixel value in the tracked zarr. |
| `SegmentID` | For btrack, LapTrack and TrackPy: the label the object had in the per-timepoint segmentation. For the propagation methods, imports, manually edited data and CSVs rebuilt from a tracked image (**Skip & Resume**): identical to `TrackID`. |
| `position_t` | Timepoint index, starting at 0. |
| `position_x`, `position_y`, `position_z` | Centroid in **physical units** (pixel position × `pixel_distance_xy` / `pixel_distance_z` from the metadata). |
| `pixel_position_x`, `pixel_position_y`, `pixel_position_z` | Centroid in **pixels**. |

Extra columns: `organoid_type` (organoids tracked together, see [Fragmentation Propagation](fragmentation_tracking.md)) and `source_cell_type` (merged multicolor output).

## Outputs

| File | Path |
|---|---|
| Tracked labels image | `<output_dir>/images/<sample>/<sample>_<cell_type>_tracked.zarr` |
| Tracks table | `<output_dir>/trackdata/<sample>/<cell_type>/<sample>_<cell_type>_tracks.csv` |

The metadata CSV is updated with the paths of the new files. Organoids tracked together additionally produce a combined `all_organoids` set, and multicolor groups a `<base>_merged` set (see above).

## After tracking

### Manual editing of tracked segments

Automated segmentation and tracking usually leave a few mistakes — split or merged labels, swapped identities, missed cells, junk objects. You correct them on the **tracked** data (`*_tracked.zarr` + tracks CSV), so **run tracking first**. The editor is opened from the **[Visualization](../../plugin_essentials/visualization.md) tab** with **✏️ Edit tracked segments**. Full instructions: **[Manual editing of tracked segments](manual_editing.md)**.

### Feature extraction

- **[Feature Extraction](../../analysis/feature_extraction.md)** uses the tracks CSV and tracked zarr to compute per-timepoint and per-track features.

```{toctree}
:hidden:
:maxdepth: 1

btrack
fragmentation_tracking
bounded_propagation
reporter_propagation
trackpy
lap
import
manual_editing
```

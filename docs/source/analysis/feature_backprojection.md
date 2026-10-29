# 🔬 Feature Backprojection

The first sub-tab of the **📊 Analysis** tab. It paints any numeric feature of a cell type — speed, elongation, `percentage_dead_mask`, a contact flag… — onto that cell type's segmentation, for the timepoint the napari time slider is on. It is a quick visual check: nothing is written to disk.

## When to use it

- To **sanity-check a feature** before analysing it: does a high `speed` really sit on the cells that move, does `percentage_dead_mask` light up the dying organoids?
- To **see the spatial pattern** of a feature across the sample over time.
- Right after [Feature Extraction](feature_extraction.md): it works on the unfiltered table too, so you can look before [Filtering](filtering.md).

## Controls

| Control | What it does |
|---|---|
| **Cell type:** | Every organoid, immune and other cell type (and any [cell-type group](index.md#grouping-cell-types)). |
| **Sample:** | The sample to show. |
| **Feature:** | Every numeric column of the cell type's track-features table (identifiers and positions excluded). The **filtered** table is used when it exists, otherwise the unfiltered one. |
| **▶ Show Preview** | Loads the preview (see below). Enabled once the feature list is loaded. |

A status line under the button reports what is shown (*"Showing '<feature>' for <cell type> / <sample> at t=<t>."*) or what is missing — e.g. no track-features table (run Feature Extraction first), or no raw or tracked image for that sample.

## What you see

**▶ Show Preview** clears the viewer, then adds:

- the raw image channels of the sample;
- **[Feature BP] <cell type> filtered TrackID** — the tracked labels, restricted to the tracks present in the table (hidden by default);
- **[Feature BP] <feature>** — each cell filled with its feature value at the current timepoint, in the *inferno* colour map at 50 % opacity. Cells with no value at that timepoint are set to 0.

Moving the time slider recomputes only the frame you are on, so scrubbing is fast. The contrast is set from the 2nd–98th percentile of the first frame shown; contrast or opacity changes you make in napari's layer controls are kept while you scrub.

```{tip}
Changing the cell type or sample removes the preview layers; press **▶ Show Preview** again. For state or trajectory-cluster labels painted onto the images, use the backprojection steps of [Single Cell](single_cell/index.md) instead.
```

## See also

- [Feature Extraction](feature_extraction.md) — what each feature column means.
- [Visualization](../plugin_essentials/visualization.md) — loading raw images, segmentations and tracks into the viewer.

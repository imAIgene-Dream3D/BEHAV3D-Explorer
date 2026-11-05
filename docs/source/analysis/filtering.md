# 🧹 Filtering

Tab 6 of the BEHAV3D EXPLORER dock widget. The Filtering tab restricts the data to the timepoints you want to analyse and drops or trims tracks that won't survive downstream analysis — short tracks, tracks past the experiment duration, undersized or already-dead starting cells — and produces:

1. A **filtered per-timepoint table** that every downstream analysis (Death Dynamics, Interaction, Invasiveness, Single Cell) reads.
2. A **summarised per-track table** with one row per surviving track (see [below](#the-summarised-csv)).
3. **Quality-control PDFs** showing how many tracks each filter removed and what the surviving tracks look like.

```{important}
**Filtering must be run even if you apply no filters at all.** The filtered table it writes is the input of Population Dynamics and the preferred input of Feature Backprojection and Single Cell. Re-run Filtering whenever Feature Extraction (or **🔄  Re-run death**, or Active Killing) has produced new results for that cell type.
```

## Workflow at a glance

1. Open the [sub-tab of a cell type](#per-cell-type-sub-tabs).
2. (Optional) [Preview the track lengths](#preview-track-lengths) to choose **Min length** and **Max length**.
3. Choose [the filters](#the-filters) (each has its own checkbox; they are applied in this order):
   1. [Restrict to a timepoint range](#restrict-to-a-timepoint-range) (shared by all cell types).
   2. [Trim the full time series to a maximum number of timepoints](#trim-full-time-series-to-max-timepoints).
   3. [Filter by minimum size at the first timepoint](#filter-by-minimum-size-at-the-first-timepoint).
   4. [Filter short tracks](#filter-short-tracks).
   5. [Trim long tracks to a maximum length](#trim-long-tracks-to-max-length).
   6. [Filter dead cells at the first timepoint](#filter-dead-cells-at-the-first-timepoint) (only with a dead channel).
   7. Pick the [unit for time-based filters](#unit-for-time-based-filters).
4. (Optional) [Copy the settings to the other sub-tabs](#apply-run-queue) with **Apply to all <Category>** or **Apply to all**.
5. [Run](#apply-run-queue) **Filter <Cell type> Tracks & Summarize**, **Run Batch Filtering (All Cell Types)**, or queue it with **+🛒**. Run it even if you apply no filters at all.
6. [Check the outputs](#outputs): the filtered and summarised tables and the [quality-control PDFs](#quality-control-pdfs).

## Per-cell-type sub-tabs

Filtering is organised as sub-tabs per cell type along the left side, with the usual colour code:

| Icon | Category |
|---|---|
| 🟣 | Organoid |
| 🔵 | Immune |
| 🟡 | Other |

Each sub-tab has its own filter settings and run button. The one exception is the **timepoint range**, which is a single setting shared by all cell types. The lower half of the tab is the shared [📄 Results panel](../plugin_essentials/results_panel.md).

## Preview track lengths

> **📊 See Track Length Distributions**

Opens a window with per-sample histograms of track length (in timepoints) from the **unfiltered** Feature Extraction table — the quickest way to choose **Min length** and **Max length** before running. Nothing is written to disk (use the window's toolbar to save the figure).

```{tip}
The Co-pilot assistant ([QueenB](../getting_started.md#co-pilot-assistant-queenb)) adds two quick actions on this tab: **Track lengths** opens the same preview, and **Cell counts** tabulates, per sample, how many cells present at timepoint 0 would survive a minimum track length of 20, 50, 100 and 200 timepoints. The table is saved as `BEHAV3D_<cell_type>_track_count_summary.csv` in the cell type's `quality_control/` folder.
```

## The filters

Every filter has its own checkbox; when a filter is unchecked, its settings are hidden and it is skipped. Defaults on a fresh project:

| Control | Default |
|---|---|
| **Restrict analysis to a timepoint range** | off (`0` – `0`) |
| **Trim full time series to max timepoints** | on, **Max timepoints:** `350` |
| **Filter tracks shorter than minimal length** | on, **Min length:** `30` |
| **Trim tracks to maximum length** | on, **Max length:** `30`, **Split long tracks into chunks** on |
| **Filter by minimal size at first timepoint** | off, **Min size:** `1000` voxels |
| **Filter dead cells at first timepoint** | off — *only shown when the metadata declares a dead channel* |
| **Unit for time-based filters:** | `frames` |

Filters are applied in this order, each one seeing only what the previous ones left: **timepoint range → max timepoints → min size → min length → max length → dead at first timepoint**.

### Restrict to a timepoint range

> ☐ **Restrict analysis to a timepoint range** — **Start timepoint:** `0` — **End timepoint:** `0`

Keeps only the rows inside the inclusive **Start**–**End** window. These are **absolute positions in the movie** (e.g. "timepoints 20 to 80"), not per-track durations. Tracks that overlap the window only partially are **kept and cropped** — **Min length** then drops those that end up too short. **End** must be greater than **Start**, otherwise the run is refused.


```{tip}
If you already know you will never need the frames outside the window, you can instead cut the images themselves with **Clip timepoints (select range)** in [Data Preparation → Convert to Zarr](../data_preparation.md#6--convert-to-zarr). Segmentation, tracking and every analysis then work on those timepoints only, and the steps run faster. 
```

```{warning}
**One range for all cell types.** The range appears on every sub-tab and editing it on one updates all the others. It cannot differ between cell types because every cross-cell-type analysis (interaction, invasiveness, multi-organoid death dynamics) lines the cell types up on the timepoint axis; different windows would silently give wrong percentages and invented data.

**Original timepoints are preserved.** `position_t` keeps its original values, so plots of a 56–78 window read 56–78, not 1–23.
```

#### First-timepoint filters with a range

> ☐ **Enable first-timepoint filters, measured at the start of this range** *(only shown while a range is active)*

The **minimal size** and **dead at first timepoint** filters normally test each track at its first frame of the whole movie, which a window usually cuts away. While a range is active they are therefore greyed out and switched off. Tick this box to re-enable them, measured at each track's first timepoint **inside** the range (a track that appears part-way through is judged where it appears). Unlike the range itself, this option is per cell type.

#### Relationship to other timepoint settings

| Setting | What it cuts | Re-run cost |
|---|---|---|
| [**Data Preparation → Clip timepoints (select range)**](../data_preparation.md#6--convert-to-zarr) | The *images*, before segmentation and tracking | Everything downstream |
| **Filtering → Restrict analysis to a timepoint range** (this one) | An already extracted feature table | Filtering and the analyses |
| **Analysis → Custom time range** | One analysis' view of the data | That analysis only |

Active Killing deliberately scans the **full, unfiltered** movie so it never has to be re-run when you change this range; the range is applied when its results are read. When an Analysis **Custom time range** is also set, the analysis uses the overlap of the two.

### Trim full time series to max timepoints

> ☑ **Trim full time series to max timepoints** — **Max timepoints:** `350`

Keeps only the first **Max timepoints** frames of the movie and drops every later row, for all tracks. Typical use: cut off the tail of a movie where the dye has bleached or the signal has degraded.

```{tip}
If you already know you will never need the frames outside the window, you can instead cut the images themselves with **Clip timepoints (select range)** in [Data Preparation → Convert to Zarr](../data_preparation.md#6--convert-to-zarr). Segmentation, tracking and every analysis then work on those timepoints only, and the steps run faster. 
```

### Filter short tracks

> ☑ **Filter tracks shorter than minimal length** — **Min length:** `30`

Drops whole tracks that span fewer than **Min length** timepoints (measured after the range, max-timepoints and min-size filters). Short tracks usually come from false detections, transient over-segmentation or cells that briefly entered the field of view, and add noise to every per-track statistic.

### Trim long tracks to max length

> ☑ **Trim tracks to maximum length** — **Max length:** `30`
> &nbsp;&nbsp;☑ **Split long tracks into chunks**

For each track longer than **Max length**:

- **Split long tracks into chunks** on: the track is cut into **consecutive full-length chunks**. The first chunk keeps the original `TrackID`; each following full chunk becomes a **new** track with a new `TrackID` (and its `relative_time` restarts at 1); a final remainder shorter than **Max length** is discarded. E.g. a 100-timepoint track at max length 30 yields three 30-timepoint tracks. The `original_TrackID` column keeps the real track number, so backprojection can still find the cell.
- **Split long tracks into chunks** off: the track is simply **cropped** to its first **Max length** timepoints.

Use it for track-level analyses that need equal-length inputs; [State Trajectory](single_cell/track_classification.md) offers an equivalent option, so this can be done here or there.

### Filter by minimum size at the first timepoint

> ☐ **Filter by minimal size at first timepoint** — **Min size:** `1000` — **Units:** `px` ⇄ `µm`

Drops whole tracks whose cell is smaller than **Min size** at its first timepoint. The size is a voxel count; the **Units** switch only changes the display (µm³ by default when the metadata has pixel sizes, voxels otherwise). Common use: removing debris or segmentation fragments.

### Filter dead cells at the first timepoint

> ☐ **Filter dead cells at first timepoint**

Drops every track that is already `dead` at its first timepoint — useful to study **dying** rather than **already-dead** cells. Only shown when the metadata declares a dead channel, and needs the `dead` column from Feature Extraction (a **Dead mask % threshold** above 0).

### Unit for time-based filters

> **Unit for time-based filters:** `frames` ⇄ `hours`

Switches the display of **Start timepoint**, **End timepoint**, **Max timepoints**, **Min length** and **Max length** between frames and hours (using the frame interval from the metadata; disabled when it is missing). Filtering always runs on frames, so the switch never changes the result.

## Apply, Run, Queue

Per sub-tab:

- **Apply to all <Category>** (e.g. **Apply to all Organoids**) — copies this sub-tab's filter settings and unit to every sub-tab of the same category (the timepoint range is already shared).
- **Apply to all** — copies them to all cell types.
- **Filter <Cell type> Tracks & Summarize** — filters and summarises this cell type. If a filtered table already exists, an **Overwrite Existing Filtered Data?** dialog asks before replacing it.

Below the sub-tabs:

- **Run Batch Filtering (All Cell Types)** — runs every cell type in turn. If filtered data exists, the **Existing Filtered Data** dialog offers the usual choices, with **Customize…** per cell type (see [When results already exist](../plugin_essentials/running_steps.md#when-results-already-exist)).
- **+🛒** — adds a Filtering step to the [Processing Queue](../plugin_essentials/processing_queue.md).

Settings are saved to `behav3d_parameters.yml`.

## Outputs

For each cell type:

```text
<output_dir>/analysis/<cell_type>/
├── track_features/
│   ├── BEHAV3D_<cell_type>_combined_track_features_filtered.csv      # per-timepoint
│   └── BEHAV3D_<cell_type>_combined_track_features_summarized.csv    # one row per track
└── quality_control/
    ├── BEHAV3D_filter_counts.pdf
    ├── BEHAV3D_<target>_touching_distribution.pdf      # one per contact column
    ├── BEHAV3D_dead_dye_distribution.pdf               # if mean_dead_dye exists
    └── BEHAV3D_dead_dye_distribution_t0.pdf            # if mean_dead_dye exists
```

### The filtered CSV

Same columns as the Feature Extraction table, restricted to the surviving rows, plus `original_TrackID` and the metadata grouping columns (`well`, `exp_nr`, `*_line_condition`). **This is the file every downstream analysis reads.**

```{warning}
If [Active Killing](population_analysis/active_killing.md) results exist for the cell type, Filtering reads Active Killing's table instead of the plain Feature Extraction table, so the killing columns survive filtering. Two consequences:

- **Re-run Filtering after every Active Killing run** — Active Killing reads the unfiltered table and reminds you when an older filtered table exists.
- If Feature Extraction (or **🔄  Re-run death**) was run **after** Active Killing, Filtering stops with an error for that cell type: re-run Active Killing first, then Filtering.
```

### The summarised CSV

One row per track (`sample_name`, `TrackID`), built from the filtered table:

| Column | How it's aggregated |
|---|---|
| `track_length` | Number of timepoints after filtering |
| `mean_<col>` | **Mean** over the track of every numeric and true/false column (e.g. `mean_speed`, `mean_volume`, `mean_<type>_contact` = fraction of timepoints in contact) |
| `dies` | `True` if the track is dead at any timepoint (needs the `dead` column) |
| `displacement_from_origin` | **Maximum** distance from the start reached by the track |
| `cumulative_displacement` | Value at the **last** timepoint (total path length) |
| `well`, `exp_nr`, `*_line_condition` | Carried over for grouping |

It is read only by the legacy *original BEHAV3D* (feature-based DTW → UMAP → KMeans) option of [State Trajectory](single_cell/track_classification.md); all other analyses work from the per-timepoint filtered CSV.

### Quality-control PDFs

Always written, one folder per cell type:

| File | Content |
|---|---|
| `BEHAV3D_filter_counts.pdf` | **(1)** Per-sample histograms of track length **before** filtering. **(2)** Per-sample bar chart of how many tracks remain after each stage (timepoint-range, min-size and dead-at-first-timepoint bars appear only when those filters are on; the min-length bar is counted after the max-length step, so splitting can raise it). **(3)** Per-sample histograms of track length **after** filtering. |
| `BEHAV3D_<target>_touching_distribution.pdf` | One per contact column (e.g. `BEHAV3D_organoid_touching_distribution.pdf` for T cells, plus `any_organoid` / `any_immune_cell`): per sample, the number of cell-timepoints touching vs not touching. |
| `BEHAV3D_dead_dye_distribution.pdf` | Violin + point plot of `mean_dead_dye` over all remaining timepoints, per sample (drawn before the dead-at-first-timepoint filter is applied). |
| `BEHAV3D_dead_dye_distribution_t0.pdf` | The same at each track's first timepoint — a check for the dead-at-first-timepoint filter. |

## Tips & best practices

- **Look at `BEHAV3D_filter_counts.pdf` after every run.** The per-stage bar chart is the fastest way to spot a filter that removes far too many or far too few tracks.
- **Mind the default max-length split.** It is on by default and turns every track longer than 30 timepoints into 30-timepoint pieces. Untick **Trim tracks to maximum length** if you want to keep full-length tracks.
- **Minimum track length is usually the most impactful filter** for behavioural analyses; check the length histograms before choosing it.
- **Re-run Filtering whenever its inputs change** (Feature Extraction, **🔄  Re-run death**, Active Killing); downstream analyses otherwise keep reading the old filtered table.

```{tip}
**What we filter in practice:**

- **Organoids** are usually tracked accurately and rarely need filtering — at most a small minimum length to drop fragments, or a large one (close to the experiment duration) to keep only organoids followed from start to end.
- **T cells / immune cells:** drop tracks shorter than roughly **10–30 timepoints**; these short fragments are typically junk.
- **Keep the minimum permissive for Behavioral State.** A strict minimum length mostly removes the cells that are hardest to follow continuously (fast-moving, or only briefly in contact) and keeps the easy ones (often static or dead), which shifts per-timepoint state proportions. Drop only fragments here; when an analysis needs fixed-length windows, let [State Trajectory](single_cell/track_classification.md#track-clustering-controls) cut them with **Divide long tracks**.
- **Often tracks are not trimmed to max length at filtering already.** At some analysis steps you can choose to also trim the tracks, so you can use full length tracks for other analysis steps where full tracks are needed
- **Trim full time series** is the right tool when the dead dye bleaches or the signal degrades late in the movie.
```

## See also

- [Feature Extraction](feature_extraction.md) — produces the table Filtering reads.
- [Population Dynamics](population_analysis/index.md) — reads the filtered per-timepoint CSV.
- [Single Cell](single_cell/index.md) — classifies per-cell behavioural states from the filtered CSV.
- [Output Directory & File Layout](../plugin_essentials/output_layout.md) — where the CSVs and QC PDFs live.

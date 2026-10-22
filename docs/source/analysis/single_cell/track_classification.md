# 🛤️ State Trajectory

The second analysis of **📊 Analysis → 🧬 Single Cell** (open it with the **State Trajectory** card's start button on the [Single Cell overview](index.md)). Where [🔬 Behavioral State](state_classification.md) labels behaviour **timepoint by timepoint**, State Trajectory groups **whole trajectories** into clusters by how their behaviour unfolds over time — two cells that follow a similar behavioural "story" (e.g. *static* ▶ *scanning* ▶ *killing*) end up in the same trajectory cluster, even if they are never in the same state at the same frame.

It runs on the cell type chosen in the shared **Cell type** dropdown at the top of Single Cell. The default method compares the **sequence of behavioural states** along each track, so you normally run [Behavioral State](state_classification.md) for that cell type first; a **Feature-based** method can cluster tracks directly on track features without it.

## Workflow at a glance

1. [Choose a clustering method](#choosing-a-clustering-method) — State-based or Feature-based (see [How it works](#how-it-works)).
2. [Track clustering](#track-clustering) — set the features and parameters, run the clustering.
3. [Renaming clusters](#renaming-clusters) — give the clusters biological names, merge, order and colour them.
4. [Reports & Plots](#reports--plots) — diagnostics, exemplar tracks, feature heatmap, composition, transitions, condition and contact reports.
5. [Backprojection](#backprojection) — paint the clusters back onto the images to validate them.
6. [Train track classifier](#train-track-classifier) — reuse the same clusters on new data.

## How it works

Tracks are compared with **dynamic time warping (DTW)**: a distance between two time series that tolerates the same behaviour happening a bit earlier, later, faster or slower. Similar tracks are then grouped into clusters. The **Trajectory clustering method** dropdown picks what is compared:

- **State-based** (default) — uses each track's per-timepoint behavioural states from [Behavioral State](state_classification.md) (`full_behavioral_state`). The **Basis** picks how:
  - **dtw** (default) — DTW over the sequence of state labels.
  - **bouts** — no DTW: each track is described by summary numbers (fraction of time in each state, bout counts and lengths, state-to-state transition probabilities, state bigrams/trigrams), and these profiles are clustered.

  Because both compare *which states occur, in what order or proportion*, they do not separate tracks by differences in raw values (e.g. exactly how fast a cell moves) within a state.
- **Feature-based** — DTW directly over per-timepoint track features you pick (speed, shape, contact, …) from the filtered track-features table of [Filtering](../filtering.md). It does not need Behavioral State. Tick **Use features used in behavioral state clustering** to reuse instead the exact features *and* scaling of the cell type's behavioural-state model.
- **Legacy original BEHAV3D** — inside Feature-based, **Advanced: run legacy BEHAV3D algorithm (DTW → UMAP → KMeans)** reproduces the trajectory analysis of the [original BEHAV3D](https://www.nature.com/articles/s41587-022-01397-w) pipeline: DTW over the features, a UMAP embedding of the distances, then K-means on the UMAP. It also reads the summarised CSV written by [Filtering](../filtering.md#the-summarised-csv).

With **dtw**, **bouts** and the newer Feature-based pipeline, the clusters are formed with **agglomerative** (hierarchical, fixed number of clusters) or **leiden** (graph communities, number of clusters found from the data) clustering (**Method**). Every method writes the same kind of result, so renaming, reports, backprojection and the classifier work the same afterwards.

## Choosing a clustering method

```{mermaid}
flowchart TD
    A[State Trajectory] --> B{Apply a saved<br/>trajectory classifier?}
    B -->|Yes| C[Apply pretrained trajectory classifier]
    B -->|No| E{Behavioral State run<br/>for this cell type?}
    E -->|No| F[Locked to Feature-based]
    E -->|Yes| G{Trajectory clustering method}
    G -->|State-based| H[Basis: dtw or bouts]
    G -->|Feature-based| I{Legacy BEHAV3D<br/>option ticked?}
    I -->|No, default| J[DTW + agglomerative / leiden<br/>on picked or state-model features]
    I -->|Yes| K[Original BEHAV3D<br/>DTW → UMAP → KMeans]
    H --> L[Rename → Reports & Plots →<br/>Backprojection → Train classifier]
    J --> L
    K --> L
```

| Method | Compares | Needs first |
|---|---|---|
| **State-based**, Basis `dtw` (default) | Sequence of behavioural states | [Behavioral State](state_classification.md) for this cell type |
| **State-based**, Basis `bouts` | State fractions, bouts, transitions, n-grams | [Behavioral State](state_classification.md) for this cell type |
| **Feature-based** | Hand-picked per-timepoint features | [Filtering](../filtering.md) |
| **Feature-based** + **Use features used in behavioral state clustering** | The state model's own features and scaling | [Behavioral State](state_classification.md) for this cell type |
| **Feature-based** + legacy option | Original BEHAV3D features (or your picks), UMAP + K-means | [Filtering](../filtering.md) |
| **Apply pretrained trajectory classifier** | Assigns the clusters of a saved classifier | A trained classifier `.pkl` + this cell type's behavioural states |

```{warning}
**Without Behavioral State results for the cell type**, the sub-tab shows *"⚠ Behavioral states not found for cell type '…'"* and locks the method to **Feature-based**. You can cluster and rename, but **Reports & Plots**, **Backprojection** and **Train track classifier** stay disabled until Behavioral State has been run for that cell type. Once its results exist, the dropdown switches back to **State-based** (unless you had chosen Feature-based yourself).
```

```{note}
**Applying a saved classifier instead of clustering.** Tick **Apply pretrained trajectory classifier** at the top: the clustering and training sections are hidden and a box asks for the **Classifier path** (`.pkl`) and **States h5ad path** (the cell type's behavioural states; auto-filled when they exist). Click **▶ Apply Pretrained Classifier**. The result replaces any trajectory clustering of this cell type and can be renamed, reported and backprojected as usual. See [Train track classifier](#train-track-classifier).
```

## Track clustering

The **Track clustering** section holds the method choice, the feature picker (Feature-based), the clustering parameters and the rename step.

### Trajectory Features (Feature-based only)

Shown when **Trajectory clustering method** is **Feature-based**. It lists the columns of the cell type's filtered track-features table (see [Feature Extraction](../feature_extraction.md#column-reference)); coordinates are not offered.

- **Use features used in behavioral state clustering** (above the picker; only shown once Behavioral State exists for the cell type) — reuses the state model's continuous and binary features with its saved scaling. The picker is greyed out and **Feature Processing** does not apply; a box lists the features used.
- **Timepoint Features** — grouped by feature family; pre-ticked: `speed`, `sphericity`, `elongation`, `extent`, `solidity`.
- **Window Features** — computed here from the cell positions, not by Feature Extraction:
  - **Window size (timepoints)** (default `5`, range 2–500) with `net_displacement`, `straightness`, `mean_square_displacement` over a rolling window (none ticked by default).
  - **Full-tracklet** features `net_displacement_tracklet` and `net_displacement_tracklet_squared` (not ticked by default): straight-line distance (or its square) from the first timepoint of each trajectory — the whole track, or each window when **Divide long tracks** is on — at every later timepoint. Not available with **Use features used in behavioral state clustering**, and not used by the legacy option.
- **Binary Features** — detected 0/1 columns (e.g. `<type>_contact`); none ticked by default. Their influence is set by **Binary weight** in Advanced Configuration.
- **Feature Processing** (collapsed): **Smooth window (timepoints)** `1` (= no smoothing), **Low percentile cap** `0.0` and **High percentile cap** `1.0` (= no clipping), **Start offset (timepoints)** `1` (drops each track's first timepoint, whose speed is always 0), and **Log scaling (log1p) of selected features** (`speed` pre-ticked).

Continuous features are processed as set, then standardized (z-scored). A summary box under the picker lists the current selection and processing.

### Track clustering controls

| Control | Default | What it does |
|---|---|---|
| **Trajectory clustering method** | State-based | **State-based** or **Feature-based** (see above). |
| **Trajectory size (timepoints)** | `100` | Number of timepoints compared per track. Shorter tracks are discarded; longer ones are trimmed (see **Trim mode**). The line below shows the real-time equivalent from the metadata; the maximum is capped at the longest track available. |
| **Divide long tracks** | off | Instead of trimming, splits long tracks into consecutive, non-overlapping windows of *Trajectory size*; each window is clustered on its own (leftover timepoints are dropped). Windows keep their `TrackID`, so backprojection still works. Required for the [Track Transition Report](#track-transition-report). |
| **N clusters** | `6` (2–200) | Number of clusters. Hidden when **Method** is `leiden`. |
| **Basis** | `dtw` | *State-based only*: `dtw` or `bouts`. |
| **Method** | `agglomerative` | `agglomerative` (cut into **N clusters**) or `leiden` (number of clusters set by **Leiden resolution**). Hidden for the legacy option, which always uses K-means. |

Click **▶ Run Track Clustering** (labelled **▶ Run Bout/Proportion Clustering** for the `bouts` basis and **▶ Run Feature-based BEHAV3D Clustering** for the legacy option). It runs in the background with a progress bar and **Log** (see [Running Steps](../../plugin_essentials/running_steps.md#background-execution)). **+🛒** adds it to the [Processing Queue](../../plugin_essentials/processing_queue.md); the queue always runs the newer pipelines, so run the legacy option with its button.

```{warning}
Running the clustering again **replaces the whole previous State Trajectory result of this cell type** — cluster names, reports and the trained classifier included. There is no Overwrite / Skip prompt.
```

When clustering finishes, a first **exemplar overview** is drawn automatically in `clustering/raw/`: `example_tracks_overview.pdf` (**Exemplars / cluster** random tracks per cluster, default 10, drawn as state bars) and, for DTW-based clusters, `medoid_tracks_overview.pdf` (the most central track of each cluster). This needs state bars, so it is skipped for the legacy option and for Feature-based clusters on hand-picked features.

### How to judge whether `N clusters` is sensible

Treat **N clusters** as a starting guess. Start around `6`, look at the exemplar overview, then decide:

1. **Does each cluster show one main pattern?** If one cluster mixes visibly different trajectory stories, raise **N clusters**.
2. **Do clusters really differ?** If two clusters look interchangeable, they may not deserve separate names.
3. **Are some clusters tiny?** Small clusters can be real but deserve scepticism; they often end up merged.
4. **Could you name each cluster with one biological label?** If not, the clustering may still be too coarse.

Prefer **slight over-splitting**: it is easier to merge similar clusters while [renaming](#renaming-clusters) than to separate mixed ones. The [Diagnostic](#diagnostic) report supports this decision but the exemplar tracks are the main evidence. With `leiden`, tune **Leiden resolution** instead (higher = more, smaller clusters).

### ⚙ Advanced Configuration

Collapsed by default; the rows shown depend on the method.

| Control | Default | Shown for | What it does |
|---|---|---|---|
| **Trim mode** | `last` | always | For tracks longer than *Trajectory size*: `last` keeps the final timepoints, `first` the first ones. (The legacy option always keeps the first timepoints.) |
| **Linkage** | `average` | `dtw` / Feature-based, agglomerative | `average`, `complete` or `single`. |
| **Bouts linkage** | `ward` | `bouts`, agglomerative | `ward`, `complete`, `average` or `single`. |
| **Leiden neighbors** / **Leiden resolution** | `15` / `1.0` | `leiden` | Neighbourhood graph size; higher resolution = more, smaller clusters. |
| **State fractions** / **Bout stats** / **Transitions** / **Bigrams/trigrams** | all on | `bouts` | Which track descriptors are clustered. |
| **PCA before clustering** | on | `bouts` | Reduces the descriptors with PCA (95 % variance kept). |
| **Log-ratio transform proportions (CLR)** | on | `bouts` | Transform suited to proportions that sum to 1. |
| **Log-transform bout lengths** | on | `bouts` | Stops a few very long bouts from dominating. |
| **Balance feature blocks (MFA)** | on | `bouts` | Gives each descriptor group a comparable weight, whatever its number of columns. |
| **Drop redundant/constant features** | on | `bouts` | Removes near-constant and highly correlated (> 0.95) descriptors. |
| **Parallel computation** | on | DTW methods | Uses several CPU cores for the distance calculation. |
| **Save distance matrix CSV** | off | DTW methods | Also saves the full track-to-track distance matrix (large for many tracks). |
| **Binary weight** | `1.0` (0–10) | Feature-based | Weight of the binary (e.g. contact) features in the distance: `1` = like one continuous feature, `0` = ignored, > 1 = contacts drive the clustering more. |
| **UMAP n_neighbors** / **UMAP min_dist** / **UMAP spread** | `15` / `0.1` / `1.0` | DTW methods | Shape of the UMAP drawn in the diagnostics only — never changes the clusters. |
| **Random seed** | `123` | always | Makes re-runs (and classifier training) reproducible. |
| **Advanced: run legacy BEHAV3D algorithm (DTW → UMAP → KMeans)** | off | Feature-based | Switches to the original BEHAV3D pipeline. A dialog reminds you that it needs tracks of equal length (filter or trim them in [Filtering](../filtering.md) first); tick **Don't show again** to hide it. |
| **Use exact original BEHAV3D settings** | on | legacy | Uses the original features — `mean_square_displacement`, `speed`, `mean_dead_dye` and every detected contact column, scaled as in the paper. Untick to use your **Trajectory Features** picks instead. |
| **UMAP n_neighbors** / **UMAP min_dist** | `15` / `0.1` | legacy | The UMAP that K-means clusters on. |

### Renaming clusters

Fresh clusters are numbered. Until a result exists the status line reads **"ℹ Run clustering first to enable renaming."**; afterwards it shows how many trajectories were loaded.

Click **✏ Rename Track Clusters** to open **Rename Track Trajectory Clusters**. Each row shows the **Current name**, a **New name** field and a **Color** button; drag rows to set the order used in every plot. Giving several clusters the **same name merges them**. **💾 Save** applies the changes, **✖ Discard** closes without saving, **↺ Reset Names** restores the original numbered clusters (after a confirmation; cannot be undone). Reset is not available for clusters from the `bouts` basis, the legacy BEHAV3D option or an applied classifier.

The names replace the labels in the `ClusterID` column; the labels from before the most recent save are kept in `ClusterID_original`. Colours are stable per name across re-runs unless you pick one. Names, order and colours are used by every report and by backprojection.

Saving regenerates the [Diagnostic](#diagnostic) report and the exemplar/medoid overviews in `clustering/` with the new names (not for the legacy option — re-run **▶ Create Diagnostics** there). Other reports keep the old names until you re-run them.

Use renaming as the curation step after inspecting the exemplars: the goal is **interpretable trajectory classes**, not keeping every split the algorithm made.

## Reports & Plots

Once a clustering exists, click **Generate analysis and plots ▸** (below the Track clustering section). A list of reports opens; **Start** on a report opens its settings, and **‹ Back** returns to the list, then to the clustering settings. Each report shows its own run button and a **👁** button that reopens its latest result (also listed in the [Results panel](../../plugin_essentials/results_panel.md)).

The settings a report ran with are saved per cell type in `behav3d_parameters.yml` and restored when you reopen the project or switch back to that cell type. The contact settings shared by all contact reports are saved by whichever contact report runs.

### Diagnostic

**▶ Create Diagnostics** writes the clustering-quality PDF to `clustering/`:

- **DTW methods**: a UMAP of the trajectories coloured by cluster (drawn with the current **UMAP** settings of Advanced Configuration, so you can redraw it without re-clustering), cluster sizes and a heatmap of the track-to-track distances (up to 200 tracks). It also refreshes the exemplar and medoid overviews.
- **bouts**: UMAP and correlation matrix, descriptor heatmap, matrix plot and cluster-occurrence ranking, plus the exemplar overview.
- **Legacy option**: the original BEHAV3D UMAP, cluster-feature heatmap and cluster percentages.

### Exemplar Tracks

Representative tracks per cluster, each drawn as a bar of its behavioural states over time. Needs Behavioral State for the cell type.

| Control | Default | What it does |
|---|---|---|
| **Exemplars / cluster** | `10` (1–100) | Number of tracks shown per cluster. |
| **Overview statebars** | on | Creates the PDFs below — keep it ticked. |

**▶ Create Exemplar PDFs** writes `example_tracks/example_tracks_overview.pdf` (all clusters side by side — the main visual check of the clustering) plus one PDF per cluster and one per exemplar rank.

### Feature Heatmap

Clusters × features: each feature is averaged over each track's trajectory timepoints, then per cluster. Cell labels show the unscaled cluster mean.

| Control | Default | What it does |
|---|---|---|
| **Available features** / **Selected features** | the clustering's features | Select rows in **Available** and click **Add ▼** (or double-click) to move them to **Selected**; **▲ Remove** (or double-click) sends them back. Drag rows in **Selected** to set the plot's row order. **Select used features** restores the clustering's own features; **Select all** / **Clear** fill or empty **Selected**. A box lists the current selection. |
| **Colour scaling** | `z-score` | `z-score`: each feature standardized across all tracks, then averaged per cluster (0 = average track, red = higher, blue = lower). `min-max`: cluster means rescaled 0–1 per feature. |

**▶ Create Feature Heatmap** writes the heatmap PDF and two tables: the cluster means and the per-track means they are built from.

### Track Composition Report

What fraction of tracks falls in each cluster, per sample — and, over time, the share of currently active tracks in each cluster.

| Control | Default | What it does |
|---|---|---|
| **Group in X** / **Group in Y** | (none) | Metadata columns (`well`, `exp_nr`, `*_line_condition`) whose levels form the columns / rows of a grid of composition bars. With neither set you get only the per-sample bars. |
| **Group conditions** | off | Next to each Group in X / Y: pools that column's levels into two groups (see the tip below). |
| **Group per page** | none selected | Further columns (Ctrl/Cmd-click for several) whose combinations get one grid page each. |
| **Time bin size** | `1` | Groups timepoints into bins of this size for the over-time plots (`1` = no binning). |

**▶ Track Composition Report** writes one PDF plus tables with, per group, the mean proportion and the number of tracks (`n_tracks`) behind each bar. Selecting three or more **Group per page** columns gives a flat panel layout rather than a grid.

```{tip}
**Pooling levels with "Group conditions".** Every **Group in X / Y** on this page and in [Behavioral State](state_classification.md) has a **Group conditions** checkbox. Ticking it shows a two-list picker: put each level of the column in the left or the right group, and that axis is reduced to one left-vs-right comparison.

Example: 6 organoid lines, 3 healthy and 3 tumor. Choose the line column for **Group in X**, tick **Group conditions**, put the healthy lines left and the tumor lines right: the report compares healthy-pooled vs. tumor-pooled instead of every line separately.
```

### Track Transition Report

How often one trajectory cluster is followed by another, pooled over all tracks and windows — the trajectory counterpart of the [State Transition Report](state_classification.md). It **needs Divide long tracks** to have been used: without windows each track has only one cluster.

Options (**⚙ Advanced Configuration**, open by default):

| Control | Default | What it does |
|---|---|---|
| **Min probability cutoff** | `3.0 %` | Transitions below this are not drawn in the circular diagram. |
| **Emphasis (gamma)** | `2.0` | How strongly line width and opacity favour the strongest transitions (1 = linear). |
| **Node labels** | On dots | Cluster names inside the dots, or in a numbered **Side legend**. |
| **Transition matrix heatmap** | on | Page with the transition counts / probabilities. |
| **Circular transition diagram** | on | Circular diagrams weighted by outgoing and by incoming probability. |
| **Include self-transitions** | off | Off: arrows show how a cluster splits among the clusters it switches to. On: probabilities include windows that stay in the same cluster, drawn as a loop beside each node. |
| **Show per-cluster breakdown grid** | on | Two pages with one small diagram per cluster (outgoing, incoming). |
| **Window transition Sankey (per sample)** | on | See below; its own **👁** reopens it. |

Click **▶ Track Transition Report**. The result is `transition_analysis.pdf`, with the matrices as tables.

#### Window transition Sankey

Reconnects the windows of each physical track (same `TrackID`) and shows, as a Sankey diagram, how a track's cluster changes from one window to the next — one diagram per sample plus a pooled page, saved as `window_transitions_all_samples.pdf`. Untick the other options to produce only the Sankey, or untick this one to skip it.

### Condition Comparison Report

Tests whether cluster proportions differ between the levels of a metadata column (e.g. organoid lines).

| Control | Default | What it does |
|---|---|---|
| **Compare condition** | first column | The column whose levels are compared pairwise for each cluster (Welch's t-test). |
| **Group in X** | (none) | Splits the comparison into side-by-side columns by a second column (with **Group conditions**, see the tip above). |
| **Group in Y** | — | Read-only: shows the **Compare condition**. |
| **Group conditions** | off | Pools the compared column's levels into two groups and runs one comparison. |
| **Group per page** | none selected | Columns that paginate the report. |

Click **▶ Condition Comparison Report**. Each cluster gets a signed bar of the proportion difference with significance stars (`*` to `****`); an extra page shows each cluster's prevalence over time per condition.

### Contact analysis

Reports that label every classified track **contact** or **no_contact** with another cell type, using the contact columns from [Feature Extraction](../feature_extraction.md#contact), and look at that split from different angles. They share one settings block and can be run one by one or together.

#### Shared settings

| Control | Default | What it does |
|---|---|---|
| **Contact column** | first detected | Which contact column to use (`*_contact` / `*_contact_on_distance`). |
| **Min. contiguous bout** | `5` timepoints | A track counts as **contact** if it has at least one unbroken run of this many contact timepoints in its classified window; otherwise **no_contact**. Lower to `1` to count any touch. |
| **Use contact cell classification** | off | Splits "contact" by the class of the **touched** cell (see the tip). |
| **Target classification** | State classification | Only with the box above: take the touched cells' classes from their **State classification** (Behavioral State) or **Track classification** (State Trajectory). |
| **Target state column** | `full_behavioral_state` | Only for **State classification**: `full_behavioral_state`, `hmm_intrinsic_behavioral_state` or `raw_hmm_state`. |
| **Group in X** / **Group in Y** / **Group per page** | (none) | Grouping of the contact grids, as in the [Track Composition Report](#track-composition-report), including **Group conditions**. |

```{tip}
**"Use contact cell classification" turns a yes/no flag into a class.** For example, if the touched population is a macrophage line classified by morphology (round / elongated / plastic), you can ask whether a trajectory cluster contacts *plastic* macrophages more than *round* ones, or contacts them longer (see [Contact Duration Comparison](#contact-duration-comparison)). It needs the touched cell type to have its own Behavioral State (or State Trajectory) results and the `touching_<type>s` column from Feature Extraction; a warning under the settings says what is missing.
```

#### Run All Contact Analyses

**▶▶ Run All Contact Analyses** runs, in order: Contact Rate → Contact Composition Grid → Contact Condition Comparison → Contact Cluster Heatmap → Contact Duration Comparison → Contact Type Comparison → Track Contact Overview. Reports whose requirements are not met are skipped with a note in the **Log**.

#### Contact Rate Report

Per sample, the % of tracks in contact; with **Use contact cell classification**, also the contact fraction per touched class. **▶ Create Contact Rate Report**.

#### Contact Composition Grid

Cluster composition of contact vs. no_contact tracks, in the Group in X / Y / page grid — do some clusters occur more among tracks that made contact? **▶ Create Contact Composition Grid**.

#### Contact Condition Comparison

Welch's t-test comparison of cluster composition between the contact and no_contact groups — the contact version of the [Condition Comparison Report](#condition-comparison-report). **▶ Create Contact Condition Comparison**.

#### Contact Cluster Heatmap

Per trajectory cluster of the tracks themselves: longest contact bout (in minutes when time metadata exist, otherwise timepoints), mean fraction of time in contact, and % of tracks with any qualifying contact — as heatmaps with the per-track distributions. **▶ Create Contact Cluster Heatmap**.

#### Contact Duration Comparison

Requires **Use contact cell classification**. For every class of the touched cells, how long tracks stay in sustained contact with that class, compared between every pair of classes and each class vs. all others pooled ("rest") — e.g. round-vs-elongated, …, round-vs-rest. Durations are shown in timepoints and, with valid time metadata, minutes.

| Control | Default | What it does |
|---|---|---|
| **Test** | Welch's t-test (unpaired) | `Welch's t-test (unpaired)` compares the per-track durations. `Paired t-test` first averages durations per **Pairing column(s)** value, then pairs the two classes' averages (values missing on one side are dropped). |
| **Pairing column(s)** | `sample_name` | Only for the paired test; Ctrl/Cmd-click for several (combined). |
| **Comparisons per page** | `12` | Number of comparison plots per PDF page. |
| **Long contact threshold** | `(off)` | Above 0, adds a page per group: among tracks that touched a class, the % with "long" vs. "short" contact. In `Percent of time window`: at least this % of the classified window in contact (gaps allowed, no time metadata needed). In `Seconds` / `Minutes` / `Hours`: longest sustained bout at least this long (needs time metadata). |
| **Long contact unit** | Minutes | Unit of the threshold. |

**▶ Create Contact Duration Comparison**.

#### Contact Type Comparison

Mean contact fraction per cluster, side by side for several contact columns (e.g. healthy vs. tumor organoid contact). Descriptive only, no statistical test (unlike [Behavioral State's contact type comparison](state_classification.md)). Select **Contact columns to compare** (Ctrl/Cmd-click for 2 or more), then **▶ Create Contact Type Comparison**.

#### Track Contact Overview

A visual check that the contact bouts behind these reports match real behaviour changes. For every track with a qualifying contact it draws the track's full behavioural-state bar with its contact bouts marked underneath, grouped by sample (and by **Group per page** combination). Needs Behavioral State for the cell type.

| Control | Default | What it does |
|---|---|---|
| **State column** | `full_behavioral_state` | Which state labels colour the bars (`full_behavioral_state`, `hmm_intrinsic_behavioral_state`, `raw_hmm_state`). |
| **Tracks per page** | `6` | Track rows per page. |

**▶ Create Track Contact Overview**.

## Backprojection

Paints the trajectory clusters back onto the images so you can check that tracks in the same cluster really behave alike. It works like the [Behavioral State backprojection](state_classification.md), colouring by trajectory cluster instead of state.

### Live overlay in napari

The **Live Napari Layer Backprojection** box:

| Control | Default | What it does |
|---|---|---|
| **Sample** | — All samples — | Sample to show (**— All samples —** shows the first sample). |
| **Color by** | `ClusterID` | `ClusterID` (current names) or `ClusterID_original` (labels before the most recent rename, once renamed). |
| **Opacity** | `80 %` | Opacity of the cluster overlay (10–100 %). |
| **Show trajectories** | on | Also draws each track's path in its cluster colour (one napari Tracks layer per cluster). |

**▶ Show Track Backprojection in Napari** replaces the viewer layers with the raw channels and a `track_cluster_class` layer that updates as you move the time slider, plus a **Track Cluster Mapping** legend. With behavioural states available, a **Track State Bar** panel shows the full state bar of any cell you click. Nothing is written to disk.

### Export backprojection

**▶ Export Track Backprojection** writes the overlay as a napari-readable `.zarr` for the selected sample, or for every sample with **— All samples —**:

```text
<output_dir>/analysis/<cell_type>/behavioral_states/backprojection/<sample>_<cell_type>_track_clusters.zarr
```

Open it in napari (drag and drop or **File → Open**). The Behavioral State export goes to the same folder with the suffix `_behavioral_states.zarr`, so both can be kept side by side.

```{note}
For a cell-type group whose tracked segments have not been built yet, Backprojection is disabled with a warning; build them from the Group Builder (**Build Tracked Segments**).
```

## Train track classifier

The collapsed **Train track classifier** section (last on the page) trains a random-forest classifier on your named clusters so the **same** clusters can be assigned to new data without re-clustering. It describes each track by its behavioural-state sequence, so it is only available for clusters built from behavioural states (State-based, or Feature-based with **Use features used in behavioral state clustering**).

**Train RF Classifier on Named Clusters**

| Control | Default | What it does |
|---|---|---|
| **Number of trees** | `300` (10–2000) | More trees = more stable but slower. |
| **Max depth** | `Unlimited (0)` | Maximum tree depth; limit it if a small dataset is overfitted. |
| **Test holdout %** | `20.0 %` (0–50) | Share of trajectories kept aside to measure accuracy (`0` = out-of-bag estimate only). |
| **⚙ Advanced Parameters** → **Min samples leaf** / **Min samples split** / **Max features** / **n_jobs** | `2` / `2` / `sqrt` / `All cores (-1)` | Usual random-forest settings. |

**▶ Train RF Classifier** saves the classifier and its validation (confusion matrices, classification report, accuracy metrics) in `classification/`. The **Random seed** of Advanced Configuration is used.

**Apply Classifier to New Data** — **Classifier path** (auto-filled after training) and **States h5ad path** (auto-filled from Behavioral State), then **▶ Apply Classifier**. It assigns a cluster to every track of the given states file and replaces the current result, adding a track-class proportion plot in `classification/`. In a project without its own clustering, use **Apply pretrained trajectory classifier** at the top of the page instead (see [Choosing a clustering method](#choosing-a-clustering-method)). One cluster is assigned per original track, even if the classifier was trained on divided windows.

## Outputs

Everything is written under `<output_dir>/analysis/<cell_type>/behavioral_trajectories/` (see [Output layout](../../plugin_essentials/output_layout.md)); the **👁** buttons and the [Results panel](../../plugin_essentials/results_panel.md) open the files directly.

```text
behavioral_trajectories/
├── BEHAV3D_<cell_type>_behavioral_trajectories.h5ad   one row per trajectory, cluster in ClusterID
├── BEHAV3D_<cell_type>_behavioral_trajectories.csv    same table, for spreadsheets
├── clustering/              diagnostics, exemplar and medoid overviews (raw/ = first version after clustering);
│                            distance matrix CSV if saved
├── example_tracks/          Exemplar Tracks (overview, per_cluster/, per_example/)
├── feature_heatmaps/        Feature Heatmap PDF + cluster / per-track mean tables
├── behavior_proportions/    Track Composition Report
├── transition_analysis/     Track Transition Report (tables in data/)
├── window_transitions/      Window transition Sankey (per-sample pages in sankey_pdf_pages/)
├── behavior_comparisons/    Condition Comparison Report
├── contact_analysis/        <report>/<contact column>/ — one folder per contact report, tables in csv/
│                            (Contact Type Comparison: <column>_vs_<column>/)
├── classification/          trained classifier (.pkl), qc/ validation, proportions after Apply
└── original_behav3d/        legacy option only: original BEHAV3D plots and tables
```

Feature-based clustering on hand-picked features also writes the per-timepoint features it used to `analysis/<cell_type>/trajectory_features/`. Exported backprojections go to `behavioral_states/backprojection/` (see [Export backprojection](#export-backprojection)).

## Tips & best practices

- **Run Behavioral State first** for the default State-based method — and for reports, backprojection and the classifier, which need it in any case.
- **Use the exemplar loop**: cluster → inspect the exemplar overview → rename and merge → re-run with a different **N clusters** only if clusters are still mixed or fragmented. Remember that re-running deletes the previous result.
- **Linkage**: keep `average`; `complete` is worth a try; `single` rarely works for these distances.
- **Leiden vs. agglomerative**: Leiden finds the number of clusters itself (tune **Leiden resolution**); agglomerative gives direct control through **N clusters**.
- **Divide long tracks** when tracks are much longer than *Trajectory size* and the discarded part likely holds different behaviour — and always if you want the Track Transition Report.
- **Match Min. contiguous bout to your time resolution**: a few timepoints ignore single-frame touches but keep real engagements.
- **Save the distance matrix only when needed** — it grows with the square of the number of tracks.

## See also

- [🔬 Behavioral State](state_classification.md) — the per-timepoint states this analysis builds on.
- [Filtering](../filtering.md) — the filtered table (and, for the legacy option, the summarised table) read here.
- [Feature Extraction](../feature_extraction.md) — the per-timepoint features, including contact columns.
- [Processing Queue](../../plugin_essentials/processing_queue.md) — run the clustering unattended.

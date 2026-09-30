# 🛤️ Track Classification

The second inner sub-tab of **Analysis → 🧬 Single Cell**. Where [State Classification](state_classification.md) labels behaviour **timepoint by timepoint**, Track Classification groups **whole trajectories** into clusters based on how their behaviour unfolds over time — so two cells that follow a similar behavioural "story" (e.g. *static*▶*scanning*▶*killing* ) end up in the same trajectory cluster, even if they are never in the exact same state at the exact same frame.

It runs on the **cell type chosen in the dropdown** at the top of the Single Cell sub-tab (immune / other only). By default it compares the **sequence of behavioural states** along each track (the **State-based** method, Basis `dtw`), so you normally run [State Classification](state_classification.md) for that cell type first — though a **Feature-based** method (below) can cluster tracks directly on raw features without it.

![Track Classification sub-tab](../../_static/screenshots/track_classification_tab.png)

```{note}
*Screenshot placeholder.*
```

## Workflow at a glance

1. [How it works](#how-it-works) — understand the State-based vs. Feature-based clustering families and their bases/methods.
2. [Choosing a clustering method](#choosing-a-clustering-method) — decide which method fits your data, using the decision flowchart.
3. [Track clustering](#track-clustering) — configure the trajectory features and run the clustering.
4. [Renaming clusters](#renaming-clusters) — give the trajectory clusters meaningful biological names.
5. [Reports & Plots](#reports-plots) — generate composition, transition, contact and exemplar reports.
6. [Backprojection](#backprojection) — paint the clusters back onto the raw images to validate them.
7. [Train track classifier](#train-track-classifier) — fit a reusable classifier to apply the same clusters to new data.

## How it works

Trajectory clustering groups whole tracks by how their behaviour unfolds over time, using one of two families of methods, set by the **Trajectory clustering method** dropdown:

- **State-based** — each track's per-timepoint behavioural-state labels (the states from [State Classification](state_classification.md)) drive the comparison, via one of two **Basis** options:
  - **dtw** (default) — the track's raw sequence of state labels, compared pairwise with **multivariate dynamic time warping** (dtaidistance), which aligns two sequences even when they differ slightly in length or timing.
  - **bouts** — trajectory-level summary features instead of a sequence comparison: per-track state proportions, per-state bout statistics (mean/max bout length, number of bouts), per-timepoint state-transition rates, and optionally bout-to-bout bigram/trigram transitions. You choose which feature groups to include.

  Both bases then cluster tracks using either **agglomerative hierarchical** or **leiden** clustering (**Method** dropdown). Because `dtw` and `bouts` compare *which states occur, in what order or proportion* — as you defined them in State Classification — they won't split clusters based on differences in raw feature values (e.g. exactly how fast a trajectory is). For that, switch to **Feature-based** and tick **"Use features used in behavioral state clustering"** (below) — it clusters on the *numeric* continuous and binary features the state model itself was built from, reusing its exact saved scaling, instead of the discrete state labels.

- **Feature-based** — clusters tracks directly on hand-picked raw per-timepoint features, without needing State Classification to have been run at all. This is the default, newer pipeline: DTW over those features, then agglomerative/leiden clustering on the distances. Its **"Use features used in behavioral state clustering"** checkbox (only available once State Classification has been run for the cell type) switches it to instead reuse the state model's own features *and* its exact saved scaling — see above.

  A buried **legacy mode**, reached by ticking **"Advanced: run legacy BEHAV3D algorithm (DTW → UMAP → KMeans)"** inside Advanced Configuration, instead reproduces the original [BEHAV3D](https://www.nature.com/articles/s41587-022-01397-w) paper's trajectory analysis: each track becomes a multi-dimensional numeric trajectory (displacement, speed, dead-dye intensity, contacts), DTW runs on those scaled features, the distances are embedded with **UMAP**, and **K-means** cuts the clusters.

Every one of these — `dtw` / `bouts` / the default Feature-based pipeline (hand-picked or state-model-preset features) / legacy Feature-based — writes to the same unified result and gets the same downstream Rename → Reports/Plots → Backprojection → Train/Apply Classifier workflow.

## Choosing a clustering method

```{mermaid}
flowchart TD
    A[Track Classification] --> B{Apply a saved classifier?}
    B -->|Yes| C[Apply pretrained trajectory classifier]
    B -->|No| D[Trajectory clustering method]
    D --> E{State Classification done<br/>for this cell type?}
    E -->|No| F[Method locked to Feature-based]
    E -->|Yes| G{State-based or Feature-based?}
    G -->|State-based| H[Basis: dtw / bouts]
    G -->|Feature-based| I{Legacy BEHAV3D toggle<br/>in Advanced Configuration?}
    I -->|off, default| J[DTW + Agglomerative/Leiden<br/>on hand-picked features]
    I -->|on| K[Original BEHAV3D DTW → UMAP → KMeans]
    H --> L[Unified behavioral_trajectories/ result]
    J --> L
    K --> L
    L --> M[Rename → Reports/Plots →<br/>Backprojection → Train/Apply Classifier]
```

| Family | Basis / mode | What it does | Requires first |
|---|---|---|---|
| **State-based** (default) | `dtw` (default) | Compares each track's sequence of behavioural-state labels | [State Classification](state_classification.md) for this cell type |
| **State-based** | `bouts` | Per-track state proportions, bout stats, transitions (optionally bigrams/trigrams) | [State Classification](state_classification.md) for this cell type |
| **Feature-based** (default pipeline) | — | DTW + agglomerative/leiden on hand-picked raw per-timepoint features | [Filtering](../filtering.md) (filtered track-features CSV) — no state classification needed |
| **Feature-based** | "Use features used in behavioral state clustering" checked | DTW + agglomerative/leiden on the state model's own numeric + binary features, with its exact saved scaling | [State Classification](state_classification.md) for this cell type |
| **Feature-based** | legacy toggle | Reproduces the original BEHAV3D paper's DTW → UMAP → KMeans trajectory analysis | [Filtering](../filtering.md) (filtered track-features CSV) |
| **Apply pretrained classifier** | — | Assigns a saved set of clusters to a new dataset | A saved classifier `.pkl` + matching states `.h5ad` |

```{note}
**Applying a saved classifier instead of clustering.** Use **Apply pretrained trajectory classifier** to select a saved classifier (`.pkl`) and its matching state data (`.h5ad`; auto-filled if State Classification output exists for the cell type), then click **▶ Apply Pretrained Classifier**. Use this to reproduce the same trajectory clusters on a new dataset.
```

```{tip}
**If State Classification hasn't been run** for the selected cell type, the sub-tab shows a warning and **locks the method to Feature-based** (the dropdown is force-set and greyed out):

> ⚠ Behavioral states not found for cell type '&lt;cell type&gt;'.
> State-based clustering requires running State Classification first and is unavailable until then.
> You can still run Feature-based clustering below.

Once State Classification exists for the cell type, the dropdown reverts to State-based automatically — but only if it had been force-locked; if you deliberately chose Feature-based yourself, your choice is kept.
```

## Track clustering

This is where you set up and run trajectory clustering, and where you rename the resulting clusters.

### Trajectory Features (Feature-based only)

Shown only when **Trajectory clustering method** is set to **Feature-based** — picks the raw per-timepoint features DTW runs on, read directly from the track-features table (no behavioral states needed).

- **"Use features used in behavioral state clustering"** checkbox, directly above the feature picker — only available once State Classification has been run for this cell type. Ticking it reuses the exact continuous and binary columns *and* the exact saved scaling/preprocessing the behavioral-state model was built from, instead of hand-picking and re-scaling features below; the **Feature Processing** settings below don't apply while it's checked. If State Classification's output disappears for this cell type (e.g. after switching cell type), this option is cleared automatically.
- Three tabs:
  - **Timepoint Features** — grouped by feature family; defaults pre-ticked: `speed`, `sphericity`, `elongation`, `extent`, `solidity` (the same defaults as State Classification's own feature picker).
  - **Window Features** — a **Window size** spinbox (default 5, range 2–500) plus `net_displacement` / `straightness` / `mean_square_displacement` checkboxes. **None of these three are pre-ticked by default** here — unlike State Classification's own window-feature picker, where `net_displacement` defaults on.
  - **Binary Features** — detected 0/1 columns (e.g. contact flags); none pre-ticked by default.
- **Feature Processing**: Smooth window (default **1** — smaller than State Classification's default of 5), Low/High percentile caps (0.0 / 1.0, off by default), Start offset (default 1), and a log-scaling grid seeded with `speed` pre-ticked.

```{tip}
Continuous features are z-scored after processing; binary columns are standardized and weighted by **Binary weight** (see Advanced Configuration below).
```

### Track clustering controls

| Control | Default | Meaning |
|---|---|---|
| **Trajectory size** | 100 | Number of timepoints each trajectory is resampled to before comparison (filters out tracks shorter than this length and cuts remaining tracks to this length). Set it to its minimum to switch to **Variable-length** mode (compare tracks at their native lengths). |
| **Divide long tracks** | off | Instead of trimming an over-length track to *Trajectory size*, split it into consecutive, non-overlapping, full-length windows. Each window keeps the same `TrackID` (so backprojection still works) but gets its own `trajectory_window_id`, so a single long track can contribute several independently clustered trajectories rather than one truncated one. |
| **N clusters** | 6 | How many trajectory clusters to cut the hierarchy into. Only shown when **Method** is `agglomerative` (not shown for `leiden`, which finds its own number of communities). |
| **Basis** | `dtw` | *State-based only* — `dtw` / `bouts` (see **How it works** above). |
| **Method** | `agglomerative` | `agglomerative` (hierarchical, cut into **N clusters**) or `leiden` (community detection on a nearest-neighbour graph, its own cluster count). |
| **Random seed** | 123 | Fixes the random parts of clustering so re-runs with identical settings are reproducible. |

Click the run button to run it in the background, with a progress bar and **Log**. It relabels depending on the current mode: **▶ Run Track Clustering** (default) / **▶ Run Bout/Proportion Clustering** (`bouts` basis) / **▶ Run Feature-based BEHAV3D Clustering** (legacy Feature-based). Next to it: **+🛒** queues the step for a batch run, and **👁** opens the result once it exists.

For the default `dtw` basis, treat **`N clusters`** as a practical starting guess rather than a number the software can determine for you biologically. Start around the default `6`, inspect the resulting exemplar overview, then decide whether the clustering needs to be split further or collapsed.

- **Increase `N clusters`** when one trajectory cluster still contains clearly different trajectories in the overview PDF, for example one cluster mixing tracks that linger, scan, and then disengage with tracks that go straight into sustained contact.
- **Decrease `N clusters` or merge afterwards** when clusters have very few examples, or when two clusters look so similar in the overview exemplars that you would describe them with the same biological name.
- **Prefer slight over-splitting to under-splitting.** It is usually safer to split a bit too much, then merge similar clusters when renaming by giving them the same final name.

```{tip}
A good cluster count for `dtw` is one where each cluster looks like a recognisable trajectory archetype when you inspect the exemplars, not just a mathematically distinct branch in the hierarchy.
```

### How to judge whether `N clusters` is sensible

For the default `dtw` workflow, the main evidence is the **exemplar overview PDF** saved as:

```text
<output_dir>/analysis/<cell_type>/behavioral_trajectories/example_tracks/example_tracks_overview.pdf
```

By default, this overview shows **10 representative tracks per cluster**, which is why it is the first thing to inspect after clustering. Use it to ask:

1. **Does each cluster show one main movement pattern?** If a single cluster contains several visually distinct trajectory stories, raise `N clusters`.
2. **Do different clusters really look different from one another?** If two clusters look nearly interchangeable across their exemplars, they may not deserve separate final names.
3. **Are some clusters too small to feel robust?** Very tiny clusters can still be real, but they deserve extra skepticism and often end up merged unless they show a very distinct pattern.
4. **Would you describe the cluster with a single biological label?** If not, the clustering may still be too coarse.

The **diagnostics PDF** is still useful, but for this specific decision it is secondary to the visual evidence in the exemplar overview. This cluster-count guidance applies to the default `dtw` basis; `bouts`, Feature-based (including with the state-features preset checked), and legacy Feature-based clustering use different diagnostics and aren't the focus here.

### ⚙ Advanced Configuration

The defaults are sensible for most data. Which rows are shown depends on **Basis** / **Method** / **Trajectory clustering method**:

| Control | Default | Shown for | Meaning |
|---|---|---|---|
| **Trim mode** | last | always | When a track is longer than *Trajectory size*, whether to trim from the `last` or `first` timepoints. |
| **Linkage** | average | `dtw` + agglomerative | Hierarchical-clustering linkage (`average`, `complete`, `single`). |
| **Bouts linkage** | ward | `bouts` + agglomerative | Hierarchical-clustering linkage for bout/proportion features (`ward`, `complete`, `average`, `single`). |
| **Leiden neighbors** | 15 | Method = `leiden` | Nearest-neighbour graph size for community detection. |
| **Leiden resolution** | 1.0 | Method = `leiden` | Higher values find more, smaller communities. |
| **State fractions** / **Bout stats** / **Transitions** / **Bigrams/trigrams** | all on | `bouts` only | Which feature groups to include: per-state time fractions, bout-length statistics, timepoint-to-timepoint transition rates, and bout-to-bout n-gram transitions. |
| **PCA before clustering** | on | `bouts` only | Reduce the assembled bout/proportion feature matrix with PCA before clustering. |
| **Log-ratio transform proportions (CLR)** | on | `bouts` only | Compositional (CLR) transform of the state-proportion features, appropriate since they sum to 1. |
| **Log-transform bout lengths** | on | `bouts` only | Log-transform bout-length statistics, which are typically right-skewed. |
| **Balance feature blocks (MFA)** | on | `bouts` only | Rescale each feature block (proportions / bout stats / transitions / n-grams) so no single block dominates the distance purely by having more columns. |
| **Drop redundant/constant features** | on | `bouts` only | Drop near-constant or highly redundant bout/proportion columns before clustering. |
| **Missing policy** | Keep as category | `dtw` | How to treat missing timepoints: `Keep as category` treats "missing" as its own state, or `Drop missing timepoints`. |
| **Parallel DTW computation** | on | not `bouts` | Use multiple CPU cores for the (expensive) pairwise distance computation. |
| **Save distance matrix CSV** | off | not `bouts` | Also write the full pairwise DTW distance matrix to CSV (large for many tracks). |
| **Binary weight** | 1.0 | Feature-based (non-legacy), including with the state-features preset checked | Relative weight of the standardized binary columns versus the continuous features in the distance calculation. |
| **QC UMAP n_neighbors / min_dist / spread** | 15 / 0.1 / 1.0 | `dtw`, Feature-based (non-legacy) | Display-only UMAP embedding shown in diagnostics for visual QC — it never affects the actual clustering. |
| **Advanced: run legacy BEHAV3D algorithm (DTW → UMAP → KMeans)** | off | Feature-based only | Switches Feature-based clustering to the original BEHAV3D pipeline (see **How it works** above). |
| **Use exact original BEHAV3D settings** | on | legacy Feature-based only | Reproduces the paper's exact feature set (mean_square_displacement / speed / mean_dead_dye, quantile-scaled, plus every detected contact column, min-max scaled) instead of using your **Trajectory Features** selections. |
| **UMAP n_neighbors / min_dist** | 15 / 0.1 | legacy Feature-based only | Not currently exposed as visible controls in the UI — the legacy pipeline always uses these default values. |

Turning the legacy toggle on shows a one-time disclaimer that the legacy engine needs equal-length tracks.

```{note}
**Divide long tracks** only affects clustering/training. **Apply Classifier** (see **Train track classifier** near the end of this page) always assigns exactly one cluster label per original track, even if the classifier it's using was trained on split windows.
```

### Renaming clusters

Freshly computed clusters are numbered, not named. This step lets you give them meaningful biological labels and merge clusters if you think they are biologically similar. You can additionally order the clusters by dragging them and give them unique colors, which are used in the following reports and backprojection.

The status line above the rename button reads **"ℹ Run clustering first to enable renaming."** until clustering has produced data; it then reports how many trajectories were loaded.

Click **✏ Rename Track Clusters** to open a dialog ("Rename Track Trajectory Clusters") where you can combine biologically similar clusters and give each trajectory cluster a meaningful name (e.g. *super-engager*, *engager*, *killer*, *scanner*, *static*). Giving two clusters the same name merges them. **👁** reopens the renamed result.

The cluster label lives in the **`ClusterID`** column; renaming overwrites it in place and keeps the original values as `ClusterID_original`. Downstream reports and **Backprojection** use whichever names/order/colors you set here.

Colors default to a **hash-stable** palette — a cluster keeps the same color across reruns and even when only a subset of clusters is plotted — unless you assign one manually here, in which case your choice is remembered instead.

Treat renaming as the **curation step** that follows the overview PDF. First inspect the representative trajectories, then decide which clusters deserve distinct names and which ones are just oversplit versions of the same movement archetype.

The goal here is **interpretable trajectory classes**, not preserving every split made by the clustering algorithm. If two clusters would end up with the same biological description, it is usually better to give them the same final name and merge them than to keep two labels that nobody can explain consistently.

```{tip}
Clicking **Apply cluster names** automatically regenerates the **diagnostics** and **track-class-proportion** plots (in Reports & Plots, below) with the new names, order and colors — no separate "regenerate" step needed for those two. The **Condition Comparison Report** and **contact-analysis** plots depend on which condition/contact column you pick, so re-run those from their own buttons after a rename.
```

## Reports & Plots

Use **Generate analysis and plots ▸** to create reports and plots.

### Diagnostic

**▶ Create Diagnostics** produces clustering-quality diagnostic plots (UMAP/correlation, heatmap, matrixplot, cluster-occurrence ranking). **👁** reopens the PDF.

For deciding whether **`N clusters`** was sensible, inspect the **exemplar overview PDF** (see **Exemplar Tracks**, near the end of this section) first and use this diagnostics PDF second — the overview tells you whether the model has split tracks into visually meaningful trajectory types; the diagnostics help support that interpretation, but they are not the primary evidence for this particular choice.

### Track Composition Report

What fraction of tracks each trajectory cluster occupies, per sample — optionally grouped by experimental condition.

| Control | Default | Meaning |
|---|---|---|
| **Group in X** | — none — | A metadata column whose levels become one axis of a 2D grid of proportion bars (one grid cell per combination). |
| **Group in Y** | — none — | A second metadata column for the other grid axis. With one or two columns set you get a true 2D grid; with neither set you just get the plain per-sample bars. |
| **Group per page** | none selected | Additional metadata column(s) (Ctrl/Cmd-click for several) whose combinations instead **paginate** the output — one full grid page per combination, rather than adding more grid axes. |
| **Time bin size** | 1 | Groups timepoints into buckets of this size before computing composition, to reduce visual noise. `1` = no binning (raw per-frame resolution). |

Click **▶ Track Composition Report**. The output is one PDF with the per-sample bars plus (if grouping is set) the grouped grid pages; a companion CSV records, per group, both the mean proportion *and* the underlying track count (`n_tracks`) behind each bar.

```{tip}
Three or more **Group per page** columns selected at once falls back to a flat, wrapped panel-per-combination layout rather than a true 2D grid — keep to at most two grouping axes (X + Y) if you want the proper grid.
```

```{tip}
**Pooling levels with "Group conditions".** Next to **Group in X** and **Group in Y** (here, and in every other report on this page and in [State Classification](state_classification.md) that has a Group in X/Y control) sits a **Group conditions** checkbox. Ticking it reveals a two-list picker where you manually assign each level of the chosen column to a left group or a right group, collapsing that axis down to one left-vs-right comparison instead of one bar/panel per level.

Worked example: you have 6 organoid lines — 3 healthy, 3 tumor. Pick the line-condition column for **Group in X**, tick **Group conditions**, put the 3 healthy lines in the left list and the 3 tumor lines in the right list. The report now compares healthy-pooled vs. tumor-pooled, instead of running (or plotting) all 15 pairwise line-vs-line combinations.
```

### Feature Heatmap

Heatmap of trajectory clusters × features: each feature is averaged over every track's trajectory timepoints, then per cluster. The features the clustering used are preselected; add any other per-timepoint feature. Cell labels show the unscaled cluster mean.

| Control | Default | Meaning |
|---|---|---|
| **Features** | the clustering's own features | Multi-select list (Ctrl/Cmd-click for multiple); **Select used features** restores the clustering's own selection, **Select all** / **Clear** adjust it in bulk. |
| **Colour scaling** | z-score | `z-score`: each feature is standardized across all tracks, then averaged per cluster; 0 = the average track, red = higher, blue = lower. `min-max`: cluster means rescaled 0–1 per feature, showing which cluster is highest/lowest. |

Click **▶ Create Feature Heatmap**. Output, under `analysis/<cell_type>/behavioral_trajectories/feature_heatmaps/`: `feature_heatmap.pdf`, `feature_heatmap_cluster_means.csv`, and `feature_heatmap_track_means.csv` (the per-track means the cluster means are built from).

### Window Transitions

Where **Track Transition Report** (below) pools everything and asks "which clusters transition into which, overall," Window Transitions asks a narrower question: **within one physical track**, does its assigned cluster drift from one "Divide long tracks" window to the next? It reconnects the sub-tracks that "Divide long tracks" split apart, via their shared parent `TrackID`, into a Sankey diagram of window-to-window cluster transitions — one diagram per sample, plus a pooled page across all samples.

This is about a single track's own windows changing cluster over time, not about different tracks influencing each other.

It **requires "Divide long tracks"** (in the **Track clustering controls**, above) to have been used — with a single window per track there is nothing to connect.

Click **▶ Create Window Transition Sankey** to run it. Output: `window_transitions_all_samples.pdf` (the merged, pooled-plus-per-sample PDF), with the individual per-sample pages also kept under a `sankey_pdf_pages/` subfolder.

### Track Transition Report

Pooled, inter-cluster transition analysis for trajectory clusters — the trajectory-cluster counterpart of [State Classification's State Transition Report](state_classification.md#reports). It uses the **same circular-diagram / Sankey engine and Advanced Configuration options** described there (min probability cutoff, emphasis gamma, node-label style, and the matrix/circular/self-transitions/per-cluster-grid/Sankey toggles) — see that page for what each control does. What's different here:

- It operates on **trajectory clusters** instead of per-timepoint states, and the window index is collapsed — every track's windows are pooled together, so this is a population-wide view, not a per-window one (see **Window Transitions** above for the per-window view).
- It **requires "Divide long tracks"** (in the **Track clustering controls**, above) to have been used when clustering. Without split windows, each track only ever has one cluster label, so there is nothing to transition between.
- Output: `transition_analysis.pdf`.

Click **▶ Track Transition Report** to run it; **+🛒** queues it, **👁** reopens the result.

```{note}
N-gram rankings are a state-sequence concept and don't carry over here — the Track Transition Report's Advanced Configuration is the matrix/circular/Sankey subset of State Transition Report's options, not the full set.
```

### Condition Comparison Report

Statistically compares trajectory-cluster proportions between the levels of one metadata column — e.g. does cluster composition differ between organoid lines?

| Control | Default | Meaning |
|---|---|---|
| **Compare condition** | — | The metadata column whose levels are compared pairwise (Welch's t-test) for each cluster. |
| **Group in X** | — none — | Splits the comparison into side-by-side columns from a second condition. |
| **Group per page** | none selected | Additional column(s) that paginate rather than add another axis. |

Click **▶ Condition Comparison Report**. Each cluster gets a signed bar showing the proportion difference between two condition levels, annotated with significance stars (`*`/`**`/`***`/`****`). When **Compare condition** has exactly two levels and a second grouping axis is also set, the report switches to a true 2D grid layout instead of one row per pairwise comparison.

**Group in X** here also supports the **Group conditions** pooling checkbox described under [Track Composition Report](#track-composition-report) above — the mechanic is identical.

### Contact Analysis

A family of reports that all start from the same question — labelling every classified track **contact** or **no_contact** with another cell type, using the `*_contact` columns from [Filtering](../filtering.md) — and then look at that split from different angles: rate, composition, condition comparison, a cluster-vs-contact heatmap, duration, and contact-type comparison. They all share one settings panel and can be run individually or all at once.

#### Shared settings

These apply to every contact report:

| Control | Default | Meaning |
|---|---|---|
| **Contact column** | first detected | Which per-timepoint contact column to use (auto-populated from columns ending in `_contact` / `_contact_on_distance`). |
| **Min. contiguous contact bout (timepoints)** | 5 | A track is labelled **contact** if it has an unbroken run of at least this many consecutive contact timepoints within its classified time window; otherwise **no_contact**. Raise it to require sustained contact and ignore fleeting touches; lower it (e.g. to 1) to count any single contact timepoint. |
| **Use contact cell classification** | off | Instead of a plain contact/no_contact flag, label each track's contact by the ***touched*** cell's own classification (see the tip below). |
| **Target classification** | State classification | Only used when **Use contact cell classification** is on: whether the touched population's classes come from its **State classification** or its **Track classification**. |
| **Target state column** | full_behavioral_state | Only relevant when **Target classification** = State classification: `full_behavioral_state`, `hmm_intrinsic_behavioral_state`, or `raw_hmm_state`. |
| **Group in X** / **Group in Y** | — none — | Same 2D-grid grouping as the plots elsewhere on this page, applied to the contact-related composition grids — including the **Group conditions** pooling checkbox described under [Track Composition Report](#track-composition-report). |
| **Group per page** | none selected | Same pagination behaviour as above. |

```{tip}
**"Use contact cell classification" turns a binary flag into a class label.** With it off, a track is simply *contact* or *no_contact*. With it on, "contact" is broken down by *which class* of the touched population it contacted. For example, if the touched population is a macrophage line already run through State Classification for morphology (classes round / elongated / plastic), you can ask whether a trajectory cluster contacts *plastic* macrophages more than *round* ones, or whether contact with a *plastic* macrophage tends to last longer than contact with a *round* one — see [Contact Duration Comparison](#contact-duration-comparison) below.
```

#### Run All Contact Analyses

**▶▶ Run All Contact Analyses** runs every report below once, using the shared settings above, in this order: Contact Rate Report → Contact Composition Grid → Contact Condition Comparison → Contact Cluster Heatmap → Contact Duration Comparison → Contact Type Comparison → Track Contact Overview.

#### Contact Rate Report

The simplest view: per-sample % of tracks in contact, plus — when **Use contact cell classification** is on — a per-target-class contact fraction. Click **▶ Create Contact Rate Report**. Output: `contact_rate.pdf`.

#### Contact Composition Grid

A cluster × contact/no_contact composition grid, faceted by **Group in X / Y / page** — do certain trajectory clusters occur more in tracks that made contact? Click **▶ Create Contact Composition Grid**. Output: `contact_composition.pdf`.

#### Contact Condition Comparison

A Welch's t-test grid comparing class composition between the contact and no_contact groups — the contact-analysis counterpart of the plain [Condition Comparison Report](#condition-comparison-report) above, always run as a binary (2D-grid-eligible) comparison. Click **▶ Create Contact Condition Comparison**. Output: `condition_comparison_<condition>.pdf`, written into `contact_analysis/contact_group_analysis/<contact_col>/` alongside `contact_analysis.pdf`.

#### Contact Cluster Heatmap

A heatmap-first complement to the Contact Duration Comparison violins below: crosses mean contact fraction and max-contact-bout-length against the **track's own** trajectory cluster (not the touched cell's class), alongside a heatmap of what fraction of tracks in each cluster made contact at all. Click **▶ Create Contact Cluster Heatmap**. Output: `contact_cluster_heatmap.pdf`.

#### Contact Duration Comparison

For every class the touched cell type's classification assigns (e.g. macrophage morphology classes round / elongated / plastic), how long tracks stay in **sustained contact** with a target of that class — the same "max contact-bout length" feature as the Contact Cluster Heatmap above, in **timepoints and minutes side by
side** (minutes needs a valid `time_interval`/`time_unit` in metadata; otherwise only timepoints are
shown). Every class is compared against every other class **and** against every other class pooled
together ("rest") — e.g. with round / elongated / plastic you get round-vs-elongated,
round-vs-plastic, elongated-vs-plastic, and round-vs-rest, elongated-vs-rest, plastic-vs-rest.

Requires **Use contact cell classification** (in the shared settings above) to be enabled — there is nothing to compare
"by class" without it.

| Control | Default | Meaning |
|---|---|---|
| **Test** | Welch's t-test (unpaired) | `Welch's t-test`: compares the raw per-touch durations between the two groups without assuming equal variances. `Paired t-test`: averages durations within each **Pairing column(s)** value first (e.g. per sample), then pairs the two groups' averages for the same value — a value missing either side is dropped from that comparison. |
| **Pairing column(s)** | `sample_name` | Only shown for the paired test. Multi-select (Ctrl/Cmd-click for multiple); defines a pairing unit from any of the same condition-like columns offered elsewhere, plus `sample_name` — when more than one is selected, durations are pooled per combination of their values first. |
| **Comparisons per page** | 12 | How many small boxplot pairs to place on each PDF page before starting a new one. |
| **Long contact threshold** | 0.0 (off) | When set above 0, adds an extra page per group: for tracks in contact with each touched class, the percentage with "long" contact — reaching this threshold, in the unit below — vs. "short" contact, restricted to tracks that touched that class at all (long vs. short, not vs. no contact). `Percent`: the track spent at least this % of its classified time window in contact with the class (contiguity-agnostic; no time metadata needed). `Seconds`/`Minutes`/`Hours`: its longest sustained contact bout with the class reached this real-time length (requires time metadata). |
| **Long contact unit** | Minutes | `Percent of time window` / `Seconds` / `Minutes` / `Hours` — see **Long contact threshold** above. |

Click **▶ Create Contact Duration Comparison**. Output:

```text
<output_dir>/analysis/<cell_type>/behavioral_trajectories/contact_analysis/contact_duration_comparison/<contact_col>/contact_duration_comparison.pdf
<output_dir>/analysis/<cell_type>/behavioral_trajectories/contact_analysis/contact_duration_comparison/<contact_col>/csv/contact_duration_comparison.csv
```

#### Contact Type Comparison

Mean contact fraction per cluster, side by side across selected contact columns — e.g. healthy vs. tumor organoid contact. Purely descriptive, no significance test (contrast with [State Classification's Contact type comparison](state_classification.md#contact-analysis), which *is* statistically tested).

Select **Contact columns to compare** (Ctrl/Cmd-click for 2 or more), then click **▶ Create Contact Type Comparison**. Output:

```text
<output_dir>/analysis/<cell_type>/behavioral_trajectories/contact_analysis/contact_type_comparison/<contact columns, "_vs_"-joined>/contact_type_comparison.pdf
<output_dir>/analysis/<cell_type>/behavioral_trajectories/contact_analysis/contact_type_comparison/<contact columns, "_vs_"-joined>/csv/contact_type_comparison.csv
```

#### Track Contact Overview

A **QC / sanity-check view**, in the same spirit as [Backprojection](#backprojection) below — not a figure meant for a manuscript, but a way to visually confirm that the contact bouts driving all the reports above actually line up with real behavioural changes. For every track whose contact meets the **Min. contiguous contact bout** threshold, it plots that track's full (untrimmed, classified-window) behavioural-state trajectory as a coloured bar, with a grey/green bar directly beneath marking every contact bout of at least that length. Pages are grouped by sample — a sample's tracks are never split across a page shared with the next sample's, even if that leaves the page under-full.

**Group per page** works here too: tracks are never split across a page shared with a different combination of the selected columns' values, and each page's title is annotated with its group combination.

Requires **State Classification** to have been run for this cell type (it reads the behavioral-states `.h5ad`), even though it lives in the Track Classification tab.

| Control | Default | Meaning |
|---|---|---|
| **Tracks per page** | 6 | How many track rows to place on each page before starting a new one. |

Click **▶ Create Track Contact Overview**. Output: `track_contact_overview.pdf`, in `contact_analysis/contact_track_overview/<contact_col>/`.

### Exemplar Tracks

| Control | Default | Meaning |
|---|---|---|
| **Exemplars / cluster** | 10 | How many representative trajectories to show per cluster. |
| **Overview statebars** | on | Include the per-cluster state-bar overview pages. |
| **Backprojection PDFs** | off | Also render per-exemplar backprojection figures to PDF. |
| **Backprojection MP4** | off | Also render exemplar backprojection movies. |

Click **▶ Create Exemplar PDFs** to produce a PDF of representative tracks per cluster (optionally with state bars and backprojection figures/movies). The most important cluster-count QC file here is the overview PDF `example_tracks_overview.pdf`, which by default shows **10 exemplars per cluster** — see **How to judge whether `N clusters` is sensible** above. **👁** reopens the PDF.

## Backprojection

The final step paints the **trajectory clusters back onto the raw images**, so you can confirm that cells assigned to the same cluster really do behave alike. It is built directly into this sub-tab and works on the cell type selected at the top of Single Cell. It behaves exactly like State Backprojection, only it colours by trajectory cluster rather than per-timepoint state.

### Live overlay in napari

The **Live Napari Layer Backprojection** panel overlays coloured cluster labels onto the selected sample's image.

| Control | Default | Meaning |
|---|---|---|
| **Sample** | — All samples — | Which sample to overlay. **— All samples —** uses the first available sample. |
| **Color by** | `ClusterID` | The trajectory-cluster column to colour by: `ClusterID` (the current cluster labels), or `ClusterID_original` (the pre-rename labels, if you've renamed/merged clusters). |
| **Opacity** | 80 % | Opacity of the coloured overlay (10–100 %). |
| **Show trajectories** | on | Overlay each track's full path as a single-colored line matching its cluster's color — adds one napari Tracks layer per class. |

Click **▶ Show Track Backprojection in Napari** to load the overlay — this produces a napari layer only and writes nothing to disk.

### Export backprojection

Click **▶ Export Track Backprojection** to write the overlay to disk as a napari-openable `.zarr`, one per sample:

```text
<output_dir>/analysis/<cell_type>/behavioral_states/backprojection/<sample>_<cell_type>_track_clusters.zarr
```

The **Log** reports where the files were written. There are no PDF/MP4/DPI export options anymore — export always writes this zarr; open it directly in napari (drag-and-drop, or **File → Open**) to view or re-view it later without regenerating.

```{tip}
State Backprojection's export writes to the same folder with a different filename suffix (`..._behavioral_states.zarr`), so exporting both for the same sample and cell type keeps both files side by side.
```

## Train track classifier

Once you are happy with the clusters, you can train a classifier so the **same** trajectory clusters can be assigned to new data without re-running the full DTW clustering. This section has two groups: **Train RF Classifier on Named Clusters** and **Apply Classifier to New Data**.

| Button | What it does | Enabled when |
|---|---|---|
| **▶ Train RF Classifier** | Fits a random-forest classifier on the current named clusters. | Clustering has been run. |
| **▶ Apply Classifier** | Assigns clusters to the data using the trained classifier (browse to a classifier and its state data, or use the auto-filled paths). | A trained classifier exists. |

Both have **+🛒** (queue) and **👁** (view) buttons.

### ⚙ Advanced Configuration (Track Classifier)

| Control | Default | Meaning |
|---|---|---|
| **n_estimators** | 100 | Number of trees in the random forest. More trees = steadier but slower. |
| **Test holdout** | 0.20 | Fraction of trajectories held out to estimate classifier accuracy (0.05–0.5). |

## Outputs

Track Classification writes its results under:

```text
<output_dir>/analysis/<cell_type>/behavioral_trajectories/
```

You will find there, depending on which steps you ran:

- The **track-cluster data** as `BEHAV3D_<cell_type>_behavioral_trajectories.h5ad` (one trajectory per row, with its cluster label in the `ClusterID` column, `ClusterID_original` added once you've renamed). Every basis and method (`dtw`, `bouts`, Feature-based — including with the state-features preset — and legacy Feature-based) writes here — there's exactly one current result per cell type.
- A **track-cluster table** (`BEHAV3D_<cell_type>_track_clusters.csv`) once the classifier is applied.
- The trained **classifier** (`classification/track_classification_random_forest_<cell_type>.pkl`).
- The **exemplar overview PDF** at `example_tracks/example_tracks_overview.pdf`, which is the main visual QC output for deciding whether `N clusters` split the trajectories sensibly.
- Additional **exemplar** PDFs under `example_tracks/` and **diagnostics** PDFs under the trajectory-clustering output folders.
- **Feature Heatmap**: `feature_heatmaps/feature_heatmap.pdf` plus `csv/feature_heatmap_cluster_means.csv` and `csv/feature_heatmap_track_means.csv`.
- **Track-class proportion plots** under `behavior_proportions/`: `track_class_proportions_by_sample_<class>.pdf`/`.csv`, plus `track_class_proportions_by_group_<class>.csv` when grouping is used.
- **Condition comparison reports** under `behavior_comparisons/`: `condition_comparison_<condition>.pdf`/`.csv`.
- **Track Transition Report**: `transition_analysis.pdf` (pooled circular diagram + transition matrix for trajectory clusters).
- **Window Transitions**: `window_transitions_all_samples.pdf` (merged pooled + per-sample Sankey), with individual per-sample pages also kept under `sankey_pdf_pages/`.
- **Contact analysis** under `contact_analysis/`, one folder per report type, each holding a `<contact_col>/` (or, for the type comparison, a `<contact_col_a>_vs_<contact_col_b>/`) subfolder with that report's PDF plus a sibling `csv/` folder:
  - `contact_group_analysis/<contact_col>/`: `contact_analysis.pdf` (the bundled "Run Contact Analysis" report) plus `condition_comparison_<condition>.pdf`.
  - `contact_rate/<contact_col>/`: `contact_rate.pdf`.
  - `contact_composition/<contact_col>/`: `contact_composition.pdf`.
  - `contact_cluster_heatmap/<contact_col>/`: `contact_cluster_heatmap.pdf`.
  - `contact_duration_comparison/<contact_col>/`: `contact_duration_comparison.pdf` plus `csv/contact_duration_comparison.csv`.
  - `contact_track_overview/<contact_col>/`: `track_contact_overview.pdf` (no `csv/`).
  - `contact_type_comparison/<contact_col_a>_vs_<contact_col_b>/`: `contact_type_comparison.pdf` plus `csv/contact_type_comparison.csv`.
- Optionally the **DTW distance matrix** CSV (if you ticked *Save distance matrix CSV*).

```{note}
The legacy **Original BEHAV3D DTW** engine also writes to this same unified location and gets the full Rename → Reports/Plots → Backprojection → Train/Apply Classifier workflow like every other basis.
```

```{tip}
The folder name on disk is `behavioral_trajectories`. The easiest way to reopen any of these results is the **👁** buttons or the shared **Results** panel rather than browsing the path by hand.
```

## Tips & best practices

- **Run State Classification first** (for the default State-based method). It compares behavioural-state sequences, so it needs the per-timepoint states to exist for the cell type — or switch to **Feature-based** if you'd rather cluster on raw features without it.
- **Use the overview-PDF decision loop.** A good practical workflow is: fit clusters → inspect `example_tracks_overview.pdf` → rename and merge similar clusters → only then rerun with higher or lower *N clusters* if the split still looks too coarse or too fragmented.
- **Slight over-splitting is safer.** If you are unsure, it is usually better to split a bit too much and merge later during renaming than to force several distinct trajectory patterns into one cluster from the start.
- **Leave Linkage on `average`.** `complete` gives comparable results and is worth trying; **`single` rarely works well** for these distances. For the legacy engine specifically, agglomerative clustering is preferred over its default k-means, with the caveat that the resulting UMAP embedding can look poor even when the clusters themselves are sensible.
- **Leiden vs. agglomerative.** Leiden finds its own cluster count from the neighbourhood graph — there's no `N clusters` to tune, `Leiden resolution` takes its place instead (higher = more, smaller clusters). Agglomerative + `N clusters` gives more direct control over exactly how many clusters you end up with.
- **Use Variable-length mode for uneven tracks.** If your tracks differ a lot in length and resampling distorts them, switch *Trajectory size* to its minimum (Variable-length) so DTW compares native lengths.
- **Save the distance matrix only when you need it.** It grows with the square of the number of tracks and is rarely needed for routine analysis.
- **Queue the heavy steps.** DTW clustering and classifier training are CPU-intensive — use the **+🛒** buttons to run them unattended behind your other pipeline steps.
- **Turn on Divide long tracks when tracks run much longer than Trajectory size** and the part that would otherwise be discarded likely holds meaningfully different behaviour — you get more, shorter, independently classified trajectories instead of one truncated one. This is also a **prerequisite** for the Track Transition Report and Window Transitions below — without split windows every track has only one cluster label, so there's nothing to transition between.
- **Tune Min. contiguous contact bout to your imaging's time resolution.** A few timepoints is usually enough to exclude noisy single-frame touches while still catching genuine sustained contact; lower it toward 1 only if you want any touch, however brief, to count.
- **Cluster colors and order set during renaming persist automatically** into every later plot — proportions, condition comparisons, contact analysis, backprojection — so you don't need to reapply them per report.

## See also

- [State Classification](state_classification.md) — the per-timepoint behavioural-state workflow this builds on.
- [Feature Extraction](../feature_extraction.md) — computes the per-track features the clustering consumes.

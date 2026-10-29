# 🔬 Behavioral State

The first analysis of **📊 Analysis → 🧬 Single Cell** (*"What behaviour states do cells pass through?"*). It assigns every cell, at every timepoint, to a recurring **behavioural state**, in two stages:

1. A **Hidden Markov Model (HMM)** clusters the continuous features you choose — by default movement and morphology — into a handful of **intrinsic states**. An HMM models behaviour as a sequence, rather than judging each timepoint on its own.
2. The **binary (true/false) features** you also select, such as a contact or death flag, are layered on top without being fitted by the HMM. They split each intrinsic state into **full behavioural states**, for example *slow-moving & in contact* vs *slow-moving & not in contact*.

It runs on the cell type chosen in the dropdown at the top of [Single Cell](index.md) and reads that cell type's per-timepoint track-features table from [Feature Extraction](../feature_extraction.md) / [Filtering](../filtering.md).

## Workflow at a glance

1. [Train a new model, or reuse an existing one](#train-a-new-model-or-reuse-an-existing-one).
2. [Behavioral state clustering](#behavioral-state-clustering):
   1. [Select the features](#select-features-for-hmm-model-fitting) the HMM is fitted on (see the [default features](#default-features)).
   2. (Optional) [Process the features](#optional-process-features-before-hmm-fitting) (log scaling, smoothing, capping).
   3. [Select binary features](#assign-groups-based-on-binary-features-eg-contact-death) (e.g. contact, death) to split the states into full behavioural states.
   4. [Advanced configuration](#optional-advanced-configuration) (leave at its defaults unless you need it).
   5. [Choose a number of states](#choose-a-number-of-states) the model will attempt to find.
   6. [Run the clustering](#running-it).
3. [Merge and rename the behavioral states](#merge-and-rename-behavioral-states) with meaningful biological names:
   1. [Check the QC outputs](#check-the-qc-outputs) and, optionally, [backproject](#backprojection) the intermediate states.
   2. [Rename the intrinsic states first](#rename-intrinsic-states-first) as broad behaviours, in the [rename dialogs](#the-rename-dialogs).
   3. [Then review the full behavioral states](#then-review-the-full-behavioral-states) and [decide which binary flag takes precedence](#decide-which-binary-flag-takes-precedence).
   4. [Backproject](#backprojection) the classes you are unsure about and repeat steps 2–3 until the states hold up.
4. [Reports](#reports): composition, transitions, feature heatmap and condition comparison, plus [Contact Analysis](#contact-analysis).
5. [Backprojection](#backprojection): paint the states back onto the images to validate them, or [find tracks that show a given sequence of states](#find-a-behavior).

The settings page shows, from top to bottom: the **Apply existing behavioral state classification** checkbox, the **Behavioral state clustering** section (steps 2–3), the **Reports & Plots** card, the **Backprojection** box, and the progress bar and **Log**.

## Train a new model, or reuse an existing one

The first control on the settings page is the **Apply existing behavioral state classification** checkbox. It chooses what the panel below it shows:

| Checkbox | The panel shows | Use it to |
|---|---|---|
| Unticked (default) | the **Behavioral state clustering** section | fit a new model on the current cell type (see [Behavioral state clustering](#behavioral-state-clustering)). |
| Ticked | the **Apply saved HMM artifact** box, in place of the clustering section | apply a model trained in an earlier run. |

To reuse a saved model:

1. Tick **Apply existing behavioral state classification**.
2. In the **Apply saved HMM artifact** box, click **Browse…** and select the model file (`hmm_state_deployment_<cell_type>.pkl`, see [Outputs](#outputs)).
3. Click **▶ Apply saved HMM artifact**.



Applying reuses everything saved with the model (features, processing, binary groups, state names, colours and order), so states stay comparable across experiments. The new data must contain the same feature and binary columns; the Log reports an error if it does not.

## Behavioral state clustering

### Select features for HMM model fitting

The lists are built from the columns of the selected cell type's features table, so you only see features you actually computed. Two tabs show the number of selected features in their titles.

- **Timepoint Features** — one value per frame, read directly from the table. They are grouped in collapsible categories (**morphology**, **movement**, **intensity**, **death**, **contact**, **invasiveness**, **active_killing**, **other**), each showing *(n/total selected)*. True/false columns are not listed here; they go to the binary groups below. Pre-ticked in a new project when present: **speed**, **elongation**, **sphericity**, **extent** and **solidity**.
- **Window Features** — short-term track-shape descriptors computed here over a trailing window of consecutive timepoints (they are not columns of the features table):
  - **Window size (timepoints):** default `5` (1–500). The line below shows the equivalent real time from your metadata `time_interval`: 5 frames is 10 min at a 2-min interval but 25 s at a 5-s interval, so judge the duration, not the number. Window features always span at least 2 timepoints, and near the start of a track the window is shorter. Around 5 suits motility at typical live-imaging intervals.
  - **net_displacement** (ticked by default), **straightness**, **mean_square_displacement**.

The HMM still assigns one state **per timepoint**, whatever the window size; the window only affects how the window features are computed. For events that last a single frame (e.g. calcium peaks), rely on timepoint features and keep window features short or unticked.

```{tip}
Start small and biological. A handful of well-chosen features (e.g. speed, a shape descriptor and one window feature such as straightness) usually gives cleaner, more interpretable states than every available column.
```

### (Optional) Process features before HMM fitting

| Control | Default | What it does |
|---|---|---|
| **Log scaling** | `speed` ticked | One checkbox per selected timepoint feature; ticked features are log-transformed (`log1p`) before all features are standardised. **📊  Preview feature distributions** shows histograms of the selected features, so you can see which are skewed. |
| **Smooth window (timepoints):** | `5` (1–100) | Trailing rolling average of the timepoint features along each track; `1` = no smoothing. Damps single-frame errors (e.g. a cell that looks elongated for one frame because two masks merged) and makes state changes more gradual. |
| **Low percentile cap:** | `0.00` | Clips values below this quantile; `0` = off. |
| **High percentile cap:** | `1.00`  | Clips values above this quantile; `1` = off (e.g. `0.99` clips the top 1 %). |

```{tip}
**`speed` is the classic case for log scaling.** Most tracked objects barely move most of the time, with a few fast excursions. Untransformed, the rare large values dominate the clustering; log scaling compresses that tail. Check the histograms first and only log-scale features that are clearly right-skewed.
```

### Default features

By default these features are ticked (only if its column exists in your features table), and processed as follows:

| Feature | Tab | Measures | Smoothed (5 timepoints) | Log-scaled |
|---|---|---|---|---|
| `speed` | Timepoint Features | distance moved since the previous timepoint | ✓ | ✓ |
| `elongation` | Timepoint Features | how stretched the cell is | ✓ | – |
| `sphericity` | Timepoint Features | how close the cell is to a sphere | ✓ | – |
| `extent` | Timepoint Features | how much of its bounding box the cell fills | ✓ | – |
| `solidity` | Timepoint Features | how much of its convex hull the cell fills | ✓ | – |
| `net_displacement` | Window Features (window of 5 timepoints) | straight-line distance travelled over the window | – (already calculated over separate window) | – |

### Assign groups based on binary features (e.g. contact, death)

Contains checkboxes for the true/false columns of the features table (for example `dead`, a `<type>_contact` column or `is_active_killing`); none is ticked by default. They are **kept out of the HMM fit**, because clustering on binary columns tends to create artificial states, and instead **subdivide** each intrinsic state afterwards. A full state is named `<flag>_<intrinsic state>` by default, `<flagA>_and_<flagB>_<intrinsic state>` when several ticked flags are true, and `no_contact_<intrinsic state>` when none is.

```{tip}
**`any_organoid_contact` / `any_immune_cell_contact` or a per-type contact column?** The `any_*` columns are true when a cell touches any organoid or any immune cell, whatever its subtype (see [contact columns](../feature_extraction.md)). Use them when engagement itself is the behaviour and the partner's subtype should not matter. Use a per-type column when *which* population is contacted is part of the behaviour (e.g. T cell vs macrophage contact); you get more, smaller full states in exchange.
```

### (Optional) Advanced configuration

Most users never need to change these.

| Control | Default | What it does |
|---|---|---|
| **Start offset:** | `1` | Leading timepoints of each track left out of the fit. Keep it at least `1`: the first timepoint of a track has no real speed (there is no previous position), so with `0` every track would start in a "static" state. |
| **Skipped frames:** | `backfill` | `backfill` gives the skipped frames the first classified state; `leave_unassigned` leaves them without a state. |
| **Covariance type:** | `full` | Shape of each state's feature distribution (`full`, `diag`, `spherical`, `tied`). `full` is the most flexible; `diag` is lighter and more stable with many features or few cells. |
| **n_iter:** | `200` | Maximum training iterations. Increase if the Log reports that the model did not converge. |
| **tol:** | `0.001` | Training stops when the fit improves by less than this. |
| **min_covar:** | `0.001` | Floor on each state's variance; guards against degenerate states. |
| **Random seed:** | `123` | Fixes the random start, so re-running with identical settings gives identical states. |

### Choose a number of states

**n_states:** (default `4`, 2–50), just below the advanced section, is the number of intrinsic states the HMM fits: how many distinct behaviour types you expect the data to contain. There is no single correct value. Treat it as a starting guess that you refine, using one of the two routes below:


#### Route A: start large and merge
*Best when you are looking for rare or subtle behaviours that a smaller model would hide. However, this route is dependent on more curation in merging and renaming the detected states.*
1. Run with a generous **n_states** (e.g. `8`–`10`).
2. In the QC outputs, find pairs of states with very similar feature distributions and mean profiles.
3. Check those pairs in backprojection. If they look the same, merge them by giving them the same name in the [rename dialogs](#the-rename-dialogs).
4. Repeat until every remaining state is clearly distinct.



#### Route B: start small and split

*Best when you expect a few clear behaviours or need a quick first overview and don't want to spend a lot of time curating the results.*

1. Run with a low **n_states** (e.g. `3`–`4`).
2. Check the [QC outputs](#check-the-qc-outputs) and [backprojection](#backprojection).
3. If one state still mixes clearly different behaviours (e.g. *static* and *actively moving* cells together), raise **n_states** by one and run again.
4. Stop once every state shows a single behaviour.

### Running it

Once the feature lists are loaded, a summary box above the run button lists the selected timepoint features, smoothing, window features and size, log-scaled features, binary groups and quantile capping.

- **▶ Run State Clustering** fits the model in the background (see [Running Steps](../../plugin_essentials/running_steps.md#background-execution)). When it finishes, the model is applied to every timepoint of the cell type, the QC outputs are written and a dialog offers to **Open folder**.
- **+🛒** adds a **🔬 State Clustering** step to the [Processing Queue](../../plugin_essentials/processing_queue.md).
- **👁** reopens the run's QC PDFs (*Clustering diagnostics*, *Feature distributions*).

The settings of a successful run are saved per cell type in `behav3d_parameters.yml` and restored when you come back to that cell type.

```{warning}
Running **▶ Run State Clustering** again deletes the cell type's whole `behavioral_states` folder before writing the new results, including reports, contact analyses and exported backprojections. Copy anything you want to keep first.
```

### Merge and rename behavioral states

Freshly fitted states are numbered. In this step you decide which states are biologically distinct, merge the ones that are not, and give them names, colours and a display order that every report and backprojection use.

Curation works from coarse to fine, in this order:

1. [Check the QC outputs](#check-the-qc-outputs) to see how the states differ.
2. [Name the intrinsic states](#rename-intrinsic-states-first) as broad behaviours.
3. [Review the full behavioral states](#then-review-the-full-behavioral-states) that the binary flags create, and [decide which flag takes precedence](#decide-which-binary-flag-takes-precedence) when several are true together.
4. [Backproject](#backprojection) the classes you are unsure about, then go back to step 2 or 3 and rename or merge again until the states hold up.

#### Check the QC outputs

Inspect the QC outputs before the reports (**👁** next to the run button, or the [Diagnostic](#diagnostic) report):

- **Clustering diagnostics** (`behavioral_clustering_diagnostics.pdf`) — a heatmap of the selected features per timepoint, grouped by state; the learned transition matrix; and the correlation between states' mean feature profiles.
- **Feature distributions** (`behavioral_clustering_feature_distributions.pdf`) — each feature's distribution, overall and per state; the best place to decide whether two states truly differ.
- **Full state counts** (`behavioral_clustering_full_state_counts.pdf`) — the number of timepoints (cell-frames) in each full state.

For each state, ask:

1. **Does it differ on the features that matter** (often speed, shape, straightness, displacement or death-related measurements)? Compare it with the other states in the *Feature distributions* and the mean profiles in the *Clustering diagnostics*.
2. **Is the difference consistent**, or do the distributions overlap heavily? States that overlap heavily are the ones you will probably merge in the next steps.

#### Rename intrinsic states first

Open **✏  Rename intrinsic HMM behavioral states (e.g. morphodynamic)** and give each intrinsic state a broad morphodynamic name that the QC plots support, e.g. *static*, *scanning*, *plastic*, *migratory* or *rounded*. Keep it coarse at this level: aim for **biological interpretability** rather than keeping every statistically separable state. If two states would be described with the same sentence in a figure legend, give them the same name, which merges them.

Do this before touching the full states, for two reasons:

- **The names carry over.** Full states are named `<flag>_<intrinsic state>`, so naming the intrinsic states first turns e.g. `organoid_contact_2` into `organoid_contact_migratory`, which makes the next step much easier to read.
- **Renaming intrinsic states resets the full-state names.** Each save in the intrinsic dialog rebuilds the full states from the new intrinsic names, replacing any full-state names you had already set.

#### Then review the full behavioral states

Open **✏  Rename full behavioral states (with binary features)**. Each full state is an intrinsic state split by the binary flags; treat each split as a candidate subtype and ask:

1. **Can you see it in the images?** Check it in [backprojection](#backprojection), which colours every cell by its state in napari, so you can see whether the split is a real subtype.
2. **Does it change the interpretation** enough to deserve its own name?
3. **Is it a real behavioural niche?** Check the [full state counts](#check-the-qc-outputs): a split with only a few timepoints, or few compared with the others, is more likely a leftover of processing than a distinct behaviour.


Keep a distinct name where the split matters; otherwise merge it into the state it most resembles, by typing the same name or with **Combine selected →**.

##### Decide which binary flag takes precedence

When several flags can be true at once, you decide the naming hierarchy; BEHAV3D EXPLORER does not impose one. For example, a class that is both **dead** and **in organoid contact** and clearly looks dead in backprojection is best named simply *dead*. If such a class does **not** look dead (it may only pick up death signal from a dying organoid it touches), name it after what you see instead. Use the flags as clues, but let the QC plots and backprojection decide.

#### The rename dialogs

Two buttons open the rename dialogs. A status line above them shows how many intrinsic and full clusters the model has.

| Button | Opens | Enabled when |
|---|---|---|
| **✏  Rename intrinsic HMM behavioral states (e.g. morphodynamic)** | **Rename Primary Dynamic State Clusters** | A model exists. |
| **✏  Rename full behavioral states (with binary features)** | **Rename Full Behavioral Clusters (Binary Groups)** | A model exists. |

In the dialog, each row shows the **Current name**, a **New name** field and a **Color** swatch (click to pick a colour). Drag rows to set the display order. **Giving two clusters the same name merges them.** The full-state dialog also has a **Sel** checkbox per row and a **Combine selected →** field with a **Combine** button, which gives all ticked rows the typed name and the first one's colour.

**💾  Save** writes the names; the new labels are then applied to the full dataset and the diagnostics are refreshed automatically (watch the Log). **✖  Discard** closes without changes. **↺  Reset Names** restores the original labels after a confirmation; it cannot be undone.


## Reports

The **Reports & Plots** card below the clustering section (*"Turn your classified behavioral states into shareable figures."*) has a **Generate analysis and plots  ▸** button, greyed out until a clustering result exists. It opens a list of report cards; each card's **Start — open the settings  ▸** shows that report's settings only, and **‹ Back** steps back to the list, then to the clustering settings.

All reports use the current state names, colours and display order, so create them after renaming. The settings a report ran with are saved per cell type in `behav3d_parameters.yml` and restored when you come back. Each report has a **👁** button to reopen its PDF; all results also appear in the [📄 Results panel](../../plugin_essentials/results_panel.md). The grouping columns offered are `exp_nr`, `well`, the `*_line_condition` columns of your metadata, and `origin_cell_type` for cell-type groups.

### Diagnostic

**▶ Create Diagnostics** regenerates the [QC outputs](#check-the-qc-outputs) with your current state names, plus a bar chart of the number of timepoints per full state. They are also refreshed automatically after each run and each rename. Its **👁** lists all these PDFs.

### State Composition Report

How the share of each **full behavioural state** changes over time, per sample and per group.

| Control | Default | What it does |
|---|---|---|
| **Group in X:** / **Group in Y:** | `(none)` | Metadata columns whose levels form the columns / rows of a grid of composition panels. With neither set you get only the per-sample pages. |
| **Group conditions** (under each axis) | off | Pools that axis's levels into two custom groups, assigned in a two-list picker (e.g. 3 healthy vs 3 tumour lines). |
| **Group per page (Ctrl/Cmd click for multiple):** | none | Columns whose combinations each get their own page set instead of another grid axis. |
| **Time bin size:** | `1` | Groups timepoints into bins of this many frames before computing composition, to reduce noise; `1` = no binning. |

**▶ State Composition Report** writes one PDF with: composition over time per sample (proportions and absolute cell counts), overall composition per sample as horizontal bars, and, when groups are set, the same per group. It also writes the plotted values and a per-sample, per-state area-under-the-curve table as CSVs.

### State Transition Report

Which full behavioural states cells switch between, and how often. Settings are in its **⚙ Advanced Configuration** section:

| Control | Default | What it does |
|---|---|---|
| **Min probability cutoff:** | `0.0 %` | In the circular diagram, transitions below this probability are dropped, so rare transitions do not clutter the plot. |
| **Emphasis (gamma):** | `1.0` (0.5–5) | How fast line width and opacity fall off for weaker transitions; `1.0` is linear, 2–3 makes the strongest transitions stand out. |
| **Node labels:** | `Side legend` | State names written **On dots**, or in a numbered **Side legend** (better for long names or many states). |
| **Transition matrix heatmap** | on | Adds the transition matrix page. |
| **Circular transition diagram** | on | Adds the circular diagram pages. |
| **Include self-transitions** | off | Off: arrows show how a state's *switches* are split between the other states (no self-loops). On: staying in the same state is drawn as a loop beside each node and every arrow shows its share of all timepoints, which makes arrows thin for states that rarely switch. |
| **Show per-cluster breakdown grid** | on | Adds two pages with one small diagram per state (outgoing and incoming), useful when the overlaid diagram is too busy. |
| **N-gram rankings** | on | Adds rankings of the most common state sequences. |
| **Sankey diagrams (all state pairs)** | off | Adds Sankey diagrams of the paths cells take from each state to each other state. The slowest part of the report. |

**▶ State Transition Report** writes:

- `transition_matrix_heatmap.pdf` (opened by **👁**):
  - a 2 × 2 heatmap page of transition probabilities and counts, with and without self-transitions;
  - the circular diagrams, outgoing (*where does this state go next?*) and incoming (*where did its arrivals come from?*), each shown unfiltered next to the version at your cutoff;
  - the per-state grids;
  - the most frequent two- and three-state sequences, pooled and per start / end state;
- `sankey_all_pairs.pdf`, only when Sankey is ticked;
- the matrices and sequence counts as CSVs.

```{tip}
**Circular diagram or Sankey?** The circular diagram shows only direct, one-step neighbours: for a state named *killer*, the state cells were in just before. The Sankey diagrams show the full routes (e.g. *static* → *scanner* → *engaging* → *killer*). Use the circular diagram for direct relationships at a glance and Sankey to trace how cells actually got somewhere.
```

### State Feature Heatmap

A heatmap of states × features: the mean of each selected per-timepoint feature in each state. Cell labels show the unscaled mean.

| Control | Default | What it does |
|---|---|---|
| **States:** | `full_behavioral_state` | `full_behavioral_state`: final states including the binary groups. `hmm_intrinsic_behavioral_state`: the HMM states before the binary split. |
| **Average:** | `Every timepoint counts` | `Each track counts once`: each track is first averaged within a state, then tracks are averaged, so long tracks do not dominate. |
| **Colour scaling:** | `z-score` | `z-score`: each feature standardised (0 = average, red = higher, blue = lower). `min-max`: state means rescaled 0–1 per feature. |
| Feature lists | the features the states were built on | Move features from **Available features** to **Selected features** with **Add ▼** / **▲ Remove** (or double-click); drag selected rows to set their order in the plot. **Select used features**, **Select all** and **Clear** act in bulk. |

**▶ Create State Feature Heatmap** writes the heatmap PDF and a CSV of the state means.

### Condition Comparison Report

Tests whether the proportions of the full behavioural states differ between the levels of one metadata column (e.g. organoid lines).

| Control | Default | What it does |
|---|---|---|
| **Compare condition:** | first column | Column whose levels are compared pairwise. |
| **Group in X:** + **Group conditions** | `(none)` | Splits the comparison into side-by-side columns by a second column, optionally pooling its levels into two groups. |
| **Group in Y:** | — | Read-only; shows the compared column (one row per pairwise comparison). |
| **Group conditions** | off | Pools the compared column's levels into two custom groups and runs a single comparison between them. |
| **Group per page (Ctrl/Cmd click for multiple):** | none | Columns that paginate the report. |

**▶ Condition Comparison Report** writes one PDF and its CSVs:

- **Proportion difference per state** — a signed bar for each pair of levels, with Welch's t-test stars (`*` to `****`), using each sample as a replicate.
- **Prevalence over time** — each state's prevalence for each level, as lines with a standard-error band. Descriptive only, with no test; the underlying values are saved as CSVs so you can test them yourself.

## Contact Analysis

Found as the **Contact analysis** card in the report list. It needs only this classification and the features table, not State Trajectory. It compares the **state composition of contacting timepoints** between contact columns, e.g. *for T cells touching organoids, does the mix of behaviours differ between healthy and tumour organoids?* This is the statistically tested counterpart of State Trajectory's descriptive [Contact Type Comparison](track_classification.md#contact-type-comparison).

| Control | Default | What it does |
|---|---|---|
| **Group per page** list | none | Splits the report into one page set per group. |
| **State column:** | `full_behavioral_state` | `full_behavioral_state`, `hmm_intrinsic_behavioral_state` or `raw_hmm_state` (original numbered states). |
| **Contact columns to compare (Ctrl/Cmd click for 2+):** | none | The contact columns of the features table; select at least two. |

**▶ Create Contact Type Comparison** (enabled when the table has at least two contact columns) writes a PDF with per-state differences between contact columns (Welch's t-test, one row per pair) and the stacked state composition per contact column, plus two CSVs. Each combination of columns gets its own folder, so different comparisons do not overwrite each other.

## Backprojection

The **Backprojection** box paints the states **back onto the images**, so you can check that the labels match what the cells are doing. This is the final validation step of the curation workflow. For a cell-type group whose tracked segments have not been built, the box is disabled and a warning explains how to build them.

To colour cells by any feature instead of by state, use [Feature Backprojection](../feature_backprojection.md).

### Live overlay in napari

The **Live Napari Layer Backprojection** box:

| Control | Default | What it does |
|---|---|---|
| **Sample:** | `— All samples —` | Sample to show; `— All samples —` shows the first sample. |
| **Color by:** | `full_behavioral_state` | `full_behavioral_state` (final, renamed states), `hmm_intrinsic_behavioral_state` (renamed HMM states) or `raw_hmm_state` (original numbered HMM states). |
| **Opacity:** | `80 %` (10–100 %) | Opacity of the state overlay; you can change it later in napari's layer controls. |
| **Trajectories:** **Show trajectories** | on | Adds a tracks layer drawing each track's path, coloured by its state at each timepoint. |

**▶ Show State Backprojection in Napari** clears the viewer and loads the sample's raw channels, a state overlay on the tracked cells (named after **Color by:**) and the trajectories. It also opens a **State Class Mapping** legend (state name ↔ colour). The overlay is computed for the frame you are on and updates as you move the time slider. Nothing is written to disk. Contrast and colormap changes you make to the raw channels are remembered for the next time.

### Export backprojection

**▶ Export State Backprojection** writes the overlay (coloured by **Color by:**) as a napari-openable `.zarr` for the selected sample, or for every sample with `— All samples —`:

```text
<output_dir>/analysis/<cell_type>/behavioral_states/backprojection/<sample>_<cell_type>_behavioral_states.zarr
```

The Log reports how many samples were written and why any were skipped. Open the file in napari (drag-and-drop or **File → Open**) to view it again without recomputing. State Trajectory exports to the same folder with the suffix `_track_clusters.zarr`, so both can sit side by side.

### Find a Behavior

The **🔎 Find a Behavior** box allows you to supply a sequence of states (e.g. static > scanning > engager). It searches the whole dataset for tracks whose trajectories contains that specific sequence of states you build, allows you to select a track of interest and allows backprojection of that specific trajectory on the data

**Build the sequence**

| Control | What it does |
|---|---|
| **🔄 Load States** | Fills the state dropdown with the labels of the current **Color by:** choice. |
| **➕ Add** | Appends the selected state as the next step. |
| **⬆** / **⬇** | Moves the selected step earlier / later. |
| **✕ Remove** / **🗑 Clear** | Removes the selected step / the whole sequence. |

**Match options** (all off by default)

- **Require no other state in between (strict)**: The provided sequence must occur in the track without any other states inbetween (other states can occur before/after the provided sequence)
- **Sequence must start the track** / **Sequence must end the track**: Matching tracks must start/end with the first/last state in the provided sequence.

**🔎 Find Matching Tracks** lists every match with its sample, TrackID, matched time range, track length and a small coloured bar of the track's full state history.

Select a row and click **▶ Visualize This Track**. The viewer is cleared and shows only that track: the raw channels, its mask coloured by state, its trajectory, the **State Class Mapping** legend, and a **State timeline** bar below the viewer with the track's state history and a time cursor. Click or drag on the bar to jump to that timepoint. The viewer opens at the start of the match.


```{tip}
Use backprojection as the final arbiter:

- scrub through time and check that colour changes coincide with visible behaviour changes;
- inspect questionable **full states one by one**;
- check that clusters you intend to merge really look alike;
- decide whether a binary combination is a genuine subtype or an artefact of neighbouring signal or thresholding.

If the PDFs suggest a difference but the images do not, revisit the naming or merging before treating the classes as distinct.
```

## Outputs

Everything is written under the cell type's `behavioral_states` folder (see [Output Layout](../../plugin_essentials/output_layout.md)):

```text
<output_dir>/analysis/<cell_type>/behavioral_states/
├── BEHAV3D_<cell_type>_behavioral_states.h5ad    # classified data, every timepoint
├── BEHAV3D_<cell_type>_behavioral_states.csv     # the same labels as a table
├── processing/
│   ├── BEHAV3D_<cell_type>_behavioral_states_modeldata.h5ad
│   └── hmm_behavioral_classification/
│       ├── hmm_state_deployment_<cell_type>.pkl  # saved model, for "Apply existing…"
│       └── quality_control/
│           ├── raw/                              # QC written by the run (numbered states)
│           └── *.pdf, *.csv                      # QC with your current names (Diagnostic)
├── state_composition/
│   ├── state_composition_report.pdf / .csv / _auc.csv
│   └── behavior_proportions/condition_comparison_<column>.pdf / .csv / _over_time_*.csv
├── state_transitions/
│   ├── transition_matrix/transition_matrix_heatmap.pdf  (+ data/ CSVs)
│   └── sankey_all_pairs.pdf                      # only with Sankey ticked
├── state_feature_heatmaps/state_feature_heatmap_<state column>.pdf (+ _cluster_means.csv)
├── contact_analysis/contact_type_comparison/<columns joined by _vs_>/contact_type_comparison.pdf (+ csv/)
└── backprojection/<sample>_<cell_type>_behavioral_states.zarr
```

The classified table has, for every cell and timepoint (`sample_name`, `TrackID`, `position_t`), its `hmm_intrinsic_behavioral_state` and `full_behavioral_state` (with your names), and `..._confidence` columns giving the model's probability for the assigned state. The QC folders also hold the per-state feature means, state counts, transition matrix and state-correlation CSVs.

## Tips & best practices

- **Choose features deliberately.** States are only as meaningful as the features you feed in; begin small and expand only if states are blurry.
- **Treat curation as iterative.** Fit the model, then check each state in the diagnostic PDFs and in backprojection. Rename the intrinsic states, review the resulting full states, backproject any that still look doubtful, and rename or merge again until every state means something clear in both the plots and the images.
- **Keep the seed fixed** (or note it) so re-runs and collaborators get the same states.
- **Reuse a trusted model** on new experiments with **Apply existing behavioral state classification**, so states stay comparable.
- **Let images overrule labels.** If a name sounds convincing in the PDFs but not in backprojection, trust the images.

## See also

- [Feature Extraction](../feature_extraction.md) — computes the per-timepoint features used here.
- [Filtering](../filtering.md) — remove short or low-quality tracks before classifying.
- [Feature Backprojection](../feature_backprojection.md) — colour cells by any single feature.
- [🛤️ State Trajectory](track_classification.md) — groups whole tracks by how their states unfold over time.

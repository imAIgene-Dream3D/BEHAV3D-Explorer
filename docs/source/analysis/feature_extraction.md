# 🧪 Feature Extraction

Tab 5 of the BEHAV3D EXPLORER dock widget. Feature Extraction takes your **tracked segments** plus the **raw image** and computes, for every cell at every timepoint, a panel of biological measurements (movement, intensity, morphology, contact with neighbours, death). The result is the table that [Filtering](filtering.md) — and through it every downstream analysis — reads from.

## What this tab produces

For each cell type, Feature Extraction writes one combined table:

```
<output_dir>/analysis/<cell_type>/track_features/
    BEHAV3D_<cell_type>_combined_track_features.csv
```

Each row = one cell (`TrackID`) at one timepoint (`position_t`); the columns are grouped into the feature families [below](#the-six-feature-families). Per-sample tables are also written to `trackdata/<sample>/<cell_type>/`, together with an `intermediate_results/` folder (intensity, contact, morphology and dead-mask measurements) that is reused on later runs.

The main run writes CSVs only — no plots. The only graphics produced on this tab are the live [dead-threshold preview](#preview-dead-threshold-in-viewer) and the optional [Active Killing](population_analysis/active_killing.md) outputs. Like the other processing tabs, the lower half of the tab is the shared [📄 Results panel](../plugin_essentials/results_panel.md), where the new CSVs appear as soon as a run finishes.

## Per-cell-type sub-tabs

Feature Extraction is organised as **sub-tabs per cell type** along the left side, colour-coded by category:

| Icon | Category |
|---|---|
| 🟣 | Organoid |
| 🔵 | Immune |
| 🟡 | Other |

Multicolor channel splits (e.g. `tcell_1_multicolor`) are **not** shown as separate sub-tabs — the merged track set is used instead.

Each sub-tab has, from top to bottom: **Features to extract**, **Death Threshold** (only when the metadata declares a dead channel), **⚙ Advanced Configuration** (same condition), **Workers**, the **Apply to all …** buttons and the run button.

## The six feature families

Each sub-tab lists the feature families as checkboxes. Families that downstream analyses depend on are **always extracted** for that cell type: their checkbox is ticked and greyed out (*"Greyed-out features are always extracted for this cell type."*). On a fresh project the optional families are ticked too.

| Family | What it measures | Always extracted for |
|---|---|---|
| **Contact** | Which neighbouring cells (of any cell type) each cell touches or is close to | All cell types |
| **Invasiveness** | How much of an immune cell's surface is engaged with an organoid (0–100 %) | Optional; **immune sub-tabs only**, shown next to Contact |
| **Movement** | Step length, distance travelled, mean square displacement, speed, directional persistence | Immune, Other |
| **Intensity** | Per-channel intensity inside each cell's mask, plus self-normalised fold changes | All cell types |
| **Morphology** | Volume, shape descriptors, surface area, principal axes, orientation | Optional |
| **Death** | Share of the cell's mask covered by the dead mask, and an alive/dead flag | All cell types — **only listed when a dead channel is declared** |

### Per-category summary

| Cell type | Always on (greyed) | Optional |
|---|---|---|
| Organoid | Contact, Intensity (+ Death if dead channel) | Movement, Morphology |
| Immune | Contact, Movement, Intensity (+ Death if dead channel) | Invasiveness, Morphology |
| Other | Contact, Movement, Intensity (+ Death if dead channel) | Morphology |

## Parameters

| Control | Default | What it does |
|---|---|---|
| **Contact units:** `px` ⇄ `µm` | `µm` | Display unit of the contact threshold. Contact is always computed in µm; a value typed in px is converted with the sample's voxel size. Disabled when the metadata has no pixel size. |
| Contact threshold (number box next to **Contact**) | `0.00` µm | Distance within which another cell counts as "in contact" for the `*_contact_on_distance` and `touching_*` columns. Also defines the neighbourhood used by **Invasiveness** and by Active Killing. |
| **Dead mask % threshold:** | `10.0 %` | Share of a cell's voxels that must overlap the dead mask for that timepoint to count as dead. Range 0–100 %, saved to `behav3d_parameters.yml` as a fraction (10 % → `0.1`). Set to **0** to skip the alive/dead classification (no `dead` column). *Only with a dead channel.* |
| **🔄  Re-run death** | — | Re-classifies alive/dead in the existing combined table with the current threshold, without re-measuring anything. Visible once the combined CSV exists; active only when the threshold differs from the one used. See [Changing the dead threshold later](#changing-the-dead-threshold-later). |
| **Preview sample:** | first sample | Sample loaded by the preview button below. |
| **👁  Preview Dead Threshold in Viewer** | — | Green/red alive/dead overlay — see [below](#preview-dead-threshold-in-viewer). |
| **⚙ Advanced Configuration → Propagate dead signal** | `on` for organoids, `off` for immune / other | **On:** once a track crosses the threshold it stays dead for the rest of the track. **Off:** only timepoints at/above threshold are dead, so the cell can return to alive — appropriate for immune cells that transiently pick up dead dye while killing. *Only with a dead channel.* |
| **Workers:** | half the CPU cores (at least 4), never more than cores − 1 | Parallel workers for the image-based measurements (morphology, intensity, contact, dead mask). Range 1 – (cores − 1). Set per cell type, not linked to the shared [Workers](../plugin_essentials/running_steps.md#workers) setting of the other tabs. |

```{important}
The **Dead mask % threshold** is **shared across all organoid sub-tabs** (*"One threshold applies to ALL organoid types equally"*): changing it on one organoid sub-tab updates every other organoid sub-tab. Immune and Other cell types each have their own threshold.
```

## Preview Dead Threshold in Viewer

**👁  Preview Dead Threshold in Viewer** is the quickest way to pick a sensible dead threshold. It:

1. Clears the napari viewer (after asking for confirmation if layers are open).
2. Loads, for the **Preview sample**, the raw channels, the dead mask and the tracked segments of every cell type.
3. Colours each cell of the current sub-tab's cell type by its dead-mask overlap at the current timepoint:

| Colour | Meaning |
|---|---|
| 🟢 Green | **Alive** — overlap **below** the threshold |
| 🔴 Red | **Dead** — overlap **at or above** the threshold |

The overlay updates live when you change **Dead mask % threshold** or move the time slider. Hover over a cell to see its track ID (plus its cell type on organoid sub-tabs) and exact dead percentage in the status bar. On organoid sub-tabs all organoid types are merged into a single overlay.

```{note}
For organoid and other (non-immune) cell types, dead-mask voxels that lie inside **immune cells** are ignored — both in the preview and in the real extraction — so dye-positive immune cells crawling over or into an organoid do not make the organoid look dead.
```

```{tip}
Starting points (fraction of the mask that must overlap the dead signal):

- **Organoids ≈ 5 %**. It sounds low, but organoids are large 3-D objects, so a few dead cells are only a small volume fraction.
- **T cells ≈ 10 %** (the default). With a clean dead-dye signal there is a clear central red dot.

Adjust until the green/red split matches the dead cells you see in the raw image at a timepoint you trust.
```

### Changing the dead threshold later

The measured overlap is stored next to the alive/dead flag, so you never need to re-measure to try another threshold. Change **Dead mask % threshold** and click **🔄  Re-run death** (on organoid sub-tabs this re-runs all organoid types, since they share one threshold). The current **Propagate dead signal** setting is applied too. A reminder then asks you to re-run [Filtering](filtering.md): every analysis reads the *filtered* table, which still holds the old classification until you do.

## Active Killing

The collapsible **🧬  Extended Analysis — Active Killing (Immune Cells)** section below the batch button appears when the metadata has at least one immune cell type. It detects, for each immune cell, the contacts after which the target's death signal rises enough to count as a kill. It is configured and run here but explained with the population analyses: **[Active Killing](population_analysis/active_killing.md)**. Run baseline Feature Extraction for the immune type first.

## Apply, Run, Queue

Each sub-tab has:

- **Apply to all <Category>** (e.g. **Apply to all Organoids**, **Apply to all Other types**) — copies this sub-tab's feature checkboxes, contact threshold, workers, **Propagate dead signal** and (between non-organoid sub-tabs) dead threshold to every sub-tab of the same category.
- **Apply to all** — the same across all cell types.
- **Run <Cell type> Feature Extraction** — runs this cell type for every sample.

Below the sub-tabs:

- **Run Batch Feature Extraction (All Cell Types)** — runs every cell type in turn.
- **+🛒** — adds a Feature Extraction step to the [Processing Queue](../plugin_essentials/processing_queue.md).

A progress bar and a **Log** box report progress (see [Running Steps](../plugin_essentials/running_steps.md#background-execution)). If results already exist, the **Existing Feature Data** dialog offers the usual choices (see [When results already exist](../plugin_essentials/running_steps.md#when-results-already-exist)). Here **Overwrite All** also discards the cached intermediate measurements, and when a dead threshold changed an extra **Re-run death only** button (batch: **Re-run death only (changed thresholds)**) re-classifies alive/dead without recomputing anything else.

When the run finishes, a **Feature Extraction Complete** reminder offers **Run Filtering with current setup**, **Go to Filtering tab** or **OK** (tick **Don't remind me this session** to silence it).

## Column reference

The combined CSV always contains a base set of identity columns, plus whatever the enabled feature families add.

### Identity / time columns (always present)

| Column | Meaning |
|---|---|
| `sample_name` | Sample identifier |
| `TrackID` | Stable track identifier across timepoints |
| `SegmentID` | Per-timepoint segment label (empty on interpolated rows) |
| `position_t` | Timepoint index (0-based) |
| `position_x` / `_y` / `_z` | Centroid in µm |
| `relative_time` | Time within this track, starting at 1 |
| `time` | Real time in hours |
| `distance_unit`, `time_unit` | Always `"um"` / `"h"` |
| `interpolated` | `True` if this row was filled in to bridge a missing frame |
| `nr_pixels`, `volume` | Voxel count and physical volume (µm³) — always computed, even without Morphology |

Missing frames inside a track are filled in before the movement features are computed: numeric values are interpolated linearly, flags and text are copied from the previous timepoint. Any extra columns in your tracks CSV (e.g. `well`, `condition`) are carried over.

### Movement

All measurements are in µm and hours.

| Column | Meaning |
|---|---|
| `displacement` | Step length from the previous timepoint (µm) |
| `cumulative_displacement` | Running total of step lengths up to this timepoint |
| `displacement_from_origin` | Straight-line distance from the track's first position |
| `mean_square_displacement` | Mean squared distance between the current position and every earlier position on the track |
| `directional_persistence` | Cosine of the angle between consecutive steps (−1 = U-turn, 0 = right-angle turn, +1 = straight ahead) |
| `speed` | Step length divided by the frame interval (µm/h) |

#### How the movement features are computed

Let $\mathbf{p}(t) = (x, y, z)$ be a cell's centroid at timepoint $t$ (in µm), $\Delta t$ the frame interval (in hours), and $\mathbf{s}(t) = \mathbf{p}(t) - \mathbf{p}(t-1)$ the step vector between consecutive frames.

$$
\text{displacement}(t) = \lVert \mathbf{s}(t) \rVert
\qquad
\text{speed}(t) = \frac{\lVert \mathbf{s}(t) \rVert}{\Delta t}
$$

$$
\begin{aligned}
\text{cumulative\_displacement}(t) &= \sum_{i \le t} \lVert \mathbf{s}(i) \rVert \\[4pt]
\text{displacement\_from\_origin}(t) &= \lVert \mathbf{p}(t) - \mathbf{p}(0) \rVert
\end{aligned}
$$

The **mean square displacement** at timepoint $t$ averages the squared distance from the current position to *every* position up to and including $t$:

$$
\text{MSD}(t) = \frac{1}{t+1} \sum_{j=0}^{t} \lVert \mathbf{p}(j) - \mathbf{p}(t) \rVert^{2}
$$

MSD grows roughly linearly in time for random (diffusive) motion and quadratically for directed motion, so its shape distinguishes wandering cells from cells under a directional cue.

**Directional persistence** is the cosine similarity of successive step vectors (0 wherever a step has zero length, and at the first two timepoints of a track):

$$
\text{persistence}(t) = \frac{\mathbf{s}(t-1) \cdot \mathbf{s}(t)}{\lVert \mathbf{s}(t-1) \rVert \, \lVert \mathbf{s}(t) \rVert}
$$

A value near +1 means the cell keeps heading the same way (persistent, directed migration); near 0 means each step's direction is independent of the last (random walk); near −1 means it reverses.

```{note}
Windowed motility statistics (net displacement, straightness and MSD over a sliding window) and the full-tracklet net-displacement descriptors are **not** columns of this table. They are computed from it later, in the [Single Cell](single_cell/index.md) analyses ([Behavioral State](single_cell/state_classification.md) and [State Trajectory](single_cell/track_classification.md)).
```

### Intensity

| Column pattern | Meaning |
|---|---|
| `mean_intensity_ch<N>` | Mean intensity of channel *N* inside the cell's mask |
| `mean_dead_dye` | The dead channel's mean intensity (replaces its `mean_intensity_ch<N>` column when the metadata declares a dead channel) |
| `mean_intensity_ch<N>_fold_change` | Mean intensity divided by the track's own baseline (its 10th-percentile value over time), so a cell that never changes stays ≈ 1 |
| `q75_mean_intensity_ch<N>`, `q90_mean_intensity_ch<N>` | Mean of the brightest 25 % / 10 % of the cell's voxels — follows a bright sub-region (e.g. a granule or local flash) that the whole-cell mean dilutes |
| `q75_…_fold_change`, `q90_…_fold_change` | The above divided by the track's own low baseline (median of its 5 lowest values) |
| `q75_…_fold_change_prevframe`, `q90_…_fold_change_prevframe` | Ratio to the previous timepoint of the same track (the denominator never drops below the track's low baseline) |
| `q75_…_flicker_score`, `q90_…_flicker_score` | The previous-frame ratio clipped to 1–3: 1 = no rise, 3 = a jump to three-fold or more — a bounded score for brief flashes |

The fold-change columns compare each cell with itself, which makes cells of different absolute brightness comparable and is the natural choice for fluctuating reporters (e.g. calcium).

### Morphology

When Morphology is on, the following are added. Units: volumes in µm³, lengths in µm, surface area in µm².

| Column | Meaning |
|---|---|
| `bbox_volume` | Volume of the axis-aligned bounding box |
| `equivalent_diameter` | Diameter of a sphere with the same volume |
| `axis1_length` / `axis2_length` / `axis3_length` | Longest / middle / shortest principal-axis length (spacing-aware) |
| `major_axis_length`, `minor_axis_length` | The longest (`axis1`) and shortest (`axis3`) principal axis |
| `elongation` | `axis1_length / axis2_length` — longest over middle axis |
| `extent` | Volume divided by the **oriented** bounding-box volume (`axis1 × axis2 × axis3`) |
| `oblateness` | Flatness: `1 − axis3/axis1` (0 = not flattened, →1 = disc-like) |
| `prolateness` | Elongation: `1 − axis2/axis1` (0 = not elongated, →1 = cigar-like) |
| `surface_area` | Physical surface area from exposed voxel faces (µm²) |
| `sphericity` | How sphere-like the object is (1 = perfect sphere) |
| `surface_to_volume_ratio` | `surface_area / volume` |
| `convex_volume` | Volume of the convex hull |
| `solidity` | `volume / convex_volume` — how much the object fills its convex hull (1 = convex) |
| `orientation_vector` | Unit vector of the longest principal axis (the cell's 3-D orientation) |
| `border_touching_segment` | `True` when the mask touches the edge of the image volume — border objects are cropped, so their morphology is unreliable |

#### How the morphology features are computed

Let $V$ be the object's physical volume and $A$ its surface area. **Surface area** is measured directly from the voxelised mask by summing the physical area of every voxel face between the object and the background (no smoothed mesh). **Sphericity** compares the object's surface area to that of a sphere of equal volume:

$$
\begin{aligned}
\Psi &= \frac{\pi^{1/3}\,(6V)^{2/3}}{A} \\[4pt]
\text{surface\_to\_volume\_ratio} &= \frac{A}{V} \\[4pt]
\text{equivalent\_diameter} &= \left(\frac{6V}{\pi}\right)^{1/3}
\end{aligned}
$$

$\Psi = 1$ for a perfect sphere and decreases as the object becomes more irregular or elongated.

The three **principal axes** ($a \ge b \ge c$ = `axis1` ≥ `axis2` ≥ `axis3`) come from a principal-component analysis of the mask's voxel coordinates in physical units. The shape descriptors are then:

$$
\begin{aligned}
\text{elongation} &= \frac{a}{b} & \qquad \text{oblateness} &= 1 - \frac{c}{a} \\[4pt]
\text{prolateness} &= 1 - \frac{b}{a} & \qquad \text{extent} &= \frac{V}{a\,b\,c}
\end{aligned}
$$

**Solidity** ($V / V_{\text{convex}}$) measures how much of the object's convex hull it actually fills — low solidity flags protrusions or concavities.

### Contact

For every cell type in the metadata — including the cell's own type, i.e. contact with *other* cells of the same type:

| Column pattern | Meaning |
|---|---|
| `<type>_contact` | `True` if this cell is **touching** at least one `<type>` cell (masks within one voxel diagonal, ≈ 1.73 px) |
| `<type>_contact_on_distance` | `True` if at least one `<type>` cell is within the **contact threshold** |
| `touching_<type>s` | Comma-separated `TrackID`s of the `<type>` cells within the contact threshold |
| `any_organoid_contact` / `any_organoid_contact_on_distance` | Aggregated over every organoid type |
| `any_immune_cell_contact` / `any_immune_cell_contact_on_distance` | Aggregated over every immune type |

```{tip}
`*_contact` answers **"is it touching?"** and `*_contact_on_distance` answers **"is it close enough to interact?"**. With the default threshold of 0 µm the two are nearly equivalent; raise it to 1–5 µm to capture near-contacts between cells that don't physically touch.

Changing the contact threshold changes the contact columns themselves, so it needs a new extraction with **Overwrite All** (Skip / Resume would reuse the cached contact measurements). Plan it as a deliberate re-run rather than interactive trial and error.
```

### Invasiveness (immune only)

When ticked, for each organoid type `<org>`:

| Column | Meaning |
|---|---|
| `<org>_invasiveness_perc` | Of the voxels within 2 µm of the immune cell's boundary (its "surface"), the percentage that lie within the contact threshold of an `<org>` cell |
| `<org>_invasiveness` | `True` when `<org>_invasiveness_perc ≥ 50 %` |
| `any_org_invasiveness_perc` | Maximum across all organoid types |
| `any_org_invasiveness` | `True` when the cell is invasive for any organoid type |

This is a **surface-engagement fraction** — how much of the immune cell's boundary is wrapped by organoid tissue — rather than a count of voxels inside the organoid. It is analysed in [Invasiveness](population_analysis/invasiveness.md).

### Death

Requires a dead channel in the metadata and the dead mask from [Segmentation](../processing/segmentation/index.md).

| Column | Meaning |
|---|---|
| `percentage_dead_mask` | Fraction (0–1) of the cell's voxels that overlap the dead mask |
| `nr_dead_mask_pixels` | Number of overlapping voxels = `nr_pixels × percentage_dead_mask` |
| `increase_dead_mask` | Change in `nr_dead_mask_pixels` compared with the same cell 10 timepoints earlier — a simple death-rate proxy |
| `smoothed_percentage_dead_mask`, `smoothed_nr_dead_mask_pixels`, `smoothed_increase_dead_mask` | Rolling means of the three signals over ≈ 10 minutes (unsmoothed copies when frames are more than 10 min apart), used by Death Dynamics to damp single-frame noise |
| `dead` | Alive/dead flag: `percentage_dead_mask` ≥ **Dead mask % threshold**. With **Propagate dead signal** on, the flag stays `True` from the first crossing onward. Not written when the threshold is 0 |

$$
\text{percentage\_dead\_mask} = \frac{\#\{\text{cell voxels overlapping the dead mask}\}}{\#\{\text{cell voxels}\}}
$$

The death signals capture death differently: `percentage_dead_mask` is **size-independent** (good when objects differ in size) but saturates; `nr_dead_mask_pixels` is an **absolute** count (pairs well with an absolute threshold — see [Active Killing](population_analysis/active_killing.md)); and `mean_dead_dye` (Intensity family) is the **mean dead-channel intensity** over the whole mask — the right choice when there is no dead-cell segmentation, or when the dye is diffuse rather than forming a discrete region.

```{note}
**The dead channel does not have to carry a death dye.** Any reporter that switches on and stays on works here; read "dead" as "past the threshold you set". A fluctuating reporter such as calcium does not belong in `dead_channel` (see [Population Dynamics](population_analysis/index.md)).
```

## Tips & best practices

- **Always use the dead-threshold preview before running.** The green/red overlay catches a wrong threshold much faster than inspecting a CSV.
- **Use Apply to all <Category> for multi-type experiments**, so no sub-tab is left with different settings.
- **Workers speed up the image measurements only.** If a run stops with an out-of-memory error, lower **Workers** and run again — finished samples and cached measurements are reused.
- **Check the Log for "skipping" messages.** A sample is skipped (with a warning) when its tracked image or tracks CSV is missing, or when Death is requested but the sample has no dead mask.
- **Re-run the downstream steps after a new extraction.** A new combined table makes existing Filtering (and Active Killing) results stale; re-run them in that order.

## See also

- [Filtering](filtering.md) — the next step: drops low-quality tracks and produces QC plots.
- [Population Dynamics](population_analysis/index.md) — death dynamics, interaction, invasiveness and active killing from the contact and death columns.
- [Single Cell](single_cell/index.md) — classifies per-cell behavioural states from this table.
- [Output Directory & File Layout](../plugin_essentials/output_layout.md) — where the CSVs live.

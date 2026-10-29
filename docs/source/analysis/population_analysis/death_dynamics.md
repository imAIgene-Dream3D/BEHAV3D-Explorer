# 💀 Death Dynamics

Quantifies how the measured signal progresses across the selected target population(s) over time, using the `dead` flag and the dead-mask signals computed during [Feature Extraction](../feature_extraction.md). If your `dead_channel` carries a reporter other than a death dye, read every "dead" below as "past the threshold you set".

The headline curve is the **cumulative percentage of dead targets at each timepoint** — the number of targets flagged dead *at or before* that timepoint, out of every target tracked in that sample. Once a target is counted dead it stays counted, so this curve never goes down. The death-signal traces (`percentage_dead_mask`, `nr_dead_mask_pixels`, `mean_dead_dye`) are also shown **baseline-normalised** — shifted so each starts at 0 — so an already-dead baseline doesn't skew the trend.

Open it with **Start** on the **Death Dynamics** card, tick your targets in **🎯 Target cells** (see [Population Dynamics](index.md#selecting-cells)), then use **Step 1 — Death Dynamics**. The step requires a `dead_channel` in the metadata.

## Controls

| Control | Default | What it does |
|---|---|---|
| **Group plots by condition column(s):** | none selected | Optional. Pick one or more metadata columns — `exp_nr`, `well` and every `…_line_condition` column — to pool targets by condition (e.g. organoid line + T-cell line). Adds grouped panels to the **per-target** PDF; the combined run ignores it. |
| **▶ Run Death Dynamics (per target)** | — | Runs the analysis separately for **each** selected target. Enabled when ≥ 1 selected target has a `dead` column. |
| **▶ Run Combined Death Dynamics (≥2 targets)** | — | One cross-target comparison. Enabled when ≥ 2 selected targets have a `dead` column. |

Each run button has a **+🛒** and a **👁** button (see [Running and viewing results](index.md#running-and-viewing-results)).

## Death thresholds (read-only)

Below the buttons, a read-only panel lists the **Dead mask % threshold** used for each target (e.g. `10 %`, or `<not configured>`). The threshold is **owned by Feature Extraction** — nothing to tune here. To change it, set a new **Dead mask % threshold** in the [Feature Extraction](../feature_extraction.md#changing-the-dead-threshold-later) tab, click **🔄  Re-run death**, then re-run [Filtering](../filtering.md) for that cell type. No full re-extraction is needed.

```{note}
If a selected target has no `dead` column, a disclaimer *"⚠ No death classification found for: …"* appears and that target is skipped. Run Feature Extraction with a **Dead mask % threshold** > 0 (then Filtering) to create the column.
```

## Death Dynamics outputs

| Run | Result PDF |
|---|---|
| Per target | `analysis/<target>/results/combined_general_<target>_dynamics_analysis.pdf` |
| Per target, per sample | `analysis/<target>/results/per_sample/<sample>/<target>/<sample>_<target>_analysis.pdf` |
| Combined | `analysis/multi_organoid_comparison/multi_organoid_death_dynamics_comparison.pdf` |

**The per-target PDF** opens with a *Death Dynamics Overview* page:

- **% dead targets over time, one line per sample** — the headline "how fast does this population die" curve;
- **end-of-experiment stacked bar** per sample: alive (blue) / dead (red) / disappeared (grey);
- when condition columns are selected, also **% dead per condition group** (targets pooled across samples, mean ± SEM).

Then one page per death signal (`percentage_dead_mask`, `nr_dead_mask_pixels`, and `mean_dead_dye` when present): mean ± SEM per condition group (when condition columns are selected), per-sample mean ± SEM, and per-sample mean ± SEM from baseline.

**The per-sample PDF** shows maximum-projection snapshots of the sample at the first, middle and last timepoints (the raw image's second channel, labelled *Live Channel*, and the dead mask with the segmentation on top), the instantaneous "still tracked and not yet dead" curve, and per-target traces of the three smoothed death signals and of volume. It is only made when the sample's dead-mask image is found.

**The combined PDF** compares target types (colours by line condition when the metadata has one, line style by target type):

- % dead over time, one line per sample × target type;
- the end-of-experiment stacked bar for every target × sample;
- **per-target death-signal traces** (smoothed % dead mask and dead-pixel count), so single dying targets are visible rather than only averages;
- the same traces made **non-decreasing**, absolute and baseline-normalised, with the dead phase in red and a ✕ where a target disappears, plus a **mean cumulative dead fraction** subpanel per condition;
- **mean ± SEM per condition and target type** of the baseline-normalised traces.

```{warning}
**Two different "percentage dead" live in these outputs.** The PDFs use the cumulative definition above. The **per-sample** table (`<sample>_<target>_general_analysis.csv`) also reports an *instantaneous* reading: `percentage_dead` there is the targets flagged dead **right now** divided by `nr_organoids_present` — the targets **still tracked** at that timepoint. A target that dies and then stops being tracked leaves both, so this curve can go **down**. The same table carries `percentage_dead_cumulative` and `nr_dead_cumulative`, which match the PDFs.
```

### Which CSV backs which plot

Every panel can be reproduced from a table. Tables derived only for a figure go to a `plot_data/` folder next to it.

**Per target**, under `analysis/<target>/results/`:

| PDF panel | Backing CSV |
|---|---|
| % dead over time per sample; end-of-experiment bar (last timepoint) | `combined_general_<target>_dynamics_analysis.csv` |
| % dead per condition group | `plot_data/grouped_fraction_dead_<target>.csv` |
| Per-sample mean ± SEM pages | `combined_mean_sem_<target>_dynamics_analysis.csv` |
| Grouped mean ± SEM panels | `combined_mean_sem_grouped_<target>_dynamics_analysis.csv` |
| Per-sample PDF: still-tracked-and-not-yet-dead curve | `per_sample/<sample>/<target>/<sample>_<target>_general_analysis.csv` |
| Per-sample PDF: per-target traces | `per_sample/<sample>/<target>/<sample>_<target>_track_analysis.csv` |

**Combined**, under `analysis/multi_organoid_comparison/`:

| PDF panel | Backing CSV |
|---|---|
| % dead per sample × target, end-state bars | `combined_multi_organoid_death_dynamics.csv` |
| Per-target smoothed traces | `plot_data/per_organoid_traces_smoothed.csv` |
| Absolute non-decreasing traces | `plot_data/per_organoid_traces_absolute_cummax.csv` |
| Baseline-normalised non-decreasing traces | `plot_data/per_organoid_traces_normalized_cummax.csv` |
| Disappearance markers (✕) | `plot_data/disappearance_markers.csv` |
| Mean cumulative dead fraction subpanel | `plot_data/mean_cumulative_dead_fraction.csv` |
| Mean ± SEM per condition and target type | `plot_data/grouped_dead_signal_meansem_<feature>.csv` |

### Reading the tables

- **Cohort columns.** `nr_alive`, `nr_dead`, `nr_disappeared` and `nr_not_yet_seen` always add up to `nr_organoids_total` (every track in that sample). `nr_organoids_t0` (tracks present at the sample's first analysed timepoint) is kept for reference; when it differs, some tracks appear later.
- **How many targets back each mean.** `combined_mean_sem_*` has a `<feature>_n` column per timepoint. It falls as tracks end, so late timepoints are averages over survivors — check it before reading a late trend.
- **Where a baseline sits.** `<feature>_from0_baseline_t` is the timepoint that was subtracted — normally the sample's first timepoint, later if a channel has no data there.
- **Smoothed traces.** Death Dynamics recomputes every `smoothed_…` signal as a rolling mean over 5 timepoints, whatever the frame interval; the ≈ 10-minute smoothed columns from Feature Extraction are not used.
- **The non-decreasing traces** are built in three steps: tracks that stop early get a **flat tail** to the sample's last timepoint; the signal is **baseline-subtracted** per track; then made **non-decreasing** (running maximum). That is why these tables have more rows than the filtered features.
- **The cumulative-dead subpanel nudges flat curves.** Groups that sit at zero get a small upward offset so they stay visible. The drawn line is `mean_cum_dead_fraction + plot_offset`; both columns are in the table, and the offsets are named in the figure footnote.

## Provenance and stale results

The per-target and combined PDFs are each written with a `<name>_provenance.json` recording the filtered table(s) they were built from (path, size, modification time), the row/track/sample counts and the run parameters.

```{warning}
Death Dynamics results are **not** refreshed when you re-run Filtering. When you open a PDF with **👁** and it is older than the table it came from, the Log warns you — re-run Death Dynamics to bring it back in sync.
```

## Tips

- **Read the "disappeared" (grey) bar as QC.** A few targets disappearing (often organoids fusing) is normal; a *large* grey bar points to a segmentation or tracking problem in that sample, not to biology.
- **Compare % dead and absolute dead-pixel counts.** The binary threshold tends to over-represent death in *small* organoids (their % crosses sooner), while the dead-pixel count leans toward *big* organoids. Looking at both — ideally on organoids of comparable size — gives the cleanest picture.

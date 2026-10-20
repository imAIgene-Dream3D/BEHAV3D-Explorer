# 💀 Death Dynamics

Quantifies how the measured signal progresses across the selected target population(s) over time, using the sticky `dead` flag and the dead-mask signal computed during Feature Extraction. If your `dead_channel` carries a reporter other than a death dye, read every "dead" below as "past the threshold you set".

The headline curve is the **cumulative percentage of dead targets at each timepoint** — the number of targets flagged dead *at or before* that timepoint, out of every target tracked in that sample. Once a target is counted dead it stays counted, so this curve never goes down. The death-signal traces (`percentage_dead_mask` or `nr_dead_mask_pixels`) are **baseline-normalised** — shifted so each starts at 0 at the track's first timepoint — so an already-dead baseline doesn't skew the trend.

```{warning}
**Two different "percentage dead" live in these outputs.** The combined overview uses the cumulative definition above. The **per-sample** file (`per_sample/.../<sample>_<target>_general_analysis.csv`) additionally reports an *instantaneous* reading: `percentage_dead` there is the targets flagged dead **right now**, divided by `nr_organoids_present` — the targets **still being tracked** at that timepoint. A target that dies and then stops being tracked leaves both, so the instantaneous curve can go **down**, and it can sit far above the cumulative one late in an experiment. The per-sample file carries `percentage_dead_cumulative` and `nr_dead_cumulative` alongside, which match the combined table row for row — use those to reproduce the overview plots.
```

| Button | What it does | Enabled when |
|---|---|---|
| **▶ Run Death Dynamics (per target)** | Runs the analysis separately for **each** selected target. | ≥ 1 selected target has a `dead` column. |
| **▶ Run Combined Death Dynamics (≥2 targets)** | Produces a single cross-target comparison. | ≥ 2 selected targets have a `dead` column. |

Next to each button:

- **+🛒** — adds the step to the [Processing Queue](../../plugin_essentials/processing_queue) to run later in a batch.
- **👁** — opens the resulting PDF in napari (enabled once it exists; if several targets are selected it offers a chooser menu).

## Death thresholds (read-only)

Below the buttons, a read-only panel lists the **Dead mask % threshold** currently configured for each target. This value is **owned by Feature Extraction** — there is nothing to tune here. To change it, go back to the [Feature Extraction](../feature_extraction) tab, set a new Dead mask % threshold, and re-run Feature Extraction (and Filtering) for that cell type.

```{note}
If a selected target shows `⚠ no dead column`, the Death Dynamics buttons stay disabled and a disclaimer tells you to re-run Feature Extraction with a Dead mask % threshold > 0. The death classification is what Death Dynamics measures, so without it there is nothing to plot.
```

## Death Dynamics outputs

| Run | Result PDF |
|---|---|
| Per target | `<output_dir>/analysis/<target>/results/combined_general_<target>_dynamics_analysis.pdf` |
| Combined | `<output_dir>/analysis/multi_organoid_comparison/multi_organoid_death_dynamics_comparison.pdf` |

**The per-target PDF contains:**

- A **line plot of the percentage of dead targets over time**, one line per sample — the headline "how fast does this population die" curve.
- A **stacked bar of the end-of-experiment state** (alive / dead / disappeared) per sample.
- **Per-sample mean ± SEM pages** for the death signals over time (both the raw and a smoothed version), so you can see the average trajectory and its spread per sample.

**The combined PDF adds cross-target context:**

- Percentage-dead-over-time with **one line per sample × target type**.
- An **end-of-experiment stacked bar** (alive / dead / disappeared) for every target × sample.
- **Per-individual-target death-signal traces** (smoothed % dead-mask and smoothed dead-pixel count), and baseline-normalised versions, so single dying targets are visible rather than only averages.

## Which CSV backs which plot

Every panel is reproducible from an exported table. Frames that are derived on the fly for a figure are written to a `plot_data/` folder next to that figure.

**Per target**, under `<output_dir>/analysis/<target>/results/`:

| PDF panel | Backing CSV |
|---|---|
| % dead over time, one line per sample | `combined_general_<target>_dynamics_analysis.csv` |
| End-of-experiment stacked bar | same file, at each sample's last timepoint |
| % dead grouped by condition (only with a *Group by* selection) | `plot_data/grouped_fraction_dead_<target>.csv` |
| Per-sample mean ± SEM feature pages | `combined_mean_sem_<target>_dynamics_analysis.csv` |
| Grouped mean ± SEM feature panel | `combined_mean_sem_grouped_<target>_dynamics_analysis.csv` |
| Per-sample PDF: tracked-and-not-yet-dead curve | `per_sample/<sample>/<target>/<sample>_<target>_general_analysis.csv` |
| Per-sample PDF: per-track feature traces | `per_sample/<sample>/<target>/<sample>_<target>_track_analysis.csv` |

**Combined**, under `<output_dir>/analysis/multi_organoid_comparison/`:

| PDF panel | Backing CSV |
|---|---|
| % dead per sample × target, and the end-state bars | `combined_multi_organoid_death_dynamics.csv` |
| Per-organoid smoothed traces | `plot_data/per_organoid_traces_smoothed.csv` |
| Absolute non-decreasing traces | `plot_data/per_organoid_traces_absolute_cummax.csv` |
| Baseline-normalised non-decreasing traces | `plot_data/per_organoid_traces_normalized_cummax.csv` |
| Disappearance markers (✕) | `plot_data/disappearance_markers.csv` |
| Mean cumulative dead fraction subpanel | `plot_data/mean_cumulative_dead_fraction.csv` |
| Grouped mean ± SEM dead-signal plots | `plot_data/grouped_dead_signal_meansem_<feature>.csv` |

### Reading the derived tables

- **Cohort columns.** `combined_general_*` reports `nr_alive`, `nr_dead`, `nr_disappeared` and `nr_not_yet_seen`, which always add up to `nr_organoids_total` (every track in that sample). `nr_organoids_t0` is kept for reference; when it differs from `nr_organoids_total`, some tracks first appear after t=0 and the run says so in its log.
- **How many organoids back each mean.** `combined_mean_sem_*` carries a `<feature>_n` column per timepoint. It falls as tracks end, so late timepoints are averages over survivors — check it before reading a trend at the end of a run.
- **Where a baseline sits.** `<feature>_from0_baseline_t` records the timepoint that was subtracted. It is normally 0, but a channel with no data in a sample's opening frame is anchored on its first timepoint that has data.
- **The traces behind plots 4 and 5** go through three steps, in this order: tracks that stop early are **extended with a flat tail** to the sample's last timepoint; the signal is **baseline-subtracted** per track; and the result is made **non-decreasing** (running maximum). The exported `plot_data` tables are post-transformation, which is why they have more rows than the filtered track features.
- **The cumulative-dead subpanel nudges flat curves.** Groups that sit at zero get a small upward offset so they stay visible. The drawn line is `mean_cum_dead_fraction + plot_offset`; both columns are in the exported table, and the offsets are named in the figure footnote.

## Provenance and stale results

Each results PDF is written with a `<name>_provenance.json` recording the track-features file it was built from (path, size, modification time), the row/track/sample counts, and the run parameters.

```{warning}
Death Dynamics results are **not** refreshed when you re-run Filtering. The plugin compares the timestamps and shows a warning when the results on disk are older than the track features they came from — re-run Death Dynamics to bring them back in sync. If you are reading older outputs that have no `_provenance.json`, check the file dates yourself before trusting them.
```

After a successful run a pop-up offers to open the results folder.

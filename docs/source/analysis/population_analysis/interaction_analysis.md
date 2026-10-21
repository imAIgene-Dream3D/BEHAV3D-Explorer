# 🤝 Interaction Analysis

Relates target–effector contacts to target death. A contact is read from the target's `<effector>_contact` column, written by [Feature Extraction](../feature_extraction.md) when the **Contact** family is on. Unlike Death Dynamics, it **can run without a dead channel** — it then skips the death-aware plots.

Open it with **Start** on the **Interaction Analysis** card, tick your targets in **🎯 Target cells**, then the effectors in **🤝 Interaction cells** inside **Step 2 — Interaction Analysis** (see [Population Dynamics](index.md#selecting-cells)). Step 2 has two runs:

| Button | What it does | Enabled when |
|---|---|---|
| **▶ Run Interaction Analysis (per target)** | **2a** — for each selected target, one PDF per effector: cumulative contact over time and alive-vs-dead plots. | ≥ 1 target **and** ≥ 1 effector selected. |
| **▶ Run Interaction Overview** | **2b** — one summary PDF across the selected target(s): contacts per target, cumulative-to-death curves and the active-killing dashboard. Works with one or more targets. | Same. |

Each run button has a **+🛒** and a **👁** button (see [Running and viewing results](index.md#running-and-viewing-results)).

```{note}
**Without death data** the following are skipped automatically — when no `dead_channel` is set (Step 2 then lists them in a yellow box), or when a selected target's filtered table has no `dead` column (a disclaimer names the target):

- alive-vs-dead plots (overall and per sample);
- survive / die statistics (`n_survives`, `n_dies`, `mean_contacts_survives`, `mean_contacts_dies`);
- cumulative-to-death curves (Interaction Overview).

The active-killing dashboard of the Overview needs [Active Killing](active_killing.md) results, which in turn need a dead channel.
```

## Interaction settings

Each run has its own collapsible settings section, closed by default.

**Per-target settings** — apply to **▶ Run Interaction Analysis (per target)**:

| Control | Default | What it does |
|---|---|---|
| **Group overall plots by immune line condition** | off | Splits the two *overall* plots by the effector's line condition (`im_<type>_line_condition` in the metadata): the cumulative-contact plot draws one line per condition, and the alive-vs-dead plot colours by condition (solid = survives, dashed = dies). Per-sample plots are unchanged. |

**Interaction Overview settings** — apply to **▶ Run Interaction Overview**:

| Control | Default | Applies to | What it does |
|---|---|---|---|
| **Before-death window:** | `60 min` | cumulative-to-death curves only | How far back before each target's time of death the curves look. Enabled only when the selected targets have death data. |
| **Temporal Range** — **All timepoints** / **Custom time range** with **Start T:** – **End T:** | **All timepoints** (`0` – `100` when custom) | contacts-per-target plot and active-killing dashboard | Restricts those plots to a window of timepoints. It always stays inside the [timepoint range set in Filtering](../filtering.md#restrict-to-a-timepoint-range), if any. |
| **Annotate by immune line condition** | off | contacts-per-target plot and active-killing dashboard | Shows a separate group per effector type and line condition instead of pooling them. |
| **Group by:** | **By target (organoid) type** | all three plots | x-axis grouping: **By target (organoid) type** keeps each target type as its own group; **By treatment (immune cell)** pools target types and groups by effector type. |

## Interaction outputs

| Run | Result PDF |
|---|---|
| Per target ↔ effector | `analysis/<target>/interaction_analysis/interaction_analysis_<target>_vs_<effector>.pdf` |
| Interaction Overview | `analysis/multi_organoid_comparison/multi_organoid_interaction_comparison_<settings>.pdf` |

The Overview file name records its settings so different runs don't overwrite each other: grouping (`orgtype` / `treatment`), time range (`Tall`, or `T<start>-<end>`), and `annoLC` when annotating by line condition — e.g. `multi_organoid_interaction_comparison_orgtype_Tall.pdf`.

**The per-target PDF contains:**

- **Cumulative contacts over time** — overall (mean ± SEM over all samples) and per sample: how contact accumulates as the movie runs.
- **If death data is present: alive vs dead** — the same curves split by whether the target survives or dies, overall and per sample: do targets that die accumulate more contact than those that survive?

**The Interaction Overview PDF contains:**

- **Cumulative interactions per target** — a violin of each target's total number of contact timepoints, split into *Live* and *Dying* targets when death data exists; the axis labels give dead / total counts per group.
- **If death data is present: cumulative-to-death curves** — mean ± SEM of contact accumulation for dying targets only, aligned on each target's time of death (x-axis in minutes before death).
- **If [Active Killing](active_killing.md) results exist: the active-killing dashboard** — *Contact event duration* per group and target fate (one point per effector–target contact event, coloured by whether it actively killed), and *Active killing efficiency*: the % of contact events in each group that actively killed.

### Which CSV backs which plot

Per target, under `analysis/<target>/interaction_analysis/`:

| File | Contents |
|---|---|
| `interaction_stats_<target>_vs_<effector>.csv` | Per-sample summary: number of targets, mean / SD % of time in contact, mean / SD total contacts, plus survive / die counts and contacts when death data exists. |
| `plot_data/cumulative_overall_<target>_vs_<effector>.csv` | Cumulative contacts over time, all samples. |
| `plot_data/cumulative_per_sample_<target>_vs_<effector>.csv` | Cumulative contacts over time, per sample. |
| `plot_data/alive_vs_dead_overall_<target>_vs_<effector>.csv` | Alive vs dead, all samples. |
| `plot_data/alive_vs_dead_per_sample_<target>_vs_<effector>.csv` | Alive vs dead, per sample. |

The `plot_data` tables carry `mean`, `std`, `count` and `sem` per timepoint: the shaded band is `mean ± sem`, and `count` tells you how many tracks back each point.

Interaction Overview, under `analysis/multi_organoid_comparison/`:

| File | Contents |
|---|---|
| `multi_organoid_interaction_summary_<settings>.csv` | One row per target: total contacts per effector type, fate (*Live* / *Dying*) and time of death. |
| `multi_organoid_cumulative_to_death_curves_min_<settings>.csv` | The cumulative-to-death curves (time in minutes). |

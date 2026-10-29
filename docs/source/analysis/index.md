# Analysis

The Analysis section covers everything that happens **after tracking**: turning the tracked segments + raw image into per-cell features, filtering for quality, and running the higher-level analyses (population signal dynamics, cell–cell interactions, behavioural-state classification), with ways to paint features and results back onto the images.

```{note}
**Which route fits depends on your readout, not on your cell types.**

- A reporter that **switches on and stays on** — a death dye, a differentiation or activation marker — belongs in [Population Dynamics](population_analysis/index.md), which measures how that signal rises across a population and whether the rise follows contact.
- A **fluctuating** reporter that goes on and off, such as calcium, belongs in [Single Cell](single_cell/index.md), where its intensity becomes a behavioural feature.
- **No reporter at all** is fine: [Single Cell](single_cell/index.md) profiles cells on motility, morphology, or contact alone.

Nothing here requires an immune-versus-target design. A single population with one readout is a valid experiment.
```

In the BEHAV3D EXPLORER dock widget these steps are spread across three tabs; the last one, **📊 Analysis**, has three sub-tabs:

```{mermaid}
flowchart LR
    T["📍 Tracked segments"] --> FE["🧪 Feature Extraction"]
    FE --> F["🧹 Filtering"]
    F --> A["📊 Analysis"]
    A --> FB["🔬 Feature Backprojection"]
    A --> PD["👥 Population Dynamics"]
    A --> SC["🧬 Single Cell"]
```

| Page | Tab in napari |
|---|---|
| [Feature Extraction](feature_extraction.md) — per-timepoint features; also hosts [Active Killing](population_analysis/active_killing.md) | 🧪 Feature Extraction |
| [Filtering](filtering.md) — quality filtering of tracks | 🧹 Filtering |
| [Feature Backprojection](feature_backprojection.md) — preview any feature on the segmentation, frame by frame | 📊 Analysis → 🔬 Feature Backprojection |
| [Population Dynamics](population_analysis/index.md) — Death Dynamics, Interaction, Invasiveness (and Active Killing) | 📊 Analysis → 👥 Population Dynamics |
| [Single Cell](single_cell/index.md) — [🔬 Behavioral State](single_cell/state_classification.md) and [🛤️ State Trajectory](single_cell/track_classification.md) | 📊 Analysis → 🧬 Single Cell |

## The 📊 Analysis tab

Until metadata is loaded the tab only shows *"Load metadata in the Data Preparation tab to see analysis options."* Once it is loaded you see, from top to bottom:

- the **🧬  Group cells together for analysis** button (see [below](#grouping-cell-types));
- the three sub-tabs **🔬 Feature Backprojection**, **👥 Population Dynamics** and **🧬 Single Cell**;
- the shared [📄 Results panel](../plugin_essentials/results_panel.md), which lists every result file and refreshes when you switch sub-tab.

All analyses run in the background; while one runs you cannot switch sub-tab or start another (see [Running Steps](../plugin_essentials/running_steps.md#background-execution)).

### Guided overview

**👥 Population Dynamics** and **🧬 Single Cell** open on a **Guided overview**: one card per analysis, with its name and the question it answers (e.g. *Death Dynamics — "How and when do target cells die?"*). Each card has:

- **ⓘ  New to this? See what this analysis does** — expands a plain-language description, **What you'll get**, and **WHAT YOU'LL DECIDE**: the choices you will make, tagged *only you know this*, *suggested set, adjustable*, *can be estimated* or *tested defaults, optional*;
- **💬  Ask the assistant** — sends a question about this analysis to the [Co-pilot assistant](../getting_started.md#co-pilot-assistant-queenb);
- **Start — open the settings  ▸** — opens the settings of that analysis only.

**‹ Back to overview** at the top of the settings returns to the cards. The overview never runs anything by itself.

### Grouping cell types

**🧬  Group cells together for analysis** merges several **already filtered** cell types (e.g. CD4 + CD8 T cells) into one population that you can analyse as a single group. It opens the **Cell Type Grouping** dialog:

| Control | What it does |
|---|---|
| **Select cell types to group (must be filtered already)** | Tick the member cell types. Types that have not been through [Filtering](filtering.md) are shown as *not yet filtered* and cannot be ticked. Existing groups appear as `<name>_merged  (group)`, with `[tracked: built/total]` when tracked segments exist. |
| **Group name*:** | Letters, digits, `-` and `_` only. The group is called `<name>_merged`. |
| **Also build tracked segments (slower)** + **Workers:** | Also builds a tracked label image per sample for the group, needed to see it in Visualization, backprojection and exemplar exports. Off by default. |
| **Create Group** | Merges the members' filtered tables and shows the group's track-length distribution. |
| **📊 Track Length Distribution** / **🧩 Build Tracked Segments** / **Remove Group** | Act on one ticked existing group: preview its track lengths, build (or rebuild) its tracked segments, or remove it (the merged table on disk is kept). |

The group then appears in Feature Backprojection, Population Dynamics and Single Cell next to the real cell types. In Population Dynamics it takes the category of its first member (an organoid group is offered as a target, an immune group as an effector, an other group as either); Single Cell lists every group, whatever its members. Every row keeps an `origin_cell_type` column, so results can still be split by original type. Your `metadata.csv` is never changed; the group is recorded in `behav3d_parameters.yml` and its table is written to `analysis/<name>_merged/track_features/`.

```{note}
The merged table is a snapshot taken when you click **Create Group**. If you later re-run Filtering for a member, remove the group and create it again.
```

```{toctree}
:hidden:
:maxdepth: 2

feature_extraction
filtering
feature_backprojection
population_analysis/index
single_cell/index
```

# 🧬 Single Cell

The **🧬 Single Cell** sub-tab of **📊 Analysis**. Where [Population Dynamics](../population_analysis/index.md) works at the **population** level, Single Cell classifies the **behaviour of individual cells** over time. It offers two analyses:

| Analysis | Question it answers | What it does |
|---|---|---|
| [🔬 Behavioral State](state_classification.md) | *What behaviour states do cells pass through?* | Assigns each cell, at each timepoint, to a recurring **behavioural state** (e.g. resting, scanning, migrating) with a Hidden Markov Model, optionally subdivided by true/false flags such as contact or death. |
| [🛤️ State Trajectory](track_classification.md) | *What movement patterns exist across whole tracks?* | Groups whole **tracks** into clusters by how their behaviour unfolds over the entire movie, either from their sequence of behavioural states (run 🔬 Behavioral State first) or directly from their per-timepoint features (the clustering itself needs no Behavioral State run). |

## How the sub-tab is organised

- **Cell type:** dropdown at the top, shared by both analyses (see below), with a status line on the right (*Ready — N cell type(s) available.*).
- Below it, the [Guided overview](../index.md#guided-overview): one card per analysis. There is no tab bar between the two analyses: **Start — open the settings  ▸** on a card opens that analysis's settings, and **‹ Back to overview** returns to the cards.
- Result files of both analyses are listed in the shared [📄 Results panel](../../plugin_essentials/results_panel.md).

## Cell-type scope and selector

The **Cell type:** dropdown lists the individually tracked populations:

- **immune** and **other** cell types (multicolour populations appear once, as their merged `<type>_merged` entry, not per colour);
- every cell-type group made with [**🧬  Group cells together for analysis…**](../index.md#grouping-cell-types).

Organoid types are not listed; they are analysed in [Population Dynamics](../population_analysis/index.md). Nothing here is specific to immune cells: declare any individually tracked population as an *other* cell type and it becomes available.

The selected cell type applies to both analyses and is remembered in `behav3d_parameters.yml`. You cannot change it while an analysis is running.

```{important}
Single Cell reads the per-timepoint **track-features table** of the chosen cell type, written by [Feature Extraction](../feature_extraction.md) and [Filtering](../filtering.md). The filtered table is used when it exists; otherwise the unfiltered one is used and a warning says so. With neither, the settings stay disabled.

If you re-run Filtering after classifying, an orange warning appears when the filtered tracks no longer match the saved Behavioral State / State Trajectory results. Re-run the analysis to bring them in line.
```

## First decide what you are profiling on

Before configuring either analysis, settle the question the feature selection answers: **what makes two cells behave differently in your experiment?** The choice determines which feature families you extract and which you feed to the model.

| Profile on | Typical features | Ask this when |
|---|---|---|
| **Motility** | speed, displacement, cumulative displacement, directional persistence, straightness | Cells move, and how they move is the readout |
| **Interaction** | contact with another population, or with their own type | Engagement is the readout, whether or not anything follows from it |
| **Morphology** | volume, sphericity, elongation and related shape descriptors | Shape change carries the signal — spreading, polarising, rounding |
| **Channel intensity** | mean or quantile intensity of a given channel, and its fold change | A reporter's level is the readout, including fluctuating reporters such as calcium |

These combine, but starting with a small, biologically interpretable set is better than selecting everything: states are easier to name and the model is easier to trust. Add families once you can explain the states you already have.

```{toctree}
:hidden:
:maxdepth: 1

state_classification
track_classification
```

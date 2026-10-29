# 👥 Population Dynamics

**Population Dynamics** is the family of analyses that describe what happens to whole populations and to the contacts between them, as opposed to [Single Cell](../single_cell/index.md), which classifies the behaviour of individual cells. It answers four related questions about your **target** populations (organoid or "other" cell types) and the **effector** populations that contact them (immune or "other" cell types). Invasiveness and Active Killing additionally need the effector to be declared as immune (`im_`), and Invasiveness is measured against organoid types only — the prefix is a processing category, not a biological claim ([Population categories](../../data_preparation.md#population-categories)).

| Page | Question it answers | Where you run it |
|---|---|---|
| [💀 Death Dynamics](death_dynamics.md) | How, and how fast, does a signal rise across each target population over the movie? | 📊 Analysis → 👥 Population Dynamics → **Step 1 — Death Dynamics** |
| [🤝 Interaction Analysis](interaction_analysis.md) | How do contacts between targets and effectors relate to that signal? | 📊 Analysis → 👥 Population Dynamics → **Step 2 — Interaction Analysis** |
| [🫳 Invasiveness Analysis](invasiveness.md) | How much of each effector — the cell plus a 2 µm shell — overlaps the target within the contact threshold (at the default 0 µm: the part of the cell inside the target), from the *effector's* perspective? | 📊 Analysis → 👥 Population Dynamics → **Step 3 — Invasiveness Analysis** |
| [🎯 Active Killing](active_killing.md) | Which *individual* effectors are associated with a signal rise in the target they touched? | 🧪 Feature Extraction tab |

```{note}
**Why Active Killing is configured somewhere else.** Conceptually it belongs with the three analyses above — it is about targets, effectors and contact. Mechanically it works on the per-timepoint feature tables and writes extra columns into the effector's own table, which [Filtering](../filtering.md) then carries forward. So you configure and run it in [Feature Extraction](../feature_extraction.md), and interpret it here.
```

```{important}
**These analyses are not restricted to cell death.** What they actually compute is the rise of a per-object signal — across a population over time (Death Dynamics), and in association with contact (Interaction, Active Killing). Cell death measured with a dye is the original and canonical use, which is why the controls are named after it.

To use them with **any reporter that switches on** — a differentiation marker, an activation reporter, a stress or damage signal — declare that reporter's channel as `dead_channel` in [Data Preparation](../../data_preparation.md). Everything downstream then works unchanged; read "dead" as "the signal has risen past your threshold".

This applies only to reporters that **go up and stay up**. A **fluctuating** reporter that switches on and off, such as calcium, is not suited to these analyses — use [Single Cell](../single_cell/index.md) analysis instead, where the reporter intensity becomes a behavioural feature.
```

## Opening an analysis

The sub-tab opens on the [Guided overview](../index.md#guided-overview) with three cards: **Death Dynamics**, **Interaction Analysis** and **Invasiveness Analysis**. **Start — open the settings  ▸** on a card shows only what that analysis needs:

| Card | Sections shown |
|---|---|
| Death Dynamics | **🎯 Target cells** + **Step 1 — Death Dynamics** |
| Interaction Analysis | **🎯 Target cells** + **Step 2 — Interaction Analysis** (with its **🤝 Interaction cells** list) |
| Invasiveness Analysis | **Step 3 — Invasiveness Analysis** (it has its own cell pickers) |

Below the settings, a progress bar and a **Log** box report each run. **‹ Back to overview** returns to the cards; your selections are kept.

## Before you start — prerequisites

All three steps read the **filtered** per-timepoint feature tables, so run [Feature Extraction](../feature_extraction.md) and [Filtering](../filtering.md) for every cell type involved first. Buttons enable themselves only when their inputs exist on disk, and the cell lists show status badges so you can see what is missing.

| Requirement | How to satisfy it | Used by |
|---|---|---|
| A **filtered** table for each cell type | Run [Filtering](../filtering.md) for that cell type | Everything |
| A **dead channel** in the metadata — the channel carrying the rising signal you measure | Set `dead_channel` in [Data Preparation](../../data_preparation.md) | Death Dynamics (required); Interaction (optional, unlocks the death-aware plots) |
| A **`dead` column** in the target's filtered table | Feature Extraction with a **Dead mask % threshold** > 0 (or a new threshold applied with **🔄  Re-run death**), then Filtering again | Death Dynamics; death-aware Interaction plots |
| **Contact columns** for the chosen effectors in the target's table | Feature Extraction with the **Contact** family on, for both cell types | Interaction Analysis |
| **Invasiveness columns** in the effector's table | Feature Extraction with **Invasiveness** on for the immune type | Invasiveness Analysis |

```{important}
Without a `dead_channel` in the metadata, Step 1 shows *"⚠ No dead channel configured in metadata. Death Dynamics is unavailable until 'dead_channel' is set."* and Step 2 lists what it will skip. That is expected if your experiment has no rising reporter.
```

## Selecting cells

### 🎯 Target cells (organoid / other)

Every **organoid** and **other** cell type in the metadata (and organoid / other [cell-type groups](../index.md#grouping-cell-types)). Each row carries a badge:

- `✅ filtered` — a filtered table exists and the checkbox can be ticked.
- `⚠ no filtered CSV (run Filtering)` — the checkbox stays disabled until you run Filtering for that cell type.
- `✅ dead-col` / `⚠ no dead column` — whether the alive/dead classification is present in that table.

### 🤝 Interaction cells (immune / other)

Inside **Step 2**: every **immune** and **other** cell type. These are the effectors whose contact with the selected targets defines an interaction — "effector" simply means the population doing the contacting, whatever its biology. Each row shows, per selected target, whether that target's table has a contact column for it (`<target>: ✅` / `<target>: ⚠`). An effector with no contact column in any selected target is disabled; with no target selected the rows read *(select target(s) above)*.

```{tip}
Select your target(s) **first** — effector availability is judged against the targets' contact columns.
```

## Running and viewing results

Next to each **▶ Run …** button:

- **+🛒** — adds the step with the current selection to the [Processing Queue](../../plugin_essentials/processing_queue.md). A queued step whose inputs are still missing when it runs (e.g. Filtering earlier in the queue has not finished) skips itself with a log message instead of stopping the queue.
- **👁** — opens the result PDF in napari, at the resolution set in the Results panel. It is enabled once the PDF exists; when several PDFs match your selection it offers a menu. If a PDF is older than the filtered table it was built from, the Log warns *"… is older than the track features it was built from — re-run this step to refresh it."* Interaction and Invasiveness PDFs do not record their source table, so for them the check compares against every filtered table of the project.

After Death Dynamics and Invasiveness runs, a pop-up offers **Open folder**. Every result file is also listed in the [📄 Results panel](../../plugin_essentials/results_panel.md) below the sub-tabs; where each analysis writes is summarised in [Output Directory & File Layout](../../plugin_essentials/output_layout.md#analysis).

## Tips & best practices

- **Run [Filtering](../filtering.md) for every cell type involved first.** All steps read the *filtered* table; a missing or stale one disables the buttons or feeds short / dead-at-start tracks into the statistics.
- **Set the dead threshold in Feature Extraction**, using its live preview. To change it later, click **🔄  Re-run death** there and re-run Filtering — no full re-extraction needed.
- **Use the combined runs for cross-condition figures.** Per-target runs are best for QC of one population; the combined outputs compare target types or treatments side by side.
- **Queue the heavy work.** For multi-sample experiments, use the +🛒 buttons to put the analyses behind your segmentation / tracking / feature steps and let the whole pipeline run unattended.

## See also

- [Feature Extraction](../feature_extraction.md) — computes the death, contact and invasiveness columns and owns the Dead mask % threshold.
- [Filtering](../filtering.md) — produces the filtered tables these analyses read.
- [Single Cell](../single_cell/index.md) — per-cell behavioural-state and trajectory classification.
- [Output Directory & File Layout](../../plugin_essentials/output_layout.md) — where the result PDFs and CSVs live.

```{toctree}
:hidden:
:maxdepth: 1

death_dynamics
interaction_analysis
invasiveness
active_killing
```

# 🫳 Invasiveness Analysis

Where Interaction Analysis looks at *contact* from the target's side, Invasiveness measures, from the **effector** cell's side, how much of an organoid lies in and right around it. The effector must be declared as an immune (`im_`) population — a processing category, not a biological claim ([Population categories](../../data_preparation.md#population-categories)). It uses the **Invasiveness** feature computed during [Feature Extraction](../feature_extraction.md#invasiveness-immune-populations): for each effector cell at each timepoint, the target voxels inside the cell or within contact distance of it, as a percentage of the cell plus a 2 µm shell around it (`<target>_invasiveness_perc`, 0–100), and an *invasive* flag that is set once this reaches ≥ 50 %. With the default contact threshold of 0 µm this is the part of the effector cell inside the organoid's mask.

Open it with **Start** on the **Invasiveness Analysis** card. **Step 3 — Invasiveness Analysis** is self-contained: it has its own cell pickers, independent of the target / interaction lists of Steps 1 and 2.

```{important}
Invasiveness Analysis only works if you ran Feature Extraction for the effector (immune) population with **Invasiveness** enabled (which also requires **Contact**), then Filtering. Otherwise the target list shows *"⚠ No invasiveness columns for: … Run Feature Extraction with 'invasiveness' enabled, then Filtering."*
```

## Controls

| Control | Default | What it does |
|---|---|---|
| **Immune cell type(s)** | first type ticked | The cell types to analyse. Immune and other types (and groups) are listed, but only immune (`im_`) types can carry invasiveness columns. Several can be ticked: they are analysed together but kept apart in every plot. |
| **🎯 Targets to compare** | all ticked | The targets found in the invasiveness columns of the ticked cell types' filtered tables. |
| **Per-movie summary stat:** | `mean` | How each movie's over-time curve is collapsed to one dot per movie: `mean`, `median`, `max` or `auc` (see below). Only affects the per-movie plots and table. |
| **Separate by immune line condition** | off | Splits results by the immune cell's line condition (`im_<type>_line_condition` in the metadata): a separate box / line per condition instead of pooling them. |
| **Temporal Range (in timepoints):** **All timepoints** / **Custom time range** with **Start T:** – **End T:** | **All timepoints** (`0` – `100` when custom) | Restricts the analysis to a window of timepoints. |
| **▶ Run Invasiveness Analysis** | — | Runs the analysis. Enabled once at least one immune type and one target are ticked. With **+🛒** and **👁** (see [Running and viewing results](index.md#running-and-viewing-results)). |

```{note}
The **Per-movie summary stat** first builds each movie's per-timepoint curve across its cells (the fraction invasive, and the mean % — or the median % when `median` is chosen), then collapses that curve over time:
- **mean** — average of the per-timepoint values.
- **median** — the middle per-timepoint value.
- **max** — the highest per-timepoint value reached.
- **auc** — the area under the curve divided by its duration: a time-weighted average, on the same scale as the curve.
```

## Invasiveness outputs

Written under `analysis/<immune_type>/invasiveness_analysis/`, or `analysis/invasiveness_analysis/` when several immune types are analysed together. File names end with a `<settings>` part recording the selection — the immune types when several (`im-…`), the targets, the summary stat, the time range (`Tall` or `T<start>-<end>`) and `annoLC` when separating by line condition — so different runs don't overwrite each other.

| File | Contents |
|---|---|
| `invasiveness_analysis_<immune>_<settings>.pdf` | All figures (below). |
| `invasiveness_fraction_over_time_<immune>_<settings>.csv` | Per-timepoint fraction of invasive cells, per sample / target. |
| `invasiveness_perc_over_time_<immune>_<settings>.csv` | Per-timepoint mean and median invasiveness %. |
| `invasiveness_per_movie_summary_<immune>_<settings>.csv` | One row per movie: both curves collapsed with the chosen stat. |
| `invasiveness_by_fate_<target>_<immune>_<settings>.csv` | One row per target organoid: mean invasiveness % of the immune cells while they touch it, number of those cells flagged invasive, and whether it died. Only when fate violins are made. |

The PDF contains:

- **Fraction invasive over time** — % of immune cells flagged invasive (invasiveness ≥ 50 %) at each timepoint, per sample.
- **Mean % over time** and **median % over time** — average / typical invasiveness % across *all* cells (including non-invasive 0 % ones).
- **Per-movie summary** of the fraction and of the % — one dot per movie, using the chosen stat.
- **Fate violins** (two per target): mean invasiveness % and number of invasive cells per target organoid, split by whether that organoid died. Made automatically when the target is an organoid type whose filtered table has a `dead` column.

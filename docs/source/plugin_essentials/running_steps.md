# ⚙️ Running Steps

Every processing and analysis step in BEHAV3D EXPLORER behaves the same way when you run it, whichever tab it belongs to. This page explains that shared behaviour once: how steps run in the background, what happens when results already exist, and the **Workers** and unit switches you find in several tabs.

## Background execution

Heavy steps (Zarr conversion, segmentation, tracking, feature extraction, filtering, analyses, project transfer) run in the background, so napari stays responsive: you can keep looking at layers and scrolling through time while they run.

While a step runs:

- a progress bar appears in the tab, with a short status text, a percentage and an estimated time left (e.g. `— 42% — ETA 3:15`); a matching entry appears in napari's activity panel;
- the run buttons of that step are disabled;
- a yellow banner at the top of the dock reads **⏳ Running in the background: … — please wait**.

```{important}
While something runs, **you cannot switch tabs or start another job**. Clicking another tab, or another run button, shows a **Please wait — something is running** dialog listing what is still running. This prevents crashes caused by two jobs touching the same data. A running [Processing Queue](processing_queue.md) counts as running work too.
```

A banner reading **⏳ Loading in the background: … — you can keep working** only signals an automatic load and does not block anything.

## When results already exist

When you start a step and some of its results are already on disk, BEHAV3D EXPLORER lists them per sample and population before doing anything:

| Symbol | Status | Meaning |
|---|---|---|
| ✔ | Complete | All results of this item exist. |
| ◐ | Incomplete | Results exist only partly (for example, an interrupted run). |
| ○ | Missing | No results yet. Missing items are always run. |

You then choose:

| Button | What it does |
|---|---|
| **Overwrite All** (**Overwrite** for a single item) | Recomputes everything and replaces the existing results. |
| **Skip Existing** (**Skip** for a single item) | Keeps complete items and runs only the missing ones. Reads **Skip & Resume** when some items are incomplete: those are then completed too. |
| **Customize…** | Opens a table with one row per sample and population, to choose item by item (see below). |
| **Cancel** (default) | Nothing runs. |

In the **Customize…** table each row has an **Action**: complete items can be set to **Skip (keep existing)** or **Overwrite**, incomplete items to **Run missing** or **Overwrite**; missing items are always **Run**. The buttons **Overwrite all**, **Skip complete (default)**, **Overwrite selected** and **Reset selected** change many rows at once. **Run** starts the step, **Back** returns to the first dialog.

```{note}
In the [Processing Queue](processing_queue.md) one dialog covers all queued steps. If an earlier step recomputes a sample, later steps that you set to skip for the same sample are re-run automatically, so results never get out of step with each other.
```

## Workers

The **Workers** setting (number of CPU processes used in parallel) is **shared**: it appears in the Data Preparation tab (Zarr conversion), in the Pixel Classifier, APOC and ConvPaint segmentation pages, in the manual track editor and in the Analysis tab's **Cell Type Grouping** dialog, and changing it in one place changes it everywhere. (The **Workers** of the Feature Extraction sub-tabs are set per cell type.) It is saved with the project.

On a fresh project it defaults to **half of your CPU cores**. Each place caps it at its own maximum (usually one less than your number of cores). More workers is faster but uses more memory; lower it if your computer runs out of memory.

## Units switches

Several parameter panels show a small switch that changes the **unit in which values are displayed**:

| Switch | Where | Options |
|---|---|---|
| **px** ⇄ **µm** (labelled **Units:**, **Distance units:** or **Contact units:**) | Size and distance parameters in Segmentation, Tracking, Feature Extraction and Filtering | **µm** (µm, µm³; default) or **px** (pixels, voxels) |
| **frames** ⇄ **hours** | Time-based filters in Filtering | **frames** (default) or **hours** |

The switch only changes the display: BEHAV3D EXPLORER converts the value back automatically, so the result is the same whichever unit you type in. The conversion uses the pixel sizes (`pixel_distance_xy`, `pixel_distance_z`) and time interval (`time_interval`, `time_unit`) of your [metadata](../data_preparation.md). If they are missing, the switch is disabled and shows *(no resolution in metadata)* or *(no time interval in metadata)*.

## See also

- [Processing Queue](processing_queue.md): run several steps in a row, unattended.
- [Results panel](results_panel.md): open the plots and tables a step produced.

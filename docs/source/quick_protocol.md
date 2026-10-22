# ⚡ Quick protocol

The whole BEHAV3D EXPLORER pipeline on one page — the minimal path and the few real choices at each tab. Work the tabs of the **BEHAV3D Explorer** panel **left to right**. Each section links to its full page when you want the detail.

```{note}
Method names and defaults here are **what the plugin does on a fresh project**, not recommended values. Derive every threshold from your own images (spacing, frame interval, object size) and **preview before batching** (see [Deriving settings for a new dataset](recommended_settings.md)).
```

## 📋 Data Preparation

*Full detail: [Data Preparation](data_preparation.md).*

The numbers below are the section numbers shown in the tab.

- **1 · Output Directory** — set it first; everything else is saved under it.
- **2 · Metadata Builder** (tick its header to expand): **Number of samples:** → **Organoid types:** / **Immune types:** / **Other types:** (+ **Include dead channel**) → **Configure Cell Types** → **Create Sample Forms** → fill Sample 1 → **Fill All from Sample 1** → **Save Metadata CSV**.
- **3 · Metadata Loader** → **Load Metadata** — the other tabs show their options only once it has loaded without errors. **4 · Metadata Overview** then shows what was loaded.
- **5 · Dimension Order (Optional)** — check the axis order of every sample (e.g. `TCZYX`) and fix any wrong row.
- **6 · Convert to Zarr** → **Convert All Images to Zarr** — optionally tick **Clip timepoints (select range)** to cut dead start/end frames or give all samples the same length.

**Population vs Line/Condition:**

| If… | Declare it as… |
|---|---|
| You can tell it apart **in the image** (own or overlapping channel) | a **population** — own segmentation + track IDs |
| Same biology, deliberately **split across colours** | **one** immune population, tick **Multicolor** |
| **Identical on screen**, differs only by treatment / donor / subclone | **one** population; use **Line** / **Condition** |

**Required per sample** (marked `*`): Name · Exp # · Well · Image path · Pixel XY / Pixel Z + Dist unit · Time interval + **Time unit** (defaults to `s` — check it) · Line per population. With a dead channel, also set **Dead ch #** (0-based channel index).

## 🦠 Segmentation

*Full detail: [Segmentation](processing/segmentation/index.md). Tune **each cell type on its own**.*

**Pick one method per cell type** (**Method:** dropdown):

| Your situation | Method |
|---|---|
| Already segmented elsewhere (Imaris, Ilastik, script) | **Import segmentation** |
| You have a trained Cellpose model for this data | **Cellpose (Deep Learning)** |
| No model, but GPU + clean channels (no bleed-through) + can wait, and its preview looks right | **Cellpose-SAM (zero-shot)** |
| Otherwise, with a GPU: start here (any cell shape) | **APOC (GPU)** |
| APOC's foreground mask stays wrong after more labels (subtle texture, low contrast) | **ConvPaint (DL pixel classifier)** |
| No GPU available | **Pixel Classifier (Random Forest)** |

**Pixel classifiers (APOC / ConvPaint / Pixel Classifier):** **Generate Training Data** (wait for the progress bar) → paint a few background / foreground pixels → train (**▶ Train <Cell type> classifier** in APOC) → preview with **Run instance segmentation** (**Test Segmentation Parameters** in Pixel Classifier) → tune post-processing → batch run (untick **Process ALL timepoints**, or **All timepoints** in Pixel Classifier, only to set a range). With a dead channel, also paint and train the **Dead** labels.

**Cellpose-SAM:** **Set up Cellpose-SAM environment** (once per BEHAV3D EXPLORER environment) → tick the channel(s) of each population → **👁 Preview segmentation** and set **Min size:** / **Max size:** → **▶ Run Cellpose-SAM**.

**Cellpose (v3):** map and save the channels (**💾 Save Channel Configuration**) → **Browse…** to the model and **📂 Load Model** → run per cell type.

**Dead mask with a Cellpose route:** on the **Cellpose (Deep Learning)** page, map the dead channel and click **☠ Run Dead Mask (Otsu)**.

**Key knobs** — pixel-classifier post-processing ([full detail](processing/segmentation/index.md#instance-post-processing-parameters)):

| Knob | What it does |
|---|---|
| **EDT threshold** / **Seed threshold** | splits touching objects — **raise** to split more, **lower** to keep them merged (EDT threshold ≤ 1 px, or Seed ≤ Mask threshold, splits nothing) |
| **Mask threshold** | foreground cutoff (probability strategies) — **lower** to keep dim cells |
| **Min size** | drops segments smaller than this (noise filter) |
| **Opening px** | smooths boundaries / removes speckles |

**EDT threshold** and **Min size** (like the Cellpose-SAM size filter) are shown in µm / µm³ by default; set **Units:** to **px** to enter pixels / voxels.

## 📍 Tracking

*Full detail: [Tracking](processing/tracking/index.md). One sub-tab per cell type.*

**Pick the method from overlap, not cell type** — between two consecutive frames, does the object still overlap where it was?

| Between frames… | Method |
|---|---|
| Large overlap, mainly **fragments**, nothing new appears (objects appearing later get no ID) | **Fragmentation Propagation** (fresh-project default for organoids) |
| Overlap, but masks **touch / join**, or new objects appear (they get new IDs) | **Bounded Propagation** |
| **Near-static**, detection flickers on/off (every object is stamped onto every frame) | **Reporter Propagation** |
| **Little / no overlap** (moves ~1 cell diameter or more) | **btrack (Bayesian)** (default for immune types) |
| Already tracked elsewhere | **Import tracking** |

A fresh project pre-selects **LapTrack** for other (`ot_`) types; like every pre-selection, it is only a starting point. LapTrack and **TrackPy** are alternatives to btrack for objects that do not overlap between frames ([Tracking](processing/tracking/index.md)).

1. Per sub-tab: **Method:** → parameters → **Run <Cell type> Tracking** (or **Run Batch Tracking (All Cell Types)**).
2. **Inspect in Visualization** — a correctly tracked cell keeps the **same ID / colour across frames**.
3. **Several organoid types in one movie?** **Track all organoids together** (on by default) gives them one shared ID space; untick it to track each type independently.
4. Fix remaining mistakes in the [manual editor](processing/tracking/manual_editing.md) (there is no undo — preview before applying).

**Key knobs** — the propagation methods have few or no parameters; **btrack** is the tunable one ([details](processing/tracking/btrack.md)):

| Knob | What it does |
|---|---|
| **Max search radius** | furthest an object may move between frames and still link — set from the fastest one-frame displacement |
| **Distance threshold** / **Time threshold** | (global optimizer) largest spatial gap / frame gap to bridge when reconnecting broken tracks |

## 🧪 Feature Extraction

*Full detail: [Feature Extraction](analysis/feature_extraction.md). One sub-tab per cell type.*

Computes, for every cell at every timepoint, the features every later step reads. Families that later analyses need are ticked and greyed out:

| Family | Extracted for |
|---|---|
| **Contact**, **Intensity** | all cell types (always) |
| **Movement** | immune / other (always); optional for organoids |
| **Death** | all cell types, when a dead channel is declared (always) |
| **Morphology** | optional, any cell type |
| **Invasiveness** | optional; immune (`im_`) populations only, measured against organoids |

1. Per sub-tab: set the contact threshold (box next to **Contact**; default `0` µm). With a dead channel, set **Dead mask % threshold:** (default `10.0 %`; check it with **👁  Preview Dead Threshold in Viewer**) and, under **⚙ Advanced Configuration**, **Propagate dead signal** (on for organoids, off for immune / other).
2. **Run <Cell type> Feature Extraction**, or **Run Batch Feature Extraction (All Cell Types)**.
3. Changed the dead threshold afterwards? **🔄  Re-run death** re-classifies without re-measuring.
4. **Active Killing** (optional; the effector population must be declared as immune, `im_`): configured and run in this tab, explained with the analyses → [Active Killing](analysis/population_analysis/active_killing.md).

## 🧹 Filtering

*Full detail: [Filtering](analysis/filtering.md).*

Run it for every cell type **even with no filters**: the analyses read its filtered table (Population Dynamics cannot run without it; the others fall back to the unfiltered table). Re-run it after every Feature Extraction, **🔄  Re-run death** or Active Killing run.

1. **📊 See Track Length Distributions** to choose the lengths.
2. Check the filters. Fresh-project defaults: **Trim full time series to max timepoints** `350`, **Filter tracks shorter than minimal length** `30`, and **Trim tracks to maximum length** `30` with **Split long tracks into chunks** — untick the last one to keep full-length tracks.
3. **Filter <Cell type> Tracks & Summarize**, or **Run Batch Filtering (All Cell Types)**.

## 📊 Analysis

*Full detail: [Analysis](analysis/index.md).*

Three sub-tabs: **🔬 Feature Backprojection** (paint any feature onto the cells), **👥 Population Dynamics** and **🧬 Single Cell**. The last two open on a **Guided overview**: click **Start — open the settings  ▸** on the card you want. To analyse several filtered cell types as one, use **🧬  Group cells together for analysis…**.

**👥 Population Dynamics** — population- and interaction-level readouts:

| Analysis | What it gives you |
|---|---|
| **[💀 Death Dynamics](analysis/population_analysis/death_dynamics.md)** | how a target signal rises across each population over time |
| **[🤝 Interaction Analysis](analysis/population_analysis/interaction_analysis.md)** | effector–target contact and how it relates to that signal |
| **[🫳 Invasiveness Analysis](analysis/population_analysis/invasiveness.md)** | how much of each effector cell lies inside or right around an organoid |
| **[🎯 Active Killing](analysis/population_analysis/active_killing.md)** | signal rises attributed to individual effectors *(run in Feature Extraction)* |

**🧬 Single Cell** — per-cell behaviour:

| Analysis | What it gives you |
|---|---|
| **[🔬 Behavioral State](analysis/single_cell/state_classification.md)** | a behavioural state per cell and timepoint (HMM) |
| **[🛤️ State Trajectory](analysis/single_cell/track_classification.md)** | clusters of whole tracks by how their behaviour unfolds |

Both can paint their result back onto the images (**Backprojection**).

**Key knobs:**

- **Active Killing:** **Observation window (tp):** (`5`), **Killing threshold multiplier:** (`1.5`) or **Use absolute threshold**, **Min contact duration (tp):** (`1`). Re-run Filtering afterwards.
- **Behavioral State:** the features (pre-ticked: speed, elongation, sphericity, extent, solidity, net_displacement), **Window size (timepoints):** and **Smooth window (timepoints):** (`5` each — judge the real time shown below them), and **n_states:** (`4`) or **State selection mode:** `auto`. Re-running the clustering deletes that cell type's previous `behavioral_states` results.
- **State Trajectory:** **Trajectory size (timepoints):** (`100`) and **N clusters:** (`6`).

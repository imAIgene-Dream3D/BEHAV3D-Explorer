# 🛒 Processing Queue

The Processing Queue is the collapsible panel at the bottom of the BEHAV3D EXPLORER dock widget, under the title bar **🛒 Processing Queue (N)** (N = number of queued steps). Click the title bar or the **▲** / **▼** button to open or close it. It batches steps from segmentation, tracking, feature extraction, filtering and analysis and runs them one after the other, in pipeline order, for every sample. Use it to leave a long pipeline running overnight.

## What it does

The queue:

- collects the steps you add with the **+🛒** buttons in the tabs, or with a preset;
- sorts them into pipeline order, whatever order you add them in;
- offers to add a missing upstream step (for example tracking before feature extraction);
- asks **once**, before starting, what to do with results that already exist;
- runs each step for all samples, in the background, showing progress per step and overall.

## Step types

| Step (as shown in the queue) | Added from | What it does |
|---|---|---|
| 🧠 Train Classifier | Segmentation → Pixel Classifier | Trains the pixel classifier. Also adds the Pixel Classifier segmentation step. |
| 🦠 Segmentation — APOC (GPU) / ConvPaint / CPU Pixel Classifier / Cellpose / Cellpose-SAM | Segmentation (each method's page) | Segments all samples. Cellpose and Cellpose-SAM steps show the population (or *all cell types*) they were queued for. |
| ☠ Dead Mask (Otsu) | Segmentation → Cellpose page | Computes the dead-cell mask from the dead-dye channel. |
| 📍 Batch Tracking | Tracking | Tracks all populations. |
| 🧪 Feature Extraction | Feature Extraction | Computes per-timepoint features. |
| 🔥 Active Killing — *immune type* | Feature Extraction → Active Killing | Detects killing events. |
| 🧹 Filtering | Filtering | Filters tracks and writes the filtered and summarised tables. |
| 💀 Death Dynamics / 💀 Combined Death Dynamics | Analysis → 👥 Population Dynamics | Per-target and combined death dynamics. |
| 🤝 Interaction Analysis / 🤝 Interaction Overview | Analysis → 👥 Population Dynamics | Per-target and combined interaction analysis. |
| 🧗 Invasiveness Analysis | Analysis → 👥 Population Dynamics | Invasiveness of an immune population into its targets. |
| 🔬 State Clustering | Analysis → 🧬 Single Cell → 🔬 Behavioral State | Fits the behavioural states. |
| 🛤️ Track Clustering | Analysis → 🧬 Single Cell → 🛤️ State Trajectory | Clusters whole trajectories. |

Each step type can be queued once; adding it again is ignored (Cellpose and Cellpose-SAM can be queued once per population). The **+🛒** next to **▶▶ Run All Available** in Population Dynamics adds every applicable analysis step at once. In Single Cell only the two clustering steps can be queued; train and apply the classifiers with their own run buttons.

```{note}
**Which settings does a queued step use?** APOC and ConvPaint segmentation, and the Population Dynamics and Active Killing steps remember their settings from the moment you click **+🛒**. Cellpose-SAM steps remember only the population(s), model, device settings, CPU threads and cooldown; Cellpose steps only the population and model. Everything else (for Cellpose and Cellpose-SAM: timepoints, channel selection, thresholds and size filter; the dead-mask timepoint range) and all settings of the other steps are read from their tab when the step starts. Set up every tab before you click **▶ Run Queue**.
```

## Adding steps

Click the **+🛒** button next to a step's run button in its tab. The queue opens automatically and the new step is sorted into place.

### Dependency prompts

When you add a step whose input data is missing for some samples, and the step that produces it is not queued yet, the queue asks whether to add it:

| You add | If missing | Dialog | **Yes** adds |
|---|---|---|---|
| 📍 Batch Tracking | segmentation | **Missing Segmentation Data** | Pixel Classifier segmentation |
| 🧪 Feature Extraction | tracking | **Missing Tracking Data** | 📍 Batch Tracking |
| 🧹 Filtering | features | **Missing Feature Data** | 🧪 Feature Extraction |
| 🔥 Active Killing | features of that immune type | **Missing Feature Data** | 🧪 Feature Extraction |

**No** cancels adding the step.

```{tip}
Add steps in pipeline order (segmentation first) to avoid these prompts. This matters especially if you do not segment with the Pixel Classifier: the tracking prompt always offers the Pixel Classifier, so queue your own segmentation method first.
```

## Presets

The **Preset:** drop-down fills the queue with a ready-made chain, replacing what was queued:

| Preset | Expands to |
|---|---|
| **Segment + Track** | Segmentation → Batch Tracking |
| **Segment → Filter** | Segmentation → Batch Tracking → Feature Extraction → Filtering |

The segmentation step is the method currently selected in the Segmentation tab. The line under the preset row shows it: **📍 Segmentation preset will use: …**. **Import** cannot be queued; with it selected, the presets are refused.

## Running the queue

Click **▶ Run Queue**. The metadata must be loaded, and nothing else may be running.

1. If results already exist for any queued step, one **Existing Data** dialog lists them for all steps, with **Overwrite All**, **Skip Existing** (or **Skip & Resume**), **Customize…** and **Cancel**. See [Running Steps](running_steps.md#when-results-already-exist).
2. The steps then run one after the other, each for every sample in the metadata:
   - the icon of each row goes ⏸️ pending → ⏳ running → ✅ done (or ❌ failed);
   - a small progress bar in the running row shows the progress of that step; when a step finishes its row shows the elapsed time (e.g. `12s`, `3.4m`);
   - the title bar and the line above **▶ Run Queue** show the overall progress (`Queue 2/4: …`), with a percentage and estimated time left.
3. While the queue runs, tabs are locked like for any [background step](running_steps.md#background-execution), and the queue cannot be edited.
4. If a step fails, a **Queue Error** dialog shows the error and the queue **stops**; the remaining steps stay pending.

```{note}
The queue refuses to run the Pixel Classifier segmentation and APOC segmentation together (**Queue Conflict: Segmentation Steps**); remove one of them.
```

## Removing / clearing

- **x** at the right of a row removes that step (pending steps only, not while the queue runs). Removing the Pixel Classifier segmentation step also removes its **🧠 Train Classifier** step.
- **Clear** empties the whole queue.
- The title bar text turns yellow while steps are queued.

## What the queue does *not* do

- It does not run samples in parallel: each step is heavy enough that running them one by one is usually fastest anyway.
- It does not restart after a crash, and the queue is not saved: after restarting napari, add the steps again. Use **Skip Existing** to continue where it stopped.

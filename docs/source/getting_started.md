# Getting Started

This page gets you from a fresh BEHAV3D EXPLORER install ([Installation](installation.md)) to a running pipeline.

## Launch the BEHAV3D EXPLORER plugin

::::{tab-set}

:::{tab-item} Recommended (one-click)

Double-click the launcher for your system in the `napari/` folder of BEHAV3D EXPLORER:

- Windows: `napari/run_behav3d_windows.bat`
- macOS: `napari/run_behav3d_macos.command`
- Linux: run `./napari/run_behav3d_linux.sh` from a terminal

The launcher starts the environment the installer created and opens napari (window title **BEHAV3D Explorer**) with the plugin already docked on the right. Keep the terminal window it opens: closing it closes napari. It shows the progress messages of every step, which are also saved in the `logs/` folder of BEHAV3D EXPLORER.

:::

:::{tab-item} Manual

```bash
mamba activate behav3d
napari
```

Use your own environment name if you chose another one during installation. Then open **Plugins → BEHAV3D Explorer** from the napari menu bar.
:::

::::

## The dock widget

On the right of the napari viewer you get the **BEHAV3D Explorer** panel and, next to it, the **BEHAV3D Assistant** panel ([QueenB](#co-pilot-assistant-queenb)). The BEHAV3D Explorer panel has seven tabs along its top, which you work through from left to right:

| # | Tab | What it does | Detailed page |
|---|---|---|---|
| 1 | 📋 **Data Preparation** | Choose the output directory, build or load `metadata.csv`, check the dimension order, convert the raw images to Zarr. | [Data Preparation](data_preparation.md) |
| 2 | 👁 **Visualization** | Load any sample (raw channels, segments, tracked segments, tracks) as napari layers; correct tracked segments by hand. Used after every step. | [Visualization](plugin_essentials/visualization.md) |
| 3 | 🦠 **Segmentation** | Six methods: APOC, ConvPaint, Pixel Classifier, Cellpose, Cellpose-SAM (zero-shot) and Import segmentation. | [Segmentation](processing/segmentation/index.md) |
| 4 | 📍 **Tracking** | One sub-tab per cell type: btrack, LapTrack or TrackPy (centroid-based), Fragmentation, Bounded or Reporter Propagation (overlap-based), or Import tracking. | [Tracking](processing/tracking/index.md) |
| 5 | 🧪 **Feature Extraction** | Per-timepoint contact, invasiveness, movement, intensity, morphology and death features for every cell; also runs Active Killing. | [Feature Extraction](analysis/feature_extraction.md) |
| 6 | 🧹 **Filtering** | Timepoint range and track-quality filters; writes the table every analysis reads. | [Filtering](analysis/filtering.md) |
| 7 | 📊 **Analysis** | Three sub-tabs: 🔬 Feature Backprojection, 👥 Population Dynamics and 🧬 Single Cell. | [Analysis](analysis/index.md) |

Tabs 2–7 show their options only once metadata is loaded in Data Preparation.

Below the tabs sits the **🛒 Processing Queue** panel (click its title bar to open it). It collects the steps you add with the **+🛒** buttons of tabs 3–7 and runs them one after the other for all samples, for example overnight. See [Processing Queue](plugin_essentials/processing_queue.md). Feature Extraction, Filtering and Analysis also show a [📄 Results panel](plugin_essentials/results_panel.md) to browse and open their result files.

```{note}
Steps run in the background. While one runs, a yellow **⏳ Running in the background: …** banner appears at the top of the panel and you cannot switch tabs or start another job. See [Running Steps](plugin_essentials/running_steps.md#background-execution).
```

## Co-pilot assistant (QueenB)

QueenB, the BEHAV3D EXPLORER co-pilot, lives in the **BEHAV3D Assistant** panel next to the main panel. No setup is needed. The round QueenB button (a crowned bee) in the bottom-right corner of the main panel hides or shows it; you can drag the button elsewhere.

- **Ask anything** in the box at the bottom, e.g. *"what does contact threshold do?"* or *"which tracking method fits cells that no longer overlap between frames?"*. QueenB reads what is open in the plugin (current tab, metadata including an unsaved Metadata Builder draft, settings, queue, logs and result files), so it does not ask you to repeat it. Try *"Explain this screen"*, or *"Is all ready?"* before running a step.
- **Quick actions** above the input box change with the tab:

  | Tab | Quick actions |
  |---|---|
  | 📋 Data Preparation | **Walk through setup**, **Check what's missing**, **Build metadata** (asks for your samples and cell types one question at a time and fills in the Metadata Builder) |
  | 👁 Visualization | **Explain this screen**, **Tell me about my data** |
  | 🦠 Segmentation | **Guide segmentation**, **Choose a method**, **Estimate EDT** (starting values for the **EDT threshold**, calculated from the XY pixel size in your metadata and a typical cell diameter, 10 µm unless you give another) |
  | 📍 Tracking | **Guide tracking**, **Choose tracking method** |
  | 🧪 Feature Extraction | **Guide setup**, **Check prerequisites** |
  | 🧹 Filtering | **Review filters**, **Track lengths**, **Cell counts** (see [Filtering](analysis/filtering.md#preview-track-lengths)) |
  | 📊 Analysis | **Choose analysis**, **Review results**; each analysis card also has a **💬  Ask the assistant** button |

- **It fills in forms, you confirm.** A suggested value appears under **Proposed changes** as a card showing the current and the new value: **Apply change** accepts it, **Keep current** rejects it, **Dismiss all** discards every pending card. Adding a step to the queue, creating a cell-type group, and saving or loading metadata are confirmed the same way. QueenB only acts without asking when nothing is changed: switching tab, opening an analysis or a result file, showing a preview or a calculation. In **Build metadata**, the answers you type go straight into the form.
- **New chat** clears the conversation. While an output directory is set, each QueenB reply is also appended to `assistant_log.md` in it.

```{note}
QueenB needs an internet connection: your questions and a text summary of the plugin state (never the images) are sent to the BEHAV3D EXPLORER assistant service. The status line under the top bar shows **Online** when it is reachable; **Check status** tests the connection. When the service cannot be reached, replies start with **Assistant offline** and QueenB can only look up parameter definitions (no guided help, no form-filling). It reconnects automatically once the service is back.
```

## Pipeline tour

This is the happy path. Each step links to its full reference page.

1. **Set up your project in 📋 Data Preparation** ([details](data_preparation.md))
   - In **1 · Output Directory**, pick the folder that will hold everything BEHAV3D EXPLORER writes.
   - New project: build `metadata.csv` in **2 · Metadata Builder** (**Configure Cell Types** → **Create Sample Forms** → fill in the forms → **Save Metadata CSV**), then click **Load Metadata**. Existing project: just load its `metadata.csv` in **3 · Metadata Loader**.
   - Check the axis order of each image in **5 · Dimension Order** (`TCZYX` when it cannot be detected).
   - Click **Convert All Images to Zarr** to convert the raw images for fast access.
2. **Look at the data in 👁 Visualization** ([details](plugin_essentials/visualization.md))
   - Pick a sample, click **Load Dataset into Napari**, and show or hide the Raw / Segments / Tracked Segments / Tracks layers.
   - Come back to this tab after every step below to inspect what it produced.
3. **Segment in 🦠 Segmentation** ([details](processing/segmentation/index.md))
   - Choose a method in the **Method:** dropdown: train a pixel classifier on a few brush strokes (APOC, ConvPaint, Pixel Classifier), use a pretrained Cellpose model, run Cellpose-SAM with no training, or import segmentations made elsewhere.
   - Check one sample in Visualization, then run all samples, or queue the step with **+🛒**.
4. **Track in 📍 Tracking** ([details](processing/tracking/index.md))
   - One sub-tab per cell type. Choose the method from how far your objects move between frames, not from what they are, or import existing tracks.
   - Correct remaining errors with **Manual Edition** in the Visualization tab ([details](processing/tracking/manual_editing.md)).
5. **Extract features in 🧪 Feature Extraction** ([details](analysis/feature_extraction.md))
   - For each cell type, tick the feature families you need; those that later analyses depend on are always extracted. With a dead channel, set the **Dead mask % threshold** and check it with **👁  Preview Dead Threshold in Viewer**.
   - Optional: **🧬  Extended Analysis — Active Killing (Immune Cells)** finds the contacts of an effector population (declared as immune, `im_`) that are followed by a rise of the target's death signal ([details](analysis/population_analysis/active_killing.md)).
6. **Filter in 🧹 Filtering** ([details](analysis/filtering.md))
   - Choose the timepoint range and the track filters (minimal and maximal track length, minimal size or dead at the first timepoint…); **📊 See Track Length Distributions** helps you choose the lengths.
   - Run Filtering even if you apply no filter: every analysis reads its output.
7. **Analyse in 📊 Analysis** ([details](analysis/index.md))
   - **🔬 Feature Backprojection**: paint any feature onto the cells, frame by frame, as a quick visual check ([details](analysis/feature_backprojection.md)).
   - **👥 Population Dynamics**: Death Dynamics, Interaction Analysis and Invasiveness Analysis ([details](analysis/population_analysis/index.md)).
   - **🧬 Single Cell**: **🔬 Behavioral State** assigns each cell a behavioural state at every timepoint, **🛤️ State Trajectory** groups whole tracks by how their behaviour unfolds; both can paint their results back onto the images ([details](analysis/single_cell/index.md)).
   - Population Dynamics and Single Cell open on a **Guided overview**: click **Start — open the settings  ▸** on the card of the analysis you want.

## Next steps

- → The whole pipeline on one page: [Quick protocol](quick_protocol.md)
- → Cross-tab tools: [Plugin Essentials](plugin_essentials/index.md)
- → Full reference: [Data Preparation](data_preparation.md), [Processing](processing/index.md), [Analysis](analysis/index.md)

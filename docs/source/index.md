# BEHAV3D EXPLORER Wiki

**BEHAV3D EXPLORER** is a toolkit for the behavioural analysis of cells in 3D fluorescent time-lapse microscopy, built for biologists: no coding required. It takes you from raw 3D movies to cell-by-cell behavioural readouts: segmentation → tracking → feature extraction → filtering → analysis.

**This wiki documents the napari plugin**, the graphical interface for day-to-day use in the lab. (Advanced users can also script the pipeline from Python with `notebooks/run_behav3d.ipynb`; that route is not covered here.)

## How you use BEHAV3D EXPLORER

[Install](installation.md) BEHAV3D EXPLORER, open the plugin (see [Getting Started](getting_started.md)) and work through the tabs of its panel from left to right:

```{mermaid}
flowchart LR
    subgraph Pipeline ["BEHAV3D EXPLORER"]
        direction LR
        DP["📋 Data Preparation"] --> Seg["🦠 Segmentation"] --> Trk["📍 Tracking"] --> FE["🧪 Feature Extraction"] --> Filt["🧹 Filtering"] --> Ana["📊 Analysis"]
    end
    Vz["👁 Visualization"] -.inspect.-> Pipeline
    PQ["🛒 Processing Queue"] -.batches.-> Pipeline
```

**Visualization** and the **Processing Queue** work across the whole pipeline: Visualization shows any sample and its results at any stage, and the Processing Queue runs segmentation, tracking, feature extraction, filtering and analysis steps one after the other for all your samples. In the **📊 Analysis** tab, **🔬 Feature Backprojection** paints any measured feature back onto the cells, and the Single Cell analyses can do the same with their behavioural states and trajectory clusters.

There is no one-size-fits-all pipeline: you pick the segmentation, tracking and analysis methods that fit your experiment and your computer. You don't have to figure the options out alone: the built-in **Co-pilot assistant (QueenB)** sits next to the panel on every tab, explains methods and parameters, and can fill in the forms for you after you confirm. See [Getting Started](getting_started.md#co-pilot-assistant-queenb).

## Where to next

```{grid} 2
:gutter: 3

::::{grid-item-card} 🚀 Installation
:link: installation
:link-type: doc

One-click installer for Windows / macOS / Linux, automatic GPU/CPU set-up, installer options, manual installation, troubleshooting.
::::

::::{grid-item-card} 🧭 Getting Started
:link: getting_started
:link-type: doc

Launch the plugin, learn the panel layout and the QueenB co-pilot, take the pipeline tour.
::::

::::{grid-item-card} ⚡ Quick protocol
:link: quick_protocol
:link-type: doc

The whole pipeline on one page: the minimal path and the few real choices at each tab.
::::

::::{grid-item-card} 🧰 Plugin Essentials
:link: plugin_essentials/index
:link-type: doc

Tools used at every step: Visualization, running steps, the Processing Queue, the Results panel, the output folder layout and moving projects.
::::

::::{grid-item-card} 📋 Data Preparation
:link: data_preparation
:link-type: doc

Choosing the output folder, building the metadata table, declaring your cell types, and converting raw images to Zarr for fast access.
::::

::::{grid-item-card} ⚙️ Processing
:link: processing/index
:link-type: doc

Segmentation (6 methods, including zero-shot Cellpose-SAM) and Tracking (btrack, Fragmentation Propagation, Bounded Propagation, Reporter Propagation, TrackPy, LapTrack, Import tracking): turning images into tracked cells.
::::

::::{grid-item-card} 📈 Analysis
:link: analysis/index
:link-type: doc

Feature Extraction, Filtering, and the Analysis tab: Feature Backprojection, Population Dynamics (Death Dynamics, Interaction, Invasiveness, Active Killing) and Single Cell (Behavioral State, State Trajectory).
::::

::::{grid-item-card} 🧠 Cellpose Training
:link: cellpose_training
:link-type: doc

Train your own Cellpose model with the `train_behav3d_cellpose.ipynb` notebook when the pretrained models aren't enough.
::::

```

## Table of contents

```{toctree}
:caption: Introduction
:maxdepth: 2

installation
getting_started
```

```{toctree}
:caption: Plugin Essentials
:maxdepth: 2

plugin_essentials/index
```

```{toctree}
:caption: Pipeline
:maxdepth: 3

quick_protocol
data_preparation
processing/index
analysis/index
```

```{toctree}
:caption: Tools
:maxdepth: 2

cellpose_training
recommended_settings
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

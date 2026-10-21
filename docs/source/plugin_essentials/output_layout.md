# Output Directory & File Layout

Every BEHAV3D EXPLORER step writes into one folder structure under the **output directory** you pick in the Data Preparation tab. This page is the reference for where things go.

## The output directory

You pick it once, in **Data Preparation → 1 · Output Directory**. All tabs read and write under that same folder, so one output directory = one project.

## Top-level tree

```text
<output_dir>/
├── metadata.csv                        # the metadata table (saved here by the Metadata Builder)
├── metadata.csv.bak                    # original copy, kept the first time paths are re-linked after a move
├── behav3d_parameters.yml              # project settings, saved automatically
│
├── images/
│   ├── PixelClassification/            # Pixel Classifier / APOC / ConvPaint classifiers and training labels
│   │                                   #   (incl. the PixelClassifier_TrainingData.zip training bundle)
│   └── <sample>/                       # one folder per sample
│       ├── <sample>.zarr                       # converted raw image
│       ├── <sample>_<cell_type>_segments.zarr  # segmentation, one per population
│       ├── <sample>_<cell_type>_segments.progress.json  # Cellpose-SAM resume record
│       ├── <sample>_mask_dead.zarr             # dead mask (only with a dead channel)
│       └── <sample>_<cell_type>_tracked.zarr   # tracked segments, one per population
│
├── trackdata/
│   └── <sample>/
│       └── <cell_type>/
│           ├── <sample>_<cell_type>_tracks.csv          # tracks table
│           ├── <sample>_<cell_type>_track_features.csv  # per-sample features (Feature Extraction)
│           └── intermediate_results/                    # per-sample feature parts
│
└── analysis/
    ├── <cell_type>/
    │   ├── track_features/             # feature tables: combined, _filtered, _summarized
    │   ├── quality_control/            # filtering QC plots and track-count tables
    │   ├── active_killing/             # immune populations: one <target>/ folder per target (+ combined/)
    │   ├── invasiveness_analysis/      # immune populations: invasiveness (single immune type)
    │   ├── results/                    # death dynamics (per target)
    │   ├── interaction_analysis/       # interaction analysis (per target)
    │   ├── behavioral_states/          # Single Cell → Behavioral State; its backprojection/ also
    │   │                               #   holds the State Trajectory backprojection exports
    │   ├── behavioral_trajectories/    # Single Cell → State Trajectory
    │   └── trajectory_features/        # per-timepoint features used by feature-based State Trajectory
    ├── <name>_merged/                  # cell-type group made in the Analysis tab (same sub-folders)
    ├── invasiveness_analysis/          # invasiveness run on several immune types together
    └── multi_organoid_comparison/      # combined death dynamics and interaction overview
```

```{note}
Folder and file names use the population name **without** its metadata prefix: `or_organoid1` and `im_tcell` become `organoid1` and `tcell`. Multicolor populations add one entry per colour (`tcell_1_multicolor`, …) plus the merged `tcell_merged`, and cell-type groups created in the Analysis tab get their own `<name>_merged` entries in the same places.

Interaction Overview and invasiveness file names end with a part recording the run's settings (selected types, time range, …), so runs with different settings do not overwrite each other. Re-running Behavioral State clustering deletes the whole `behavioral_states/` folder of that cell type, including the exported backprojections of both Single Cell analyses.
```

## Per-step writes

### Data Preparation → Convert to Zarr

| Writes | Path |
|---|---|
| Converted raw image | `images/{sample}/{sample}.zarr` |
| Conversion record (original dimensions, timepoint cut) | `behav3d_parameters.yml` |

### Segmentation

| Writes | Path |
|---|---|
| Segments | `images/{sample}/{sample}_{cell_type}_segments.zarr` |
| Cellpose-SAM resume record | `images/{sample}/{sample}_{cell_type}_segments.progress.json` |
| Dead mask | `images/{sample}/{sample}_mask_dead.zarr` |
| Trained pixel classifiers, training labels and training bundle (`PixelClassifier_TrainingData.zip`) | `images/PixelClassification/` |

### Tracking

| Writes | Path |
|---|---|
| Tracked segments | `images/{sample}/{sample}_{cell_type}_tracked.zarr` |
| Tracks table | `trackdata/{sample}/{cell_type}/{sample}_{cell_type}_tracks.csv` |

### Feature Extraction

| Writes | Path |
|---|---|
| Per-sample features and intermediate parts | `trackdata/{sample}/{cell_type}/` |
| Combined feature table (all samples) | `analysis/{cell_type}/track_features/BEHAV3D_{cell_type}_combined_track_features.csv` |
| Active Killing (immune only) | `analysis/{immune_type}/active_killing/{target}/`, and `…/active_killing/combined/` when several targets are selected |

### Filtering

| Writes | Path |
|---|---|
| Filtered and summarised feature tables | `analysis/{cell_type}/track_features/` (`…_filtered.csv`, `…_summarized.csv`) |
| Quality-control plots and track counts | `analysis/{cell_type}/quality_control/` |

### Analysis

| Writes | Path |
|---|---|
| Death Dynamics (per target) | `analysis/{target}/results/` |
| Interaction Analysis (per target) | `analysis/{target}/interaction_analysis/` |
| Invasiveness Analysis | `analysis/{immune_type}/invasiveness_analysis/`, or `analysis/invasiveness_analysis/` for several immune types together |
| Combined Death Dynamics / Interaction Overview | `analysis/multi_organoid_comparison/` |
| Behavioral State classification | `analysis/{cell_type}/behavioral_states/` |
| State Trajectory classification | `analysis/{cell_type}/behavioral_trajectories/` and `analysis/{cell_type}/trajectory_features/` |
| Backprojection exports (both Single Cell analyses) | `analysis/{cell_type}/behavioral_states/backprojection/` |
| Cell-type group table | `analysis/{name}_merged/track_features/` |

```{tip}
You rarely need to browse these folders by hand: the [Results panel](results_panel.md) at the bottom of the Feature Extraction, Filtering and Analysis tabs lists everything under `analysis/` and opens it for you.
```

## The project settings file

`behav3d_parameters.yml` is created in the output directory the first time you click **Load Metadata**, and updated automatically as you work. It stores:

- the location of the metadata.csv and of the output directory;
- the last settings you used in every tab, so they are restored when you load the project again;
- the shared **Workers** value (see [Running Steps](running_steps.md#workers));
- the colours and contrast of the raw channels in the Visualization tab;
- cell-type groups created in the Analysis tab;
- the Zarr conversion record: each sample's original shape and axis order, the timepoint cut (if any) and when the conversion was done.

It is a plain text file you can open to check which settings produced a result. Deleting it resets every tab to its defaults the next time you load the metadata.

## How tabs find prior outputs

Tabs find earlier results through the paths stored in metadata.csv and the fixed layout above. As a result:

- If you delete a `*_tracked.zarr`, that sample shows no tracked segments in the Visualization tab and counts as "missing" the next time you run tracking.
- If you move or copy the output directory, select the new location in Data Preparation and click **Load Metadata**: paths in metadata.csv are re-linked to the new folder automatically. To move a project to another drive or computer quickly and safely, see [Moving Projects](moving_projects.md).
- When you run a step, BEHAV3D EXPLORER uses the same layout to check which results already exist (see [Running Steps](running_steps.md#when-results-already-exist)).

## Importing pre-existing data

Segmentations or tracks made outside BEHAV3D EXPLORER are brought in with the **Import** option of the Segmentation or Tracking tab, which checks them and converts or copies them into this layout where needed (a valid segmentation Zarr is used where it is). See:

- [Import segmentation](../processing/segmentation/import.md)
- [Import tracking](../processing/tracking/import.md)

## Cleaning up

Safe to delete (recomputed the next time you run the step):

- `analysis/`: analysis outputs only (feature extraction, filtering and analyses must be rerun).
- A specific `*_tracked.zarr` and its `trackdata/` folder if you want to re-track.
- A specific `*_segments.zarr` if you want to re-segment.

Be careful with:

- `images/{sample}/{sample}.zarr`: the converted raw image. If you deleted the original raw file after conversion, this is your only copy.
- `images/PixelClassification/`: your trained classifiers and painted training labels. Keep them unless you intend to retrain.
- `behav3d_parameters.yml`: the saved settings (see above).
- `metadata.csv`: the project's index of samples and files.

# BEHAV3D Explorer

BEHAV3D Explorer is a tool for analyzing how cells behave in 3D microscopy images: no coding required, built for biologists.

It works with pretty much any 3D microscopy data, for example 3D live co-cultures or intravital microscopy, and turns your raw movies into quantitative, cell-by-cell readouts. Here's the kind of analysis you can get out of it:

<p align="center">
  <img src="docs/source/_static/icons/death_dynamics.jpg" width="200" alt="Death dynamics"/>
  <img src="docs/source/_static/icons/invasion.jpg" width="100" alt="Invasion"/>
  <img src="docs/source/_static/icons/active_killing.jpg" width="200" alt="Active killing"/>
  <img src="docs/source/_static/icons/single_cell_analysis.jpg" width="200" alt="Single cell analysis"/>
</p>

There's no one-size-fits-all solution, so BEHAV3D-Explorer is built as a set of flexible, modular tools: you pick and combine the segmentation, tracking and analysis methods that fit your experiment, or your computational power, instead of being locked into one fixed pipeline. Navigating a complex software with so many options is difficult, but you don't have to figure it out alone: **a built-in Co-pilot assistant** (QueenB) sits next to the panel, explains what each parameter does, and can fill in the forms for you.

📖 [Full wiki](https://imaigene-dream3d.github.io/BEHAV3D-Explorer/)

![BEHAV3D panel in napari](docs/source/_static/gif/Github_readme_GUI_intro_v3_cropped.gif)


> [!TIP]
> ## Try it online, no installation
>
> **This is a quick, hands-on preview of the BEHAV3D Explorer interface, not a way to run BEHAV3D itself.** It's free but limited by Colab's shared cloud resources (CPU-only, capped time and RAM). Install it locally for full capability.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imAIgene-Dream3D/BEHAV3D-Explorer/blob/main/demo/colab/BEHAV3D_Explorer_Colab.ipynb)
>
> Want to look before you install? The badge above opens the **real** BEHAV3D Explorer GUI on a free
> Google Colab machine and streams the napari window to your browser: `Runtime > Run all`, wait ~4
> minutes, and the GUI appears embedded in the notebook (or click **Link to the BEHAV3D Explorer
> tab** for a full browser tab). Nothing is installed on your computer, you get your own private
> session, and the demo dataset opens with segmentation and tracking already computed so you go
> straight to visualisation, track editing and the analysis tabs.
>
> Colab disconnects after ~90 minutes idle and saves nothing, so it is a test drive, not a workspace.
> For real analysis, install it below. Full details, and the maintainer guide for building the demo
> from source, are in [`demo/README.md`](demo/README.md).

## Installation (3 steps, no code)

1. Download this repository: green **Code > Download ZIP** button, then unzip it.
2. Open the `installation` folder and double-click the installer for your system:
   - **Windows:** `install_behav3d_windows.bat`
   - **macOS:** `install_behav3d_macOS.command`
3. Follow the on-screen prompts. The installer handles everything (conda, GPU, PyTorch, Cellpose) automatically.

> Using Linux, or prefer the terminal? See the manual installation and advanced options below.

## How to open the program

Double-click the launcher for your system, inside the `napari` folder:

- **Windows:** `napari/run_behav3d_windows.bat`
- **macOS:** `napari/run_behav3d_macOS.command`
- **Linux:** `napari/run_behav3d_linux.sh`

The BEHAV3D plugin opens automatically inside napari, covering the full pipeline: data preparation → segmentation → tracking → feature extraction → filtering.

## Files you need before starting

**Required:**
- Per sample, a microscopy image (.czi, .lif .tiff, .zarr).
- A [metadata.csv file](./configs/metadata.csv) listing the samples to analyze. You can build this directly inside BEHAV3D-Explorer with the built-in metadata editor, no need to prepare it beforehand. If you prefer editing it outside the app, on Windows we recommend https://www.moderncsv.com/, as it doesn't alter the format on save.

**Optional:**
- Already segmented data per sample.
- Already tracked data per sample.

See the exact `metadata.csv` structure [below](#metadatacsv-structure).

## Co-pilot assistant

No setup needed: the Co-pilot dock on the right of the napari panel is ready to use as soon as you open BEHAV3D. Ask it about any parameter or method, and confirm with one click to let it fill in the form for you.

Developers extending the assistant should start with the
[assistant architecture and change guide](chatbot/README.md).

---

## FAQ

<details>
<summary><strong>The installer doesn't work / PyTorch errors out</strong></summary>

Try one of these two options from the terminal, inside the `behav3d` environment:

**Option 1:** Install `nomkl` **before** installing PyTorch (avoids conflicts with OpenMP)
```bash
mamba install -c conda-forge igraph "blas=*=openblas"
mamba install nomkl
```

**Option 2:** Reinstall Cellpose with conda instead of pip
```bash
pip uninstall cellpose -y
mamba install -c conda-forge cellpose
```
</details>

---

## Technical documentation (advanced)

<details>
<summary><strong>Manual installation (Linux, or if you prefer the terminal)</strong></summary>

### Linux
```bash
cd path/to/BEHAV3D/installation
chmod +x install_behav3d_linux.sh
./install_behav3d.sh
```

### macOS via terminal
```bash
cd path/to/BEHAV3D/installation
chmod +x install_behav3d_linux.sh
./install_behav3d.sh
```

The installer automatically:
- Installs Miniforge (conda) if not found
- Detects your GPU (NVIDIA CUDA / Apple Silicon MPS / CPU-only)
- Creates the conda environment with all dependencies
- Installs PyTorch with the appropriate backend
- Installs Cellpose for cell segmentation

### Fully manual installation

**Step 1: Install Miniforge**

If you don't have conda/mamba installed, download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/) (recommended, includes `mamba`) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

> We recommend `mamba` over `conda` for faster dependency resolution. Miniforge includes it by default. All `mamba` commands below work the same with `conda` if you're using Miniconda/Anaconda.

**Step 2: Create the environment**

```bash
cd /path/to/BEHAV3D

# Create environment from the yml file
mamba env create -f environment.yml
mamba activate behav3d

# Install Cellpose and ConvPaint
pip install cellpose>=3.0 napari-convpaint

# Install PyTorch (choose ONE option based on your system):

# First remove any preexisting pytorch installations:
mamba remove pytorch torchvision torchaudio

# CPU only (all platforms):
mamba install pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 -c pytorch

# CUDA 12.1 (Windows/Linux with NVIDIA GPU):
mamba install pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install the BEHAV3D package
pip install -e .

# Register the Jupyter kernel
python -m ipykernel install --user --name=behav3d --display-name "behav3d"
```

### Installer options

```bash
# Check system info before installing
python install_behav3d.py --check

# Force CPU-only installation (even if a GPU is detected)
python install_behav3d.py --cpu-only

# Only install PyTorch/Cellpose (if the environment already exists)
python install_behav3d.py --pytorch-only
```

### Manual napari launch

```bash
mamba activate behav3d
napari
```
Then open the BEHAV3D plugin from the **Plugins** menu.

> The launcher reads its environment configuration from `napari/.config/behav3d_env.json`, generated by the installer.
</details>

<a id="metadatacsv-structure"></a><details>
<summary><strong>metadata.csv structure</strong></summary>

### Required
| Column Name                    | Explanation                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| sample_name                   | Name to assign to the sample; used for naming output files/folders.                                |
| organoid_line                 | Name of the multicellular structure line used in the experiment (e.g. `lineA`, `lineB`, such as a patient-derived organoid line, a spheroid line, etc.). Analysis visual results will be split on organoid lines                                     |
| tcell_line                    | Name or type of immune population used (e.g. `lineA`, such as a CAR-T product, an engineered T-cell line, a donor identifier). Analysis visual results will be split on tcell lines                                 |
| exp_nr                        | Experiment number for tracking and reproducibility.                                                |
| well                          | Well identifier on the experimental plate (e.g. `well01`).                                          |
| tcell_channel                 | Index of the fluorescence channel representing T cells in the raw image. (Channel indexing starts at 0)                                      |
| live_channel                  | Index of the fluorescence channel representing alive cells in the raw image. (Channel indexing starts at 0)             |
| dead_channel                  | Index of the fluorescence channel representing dead signal in the raw image. (Channel indexing starts at 0)                                  |
| contact_threshold             | Distance threshold (in `distance_unit`) to define T cell-organoid or T cell - T cell contact.                        |
| pixel_distance_xy             | Physical pixel spacing in the XY plane (usually in micrometers per pixel).                         |
| pixel_distance_z              | Physical spacing between Z slices (z-step), in same unit as `pixel_distance_xy`.                   |
| distance_unit                 | Unit for all provided spatial distances (e.g. `μm` for micrometers).                                        |
| time_interval                 | Time between consecutive frames (usually in seconds).                                              |
| time_unit                     | Unit of the time interval (e.g. `s` for seconds, 'm' for minutes, 'h' for hours).                                                  |
| raw_image_path                | Full path to the original CZI or raw imaging file.                                                 |

### Optional
| Column Name                    | Explanation                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| tcell_segments_image_path     | Path to the segmented T cell image (e.g. .zarr file).                                              |
| tcell_tracks_image_path       | Path to the image showing T cell tracks over time.                                                 |
| tcell_tracks_csv_path         | Path to the CSV file containing T cell tracking data.                                              |
| organoid_segments_image_path  | Path to the segmented organoid image (e.g. .zarr file).                                            |
| organoid_tracks_image_path    | Path to the image showing organoid tracks over time.                                               |
| organoid_tracks_csv_path      | Path to the CSV file containing organoid tracking data.                                            |

</details>

<details>
<summary><strong>Segmentation with Cellpose</strong></summary>

**Cellpose** performs 3D segmentation with pretrained models. You run it in the **Segmentation** tab of the napari plugin.

### Workflow in the napari plugin

1. **Get a model** (see [Pretrained Models](#pretrained-models) below). Models are single files with no extension.
2. **Prepare your data** in the **Data Preparation** tab: load or build `metadata.csv` and click **Convert All Images to Zarr** (samples that are not converted are skipped).
3. Open the **Segmentation** tab and pick **Cellpose (Deep Learning)** in **Method:**.
4. **Section 1 · Channel Configuration**: choose **Same for all** or **Per sample**, assign a population (or `dead`) to each channel, then click **💾 Save Channel Configuration**. Check this mapping carefully: the model only sees the channel mapped to the population you segment.
5. **Section 2 · Model & Cell Type**: **Browse…** to the model file and click **📂 Load Model**. Choose the **Cell type to segment** that matches the model. Keep **Process all timepoints** ticked, or set **Start:** / **End:**.
6. Click **▶ Run Cellpose — *cell type***. It runs in the background on every sample, with no preview, so check the first result in the **Visualization** tab before processing everything.
7. To segment another population (e.g. organoids and T cells), repeat steps 5–6 with the matching model.
8. **Dead-stain channel?** Click **☠ Run Dead Mask (Otsu)** to create the dead mask used later by Feature Extraction.
9. Optionally, queue the runs with the **+🛒** buttons to launch them together with the rest of the pipeline.

> [!NOTE]
> A run on a timepoint range replaces the whole segmentation of that population: timepoints outside the range are left empty. Test a model on a short range, then run all timepoints.

Full details, every control, tuning tips and output paths are in the wiki: [Cellpose segmentation page](https://imaigene-dream3d.github.io/BEHAV3D-Explorer/processing/segmentation/cellpose.html).

### Pretrained Models
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18872977.svg)](https://doi.org/10.5281/zenodo.18872977)

You can find and download pretrained T cell and Organoid Cellpose models on [Zenodo](https://doi.org/10.5281/zenodo.18872977) (together with the demo dataset and the manuscript config files) or in the `behav3d/preprocessing/segmentation/models` folder.


### Prerequisites

1.  **Metadata**: Ensure your `metadata.csv` is correctly populated with `sample_name`, `raw_image_path`, and cell type categories (Organoid, Immune, Other).
2.  **Image Format**: Input images should be converted to **Zarr** format (using **Convert All Images to Zarr** in the napari Data Preparation tab) for efficient multiprocessing.


---

*Note: To train a new Cellpose model, refer to the `train_behav3d_cellpose.ipynb` notebook.*

</details>

<details>
<summary><strong>Tracking with btrack</strong></summary>

## btrack Tracking

[btrack](https://github.com/quantumjot/btrack) is a Bayesian multi-object tracking library designed for cell tracking in time-lapse imaging. It uses a motion model and object features to build consistent cell trajectories across timepoints, handling cell divisions, disappearances, and re-appearances robustly.

BEHAV3D uses btrack to link segmented cell instances across frames into full tracks, producing per-cell trajectory data used in downstream behavioral analysis.

### Provided Default Configurations

Pre-configured btrack motion model config files (`.json`) for common cell types are included in the `behav3d/preprocessing/tracking/models/` folder:

| Config file | Suitable for |
|---|---|
| `cell_config.json` | Generic cell tracking (moderate speed, large objects) |
| `particle_config.json` | Fast, small objects (e.g. T cells, immune cells) |

These configs define the motion model parameters (e.g., max search radius, noise covariance) and are passed directly to the btrack `BayesianTracker`.

### Workflow in the napari plugin

1. Open the **Tracking** tab and pick **btrack** as the method.
2. Choose a **Config preset**: start with **Cell (bundled)**, and try **Particle (bundled)** or **Custom JSON…** (your own `.json` file) only if the result is poor.
3. Set **Max search radius** to the fastest plausible one-frame displacement plus a margin of 10-25 %.
4. Run the tracking and check the tracks in the **Visualization** tab.

All parameters, tuning tips and troubleshooting are on the wiki: [btrack page](https://imaigene-dream3d.github.io/BEHAV3D-Explorer/processing/tracking/btrack.html).

### Output Files

| File | Contents |
|---|---|
| `{sample_name}_{cell_type}_tracks.csv` | Per-cell track data (position, time, track ID) |
| `{sample_name}_{cell_type}_tracks.zarr` | Tracked labels as a 4D array (T, Z, Y, X) |

These files are automatically picked up by the downstream **Analysis** module to compute behavioral metrics (speed, contact duration, killing events, etc.).

---

*Note: For custom motion models, refer to the [btrack documentation](https://btrack.readthedocs.io).*

</details>

<details>
<summary><strong>Chatbot deployment (maintainers only)</strong></summary>

## Deploying / Managing the Chatbot

The co-pilot runs as a **CPU-only [Modal](https://modal.com) service** that proxies requests to the DeepSeek API. The DeepSeek key stays server-side; the endpoint is public. This section is only relevant if you are managing the hosted service.

### Prerequisites

- A [Modal](https://modal.com) account with the CLI installed (`pip install modal`)
- A [DeepSeek API](https://platform.deepseek.com) key
- Authenticated: `python -m modal token new`

### 1. Store the DeepSeek API key as a Modal secret

```bash
python -m modal secret create deepseek-api-key DEEPSEEK_API_KEY=<your-key>
```

> Set a **monthly spend limit** in the DeepSeek dashboard to cap costs.

### 2. Build the RAG index

The RAG index embeds all BEHAV3D documentation and parameter descriptions so the bot can retrieve relevant context for every question.

```bash
python -m modal run chatbot/app.py::ingest
```

Re-run this whenever files in `chatbot/knowledge/` or `chatbot/schema_cards.json` change.

### 3. Deploy

```bash
python -m modal deploy chatbot/app.py
```

The deploy output will print the endpoint URL. Update `napari/assistant_config.json` if it changes:

```json
{ "endpoint": "https://<your-modal-url>.modal.run", "timeout": 60 }
```

The service runs on CPU with `min_containers=1` (always warm, ~$6/month at default specs). No GPU or model weights are downloaded, so cold start only takes a few seconds.

### 4. Development / hot-reload

```bash
python -m modal serve chatbot/app.py
```

This starts a temporary endpoint that hot-reloads on file changes. Useful for testing prompt changes without a full deploy.

### Changing the model

The service defaults to `deepseek-v4-flash`. To switch models, set the `DEEPSEEK_MODEL` environment variable before deploying:

```bash
DEEPSEEK_MODEL=deepseek-v4-pro python -m modal deploy chatbot/app.py
```

Available options: `deepseek-v4-flash` (fast, cheap), `deepseek-v4-pro` (stronger, pricier).

### Researchers' own providers

Researchers can choose OpenAI, Anthropic, Gemini, Mistral or DeepSeek with their own key in the assistant's **⚙** settings. The plugin calls the provider directly, so their key never reaches this service and their usage is not billed to your DeepSeek account. See `chatbot/README.md`.

### Monitoring costs

- DeepSeek usage: [platform.deepseek.com](https://platform.deepseek.com) → Usage
- Modal compute: [modal.com/apps](https://modal.com/apps), under the `behav3d-assistant` app
- Set a DeepSeek spend limit to prevent runaway costs from unexpected traffic

</details>

<details>
<summary><strong>Developer options</strong></summary>

## Developer Options

### Developer Mode

BEHAV3D supports a lightweight developer mode that can be activated without modifying any code.

**How to enable:**

Create an empty file named `.behav3d_dev` in the BEHAV3D project root directory (the same folder that contains `README.md`):

```bash
# Linux / macOS
touch .behav3d_dev

# Windows PowerShell
New-Item -ItemType File .behav3d_dev
```

**What it does:**

| Effect | Detail |
|---|---|
| Window title | Adds `[DEV MODE]` to the napari window title bar |
| Environment variable | Sets `BEHAV3D_DEV_MODE=1` for the duration of the session |
| Console output | Enables extra verbose logging in some pipeline stages |
| BEHAV3D Assistant | Can discuss and set developer options; without dev mode it leaves them out of its answers |

**How to disable:**

Delete or rename the `.behav3d_dev` file:

```bash
# Linux / macOS / Windows
del .behav3d_dev        # Windows CMD
Remove-Item .behav3d_dev  # Windows PowerShell
```

**For developers, checking dev mode in code:**

```python
import os
if os.environ.get("BEHAV3D_DEV_MODE") == "1":
    print("Running in developer mode")
```

> **Note:** The `.behav3d_dev` file is listed in `.gitignore` and will never be committed to version control. It is a purely local, per-machine toggle.

</details>

---

## About AI Usage

*The BEHAV3D Explorer toolkit was developed with the assistance of AI models from OpenAI (Codex, ChatGPT), Anthropic (Sonnet, Haiku, Opus), Cursor and Google (Gemini Flash, Gemini Pro).*
*All AI model code edits have been supervised by our team of developers and have been tested thoroughly among the team of developers and by outside testers.*

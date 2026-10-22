# Installation

BEHAV3D EXPLORER runs on Windows, macOS and Linux. The easiest way to install it is the automatic installer, which sets up everything for you; manual steps are given below for those who want full control.

```{tip}
**Want to look before you install?** The [Colab demo](https://colab.research.google.com/github/imAIgene-Dream3D/BEHAV3D-Explorer/blob/main/demo/colab/BEHAV3D_Explorer_Colab.ipynb) opens the real BEHAV3D EXPLORER interface in your browser on a free Google Colab machine, with a demo dataset that is already segmented and tracked. It is a CPU-only test drive that saves nothing, not a way to analyse your own data.
```

## 📋 Requirements

- **Operating system**: Windows 10/11 (x64), macOS (Intel or Apple Silicon) or Linux (x64).
- **Internet connection** during installation, and several GB of free disk space.
- **Conda** (Miniforge recommended): if none is found, the installer offers to install Miniforge for you. The installer creates its own environment with Python 3.12, so the Python version already on your computer does not matter.
- **GPU (optional but recommended)**: an NVIDIA GPU with an up-to-date driver (Windows/Linux) speeds up Cellpose, Cellpose-SAM and ConvPaint a lot; Apple Silicon Macs use their built-in GPU automatically. Everything also runs on the CPU, only slower. APOC needs OpenCL drivers for your GPU (see [Segmentation](processing/segmentation/index.md)).

## 📁 Files overview

Download the repository (green **Code → Download ZIP** button on GitHub, then unzip it). The installer files are in its `installation/` folder:

| File | Description |
|------|-------------|
| `install_behav3d_windows.bat` | Windows installer (double-click to run) |
| `install_behav3d_macOS.command` | macOS installer (double-click to run) |
| `install_behav3d_linux.sh` | Linux installer (also works on macOS from a terminal) |
| `install_behav3d.py` | The cross-platform installer the three files above start; can also be run directly |
| `environment.yml` | Conda environment specification (used by the installer and for manual installation) |
| `requirements.txt` | Pip requirements (alternative to conda, advanced) |

## 🚀 Quick start

### Windows

**Option 1 (recommended):** double-click `install_behav3d_windows.bat`.

**Option 2:** open a Command Prompt and run:

```bat
cd path\to\BEHAV3D-Explorer\installation
install_behav3d_windows.bat
```

### macOS

**Option 1 (recommended):** double-click `install_behav3d_macOS.command`.

**Option 2:** if macOS refuses to open it, open Terminal and run:

```bash
cd path/to/BEHAV3D-Explorer/installation
bash install_behav3d_macOS.command
```

### Linux

```bash
cd path/to/BEHAV3D-Explorer/installation
bash install_behav3d_linux.sh
```

### What the installer does

Without options, the installer first asks for the **environment name** (press Enter to keep `behav3d`), then:

1. Detects your system and GPU: **NVIDIA (CUDA)**, **Apple Silicon (MPS)** or **CPU only**.
2. Finds conda (it prefers `mamba`, then `micromamba`, then `conda`). If none is found it offers to install **Miniforge**; you may then be asked to close the window and start the installer again.
3. **Windows only:** checks *long path support* (see the warning below).
4. If the environment already exists, asks whether to **[1] Remove and reinstall** it (clean install), **[2] Keep it and only update PyTorch/Cellpose**, or **[3] Cancel**.
5. Creates the conda environment with all dependencies, including napari.
6. Installs **PyTorch** for your hardware: with CUDA support when an NVIDIA GPU is detected, with Apple-GPU (MPS) support on macOS, otherwise the CPU-only version.
7. Installs **Cellpose v3** (the version used by the Cellpose method and its models), **ConvPaint** and the **BEHAV3D EXPLORER** plugin itself, plus the OpenCL support APOC needs on macOS/Linux.
8. Checks the installation and prints a summary (PyTorch version, `CUDA: True/False`, Cellpose, ConvPaint, napari, BEHAV3D EXPLORER plugin).
9. Saves the environment it used for the launchers in `napari/`, then asks whether to **launch napari now**.

```{warning}
**Windows: long path support.** Windows limits file paths to 260 characters by default, which makes some packages fail to unpack. If long path support is off, the installer offers to **[1] Enable long path support now** (recommended; Windows asks for **administrator** permission), **[2] Continue without it** (the installer then uses `conda` instead of `mamba`: slower, but avoids long paths) or **[3] Cancel installation**. If you have no administrator rights, choose 2 or ask your IT department to enable long paths.
```

```{note}
**Cellpose-SAM is set up separately.** The [Cellpose-SAM](processing/segmentation/cellpose_sam.md) method needs **cellpose v4**, which cannot share an environment with the cellpose v3 used everywhere else. It goes into a small one-time side environment that reuses your PyTorch install and leaves cellpose v3 and your trained models untouched. Either click **Set up Cellpose-SAM environment** the first time you open the Cellpose-SAM method in the plugin, or add `--with-cellpose-sam` when running the installer. You only need it if you intend to use Cellpose-SAM.
```

## ⚙️ Installation options

All installers accept these options (add them after the installer name, e.g. `bash install_behav3d_linux.sh --cpu-only` or `install_behav3d_windows.bat -n my_env`):

| Option | Description |
|--------|-------------|
| `-n NAME` / `--name NAME` | Environment name (default: `behav3d`). Without any option, the installer asks for it. |
| `--cpu-only` | Install the CPU-only version of PyTorch, even if a GPU is detected. |
| `--reinstall` | If the environment exists, remove and recreate it without asking. |
| `--keep-existing` | If the environment exists, keep it and only update PyTorch/Cellpose (and re-install the plugin) without asking. |
| `--pytorch-only` | Only (re)install PyTorch, Cellpose and ConvPaint into an existing environment. |
| `--with-cellpose-sam` | Also set up the Cellpose-SAM side environment. |
| `--skip-behav3d` | Do not install the BEHAV3D EXPLORER plugin itself (advanced). |
| `--check` | Only show system information (OS, GPU, recommended PyTorch version); install nothing. |

### Examples

These examples run the Python installer directly; type them in the `installation` folder.

```bash
# Check what the installer detects (GPU etc.), without installing
python install_behav3d.py --check

# Install with a custom environment name
python install_behav3d.py -n my_behav3d_env

# Force a CPU-only installation
python install_behav3d.py --cpu-only

# Start again from scratch
python install_behav3d.py --reinstall
```

```{note}
The launchers in `napari/` always open the environment set up by the **most recent** installer run, whatever its name.
```

## 🔧 Manual installation

Use this only if the installer does not work for you. On Windows, prefer the installer: it also applies two Windows-specific fixes that a manual installation lacks (see [Troubleshooting](#napari-does-not-start-on-windows)).

### Step 1: Install Miniforge

Download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/) (recommended, includes `mamba`) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```{tip}
`mamba` resolves dependencies much faster than `conda`. All `mamba` commands below also work with `conda`.
```

### Step 2: Create the environment

```bash
cd path/to/BEHAV3D-Explorer/installation
mamba env create -f environment.yml
mamba activate behav3d
# Avoid OpenMP/MKL conflicts with PyTorch (the installer does this too)
mamba install -c conda-forge igraph "blas=*=openblas"
mamba install nomkl
```

### Step 3: Install PyTorch, Cellpose and ConvPaint

```bash
# PyTorch: choose ONE line
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (Windows/Linux)
pip install torch torchvision torchaudio                                                     # macOS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

pip install "cellpose[gui]==3.1.1.2" napari-convpaint
```

For APOC, also install the OpenCL loader: `mamba install -c conda-forge ocl_icd_wrapper_apple` on macOS, `mamba install -c conda-forge ocl-icd-system` on Linux (not needed on Windows).

### Step 4: Install BEHAV3D EXPLORER itself

```bash
cd path/to/BEHAV3D-Explorer
pip install -e .
```

The one-click launchers need the settings file the installer writes; after a manual installation, start the plugin the manual way (see [Getting Started](getting_started.md)).

## ✅ Verify the installation

The installer ends with a summary: check that the plugin line reads `Plugin OK` and, on a computer with an NVIDIA GPU, that it shows `CUDA: True`. Then launch the plugin as described in [Getting Started](getting_started.md). If the **BEHAV3D Explorer** panel shows its seven tabs (📋 Data Preparation · 👁 Visualization · 🦠 Segmentation · 📍 Tracking · 🧪 Feature Extraction · 🧹 Filtering · 📊 Analysis) with the **🛒 Processing Queue** below them, and the QueenB assistant panel opens next to it, you are good to go.

```{note}
**Jupyter notebooks.** Jupyter is installed in the same environment. You only need it to train your own Cellpose models with `notebooks/train_behav3d_cellpose.ipynb` (see [Cellpose Training](cellpose_training.md)): open it in [VS Code](https://code.visualstudio.com/Download) (Python and Jupyter extensions) with the `behav3d` environment as kernel, or run `mamba activate behav3d` and then `jupyter notebook`.
```

## ❓ Troubleshooting

### "CUDA not detected" although you have an NVIDIA card

1. Make sure the NVIDIA driver is installed and up to date: run `nvidia-smi` in a terminal; it should list your card. The installer uses the same check.
2. Run `python install_behav3d.py --check` to see what the installer detects.
3. Re-install PyTorch with `python install_behav3d.py --keep-existing`, or start again with `--reinstall`.

### Conda not found

The installer offers to download and install Miniforge. Afterwards, close the window (or terminal) and start the installer again so that conda is found.

### Environment already exists

```bash
python install_behav3d.py --reinstall       # wipe and recreate
python install_behav3d.py --keep-existing   # keep it, update PyTorch/Cellpose only
python install_behav3d.py -n behav3d_v2     # create a second environment under a new name
```

### Windows: "extraction failed" / "remove_all: The directory is not empty"

Some packages have paths longer than Windows' 260-character limit. Re-run the installer and accept **Enable long path support now**, or enable it by hand in an **administrator** terminal:

```bat
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Then delete the half-unpacked package cache (a normal delete fails on the long paths) and run the installer again:

```bat
rmdir /s /q "\\?\%USERPROFILE%\miniforge3\pkgs\https"
```

### napari does not start on Windows

If napari stops with `DLL load failed while importing QtCore`, or with `Freetype library not found`, re-run the installer with `--keep-existing`: it re-applies the Windows fixes for these two problems.

### The launcher closes with "behav3d_env.json not found" or "Package manager not found"

The launcher cannot find the environment: the installer has not finished, or conda or the environment was moved or deleted. Run the installer again.

### BEHAV3D Explorer is missing from the Plugins menu, or import errors

Re-install the plugin into the environment:

```bash
mamba activate behav3d
cd path/to/BEHAV3D-Explorer
pip install -e .
```

### OpenMP/MKL conflicts (`nomkl` fails to install)

Only relevant for manual installations; the installer handles it. If installing `nomkl` fails with `LibMambaUnsatisfiableError`, switch the BLAS backend first:

```bash
mamba install -c conda-forge igraph "blas=*=openblas"
mamba install nomkl
```

### Cellpose errors

BEHAV3D EXPLORER needs **cellpose 3.1.1.2** in its main environment; do not upgrade it to cellpose 4 there (Cellpose-SAM has its own environment, see above). To repair it:

```bash
mamba activate behav3d
pip install "cellpose[gui]==3.1.1.2"
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above.
2. Run `python install_behav3d.py --check` and keep the output.
3. Open an issue on the [BEHAV3D EXPLORER GitHub repository](https://github.com/imAIgene-Dream3D/BEHAV3D-Explorer/issues), with that output and, for problems after launching, the latest file from the `logs/` folder of BEHAV3D EXPLORER.

## 📂 What you need to start

- **Per sample, a raw microscopy movie**: `.czi`, `.lif`, `.tif`/`.tiff`, `.ims` or `.zarr`.
- **A `metadata.csv`** listing your samples. You don't need to prepare it beforehand: the plugin's Metadata Builder creates it (see [Data Preparation](data_preparation.md)).
- Optional: segmentations or tracks made with other software, which you can bring in with [Import segmentation](processing/segmentation/import.md) and [Import tracking](processing/tracking/import.md).

```{tip}
If you edit `metadata.csv` outside BEHAV3D EXPLORER on Windows, we recommend [Modern CSV](https://www.moderncsv.com/): unlike Excel, it doesn't silently reformat values.
```

## Next step

→ [Getting Started](getting_started.md)

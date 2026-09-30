# BEHAV3D Explorer — online demo

Try the **real** BEHAV3D Explorer GUI in your browser, with no installation, for free, on Google
Colab. **Part 1** is for anyone who just wants to run it. **Part 2** is the maintainer guide for
building and hosting the demo from source.

---

# Part 1 · Run the demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imAIgene-Dream3D/BEHAV3D-Explorer/blob/main/demo/colab/BEHAV3D_Explorer_Colab.ipynb)

1. Click the badge above — it opens `demo/colab/BEHAV3D_Explorer_Colab.ipynb` in Google Colab
   (you need a Google account; nothing is installed on your computer).
2. **Runtime ▸ Run all** and wait ~4 minutes while the prebuilt environment and the demo dataset
   download.
3. Click the link the last cell prints — the actual napari GUI opens in a **new browser tab**,
   streamed from your own private Colab machine.

**What you get.** The identical GUI to a local install — same window, menus, dock widget and layer
list. The demo dataset ships with **segmentation and tracking already computed**, so you land
straight on the interesting parts: visualisation, track editing, feature extraction and the
behaviour analysis tabs.

**Good to know.**

* It runs entirely on **CPU** — no GPU runtime needed. Software OpenGL makes 2D views smooth and
  the 3D view slow, so stay in 2D for a comfortable tour.
* Colab disconnects after ~90 minutes idle (12 h max) and **saves nothing** — it is a test drive,
  not a workspace. Download anything you want to keep.
* To point the demo at **your own data**, see the *Using your own data* section at the bottom of
  the notebook.

For real analysis, [install BEHAV3D locally](../README.md#installation).

---

# Part 2 · Build the demo from source (maintainer guide)

```
      Colab VM (free, ~12.7 GB RAM, CPU-only)
 ┌───────────────────────────────────────────────────────────┐
 │  napari (PyQt5) + BEHAV3DWidget                           │
 │        │ renders into                                     │
 │  Xvfb :99  (software OpenGL / llvmpipe)                   │
 │        │ captured by                                      │
 │  x11vnc :5900  ──►  websockify/noVNC :6080                │
 └────────────────────────────────┬──────────────────────────┘
                                  │ Colab port proxy (or a Cloudflare quick tunnel)
                                  ▼
                     the visitor's browser tab
```

The visitor sees the identical GUI to a local install. Nothing in `behav3d/` is reimplemented;
only `$DISPLAY` differs. napari is started through the repository's own launcher,
`napari/.config/launch_napari.py --internal`.

### Files here

| File | What it is |
|---|---|
| `colab/BEHAV3D_Explorer_Colab.ipynb` | What visitors open. Three steps, "Run all". |
| `colab/colab_setup.py` | All the machinery: apt, environment, data, display stack, launch, tunnel fallback, health checks. |
| `colab/prepare_demo.py` | Rewrites the absolute paths inside a demo bundle for the machine it landed on. |
| `colab/repair_env.py` | Reconciles packages that conda and pip disagree about after conda-pack. Run by the bootstrap; `--check` is the same comparison as a guard. |
| `colab/apt_packages.txt` | The headless-Qt / OpenGL / VNC package list, shared by Colab and the Docker test. |
| `build_env.sh` | Builds the prebuilt environment tarball (and can run a local dry-run of the GUI). |

**Cost: $0.** Colab's free tier runs the demo; Zenodo and Hugging Face dataset repos host the
files for free. Visitors need a Google account (unavoidable at zero cost).

---

## Step 1 — Build the environment tarball

Solving `installation/environment.yml` inside Colab takes 10+ minutes and breaks whenever
conda-forge moves. Solve it once here instead, inside Ubuntu 22.04 (Colab's own base), and ship the
result as a [conda-pack](https://conda.github.io/conda-pack/) archive that restores in ~2 minutes.

```bash
./demo/build_env.sh
```

* Needs Docker. Takes 20–40 min the first time.
* Produces `dist/behav3d-env.tar.gz` (~2.5 GB).
* CPU-only by design: the Colab X display has no GPU, so PyTorch is always the CPU wheel and a
  CUDA build would only make the download bigger for no benefit.
* The build fails loudly if `napari`, `torch`, `cellpose`, `pyopencl` or `qtpy` cannot be imported —
  better here than in front of a visitor.

## Step 2 — Host the tarball

A free Hugging Face account can host **dataset** repos (only *Spaces* need PRO), files up to 50 GB,
on a fast CDN. GitHub Releases cap at 2 GB per file, which this tarball exceeds.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create behav3d-demo-env --type dataset
huggingface-cli upload <your-org>/behav3d-demo-env dist/behav3d-env.tar.gz \
    behav3d-env.tar.gz --repo-type dataset
```

(`huggingface_hub` 0.34+ renames this command to `hf`, e.g. `hf upload ...`; the CI workflow uses that newer name.)

The direct URL is then:

```
https://huggingface.co/datasets/<your-org>/behav3d-demo-env/resolve/main/behav3d-env.tar.gz
```

## Step 3 — Assemble the demo bundle

Run BEHAV3D locally once on the demo sample so the heavy steps are already done, keeping
**everything under one common folder** (the path rewriter finds the old root as the common parent
of all recorded paths):

```
behav3d_demo/
├── raw/                     # the ~400 MB images
├── metadata.csv             # produced by the Data Preparation tab
├── behav3d_parameters.yml   # the GUI's parameter snapshot, written in the output folder
└── analysis/ ...            # segmentation, tracking, features, analysis results
```

The **output folder is whatever folder you pointed the GUI at** - it has no required name and is
normally the bundle root itself, as above. `prepare_demo.py` records the pair it settled on in
`behav3d_parameters.yml`, and `colab_setup.print_paths()` reads it back, so the two paths a visitor
pastes are always the real ones.

Trim it so it stays pleasant on a free runtime: one sample, cropped in Z/T if needed, target
≤ 500 MB raw. Then:

```bash
tar -czf behav3d_demo.tar.gz behav3d_demo/
```

Sanity-check the rewriter before uploading anything:

```bash
python demo/colab/prepare_demo.py --root /path/to/behav3d_demo --dry-run
```

It should report the old root, the number of paths it would rewrite, and **no missing files**.

## Step 4 — Publish the bundle on Zenodo

Zenodo is free, gives 50 GB per record and a citable DOI, and you already use it for the Cellpose
models (record 18872978). Create a new record, upload `behav3d_demo.tar.gz`, publish, then take the
direct file URL:

```
https://zenodo.org/records/<RECORD_ID>/files/behav3d_demo.tar.gz?download=1
```

## Step 5 — Fill in the two URLs

In `demo/colab/colab_setup.py`, replace the two placeholders:

```python
ENV_URL  = "https://huggingface.co/datasets/<your-org>/behav3d-demo-env/resolve/main/behav3d-env.tar.gz"
DEMO_URL = "https://zenodo.org/records/<RECORD_ID>/files/behav3d_demo.tar.gz?download=1"
```

Both are also overridable at runtime via `BEHAV3D_ENV_URL` / `BEHAV3D_DEMO_URL`, which is how the
notebook lets visitors point at their own data. The bootstrap refuses to run while a placeholder is
still in place, with a message saying which step to go back to.

## Step 6 — Dry run locally (do this before touching Colab)

This catches every Qt / xcb / OpenGL problem on your own machine, in the same Ubuntu base Colab
uses, in about a minute:

```bash
./demo/build_env.sh --test
```

Then open the URL the script prints:

```
http://localhost:6080/vnc.html?autoconnect=true&resize=scale&reconnect=true&quality=6
```

The query parameters matter: `resize=scale` makes noVNC shrink the 1920x1080 virtual desktop to fit
your browser window, so the whole GUI is visible with no scrollbars. Opening the bare `/vnc.html`
gives you a 1:1 view and scrollbars instead.

Two options for the dry run:

```bash
./demo/build_env.sh --test --screen 1600x900          # match your own screen: crisp, unscaled
./demo/build_env.sh --test --data /path/to/behav3d_demo   # mount a bundle so the GUI has data
```

`--data` mounts the folder at `/data` and runs `prepare_demo.py` on it, which rewrites its
`metadata.csv` **in place** for the container paths (the original is kept as
`metadata.original.csv`). The run then prints the two paths to paste into the Data Preparation tab.

The napari window with the BEHAV3D Explorer dock widget must appear. If it does not, read
`/tmp/behav3d_logs/napari.log` inside the container.

## Step 7 — Test on Colab

Push to `main`, then open:

```
https://colab.research.google.com/github/imAIgene-Dream3D/BEHAV3D-Explorer/blob/main/demo/colab/BEHAV3D_Explorer_Colab.ipynb
```

Check, in order:

1. Run all completes in under ~5 minutes on a cold runtime.
2. The new tab shows the GUI and it is **interactive** — drag a slider, open a menu, switch tabs.
3. The two paths printed by Step 2 load in the Data Preparation tab.
4. Run the `start_cloudflared()` cell and confirm that route works too.
5. Watch RAM in the Colab resource panel with the demo loaded — stay under ~10 GB.

> The demo tracks `main`. The branch name `main` is baked into: the Colab link above, the badge in
> Part 1, the badge in the root `README.md`, and the notebook's `REPO_REF` + its screenshot URL, and
> `REPO_REF` in `colab_setup.py` (`grep -rn "blob/main\|BEHAV3D_REPO_REF" demo` finds them). If you
> ever stage demo changes on a separate branch, point those at it while testing, then back to `main`.

## Step 8 — Maintenance

The tarball only needs rebuilding when `installation/environment.yml` changes; nothing else moves.
`.github/workflows/build-demo-env.yml` handles it two ways:

* **automatically** — any push to `main` that touches `installation/environment.yml` rebuilds the
  tarball and re-uploads it to Hugging Face. No action needed.
* **manually** — Actions ▸ *Build Colab demo environment* ▸ *Run workflow*, with a checkbox to
  build without uploading (the tarball is then kept as a workflow artifact for 7 days).

Both paths need these two repository settings:

| Setting | Kind | Where | Value |
|---|---|---|---|
| `HF_TOKEN` | secret | Settings ▸ Secrets and variables ▸ Actions ▸ *Secrets* | a Hugging Face **write** token from <https://huggingface.co/settings/tokens> |
| `HF_REPO` | variable | Settings ▸ Secrets and variables ▸ Actions ▸ *Variables* | the dataset repo id, e.g. `Erios12/behav3d-demo-env` |

The dataset repo must already exist (`hf repo create behav3d-demo-env --type dataset`). The
workflow uploads to `behav3d-env.tar.gz` at the repo root, which is the path
`demo/colab/colab_setup.py` (`ENV_URL`) downloads.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not load the Qt platform plugin "xcb"` | a missing `libxcb-*` | add it to `colab/apt_packages.txt`; run napari with `QT_DEBUG_PLUGINS=1` to see which one |
| Tab opens but stays black / never connects | Colab's proxy dropped WebSockets | run the `start_cloudflared()` cell |
| `no python at /opt/behav3d/bin/python` | the tarball is not a conda-pack archive, or it was built for a different base | rebuild with `demo/build_env.sh` |
| `No module named 'numpy._utils._conversions'` / `'scipy._external'`, or any import error straight after unpacking | a pip step in the build downgraded a conda package (`cellpose==3.1.1.2` requires `numpy<2.1`, which walks numpy, scipy, numba and llvmlite back). conda-pack then packed conda's files next to pip's `.dist-info` | the bootstrap repairs this itself via `colab/repair_env.py`. To stop shipping it, pin `numpy>=2.0,<2.1` in the `micromamba create` step of `build_env.sh` so pip has nothing to downgrade |
| napari exits immediately | see `colab_setup.tail("napari", 60)` | usually a missing library or a bad `PYTHONPATH` |
| Session dies while loading the images | out of RAM | crop the demo bundle further |
| Everything is very slow in 3D | software OpenGL (llvmpipe), expected | stay in 2D; a GPU runtime does not help, since the X display has no GPU |
| Scrollbars in the noVNC tab; the window does not fit | the URL was opened without its query parameters | use the printed URL (`resize=scale`), or set noVNC ▸ Settings ▸ Scaling Mode to "Local Scaling"; `--screen WxH` runs the desktop at your own size |
| Data Preparation cannot find `metadata.csv` | the dry run has no dataset unless `--data` is given; on Colab the paths are under `/content/BEHAV3D_demo` | paste the two paths printed by `colab_setup.print_paths()` - they are read back from `behav3d_parameters.yml` |

## Other hosting routes considered

* **GitHub Codespaces** — works well with the `desktop-lite` devcontainer feature (8 GB, no GPU,
  spends the *visitor's* free 120 core-hours). A good second front door; the same
  `colab_setup.py --local` entry point drives it.
* **Hugging Face Docker Space** — the only genuinely no-login option, but creating a Docker Space
  now requires PRO ($9/month) and a Space is a *single shared container*: two simultaneous visitors
  would fight over one mouse.
* **mybinder.org** — free and login-free, but capped at 2 GB RAM; napari + torch + the dataset do
  not fit.
* **usegalaxy.eu** — napari is already an interactive tool there, sessions are per-user and last up
  to 30 days. Free and durable, but it needs the Galaxy team to accept a BEHAV3D wrapper.

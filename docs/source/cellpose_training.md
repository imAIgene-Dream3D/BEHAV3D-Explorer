# Cellpose Training Notebook

When the [pretrained Cellpose models](processing/segmentation/cellpose.md#pretrained-models) don't give clean segmentations for your data, the next step is to train your own Cellpose model. The napari plugin cannot train Cellpose models; BEHAV3D EXPLORER ships a dedicated Jupyter notebook for this:

```
notebooks/train_behav3d_cellpose.ipynb
```

```{important}
This is the **only** notebook in the BEHAV3D EXPLORER workflow that is meant for routine use. Everything else — segmentation, tracking, feature extraction, filtering, analysis — happens inside the napari plugin. The other notebook (`run_behav3d.ipynb` in `notebooks/`) is for **advanced users** who want to run the pipeline from Python; it is not covered in this wiki.
```

```{tip}
No training data, or no time to annotate? Try [Cellpose-SAM (zero-shot)](processing/segmentation/cellpose_sam.md) first — it needs no model and no training.
```

## What it does

Open the notebook in Jupyter (or VS Code) with the BEHAV3D environment active and **run the cells from top to bottom** — each step uses the settings of the previous one. The notebook produces a custom Cellpose model file that you then load in [Segmentation → Cellpose](processing/segmentation/cellpose.md).

```{mermaid}
flowchart LR
    A["1. Load packages"] --> B["2. Pick model type<br/>(organoid / immune-other)"]
    B --> C["3. Set train + val folders"]
    C --> D["4. Image & channel settings"]
    D --> E["5. Train Cellpose"]
    E --> F["6. Check predictions<br/>in napari"]
```

## Requirements

| Requirement | Detail |
|---|---|
| Training set | 3-D annotated images, ~70 % of your data, e.g. in `training_data/train/`. |
| Validation set | 3-D annotated images, ~30 % of your data, e.g. in `training_data/val/`. Must be images the model has **never** seen during training. |
| Image format | `png`, `jpg`, `jpeg`, `tif`, `tiff`. |
| Mask pairing | Each image has a mask file (full instance labels: `0` = background, one integer per cell) in the same folder, with the same name except for an identifier (e.g. `sample01_img.tif` + `sample01_masks.tif`). |
| Hardware | A CUDA GPU (training on CPU is impractically slow). |

Example folder layout:

```text
training_data/
├── train/
│   ├── sample01_img.tif
│   ├── sample01_masks.tif
│   ├── sample02_img.tif
│   └── sample02_masks.tif
└── val/
    ├── sample03_img.tif
    └── sample03_masks.tif
```

## Step-by-step

### Step 1 — Load packages

Run the first two code cells. They load Cellpose, napari and the BEHAV3D helpers; nothing to set.

### Step 2 — Pick the model type

In the dropdown, choose **Organoid** or **Immune cells/Other cell types**. This sets:

- the **starting model** that is fine-tuned: Cellpose `cyto3` for organoids, `nuclei` for immune / other cells;
- the default **mask identifier** in Step 3 (`_organoids` or `_immune_other`) and the default model name in Step 5;
- the **minimum object size** in the Step 6 check — **1000** voxels for organoids, **10** for immune / other, the same values the plugin uses.

### Step 3 — Set training & validation folders

- **Train directory:** and **Validation directory:** — your two folders (default `training_data/train` and `training_data/val`).
- **Identifier for images** (default `_img`) and **Identifier for masks** — the part of the file name that differs between an image and its mask.

```{warning}
The mask identifier is pre-filled with `_organoids` or `_immune_other`. If your masks are named like `sample01_masks.tif`, change it to `_masks`.
```

### Step 4 — Image and channel settings

In the **Data/Channel config** panel:

- **Z pixel size (micrometers):** and **XY pixel size (micrometers):** of your training images (used to correct for the Z/XY anisotropy);
- **Dimension order:** `ZYX` (Z first, most TIFF stacks) or `YXZ` (Z last);
- **Number of channels:** in the training images, and a **Label for channel *i*** for each (e.g. `tcell`, `organoid`). These labels are only used to pick the training channel(s) in Step 5 and end up in the model's file name; they do not have to match population names in the plugin.

Click **Save configuration** (use **Change configuration** to edit it again).

### Step 5 — Train

Run the cell; the **Training Cellpose** panel appears:

- tick the channel(s) to use as Cellpose input;
- check the model name in **Basename for the … model:** (default `cellpose_organoids` or `cellpose_immune_other`);
- click **Train the Cellpose model**.

The notebook first cuts 2-D training images (up to 512 × 512 pixels) from your 3-D volumes — up to 10 planes per view (XY, ZY, ZX) per volume, corrected for anisotropy — and stores them in a `2D_cellpose<mask identifier>` subfolder of the train and validation folders. It then fine-tunes the model for **400 epochs** with fixed training settings; the notebook does not offer controls to change them. Training takes from tens of minutes to a few hours, depending on data size and GPU.

```{warning}
**Train on one channel.** The plugin's Cellpose method gives the model **one** channel — the one mapped to the population you segment. A model trained on two or more channels does not receive its other channels in the plugin, so it performs poorly or fails to run.
```

### Step 6 — Check the predictions in napari

In the last cell, pick a validation image under **Sample:**, check **Model path:** (pre-filled with the model just trained; you can also pick an older one) and click **Visualize the results**. napari opens with the image channel(s), the ground-truth masks (`GT`) and the model's masks (`prediction`).

Look for:

- under-segmentation (one mask covers two cells);
- over-segmentation (one cell split into several masks);
- missed cells;
- poorly placed boundaries.

If it looks bad, improve or add training masks and train again.

## Output: your custom model

The model is a single file saved in a `models/` folder next to the notebook. Its name is the basename followed by the channel(s) it was trained on, for example:

```text
models/cellpose_organoids__channel0-organoid
models/cellpose_immune_other__channel0-tcell
```

Training again with the same basename and channel replaces the earlier file — change the basename to keep both.

## After training: use the model

1. In napari, open **🦠 Segmentation** → **Cellpose (Deep Learning)**.
2. In **1. Channel Configuration**, map each channel of your experiment to its population and save.
3. In **2. Model & Cell Type**, **Browse…** to the model file and click **📂 Load Model** (the grey line repeats the channel name from the file name, as a reminder).
4. Select the **Cell type to segment:** — the population whose channel shows what the model was trained on — and click **▶ Run Cellpose — *cell type*** (or queue it).

See [Cellpose](processing/segmentation/cellpose.md) for details.

## Tips

- **Annotate fewer images well rather than many images badly.** A handful of clean masks beats a hundred shaky ones.
- **Check in Step 6** before running a full dataset in the plugin.
- **Match the acquisition.** Train on images with the same magnification and pixel size as the data you will segment: the model expects cells at roughly the size it was trained on.
- Training uses **2-D** planes; the plugin applies the model in **3-D** to the whole volume.

## See also

- [Cellpose segmentation](processing/segmentation/cellpose.md) — where you use the model.
- [Cellpose documentation](https://cellpose.readthedocs.io/) — background on Cellpose training.
- [Pretrained Cellpose models on Zenodo](https://zenodo.org/records/18872978) — try these before training your own.

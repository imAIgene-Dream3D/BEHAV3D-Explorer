# 🔬 Cellpose

Cellpose is the **classic deep-learning instance segmenter** (cellpose v3) in BEHAV3D EXPLORER. Unlike the pixel-classifier methods, Cellpose is *not* a pixel classifier — it outputs cell-level instance labels directly from raw intensities, using a pretrained convolutional neural network ([Stringer et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01018-x)).

```{note}
This page covers **classic Cellpose (v3)**, where you load a pretrained or retrained model file. If you want the newer zero-shot foundation model that needs **no model and no training**, see [Cellpose-SAM (zero-shot)](cellpose_sam.md) instead.
```

Classic Cellpose needs a model file. Models are trained on full segmentation masks, not painted points, so they cannot be trained from the napari plugin itself — use the [Cellpose Training Notebook](../../cellpose_training.md). For method choice across all six segmentation options, see the [segmentation overview](./index.md#how-to-pick-a-method).

![Cellpose tab](../../_static/screenshots/segmentation_tab_cellpose.png)

## Pretrained models

Two pretrained models are available, one for **organoids** (`cellpose_organoids__channel0-organoid`) and one for **T cells** (`cellpose_tcells__channel0-tcell`). Each model is a single file (no extension). Get them from:

- **[BEHAV3D EXPLORER pretrained Cellpose models on Zenodo](https://zenodo.org/records/18872978)** — the imaging conditions they were trained on are documented there;
- or the copies that come with BEHAV3D EXPLORER: in the **Browse…** dialog, open the folder where you downloaded BEHAV3D EXPLORER, then `behav3d` → `preprocessing` → `segmentation` → `models`.

The models work best on data imaged under those conditions. On different data (another microscope, magnification or stain), check the result on one sample first. If none of them fits your sample, train your own with the [Cellpose Training Notebook](../../cellpose_training.md).

## Step-by-step in the napari plugin

1. **Get a model** (see above).
2. **Open the Segmentation tab** → pick **Cellpose (Deep Learning)** in **Method:**.
3. **Section 1 · Channel Configuration** — tell BEHAV3D EXPLORER which raw channel holds which population: pick **Same for all** or **Per sample**, choose a population (or `dead`) for each channel, then click **💾 Save Channel Configuration**.
4. **Section 2 · Model & Cell Type** — **Browse…** to the model file and click **📂 Load Model**. Pick the **Cell type to segment** that matches the model, and keep **Process all timepoints** ticked (or set **Start:** / **End:**).
5. **Click ▶ Run Cellpose — *cell type*.** The run goes over every sample directly (there is no preview), so check the first result in the [Visualization tab](../../plugin_essentials/visualization.md) before processing everything.
6. **Other populations** (e.g. organoids and T cells): repeat steps 4–5 with the matching model.
7. **Dead-stain channel?** Click **☠ Run Dead Mask (Otsu)** to produce the dead mask used later by Feature Extraction.
8. Or **queue** the runs with the **+🛒** buttons to launch them together with the rest of the pipeline — see [Processing Queue](../../plugin_essentials/processing_queue.md).

### Section 1 · Channel Configuration

This section maps **raw image channel → population**. Only the channel mapped to the population you segment is passed to the model.

| Control | Effect |
|---|---|
| **Detected: N channel(s)** | Number of channels expected from the metadata (one per population, plus one when a dead channel is declared). The grey line below breaks it down (**Organoids**, **Immune**, **Other**, **Dead**). |
| **⚙ Set Manually** → **Channels:** + **Apply** / **↺** | Use when the raw image has more channels than populations (e.g. an unused channel). Set the real count and click **Apply**; the extra channels get the option `none`. **↺** returns to the detected count. |
| **Mode:** **Same for all** / **Per sample** | **Same for all** (default) uses one mapping for every sample. **Per sample** shows one set of dropdowns per sample, for experiments where the channel order differs. |
| **Channel 0:**, **Channel 1:**, … | One dropdown per channel. Options: the populations in the metadata, `dead` (only when a dead channel is declared) and `none` (only after **Set Manually**). They are pre-filled in metadata order — always check them. |
| **💾 Save Channel Configuration** | Saves the mapping in the project settings (`behav3d_parameters.yml`). **▶ Run** also saves it, but the **+🛒** queue button requires it to be saved first. |

```{warning}
The channel mapping is critical. If a population is mapped to the wrong channel, the model sees the wrong intensities and the masks are meaningless. Check the mapping before every batch.
```

### Section 2 · Model & Cell Type

| Control | Effect |
|---|---|
| **Model:** + **Browse…** | Path to the Cellpose model file (downloaded or trained yourself). You can also paste the path. |
| **📂 Load Model** | Selects that model for the next runs. The grey line below shows **Model loaded · Channels: …**, the channel name stored in the model's file name, as a reminder of what the model was trained on. |
| **Cell type to segment:** | Population to segment. The run button reads **▶ Run Cellpose — *cell type***. |
| **Process all timepoints** | Default ON: segments every timepoint. Uncheck to enable **Start:** / **End:**. |
| **Start:** / **End:** | Inclusive, 0-based timepoint range (default `0` / `0`). |

```{warning}
A run on a timepoint range **replaces the whole segmentation** of that population in each processed sample: timepoints outside the range are left empty. Use a short range to test a model, then run all timepoints.
```

### Section 3 · Run

| Control | Effect |
|---|---|
| **▶ Run Cellpose — *cell type*** | Segments the selected population in every sample of the metadata with the loaded model, [in the background](../../plugin_essentials/running_steps.md#background-execution). Samples whose raw image has not been converted to Zarr are skipped. When segmentations already exist, you choose what to do (see [When results already exist](../../plugin_essentials/running_steps.md#when-results-already-exist)). The result paths are written into metadata.csv. |
| **+🛒** | Adds **🦠 Segmentation — Cellpose** for this population to the [Processing Queue](../../plugin_essentials/processing_queue.md). Queue one step per population (each with its own model). The step remembers the population and the model; the timepoint range and channel mapping are read when it starts. Shown once metadata is loaded. |
| **☠ Run Dead Mask (Otsu)** | Only when the metadata declares a dead channel. Thresholds the channel mapped to `dead` (samples without a `dead` channel in the mapping are skipped). For each sample one automatic (Otsu) threshold is computed over all selected timepoints and applied to every timepoint, giving a binary dead mask. Uses the same timepoint setting as Section 2; timepoints outside the range stay empty. |
| **+🛒** (next to it) | Adds **☠ Dead Mask (Otsu)** to the queue. |

## What runs under the hood

For each timepoint, the plugin passes the 3-D volume of the mapped channel to the model and writes the resulting instance labels. These settings are fixed (not in the GUI):

| Setting | Value |
|---|---|
| 3-D mode | Always on: each volume is segmented in 3-D, not slice by slice. |
| Intensity normalization | Always on, per volume. |
| Cell size | Not estimated from your images: the model expects cells at about the size (in pixels) it was trained on. |
| Z scaling (anisotropy) | Computed per sample from the metadata, $\mathrm{pixel\_distance\_z} / \mathrm{pixel\_distance\_xy}$. |
| Minimum object size | **1000 voxels** for organoids, **10 voxels** for immune and other populations. |
| Flat objects | Objects only one voxel thick in Z, Y or X are removed. |
| Device | CUDA GPU when available, otherwise CPU (much slower). |

```{important}
Z scaling comes straight from your metadata pixel sizes. If they are wrong, the model's behaviour in Z will be off — fix the pixel sizes in the metadata first.
```

## Tuning the result

Almost none of Cellpose's parameters are exposed in the plugin. The result depends on:

1. **Which model you load** — by far the most important choice.
2. **The channel mapping** (Section 1).
3. **The pixel sizes in your metadata** (Z scaling).

Tuning Cellpose therefore means choosing a model or **retraining one on your own data**, not changing GUI values.

**Many cells missed, or blobs in the background**

- Almost always a **channel-mapping problem**. Re-check Section 1.
- If the mapping is correct, the model has probably not seen images like yours — retrain it on your data.

**Two touching cells fused into one label**

- Cellpose has no watershed-style splitting afterwards.
- The proper fix is to **retrain the model** with examples of touching-but-distinct cells ([Cellpose Training Notebook](../../cellpose_training.md)).
- A quick fix is to **split** the labels by hand after tracking ([Manual Editing](../tracking/manual_editing.md)).

**Cells split into several labels**

- Check the pixel sizes in your metadata first (they set the Z scaling).
- If they are right but your Z resolution or cell size (in pixels) differs a lot from the training data, retrain on your own data.

**Inference is very slow**

- Check that the GPU is used (GPU load in your task manager). On CPU, 3-D Cellpose is slow.
- Runtime depends on GPU, volume size and number of objects, so test on a short timepoint range first, then launch the full batch.

**The dead mask looks wrong (too dim or too noisy)**

- One Otsu threshold is used per sample for the whole timepoint range. If the dead-stain intensity drifts strongly over time, it may suit early frames but not late ones.
- Restrict the range to a stable interval, or create the dead mask with a pixel classifier ([APOC](apoc.md), [ConvPaint](convpaint.md) or [Pixel Classifier](pixel_classifier.md)), which can be trained on the dead channel.

## Outputs

| File | Path |
|---|---|
| Cellpose instance labels | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr` |
| Dead mask (Otsu, if run) | `<output_dir>/images/<sample>/<sample>_mask_dead.zarr` |

The paths are written into metadata.csv (the population's `…_segments_image_path` column and `dead_mask_path`). Model files are **not** stored under `<output_dir>`; they stay wherever you downloaded or trained them.

## See also

- [Segmentation overview](./index.md) — how Cellpose fits with the other methods.
- [Cellpose Training Notebook](../../cellpose_training.md) — train a model on your own masks.
- [Cellpose-SAM (zero-shot)](cellpose_sam.md) — no model needed.
- [Zenodo pretrained models](https://zenodo.org/records/18872978).
- [Cellpose project on GitHub](https://github.com/MouseLand/cellpose) — upstream documentation.
- [Processing Queue](../../plugin_essentials/processing_queue.md) — batching several populations.

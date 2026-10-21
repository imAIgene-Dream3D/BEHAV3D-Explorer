# 🎨 ConvPaint

ConvPaint is the **deep-features pixel classifier**. Instead of computing classical features (Gaussian / edge / blob filters like [APOC](apoc.md) or [Pixel Classifier](pixel_classifier.md)), it runs your image through a pretrained CNN or transformer feature extractor (VGG16, DINOv2, …) and then fits a CatBoost gradient-boosted classifier on top of the extracted features. The result is a single multi-class pixel classifier that predicts *every* cell type at once.

```{important}
ConvPaint is **multi-class single-model**: one classifier maps each pixel to background or *any* of the cell types. This is different from [APOC](apoc.md) and [Pixel Classifier](pixel_classifier.md), where you train one binary classifier per cell type. In practice you paint all classes onto a single `User Provided Labels` layer; the **Annotation legend** tells you which label value belongs to which cell type. The dead channel (when declared) gets its own separate binary model.
```

ConvPaint runs on a PyTorch-compatible device (CPU, or a CUDA GPU — strongly preferred). For method choice across all six segmentation options, see the [segmentation overview](./index.md#how-to-pick-a-method).

## Strategies

The **Strategy:** dropdown controls how the classifier output becomes instance labels:

| Strategy | What it does |
|---|---|
| **ConvPaint Mask + EDT/Watershed** (default) | Takes ConvPaint's predicted mask for the cell type → fill holes / opening → distance transform → watershed seeds above the EDT threshold → watershed → size filter. |
| **ConvPaint Probability + Watershed** | Uses the cell type's probability map: a mask threshold defines foreground, a higher seed threshold defines watershed seeds. |
| **Advanced (per cell type)** | Each cell-type tab gets its own **↳ Cell type strategy:** dropdown with the two strategies above. |

## Step-by-step in the napari plugin

A full session usually goes like this:

1. **Open the Segmentation tab** → pick **ConvPaint (DL pixel classifier)** from the **Method:** dropdown.
2. **Pick a device** in **Device:** (CUDA GPU strongly preferred).
3. **Pick a strategy.** Start with **ConvPaint Mask + EDT/Watershed** (default). Switch to **ConvPaint Probability + Watershed** only if the mask boundary is noisy/uncertain. The strategy and the per-cell-type post-processing parameters can be set as soon as metadata is loaded; the feature-extractor and classifier settings stay greyed out until training data is loaded.
4. **Set Examples/sample** (default `3`) and click **Generate Training Data**. The viewer loads evenly spaced timepoints of every sample as `Channel …` image layers plus **one** `User Provided Labels` layer (and `User Provided Labels (Dead)` when a dead channel is declared). See [Generate Training Data](./index.md#common-structure-of-the-pixel-classifier-pages).
5. **Open the Annotation legend** (shown under **Generate Training Data**) to see which label value maps to which cell type (`1` = background, `2` = first cell type, `3` = next, …). Click a row to paint that class.
6. **Paint each class on the single label layer.** A few hundred pixels per class is usually enough; bias your strokes toward boundaries and confusing regions.
7. **Feature Extractor** group: usually leave it on **VGG16 (default)** + **Multichannel** + **Normalize stack** + Downsample `1`. Switch to DINOv2 / JAFAR DINOv2 only if VGG16's mask is "too smooth" or fails on subtle textures.
8. **(Optional) Classifier (CatBoost)** group: the defaults (`100` iterations, depth `5`, learning rate `0.1`, **Balanced (default)** class weights) work for most datasets.
9. **Click ▶ Train Classifier.** This trains the multi-class classifier from `User Provided Labels` (and the separate binary death classifier if dead labels are present), then runs it on all loaded training frames and builds a preview for every cell type.
10. **Inspect and tune.** Adjust a tab's post-processing parameters and click its **Run instance segmentation** — this re-applies only the post-processing to the stored classifier output (no feature extraction), so it is fast. Retrain after changing labels, feature-extractor or classifier settings.
11. **Batch segmentation** → choose the timepoint range → click **▶ Run ConvPaint Batch Segmentation** for an immediate run, or **+🛒** to queue.

```{warning}
Changing the feature extractor **Model:** invalidates the trained classifier. You **must** retrain after switching models.
```

### 1 · Device

| Control | Default | Effect |
|---|---|---|
| **Device:** | First CUDA GPU when one is available, otherwise **Auto (best available)** (your last choice is remembered) | Lists **CPU**, **Auto (best available)** and one **GPU n: <name>** entry per CUDA GPU. A CUDA GPU is strongly preferred for any non-trivial volume. |

### 2 · Strategy + Training data

| Control | Default | Effect |
|---|---|---|
| **Examples/sample:** | `3` | Number of evenly spaced timepoints taken from each sample as training data (range 1–10). |
| **Generate Training Data** | — | Reads the timepoints in the background, then clears the viewer and loads them + the `User Provided Labels` layer (previously saved labels are restored if the cell types are unchanged). Asks **Generate New / Load Existing** when training images already exist. |
| **Strategy:** | ConvPaint Mask + EDT/Watershed | One of the strategies listed above. |

The **Annotation legend** lists **Idx**, **Color**, **Cell type** and **Annotated voxels** for each class; **Refresh counts & colors** updates the voxel counts. Background must always be label `1`. When a dead channel is declared, the legend also has a **Dead** row with **0: Eraser**, **1: Background**, **2: Foreground** buttons for the `User Provided Labels (Dead)` layer. The shared labeling guidance (where and how much to paint) applies here too — see [Labeling foreground and background](./index.md#labeling-foreground-and-background).

**Input channels are chosen automatically.** When the metadata declares one dead channel, the multi-class model uses all other channels and the death model uses only the dead channel; otherwise both use all channels.

### 3 · Feature Extractor (applies to all cell types)

This group controls how raw pixels are turned into per-pixel feature vectors *before* the classifier sees them.

| Control | Options | Default | Effect |
|---|---|---|---|
| **Model:** | VGG16 (default), VGG16 Medium, VGG16 Large, DINOv2, JAFAR DINOv2, Gaussian, Cellpose, Ilastik | VGG16 (default) | Pretrained network used to extract features. VGG16 is fast and lightweight; DINOv2 / JAFAR DINOv2 are transformer-based and often more expressive (slower). **Gaussian** uses a classical filter bank — useful as a quick non-deep-learning baseline. |
| **Channels:** | Single channel, Multichannel, RGB (3-ch color images) | Multichannel | How the channels are fed to the extractor. **Multichannel** processes each channel separately and combines the features (more features, more memory); **Single channel** and **RGB** mix the channels before extraction. |
| **Normalize:** | No normalization, Normalize stack, Normalize per plane | Normalize stack | Intensity normalization before feature extraction: none, over the whole stack, or per image plane. |
| **Downsample:** | −4 to 16 | `1` | Downsampling factor applied before feature extraction; predictions are scaled back to full size. `1` = native resolution; >1 = faster but coarser; <0 upsamples (rarely useful). |
| **Smoothen:** | 0 to 20 | `1` | Smoothing applied to the predicted label map. Reduces single-pixel noise but can erode thin structures. |

### 4 · Classifier (CatBoost)

ConvPaint uses [CatBoost](https://catboost.ai), a gradient-boosted decision-tree classifier (not a Random Forest).

| Control | Range | Default | Effect |
|---|---|---|---|
| **Iterations:** | 10–2000 | `100` | Number of boosting rounds (trees). More iterations = richer model, slower to train, more risk of overfit. |
| **Depth:** | 1–16 | `5` | Maximum depth of each tree. 3–6 is a good starting range. |
| **Learning rate:** | 0.001–1.0 | `0.1` | Shrinkage applied to each new tree's contribution. Lower values need more iterations but often generalise better. |
| **Class weights:** | Balanced (default), SqrtBalanced, None | Balanced | How to handle classes with far fewer painted pixels. **Balanced** boosts rare classes (inverse-frequency weights); **SqrtBalanced** is a milder version; **None** trains as-is. |
| **Tile annotations** | checkbox | OFF | Training only: extracts features only in boxes around the painted regions instead of the whole image — faster when annotations are small/sparse. No effect on prediction. |

A separate **Prediction Performance (large images)** group affects prediction (training preview, **Run preview (death mask)**, batch) and stays available even when no training images are loaded:

| Checkbox | Default | Effect |
|---|---|---|
| **Tile image** | OFF | Splits the image into ~1000 × 1000 px blocks (50 px overlap) for feature extraction and classification, then stitches the result. Lower peak memory, somewhat slower. Turn on when a large image runs out of memory during prediction. |
| **Use Dask** | OFF | Beta. Processes the tiles in parallel; checking it also checks **Tile image**. Can speed up tiled prediction on machines with free CPU cores, but does not lower memory further (peak memory can be higher). |

### 5 · Annotation & Segmentation tabs

The **Annotation & Segmentation** group has one tab per cell type, plus a **Dead** tab when a dead channel is declared. Each cell-type tab shows the active strategy, a **Units:** px ⇄ µm switch, the post-processing parameters of that strategy, and a **Run instance segmentation** button. With **Advanced (per cell type)**, a **↳ Cell type strategy:** dropdown appears at the top of each cell-type tab. The **Dead** tab has no post-processing parameters; its **Run preview (death mask)** button re-runs the saved death model on the training frames.

```{seealso}
**These instance post-processing parameters are explained once, centrally — read them there.** What each control does (EDT threshold, Mask/Seed threshold, Min size, Opening px, Fill holes, Units): [Instance post-processing parameters](./index.md#instance-post-processing-parameters). What to change when the result looks wrong: [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning). Starting points for large vs small objects: [Advice: large vs small objects](./index.md#advice-large-vs-small-objects).
```

| Strategy | Parameters shown (ConvPaint defaults, in pixels/voxels) |
|---|---|
| **ConvPaint Mask + EDT/Watershed** | **EDT threshold** `1.0`, **Opening px** `0` (0–50), **Min size** `10`, **Fill holes** ON |
| **ConvPaint Probability + Watershed** | **Mask threshold** `0.5`, **Seed threshold** `0.8`, **Opening px** `0` (0–50), **Min size** `10` |

With pixel sizes in the metadata, EDT threshold and Min size are displayed in µm / µm³ by default (switch **Units:** to **px** to see the pixel values above).

### 6 · Train + Preview

- **▶ Train Classifier** fits the multi-class CatBoost classifier on the painted labels using features from the chosen extractor (plus the binary death classifier when dead labels exist). Labels are saved first. The classifier is then run on all loaded training frames and every cell type's preview is built.
- **Run instance segmentation** (per cell-type tab) re-applies that tab's post-processing to the stored classifier output, without re-extracting features — iterate on watershed parameters with it.
- **🔁 Re-run preview** does the same for the currently selected tab. Prefer the tab's own **Run instance segmentation** button.
- **💾 Save User Labels** saves the label layers (see [Labeling foreground and background](./index.md#labeling-foreground-and-background)).

Preview layers in the viewer: `Pixel Classification (Unified)` (the multi-class prediction), `<Cell type> Segments` (instance preview), `Probability Map (<Cell type>)` (probability strategy only; hidden by default) and `Pixel Classification (Dead)` (death mask). They are stored in the `PixelClassification` folder and restored the next time you load the training data.

## Tuning the parameters

Use the per-cell-type **Run instance segmentation** button between post-processing changes, and **▶ Train Classifier** after any change to labels, the feature extractor or the classifier. There are three groups of parameters that can affect the result: **feature extractor**, **CatBoost classifier**, and **per cell-type post-processing**.

**Feature Extractor — Model**

- **VGG16 (default)** works for most datasets and is the fastest. Try it first.
- The mask is "too smooth" — classes that should be separable are bleeding into each other → switch to **DINOv2** or **JAFAR DINOv2** (more expressive, transformer-based features) and retrain.
- Out-of-memory error during training or prediction → switch to a smaller model (**VGG16** instead of **VGG16 Large**, or **Gaussian** for a quick non-deep-learning baseline), raise **Downsample:**, or turn on **Tile image**.

**Feature Extractor — Channels**

- **Multichannel** (default) preserves per-channel information and is almost always what you want.
- Use **Single channel** only when the channels together are not more informative than their average.
- Use **RGB (3-ch color images)** only for true 3-channel colour images (rare in fluorescence microscopy).

```{note}
**Normalize**, **Downsample** and **Smoothen** follow the
napari-convpaint definitions; no BEHAV3D-specific tuning guidance has been validated for
them. See the
[napari-convpaint documentation](https://guiwitz.github.io/napari-convpaint/book/Params_settings.html).
```

**CatBoost — Iterations**

- Default `100` is sensible. Raise to 200–500 if the classifier is clearly under-fitting *and* the painted set is small enough that training stays fast (a few minutes).
- Lower **Learning rate:** if you raise iterations a lot, otherwise CatBoost overfits.

**CatBoost — Depth**

- Default `5` is a balanced starting point. Most ConvPaint setups stay in the 3–6 range.
- Raise to 7–10 only for clearly under-fit textures with lots of painted pixels.

**CatBoost — Learning rate**

- Default `0.1` is fine. If you increase **Iterations:** substantially (e.g. 100 → 500), also lower **Learning rate:** (e.g. 0.1 → 0.03) — the classic gradient-boosting trade-off for better generalisation.

**CatBoost — Class weights**

- **Balanced** (default) handles cell-type imbalance well — e.g. when background is 95 % of the painted pixels and the other classes are under-predicted.
- **None** is rarely the right answer unless your label counts are roughly equal across classes.
- **SqrtBalanced** is the conservative middle ground when **Balanced** overcorrects (e.g. random "rare" pixels appear everywhere).

**Per cell-type post-processing**

The EDT / Mask / Seed thresholds, Opening px, Min size and Fill holes follow the shared rules — see [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning) for what to change when something looks wrong, and [Advice: large vs small objects](./index.md#advice-large-vs-small-objects) for large vs small starting points.

### 7 · Batch segmentation

Standard batch controls in the **🚀 Batch Segmentation** group — see [Batch segmentation controls](./index.md#batch-segmentation-controls). The run/queue buttons here are **▶ Run ConvPaint Batch Segmentation** and **+🛒**. The run needs a trained multi-class model; it uses each tab's strategy and post-processing parameters, and **Tile image** / **Use Dask** when checked. **Workers:** has no effect for ConvPaint; it is the shared [Workers](../../plugin_essentials/running_steps.md#workers) setting, so changing it here also changes it for Zarr conversion and the Pixel Classifier.

## Outputs

| File | Path |
|---|---|
| Per-cell-type instance labels | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr` |
| Per-cell-type foreground mask | `<output_dir>/images/<sample>/<sample>_<cell_type>_mask.zarr` |
| Death mask (dead channel only) | `<output_dir>/images/<sample>/<sample>_mask_dead.zarr` |
| Trained ConvPaint models | `<output_dir>/images/PixelClassification/` — `ConvPaintModel_Unified.pkl` (multi-class model, with its label map) and `ConvPaintModel_Dead.pkl` (death model). Treat the files as a set; don't edit them. |
| Saved labels and previews | `<output_dir>/images/PixelClassification/` |

## Good practices & tips

- **Start with VGG16.** It is the default and runs fastest on common GPUs. Only switch to DINOv2 / JAFAR DINOv2 if VGG16's prediction is "too smooth" or fails on subtle textures — these heavier extractors are slower and need more GPU memory.
- **CatBoost defaults are sensible.** 100 iterations, depth 5, learning rate 0.1 is a balanced starting point. If you see overfitting (great on the labelled frames, poor elsewhere), lower the learning rate to ~0.03 and proportionally raise iterations.
- **Class weights = Balanced is the safe default.** If one cell type has far fewer pixels than the others, **None** gives a classifier that systematically under-predicts that class.
- **Downsample > 1 if GPU memory is tight.** Prediction time scales with the number of feature vectors; a downsample of 2 cuts the work roughly 4× per plane. The predictions are scaled back to full size at the end.
- **Switching feature extractors invalidates the model.** Retrain after any change to **Model:**.
- **Save labels before switching method** — switching to another method removes the training layers without asking (leaving the tab does ask).

## Things this page does **not** claim

- That DINOv2 always beats VGG16 (or vice versa) — this is dataset-dependent. The in-app help describes DINOv2 as "stronger feature representations… slower but often more accurate", which matches typical published results, but BEHAV3D EXPLORER ships no benchmark validating this on organoid / immune-cell data.
- Specific per-timepoint inference numbers — these depend strongly on GPU, volume size, model, and downsample. Profile your own data.

## See also

- [napari-convpaint documentation](https://github.com/guiwitz/napari-convpaint) — upstream project, including details on the feature extractors and CatBoost integration.
- [APOC](apoc.md) — classical-features alternative when speed and GPU memory matter more than feature richness.
- [Pixel Classifier](pixel_classifier.md) — CPU-only baseline.
- [Processing Queue](../../plugin_essentials/processing_queue.md) for batching segmentation steps.

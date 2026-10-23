# 🧪 Pixel Classifier

The Pixel Classifier is the **CPU-only** pixel classification method bundled with BEHAV3D EXPLORER. For each cell type, you paint a few background / foreground pixels in the napari viewer, the plugin computes multiscale image features, fits a Random Forest classifier, and uses it to label every voxel as foreground or background. The binary mask is then turned into instance labels by a post-processing pipeline (fill holes → opening → distance-transform watershed → size filter).

```{important}
The Pixel Classifier is **binary per cell type**: foreground vs. background. If you have N cell types, you train N separate classifiers, one per cell type (plus one for the dead channel when declared). Multi-class single-model training is the job of [ConvPaint](convpaint.md).
```

The Pixel Classifier runs entirely on the CPU. For method choice across all six segmentation options, see the [segmentation overview](./index.md#how-to-pick-a-method).

## What runs under the hood

For each cell type the plugin:

1. Builds a feature stack from all image channels — Gaussian-smoothed intensity plus edge strength (Sobel of the smoothed image), computed in 3-D at four fixed physical scales (`0.89, 1.77, 3.54, 14.16` µm). The scales are converted to pixels separately for XY and Z with the pixel sizes from your metadata, so the same features work regardless of acquisition resolution. The features are not editable in the GUI.
2. Fits a binary Random Forest (50 trees, max depth 20) on the pixels you painted. The forest settings are **not exposed in the GUI**.
3. Predicts a foreground mask, post-processes it (fill holes → morphological opening → distance transform, measured within each Z slice → watershed → size filter), and writes the resulting instance labels to disk.

## Step-by-step in the napari plugin

The widget is organised in three numbered group-boxes. A full session usually goes like this:

1. **Open the Segmentation tab** → pick **Pixel Classifier (Random Forest)** from the **Method:** dropdown.
2. **Set Examples/sample** (default `3`).
3. **Click Generate Training Data.** Images and features are prepared in the background (progress bar; napari stays responsive and the other tabs are locked meanwhile), then the viewer clears and loads evenly spaced timepoints of every sample as `Channel …` image layers plus one empty label layer per cell type, named `User Provided Labels (<Cell type>)` (and `User Provided Labels (Dead)` when a dead channel is declared). See [Generate Training Data](./index.md#common-structure-of-the-pixel-classifier-pages).
4. **Paint a few foreground (`2`) and background (`1`) pixels** on the label layer of each cell type, using the **Annotation legend** buttons. See [Labeling foreground and background](./index.md#labeling-foreground-and-background) for the painting convention and tips.
5. **Click Save Labels** to save your work to disk (do this often).
6. **Click Train Classifier.** Labels are saved, one classifier per cell type (and dead channel) is trained, and the result on all loaded training frames appears as `Pixel Classification (<Cell type>)` (the foreground mask) and `<Cell type> Segments` (the instances, using the current parameters of section 2).
7. **Section 2 (Fine-tune Segmentation Parameters)** → for each cell type, check **EDT:**, **Min size:**, **Opening:**, **Fill holes**. Defaults depend on the cell-type category (organoid / immune / other) and usually only need small adjustments — see also [Advice: large vs small objects](./index.md#advice-large-vs-small-objects).
8. **Click Test Segmentation Parameters** to re-segment the **currently displayed timepoint** with the new parameters (no retraining). The `<Cell type> Segments` layers are updated for that timepoint — toggle them on/off to compare with the raw image.
9. **Inspect the result.** If cells are merged together, split apart, or contain noise, see [Tuning the parameters](#tuning-the-parameters) below. Adjust → re-test → repeat.
10. **Section 3 (Batch Segmentation)** → pick the timepoint range → click **Run Batch Segmentation** to run now, or **+🛒** to queue the segmentation step in the [Processing Queue](../../plugin_essentials/processing_queue.md).

```{tip}
Iteration is cheap: re-training is quick and **Test Segmentation Parameters** re-segments a single timepoint. A typical first pass is *paint → train → test → paint more on the worst-looking regions → retrain*.
```

### 1 · Labeling & Training

| Control | Default | Effect |
|---|---|---|
| **Examples/sample:** | `3` | Number of evenly spaced timepoints (from the first to the last timepoint; with `1`, only the first) taken from each sample as training data (range 1–10). More timepoints = more variability for the classifier to learn from, at the cost of a longer labeling session. |
| **Generate Training Data** | — | Prepares (in the background) and then loads $\text{examples} \times \text{samples}$ timepoints as image layers, plus a `User Provided Labels (…)` layer per cell type. When cached training data exists, asks **Generate New / Load Existing / Cancel**. |
| **Save Labels** | — | Saves every `User Provided Labels (…)` layer to disk, so the labels survive plugin restarts. |
| **Clear Layer** / **Clear All** | — | Erase the painted labels in the selected `User Provided Labels` layer / in all of them (**Clear All** asks for confirmation). |
| **Train Classifier** | — | Trains one Random Forest per cell type (and one for the dead channel when it has labels) and shows the prediction and instance preview. |
| **+🛒** (next to **Train Classifier**) | — | Queues a training step in the [Processing Queue](../../plugin_essentials/processing_queue.md). |

The buttons below **Generate Training Data** appear once training data is loaded, together with the hint *Select a 'User Provided Labels ...' layer and annotate your ground truth data in the left panel* and the **Annotation legend**.

```{warning}
**Generate New** discards the previously saved labels and predictions of the Pixel Classifier (not those of APOC or ConvPaint). Choose **Load Existing** to keep painting on your saved labels.
```

The painting convention (`0` = eraser, `1` = background, `2` = foreground) and labeling tips are shared across methods — see [Labeling foreground and background](./index.md#labeling-foreground-and-background).

### 2 · Fine-tune Segmentation Parameters (Optional) — per cell type

```{seealso}
**These instance post-processing parameters are explained once, centrally — read them there.** What each control does (EDT threshold, Min size, Opening, Fill holes, Units): [Instance post-processing parameters](./index.md#instance-post-processing-parameters). What to change when the result looks wrong: [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning). Starting points for large vs small objects: [Advice: large vs small objects](./index.md#advice-large-vs-small-objects). The table below just lists this method's defaults.
```

This section appears once metadata is loaded. It starts with **Workers:** (the number of CPU workers used for batch segmentation, at most one less than your CPU core count; this is the shared [Workers](../../plugin_essentials/running_steps.md#workers) setting, half of your CPU cores on a fresh project), followed by one group-box per cell type with a **Units:** px ⇄ µm switch and category-dependent defaults:

| Parameter | Default for organoids | Default for immune cells | Default for "other" |
|---|---|---|---|
| **EDT:** (px) | `12.0` | `2.5` | `1.0` |
| **Min size:** (voxels) | `1000` | `10` | `10` |
| **Opening:** (px, 0–10) | `3` | `0` | `0` |
| **Fill holes** | ON | ON | ON |

With pixel sizes in the metadata, **EDT:** and **Min size:** are displayed in µm / µm³ by default; switch **Units:** to **px** to see the pixel values in the table. **EDT:** works as described centrally: **higher → more splitting** of touching objects (no splitting at `1.0` px or below, so the "other" default does not split). The **Opening:** and **Fill holes** settings are also applied to the predicted mask, and to your painted foreground before training — painted foreground strokes narrower than about twice the **Opening:** radius are therefore dropped from the training labels. The dead channel has no parameter box.

A **Test Segmentation Parameters** button at the bottom of this section re-segments the currently displayed timepoint with the current parameters and updates the `<Cell type> Segments` layers — useful for iterating on EDT / Min size / Opening / Fill holes without committing to a batch run.

### 3 · Batch Segmentation

Standard batch run — see [Batch segmentation controls](./index.md#batch-segmentation-controls) for the shared behaviour. This method's controls differ slightly in wording/placement:

- The checkbox is labelled **All timepoints** (ON); the range is **Range: T:** start – end (`0` – `0`), editable when the box is off.
- The run button is **Run Batch Segmentation**, next to **+🛒**.
- **Workers:** lives in section 2 (not here). Existing results trigger the **Overwrite All / Skip Existing / Customize… / Cancel** prompt.

## Tuning the parameters

Use **Test Segmentation Parameters** between every change. The rules below are the ones that actually matter on real data.

**Post-processing parameters (EDT / Min size / Opening / Fill holes)**

- For this method **EDT:** is the most important parameter — **raise it to split touching cells more aggressively, lower it to keep them merged**. The full failure-mode → fix heuristics for all four parameters are shared and documented once in [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning) on the [segmentation overview](./index.md).

**Examples/sample**

- Result looks good on the timepoints you trained on but poor on others → the classifier hasn't seen enough variability. Generate training data again with a higher value (e.g. 5–8), paint a few more representative pixels, and retrain. Remember that **Generate New** starts from empty label layers.

**If the classifier itself is the problem (not post-processing)**

The Random Forest learns only from the painted pixels. If the *raw foreground mask* (`Pixel Classification (<Cell type>)`) is wrong, no amount of post-processing will save it.

- Foreground bleeds into background → add more **background labels**, especially near misclassified regions.
- Cells are not detected at all → add more **foreground labels**, especially at the missed cells' boundaries.
- Behaviour is fine on bright cells but fails on dim ones → either include some dim cells in your labels (best fix), or consider switching to [APOC](apoc.md) or [ConvPaint](convpaint.md), which offer richer, tunable features.

## Outputs

| File | Path |
|---|---|
| Per-cell-type instance labels | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr` |
| Per-cell-type foreground mask | `<output_dir>/images/<sample>/<sample>_<cell_type>_mask.zarr` |
| Death mask (dead channel only) | `<output_dir>/images/<sample>/<sample>_mask_dead.zarr` |
| Trained classifiers | `<output_dir>/images/PixelClassification/PixelClassifier_<CellType>.joblib` (one per cell type, plus one for the dead channel) |
| Saved labels, training images/features and predictions | `<output_dir>/images/PixelClassification/` |

## Good practices & tips

- **Always click *Test Segmentation Parameters* on at least one timepoint before launching a batch run.** EDT and Min size have a huge effect on the result and are much faster to iterate per timepoint than per batch.
- **Annotate boundary pixels.** This is where misclassifications hurt downstream segmentation the most — the watershed is seeded from the distance transform inside the foreground, so noisy boundaries propagate into the instance labels.
- **Workers.** The Pixel Classifier is purely CPU. More workers ≈ faster batch inference, up to the cap of "one less than your CPU core count" that the widget enforces. The actual time per timepoint depends on volume size and channel count.
- **Save labels often.** **Save Labels** writes the painted annotations to disk. Leaving the Segmentation tab asks whether to save; switching to another method does not.
- **Re-training is cheap.** Only the labelled pixels are used for fitting, so iterate liberally: add a few more labels → retrain → test → repeat.

## See also

- [APOC](apoc.md) — same family of classifier, GPU-accelerated, with tunable features.
- [ConvPaint](convpaint.md) — alternative using deep features instead of classical ones.
- [Processing Queue](../../plugin_essentials/processing_queue.md) — queue training and segmentation to run sequentially.

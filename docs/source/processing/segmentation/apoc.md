# ⚡ APOC (GPU)

APOC ([Accelerated Pixel and Object Classifiers](https://github.com/haesleinhuepf/apoc)) is the **GPU pixel classifier** option in BEHAV3D EXPLORER. It computes 3-D image features on an OpenCL GPU, trains a Random Forest classifier on the features of the painted voxels, and runs that classifier on the GPU to predict the class of every voxel.

```{important}
APOC trains a separate binary classifier (foreground vs. background) for each cell type. If you set up N cell types in BEHAV3D EXPLORER, APOC trains N classifiers — plus one for the dead channel when one is declared in the metadata.
```

For method choice across all six segmentation options, see the [segmentation overview](./index.md#how-to-pick-a-method). No pixel classifier is best on every dataset — APOC, [ConvPaint](convpaint.md) and the Pixel Classifier can each win depending on the data, and with APOC the choice of features (see [Tuning the parameters](#tuning-the-parameters)) usually matters more than the method itself.

## Strategies

The **Strategy:** dropdown at the top of the training area controls how a trained classifier is turned into instance labels:

| Strategy | What APOC outputs | How it becomes instance labels |
|---|---|---|
| **APOC (Direct Instance Segmentation)** | Foreground mask. | Each connected foreground object becomes one instance — no splitting, no size filter, no post-processing controls. |
| **APOC Mask + EDT/Watershed Resegmentation** | Foreground vs. background mask. | Fill holes → opening → distance transform of the mask → threshold → watershed seeds → watershed → size filter. |
| **APOC Probability Map + Watershed** (default) | Per-voxel foreground probability. | A mask threshold defines foreground; a higher seed threshold defines watershed seeds; watershed splits touching objects on the probability map. |
| **Advanced (per cell type)** | — | Each cell-type tab gets its own **↳ Cell type strategy:** dropdown with the three strategies above. |

```{tip}
The plugin opens on **APOC Probability Map + Watershed**: it places the split between touching objects using the classifier's confidence, which helps when boundaries are fuzzy. **Direct Instance Segmentation** is the simplest and fastest, but assumes your cells are already well separated by the pixel classifier alone. The watershed variants exist because biological cells in 3D very often touch — pick the one whose post-processing matches your object morphology and tune its parameters in each cell-type tab. Switch to **APOC Mask + EDT/Watershed Resegmentation** if the probability-map split isn't clean.
```

## Workflow at a glance

1. **Open the Segmentation tab** and pick **APOC (GPU)** from the **Method:** dropdown.
2. [Pick the GPU](#1--gpu-device) from the **GPU Device:** dropdown.
3. [Pick a strategy](#2--strategy--training) in the **Strategy:** dropdown (default **APOC Probability Map + Watershed**; see [Strategies](#strategies)).
4. [Generate and label the training data](#2--strategy--training):
   1. Set **Examples/sample** (default `3`) and click **Generate Training Data**. The viewer loads evenly spaced timepoints of every sample as `Channel …` image layers plus one empty `User Provided Labels (<Cell type>)` layer per cell type (see [Generate Training Data](./index.md#common-structure-of-the-pixel-classifier-pages)).
   2. Paint a few foreground and background pixels on the label layer of each cell type, using the **Annotation legend** buttons (see [Labeling foreground and background](./index.md#labeling-foreground-and-background)).
   3. (Optional) [Reuse training data](#5--reusing-training-data-import--export) from a previous experiment.
5. [Review each cell-type tab](#3--per-cell-type-tab): **Image Channel Inputs**, **Preset:** / **Tune Features** and the Random Forest parameters. Defaults are sensible; the most common change is unchecking irrelevant channels. Use **⬇ Apply config to all tabs** to copy one tab's settings to the others.
6. [Train and preview](#4--train--preview): click **▶ Train <Cell type> classifier** for the active tab, or **▶▶ Train ALL classifiers**. The prediction appears as `<Cell type> Segments` and a **Classifier Statistics** window opens.
7. [Tune the result](#tuning-the-parameters) and repeat steps 4–6 until it holds up:
   - For the watershed strategies, adjust the **Instance Segmentation Preview** group and click **Run instance segmentation** to re-apply it without retraining.
   - With **APOC (Direct Instance Segmentation)** there is nothing to tune after the classifier; switch to a watershed strategy if you need splitting.
8. [Batch segmentation](#6--batch-segmentation): choose the timepoint range, then click **▶ Run APOC Batch Segmentation** to run now or **+🛒** to queue. To (re)segment just one cell type, use **▶ Run <Cell type> Segmentation**.

Each numbered step is described in the sections below.

## Step-by-step in the napari plugin

### 1 · GPU device

| Control | Default | Effect |
|---|---|---|
| **GPU Device:** | First OpenCL device found (your last choice is remembered) | Lists every OpenCL device the system can see. Pick a discrete GPU if you have one; integrated GPUs work but are much slower for large volumes. Shows **Auto (default OpenCL device)** (greyed out) when no device can be listed. |
| **Force CPU-only processing** | OFF | Ignores the GPU selection and runs APOC on the CPU. Needs an OpenCL CPU runtime (e.g. Intel CPU Runtime for OpenCL); without one, APOC falls back to whatever OpenCL device it finds and a warning is logged. Use it when no compatible GPU is available or GPU processing is unstable. |

The device cannot be changed while a training or segmentation run is in progress.

### 2 · Strategy + Training

All of these live in the **🎯 APOC Classifier Training** group.

| Control | Default | Effect |
|---|---|---|
| **Examples/sample:** | `3` | Number of evenly spaced timepoints taken from each sample as training data (range 1–10). |
| **Generate Training Data** | — | Reads those timepoints in the background, then clears the viewer and loads them as image layers + one label layer per cell type (saved labels are restored). Asks **Generate New / Load Existing** when training images already exist. |
| **Import Training Data** | — | Visible once training data is loaded. Reuses labelled pixels from a previous experiment — see section 5 below. |
| **Export Training Bundle…** | — | Enabled once at least one classifier has been trained. Saves this experiment's training data and classifiers as a shareable `.zip`. |
| **Strategy:** | APOC Probability Map + Watershed | One of the strategies listed above. Changing it rebuilds the **Instance Segmentation Preview** group in every tab. |
| **⛓ Apply same settings to all organoids** | OFF | Only shown when the metadata has two or more organoid cell types. When checked, copies the first organoid tab's configuration (channels, features, strategy, post-processing parameters) to the other organoid tabs, and keeps mirroring edits made on an organoid tab while it stays checked. Immune and other cell types are not affected. |

APOC is a binary classifier with one label layer per cell type (`1` = background, `2` = foreground; the hint **Labels: 1 = background 2 = foreground** is repeated in each tab). See [Labeling foreground and background](./index.md#labeling-foreground-and-background) for the painting convention and labeling tips shared by all pixel-classifier methods.

### 3 · Per cell-type tab

A tab is created for each cell type detected in the metadata (and a **Dead** tab when a dead channel is declared). Each tab exposes:

| Control | Default | Effect |
|---|---|---|
| **Image Channel Inputs** | All channels checked | Tick the image channels this classifier should see. |
| **↳ Cell type strategy:** (only with **Advanced (per cell type)**) | Same as the global strategy | Strategy for this cell type only. Not shown on the **Dead** tab. |
| **Preset:** | `large_preset` | Pre-configured grid of feature filters × sigmas: `small_preset` (sigmas 1, 2, 5), `medium_preset` (1, 2, 5, 15), `large_preset` (1, 2, 5, 10, 25). Picking one of these resets the grid (and the **Custom Sigmas:** list) to that pattern; picking `custom` keeps the current grid for you to edit. |
| **Tune Features** (tick the box to expand) | Collapsed | Feature × sigma checkbox grid (rows **Gauss**, **DoG**, **LoG**, **SoG**) plus a **Custom Sigmas:** field (comma/space-separated; default `0.3, 0.5, 1, 2, 3, 4, 5, 10, 15, 25`). Click **Update Grid** after editing the sigmas — this switches the preset to `custom`. A **Features:** line below summarises the current selection. |
| **Consider original image as well** | ON | Adds raw pixel intensity as an extra feature. Picking one of the three standard presets turns it on; picking `custom` leaves it as it is. |
| **Show classifier statistics** | — | Opens the feature-importance table of the trained classifier (also shown automatically after training). |
| **Max depth:** | `5` | Maximum depth of each decision tree (range 1–20). Deeper trees fit the painted pixels more closely (risk of overfitting a small label set) and make the classifier larger and slower to run. |
| **Trees:** | `100` | Number of decision trees in the random forest (range 10–1000, step 10). |

Channel inputs, preset, features and forest parameters are only visible once training data is loaded; the **Instance Segmentation Preview** group is always visible so you can set post-processing in advance.

**Instance Segmentation Preview** (cell-type tabs only, not shown for Direct strategy or on the **Dead** tab) holds a **Units:** px ⇄ µm switch, the post-processing parameters of the active strategy, and the **Run instance segmentation** button.

```{seealso}
**These instance post-processing parameters are explained once, centrally — read them there.** What each control does (EDT threshold, Mask/Seed threshold, Min size, Opening px, Fill holes, Units): [Instance post-processing parameters](./index.md#instance-post-processing-parameters). What to change when the result looks wrong: [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning). Starting points for large vs small objects: [Advice: large vs small objects](./index.md#advice-large-vs-small-objects). The table below just lists which controls each strategy shows and their APOC defaults.
```

| Strategy | Parameters shown (APOC defaults, in pixels/voxels) |
|---|---|
| **APOC (Direct Instance Segmentation)** | none |
| **APOC Mask + EDT/Watershed Resegmentation** | **EDT threshold** `1.0`, **Opening px** `0` (0–10), **Min size** `0`, **Fill holes** ON |
| **APOC Probability Map + Watershed** | **Mask threshold** `0.5`, **Seed threshold** `0.8`, **Opening px** `0` (0–10), **Min size** `0` |

The defaults are the **same on every tab** the first time you open APOC — only a starting point. With **EDT threshold** at `1.0` px nothing is split yet; raise it to split touching cells (see [Instance post-processing parameters](./index.md#instance-post-processing-parameters)). With pixel sizes in the metadata, EDT threshold and Min size are displayed in µm / µm³ by default (switch **Units:** to **px** to see the pixel values above).

The `0.3` and `0.5` entries in **Tune Features** are filter scales in pixels. They are not a recommended range for Mask threshold or Seed threshold. For Probability Map + Watershed, start from **0.5 Mask / 0.8 Seed** and tune against the instance preview.

### 4 · Train + Preview

- **▶ Train <Cell type> classifier** (the label follows the active tab) / **▶▶ Train ALL classifiers** fit the classifier(s) in the background. Labels are saved first. Tabs without any painted label are skipped. After training, the classifier is run on all loaded training frames and the result is shown in the viewer; a **Classifier Statistics** window opens for each trained cell type.
- **Run instance segmentation** (per cell-type tab, enabled once training data is loaded) re-applies the tab's post-processing to the classifier output on all loaded training frames, so you can iterate on EDT / Mask & Seed thresholds, Opening and Min size without retraining. With the probability strategy the stored probability map is reused (fast); with Mask + EDT the classifier is re-run first.
- **Save User Labels** saves all label layers (see [Labeling tips](./index.md#labeling-foreground-and-background)).

Preview layers in the viewer:

- `<Cell type> Segments` — the instance preview (for the **Dead** tab: `Pixel Classification (Dead)`, the death mask).
- `Probability Map (<Cell type>)` — per-voxel probability (hidden by default; `Probability Map (Dead)` for the dead classifier).

The previews are stored in the `PixelClassification` folder and restored the next time you load the training data. Full per-sample outputs are only produced by the batch run.

### 5 · Reusing training data (Import / Export)

Every training run also stores the labelled pixels in a training bundle (`PixelClassifier_TrainingData.zip`), so they can be reused in another experiment:

1. In the source experiment, click **Export Training Bundle…** and save the `.zip` (default name `behav3d_training_bundle.zip`).
2. In the new experiment, click **Generate Training Data** first (importing adds labelled pixels; it does not load images or replace labeling), then **Import Training Data** and pick the `.zip` bundle (or its `training_metadata.yml`).
3. If cell-type names differ, a **Cell Type Mapping** dialog lets you map each imported type to a local one, or **(skip)** it. If the pixel sizes differ by more than 10 %, a **Pixel Size Mismatch** warning asks whether to proceed (feature scales are in pixels, so the same sigma covers a different physical size).
4. A summary dialog lists the imported pixels per cell type (**Total px**, **Fg / Bg**); untick a row to exclude it, then click **Proceed with Import**.

After import, the **📦 Imported Training Data** panel shows the source and pixel counts, and each mapped tab's features and sigmas are **locked** to those of the imported classifier. Paint at least a few labels for each cell type you want to train — the imported pixels are added to your new labels when you click **▶ Train …**, and the combined set is saved so this experiment can itself be exported later. **Clear Import** removes the imported data and unlocks the feature controls.

## Tuning the parameters

Use the per-cell-type **Run instance segmentation** button between every post-processing change, and retrain after classifier changes. Two groups of parameters matter: the **classifier** (features + random forest) and the **post-processing** (the watershed parameters of the chosen strategy).

**Classifier — Preset and the Tune Features grid**

APOC does not classify raw voxels directly. For each checked **Image Channel Inputs** channel, it builds a stack of derived images (one per checked **filter × sigma** in the grid, plus optional raw intensity), and the Random Forest learns from that stack. More checked boxes → more columns → usually more detail but slower training and inference and more GPU memory.

**What each filter row means** (names in the grid match the UI):

| Row | What it emphasises | Typical use |
|---|---|---|
| **Gauss** | Smoothed intensity at scale σ — gradual intensity changes, with noise removed. | Large, soft objects; overall brightness of a region. |
| **DoG** | Difference of two Gaussian blurs (σ × 0.9 and σ × 1.1) — a band-pass filter that responds to bright/dark spots and lines about σ wide and changes sign at their edges. | Structures of a given size; included in all presets. |
| **LoG** | Laplacian of the Gaussian-blurred image — a similar spot/line response at scale σ (peaks and valleys). | Round nuclei, organoid cores, spot-like markers, thin processes. |
| **SoG** | Sobel of the Gaussian-blurred image — **edge strength** (gradient magnitude, in any direction) at scale σ. | Cell borders, membranes. |

**Sigma (σ)** is the width of the Gaussian blur **in pixels** before each filter is applied. It is **not** a probability and not microns unless your data happen to be 1 pixel = 1 µm. The filters work in 3-D with the same σ in X, Y and Z (in pixels), so when the Z step is larger than the XY pixel, a given σ covers a larger physical distance in Z.

| σ value | Rough neighbourhood size | What the classifier "sees" at that scale |
|---|---|---|
| **0.3 – 0.5** | Sub-pixel to ~1 px | Fine texture, shot noise, very thin edges — only useful if objects are only a few pixels wide or you need micro-detail. |
| **1 – 2** | ~2–4 px across | Small cells, nuclei, tight boundaries (common in `small_preset` / `medium_preset`). |
| **3 – 5** | ~6–10 px across | Medium cells, organoid rims, local context around one object. |
| **10 – 15** | ~20–30 px across | Large clusters, organoid bodies, slow background gradients over a patch. |
| **25** | ~50 px across | Very coarse field context; whole-organoid or tissue-scale shading. |

**Rule of thumb:** pick σ in the same ballpark as **half the diameter (in pixels)** of the structure you want that filter to describe. Example: a nucleus ~8 px wide → try **LoG** or **DoG** at σ `2–4`; an organoid ~200 px wide → add σ `10–25` (as in `large_preset`), not only σ `0.5`.

The grid columns come from the **Custom Sigmas:** list. A preset is just a pre-checked subset of filter × sigma pairs:

| Preset | Checked pairs | When to use |
|---|---|---|
| `small_preset` | 12 (all 4 filters × σ 1, 2, 5) | Fast iteration; small objects; limited GPU memory. |
| `medium_preset` | 16 (× σ 1, 2, 5, 15) | Balanced middle ground. |
| `large_preset` | 20 (× σ 1, 2, 5, 10, 25) | Default; noisy or multi-scale data; objects vary a lot in size within one field. |
| `custom` | You choose | After you know which rows matter; edit the grid manually. |

**Failure modes → what to try**

- Mask looks **fragmented or speckled** → try `large_preset`, or add larger σ values in **Tune Features** (e.g. include `5`, `10`) so the classifier sees broader context.
- **Training or preview is too slow** / GPU runs out of memory → use `small_preset` or `medium_preset`; uncheck redundant **channels**; in **Tune Features**, uncheck whole filter rows or large σ columns you do not need.
- **Under-segmentation** (merged objects) with a good mask → usually post-processing (strategy / EDT threshold), not features — but if boundaries look blurry in the probability map, add **DoG** or **SoG** at σ `1–2` before increasing **Max depth**.

**Which channels to keep in Image Channel Inputs**

Each checkbox is an **image channel** the classifier may use. The forest does **not** know your metadata — it only learns "label `2` = foreground for **this** cell-type tab." Check channels that help it **tell foreground apart from everything else**, including other objects in the same volume.

Typical pattern on the **organoid** tab:

- Check the channel(s) where **organoids** are.
- Also check a channel where **confounders** stand out — e.g. a **T-cell** channel — so the model learns that T-cell voxels are **not** organoid foreground. Without that cue, organoid foreground can spill into space occupied by T cells.

Practical steps:

1. In napari, toggle channel visibility and overlay the label layer for this tab. For each candidate channel, ask: *"Does this channel help discriminate what I painted as foreground from what I left as background (and from other cell types)?"* If yes, leave it checked.
2. Start with a sensible superset (often all channels) and `medium_preset`, train once, and look at the preview.
3. Uncheck channels that add **no** discrimination.
4. **Ablation:** uncheck one channel, retrain, compare preview — if organoids bleed into T-cell regions (or vice versa), put that discriminative channel back.

More checked channels → more feature columns and slower runs; keep the set that improves **separation**, not every channel "just in case."

**Show classifier statistics and "dominating" features**

After training (or via **Show classifier statistics**), the **Classifier Statistics — <Cell type>** window lists **Feature** and **Importance**, one row per filter × sigma × channel combination (e.g. `difference_of_gaussian_sigma2_channel0`). Higher importance means the forest relied on that column more when splitting. The Importance cells are colour-coded from **green (more informative)** toward **red (less informative)**.

Start with the proposed preset and, when object scale is uncertain, a broader candidate set; then use this table to remove consistently low-importance rows before retraining and checking the preview.

- **One row dominates** (much higher than the rest). Often the classifier found one strong cue (e.g. a T-cell channel at σ `2`). It is a problem if the preview looks **overfitted** to your painted labels, or if importance is high on a **noisy or artefact** channel you should disable.
- **Many rows near zero** → safe to uncheck those filter × sigma boxes in **Tune Features** (or drop channels) and retrain for a faster, leaner model.
- **Dominating filter** means a **specific cell of the grid** (e.g. **DoG** at σ `1`). Fix by unchecking it if it is redundant, or by fixing labels/channels if the model is latching onto the wrong cue.

**Consider original image as well** is on in the standard presets. Raw intensity helps most when it alone separates foreground and background (very bright nuclei, etc.); otherwise the multiscale filters usually carry the information.

**Classifier — Max depth and Trees**

- The defaults (depth `5`, `100` trees) keep trees fairly shallow because pixel classification uses small label sets (APOC's own default depth is `2`). The workshops use **Max depth** 5 and **Trees** 100.
- **Lower Max depth** (e.g. 2–3) if the model looks great on the painted pixels but poor elsewhere; **raise it** a little only if it clearly under-fits (e.g. on textured backgrounds) — deeper trees fit the painted pixels more closely and make the classifier slower.
- **Raise Trees** (e.g. 100 → 200) for smoother predictions, with diminishing returns, at the cost of training and prediction time.

**Strategy choice**

- Cells in your data **never touch** → use **APOC (Direct Instance Segmentation)**. No watershed parameters to tune.
- Cells touch and the binary mask is clean → use **APOC Mask + EDT/Watershed Resegmentation**.
- The classifier output is **uncertain at boundaries** → use **APOC Probability Map + Watershed** and tune **Mask threshold** / **Seed threshold**.

**Post-processing parameters** — if watershed results look wrong (merged cells, speckles, etc.), see [Tuning (failure mode → fix)](./index.md#instance-post-processing-tuning) on the [segmentation overview](./index.md); the same tips apply to APOC, ConvPaint, and Pixel Classifier.

### 6 · Batch segmentation

Standard batch controls in the **🚀 Batch Segmentation** group — see [Batch segmentation controls](./index.md#batch-segmentation-controls). The run/queue buttons here are **▶ Run APOC Batch Segmentation** and **+🛒**. The batch uses each tab's strategy, channels and post-processing parameters, and refuses to start if a classifier was trained with an incompatible feature set (retrain the listed cell types). With **Advanced (per cell type)**, queuing requires a trained classifier for every cell type.

**Segmenting a single cell type.** Below the train buttons, **▶ Run <Cell type> Segmentation** (the label follows the active tab) runs the full batch — all samples — for the **active tab only**. It has its own **Process all timepoints** checkbox (ON) with **From t:** / **to t:** fields (`0`–`0`) when unchecked, and asks about existing results for that cell type.

- Only that cell type's classifier is used, and only its `…_segments.zarr` / `…_mask.zarr` are written. Other cell types' outputs are not touched.
- The **Dead** tab runs only the death mask; a cell-type tab never recomputes the death mask.
- If the dataset has multicolor channels (`<name>_<n>_multicolor`), the overlap cleanup that follows still processes the whole family, because resolving overlaps needs every member.
- Each sample is read from disk once per run no matter how many cell types are selected, so running one cell type is faster mainly in the classifier and post-processing part.

## Outputs

| File | Path |
|---|---|
| Per-cell-type instance labels | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr` |
| Per-cell-type foreground mask | `<output_dir>/images/<sample>/<sample>_<cell_type>_mask.zarr` |
| Death mask (dead channel only) | `<output_dir>/images/<sample>/<sample>_mask_dead.zarr` |
| Trained APOC classifiers | `<output_dir>/images/PixelClassification/PixelClassifier_<CellType>.cl` (one per cell type; `PixelClassifier_Death.cl` for the dead channel) |
| Training bundle (for Import / Export) | `<output_dir>/images/PixelClassification/PixelClassifier_TrainingData.zip` |
| Saved labels and previews | `<output_dir>/images/PixelClassification/` |

## Good practices & tips

- **Pick the preset to match your object size, with APOC Probability Map + Watershed.** Use `small_preset` for small objects such as nuclei or T cells, and `medium_preset` or `large_preset` when objects are large or vary a lot in size (see [Tuning the parameters](#tuning-the-parameters)). If training is slow or GPU memory is tight, a smaller preset is often enough. Probability Map + Watershed handles touching cells well. Switch to Direct Instance Segmentation only if your cells never touch.
- **Save labels before switching method.** Labels are saved when you train, and you are asked whether to save when you leave the Segmentation tab — but switching to another method removes the training layers without asking. Click **Save User Labels** after any substantial painting.
- **Keep RF trees fairly shallow (Max depth ~5, the default).** Interactive labeling gives small label sets, which deep trees can fit too closely.
- **Use the classifier statistics to slim down the feature set.** Rows with importance near zero can be unchecked in **Tune Features**, or their channel dropped. After retraining, fewer features need computing, which cuts run time and GPU memory. One dominating feature is often just a strong cue, such as a marker channel. It points to a problem only if the preview looks overfitted to your labels or the feature comes from a noisy or artefact channel (see [Tuning the parameters](#tuning-the-parameters)).
- **Workers do not speed up APOC.** APOC inference is GPU-bound and runs sample by sample, so **Workers:** has no effect here. How long each timepoint takes depends mostly on your GPU, the volume size and the number of checked channels and features, so time a short timepoint range on your own data before queuing a full batch. The setting is shared: changing it also changes it for Zarr conversion and the Pixel Classifier (see [Workers](../../plugin_essentials/running_steps.md#workers)).
- **Several cell types (large and small)?** See [Advice: large vs small objects](./index.md#advice-large-vs-small-objects) on the overview; use **Advanced (per cell type)** when each type needs a different strategy, and **⛓ Apply same settings to all organoids** when several organoid types should share one setup.
- **Reuse labels across experiments** with the same imaging setup via **Export Training Bundle…** / **Import Training Data** (section 5) instead of repainting from scratch.

## See also

- [Segmentation overview](./index.md) for the method comparison.
- [ConvPaint](convpaint.md) — deep-features alternative when classical features aren't enough.
- [Processing Queue](../../plugin_essentials/processing_queue.md) for batching segmentation steps.
- [APOC project on GitHub](https://github.com/haesleinhuepf/apoc) — upstream documentation.

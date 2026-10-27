# 🩹 Troubleshooting

Symptom-first quick reference. Find your **method** below, then your **problem**. For full explanations see [Instance post-processing parameters](./index.md#instance-post-processing-parameters) and the per-method pages ([APOC](apoc.md), [ConvPaint](convpaint.md), [Pixel Classifier](pixel_classifier.md), [Cellpose](cellpose.md)).

## Which method should I use?

| Your situation | Use |
|---|---|
| Already segmented elsewhere (Imaris, Ilastik, script, …) | **Import segmentation** |
| A Cellpose model trained on images like yours exists | **Cellpose** |
| Strong GPU / HPC, clean channels (no bleed-through), no time to label, and its preview looks right | **Cellpose-SAM** (zero-shot) |
| GPU with working OpenCL, can paint a few labels (any cell shape) | **APOC** |
| APOC's mask stays wrong after more labels (subtle texture, weak boundaries), PyTorch device available | **ConvPaint** |
| No GPU available | **Pixel Classifier** (CPU) |

Full decision tree: [How to pick a method](./index.md#how-to-pick-a-method).

---

## APOC

### The raw mask is wrong (before splitting)

| Symptom | Fix |
|---|---|
| Foreground bleeds into background | Paint more **background** labels near the misclassified region, retrain |
| Cells not detected at all | Paint more **foreground** labels at the missed cells' edges, retrain |
| Good on trained timepoints, poor elsewhere | Raise **Examples/sample**, generate new training data, retrain. If it still fits the painted pixels much better than the rest, the model may be overfitting: **lower Max depth** (e.g. 2–3) and retrain |
| Mask fragmented / speckled | Switch **Preset:** to `large_preset`, or add larger σ in **Tune Features** |
| Training/preview slow, GPU low on memory | Switch to `small_preset`/`medium_preset`, uncheck unneeded channels in **Image Channel Inputs** |
| No GPU listed / GPU unstable | Tick **Force CPU-only processing** (needs an OpenCL CPU runtime), or use the Pixel Classifier |

```{tip}
Filter rows: **Gauss** = smoothed intensity (large/soft objects), **DoG** and **LoG** = spot/line responses of about σ size (nuclei, cores, thin processes), **SoG** = edge strength in any direction (cell borders, membranes). σ scales roughly with **half the pixel-diameter** of the structure you want that filter to describe. Full table: [APOC — Tuning the parameters](apoc.md#tuning-the-parameters).
```

### Touching cells are merged, or one cell is split into pieces

APOC has two watershed strategies for splitting objects — check which one you're using (**Strategy:** dropdown above the cell-type tabs, or the tab's **↳ Cell type strategy:** in Advanced mode). **APOC (Direct Instance Segmentation)** never splits; switch to one of these if you need splitting.

**If you're using "APOC Mask + EDT/Watershed Resegmentation":**

| Symptom | Fix |
|---|---|
| Touching cells merged into one label | **Raise EDT threshold** (the default `1.0` px splits nothing); if it is already high, try **lowering** it ([details](#raising-edtseed-threshold-doesnt-seem-to-help)) |
| One cell split into several labels | **Lower EDT threshold** |

**If you're using "APOC Probability Map + Watershed" (the default strategy):**

| Symptom | Fix |
|---|---|
| Touching cells merged into one label | **Raise Seed threshold**; if it is already high, try **lowering** it ([details](#raising-edtseed-threshold-doesnt-seem-to-help)) |
| One cell split into several labels | **Lower Seed threshold** |

```{note}
Don't mix these up — EDT threshold and Seed threshold are different fields that only apply to their own strategy. See [Common post-processing problems](#common-post-processing-problems-any-method) below for Mask threshold, Min size, Opening, and the "no sweet spot found" issue.
```

**Finer features can help too, not just post-processing:** if the mask/probability boundary itself is blurry between two touching cells (not just a post-processing problem), adding smaller-σ features (e.g. DoG or SoG at σ 1–2 in **Tune Features**) can sharpen the boundary enough that watershed splits it correctly without needing extreme threshold values — see [APOC — Tuning the parameters](apoc.md#tuning-the-parameters).

---

## ConvPaint

### The raw mask is wrong (before splitting)

| Symptom | Fix |
|---|---|
| Foreground bleeds into background | Paint more **background** labels near the misclassified region, retrain |
| Cells not detected at all | Paint more **foreground** labels at the missed cells' edges, retrain |
| Classes stay confused after more labels | Switch **Model:** to VGG16 Medium / Large or DINOv2 / JAFAR DINOv2, retrain |
| Out-of-memory error during training/prediction | Switch to a smaller **Model:** (e.g. VGG16 instead of VGG16 Large), raise **Downsample:**, or tick **Tile image** |
| Speckly mask with single-pixel noise | Raise **Smoothen:** (values from `2`; it acts on the mask, not on the probability map) |
| Thin structures eroded | Lower **Smoothen:** (`0` or `1` = off) |
| A rare cell type is under-predicted | Paint more pixels of that cell type, retrain |
| Preview says the cell type is not in the trained model | Paint that cell type's label value and retrain |

### Touching cells are merged, or one cell is split into pieces

ConvPaint has two strategies (**Strategy:** dropdown, or the tab's **↳ Cell type strategy:** in Advanced mode):

**If you're using "ConvPaint Mask + EDT/Watershed" (the default strategy):**

| Symptom | Fix |
|---|---|
| Touching cells merged into one label | **Raise EDT threshold** (the default `1.0` px splits nothing); if it is already high, try **lowering** it ([details](#raising-edtseed-threshold-doesnt-seem-to-help)) |
| One cell split into several labels | **Lower EDT threshold** |

**If you're using "ConvPaint Probability + Watershed":**

| Symptom | Fix |
|---|---|
| Touching cells merged into one label | **Raise Seed threshold**; if it is already high, try **lowering** it ([details](#raising-edtseed-threshold-doesnt-seem-to-help)) |
| One cell split into several labels | **Lower Seed threshold** |

See [Common post-processing problems](#common-post-processing-problems-any-method) below for Mask threshold, Min size, Opening, and the "no sweet spot found" issue.

---

## Pixel Classifier

### The raw mask is wrong (before splitting)

| Symptom | Fix |
|---|---|
| Foreground bleeds into background | Paint more **background** labels near the misclassified region, retrain |
| Cells not detected at all | Paint more **foreground** labels at the missed cells' edges, retrain |
| Good on trained timepoints, poor elsewhere | Raise **Examples/sample**, generate new training data (this starts from empty labels), retrain |
| Good on bright cells, fails on dim ones | Include some dim cells in labels, or switch to APOC/ConvPaint (richer features) |

### Touching cells are merged, or one cell is split into pieces

Pixel Classifier always uses the EDT/Watershed mechanism — there is no strategy dropdown (the field is called **EDT:**):

| Symptom | Fix |
|---|---|
| Touching cells merged into one label | **Raise EDT threshold**; if it is already high, try **lowering** it ([details](#raising-edtseed-threshold-doesnt-seem-to-help)) |
| One cell split into several labels | **Lower EDT threshold** |

See [Common post-processing problems](#common-post-processing-problems-any-method) below for Min size and Opening.

---

## Cellpose

Cellpose exposes almost no GUI parameters — tuning means picking the right model, fixing metadata, or fine-tuning a model.

| Symptom | Fix |
|---|---|
| Cells missed, or random blobs in background | Re-check the channel mapping (**1. Channel Configuration**) first, then reload the model |
| Two touching cells fused into one label | Fine-tune the model on touching-cell examples, or manually split labels after tracking |
| Cells split into multiple labels | Fix pixel sizes (Z anisotropy) in your metadata |
| Inference very slow | Confirm the GPU is being used; test on a small timepoint range first |

---

## Cellpose-SAM

Zero-shot — no model to load or train. Tuning means the two thresholds, do_3D vs 2D+stitch, and the size filter. Full page: [Cellpose-SAM](cellpose_sam.md).

| Symptom | Fix |
|---|---|
| First run fails / "environment not ready" | Click **Set up Cellpose-SAM environment** once (Section 1) before running |
| CUDA out-of-memory | **Lower Batch size** first (Section 4) |
| Touching cells fused into one label | Raise **Cell prob threshold** slightly, or split manually after tracking (Cellpose has no watershed splitting) |
| Many small / flat fragments | Keep **Remove flat (single-slice) segments** on; set a **Min size** in Section 5 (preview first) |
| Large organoids silently vanish | Check **Max size fraction** = 1.0 (BEHAV3D EXPLORER default), not Cellpose's 0.4 |
| Cells over-split in Z | Fix pixel sizes in metadata (they set the Z anisotropy) |
| Machine powers off mid-run (laptop) | Lower **Batch size**, cap **CPU threads (0=all)** and add **Cooldown between frames (s)**; re-run and choose **Skip & Resume** to continue |

---

(common-post-processing-problems-any-method)=
## Common post-processing problems (any method)

These parameters mean the same thing regardless of method — but not all of them are shown for both strategies (noted per row). The only case with *no* post-processing controls at all is APOC's **Direct Instance Segmentation** strategy (and the **Dead** tab).

| Symptom | Fix |
|---|---|
| Background labelled as cells | **Paint more background** where background is called foreground (the [probability map](index.md#reading-the-probability-map) shows where it reads high) and retrain. Last resort: **raise Mask threshold** (Probability Map strategy only); aim to keep the default `0.5` |
| Real cells lost at the edges | **Paint more foreground** along the cell edges where the [probability map](index.md#reading-the-probability-map) reads low and retrain. Last resort: **lower Mask threshold** (Probability Map strategy only); aim to keep the default `0.5` |
| Object has a hole / donut-shaped gap in the middle | **Paint more foreground** in the object centres where the [probability map](index.md#reading-the-probability-map) reads low and retrain. With Mask + EDT/Watershed you can also **enable Fill holes** (Probability Map always leaves holes unfilled; there's no toggle for it) |
| Tiny speckles appear as labels | **Raise Min size** (both strategies) |
| Real small cells dropped | **Lower Min size** (both strategies) |
| Fuzzy / jagged boundaries, stray voxels | **Raise Opening px** (both strategies) |
| Thin structures eroded | **Lower Opening px** (or `0`) (both strategies) |
| EDT threshold / Min size values look very different from a protocol | Check the **Units:** switch — values are shown in µm / µm³ by default; switch to **px** to compare with pixel values |

### Raising EDT/Seed threshold doesn't seem to help

There's a working **range**, not just a direction — both extremes look "merged":

- **Too low**: the seed region for a touching pair is still one connected blob → no split happens, same visible result as "merged." (EDT threshold up to 1 px, or Seed threshold at or below Mask threshold, never splits.)
- **Too high**: the smaller cell's core, and eventually every core, falls below the threshold → fewer than two seeds are left → the object is kept *whole, unsplit* → again looks "merged," even though you raised the threshold.

So raising EDT/Seed threshold and seeing no change doesn't necessarily mean you're going the wrong direction — you may have overshot past the object's own range. **Step through values with Run instance segmentation (Test Segmentation Parameters in Pixel Classifier) and watch the `<Cell type> Segments` layer** (for the probability strategies, the `Probability Map (<Cell type>)` layer shows where the cores are): merged (one seed) → split (two or more) → merged again (fewer than two) is the expected shape of the curve, and the sweet spot is the middle range, not one edge. In APOC and ConvPaint, an EDT threshold above about half the number of Z slices an object spans leaves it no seed (see [How the distance is measured](./index.md#instance-post-processing-parameters)).

## Big vs small objects — starting points

| Object type | EDT / Seed threshold starting point | Min size |
|---|---|---|
| Big (organoids) | EDT ~8–15 px | large, e.g. 1000 voxels |
| Small / crowded (immune, T cells) | EDT ~1.5–4 px | small, e.g. 10 voxels |

These are pixel values (the Pixel Classifier's built-in starting points); set **Units:** to **px** to enter them. In APOC and ConvPaint the EDT threshold must also stay below about half the number of Z slices the object spans. Tune **each cell type on its own tab** — settings on one tab do not carry over to another. Details: [Advice: large vs small objects](./index.md#advice-large-vs-small-objects).

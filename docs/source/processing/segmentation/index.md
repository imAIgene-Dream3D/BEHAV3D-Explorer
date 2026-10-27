# 🦠 Segmentation

Tab 3 of the BEHAV3D EXPLORER dock widget. Segmentation turns each timepoint of the raw zarr into **instance labels**, one integer label per cell, written to:

```
<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr
```

The zarr is a 4-D `(T, Z, Y, X)` `uint16` array; pixel value `0` is background; positive integers are individual cell IDs *within a timepoint* (track IDs are assigned later by the [Tracking](../tracking/index.md)).

The tab shows a placeholder until metadata is loaded in the Data Preparation tab. A log box at the bottom of the tab records every step.

```{tip}
In a hurry? [Troubleshooting](troubleshooting.md) is a one-page, symptom → fix quick reference for method choice, classifier issues, and instance-processing parameters.
```

## The six methods

The **Method:** dropdown at the top (group **Segmentation Method**) picks the segmentation method; the body of the tab swaps to show that method's controls. It opens on **APOC (GPU)**.

| Method (dropdown label) | What it is | Hardware |
|---|---|---|
| **APOC (GPU)** | Pixel classifier trained on sparse user labels — a random forest on classical image filters (smoothed intensity, edges, spot/line filters) computed in 3-D at scales you choose | GPU with OpenCL drivers (a **Force CPU-only processing** option needs an OpenCL CPU runtime) |
| **ConvPaint (DL pixel classifier)** | Pixel classifier trained on sparse user labels — features come from a pretrained deep network (VGG16, DINOv2, …), computed in each Z slice separately, and one gradient-boosted classifier predicts every cell type at once | PyTorch (CUDA GPU recommended; CPU possible) |
| **Pixel Classifier (Random Forest)** | CPU pixel classifier trained on sparse user labels — random forest on smoothed-intensity and edge features computed in 3-D at fixed physical scales | CPU |
| **Cellpose (Deep Learning)** | Deep-learning instance segmenter (cellpose v3) that runs a model file trained for your kind of images — outputs cell-level masks directly | PyTorch (CUDA GPU strongly recommended) |
| **Cellpose-SAM (zero-shot)** | Foundation-model instance segmenter (cellpose v4) — segments many cell shapes with no training or annotation. Its authors report the best out-of-the-box accuracy on their benchmarks; it is also the most compute-hungry method here. | PyTorch (good CUDA GPU or HPC) |
| **Import segmentation** | Checks segmentations produced outside BEHAV3D EXPLORER (`.tif`/`.tiff`/`.zarr`) and records them in the metadata: valid Zarrs are linked where they are, TIFFs and badly laid-out Zarrs are converted | n/a |

```{note}
**Two Cellpose entries.** **Cellpose (Deep Learning)** is the classic cellpose v3 backbone you load a pretrained/retrained model into (see [Cellpose](cellpose.md)). **Cellpose-SAM (zero-shot)** is the newer cellpose v4 foundation model that needs no model file and no training (see [Cellpose-SAM](cellpose_sam.md)). It runs in its own Python environment, which you set up once from its page.
```

```{warning}
Switching to another method while APOC, ConvPaint or Pixel Classifier training layers are loaded asks for confirmation and then **removes those layers without saving your labels**. Save your labels first (see [Labeling foreground and background](#labeling-foreground-and-background)).
```

## How to pick a method


```{mermaid}
flowchart TD
    A["Your situation"] --> Q1{"Already segmented<br/>elsewhere outside <br/>BEHAV3D EXPLORER? 
    (Imaris,Ilastik, custom script, …)"}
    Q1 -- "Yes" --> Import["Import segmentation"]
    Q1 -- "No" --> Q2{"Trained Cellpose model<br/>for this kind of data?"}

    Q2 -- "Yes" --> Cellpose["Cellpose v3<br/>(load the model)"]
    Q2 -- "No" --> QG{"Do you have a GPU?"}

    QG -- "No" --> PCrf
    QG -- "Yes" --> Q3{"Channels clean?<br/>(one cell type per channel,<br/>no bleed-through)"}

    Q3 -- "No" --> APOC
    Q3 -- "Yes" --> Q5{"Can you tolerate a long<br/>runtime?"}

    Q5 -- "No" --> APOC
    Q5 -- "Yes" --> Q6{"Cellpose-SAM preview right?<br/>(long, thin cells not split,<br/>neighbours not fused)"}

    Q6 -- "Yes" --> SAM["Cellpose-SAM<br/>(zero-shot)"]
    Q6 -- "No" --> APOC

    subgraph PC ["Pixel classifiers (you paint sparse labels)"]
        PCrf["Pixel Classifier<br/>(Random Forest, CPU)"]
        APOC["APOC<br/>(start here)"]
        APOC --> Q9{"Foreground mask right<br/>after adding labels?"}
        Q9 -- "No" --> ConvPaint["ConvPaint"]
        Q9 -- "Yes" --> Tune["Tune the splitting<br/>(post-processing)"]
        ConvPaint --> Tune
        PCrf --> Tune
    end

    classDef method fill:#d7f0e4,stroke:#1b7a5c,stroke-width:2px,color:#0b3d2e;
    class Import,Cellpose,SAM,PCrf,APOC,ConvPaint method;
```

```{note}
**Cellpose-SAM vs. Cellpose v3.** **Cellpose-SAM** is zero-shot (works out of the box, no training or annotation, but slowest/most compute-heavy). Classic **Cellpose v3** needs a pretrained or retrained model — take that path only if one already exists for your sample type.
```

Practical rules:

- **Import segmentation** is *not* a way to create segmentations: it checks existing masks (converting them to Zarr where needed) so the rest of the pipeline can consume them. Use it first if this covers your case — it skips everything else.
- **Cellpose (v3)** is the quickest route when the lab already has a model trained on images like yours (see [Cellpose page](cellpose.md) and [Zenodo](https://zenodo.org/records/18872978)), or when you can train one on full masks of your own. The page runs the model file you give it; without a suitable model, use Cellpose-SAM or a pixel classifier instead.
- **Cellpose-SAM (v4, zero-shot)** is worth trying when you have a GPU, clean channels (no bleed-through, one cell type per channel) and time for a long run. It needs **no training and no annotation**, but it is the most compute-hungry method (see [Cellpose-SAM page](cellpose_sam.md)). Check its preview on a representative timepoint first: like other Cellpose models it can split long, thin cells (the Cellpose documentation suggests more **niter dynamics** iterations for long objects), and because it has no labeling step it cannot learn to ignore bleed-through between channels, whereas the pixel classifiers can be taught to by painting it as background (see [Labeling tips](#labeling-tips)). If the preview is wrong, use a pixel classifier.
- Otherwise, train a **pixel classifier** — pick by hardware, then by how well the classifier learns your cells:
  - **APOC** needs a GPU with working OpenCL drivers. It classifies every voxel from classical filters, so it is not limited to round, blob-like objects (see below). You paint a few foreground/background pixels per cell type. Training is quick; inference is fast on a modern GPU. Start here when you have a GPU.
  - **ConvPaint** uses pretrained deep features instead of classical filters — reach for it when APOC's foreground mask stays wrong after you add labels (subtle texture differences, low contrast, ambiguous boundaries). Needs a PyTorch-compatible device (CUDA GPU strongly preferred; CPU works but is slow).
  - **Pixel Classifier (Random Forest)** is the CPU-only fallback. Use it when no GPU is available.
  - All three split touching cells with the **same** post-processing. If the mask is right but neighbouring cells stay fused, tune the [post-processing](#instance-post-processing-tuning) (or paint a thin background line between them, see [Labeling tips](#labeling-tips)) rather than switching method.

```{note}
**Cellpose** and **Cellpose-SAM** perform **true instance segmentation in one shot** — they directly output separated cell labels, no pixel labeling and no watershed post-processing. The three pixel classifiers (APOC, ConvPaint, Pixel Classifier) instead output a foreground/background mask or a per-class probability, which BEHAV3D EXPLORER turns into instance labels by [watershed post-processing](#instance-post-processing-parameters).
```

**Choosing between the three pixel classifiers.** All three are trained by painting a few labels (see [Labeling foreground and background](#labeling-foreground-and-background)) and differ mainly in *where their features come from* and *what hardware they need*:

| Method | Hardware | Features | Speed / cost | Choose when |
|---|---|---|---|---|
| **APOC** | GPU with working OpenCL | Classical filters (Gauss, DoG, LoG, SoG) at scales you choose, computed in 3-D on the GPU | Fast to train and run; GPU memory grows with the number of filters × channels | **Start here** whenever you have a GPU |
| **ConvPaint** | CUDA GPU recommended | Deep features from a pretrained network (VGG16, DINOv2, …), computed in each Z slice | Slower feature extraction, more GPU memory | APOC's foreground mask stays wrong after adding labels — see below |
| **Pixel Classifier (Random Forest)** | CPU only | Fixed set: smoothed intensity + edges at four physical scales, in 3-D (not editable) | Runs on the CPU, so usually the slowest | No GPU is available at all |

The cutoff between APOC and ConvPaint is **feature complexity vs. speed/memory**.

**APOC**'s classical filters each describe one simple property around a voxel at one scale — smoothed intensity (Gauss), edge strength (SoG) or a spot/line response (DoG, LoG). The random forest combines many of them voxel by voxel, so APOC is not tied to a cell shape: round or elongated cells are learned alike, as long as their local intensity, edges or texture at these scales differ from the background. It does so fast and cheap.

**ConvPaint** instead runs each Z slice through a pretrained CNN or transformer, whose features encode learned, higher-order visual patterns (texture, local context). That can help when there are subtle texture differences or low-contrast, ambiguous boundaries that classical filters do not capture. The price is slower iteration, more GPU memory, no 3-D context in the features, and a feature space that's harder to inspect than APOC's feature-importance table — so try APOC first and reach for ConvPaint when APOC's mask stays wrong despite more labels. Splitting of touching cells is the same in both (see [Instance post-processing parameters](#instance-post-processing-parameters)).

## Common structure of the pixel-classifier pages

The three pixel-classifier methods (APOC, ConvPaint, Pixel Classifier) follow the same workflow:

1. **Training** — click **Generate Training Data**, paint labels, train the classifier(s).
2. **Preview / fine-tune** — set the post-processing parameters per cell type and preview the instance labels before committing to a batch run.
3. **Batch segmentation** — process all timepoints or a custom range across every sample in the metadata. See [Batch segmentation controls](#batch-segmentation-controls).

**Generate Training Data** (same for all three):

- **Examples/sample** (default `3`, range 1–10) sets how many timepoints are taken from each sample. They are **evenly spaced** from the first to the last timepoint of the movie, not random.
- The images are read in the background with a progress bar; napari stays responsive and the other tabs are locked meanwhile. The viewer is only cleared once the new layers are ready, so a failed load leaves your previous session untouched.
- If training images were already generated for this project, a **Training Data Detected** dialog offers **Generate New**, **Load Existing** or **Cancel**. Previously saved labels are restored when they still fit the loaded images (Pixel Classifier: only with **Load Existing** — **Generate New** clears its saved labels).
- The viewer then shows one image layer per channel (`Channel …`) plus empty or restored label layers to paint into.
- Leaving the Segmentation tab while training layers are loaded asks whether to leave and then whether to save your labels; the training layers are then removed.

The per-method pages document the exact widgets, parameters, and tips for each one. The labeling step, the post-processing parameters, and the batch controls are shared and described once below.

(labeling-foreground-and-background)=
## Labeling foreground and background

The pixel classifiers are trained by painting a few labels into the label layers created by **Generate Training Data**. How the labels are organised differs:

- **APOC and Pixel Classifier are binary, one label layer per cell type**, named `User Provided Labels (<Cell type>)`. You paint two classes:

| Brush value | Meaning | Colour |
|---|---|---|
| `0` | Eraser | — |
| `1` | Background | red |
| `2` | Foreground (the cell type you are training) | cyan |

  An **Annotation legend** appears below the training controls with one row per cell type and three buttons — **0: Eraser**, **1: Background**, **2: Foreground**. Clicking a button selects that cell type's label layer, switches it to paint mode and sets the brush value.

- **ConvPaint is multi-class on a single layer**, `User Provided Labels`. Background is always `1`; each cell type gets its own value (`2`, `3`, … in the order shown in the legend). The **Annotation legend** table lists **Idx**, **Color**, **Cell type** and **Annotated voxels**; click a row to paint that class, and **Refresh counts & colors** to update the voxel counts.

- **Dead channel** (only when a dead channel is declared in the metadata): every method adds a separate binary layer `User Provided Labels (Dead)` (`1` = background, `2` = dead signal), with its own **Dead** row of legend buttons.

### Labeling tips

Practical tips that apply to all three methods and can give cleaner segmentations:

- **You only need enough pixels to disambiguate the classes** - You do *not* necessarily need to paint whole cells every time. Parts of the cell/organoid are often enough
- **Label examples from different timepoints and Z-slices** (e.g. start / middle / end and a few Z levels) so the classifier sees the variability in your data.
- **Focus labeling on regions where segmenation is liekly going to be more difficult** - don't only label the easiest cells, label across the complexity of the dataset.
- **Draw background first, then overlay foreground.** Lay down the background class along and just over the object edge, then paint the foreground over the part that sits on the object. Layering it this way produces a very clean border.
- **You don't have to always label cells completely**. Often focusing on the boundary of background and foreground for a part of the cell or any visually-confusing region (touching cells, low-contrast areas) is enough — that is where pixel classifiers fail most.
- **Split touching objects with a thin background line.** Where two objects touch, draw a thin background line between them, bordered by foreground, so the classifier learns to separate them. For objects tracked by **fragmentation propagation**, you only need to get this right on **timepoint 1** of each sample — the method copies the segmentation forward.
- **Draw thin / elongated cells thicker than the signal (~3 px).** Segmentation connects adjacent pixels, so a one-pixel-wide stroke on an elongated T cell leaves gaps; a slightly over-thick line yields one clean segment.
- **Paint channel bleed-through as background.** Where another channel's signal bleeds into the one you are training (e.g. organoid signal appearing in the T-cell channel at higher Z), explicitly paint it as background — even when it is much brighter — so the model learns to ignore it.
- **For the dead mask, label the bright dead blobs.** Concentrate on the clear high-intensity dead signal; including very dim red is optional and tends to be noisier.
- **Save your labels often** (**Save User Labels** in APOC, **💾 Save User Labels** in ConvPaint, **Save Labels** in Pixel Classifier). Labels are also saved automatically when training starts. Saving runs in the background: the labels are copied at the moment you click, so you can keep painting while they are written (strokes made after the click go in the next save).
- **Use the prediction and [probability map](#reading-the-probability-map) after training to find regions that need more labeling** After training, look at the prediction (and the [probability map](#reading-the-probability-map), where available) to see where the model struggles and label more data there. This would for example be background where the probability map has values that are close to being called foreground or clear foreground that is not labeled as foreground yet.

### Reading the probability map

APOC (always) and ConvPaint (with **ConvPaint Probability + Watershed**) add a `Probability Map (<Cell type>)` layer (hidden by default; toggle it on in the layer list) with the classifier's confidence per voxel (bright = confident foreground, dark = low confidence). Use it to target corrections:

- Aim for **background clearly below ~0.3–0.4** and **foreground above ~0.7**. With the default **Mask threshold** `0.5`, anything above 0.5 becomes foreground, so keep borderline pixels away from that line to stop them flickering across slices.
- Where an area *should* be foreground but reads low, paint a little more **foreground** there (or **background** where it wrongly reads high) and **retrain** — only small tweaks are usually needed.

(instance-post-processing-parameters)=
## Instance post-processing parameters

APOC, ConvPaint and the Pixel Classifier are *pixel* classifiers: their raw output is a foreground/background mask or a per-class probability map. BEHAV3D EXPLORER turns that into **instance labels** with a watershed-based post-processing pipeline: fill holes → opening → seeds → watershed → size filter. The same parameters appear (with method-specific defaults) in each cell type's settings; their meaning is identical everywhere:

| Parameter (GUI label) | Meaning |
|---|---|
| **EDT threshold** (**EDT** in Pixel Classifier) | Threshold on the Euclidean Distance Transform of the mask, i.e. each foreground voxel's distance (in pixels) to the nearest background voxel. Each connected group of voxels at least this deep inside the mask is one **seed**; an object containing two or more seeds is cut between them by a watershed. **Higher → seeds shrink toward each object's core, breaking the thin neck between touching objects → more splitting**; **lower → neighbouring cores stay connected → objects stay merged**. Every voxel of a mask is at least 1 px from the background, so values up to 1 px (`0` and the APOC/ConvPaint default `1.0` included) leave one seed per object: nothing is split. |
| **Opening px** (**Opening** in Pixel Classifier) | Morphological opening of the mask with a disk of this radius (pixels), in each Z slice, before watershed. Smooths boundaries and removes protrusions and specks narrower than about twice the radius. `0` disables. |
| **Min size** | Objects smaller than this volume are discarded at the end (noise / debris filter). With the EDT strategies it also limits splitting: a piece smaller than Min size produced by the cut is merged back into its neighbour. |
| **Fill holes** | Fill gaps fully enclosed by the object in each Z slice, before watershed (recommended for solid objects). Not used by the probability strategies. |
| **Mask threshold** *(probability strategies)* | Foreground cutoff applied to the probability map: voxels above it are foreground. |
| **Seed threshold** *(probability strategies)* | Higher cutoff used to place watershed seeds: each connected group of foreground voxels above it is one seed, and the watershed runs on the probability map. **Higher → keeps only each object's confident core as a seed, splitting more** touching objects; lower merges neighbouring cores together, splitting fewer. Same direction as EDT threshold above. At or below **Mask threshold** every object is one seed, so nothing is split. |

Additional behaviour to know:

- **Units: px ⇄ µm.** Each cell type's parameter block has a **Units:** switch. When the metadata contains pixel sizes, **EDT threshold** and **Min size** are shown in **µm** and **µm³** by default; flip the switch to **px** to see pixels/voxels. This only changes the display — values are always saved and used in pixels/voxels (EDT threshold is converted with the XY pixel size, Min size with the voxel volume). **Opening** is always in pixels; Mask/Seed thresholds are probabilities (0–1).
- **How the distance is measured.** The Pixel Classifier measures it within each Z slice (in-plane distance). APOC and ConvPaint measure it in 3-D and count one Z step as one pixel, whatever the Z spacing; in a stack whose Z step is larger than the XY pixel, the deepest point of a cell is therefore about half the number of Z slices the cell spans, and a higher **EDT threshold** leaves no seed at all.
- An object is only split when it contains **two or more seeds**; with fewer it is kept whole (never deleted), so raising the threshold too far — until the smaller cell's core or every core disappears — looks "merged" again (see [Troubleshooting](troubleshooting.md)).
- Objects that are flat (only one voxel thick in Z, Y or X) are always removed by this post-processing.

(instance-post-processing-tuning)=
### Instance post-processing tuning

- Touching cells **merged** into one label → **raise EDT threshold** (Mask + EDT/Watershed strategy, Pixel Classifier; it has no effect until it is above 1 px), or **raise Seed threshold** (Probability strategies; keep it above Mask threshold). If it is already high, try **lowering** it instead ([details](troubleshooting.md#raising-edtseed-threshold-doesnt-seem-to-help)).
- One cell **split** into several labels → **lower EDT threshold**, or **lower Seed threshold**.
- Background labelled as cells → paint more background where the [probability map](#reading-the-probability-map) reads high and retrain; only as a last resort **raise Mask threshold** (aim for the default `0.5`).
- Cells lost at the edges → paint more foreground at the edges where the [probability map](#reading-the-probability-map) reads low and retrain; only as a last resort **lower Mask threshold** (aim for the default `0.5`).
- Small speckles appear as tiny labels → **raise Min size**; real small cells dropped → **lower Min size**.
- Fuzzy / jagged boundaries or single stray voxels → **raise Opening**; thin structures eroded → **lower Opening** (or `0`).

(advice-large-vs-small-objects)=
### Advice: large vs small objects

Tune **each cell type on its own tab** (or, for the Pixel Classifier, its own group box). Settings for **big** objects (e.g. organoids) are usually wrong for **small, crowded** ones (e.g. immune cells / T cells).

- Work **one cell type at a time**: train (or paint), preview, look at the labels, adjust, repeat. Settings for one cell type do not apply to another.
- **Big** (e.g. organoids): use a larger minimum size than for single cells. Stuck together → **raise** EDT threshold. Fuzzy outline → try a **probability** strategy (APOC, ConvPaint).
- **Small / crowded** (e.g. immune, T cells): **small** minimum size; **lower** EDT threshold to start. One cell split into many → **lower** it further.
- **Calculate before copying:** estimate a single cell's diameter from the image. With the **Units** switch on **µm** you can enter physical sizes directly. For **Min size**, a tolerant first preview is about half the estimated object volume. For **EDT threshold**, preview about 20–30 % of the object's diameter, then adjust; in APOC and ConvPaint also keep it (in px) below about half the number of Z slices the object spans (see [How the distance is measured](#instance-post-processing-parameters) and [Recommended settings](../../recommended_settings.md)). Values from other experiments only transfer when the acquisition (resolution, magnification) matches.
- **Several cell types in one experiment?** In APOC or ConvPaint, **Advanced (per cell type)** lets each type use a different strategy; still tune post-processing separately on each tab.

```{note}
Which parameters you see depends on the strategy you pick, and default numbers differ by method — see each method page. Treat built-in values as UI defaults, not biological recommendations. Always adjust for the current resolution and the **nature of your data** (big vs small, crowded vs sparse, sharp vs fuzzy boundaries).
```

(batch-segmentation-controls)=
## Batch segmentation controls

The pixel-classifier methods share the same batch-run controls (group **🚀 Batch Segmentation** in APOC and ConvPaint, **3. Batch Segmentation** in Pixel Classifier):

| Control | Default | Effect |
|---|---|---|
| **Process ALL timepoints** (**All timepoints** in Pixel Classifier) | ON | Segment the full movie of every sample. Uncheck to set a range. |
| **From t:** / **to t:** (Pixel Classifier: **Range: T:** start – end) | `0` / `100` (Pixel Classifier `0` / `0`) | Inclusive timepoint range, used only when the box above is unchecked (APOC/ConvPaint show it only then). |
| **Workers:** | half of your CPU cores | The shared [Workers](../../plugin_essentials/running_steps.md#workers) setting (changing it also changes it for Zarr conversion and the other segmentation methods). Only the Pixel Classifier uses it; APOC and ConvPaint process one sample at a time on the GPU and ignore it. In Pixel Classifier it sits in section 2. |
| **▶ Run …** | — | Runs immediately [in the background](../../plugin_essentials/running_steps.md#background-execution). When done, a **Segmentation Finished** dialog offers to open the Visualization tab with the first sample. |
| **+🛒** | — | Queues the step in the [Processing Queue](../../plugin_essentials/processing_queue.md) to run later, sequentially with the rest of the pipeline. |

**Existing results.** There is no overwrite checkbox: when outputs already exist, **▶ Run …** lists them and asks what to do (see [When results already exist](../../plugin_essentials/running_steps.md#when-results-already-exist)).

**Multicolor channels.** If the dataset has multicolor cell types (`<name>_<n>_multicolor`), an overlap cleanup runs automatically after APOC and Pixel Classifier batches. ConvPaint does not need it — its classes are mutually exclusive by construction.

```{note}
Cellpose and Cellpose-SAM have their own Run / **+🛒** controls, and Import uses Convert buttons instead — see their pages.
```

## Outputs

Every method writes its instance labels to the same canonical path:

```
<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr
```

The pixel classifiers also write the binary foreground mask `<sample>_<cell_type>_mask.zarr`, and — when a dead channel is declared — the death mask `<sample>_mask_dead.zarr`, in the same folder.

Training images, saved labels, previews and trained models of the pixel classifiers (APOC, ConvPaint, Pixel Classifier) are written under `<output_dir>/images/PixelClassification/`. Cellpose models are *not* stored under the output directory; they are loaded from wherever you point the **Browse…** field.

See the [Output Directory & File Layout](../../plugin_essentials/output_layout.md) page for the full canonical layout.

```{toctree}
:hidden:
:maxdepth: 1

apoc
convpaint
pixel_classifier
cellpose
cellpose_sam
import
troubleshooting
```

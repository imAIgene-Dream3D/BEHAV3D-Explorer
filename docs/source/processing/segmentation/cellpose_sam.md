# ✨ Cellpose-SAM (zero-shot)

Cellpose-SAM is the **foundation-model instance segmenter** in BEHAV3D EXPLORER. Like classic [Cellpose](cellpose.md) it outputs cell-level instance labels directly from raw intensities (no pixel labeling, no watershed post-processing), but it is built on the newer **Cellpose-SAM** model ([cellpose v4](https://github.com/MouseLand/cellpose)) and is **zero-shot** — it segments a wide range of cell shapes with *no training and no annotation*.

For method choice across all six segmentation options (including when Cellpose-SAM is the recommended choice versus APOC / ConvPaint / classic Cellpose), see the [segmentation overview](./index.md#how-to-pick-a-method).

```{important}
Cellpose-SAM segments **one population at a time**, from the channel(s) you select for it. T cells and organoids cannot be predicted in a single pass — run the tool once per population, or tick **Run all cell types in one batch** to loop through every population automatically.
```

## Section 1 · Cellpose-SAM Environment (one-time setup)

Cellpose-SAM needs **cellpose v4**, which cannot share a Python environment with the **cellpose v3** that the rest of BEHAV3D EXPLORER (and your classic Cellpose models) uses. BEHAV3D EXPLORER therefore installs cellpose v4 in a small **side ("sidecar") environment**.

| Control | Effect |
|---|---|
| Status line | **✅ Ready — cellpose …, torch … (*GPU*)** once set up; otherwise a ⚠️ message (e.g. *Cellpose-SAM environment not set up yet.*). **👁 Preview segmentation** and **▶ Run Cellpose-SAM** stay disabled until it is ready. |
| **Set up Cellpose-SAM environment** | Creates the sidecar environment in the background (about a minute; needs an internet connection). **Once per BEHAV3D EXPLORER Python environment** (redo it after reinstalling BEHAV3D EXPLORER into a new environment). It reuses your existing PyTorch install, so it only adds a few MB, and leaves your cellpose v3 install and models untouched. |

```{note}
The first time you use a model (**cpsam** or **cpsam_v2**), its weights (~1.15 GB) are downloaded and stored for later runs, so the first preview or run needs an internet connection and takes longer.
```

## Section 2 · What to Segment

| Control | Default | Effect |
|---|---|---|
| **Cell type:** | first population | The population to segment. Determines the output file and which channel selection and size filter are used. |
| **Channel Selection** | see below | One checkbox per raw channel (**Channel 0**, **Channel 1**, …): tick the channel(s) that show the selected population. Each population has its own selection, used for all samples. At most **3** channels are used (if you tick more, only the first three are). |
| **Run all cell types in one batch** | OFF | Previews and segments every population in one go, each with its own channel selection and size filter. Greys out **Cell type:**. |
| **Process all timepoints** | ON | Segments the whole movie. Uncheck to enable **From:** / **To:**. |
| **From:** / **To:** | `0` / `0` | Inclusive, 0-based timepoint range. Timepoints outside the range keep what they already contain — a range run never clears other timepoints. |

The channel selection is independent of the classic Cellpose page. The first time, each population is pre-ticked with the channel it has in the classic Cellpose **Same for all** mapping, if you saved one; otherwise nothing is ticked. A population with no channel ticked cannot be run or queued.

## Section 3 · Device

| Control | Default | Effect |
|---|---|---|
| **GPU Device:** | auto | CUDA GPU to run on. |
| **Force CPU-only processing** | OFF | Hides all GPUs from Cellpose-SAM and greys out **GPU Device:**. Much slower for 3-D volumes — only use it when the GPU is busy or unavailable. |

## Section 4 · Segmentation Settings

The **Units:** switch at the top (px ⇄ µm, **µm** by default when the metadata has pixel sizes) applies to every size-like field on this page: **Diameter**, **flow3D smooth**, **Sharpen radius**, **Smooth radius**, **Tile norm blocksize** and the Section 5 **Min size** / **Max size**. See [Units switches](../../plugin_essentials/running_steps.md#units-switches).

| Parameter | Default | Range | What it does |
|---|---|---|---|
| **Model:** | `cpsam` | `cpsam`, `cpsam_v2` | `cpsam` is the original Cellpose-SAM model (April 2025), `cpsam_v2` a newer version (June 2026). Both are zero-shot; switch freely to compare. |
| **Diameter (0=auto):** | `0` (no rescaling) | ≥ 0 | Expected object diameter. `0` means no rescaling: Cellpose-SAM does not estimate a diameter, it segments the image at its native resolution. Leave at `0`: Cellpose-SAM was trained on diameters of 7.5–120 px (mean 30 px) and is robust to size. Set it only if your objects are far outside that range. |
| **Flow threshold:** | `0.4` | 0–3 | Maximum flow error for a mask to be kept. **Raise** to detect more (possibly poorly shaped) objects; **lower** to keep only clean shapes. Ignored when 3-D segmentation is on: the label then reads **Flow threshold (ignored in 3D):** and the field is greyed out. |
| **Cell prob threshold:** | `0.0` | −6 to 6 | Pixels above this value are used to build masks. **Lower** to detect more / larger objects; **raise** to drop dim or uncertain detections. |
| **3D segmentation (do_3D)** | ON | — | **On:** true 3-D segmentation — most accurate, but much slower (often several times slower, depending on your volume size). **Off:** each Z slice is segmented in 2-D and slices are joined by overlap (see **Stitch threshold:**). |
| **Stitch threshold:** | `0.0` | 0–1 | Overlap needed between masks on neighbouring Z slices to join them into one object. Only editable when 3-D segmentation is off. |
| **Batch size:** | `8` | 1–128 | Number of image patches processed together on the GPU. **Lower it first if you get a CUDA out-of-memory error**; it does not change the result. |
| **Remove flat (single-slice) segments** | ON | — | Removes objects that occupy a single Z, Y or X slice — typically small fragments on the first/last Z slice, most common in 2-D + stitch mode. Turn off only if your objects really occupy one slice. |

```{tip}
**3-D or 2-D + stitch?** Keep **3D segmentation (do_3D)** on for the most accurate result. Switch it off for speed, or when the data are so anisotropic that the side views (ZY/XZ) are unusable — then tune **Stitch threshold:** instead.
```

### Advanced (normalization, filters)

Collapsed by default. All values default to Cellpose's own recommendation or "off" (except **Max size fraction:**), so you can **leave them untouched for routine runs**.

| Parameter | Default | What it does |
|---|---|---|
| **Norm percentiles:** low / high | `1` / `99` | Intensity percentiles rescaled to 0 and 1 before segmentation. |
| **Normalize across the whole Z stack (norm3D)** | ON | On: one normalization for the whole stack. Off: each Z slice is normalized separately — use it if brightness drifts a lot between slices. Only takes effect when **3D segmentation (do_3D)** is off and **Stitch threshold:** is above 0; with 3-D segmentation on, cellpose always normalizes the whole stack. |
| **niter dynamics:** | `0` (auto) | Iterations of the flow dynamics; `0` = automatic. Long, thin objects may need more (e.g. ~2000). |
| **flow3D smooth:** | `0` (off) | Smooths the 3-D flows before the dynamics; increase it if you see ring artifacts. |
| **Sharpen radius:** | `0` (off) | Sharpening before segmentation, for dim, blurry images (about 1/4–1/8 of the object diameter). |
| **Smooth radius:** | `0` (off) | Smoothing before segmentation, for noisy images. |
| **Tile norm blocksize:** | `0` (off) | Normalizes in tiles of this size instead of over the whole frame; brightens dark areas of unevenly illuminated images. |
| **Tile norm smooth3D:** | `0` | Smooths the tiled normalization across Z (`0` = Cellpose's own default). Only used when **Tile norm blocksize:** > 0. |
| **Max size fraction:** | `1.0` | Masks larger than this fraction of the image are deleted. Cellpose's own default (0.4) silently removes large organoids, so BEHAV3D EXPLORER keeps everything and relies on the Section 5 size filter. |
| **CPU threads (0=all):** | `0` | Caps the CPU threads used for post-processing and saving. |
| **Cooldown between frames (s):** | `0` | Pause after each timepoint (0–120 s). |

```{warning}
**Laptops switching off mid-run.** A Cellpose-SAM run loads the GPU *and* every CPU core at once, which can draw more power than some laptops can deliver, causing an abrupt shutdown. If that happens, cap **CPU threads (0=all):** at about a quarter of your cores, add about **5 s** of cooldown and lower **Batch size:**. Progress is saved after every timepoint: run again and choose **Skip & Resume** to continue where it stopped (see [When results already exist](../../plugin_essentials/running_steps.md#when-results-already-exist)). Resuming is refused if you changed the segmentation settings in between — then choose **Overwrite All**.
```

## Section 5 · Size Filter (preview before batch)

A **post-processing** step that **removes** objects outside a volume range. A removed object is deleted, not merged into a neighbour — unless it lies completely inside kept objects, in which case its voxels go to the nearest kept object so no hole is left. **Min size:** and **Max size:** are saved separately for each population.

| Control | Default | Effect |
|---|---|---|
| **Sample:** / **t:** | first sample / `0` | The sample and timepoint used by **👁 Preview segmentation**. Pick a representative one. |
| **Min size:** | `0` (no bound) | Objects smaller than this volume are removed (debris, fragments). |
| **Max size:** | `0` (no bound) | Objects larger than this volume are removed. |
| **👁 Preview segmentation** | — | Segments only that timepoint, **without** the size filter. The viewer is cleared and shows the raw channels of that timepoint plus a `… · preview` labels layer per population. The log lists the object count and the smallest / median / largest volume (in voxels). |

After a preview, changing **Min size:** or **Max size:** re-filters the preview **instantly** (no new Cellpose-SAM run), and the line under the fields reports how many objects are kept and removed. Use it to pick thresholds that remove debris but keep real cells.

## Run

| Control | Effect |
|---|---|
| **▶ Run Cellpose-SAM** | Segments the selected population (or all of them) in every sample with the settings of Sections 2–5, which are saved automatically. Runs [in the background](../../plugin_essentials/running_steps.md#background-execution); when segmentations already exist you choose what to do (see [When results already exist](../../plugin_essentials/running_steps.md#when-results-already-exist)). Samples whose raw image has not been converted to Zarr are skipped. Result paths are written into metadata.csv. |
| **+🛒** | Adds **🦠 Segmentation — Cellpose-SAM** to the [Processing Queue](../../plugin_essentials/processing_queue.md). The step remembers the population (or *all cell types*), **Model:**, the device settings, **CPU threads** and **Cooldown**; the timepoint range, channel selection, Section 4 settings and size filter are read when the step starts. |

## Outputs

| File | Path |
|---|---|
| Per-population instance labels | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.zarr` |
| Progress record (used to resume an interrupted run) | `<output_dir>/images/<sample>/<sample>_<cell_type>_segments.progress.json` |

Cellpose-SAM writes to the **same path** as every other method, so tracking and all later steps use it in the same way. There is no model file to manage.

## Tuning the result

Because Cellpose-SAM is zero-shot, tuning is mostly choosing **3-D vs 2-D + stitch**, the two thresholds and the channels, then filtering by size:

- **Touching cells fused into one label** → Cellpose-SAM has no watershed splitting. Try raising **Cell prob threshold:** slightly, or set a **Diameter** if your objects are far from the ~30 px the model expects (`0` means no rescaling, not an estimated diameter). If it persists, split labels by hand after tracking ([Manual Editing](../tracking/manual_editing.md)), or use a watershed-based method ([APOC](apoc.md) / [ConvPaint](convpaint.md)) for that population.
- **Many small or flat fragments** → keep **Remove flat (single-slice) segments** on and set a **Min size:** (preview first to read real object sizes).
- **Large organoids disappearing** → check that **Max size fraction:** is `1.0` and **Max size:** is `0` or large enough.
- **CUDA out of memory** → lower **Batch size:** first.
- **Objects over-split in Z** → check the pixel sizes in your metadata (they set the Z scaling); with very anisotropic data, try 2-D + stitch.
- **Other populations detected too** → check **Channel Selection**: tick only the channel(s) of the population you segment.

## Things this page does **not** claim

- Specific runtimes — they depend strongly on your GPU, volume size, number of channels and 3-D vs 2-D + stitch. Profile your own data.
- That `cpsam_v2` always beats `cpsam` — compare them on your data.

## See also

- [Segmentation overview](./index.md) — how Cellpose-SAM fits with the other methods.
- [Cellpose (classic)](cellpose.md) — the trainable v3 model, and the **dead mask (Otsu)** step.
- [Cellpose project on GitHub](https://github.com/MouseLand/cellpose) — upstream documentation of the model and its settings.
- [Processing Queue](../../plugin_essentials/processing_queue.md) — batching segmentation steps.

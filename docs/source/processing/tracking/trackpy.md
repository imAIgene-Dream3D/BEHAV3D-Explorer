# 🎯 TrackPy

[TrackPy](http://soft-matter.github.io/trackpy/) is a particle-tracking library originally written for colloidal-particle tracking (Crocker–Grier algorithm). For each group of nearby candidates it picks the links between consecutive frames with the **smallest total (squared) displacement**. In BEHAV3D EXPLORER it is a **lighter alternative to LapTrack** when cells are well separated and you need neither a separate gap-closing distance nor merge / split detection.

```{note}
**Not pre-selected for any cell type.** For objects that do not overlap between frames this
documentation treats **[btrack](btrack.md)** as the routine tracker and TrackPy as an
alternative; for objects that do, use an
[overlap-based method](index.md#overlap-based-tracking-methods-propagation). Choose from your
data and check the result.
```

## When to use it

It only works when frame-to-frame links are **unambiguous**: cells **well-separated relative to how far they move per frame**. Minimal-displacement linking goes wrong once cells move about as far between frames as the distance to their neighbours, so the limit is crowding-vs-speed, not slow motion alone. It has no motion prediction and no merge / split handling.

```{warning}
"Lighter" holds only while links stay unambiguous. On crowded data the adaptive search keeps
shrinking the radius (and may fall back to a slower strategy or stop with an error), so it can
get **slower**, not faster — use [btrack](btrack.md) there.
```

## What runs under the hood

For each sample BEHAV3D EXPLORER measures the centroid of every segment at every timepoint in **physical units** (µm, from the metadata pixel size, z included) and links them with TrackPy.

## Parameters

A **Distance units:** switch (**px** ⇄ **µm**) sits at the top of the panel and is on **µm** by default — TrackPy links on physical coordinates (see [Distance units](index.md#distance-units)). **Search range** follows the switch; **Adaptive stop** is always in µm.

| Field | Default | Range | Effect |
|---|---|---|---|
| **Search range** | `31` | 1–999999 | Maximum centroid distance between consecutive frames considered for linking. Set roughly to the displacement of the fastest cell between frames. |
| **Memory (frames)** | `2` | 0–100 | Number of frames a cell can disappear and still be reconnected to its previous track — similar to LapTrack's **Gap close max frames** (Memory = N allows N missing frames; Gap close max frames = N allows N − 1). |
| **Adaptive stop** | `10.0` | 0.0–100.0 | When a group of nearby cells is too large to solve (an oversized "subnet"), TrackPy retries that group with a smaller search range (see **Adaptive step**). This is the smallest value the search range may shrink to before TrackPy gives up and linking is retried with a slower strategy. Lower = more retries (more robust, slower on dense data); higher = gives up sooner. |
| **Adaptive step** | `0.95` | 0.01–1.0 | Multiplier applied to the search range at each retry. Close to 1.0 → slow, gentle reduction; close to 0.5 → aggressive. |

## Step-by-step in the napari plugin

1. **Open the Tracking tab** and pick the sub-tab for the cell type you want to track.
2. From the **Method:** dropdown, pick **TrackPy**.
3. **Estimate the fastest single-frame displacement in your data** — load a sample in Visualization, scrub two consecutive frames and measure the fastest cell's step.
4. **Set Search range** ≈ that displacement, in the unit of the **Distance units:** switch (µm by default).
5. **Set Memory (frames)** = how many consecutive missing frames a cell can have and still be reconnected.
6. **Leave Adaptive stop and Adaptive step at their defaults** unless TrackPy is very slow on a dense sample.
7. (Optional) **Use the Apply buttons** to copy these settings to other sub-tabs.
8. **Click *Run … Tracking*** (the green button at the bottom of the sub-tab, named after the cell type). For batch or queued runs, see [Generic workflow](index.md#generic-workflow-per-sub-tab).
9. **Inspect the result** in the Visualization tab; if you see fragmented tracks, ID swaps, or subnet warnings in the **Log**, see [Tuning the parameters](#tuning-the-parameters).

## Tuning the parameters

**Tracks are fragmented (one cell has several short TrackIDs over time)**

- Cause: the cell moves further between frames than **Search range** allows.
- Fix: **raise Search range** to roughly the fastest cell's single-frame displacement.

**Track IDs swap between neighbouring cells**

- Cause: **Search range** is so large that two close cells become indistinguishable nearest neighbours.
- Fix: **lower Search range**. If that fragments other tracks, the dataset is genuinely ambiguous — consider btrack, whose motion model and **Use visual features** add cues besides position.

**A cell flashes off for one or two frames and comes back as a new TrackID**

- Cause: **Memory (frames)** is too low.
- Fix: **raise Memory (frames)** to at least the longest expected disappearance (e.g. 2 → 5). Large values invite spurious bridges between unrelated cells.

**"Numba linking hit subnet limit … Falling back to recursive (slower) linking strategy" warning in the Log**

- Cause: even after shrinking the search range down to **Adaptive stop**, a group of nearby cells was too large, so linking is retried with TrackPy's slower strategy. If that retry fails too, tracking stops with an error.
- Fix: **lower Search range** to the smallest value that still gives continuous tracks for the fastest cells. If you can't, switch to LapTrack or btrack.

A Log line saying TrackPy's subnet size limit was raised (from 30 to 300) appears for every sample; you can ignore it.

**TrackPy is extremely slow on a particular sample**

- Cause: adaptive search keeps shrinking the search range because subnets are too large.
- Fix: **lower Search range**, or **lower Adaptive step** (e.g. 0.95 → 0.7 or 0.5) so each retry shrinks the range more aggressively, at the cost of occasionally slightly worse links. Raising **Adaptive stop** makes TrackPy give up sooner, which leads to the slower fallback described above rather than a speed-up.

**Cells fuse or divide in your dataset**

- TrackPy has no merge / split detection. Switch to LapTrack (with **Merging cost** / **Splitting cost** > 0) or [btrack](btrack.md).

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

## Good practices & tips

- **Leave Adaptive stop and Adaptive step at their defaults** unless tracking is failing or very slow.
- **Set Search range = the maximum reasonable single-frame displacement** — too small = fragmented tracks; too large = ID swaps between neighbouring cells.
- **Memory (frames) recovers brief segmentation gaps.** Start at 2; raise it if TrackIDs flip when a cell briefly disappears.
- **No division handling.** For dividing cells, see [btrack](btrack.md) (**P_branch**) or LapTrack (**Splitting cost**).
- **TrackPy vs LapTrack:** neither is reliably faster or more accurate; it depends on your data. If you are unsure, run both on one sample and compare the tracks.

## See also

- [TrackPy documentation](http://soft-matter.github.io/trackpy/) — upstream library, including a detailed explanation of the adaptive-search algorithm.
- [LapTrack](lap.md) for a separate gap-closing distance and merge/split detection.
- [btrack](btrack.md) for a Bayesian motion model.

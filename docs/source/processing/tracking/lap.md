# 🔗 LapTrack

LapTrack — Linear Assignment Problem tracking via the [laptrack](https://github.com/yfukai/laptrack) library — works in two steps (after Jaqaman et al., 2008): it links the detections of each pair of consecutive frames by the optimal assignment on centroid distances, then solves a second assignment over all resulting track pieces to close gaps and, optionally, detect merges and splits.

```{note}
**Pre-selected, but not the routine choice.** LapTrack is pre-selected for 🟡 *Other* cell
types in a fresh project; like every pre-selection, this is only a starting point. For objects
that do not overlap between frames this documentation treats **[btrack](btrack.md)** as the
routine tracker and LapTrack as an alternative; for objects that do, use an
[overlap-based method](index.md#overlap-based-tracking-methods-propagation). Choose from your
data and check the result.
```

## When to use it

A centroid-distance tracker without a motion model: it links on positions only, whereas [btrack](btrack.md) predicts each cell's next position from its recent movement. Compared with [TrackPy](trackpy.md), which links frame to frame in a similar way, LapTrack adds a separate **gap-closing** step with its own distance limit and optional **merge / split** detection.

## What runs under the hood

For each sample BEHAV3D EXPLORER measures the centroid of every segment at every timepoint in **physical units** (µm, from the metadata pixel size) and LapTrack links them. Linking uses the full **3-D** centroid distance (z scaled by `pixel_distance_z`).

When a merge or split is detected, the track(s) end at the event and the object(s) after it continue under new TrackIDs; the parent–daughter links are not written to the tracks CSV.

## Parameters

A **Distance units:** switch (**px** ⇄ **µm**) sits at the top of the panel and is on **µm** by default (see [Distance units](index.md#distance-units)). All the costs below are distances in the selected unit.

```{note}
Enter a plain **distance**. The [laptrack documentation](https://laptrack.readthedocs.io/)
describes its cost cutoffs as *squared* distances; the plugin does the squaring for you, so
don't square the value yourself.
```

| Field | Default | Range | Effect |
|---|---|---|---|
| **Track cost** | `45` | 1–999999 | Maximum distance a cell can move between two **consecutive** frames to be linked as the same track. |
| **Gap close cost** | `60` | 1–999999 | Maximum distance allowed when reconnecting track pieces in the gap-closing step. Should be ≥ Track cost. This step also considers a piece that restarts in the very next frame, so with gap closing on, a link between consecutive frames can be made up to this distance. |
| **Gap close max frames** | `3` | 0–100 | Largest frame distance bridged by gap closing: a value of N reconnects across at most N − 1 *consecutive* missing frames (default `3` → 2 missing frames); `0` switches gap closing off. |
| **Merging cost** | `0` | 0–999999 | Maximum distance for detecting merge events (two tracks → one). `0` disables merging detection. |
| **Splitting cost** | `0` | 0–999999 | Maximum distance for detecting split events (one track → two). `0` disables splitting detection. |

## Step-by-step in the napari plugin

1. **Open the Tracking tab** and pick the **sub-tab** for the cell type you want to track.
2. From the **Method:** dropdown, pick **LapTrack**.
3. **Estimate the fastest single-frame displacement in your data**, in the unit of the **Distance units:** switch (µm by default): open the Visualization tab, load one sample, scrub through two consecutive frames and measure how far the fastest cell moves.
4. **Set Track cost ≈ that displacement**, with a small margin (e.g. fastest cell moves ~30 µm → 35–45). The default is only a starting point (see [Tuning the parameters](#tuning-the-parameters)).
5. **Set Gap close cost ≈ Track cost × 1.2–2.** A cell that disappears for a few frames may have moved further than in a single frame.
6. **Set Gap close max frames** = how many consecutive missing frames you are willing to bridge, plus 1.
7. **Leave Merging cost and Splitting cost at 0** unless you actually expect cell fusion or division — both are very disruptive when triggered by mistake.
8. (Optional) **Use the Apply buttons** (**Apply to all *Category*** / **Apply to all**) to copy these settings to cell types with similar dynamics.
9. **Click *Run … Tracking*** (the green button at the bottom of the sub-tab, named after the cell type). For batch or queued runs, see [Generic workflow](index.md#generic-workflow-per-sub-tab).
10. **Inspect the result** in the Visualization tab. If you see fragmented tracks, ID swaps, or cells that should have been bridged across a gap but weren't, see [Tuning the parameters](#tuning-the-parameters).

## Tuning the parameters

After looking at the result in the Visualization tab, identify which failure mode you're seeing and tune accordingly:

**Tracks are fragmented (one biological cell has several short TrackIDs over time)**

- Cause: the cell moves further between consecutive frames than `Track cost` allows.
- Fix: **raise `Track cost`** (e.g. 45 → 60). Re-run and re-inspect.

**Track IDs swap between neighbouring cells**

- Cause: `Track cost` (or `Gap close cost`, which also links consecutive frames) is large enough that two nearby cells' candidate links cross — LapTrack picks the wrong assignment.
- Fix: **lower `Track cost`** (and `Gap close cost` if it is much larger). If that then fragments tracks, the dataset is genuinely ambiguous; try btrack, whose motion model and **Use visual features** add cues besides position.

**A cell disappears for a few frames and then comes back as a new TrackID**

- Cause: gap closing didn't bridge the disappearance.
- Fix: **raise `Gap close cost`** so the cost of bridging the gap is high enough that LapTrack picks it over starting a new track. Also check `Gap close max frames` is greater than the number of missing frames.

**False bridges: two unrelated cells are linked across a gap**

- Cause: `Gap close cost` is too generous, or `Gap close max frames` is too high.
- Fix: **lower one of them**. Typically the right move is to lower `Gap close max frames` first (it's easier to interpret).

**One cell becomes two (split detected where there shouldn't be one)** / **Two unrelated cells are merged**

- Cause: `Splitting cost` or `Merging cost` is > 0 and is being triggered by false positives.
- Fix: **set the offending cost to 0** (disabled). Only re-enable if you actually see biological splits / merges and the cost was previously too low to catch them.

**LapTrack takes too long on dense data**

- Cause: LapTrack computes the distance between every pair of detections in consecutive frames, so the cost grows roughly with the square of the number of detections per frame.
- Fix: try [btrack](btrack.md) with **Update method** = `APPROXIMATE`, which is meant for very dense data. TrackPy is not a remedy here: it also slows down on crowded data.

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

## Good practices & tips

- **The defaults are a generic starting point**, not a tuned recommendation — look at the result and tune for the failure modes above.
- **Too small Track cost = fragmented tracks; too large = false links across cells.**
- **Keep merging / splitting at 0 unless you actually see fusion / division events** — every detected event ends the track and starts new TrackIDs, so false events fragment your tracks.
- **LapTrack gets expensive on very dense data** (thousands of cells per timepoint); consider btrack (`APPROXIMATE` update) there. Runtime grows with cells per frame and number of frames, so time one sample before running the whole batch.

## See also

- [laptrack documentation](https://laptrack.readthedocs.io/) — upstream library, including a precise definition of the cost function.
- [TrackPy](trackpy.md) — lighter alternative without a separate gap-closing step or merge/split detection.
- [btrack](btrack.md) — Bayesian alternative with motion model + global hypothesis solver.

# 🧠 btrack (Bayesian)

[btrack](https://btrack.readthedocs.io/) is a Bayesian multi-object tracker. Its
**motion model** is *"used to make forward predictions about the location of objects
using historical information and estimates of the error in the measurements"*, and
its **hypothesis model** is *"used by the global optimizer to build the final set of
tracks"* ([btrack documentation](https://btrack.readthedocs.io/en/stable/user_guide/configuration.html)).

In BEHAV3D EXPLORER, btrack runs in **two steps** on a cell type's per-timepoint
segmentation. **Step 1** links detections from frame to frame with a Kalman filter.
**Step 2**, the global hypothesis optimizer, takes the track pieces left by Step 1 and
decides across the whole movie which pieces to join. Step 2 is **on by default**: it
roughly halves the number of broken tracks. Choose btrack when objects lose frame-to-frame
overlap, regardless of their biological identity (see [Tracking overview](index.md)). It
is the method pre-selected for immune (🔵) cell types in a fresh project.

All distances are in **µm** and all variances in **µm²**, because btrack links on physical
coordinates. The **Distance units:** switch at the top of the panel only changes how
values are displayed (see [Distance units](index.md#distance-units)).

## Workflow at a glance

1. **Open the Tracking tab** and pick the sub-tab for the cell type you want to track.
2. From the **Method:** dropdown, pick **btrack (Bayesian)**.
3. Choose a **Preset**, or click **Suggest…** and let BEHAV3D pick one from your data. Selecting a preset fills every field below.
4. Optionally adjust the fields ([Step 1](#step-1--frame-to-frame-linking), [Step 2](#step-2--global-optimizer), [Advanced](#advanced)). Edits never change the preset file; the preset name shows *(modified)*.
5. **Click *Run … Tracking*** (the green button at the bottom of the sub-tab, named after the cell type) and inspect the result in the Visualization tab.
6. [Iterate](#tuning-the-parameters). Use **Reset to preset** to go back, or **Export JSON…** to save your values as a new preset file. For batch or queued runs, see [Generic workflow](index.md#generic-workflow-per-sub-tab).

When you reopen a project, each field shows the value used in the previous run of that
experiment (if the preset is unchanged); otherwise it shows the preset value.

## Presets

| Preset | Use when |
|---|---|
| **T cells - tuned** (default) | Motile cells imaged every ~2–6 min with a reasonable segmentation. |
| **T cells - full optimum** | Same situations; also uses settings that are not shown in the panel. |
| **T cells - dense** | Crowded fields, or segmentation that often misses or splits cells. |
| **T cells - long interval** | Frame interval around 8–10 min, or fast cells. |
| **T cells - noisy segmentation** | Tuned for noisy centroids; kept for reference. |
| **btrack default - cell / particle (untuned)** | btrack's own example settings; kept so old projects still load. |
| **Custom JSON…** | Your own btrack configuration file; the fields show its values. |

Frame intervals above about 10 min can reduce tracking accuracy whatever parameters you
choose, because cells move too far between frames to be matched reliably.

### Suggest…

**Suggest…** opens a window that measures your segmentation (no tracking needed):

- **Motion scale (D)**: how far cells travel between two frames. For every cell it takes the distance to the closest cell in the next frame and reports the value that 95 % of cells stay below.
- **Crowding (N)**: how close neighbouring cells sit within one frame; the nearest-neighbour distance that only 5 % of cells fall below.
- **Ambiguity (D / N)**: motion scale divided by crowding. Above about 0.9 a cell could plausibly be matched to a neighbour, so the more careful presets are chosen.

Then answer two questions: does the segmentation miss or split cells (no / some / often),
and do the cells divide. The window shows the recommended preset with the reason and the
fields it will change (search radius and gap distance are raised to at least 3× and 4× the
motion scale, and gap frames to 10 when the segmentation is not perfect). **Apply to
fields** fills the panel.

## Step 1 — Frame-to-frame linking

| Control | Default | What it does |
|---|---|---|
| **Max search radius** | 125 µm | Longest single-frame jump that may be linked. It only clips links; it does not choose between candidates. Too small a value loses cells; any value comfortably above the fastest step behaves the same. |
| **Assignment tolerance** | 8 µm | Half-width of the box over which the predicted position is compared with each detection (`accuracy`). **Must be between 0.1 and 10 µm**; btrack crashes above 10. Below about 5 µm almost nothing is linked with the tuned presets. |
| **New-track position uncertainty** | 40 µm² | How uncertain the position of a newly started track is. Too large: new tracks never find their second detection. Too small: tracks cannot follow a moving cell. |
| **Lost-track probability** | 0.1 | Prior probability that a track has no detection in the next frame. Higher: tracks break more readily (Step 2 can rejoin them). Lower: tracks hold on and may grab a wrong neighbour. |
| **Workers** | 1 | CPU cores used to measure the objects and write the result (1 to cores − 1). The tracker itself runs on one core. |

These three settings work together: the tolerance is judged against the predicted spread,
which comes from the new-track uncertainty and the measurement noise. Change them one at a
time.

## Step 2 — Global optimizer

| Control | Default | What it does |
|---|---|---|
| **Enable global track optimization** | on | Joins broken tracks, bridges gaps and removes spurious fragments. Recommended. |
| **Gap-bridging distance** | 40 µm | Track pieces whose ends are farther apart than this are never joined. About 4–6× the typical one-frame step. |
| **Gap-bridging frames** | 5 | Longest disappearance that can be bridged. Use 10 if the segmentation misses cells. |
| **Link tolerance** | 450 µm | How quickly the probability of joining two pieces falls with distance. Lower: only close pieces are joined (purer, more fragments). Higher: more gaps bridged, more wrong joins. |

Link tolerance and the track start/end penalty work as a pair: the optimizer compares
*joining* two pieces against *ending one track and starting another*. Only their ratio
matters, which is why presets with a strict link tolerance also use a very small penalty.

## Advanced

| Control | Default | What it does |
|---|---|---|
| **Optimizer time limit** | 60 s (600 s in **T cells - dense**) | Stops the Step 2 solver after this long; the best solution found so far is used. Dense fields need more solver time, and a solver that is cut short links less accurately. |
| **Measurement noise** | 5 µm² | How much the filter trusts each centroid. Too small: it follows segmentation jitter. Too large: sluggish prediction. Raise it for noisy centroids (split or merged objects). |
| **Track start/end penalty** | 0.5 µm | How easily a track may begin or end away from the image border. Higher: fewer wrong joins, more fragments. Lower: more complete tracks, more risk of wrong joins. |
| **Allow divisions** | off | Adds the `P_branch` hypothesis for dividing cells. The parent track ends at the division and each daughter starts a new TrackID; the parent–daughter link is not written to the tracks CSV. Leave off for cells that do not divide. |

## Tuning the parameters

After inspecting the result in the Visualization tab, identify the failure mode:

*Tracks are broken into many short pieces*
- Make sure **Enable global track optimization** is on.
- Raise **Gap-bridging frames** (up to 10) and **Gap-bridging distance**, or raise **Link tolerance**.

*Two cells swap identity, or tracks jump between neighbours*
- Lower **Link tolerance** or **Gap-bridging distance**, or raise **Track start/end penalty** (Advanced).
- In a crowded field, try the **T cells - dense** preset.

*Almost nothing is linked*
- **Assignment tolerance** is too small (use 6–10 µm), or **New-track position uncertainty** is too large.

*Tracking takes minutes or never ends*
- Tighten **Gap-bridging distance** and keep **Allow divisions** off on dense data.
- Lowering **Optimizer time limit** speeds up the run but can cost linking accuracy, because the solver stops before it finds the best solution.

*Spurious divisions appear*
- Untick **Allow divisions**.

*A custom JSON is rejected before tracking starts*
- BEHAV3D checks `accuracy` (0–10), `max_lost` (0–10) and `prob_not_assign` (0–1) before running; the message names the offending value.

## Python API

`run_btracking` and `btrack_image` take the same parameters as keyword arguments
(`assignment_tolerance`, `new_track_position_variance`, `lost_track_probability`,
`measurement_noise`, `link_tolerance`, `track_start_end_penalty`, `allow_divisions`,
`optimizer_time_limit_s`, plus `max_search_radius`, `dist_thresh`, `time_thresh`,
`use_optimize`). Anything left as `None` keeps the value stored in the preset. The old
`update_method`, `step_size`, `hypotheses` and `use_visual_features` arguments are still
accepted for existing scripts; update method and step size do not change the tracks, and
visual features are deprecated.

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

## Good practices & tips

- **Start from a preset** rather than from scratch, and change one field at a time.
- **Keep the optimizer on.** A poorly tuned Step 1 can be partly repaired by Step 2, but not the other way round.
- **Fix the segmentation first** if cells are often missed or split: that costs more accuracy than any parameter can recover.
- **Workers are capped at one less than your CPU core count**, so the computer stays responsive.

## See also

- [btrack documentation](https://btrack.readthedocs.io/) — upstream library, including precise definitions of every hypothesis and motion-model parameter.
- [Fragmentation Propagation](fragmentation_tracking.md) — overlap-based method for objects that stay put and change only by fragmenting.
- [LapTrack](lap.md) and [TrackPy](trackpy.md) — alternative centroid-based trackers.

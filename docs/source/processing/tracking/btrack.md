# 🧠 btrack (Bayesian)

[btrack](https://btrack.readthedocs.io/) is a Bayesian multi-object tracker. Its
**motion model** is *"used to make forward predictions about the location of objects
using historical information and estimates of the error in the measurements"*, and
its **hypothesis model** is *"used by the global optimizer to build the final set of
tracks"* ([btrack documentation](https://btrack.readthedocs.io/en/stable/user_guide/configuration.html)).

In BEHAV3D EXPLORER, btrack runs in **two steps** on a cell type's per-timepoint
segmentation: **Step 1**, the Kalman-filter tracker, always runs; **Step 2**, the global
hypothesis optimiser, is optional and should only be switched on once Step 1 looks
correct. Choose btrack when objects lose frame-to-frame overlap, regardless of their
biological identity (see [Tracking overview](index.md)). It is the method pre-selected for
immune (🔵) cell types in a fresh project.

## The two steps

- **Step 1 — Kalman Filter Tracking** — links detections between frames using the motion
  model.
- **Step 2 — Global Hypothesis Optimizer** — resolves track merges, splits, terminations
  and false positives across the whole track set. It only runs when **Enable global
  track optimization** is ticked (off by default); until then its controls are greyed out.

Distances are entered in the unit of the **Distance units:** switch at the top of the
panel (**µm** by default; see [Distance units](index.md#distance-units)).

### Step 1 — Kalman Filter Tracking parameters

| Control | Default | Range | What it does |
|---|---|---|---|
| **Config preset** | `Cell (bundled)` | `Cell (bundled)` / `Particle (bundled)` / `Custom JSON…` | Motion and hypothesis model. **Cell** is tuned for biological cells, **Particle** for small particles. **Custom JSON…** activates the path field and **Browse** button below it to load your own btrack configuration file. |
| **Max search radius** | `100` | 1–999999 | Maximum distance to search for the same object in the next frame. Derive it from the fastest measured one-frame displacement plus a modest margin — the default is not a recommendation. |
| **Update method** | `EXACT` | `EXACT` / `APPROXIMATE` | **EXACT** = full Bayesian belief matrix (accurate, slower). **APPROXIMATE** = local spatial search (faster; for very crowded frames, > 1000 cells per frame). |
| **Step size (frames)** | `100` | 1–10000 | Frames processed per tracking iteration. Lower it (e.g. 50 or 20) only if a long movie runs out of memory: each chunk then uses less RAM, at a small speed cost. |
| **Use visual features** | off | checkbox | Also feeds each object's size and shape (area, major and minor axis length) and its mean intensity in every raw-image channel to the Kalman filter as extra linking cues. Needs the raw image (`raw_image_path`) in the metadata. |
| **Workers** | `1` | 1 to (CPU cores − 1) | CPU cores used to measure the objects and write the result. The tracker itself runs on one core. |

### Step 2 — Global Hypothesis Optimizer parameters

| Control | Default | What it does |
|---|---|---|
| **Enable global track optimization** | off | Runs the optimiser after Step 1. |
| **Hypotheses** | `P_FP`, `P_init`, `P_term`, `P_link` ticked | Which explanations the optimiser may assign to a tracklet (a short track piece left by Step 1). **P_FP — False positive** is always on (greyed out). Add **P_branch**, **P_dead** or **P_merge** only when your biology requires them (definitions below). |
| **Distance threshold** | `60` | Maximum distance for generating link/branch hypotheses (in the selected distance unit). |
| **Time threshold** | `3` | Maximum frame gap for generating link hypotheses (1–999 frames). |

**Hypotheses** — checkbox label in the GUI, with the definition from the [btrack documentation](https://btrack.readthedocs.io/en/stable/user_guide/configuration.html):

| Checkbox | btrack definition |
|---|---|
| **P_FP — False positive** | "Hypothesis that a tracklet is a false positive detection." |
| **P_init — Track initialization** | "Hypothesis that a tracklet starts at the beginning of the movie or edge of the FOV." |
| **P_term — Track termination** | "Hypothesis that a tracklet ends at the end of the movie or edge of the FOV." |
| **P_link — Track linking** | "Hypothesis that two tracklets should be linked together." |
| **P_branch — Track branching** | "Hypothesis that a tracklet can split onto two daughter tracklets." |
| **P_dead — Cell death** | "Hypothesis that a tracklet terminates without leaving the FOV." |
| **P_merge — Track merging** | "Hypothesis that two tracklets merge into one tracklet." |

`P_link`, `P_branch` and `P_merge` are limited by the distance and time thresholds.

## Step-by-step in the napari plugin

1. **Open the Tracking tab** and pick the sub-tab for the cell type you want to track.
2. From the **Method:** dropdown, pick **btrack (Bayesian)**.
3. **Configure Step 1**:
   - **Config preset**: start with **Cell (bundled)**. Switch to **Particle (bundled)** only for very small objects with near-random motion.
   - **Max search radius**: measure the fastest plausible one-frame displacement, then add a 10–25 % margin.
   - **Update method**: keep **EXACT**; switch to **APPROXIMATE** only if tracking is too slow.
   - **Step size (frames)**: `100` suits most datasets; lower it if a long movie runs out of memory.
   - **Use visual features**: tick to add size, shape and intensity as linking cues (needs the raw image).
4. **Click *Run … Tracking*** (the green button at the bottom of the sub-tab, named after the cell type) and **inspect the result** in the Visualization tab.
5. **Optionally enable Step 2**: tick **Enable global track optimization**, tick only the hypotheses your data needs (`P_branch` for division, `P_init`/`P_term` for entry or exit at the field borders, `P_dead` for termination inside the field, `P_merge` for genuine fusion), and derive **Distance threshold** / **Time threshold** from the largest spatial and missing-frame gaps that should be bridged. Re-run and re-inspect.
6. **Iterate** (see [Tuning the parameters](#tuning-the-parameters)). For batch or queued runs, see [Generic workflow](index.md#generic-workflow-per-sub-tab).

## Tuning the parameters

After inspecting the result in the Visualization tab, identify the failure mode and tune accordingly.

**Step 1 — Kalman filter**

*Tracks are fragmented*
- Cause: **Max search radius** is smaller than the fastest cell's single-frame displacement.
- Fix: **raise Max search radius** until the fragmentation goes away.

*IDs jump between neighbouring cells*
- Cause: the search radius is generous and positions alone are ambiguous.
- Fix: lower **Max search radius**, or tick **Use visual features** so size, shape and intensity help tell neighbours apart.

*Tracking is very slow / runs out of memory*
- Cause: many cells per frame with **EXACT** update, or a long movie.
- Fix: switch **Update method** to **APPROXIMATE**, and/or lower **Step size (frames)** (e.g. 100 → 50).

*The Cell preset doesn't suit your data*
- Cause: objects that are unusually small or move very differently from cells.
- Fix: try **Particle (bundled)**, or load your own configuration with **Custom JSON…**.

**Step 2 — Global optimiser**

*Cell divisions are tracked as unrelated tracks*
- Fix: **tick P_branch** and check that divisions now appear as parent → two daughters.

*Spurious branches / merges appear*
- Cause: too many hypotheses are enabled.
- Fix: **untick the hypotheses you did not biologically expect**. The defaults (`P_FP, P_init, P_term, P_link`) are deliberately conservative.

*Gaps you expected to be bridged are not*
- Cause: **Time threshold** or **Distance threshold** is too small.
- Fix: set each just above the largest valid gap you measured (convert frames to time with your frame interval).

*Unrelated cells are bridged together*
- Cause: the thresholds are too generous.
- Fix: **lower one of them**, usually **Time threshold** first.

*Step 2 is very slow*
- Cause: many hypotheses combined with large thresholds.
- Fix: tighten the thresholds.

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

## Good practices & tips

- **Validate Step 1 before enabling Step 2.** The optimiser cannot fix a poorly tuned search radius.
- **Use EXACT unless tracking is too slow.**
- **P_branch is the right hypothesis for cell division.** Without it, mitosis appears as new tracks starting where the parent ended.
- **Workers are capped at one less than your CPU core count**, so the computer stays responsive.

## Things this page does **not** claim

- Specific per-sample runtimes — they depend on cells per frame, number of frames and the optimiser's hypothesis set.

## See also

- [btrack documentation](https://btrack.readthedocs.io/) — upstream library, including precise definitions of every hypothesis and motion-model parameter.
- [Fragmentation Propagation](fragmentation_tracking.md) — overlap-based method for objects that stay put and change only by fragmenting.
- [LapTrack](lap.md) and [TrackPy](trackpy.md) — alternative centroid-based trackers.

# 🧩 Fragmentation Propagation

Fragmentation Propagation is **not** a centroid-distance method. It links objects by **spatial overlap**: for each pair of consecutive frames `t → t+1`, the labels of frame `t` are propagated into the segmentation mask of frame `t+1` (watershed-based overlap tracking).

## How it works

- The labels of the **first timepoint** become the track IDs.
- At every following timepoint, each foreground piece is flooded with the ID of the previous-frame object it overlaps (a piece that overlaps several objects is divided between them). A second pass on the mask dilated by **2 pixels in XY** lets an ID also reach pieces lying up to about 4 pixels away within the same Z slice.
- Because every piece that overlaps an object's previous footprint inherits its ID, an object that **breaks into fragments keeps one identity** — which is what lets, for example, death-dye signal be quantified while an organoid falls apart. Objects that fuse keep their own IDs; the joined mask is divided between them.
- After each timepoint, a track whose **total size in that frame is below 100 voxels** is removed (fixed, not adjustable). Small fragments of a larger object are kept. Because each frame is propagated from the previous one, a removed track does not come back.
- Foreground that neither overlaps an existing track nor lies within that small gap of one receives **no ID**: objects that appear part-way through the movie are not tracked (new objects touching an existing track take that track's ID).

## When it applies

Because linking is by spatial overlap, the method relies on **each object substantially overlapping itself between consecutive frames**. That single condition, measured in your own data, decides whether this method fits. It is not a property of a cell type: the same object overlaps itself at a short frame interval and does not at a long one.

It suits populations that:

- **stay put**, or drift only slightly, so consecutive frames overlap;
- are a **closed set** — no new objects appear part-way through the movie, whether by entering the field or by dividing;
- change mainly by **fragmenting**, for example as a structure breaks up on death.

Multicellular structures such as organoids and spheroids are the usual case. If new objects appear during the movie, or touching objects must keep separate identities, use [Bounded Propagation](bounded_propagation.md); if objects no longer overlap themselves between frames, use [btrack](btrack.md).

## Parameters

Fragmentation Propagation has **no tunable parameters**; the panel only shows *"No tunable parameters for Fragmentation Propagation."* (plus the checkbox below on organoid sub-tabs).

## "Track all organoids together" checkbox (organoids only)

By **default**, all organoid cell types are tracked as one cohort in a single **🟣 All Organoids** sub-tab. It shows:

- **Method: Fragmentation Propagation** (fixed — the method cannot be changed here);
- an orange banner: *"Fragmentation Propagation will be applied to ALL organoid types simultaneously. To run individual tracking, uncheck 'Track all organoids together'"*;
- the ticked **Track all organoids together** checkbox;
- the button **▶ Run All Organoids Tracking (*organoid types*)**.

```{admonition} When to set it
:class: important

- **Leave it ON (default)** to track all organoid types **in one scene with a single, shared set of track IDs**. The masks of all organoid types are combined at every timepoint; IDs are assigned at the first timepoint (unique across types) and propagated through the combined masks, so where two types touch, a track can spread into the other type's mask. You get the combined `all_organoids` output **and** per-type outputs split from the same run, so each type can still be analysed on its own.
- **Turn it OFF** to track each organoid type **separately** — its own run, its own independent IDs, and each sub-tab free to use a **different tracking method**.
```

Keep it on when several organoid types coexist in the same well and should share one overlap-based scene (e.g. `organoidA`, `organoidB`), or when you want a single "any organoid" cohort. Turn it off when the types must not influence one another during tracking or need different methods. Signal bleed-through between channels should be resolved during segmentation; it is not by itself a reason to choose either option.

In combined mode, the CSVs gain an `organoid_type` column naming each track's type. An organoid type with no labels at the first timepoint gets no tracks.

**Unticking** the checkbox replaces the All Organoids sub-tab by one sub-tab per organoid type. Each of them shows a (now unticked) **Track all organoids together** checkbox while **Fragmentation Propagation** is selected, so you can switch back. The choice is saved in `behav3d_parameters.yml` straight away.

## Step-by-step in the napari plugin

1. **Open the Tracking tab.** By default organoids appear as a single **🟣 All Organoids** sub-tab; if you unticked **Track all organoids together**, pick the individual organoid sub-tab (🟣) instead.
2. **Method is Fragmentation Propagation.** On All Organoids it is fixed; on an individual sub-tab pick it from the **Method:** dropdown (it is the default for organoid sub-tabs). There is nothing to tune.
3. **Decide whether to keep Track all organoids together** (see above).
4. **Run it.** On All Organoids click **▶ Run All Organoids Tracking (…)**; on an individual sub-tab click **Run *Organoid* Tracking**. Batch and queued runs: see [Generic workflow](index.md#generic-workflow-per-sub-tab).
5. **Inspect the result** in the Visualization tab — see [Tuning the result](#tuning-the-result).

## Tuning the result

There are no parameters to change. Tuning means **improving the segmentation**, **switching to another tracking method**, or **correcting by hand** with [Manual editing](manual_editing.md).

**An object that appears part-way through the movie is missing from the tracked output**

- Cause: it overlaps no existing track, so it never gets an ID (closed-set assumption).
- Fix: use [Bounded Propagation](bounded_propagation.md), which gives new objects new IDs, or add it afterwards with the **Create** tool in [Manual editing](manual_editing.md).

**Tracks are lost when the object moves**

- Cause: the object moves so far between frames that it no longer overlaps its previous position.
- Fix: switch to [btrack](btrack.md), which follows motion with a centroid-based model.

**Two objects that separated still share one ID**

- Cause: by design, every piece that overlaps an object's previous footprint inherits its ID. That is right for a fragmenting organoid, wrong for two objects that were fused in the first frame and later separate.
- Fix: use [Bounded Propagation](bounded_propagation.md), or split the label with **Split** in [Manual editing](manual_editing.md).

**Two touching objects exchange territory or swap IDs**

- Cause: where objects touch, the watershed divides the joined region between their IDs and the boundary may shift.
- Fix: improve segmentation so the objects stay separated, or correct with **Swap** / **Split** in [Manual editing](manual_editing.md).

**Small objects disappear**

- Cause: a track whose total size in a frame falls below 100 voxels is removed and does not come back (fragments of a larger object are kept).
- Fix: usually harmless debris. If these objects matter, [Bounded Propagation](bounded_propagation.md) has an adjustable size limit — but it gives separated pieces their own IDs.

**You tracked all organoids together but needed fully separate tracking**

- Untick **Track all organoids together**, then re-track each organoid sub-tab separately.

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

When **Track all organoids together** is on, you get **both**:

- **Combined:** `<sample>_all_organoids_tracked.zarr` and `trackdata/<sample>/all_organoids/<sample>_all_organoids_tracks.csv` — all organoid types in one cohort.
- **Per type:** the usual files for each organoid cell type, split from the same run.

The metadata columns of each organoid type point to the **per-type** files, not to `all_organoids`.

## Good practices & tips

- **It assumes frame-to-frame overlap.** Measure that overlap rather than assuming it from the cell type, and use [btrack](btrack.md) when it fails.
- **Because it is overlap-based, segmentation quality dominates.** Fix the segmentation first, or correct with [Manual editing](manual_editing.md) after tracking.
- **Choose Track all organoids together based on how tracking should run**, not on whether you need per-type results — per-type outputs are written either way.

## See also

- [Tracking overview](index.md).
- [Bounded Propagation](bounded_propagation.md).
- [Manual editing](manual_editing.md).

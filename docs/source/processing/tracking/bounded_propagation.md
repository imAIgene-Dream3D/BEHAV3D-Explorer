# 🧱 Bounded Propagation

Bounded Propagation is an overlap-based tracker for static or slow-moving objects that remain spatially
overlapping between consecutive frames but may touch or form joined segmentation
regions. It uses watershed propagation like [Fragmentation Propagation](fragmentation_tracking.md), with one extra
topology rule: a track ID cannot span more than one disconnected region.

## When to use it

Use Bounded Propagation when all of the following are true:

- objects retain substantial frame-to-frame overlap (static/slow-moving);
- masks can touch, merge, or change shape but should keep separate identities;
- objects that separate during the experiment should become separate tracks, and new objects may appear during the movie.

If detections lose overlap because displacement is large relative to the frame
interval, use [btrack](btrack.md) instead. If spatially separated pieces should stay in the same track (dying objects that fall apart), use [Fragmentation Propagation](fragmentation_tracking.md). Try both on one sample when either interpretation is plausible.

## How it works

Before watershed propagation, the current frame's mask is split into **connected regions**. Each existing track keeps its seed only in the **one region its previous footprint overlaps most**; its markers everywhere else are discarded. Watershed then floods each region from the seeds it kept:

- a region only one track kept → flooded entirely to that track;
- a region several tracks kept → split between them by watershed (genuine merges/splits are preserved);
- a region no track kept → given a **single new track ID per disconnected patch** (so one blob does not splinter into several new tracks).

So when an object breaks apart, its ID stays with the largest-overlap piece and the other pieces become new tracks. There is no gap-closing dilation here, so an ID can never leak across a background gap into a disconnected blob.

## Parameters

| Control | Default | Range | What it does |
|---|---|---|---|
| **Min overlap fraction** | `0.00` | 0.00–1.00 | Fraction of a current-frame region's area that a track's previous-frame footprint must cover before the track may claim that region. `0` = any shared pixel qualifies; raise it to stop a track claiming a region it barely touches. |
| **Min segment size** | `20` | ≥ 1 (voxels) | Minimum segment size, applied **after** watershed. It cleans up small fragments the watershed splitting itself can create (a shared region divided between tracks, or a leftover sliver turned into a new track), so it is normally set **much lower** than a segmentation-stage size filter. Flat segments (one voxel thick along any axis) are always removed. |

## Step-by-step in the napari plugin

1. **Open the Tracking tab** and pick the sub-tab for the cell type you want to track.
2. From the **Method:** dropdown, pick **Bounded Propagation**.
3. Adjust **Min overlap fraction** and **Min segment size** if needed.
4. Click **Run *Cell type* Tracking**, or use the batch / queue controls — see [Generic workflow](index.md#generic-workflow-per-sub-tab).

## Validation

Inspect at least one representative movie in Visualization. Verify that IDs remain
stable through touching events, that one ID does not cover disconnected objects, and
that small real objects are not removed by the size threshold.

## See also

- [Tracking overview](index.md)
- [Fragmentation Propagation](fragmentation_tracking.md)
- [btrack](btrack.md)

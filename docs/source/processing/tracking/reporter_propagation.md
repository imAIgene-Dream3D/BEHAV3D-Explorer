# 🔆 Reporter Propagation

Reporter Propagation is a tracking method for **near-static objects whose
segmentation is unreliable or intermittent over time** — objects that flicker on
and off between timepoints because their fluorescent signal is not always visible
(for example a calcium or other reporter that switches on and off while the cell
itself stays in place).

Instead of linking detections frame to frame, it works across the whole movie at
once:

1. It pools **every** segment detected at **every** timepoint.
2. It groups together segments that **spatially overlap**, regardless of how far
   apart in time they were detected: each segment is compared with the largest
   instance found so far for each object.
3. For each group it takes the **single largest** detected instance and uses that
   one shape as the object's mask, **stamped onto every timepoint** unchanged.

Because the same static mask is reused at every frame, an intensity trace can be
read for each object even at the timepoints where it was not segmentable on its own.
This also applies to objects first detected part-way through the movie: they get an
ID and their mask is present at every timepoint, including those before they
appeared. There are no linking or gap parameters: grouping is purely spatial.

```{important}
Reporter Propagation reuses **one static shape** for every timepoint, so it assumes
each object's shape and position are genuinely static. Any real movement or
morphology change is masked out, because the same shape is pasted into every frame.
```

If the reporter flickers **and the object moves**, do not use Reporter Propagation.
Track a constitutive channel and extract the reporter as an intensity feature. If no
constitutive channel exists, use permissive segmentation that accepts some false
positives and a moving-object tracker such as btrack. Reporter flicker and static
position are both required for this method.

## Parameters

| Control | Default | Range | What it does |
|---|---|---|---|
| **Min overlap fraction** | `0.10` | 0.00–1.00 | Minimum spatial overlap between a segment and an object's largest instance so far (from any timepoint, however far apart) — as a fraction of the **smaller** of the two volumes — for the segment to join that object. `0` means any shared pixel counts. Raise it if unrelated nearby objects are being grouped together. |
| **Min segment size** | `100` | ≥ 1 (voxels) | Segments smaller than this are ignored entirely (treated as noise), both as candidates for the "largest instance" and when grouping. |

## Step-by-step in the napari plugin

1. **Open the Tracking tab** and pick the sub-tab for the cell type you want to track.
2. From the **Method:** dropdown, pick **Reporter Propagation**.
3. Set **Min overlap fraction** and **Min segment size** if the defaults need adjusting.
4. Click **Run *Cell type* Tracking** (the green button at the bottom of the sub-tab),
   or use the batch / queue controls — see
   [Generic workflow](index.md#generic-workflow-per-sub-tab).
5. **Inspect the result** in the [Visualization](../../plugin_essentials/visualization.md) tab.

## Outputs

Standard tracking outputs (tracked zarr + tracks CSV) — see [Tracking overview](index.md).

## See also

- [Tracking overview](index.md).
- [Fragmentation Propagation](fragmentation_tracking.md) — overlap-based tracking for objects
  that are segmented reliably at every timepoint.

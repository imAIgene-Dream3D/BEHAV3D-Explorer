# ✏️ Manual editing of tracked segments

After automated segmentation and tracking you will often still find mistakes — split labels, wrong merges, swapped identities, missed cells, junk objects. The manual editor lets you fix them inside napari, preview every change on one frame before applying it to the whole track, and save the result back to disk.

```{important}
Manual editing operates on **tracked segments** (`*_tracked.zarr`), not on the raw segmentation output (`*_segments.zarr`). Run [Tracking](index.md) first so each cell has a stable ID over time — then come back here to clean up.
```

## Where to find it

The editor is opened from the **Manual Edition** group of the [Visualization](../../plugin_essentials/visualization.md) tab:

1. Load a sample with **Load Dataset into Napari**. The **Manual Edition** group appears once at least one tracked-segments layer is in the viewer.
2. *(Multicolor samples only)* choose the **Edit target:** — **Merged (combined)** to correct the `_merged` result directly, or **Independent channels** to correct each channel separately. Only one of the two is editable at a time.
3. Pick the layer in the **Tracked segments:** dropdown.
4. Click **✏️ Edit tracked segments**. The button turns into a red **⏹ Stop editing**.

The viewer is then cleared and reloaded with only the raw image channels, the chosen tracked segments and their tracks. The editor panel (**Manual edition**, headed *sample · cell type ● Editing*) opens below the button. While *"Building label index…"* is shown, everything already works; label clicks just become faster once it is done.

```{tip}
**Multicolor, independent channels:** after correcting the channels, stop editing and click **🔁 Rebuild merged** to regenerate the `_merged` output from them (for every sample). The channels are first checked for overlap: if two channels claim the same voxel, the existing merged output is kept and the conflict is reported in the log. Rebuilding renumbers the merged TrackIDs, so re-run Feature Extraction for the merged type (a warning appears if its features already exist).
```

## Selecting labels

| Input | Effect |
|---|---|
| **Left-click** on a label | Selects that label (replaces the current selection). |
| **Shift + left-click** | Adds the label to the selection (or removes it if already selected). |
| **Right-click** | Removes the label from the selection. |
| Type a TrackID in **Add by ID** and press **Enter** | Adds that ID — handy when you already know it. |
| **Clear** | Empties the selection. |

The selection is listed next to **Selected labels:** and outlined in **yellow** in the viewer. Tools that act on one label use the **first** selected label.

## The tools

Choose a tool with the row of buttons — **Split**, **Merge**, **Swap**, **Erode**, **Dilate**, **Delete**, **Create**, **Fill Holes**, **Fill Gaps**. Each tool page has a **👁 Preview …** button and an **Apply …** button (see [Preview and apply](#preview-and-apply)).

| Tool | Select | What it does | Controls |
|---|---|---|---|
| **Split** | 1 label | Splits the label into pieces grown (3-D watershed) from **≥ 2 seed points** placed inside it on the current frame; each piece then keeps its identity forward and backward within the time range, up to the first frame where the label is missing — frames after a tracking gap stay unsplit and must be split separately. | **Add seed points**; **First seed keeps the original TrackID** (on by default — the other pieces get new IDs) |
| **Merge** | ≥ 2 labels | Rewrites every selected label to one ID. | **Target ID:** (chosen from the selection) |
| **Swap** | exactly 2 labels | Exchanges the two TrackIDs over the time range, fixing an identity swap made by the tracker. In frames where only one of the two exists, it is rewritten to the other ID so gaps stay consistent. | — (use a **Custom range**, see warning below) |
| **Erode** | 1 label | Shrinks the label with a 3-D ball. | **Radius XY (px):** `1` (0–50); the matching radius in µm and in Z slices is shown next to it, so the ball is isotropic |
| **Dilate** | 1 label | Grows the label by the XY radius in each slice and by the matching number of Z slices (isotropic extent, cylinder-shaped growth); never grows into another label. | **Radius XY (px):** `1` (0–50) |
| **Delete** | ≥ 1 label | Erases the selected labels over the time range. | **Yes, erase the selected label(s)** must be ticked before **Apply delete** (not needed for the preview); it is unticked again after each delete |
| **Create** | none | Creates **one new label per seed** placed on unlabelled voxels of the current frame, and follows it forward and backward in time. | **Image channel:** a raw channel, or **(none — Voronoi fallback)**; **Add seed points** |
| **Fill Holes** | 1 label | Fills internal cavities of a hollow (shell-like) label, e.g. organoids segmented only on their outer shell. Only cavities fully enclosed in 3-D are filled, and only background voxels are written. | — |
| **Fill Gaps** | 1 label | Rebuilds timepoints where the track briefly disappears, by morphing the mask between the frame before and the frame after the gap. Only background voxels are filled; a gap whose reconstruction would overlap another label is skipped as *conflicted*. | **Max gap size (timepoints):** `5` (1–100); longer gaps are skipped and listed in the log |

```{warning}
**Swap over the full lifetime only renames the two tracks.** To repair a swap that happened part-way, select **Custom range**, set **Start T** to the frame where the swap began, and check that **End T** reaches the last frame of both tracks (it is pre-filled with the first selected label's lifetime only).
```

**Create** with an **Image channel** restricts the new label to bright voxels (above an automatic Otsu threshold of that channel, computed per frame) and places boundaries along dim regions between objects. With **(none — Voronoi fallback)** every unlabelled voxel connected to a seed is given to the nearest seed, in every frame of the range — a single seed fills the whole unlabelled space, so choose an **Image channel** for a single object. With **Full lifetime** selected, Create works over the whole movie.

**Delete short tracks** (on the **Delete** page): tick **Delete all labels shorter than** and set the number of **timepoints** (`5` by default). **👁 Preview short-track labels** lists the affected labels in the log; **Delete short-track labels** removes every label whose span from first to last appearance is shorter than that, without asking again. It works on the whole movie and needs no selection (while ticked, the time range is locked to **Full lifetime**).

### Placing seed points (Split and Create)

**Add seed points** creates a temporary layer of red points and, if the viewer is in 3-D, switches it to 2-D so clicks land precisely. Click inside the label (Split) or on unlabelled voxels (Create); the counter shows **N seed(s) on current frame**. Seeds may lie on different Z slices — move the Z slider between clicks — but must all be on the **current timepoint**. The seed layer is removed after **Apply**.

The **? How to place seeds** button opens a five-step illustrated guide, **Placing seed points** (**< Back** / **Next >** / **Close**): select the label (Split only) → click **Add seed points** → place one seed per sub-region (Split) or per new label (Create) → **Preview** → **Apply**.

### Preview and apply

- **👁 Preview** runs the tool on the displayed timepoint only (for **Fill Gaps**: on the gap around it) and shows the result in the layer without keeping it. The preview is cleared when you apply, switch tool or discard.
- **Apply** runs the tool over the chosen time range. While it runs the buttons are disabled and a progress bar is shown in the panel (and in napari's activity panel).

## Time range

Each **Apply** acts on a time range:

- **Full lifetime of the selected label(s) — recommended** (default): from the label's first to its last appearance. For **Merge** and **Swap**, the combined lifetime of all selected labels; for **Create**, the whole movie.
- **Custom range** — **Start T:** / **End T:**, pre-filled with the lifetime of the first selected label. Useful for "only from t = 30 onwards" edits, and required for a partial **Swap**.

**Workers:** sets the number of CPU cores used to process timepoints in parallel; it is the same value as the other **Workers** fields of the plugin.

## Saving, discarding and closing

There is **no undo/redo**. Every applied edit is kept in memory until you save; the line under **Workers:** shows *No unsaved changes* or *⚠️ N pending frame change(s) (unsaved) — use Discard to revert all*.

- **💾 Save tracked segments** — writes the changed frames into the `*_tracked.zarr` and rebuilds the tracks CSV from the edited labels.
- **✗ Discard all changes** — after a **Discard all edits?** confirmation, reverts everything since the last save. This cannot be undone.
- **Stop editing** (panel footer) or **⏹ Stop editing** (Visualization tab) — closes the editor. With unsaved edits, an **Unsaved tracked-segment edits** dialog offers **Save and close**, **Discard and close** or **Cancel**. The same happens if you switch to another tab while editing. The editor cannot be closed while an operation is still running.

## Tips & best practices

- **Track first, then edit.** Editing the segmentation before tracking gives you back the same broken IDs after tracking.
- **Preview before applying, and save before a risky edit** — with no undo, **✗ Discard all changes** only goes back to the last save.
- **Use Full lifetime by default**; Custom ranges are easy to get wrong when a track extends further than you remember (Swap is the exception).
- **Save before closing napari** — unsaved edits live only in memory.
- **Re-run Feature Extraction** for the edited samples after saving, so features reflect the corrected tracks.
- **Junk objects:** use **Delete**, or **Delete all labels shorter than** for many short-lived fragments. If there are very many, stricter segmentation filtering (e.g. a larger minimum object size) is usually cheaper.

## See also

- [Visualization tab](../../plugin_essentials/visualization.md) — where the editor is launched from.
- [Tracking overview](index.md) — get tracked segments before editing.

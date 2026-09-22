"""
BEHAV3D napari plugin – Feature Extraction Tab.

Provides per-cell-type sub-tabs with feature checkboxes (movement, intensity,
morphology, contact, death), thresholds, workers, and batch-run capability.

Dead threshold logic
--------------------
- **Organoids**: a single *global* threshold is shared across all organoid types.
  The threshold is set in the "Death Classification — Organoids" group at the top
  of this tab, and the same value is used for every organoid cell type.
- **Immune / other cell types**: each type has its *own* threshold spinner, set
  inside its individual sub-tab panel.
"""

import sys
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QTabWidget, QTextEdit, QCheckBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QMessageBox, QScrollArea,
    QComboBox, QToolTip, QSplitter, QListWidget
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor

from behav3d.core.qt_help import HelpButton, make_help_row, reset_scroll_on_page_change
from behav3d.napari._analysis import CollapsibleSection
from behav3d.napari._units import UnitGroupManager
from behav3d.napari._results_panel import (
    ResultsPanel,
    notify_results_changed,
)
from behav3d.napari._background_runner import (
    BackgroundOperation,
    ProgressBarRow,
    fire_extra_callback,
)
from behav3d.napari._preview_dims import (
    clear_viewer_layers,
    disconnect_all_preview_dims_listeners,
    register_preview_dims_listener,
    unregister_preview_dims_listener,
)

# Colormaps for raw channel layers (same order as the Visualization tab)
_CHANNEL_COLORS = ["cyan", "yellow", "green", "red", "blue", "magenta"]

# Prefix used for all temporary preview layers so they can be cleaned up easily
_PREVIEW_PREFIX = "[Preview]"

# The Dead/Alive preview overlay is a single shared napari layer (one
# "[Preview] Dead/Alive" layer for the whole viewer, even when several
# organoid cell-type panels share it — see ``_propagate_org_preview_to_panels``,
# which re-attaches a hover callback on *every* organoid panel each time the
# frame is recomputed). Track the single currently-active hover callback here
# (keyed by ``id(viewer)``) so attaching a new one always removes whichever
# callback — from *any* panel instance — is currently registered, instead of
# only removing the calling panel's own previous callback. Without this,
# multiple simultaneously-registered callbacks race on every mouse move and
# most of them overwrite ``viewer.status`` with an empty/wrong result.
_ACTIVE_PREVIEW_HOVER: dict = {}

# Likewise, only one dims (time-slider) listener should ever be registered
# on ``viewer.dims.events.current_step`` for the preview feature at a time —
# whether it's a single cell-type panel's own listener or the organoid-shared
# one on FeatureExtractionTab. Switching from one preview (e.g. a "tcell"
# panel) to another (e.g. the organoid panel) must drop whichever listener —
# from *any* panel/tab instance — is currently registered, not just the
# newly-active side's own previous listener. Otherwise the stale listener
# keeps firing against layers/arrays that were just cleared/replaced,
# producing missing or clobbered "Dead/Alive" / "% Dead Mask" layers (or a
# crash if it reaches for layers mid-teardown).
#
# This dict/helper only arbitrates between this *file's own* two preview
# panels. Every connect/disconnect below also registers/unregisters with
# ``behav3d.napari._preview_dims``'s viewer-wide registry, so that *other*
# tabs (Visualization, State/Track Classification, Feature Backprojection)
# can drop this tab's listener too via ``disconnect_all_preview_dims_listeners``
# before they bulk-mutate ``viewer.layers`` — the cross-tab version of the
# same "stale listener fires mid-teardown" hazard described above.
_ACTIVE_PREVIEW_DIMS: dict = {}


def _disconnect_any_active_preview_dims(viewer):
    """Remove whichever preview dims listener is currently active on
    ``viewer``, regardless of which panel/tab registered it."""
    if viewer is None:
        return
    cb = _ACTIVE_PREVIEW_DIMS.pop(id(viewer), None)
    if cb is None:
        return
    try:
        viewer.dims.events.current_step.disconnect(cb)
    except Exception:
        pass


def _find_main_widget(start):
    """Walk up the parent chain to the top-level ``BEHAV3DWidget``."""
    w = start
    while w is not None and not hasattr(w, "tabs"):
        w = w.parent() if hasattr(w, "parent") else None
    return w


def _notify_post_extraction(source_widget):
    """Show a 'Run Filtering before Analysis' reminder after extraction.

    Honours a session-level opt-out flag stored on the top ``BEHAV3DWidget``
    (``_skip_filter_reminder``). The reminder is never raised during queued
    or non-interactive runs (those should call this only from the
    interactive paths).
    """
    main = _find_main_widget(source_widget)
    if main is None:
        return
    if getattr(main, "_skip_filter_reminder", False):
        return

    box = QMessageBox(source_widget)
    box.setWindowTitle("Feature Extraction Complete")
    box.setIcon(QMessageBox.Information)
    box.setText(
        "Feature extraction is complete.\n\n"
        "Remember to run Filtering before any Analysis step \u2014 "
        "analysis reads the filtered track-features CSV."
    )

    btn_run = box.addButton("Run Filtering with current setup", QMessageBox.AcceptRole)
    btn_goto = box.addButton("Go to Filtering tab", QMessageBox.ActionRole)
    btn_ok = box.addButton("OK", QMessageBox.RejectRole)
    box.setDefaultButton(btn_ok)

    optout = QCheckBox("Don't remind me this session")
    box.setCheckBox(optout)

    box.exec_()
    if optout.isChecked():
        try:
            main._skip_filter_reminder = True
        except Exception:
            pass

    clicked = box.clickedButton()
    if clicked is btn_run:
        filt = getattr(main, "filtering_tab", None)
        if filt is not None and hasattr(filt, "run_batch_filtering"):
            try:
                filt.run_batch_filtering(interactive=True, block=False)
            except Exception:
                traceback.print_exc()
    elif clicked is btn_goto:
        try:
            tabs = getattr(main, "tabs", None)
            filt = getattr(main, "filtering_tab", None)
            if tabs is not None and filt is not None:
                idx = tabs.indexOf(filt)
                if idx >= 0:
                    tabs.setCurrentIndex(idx)
        except Exception:
            traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
# Shared preview helpers (module-level so panels + tab can both use them)
# ═══════════════════════════════════════════════════════════════════════════

def _find_sample_with_segments(md: pd.DataFrame, cell_type: str):
    """Return (row, seg_path_str) for the first sample that has a valid
    segments file for *cell_type*, or (None, None) if none found."""
    for _, row in md.iterrows():
        for prefix in ("or", "im", "ot"):
            col = f"{prefix}_{cell_type}_segments_image_path"
            if col in row.index and pd.notna(row.get(col)):
                p = str(row[col]).strip()
                if p and Path(p).exists():
                    return row, p
    return None, None


def _resolve_dead_mask(sample_row: pd.Series, output_dir: Path, log_fn=None):
    """Try to find the dead mask array for a sample.

    Resolution order:
      1. ``dead_mask_path`` column in metadata (set by APOC segmentation).
      2. Several constructed paths under ``output_dir/images/{sample}/``.
      3. Raw dead channel -> Otsu threshold on first timepoint (last resort).

    Parameters
    ----------
    log_fn : callable | None
        Optional log function; each tried path is reported through it.

    Returns
    -------
    arr : dask.Array | None
    method : str | None   'zarr' | 'raw' | None
    tried : list[str]     All paths checked (for diagnostics).
    """
    import dask.array as da
    _log = log_fn or (lambda m: None)
    tried: list = []

    def _try_load(p):
        p = Path(p)
        tried.append(str(p))
        if not p.exists():
            return None
        try:
            from behav3d.io.images import load_image
            arr = load_image(p)
            if not isinstance(arr, da.Array):
                arr = da.from_array(np.asarray(arr))
            return arr
        except Exception as exc:
            _log(f"  Found {p.name} but failed to load: {exc}")
            return None

    # 1. PRIMARY: dead_mask_path column in metadata
    #    Written by ALL segmentation pipelines:
    #      - APOC          (apoc_segment.py)
    #      - Cellpose      (cellpose_prediction.py)
    #      - Pixel classifier (napari_pixelclassifier.py)
    dm_val = sample_row.get("dead_mask_path")
    if dm_val and pd.notna(dm_val):
        _log(f"  [dead mask] metadata dead_mask_path = {dm_val}")
        arr = _try_load(str(dm_val).strip())
        if arr is not None:
            return arr, "zarr", tried
        _log("  [dead mask] path from metadata does not exist on disk — trying fallbacks")
    else:
        _log(
            "  [dead mask] dead_mask_path column is empty or missing in metadata. "
            "Make sure segmentation has been run and the metadata CSV was saved afterwards."
        )

    # 2. FALLBACK: Constructed path using the canonical naming convention
    #    ({sample}_mask_dead.zarr is what all pipelines actually write)
    sample_name = str(sample_row.get("sample_name", ""))
    img_dir = output_dir / "images" / sample_name
    for name in (
        f"{sample_name}_mask_dead.zarr",   # APOC / pixelclassifier standard
        f"{sample_name}_dead_mask.zarr",   # Cellpose alternate naming
    ):
        arr = _try_load(img_dir / name)
        if arr is not None:
            _log(f"  [dead mask] found via fallback path: {img_dir / name}")
            return arr, "zarr", tried


    # 3. Raw dead channel -> Otsu (last resort; only first timepoint)
    raw_path = sample_row.get("raw_image_path")
    dead_ch_idx = sample_row.get("dead_channel")
    if raw_path and pd.notna(raw_path) and pd.notna(dead_ch_idx):
        label = f"raw_channel[{int(dead_ch_idx)}] via Otsu"
        tried.append(label)
        try:
            from behav3d.io.images import load_image
            raw = load_image(raw_path)
            ch = int(dead_ch_idx)
            dead_ch = raw[:, ch, ...] if raw.ndim >= 5 else raw[ch, ...]
            t0 = np.asarray(dead_ch[0] if dead_ch.ndim >= 4 else dead_ch)
            from skimage.filters import threshold_otsu
            try:
                otsu_thr = threshold_otsu(t0[t0 > 0]) if (t0 > 0).any() else float(t0.mean())
            except Exception:
                otsu_thr = float(t0.mean())
            dead_mask_t0 = (t0 > otsu_thr).astype(np.uint8)
            return da.from_array(dead_mask_t0[np.newaxis, ...]), "raw", tried
        except Exception:
            pass

    return None, None, tried


def _load_raw_dask(sample_row: pd.Series, output_dir: Path):
    """Load the raw image as a dask array (T, C, Z, Y, X).  Returns None on failure."""
    import dask.array as da

    sample_name = str(sample_row.get("sample_name", ""))
    raw_path_val = sample_row.get("raw_image_path", "")
    if not raw_path_val or pd.isna(raw_path_val):
        return None

    raw_p = Path(str(raw_path_val).strip())

    # Prefer zarr (lazy loading)
    zarr_candidates = [
        raw_p if raw_p.suffix == ".zarr" else None,
        output_dir / "images" / sample_name / f"{sample_name}.zarr",
        raw_p.with_suffix(".zarr"),
    ]
    for zp in zarr_candidates:
        if zp and zp.exists():
            try:
                from behav3d.io.formats.zarr import load_zarr
                return load_zarr(zp)
            except Exception:
                pass

    # Fallback: load full image
    if raw_p.exists():
        try:
            from behav3d.io.images import load_image
            img = load_image(raw_p)
            if not isinstance(img, da.Array):
                img = da.from_array(np.asarray(img), chunks=(1,) + np.asarray(img).shape[1:])
            return img
        except Exception:
            pass

    return None


def _clean_dead_frame(dead_frame, immune_segs_dict, frame_idx):
    """Zero dead-mask voxels under immune segments for a single timepoint.

    Parameters
    ----------
    dead_frame : array-like
        3-D dead mask volume (Z, Y, X) for one timepoint (already materialized).
    immune_segs_dict : dict[str, array-like]
        Mapping of immune cell type name -> full (T, Z, Y, X) segment array
        (may be lazy/dask). Only the slice at *frame_idx* is materialized.
    frame_idx : int
        Timepoint index to extract from each immune segment array.

    Returns
    -------
    np.ndarray
        Cleaned dead mask (same shape as input).
    """
    cleaned = np.array(dead_frame, copy=True)
    for _name, im_arr in immune_segs_dict.items():
        try:
            im_frame = np.asarray(im_arr[frame_idx])
        except (IndexError, Exception):
            continue
        if im_frame.ndim > 3:
            im_frame = im_frame[0]
        if im_frame.shape != cleaned.shape:
            continue
        cleaned[im_frame > 0] = 0
    return cleaned


def _merge_org_segments_frame(segs_dict, org_cell_types, frame_idx):
    """Merge organoid segment arrays for a single timepoint.

    Like :func:`_merge_org_segments` but only materializes the slice at
    *frame_idx* from each (potentially lazy) segment array.

    Returns
    -------
    merged_vol : np.ndarray (Z, Y, X) | None
    label_type_map : dict  merged_label -> (cell_type, original_label)
    """
    available = [ct for ct in org_cell_types if ct in segs_dict]
    if not available:
        return None, {}

    def _get_frame(arr, fidx):
        vol = np.asarray(arr[fidx])
        if vol.ndim > 3:
            vol = vol[0]
        return vol.astype(np.int32)

    if len(available) == 1:
        ct = available[0]
        vol = _get_frame(segs_dict[ct], frame_idx)
        label_map = {int(lbl): (ct, int(lbl)) for lbl in np.unique(vol) if lbl > 0}
        return vol, label_map

    ref = _get_frame(segs_dict[available[0]], frame_idx)
    merged = np.zeros(ref.shape, dtype=np.int32)
    label_map = {}
    offset = 0
    for ct in available:
        vol = _get_frame(segs_dict[ct], frame_idx)
        if vol.shape != merged.shape:
            continue
        max_lbl = int(vol.max()) if vol.size > 0 else 0
        for lbl in np.unique(vol):
            if lbl <= 0:
                continue
            label_map[int(lbl + offset)] = (ct, int(lbl))
        shifted = np.where(vol > 0, vol + offset, 0)
        merged = np.where(shifted > 0, shifted, merged)
        offset += max_lbl + 1
    return merged, label_map


def _load_all_segments_for_sample(
    sample_row: pd.Series,
    all_cell_types: list,
    output_dir: Path,
    lazy: bool = False,
) -> "tuple[dict, dict[str, str]]":
    """Load segment arrays for every cell type that has data for *sample_row*.

    Returns a dict mapping cell_type -> array, typically:
      - 4-D (T, Z, Y, X) for timelapse labels, or
      - 3-D (Z, Y, X) for single-volume labels.
    Missing or unreadable types are silently omitted.

    When *lazy* is True the arrays are kept as dask (or whatever
    ``load_image`` returns) instead of being materialized with
    ``np.asarray``. This is useful for the preview where only one
    timepoint at a time is needed.

    Segment source priority is tracked-first:
      1) metadata tracked path
      2) constructed tracked path
      3) metadata untracked path
      4) constructed untracked path

    Returns
    -------
    segs : dict[str, array-like]
        Loaded segment arrays by cell type.
    sources : dict[str, str]
        Source tag per loaded cell type: "tracked" or "untracked".
    """
    from behav3d.io.images import load_image as _li
    result = {}
    sources = {}
    sample_name = str(sample_row.get("sample_name", ""))

    def _normalize(arr):
        if lazy:
            if arr.ndim == 5:
                return arr[:, 0, ...]
            return arr
        arr_np = np.asarray(arr)
        if arr_np.ndim == 5:
            arr_np = arr_np[:, 0, ...]
        return arr_np

    def _try_load(p: Path):
        if not p.exists():
            return None
        try:
            return _normalize(_li(str(p)))
        except Exception:
            return None

    for ct in all_cell_types:
        # 1. Metadata tracked path (preferred)
        tracked_cols = [
            f"{pfx}_{ct}_tracks_image_path" for pfx in ("or", "im", "ot")
        ] + [f"{ct}_tracks_image_path"]
        for col in tracked_cols:
            val = sample_row.get(col)
            if val and pd.notna(val):
                arr_np = _try_load(Path(str(val).strip()))
                if arr_np is not None:
                    result[ct] = arr_np
                    sources[ct] = "tracked"
                    break
        if ct in result:
            continue

        # 2. Constructed tracked fallback
        for suffix in (
            f"{sample_name}_{ct}_tracked.zarr",
            f"{ct}_tracked.zarr",
        ):
            p = output_dir / "images" / sample_name / suffix
            arr_np = _try_load(p)
            if arr_np is not None:
                result[ct] = arr_np
                sources[ct] = "tracked"
                break
        if ct in result:
            continue

        # 3. Metadata untracked path
        for pfx in ("or", "im", "ot"):
            col = f"{pfx}_{ct}_segments_image_path"
            val = sample_row.get(col)
            if val and pd.notna(val):
                arr_np = _try_load(Path(str(val).strip()))
                if arr_np is not None:
                    result[ct] = arr_np
                    sources[ct] = "untracked"
                    break
        if ct in result:
            continue

        # 4. Constructed untracked fallback
        for suffix in (
            f"{sample_name}_{ct}_segments.zarr",
            f"{ct}_segments.zarr",
        ):
            p = output_dir / "images" / sample_name / suffix
            arr_np = _try_load(p)
            if arr_np is not None:
                result[ct] = arr_np
                sources[ct] = "untracked"
                break
    return result, sources


def _add_channel_layers(viewer, dask_img, sample_name: str, prefix: str = _PREVIEW_PREFIX):
    """Split a (T, C, Z, Y, X) dask array along C and add each channel as a
    napari Image layer with additive blending."""
    if dask_img is None:
        return
    n_channels = dask_img.shape[1] if dask_img.ndim >= 5 else 1
    for c in range(n_channels):
        ch_data = dask_img[:, c, ...] if dask_img.ndim >= 5 else dask_img
        color = _CHANNEL_COLORS[c % len(_CHANNEL_COLORS)]
        layer = viewer.add_image(
            ch_data,
            name=f"{prefix} {sample_name} – Ch{c}",
            colormap=color,
            blending="additive",
            opacity=0.7,
            visible=True,
        )
        # Auto-contrast on non-zero pixels
        try:
            flat = np.asarray(ch_data[0]).ravel()
            flat = flat[flat > 0]
            if flat.size > 0:
                layer.contrast_limits = (0, float(np.percentile(flat, 99.8)))
        except Exception:
            pass


def _overlay_for_volume(seg_vol, dead_vol, threshold_pct, frame_label="", log_fn=None):
    """Single-volume primitive: compute Dead/Alive overlay + per-region stats.

    Returns
    -------
    overlay_vol : np.ndarray (uint8)
        Same shape as ``seg_vol``. 0 = bg, 1 = alive, 2 = dead.
    pct_vol : np.ndarray (float32)
        Same shape as ``seg_vol``. Each segment's voxels contain its dead
        coverage as a percentage (range 0.0–100.0), matching ``pct_dead`` in
        ``region_stats`` below and the "% Dead Mask" layer's own label.
        Background is 0.
    region_stats : list[dict]
        One entry per labelled segment with keys ``label``, ``centroid``,
        ``pct_dead``, ``extent_x``.
    """
    import datetime
    from skimage.measure import regionprops

    _log = log_fn or (lambda m: None)

    def _ts():
        return datetime.datetime.now().strftime("%H:%M:%S")

    overlay_vol = np.zeros_like(seg_vol, dtype=np.uint8)
    pct_vol = np.zeros_like(seg_vol, dtype=np.float32)
    dead_binary = dead_vol > 0
    region_stats = []

    seg_int = seg_vol.astype(np.int32)
    ndim = seg_int.ndim
    regions = regionprops(seg_int)
    n_regions = len(regions)
    if frame_label:
        msg = f"[{_ts()}] [Preview] Computing overlay{frame_label}: {n_regions} segments…"
        print(msg)
        _log(f"  ⏳ Overlay{frame_label}: {n_regions} segments…")
    for region in regions:
        label_val = region.label
        mask = seg_int == label_val
        total_pixels = int(mask.sum())
        if total_pixels == 0:
            continue
        dead_pixels = int((mask & dead_binary).sum())
        pct = (dead_pixels / total_pixels) * 100.0
        overlay_vol[mask] = 2 if (pct >= threshold_pct and threshold_pct > 0) else 1
        pct_vol[mask] = pct

        bbox_min = region.bbox[:ndim]
        bbox_max = region.bbox[ndim:]
        extent_x = float(bbox_max[-1] - bbox_min[-1])
        region_stats.append(
            {
                "label": int(label_val),
                "centroid": tuple(float(v) for v in region.centroid),
                "pct_dead": float(pct),
                "extent_x": extent_x,
            }
        )
    return overlay_vol, pct_vol, region_stats


def _dead_alive_colormap():
    """Return a DirectLabelColormap for the Dead/Alive overlay, or ``None``."""
    try:
        from napari.utils.colormaps import DirectLabelColormap
        return DirectLabelColormap(
            color_dict={
                None: [0, 0, 0, 0],      # default for any unlisted label
                0:    [0, 0, 0, 0],      # background → transparent
                1:    [0, 0.8, 0, 0.6],  # alive → green
                2:    [0.9, 0, 0, 0.6],  # dead → red
            }
        )
    except Exception as exc:
        print(f"[BEHAV3D] DirectLabelColormap failed ({exc}); will use fallback color dict")
        return None


def _apply_dead_alive_colors(layer):
    """Apply dead/alive colors directly to a labels layer (fallback when
    DirectLabelColormap is unavailable)."""
    try:
        layer.color = {1: (0, 0.8, 0, 0.6), 2: (0.9, 0, 0, 0.6)}
    except Exception:
        pass



def _remove_layer_if_exists(viewer, layer_name: str):
    try:
        layer = viewer.layers[layer_name]
        viewer.layers.remove(layer)
    except Exception:
        pass


def _build_dead_pct_map(region_stats):
    """Build a {label_id: pct_dead} map from a single frame's region stats."""
    frame_map: dict = {}
    for r in region_stats or []:
        label_id = int(r.get("label", 0))
        if label_id > 0:
            frame_map[label_id] = float(r.get("pct_dead", 0.0))
    return frame_map


def _merge_org_segments(
    segs_dict: dict, org_cell_types: list
) -> "tuple[np.ndarray | None, dict]":
    """Merge segment arrays for multiple organoid types into one array with
    non-overlapping labels so the Dead/Alive overlay can cover all types.

    Returns
    -------
    merged_arr : ndarray | None
    label_type_map : dict
        merged_label_id -> (cell_type, original_label_id)
    """
    available = [ct for ct in org_cell_types if ct in segs_dict]
    if not available:
        return None, {}
    if len(available) == 1:
        ct = available[0]
        arr = np.asarray(segs_dict[ct])
        label_map = {
            int(lbl): (ct, int(lbl))
            for lbl in np.unique(arr)
            if lbl > 0
        }
        return arr.copy(), label_map

    ref = np.asarray(segs_dict[available[0]])
    merged = np.zeros(ref.shape, dtype=np.int32)
    label_map: dict = {}
    offset = 0
    for ct in available:
        arr = np.asarray(segs_dict[ct]).astype(np.int32)
        if arr.shape != merged.shape:
            continue
        max_lbl = int(arr.max()) if arr.size > 0 else 0
        for lbl in np.unique(arr):
            if lbl <= 0:
                continue
            label_map[int(lbl + offset)] = (ct, int(lbl))
        shifted = np.where(arr > 0, arr + offset, 0)
        merged = np.where(shifted > 0, shifted, merged)
        offset += max_lbl + 1
    return merged, label_map


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell-type panel
# ═══════════════════════════════════════════════════════════════════════════
class CellTypeFeaturePanel(QWidget):
    """Feature extraction controls for one cell type.

    Parameters
    ----------
    is_organoid : bool
        When *True* the dead threshold is controlled by the global spinner in
        ``FeatureExtractionTab``; no per-panel spinner is shown.
        When *False* a per-panel threshold spinner + preview button are shown.
    threshold_getter : callable | None
        Callable that returns the effective dead threshold value (float).
        Required when ``is_organoid=True``; otherwise the panel uses its own
        ``spin_dead_threshold`` widget.
    """

    def __init__(
        self,
        cell_type: str,
        category: str,
        metadata_loader,
        all_cell_types: list,
        category_types: list,
        log_callback=None,
        is_organoid: bool = False,
        threshold_getter=None,
        parent=None,
        tab_progress_row=None,
    ):
        super().__init__(parent)
        self.cell_type = cell_type
        self.category = category
        self.metadata_loader = metadata_loader
        self.all_cell_types = all_cell_types
        self.category_types = category_types
        self.log = log_callback or (lambda m: None)
        self.viewer = None
        self._is_organoid = is_organoid
        self._threshold_getter = threshold_getter  # kept for backward compat
        self._preview_connected = False
        # Background-execution infrastructure.
        self.tab_progress_row = tab_progress_row
        self._bg = BackgroundOperation(self)
        # List of organoid cell types (set by FeatureExtractionTab for org panels)
        self._org_cell_types: list = []

        # Shared cache dict (for organoid panels, set by FeatureExtractionTab)
        self._org_preview_cache: dict | None = None

        # Cached arrays for live overlay updates
        self._preview_seg_t = None
        self._preview_dead_t = None
        self._preview_segs_dict: dict = {}
        self._preview_immune_segs: dict = {}
        # merged_label_id -> (cell_type, original_label_id)  — organoid panels
        self._preview_label_type_map: dict = {}
        self._preview_hover_callback = None

        # Per-frame caches for the dead-threshold preview.
        # Overlay/pct arrays hold only the CURRENT frame (Z,Y,X) and are
        # replaced entirely when the user navigates to a different timepoint.
        self._preview_overlay_arr = None    # uint8  (Z,Y,X) — current frame
        self._preview_pct_overlay_arr = None # float32 (Z,Y,X) — current frame
        self._preview_label_arr = None      # int32  (Z,Y,X) — current frame, cell label ids
        self._preview_current_frame: int | None = None
        self._preview_pct_maps_by_frame: dict = {}
        self._preview_stats_by_frame: dict = {}
        self._preview_computed_frames: set = set()
        self._preview_pct_computed_frames: set = set()
        self._preview_current_thr: float | None = None
        self._preview_dims_callback = None
        self._preview_is_timelapse: bool = False

        # Read saved config
        params = self.metadata_loader.behav3d_parameters
        fcfg = params.get("features", {}).get(self.cell_type, {}) or {}

        # Detect dead channel presence
        md = metadata_loader.metadata
        self._has_dead = False
        if md is not None:
            self._has_dead = bool(
                "dead_channel" in md.columns and md["dead_channel"].notna().any()
            )

        # Will be created inside ``_init_ui`` if a dead channel is configured.
        self.btn_rerun_death = None

        self._init_ui(fcfg)
        self._refresh_rerun_death_button()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _init_ui(self, fcfg):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Feature checkboxes ────────────────────────────────────────────
        feat_group = QGroupBox("Features to extract")
        feat_lay = QVBoxLayout(feat_group)
        feat_lay.setSpacing(2)

        all_feats = ["movement", "intensity", "morphology", "contact", "invasiveness", "death"]

        # ── Mandatory features (same logic as the notebook widget) ─────────
        # intensity + contact are mandatory for all types.
        # movement is mandatory for immune / other cells.
        # death is mandatory whenever a dead channel is present.
        _mandatory: set = {"intensity", "contact"}
        if self.category in {"immune", "other"}:
            _mandatory.add("movement")
        if self._has_dead:
            _mandatory.add("death")
        self._mandatory_features: set = _mandatory

        default_feats = fcfg.get("features_choice", all_feats)
        if not isinstance(default_feats, (list, tuple)):
            default_feats = all_feats
        # Ensure mandatory features are always present in the saved config
        default_feats = list(default_feats)
        for _mf in self._mandatory_features:
            if _mf not in default_feats:
                default_feats.append(_mf)

        self.feature_checks: dict[str, QCheckBox] = {}
        for f in all_feats:
            if f == "death" and not self._has_dead:
                continue
            if f == "invasiveness" and self.category != "immune":
                # Invasiveness only makes sense for immune cells (counts immune
                # voxels inside organoid masks). Mirrors the notebook widget.
                continue
            label = (
                "Invasiveness"
                if f == "invasiveness" else f.capitalize()
            )
            cb = QCheckBox(label)
            cb.setChecked(f in default_feats)
            self.feature_checks[f] = cb

        # Force-check and disable mandatory checkboxes (cannot be toggled off)
        for _mf in self._mandatory_features:
            if _mf in self.feature_checks:
                self.feature_checks[_mf].setChecked(True)
                self.feature_checks[_mf].setEnabled(False)

        # Explanatory note shown at the top of the feature group
        _mandatory_note = QLabel(
            "ℹ  Greyed-out features are always extracted for this cell type."
        )
        _mandatory_note.setWordWrap(True)
        _mandatory_note.setMinimumWidth(0)
        _mandatory_note.setStyleSheet(
            "color: #90A4AE; font-style: italic; font-size: 10px;"
        )
        feat_lay.addWidget(_mandatory_note)

        # ── Contact threshold (top of the box; enabled only when Contact is
        # checked). Contact distance is computed in real (µm) space using the
        # sample's voxel spacing, so the native unit is physical — the toggle
        # below only changes how the value is *displayed*.
        self._contact_unit_mgr = UnitGroupManager(
            self.metadata_loader.metadata, default_physical=True,
        )
        contact_unit_row = QHBoxLayout()
        contact_unit_row.setSpacing(4)
        contact_unit_row.addWidget(QLabel("Contact units:"))
        contact_unit_row.addWidget(self._contact_unit_mgr.header_row(label=""))
        contact_unit_wrap = QWidget()
        contact_unit_wrap.setLayout(contact_unit_row)
        feat_lay.addWidget(contact_unit_wrap)

        self.contact_threshold = QDoubleSpinBox()
        self.contact_threshold.setRange(0.0, 100000.0)
        self.contact_threshold.setSingleStep(0.5)
        self.contact_threshold.setDecimals(2)
        self.contact_threshold.setMaximumWidth(90)

        contact_row = QHBoxLayout()
        contact_row.setSpacing(6)
        contact_row.addWidget(self.feature_checks.get("contact", QCheckBox()))
        contact_help = make_help_row(
            self.contact_threshold,
            "Contact Threshold",
            "Distance used to decide whether two cell segments are 'in "
            "contact'.\n\n"
            "Entered in the unit selected above (µm by default; contact "
            "distance is computed in real (µm) space using the sample's "
            "voxel spacing, so pixel-entered values are converted to µm "
            "before use).\n\n"
            "A distance transform (using real voxel spacing) is computed "
            "around each segment's border; any other-cell-type segment "
            "found within this distance sets '{type}_contact_on_distance' "
            "= True and adds its TrackID to 'touching_{type}s'.\n\n"
            "This value also defines the neighbourhood used by the Organoid "
            "Invasiveness calculation and by the Active Killing contact/target "
            "matching.",
        )
        contact_row.addLayout(contact_help)
        self._contact_unit_mgr.register(
            self.contact_threshold, "distance",
            float(fcfg.get("contact_threshold", 0.0)), native_unit="physical",
        )

        # Group the optional 'Organoid Invasiveness' checkbox inline with the
        # contact row (immune cells only, mirroring the notebook widget).
        if self.category == "immune" and "invasiveness" in self.feature_checks:
            inv_cb = self.feature_checks["invasiveness"]
            inv_cb.setToolTip(
                "Count how many immune voxels lie inside organoid masks at each "
                "timepoint. Requires both immune segments and organoid masks; "
                "automatically aggregated downstream."
            )
            contact_row.addWidget(inv_cb)
            contact_row.addWidget(HelpButton(
                "Organoid Invasiveness (Advanced)",
                "Per-timepoint measure of how much of this immune cell's surface\n"
                "is embedded in an organoid, not a voxel count inside the mask.\n\n"
                "'Surface' pixels are those within 2 µm of the immune cell's\n"
                "segment boundary. Of those, the fraction that also lies within\n"
                "the Contact Threshold distance of an organoid segment gives\n"
                "'{organoid}_invasiveness_perc' (0-100%).\n\n"
                "'{organoid}_invasiveness' (bool) is True when that percentage is\n"
                "≥ 50%. 'any_org_invasiveness[_perc]' aggregates across all\n"
                "organoid types (max / any).\n\n"
                "Requires both immune segments and organoid masks; only available "
                "when Contact is enabled."
            ))
        contact_row.addStretch()

        def _toggle_ct(state=None):
            contact_on = self.feature_checks.get("contact", QCheckBox()).isChecked()
            self.contact_threshold.setEnabled(contact_on)
            # Invasiveness depends on contact (notebook parity).
            if self.category == "immune" and "invasiveness" in self.feature_checks:
                self.feature_checks["invasiveness"].setVisible(contact_on)

        if "contact" in self.feature_checks:
            self.feature_checks["contact"].stateChanged.connect(_toggle_ct)

        # Contact (with its unit toggle) sits at the top of the box; the
        # remaining checkboxes follow below in their usual order.
        feat_lay.addLayout(contact_row)
        # Run only after contact_row is parented, so setVisible() inside
        # _toggle_ct() doesn't hit a still-unparented (top-level) checkbox.
        _toggle_ct()

        for f in all_feats:
            if f in ("contact", "invasiveness"):
                # Contact was already added above; invasiveness is rendered
                # inline within contact_row (or skipped for non-immune types).
                continue
            elif f in self.feature_checks:
                feat_lay.addWidget(self.feature_checks[f])

        layout.addWidget(feat_group)

        # ── Death Threshold ───────────────────────────────────────────────

        dead_group = QGroupBox("Death Threshold")
        dead_lay = QVBoxLayout(dead_group)
        dead_lay.setSpacing(4)

        # -- Death threshold spinner (ALL panels own one; organoid panels sync)
        dead_desc = QLabel(
            f"Percentage of dead-mask pixels overlapping a segment\n"
            f"above which {self.cell_type} cells are classified as 'dead'.\n"
            f"Shown here as a %, saved to config as a fraction (e.g. 3 % → 0.03)."
        )
        dead_desc.setWordWrap(True)
        dead_desc.setMinimumWidth(0)
        dead_desc.setStyleSheet("color: #888; font-size: 10px;")
        dead_lay.addWidget(dead_desc)

        if self._is_organoid:
            shared_note = QLabel(
                "\u26a0\ufe0f  One threshold applies to ALL organoid types equally.\n"
                "Adjusting it here updates every other organoid tab in real time."
            )
            shared_note.setWordWrap(True)
            shared_note.setMinimumWidth(0)
            shared_note.setStyleSheet(
                "color: #FFB74D; font-style: italic; font-size: 10px;"
            )
            dead_lay.addWidget(shared_note)

        dead_form = QFormLayout()
        dead_form.setContentsMargins(0, 0, 0, 0)
        dead_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        self.spin_dead_threshold = QDoubleSpinBox()
        self.spin_dead_threshold.setRange(0.0, 100.0)
        self.spin_dead_threshold.setSingleStep(0.1)
        self.spin_dead_threshold.setDecimals(1)
        # ``dead_mask_percentage_threshold`` is persisted as a FRACTION
        # (0.0-1.0), matching the scale of the ``percentage_dead_mask``
        # feature it's thresholded against. The spinbox only displays it
        # as a percentage for readability, so convert on load/save.
        saved_thr = fcfg.get("dead_mask_percentage_threshold", 0.1)
        self.spin_dead_threshold.setValue(float(saved_thr) * 100.0 if saved_thr else 10.0)
        self.spin_dead_threshold.setSuffix(" %")
        self.spin_dead_threshold.setMaximumWidth(100)

        # Track the threshold that is currently baked into the combined CSV
        # on disk, as a FRACTION (same scale as the persisted config value).
        # ``None`` means feature extraction has never run for this
        # cell type (or no death classification was performed). Updated by
        # ``_persist()`` and by ``_on_rerun_death_clicked``.
        self._last_persisted_threshold = (
            float(saved_thr) if saved_thr not in (None, 0, 0.0) else None
        )

        # "Re-run death features" button — always visible when a combined
        # feature CSV exists; dimmed when the spinner value matches the
        # persisted threshold. Placed to the right of the spinner+help row.
        self.btn_rerun_death = QPushButton("\U0001F504  Re-run death")
        self.btn_rerun_death.setMaximumWidth(150)
        self.btn_rerun_death.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; padding: 4px 8px; "
            "border-radius: 3px; font-size: 11px; font-weight: bold; } "
            "QPushButton:disabled { background: #455A64; color: #B0BEC5; } "
            "QPushButton:hover:!disabled { background: #1565C0; }"
        )
        self.btn_rerun_death.clicked.connect(self._on_rerun_death_clicked)

        spinner_row = make_help_row(
            self.spin_dead_threshold,
            "Dead Mask Percentage Threshold",
            "Percentage of dead-mask pixels overlapping a segment's volume\n"
            "required to classify the cell as dead.\n\n"
            "By default, classification is sticky: once a track crosses this\n"
            "threshold at any timepoint, it is marked 'dead' from that timepoint\n"
            "onward for the rest of the track, even if the percentage later\n"
            "drops again. This can be turned off in Advanced Configuration\n"
            "below ('Propagate dead signal').\n\n"
            "Set to 0 to skip dead classification.\n"
            "Entered as a percentage here; saved to behav3d_parameters.yml\n"
            "as a fraction (e.g. 3 % is saved as 0.03).",
        )
        spinner_row.addWidget(self.btn_rerun_death)
        dead_form.addRow("Dead mask % threshold:", spinner_row)
        dead_lay.addLayout(dead_form)

        # Keep button state in sync with the spinner value.
        self.spin_dead_threshold.valueChanged.connect(
            lambda _v: self._refresh_rerun_death_button()
        )

        # Per-panel sample selector
        prev_sample_row_layout = QHBoxLayout()
        prev_sample_row_layout.addWidget(QLabel("Preview sample:"))
        self.preview_sample_combo = QComboBox()
        self.preview_sample_combo.setMinimumWidth(180)
        # Long sample names must not be allowed to size this combo (and, via
        # the enclosing QTabWidget which sizes to the widest tab across ALL
        # cell types, the entire Feature Extraction panel) to fit the widest
        # item unelided — cap it and rely on the tooltip/elided text instead.
        self.preview_sample_combo.setMaximumWidth(240)
        self.preview_sample_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.preview_sample_combo.setMinimumContentsLength(18)
        self.preview_sample_combo.setToolTip(
            "Select which sample to load for the dead threshold preview."
        )
        prev_sample_row_layout.addWidget(self.preview_sample_combo, stretch=1)
        dead_lay.addLayout(prev_sample_row_layout)

        # Populate combo
        _md = self.metadata_loader.metadata if self.metadata_loader else None
        if _md is not None and not _md.empty:
            _samples = sorted(str(s) for s in _md["sample_name"].unique())
            self.preview_sample_combo.addItems(_samples)

        btn_preview = QPushButton("\U0001F441  Preview Dead Threshold in Viewer")
        btn_preview.setStyleSheet(
            "QPushButton { background: #37474F; color: white; padding: 5px 10px; "
            "border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background: #546E7A; }"
        )
        btn_preview.setToolTip(
            f"Load raw channels + dead mask for {self.cell_type} and show\n"
            "a green (alive) / red (dead) per-cell overlay.\n"
            "Adjust the threshold above to see live changes."
        )
        btn_preview.clicked.connect(self._on_preview_dead_clicked)
        dead_lay.addWidget(btn_preview)

        # Legend for the dead/alive overlay colours (matches _dead_alive_colormap).
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(2, 0, 2, 0)
        legend_row.setSpacing(12)
        legend_label = QLabel(
            "<span style='color:#00cc00;'>■</span> Alive"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#e60000;'>■</span> Dead"
        )
        legend_label.setToolTip("Overlay colours in the Dead Threshold preview.")
        legend_label.setStyleSheet("font-size: 11px;")
        legend_row.addWidget(legend_label)
        legend_row.addStretch(1)
        dead_lay.addLayout(legend_row)

        # Organoid spinners notify parent tab -> all org panels stay in sync
        if self._is_organoid:
            def _org_sync(value, _ct=self.cell_type):
                pt = self.parent()
                while pt and not hasattr(pt, '_notify_organoid_threshold_changed'):
                    pt = pt.parent()
                if pt:
                    pt._notify_organoid_threshold_changed(_ct, value)
            self.spin_dead_threshold.valueChanged.connect(_org_sync)
        layout.addWidget(dead_group)
        dead_group.setVisible(self._has_dead)
        self._dead_group = dead_group

        # ── Advanced Configuration (death propagation) ──────────────────────
        adv_dead_section = CollapsibleSection("⚙ Advanced Configuration", expanded=False)
        self.check_propagate_dead = QCheckBox("Propagate dead signal")
        self.check_propagate_dead.setChecked(
            bool(fcfg.get("propagate_dead_signal", self.category == "organoid"))
        )
        adv_dead_row = make_help_row(
            self.check_propagate_dead,
            "Propagate Dead Signal",
            "ON: once a track crosses the dead threshold at any timepoint, it\n"
            "stays 'dead' for every later timepoint, even if the signal drops\n"
            "again afterwards. Recommended for organoids - a dead organoid\n"
            "doesn't come back to life.\n\n"
            "OFF: only the timepoints where the signal is actually at/above\n"
            "threshold are marked 'dead'; the track can go back to 'alive'\n"
            "afterwards. Recommended for immune/other cells, which can\n"
            "transiently pick up dead dye while killing a dying organoid\n"
            "without actually dying themselves.",
        )
        adv_dead_section.addLayout(adv_dead_row)
        layout.addWidget(adv_dead_section)
        adv_dead_section.setVisible(self._has_dead)
        self._adv_dead_section = adv_dead_section

        def _toggle_dead_group(state=None):
            cb = self.feature_checks.get("death")
            visible = self._has_dead and (cb is not None and cb.isChecked())
            self._dead_group.setVisible(visible)
            self._adv_dead_section.setVisible(visible)

        if "death" in self.feature_checks:
            self.feature_checks["death"].stateChanged.connect(_toggle_dead_group)
        _toggle_dead_group()

        # ── Workers ───────────────────────────────────────────────────────
        n_cores = os.cpu_count() or 4
        max_allowed = max(1, n_cores - 1)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, max_allowed)
        default_workers = min(int(fcfg.get("n_workers", max(4, n_cores // 2))), max_allowed)
        self.spin_workers.setValue(default_workers)
        self.spin_workers.setMaximumWidth(60)
        self.spin_workers.valueChanged.connect(self._on_workers_changed)

        workers_form = QFormLayout()
        workers_form.setContentsMargins(0, 0, 0, 0)
        workers_form.addRow(
            "Workers:",
            make_help_row(
                self.spin_workers,
                "Number of Workers",
                f"Number of CPU cores to use for parallel processing.\n\n"
                f"Your machine has {n_cores} cores.\n"
                f"Recommendation: Use at most {max(1, n_cores - 1)} cores to keep the system responsive.",
            ),
        )
        layout.addLayout(workers_form)

        # ── Sync-to-others buttons ────────────────────────────────────────
        cat_label = (
            self.category.capitalize() + "s" if self.category != "other" else "Other types"
        )
        sync_row = QHBoxLayout()
        btn_apply_cat = QPushButton(f"Apply to all {cat_label}")
        btn_apply_cat.clicked.connect(lambda: self._apply_to_others(category_only=True))
        btn_apply_all = QPushButton("Apply to all")
        btn_apply_all.clicked.connect(lambda: self._apply_to_others(category_only=False))
        sync_row.addWidget(btn_apply_cat)
        sync_row.addWidget(btn_apply_all)
        layout.addLayout(sync_row)

        # ── Run button ────────────────────────────────────────────────────
        self.btn_run = QPushButton(
            f"Run {self.cell_type.capitalize()} Feature Extraction"
        )
        self.btn_run.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        # ``clicked`` emits a ``bool`` (checked state) that would override the
        # ``interactive=True`` default of ``_on_run_clicked``; wrap in a lambda
        # so the overwrite prompt is not silently skipped on button presses.
        self.btn_run.clicked.connect(lambda: self._on_run_clicked(interactive=True))
        layout.addWidget(self.btn_run)

        layout.addStretch()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _get_threshold(self) -> float:
        """Return the effective dead-mask threshold as shown in the UI (percent, 0-100)."""
        if self.spin_dead_threshold is not None:
            return round(float(self.spin_dead_threshold.value()), 2)
        if self._threshold_getter is not None:
            return round(float(self._threshold_getter()), 2)
        return 0.0

    def _threshold_as_fraction(self, pct: float | None = None) -> float:
        """Convert a UI percent-scale threshold to the fraction (0.0-1.0)
        that ``percentage_dead_mask`` is measured in. This is the scale
        persisted to config and passed into ``calculate_death``/
        ``rerun_death_classification``."""
        value = self._get_threshold() if pct is None else float(pct)
        return round(value / 100.0, 4)

    def _disconnect_preview_dead_hover(self):
        """Remove this panel's own previously-attached hover callback, if any."""
        if self.viewer is None or self._preview_hover_callback is None:
            self._preview_hover_callback = None
            return
        try:
            self.viewer.mouse_move_callbacks.remove(self._preview_hover_callback)
        except Exception:
            pass
        if _ACTIVE_PREVIEW_HOVER.get(id(self.viewer)) is self._preview_hover_callback:
            _ACTIVE_PREVIEW_HOVER.pop(id(self.viewer), None)
        self._preview_hover_callback = None

    @staticmethod
    def _disconnect_any_active_preview_hover(viewer):
        """Remove whichever preview-hover callback is currently active on
        ``viewer``, regardless of which panel instance attached it.

        The Dead/Alive preview overlay is a single shared layer; only one
        hover callback should ever be registered for it at a time (see
        ``_ACTIVE_PREVIEW_HOVER`` docstring above).
        """
        if viewer is None:
            return
        cb = _ACTIVE_PREVIEW_HOVER.pop(id(viewer), None)
        if cb is None:
            return
        try:
            viewer.mouse_move_callbacks.remove(cb)
        except Exception:
            pass

    def _disconnect_preview_dims(self):
        """Disconnect the napari time-slider listener if connected."""
        viewer = self.viewer
        if viewer is None or self._preview_dims_callback is None:
            self._preview_dims_callback = None
            return
        try:
            viewer.dims.events.current_step.disconnect(self._preview_dims_callback)
        except Exception:
            pass
        if _ACTIVE_PREVIEW_DIMS.get(id(viewer)) is self._preview_dims_callback:
            _ACTIVE_PREVIEW_DIMS.pop(id(viewer), None)
        unregister_preview_dims_listener(viewer, self)
        self._preview_dims_callback = None

    def _cleanup_preview(self):
        """Disconnect all preview callbacks and reset cached preview state.

        Does NOT remove layers — the caller handles that via
        ``clear_viewer_layers(self.viewer)``.
        """
        self._disconnect_preview_dead_hover()
        self._disconnect_preview_dims()
        self._preview_seg_t = None
        self._preview_dead_t = None
        self._preview_segs_dict = {}
        self._preview_immune_segs = {}
        self._preview_label_type_map = {}
        self._preview_overlay_arr = None
        self._preview_pct_overlay_arr = None
        self._preview_label_arr = None
        self._preview_current_frame = None
        self._preview_pct_maps_by_frame = {}
        self._preview_stats_by_frame = {}
        self._preview_computed_frames = set()
        self._preview_pct_computed_frames = set()
        self._preview_current_thr = None
        self._preview_is_timelapse = False

    def _connect_preview_dims(self):
        """Connect a viewer.dims.current_step listener that triggers an
        on-demand recomputation for whatever frame the slider moved to.

        Safe to call multiple times — any previous listener is dropped first.
        """
        viewer = self.viewer
        if viewer is None:
            return
        self._disconnect_preview_dims()
        _disconnect_any_active_preview_dims(viewer)

        def _on_step(*_):
            # Only act if we still have valid preview state
            if (
                self._preview_seg_t is None
                or self._preview_dead_t is None
                or not self._preview_is_timelapse
            ):
                return
            try:
                self._refresh_preview_dead_layers(value=None)
            except Exception as exc:
                # A failed per-frame recompute otherwise leaves
                # ``_preview_current_frame`` silently stuck on the last
                # successfully-computed frame, which makes the hover
                # tooltip's frame-mismatch guard suppress every hover on
                # this frame with no visible cause.
                self.log(f"⚠️ [Preview] Failed to recompute frame: {exc}")

        try:
            viewer.dims.events.current_step.connect(_on_step)
            self._preview_dims_callback = _on_step
            _ACTIVE_PREVIEW_DIMS[id(viewer)] = _on_step
            register_preview_dims_listener(viewer, self, _on_step)
        except Exception:
            self._preview_dims_callback = None

    def _attach_preview_dead_hover(self, layer_name: str = f"{_PREVIEW_PREFIX} Dead/Alive"):
        viewer = self.viewer
        # Drop this panel's own previous callback *and* whichever callback
        # (from any other organoid panel sharing this layer) is currently
        # active, so at most one hover callback is ever registered.
        self._disconnect_preview_dead_hover()
        self._disconnect_any_active_preview_hover(viewer)
        if (
            viewer is None
            or self._preview_seg_t is None
            or not self._preview_pct_maps_by_frame
        ):
            return
        try:
            viewer.layers[layer_name]
        except Exception:
            return

        def _show_tooltip(text: str):
            try:
                QToolTip.showText(QCursor.pos(), text, self.viewer.window._qt_window)
            except Exception:
                try:
                    QToolTip.showText(QCursor.pos(), text)
                except Exception:
                    pass

        def _hide_tooltip():
            try:
                QToolTip.hideText()
            except Exception:
                pass

        def _values_at_cursor(position):
            """Single coordinate lookup shared by the label id and the dead
            percentage: both ``_preview_label_arr`` and
            ``_preview_pct_overlay_arr`` are the same-shaped, same-frame
            arrays produced together by ``_overlay_for_volume``, so one
            ``world_to_data`` transform (via the Dead/Alive layer) is enough
            to index both. Returns ``(label_id, pct_dead)``, with either
            element ``None`` when unavailable.
            """
            label_arr = self._preview_label_arr
            pct_arr = self._preview_pct_overlay_arr
            if label_arr is None and pct_arr is None:
                return None, None
            try:
                da_layer = self.viewer.layers[layer_name]
                ref_arr = label_arr if label_arr is not None else pct_arr
                data_pos = da_layer.world_to_data(position)
                coords = tuple(int(round(float(c))) for c in data_pos[-ref_arr.ndim:])
                for i, c in enumerate(coords):
                    if c < 0 or c >= ref_arr.shape[i]:
                        return None, None
                label_id = int(label_arr[coords]) if label_arr is not None else None
                pct_dead = float(pct_arr[coords]) if pct_arr is not None else None
                return label_id, pct_dead
            except Exception as exc:
                # Throttled: mouse-move fires very frequently, so only log
                # when the failure reason actually changes, instead of
                # silently swallowing every occurrence (which previously
                # made "hover sometimes shows nothing" undiagnosable).
                msg = str(exc)
                if getattr(self, "_last_hover_error_msg", None) != msg:
                    self._last_hover_error_msg = msg
                    self.log(f"⚠️ [Preview] Hover lookup failed: {msg}")
                return None, None

        def _on_mouse_move(*args):
            """Works with both napari ≤0.4.18 (event,) and ≥0.4.19 (viewer,event)."""
            if self.viewer is None:
                return
            position = getattr(self.viewer.cursor, "position", None)
            if position is None:
                self.viewer.status = ""
                _hide_tooltip()
                return

            frame_idx = self._current_viewer_frame() if self._preview_is_timelapse else 0
            if frame_idx != self._preview_current_frame:
                self.viewer.status = ""
                _hide_tooltip()
                return

            label_id, pct_dead = _values_at_cursor(position)
            if not label_id:
                self.viewer.status = ""
                _hide_tooltip()
                return

            # Resolve cell type + original label from the (merged organoid,
            # when applicable) label-type map captured at attach time.
            _label_type_map = getattr(self, "_preview_label_type_map", {})
            type_info = _label_type_map.get(label_id)
            if type_info is not None:
                ct_name, orig_lbl = type_info
                cell_label = f"{ct_name} #{orig_lbl}"
            else:
                cell_label = f"Cell #{label_id}"
            status_text = (
                f"{cell_label} | {pct_dead:.2f}% dead" if pct_dead is not None else cell_label
            )
            self.viewer.status = status_text
            _show_tooltip(status_text)

        self._preview_hover_callback = _on_mouse_move
        _ACTIVE_PREVIEW_HOVER[id(viewer)] = _on_mouse_move
        # viewer.mouse_move_callbacks is the napari-version-agnostic API
        viewer.mouse_move_callbacks.append(_on_mouse_move)

    def _current_viewer_frame(self) -> int:
        """Return the napari viewer's current time-axis index (0 if no time)."""
        viewer = self.viewer
        if viewer is None:
            return 0
        try:
            return int(viewer.dims.current_step[0])
        except Exception:
            return 0

    def _invalidate_preview_cache(self):
        """Drop cached Dead/Alive overlays on threshold change.

        The ``% Dead Mask`` layer is threshold-independent so its cached frames
        (``_preview_pct_computed_frames``) are preserved.
        """
        self._preview_computed_frames.clear()
        self._preview_pct_maps_by_frame.clear()
        self._preview_stats_by_frame.clear()
        self._preview_overlay_arr = None
        self._preview_label_arr = None
        self._preview_current_frame = None

    def _materialize_frame(self, frame_idx):
        """Materialize segment + dead mask for a single timepoint.

        Handles per-frame immune cleaning and organoid merging so that the
        caller receives ready-to-use 3-D numpy arrays.

        Returns ``(seg_vol, dead_vol, label_type_map)``.
        """
        dead_data = self._preview_dead_t
        immune_segs = getattr(self, "_preview_immune_segs", {})

        # Dead mask — materialize single frame + clean
        dead_vol = np.asarray(dead_data[frame_idx])
        if dead_vol.ndim > 3:
            dead_vol = dead_vol[0]
        if immune_segs:
            dead_vol = _clean_dead_frame(dead_vol, immune_segs, frame_idx)

        # Segments — per-frame organoid merge or direct frame extraction
        segs_dict = getattr(self, "_preview_segs_dict", None)
        label_type_map = {}
        if self._is_organoid and self._org_cell_types and segs_dict:
            seg_vol, label_type_map = _merge_org_segments_frame(
                segs_dict, self._org_cell_types, frame_idx
            )
            if seg_vol is None:
                seg_vol = np.asarray(self._preview_seg_t[frame_idx])
        else:
            seg_vol = np.asarray(self._preview_seg_t[frame_idx])

        if seg_vol.ndim > 3:
            seg_vol = seg_vol[0]
        if dead_vol.ndim < seg_vol.ndim:
            dead_vol = np.broadcast_to(dead_vol, seg_vol.shape)
        return seg_vol, dead_vol, label_type_map

    def _refresh_preview_dead_layers(
        self,
        value: float | None = None,
        frame_idx: int | None = None,
    ):
        """Compute (or refresh) the Dead/Alive overlay for a single timepoint.

        Data is materialized on-demand: only the requested frame is loaded
        from the lazy (dask) source arrays.
        """
        viewer = self.viewer
        if (
            viewer is None
            or self._preview_seg_t is None
            or self._preview_dead_t is None
        ):
            return

        thr = self._get_threshold() if value is None else round(float(value), 2)

        # Threshold changed -> invalidate Dead/Alive cache (pct is kept)
        if self._preview_current_thr is None or thr != self._preview_current_thr:
            self._invalidate_preview_cache()
            self._preview_current_thr = thr

        if frame_idx is None:
            frame_idx = self._current_viewer_frame() if self._preview_is_timelapse else 0
        frame_idx = int(frame_idx)

        # Fast path: same frame + same threshold → just refresh layers
        if (
            frame_idx == self._preview_current_frame
            and frame_idx in self._preview_computed_frames
        ):
            try:
                viewer.layers[f"{_PREVIEW_PREFIX} Dead/Alive"].refresh()
            except Exception:
                pass
            try:
                viewer.layers[f"{_PREVIEW_PREFIX} % Dead Mask"].refresh()
            except Exception:
                pass
            self._attach_preview_dead_hover()
            return

        import datetime
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Preview] "
            f"Refreshing Dead/Alive overlay — threshold={thr:.2f}% "
            f"(frame={frame_idx}, {self.cell_type})"
        )

        # Materialize only this frame (lazy → numpy, clean, merge)
        seg_vol, dead_vol, label_type_map = self._materialize_frame(frame_idx)
        if label_type_map:
            self._preview_label_type_map = label_type_map

        overlay_t, pct_t, frame_stats = _overlay_for_volume(
            seg_vol, dead_vol, thr,
            frame_label=f" t={frame_idx}", log_fn=self.log,
        )

        # Store single-frame (Z,Y,X) arrays and replace layer data. ``seg_vol``
        # is stored verbatim as the label-id array for hover lookups — it is
        # already the (merged, when applicable) array used to build both
        # ``overlay_t``/``pct_t`` and ``label_type_map`` above, so it is
        # guaranteed consistent with them.
        self._preview_overlay_arr = overlay_t
        self._preview_pct_overlay_arr = pct_t
        self._preview_label_arr = seg_vol
        self._preview_current_frame = frame_idx

        cmap = _dead_alive_colormap()
        da_name = f"{_PREVIEW_PREFIX} Dead/Alive"
        try:
            layer = viewer.layers[da_name]
            layer.data = overlay_t
            layer.opacity = 1.0
            if cmap is not None:
                layer.colormap = cmap
            else:
                _apply_dead_alive_colors(layer)
            layer.refresh()
        except (KeyError, ValueError):
            kw = dict(name=da_name, opacity=1.0)
            if cmap is not None:
                kw["colormap"] = cmap
            layer = viewer.add_labels(overlay_t, **kw)
            if cmap is None:
                _apply_dead_alive_colors(layer)

        pct_name = f"{_PREVIEW_PREFIX} % Dead Mask"
        try:
            pct_layer = viewer.layers[pct_name]
            pct_layer.data = pct_t
            pct_layer.refresh()
        except (KeyError, ValueError):
            viewer.add_image(
                pct_t,
                name=pct_name,
                colormap="inferno",
                contrast_limits=(0, 100),
                blending="translucent",
                opacity=0.7,
                visible=False,
            )

        self._preview_stats_by_frame[frame_idx] = frame_stats
        self._preview_pct_maps_by_frame[frame_idx] = _build_dead_pct_map(frame_stats)
        self._preview_computed_frames.add(frame_idx)
        self._preview_pct_computed_frames.add(frame_idx)
        self._attach_preview_dead_hover()

        if self._is_organoid and self._org_preview_cache is not None:
            self._org_preview_cache["overlay_arr"] = self._preview_overlay_arr
            self._org_preview_cache["pct_overlay_arr"] = self._preview_pct_overlay_arr
            self._org_preview_cache["label_arr"] = self._preview_label_arr
            self._org_preview_cache["stats_by_frame"] = self._preview_stats_by_frame
            self._org_preview_cache["pct_maps_by_frame"] = self._preview_pct_maps_by_frame
            self._org_preview_cache["computed_frames"] = self._preview_computed_frames
            self._org_preview_cache["current_thr"] = thr

    def _on_workers_changed(self, value):
        n_cores = os.cpu_count() or 4
        max_allowed = max(1, n_cores - 1)
        if value > max_allowed:
            self.spin_workers.setValue(max_allowed)
            self.log(
                f"⚠️ Workers capped to {max_allowed} (system has {n_cores} cores). "
                "Using all cores can freeze the system."
            )

    def _selected_features(self) -> list:
        result = [f for f, cb in self.feature_checks.items() if cb.isChecked()]
        # Defensive: mandatory features are always included regardless of checkbox state
        for _mf in getattr(self, "_mandatory_features", set()):
            if _mf in self.feature_checks and _mf not in result:
                result.append(_mf)
        return result

    def _collect_params(self) -> dict:
        thr = self._get_threshold()
        return {
            "features_choice": self._selected_features(),
            "contact_threshold": float(self._contact_unit_mgr.get_native(self.contact_threshold)),
            # Persisted/consumed as a fraction (0.0-1.0); ``thr`` is the
            # percent-scale value shown in the spinbox.
            "dead_mask_percentage_threshold": self._threshold_as_fraction(thr) if thr > 0 else None,
            "propagate_dead_signal": bool(self.check_propagate_dead.isChecked()),
            "n_workers": int(self.spin_workers.value()),
        }

    def _apply_to_others(self, category_only=False):
        parent_tab = self.parent()
        while parent_tab and not hasattr(parent_tab, "panels"):
            parent_tab = parent_tab.parent()
        if not parent_tab:
            return

        settings = self._collect_params()
        targets = self.category_types if category_only else self.all_cell_types
        count = 0
        for ct in targets:
            if ct == self.cell_type:
                continue
            if ct in parent_tab.panels:
                p = parent_tab.panels[ct]
                for f, cb in p.feature_checks.items():
                    cb.setChecked(f in settings["features_choice"])
                p._contact_unit_mgr.set_native(p.contact_threshold, settings["contact_threshold"])
                p.spin_workers.setValue(settings["n_workers"])
                # Only sync dead threshold for non-organoid panels that own a spinner.
                # Use the percent-scale UI value directly (not the fraction
                # stored in ``settings``, which is scaled for persistence).
                if not p._is_organoid and p.spin_dead_threshold is not None:
                    p.spin_dead_threshold.setValue(self._get_threshold())
                p.check_propagate_dead.setChecked(settings["propagate_dead_signal"])
                count += 1
        scope = "category" if category_only else "all"
        self.log(f"Applied feature settings to {count} other cell type(s) ({scope}).")

    def _persist(self):
        params = self.metadata_loader.behav3d_parameters
        features = params.setdefault("features", {})
        features[self.cell_type] = self._collect_params()

        out_dir = self.metadata_loader.output_dir
        if out_dir:
            params_path = Path(out_dir) / "behav3d_parameters.yml"
            try:
                with open(params_path, "w") as f:
                    yaml.safe_dump(params, f, sort_keys=False)
            except Exception as e:
                self.log(f"Warning: Could not save parameters: {e}")

        # Remember the threshold value that is now baked into the on-disk
        # config (and, after run, into the combined CSV), as a fraction
        # (same scale as the persisted config value).
        thr = self._get_threshold()
        self._last_persisted_threshold = self._threshold_as_fraction(thr) if thr > 0 else None
        self._refresh_rerun_death_button()

    def _combined_csv_path(self, cell_type: str | None = None) -> Path:
        """Path to the combined features CSV for a given cell type."""
        ct = cell_type or self.cell_type
        return (
            Path(self.metadata_loader.output_dir)
            / "analysis"
            / ct
            / "track_features"
            / f"BEHAV3D_{ct}_combined_track_features.csv"
        )

    def _has_combined_csv(self) -> bool:
        try:
            return self._combined_csv_path().exists()
        except Exception:
            return False

    def _threshold_changed(self) -> bool:
        """Return True when current spinner value differs from the persisted one."""
        if not self._has_combined_csv():
            return False
        try:
            # Compare on the same (fraction) scale as ``_last_persisted_threshold``.
            current = self._threshold_as_fraction()
        except Exception:
            return False
        persisted = self._last_persisted_threshold
        if persisted is None:
            return current > 0
        return abs(current - round(float(persisted), 4)) > 1e-6

    def _refresh_rerun_death_button(self):
        """Show/hide and enable/disable the Re-run death button per current state."""
        if not hasattr(self, "btn_rerun_death") or self.btn_rerun_death is None:
            return
        if not self._has_dead:
            self.btn_rerun_death.setVisible(False)
            return
        has_csv = self._has_combined_csv()
        self.btn_rerun_death.setVisible(has_csv)
        if not has_csv:
            return
        changed = self._threshold_changed()
        self.btn_rerun_death.setEnabled(changed)
        if changed:
            self.btn_rerun_death.setToolTip(
                "Recompute the 'dead' column using the new threshold "
                "without re-extracting all features."
            )
        else:
            self.btn_rerun_death.setToolTip(
                "Current threshold matches the value used in the existing "
                "features. Nothing to recompute."
            )

    def _on_rerun_death_clicked(self):
        from behav3d.features.timepoint_features import rerun_death_classification

        new_thr = self._get_threshold()
        if new_thr <= 0:
            self.log(
                "\u26a0\ufe0f Dead threshold is 0 \u2014 set a positive value before re-running."
            )
            return

        # Organoid panels share a single global threshold (kept in sync via
        # _notify_organoid_threshold_changed). Re-running death for only the
        # panel that was clicked would leave sibling organoid types'
        # combined CSVs classified under a different, now-stale threshold
        # even though the UI shows one shared value everywhere. Re-run for
        # every organoid type at once in that case.
        targets = [self.cell_type]
        panels_by_type = {self.cell_type: self}
        if self._is_organoid:
            parent_tab = self.parent()
            while parent_tab and not hasattr(parent_tab, "panels"):
                parent_tab = parent_tab.parent()
            if parent_tab is not None:
                org_types = list(getattr(parent_tab, "_org_types", []))
                found = {
                    ct: parent_tab.panels[ct]
                    for ct in org_types
                    if ct in parent_tab.panels
                }
                if found:
                    panels_by_type = found
                    panels_by_type.setdefault(self.cell_type, self)
                    targets = list(panels_by_type.keys())

        ran_any = False
        for ct in targets:
            panel = panels_by_type[ct]
            if not panel._has_combined_csv():
                self.log(
                    f"\u26a0\ufe0f No combined features CSV for {ct} \u2014 "
                    "run full feature extraction first."
                )
                continue
            try:
                rerun_death_classification(
                    output_dir=str(Path(self.metadata_loader.output_dir).expanduser()),
                    cell_type=ct,
                    new_threshold=self._threshold_as_fraction(new_thr),
                    propagate=bool(panel.check_propagate_dead.isChecked()),
                )
                panel._persist()
                self.log(
                    f"\u2705 Re-ran death classification for {ct} "
                    f"with threshold={new_thr}%."
                )
                ran_any = True
            except Exception as e:
                traceback.print_exc()
                self.log(f"Error during death re-run for {ct}: {e}")

        if ran_any:
            _notify_post_extraction(self)

    def _run_feature_extraction_for(self, cell_type: str, overwrite: bool = False,
                                    params: dict = None, progress_cb=None):
        """Run feature extraction for a single cell type.

        ``params`` is an optional Qt-thread-safe snapshot from
        :meth:`_collect_params`; when supplied the widget values are not
        re-read (required when called from a background worker).
        ``progress_cb`` is forwarded to ``run_feature_extraction``.
        """
        from behav3d.features.timepoint_features import run_feature_extraction

        out_dir = str(Path(self.metadata_loader.output_dir).expanduser())
        if params is None:
            params = self._collect_params()

        run_feature_extraction(
            metadata=self.metadata_loader.metadata,
            output_dir=out_dir,
            cell_type=cell_type,
            features_choice=list(params["features_choice"]),
            contact_threshold=float(params["contact_threshold"]),
            dead_mask_percentage_threshold=params["dead_mask_percentage_threshold"],
            propagate_dead_signal=bool(params.get("propagate_dead_signal", True)),
            n_workers=int(params["n_workers"]),
            overwrite=overwrite,
            progress_cb=progress_cb,
        )

    def _run_death_only_for(self, cell_type: str, params: dict = None):
        """Re-apply only the death classification without recomputing features.

        ``params`` is an optional snapshot from :meth:`_collect_params`;
        when supplied the threshold widget is not re-read.
        """
        from behav3d.features.timepoint_features import rerun_death_classification

        out_dir = str(Path(self.metadata_loader.output_dir).expanduser())
        if params is not None:
            new_thr = params["dead_mask_percentage_threshold"] or 0.0
            propagate = bool(params.get("propagate_dead_signal", True))
        else:
            new_thr = self._threshold_as_fraction()
            propagate = bool(self.check_propagate_dead.isChecked())
        if new_thr <= 0:
            self.log(
                f"\u26a0\ufe0f Skipping death-only re-run for {cell_type}: "
                "threshold is 0."
            )
            return
        rerun_death_classification(
            output_dir=out_dir,
            cell_type=cell_type,
            new_threshold=float(new_thr),
            propagate=propagate,
        )

    def _check_existing_features(self, cell_types: list) -> list:
        warnings = []
        out_dir = Path(self.metadata_loader.output_dir)
        for ct in cell_types:
            feat_dir = out_dir / "analysis" / ct / "track_features"
            combined = feat_dir / f"BEHAV3D_{ct}_combined_track_features.csv"
            if combined.exists():
                warnings.append(f"{ct} feature data ({combined.name})")
        return warnings

    # ── Click handler ────────────────────────────────────────────────────────
    def _on_run_clicked(self, interactive=True):
        """Background-execute feature extraction for this cell type.

        Sample-level progress is forwarded from ``run_feature_extraction``.
        The death-only re-run path uses indeterminate progress because
        ``rerun_death_classification`` has no top-level sample loop.
        """
        from behav3d.napari._overwrite_prompt import prompt_overwrite_single
        from qtpy.QtWidgets import QMessageBox as _QMB

        if self._bg.is_running():
            self.log("⚠️ A feature-extraction run is already in progress for this panel.")
            return

        self.log(f"Running feature extraction for: {self.cell_type}")

        overwrite = False
        death_only = False
        existing = self._check_existing_features([self.cell_type])
        threshold_changed = self._threshold_changed()

        if existing:
            if interactive:
                extra = None
                if threshold_changed:
                    extra = [
                        (
                            "Re-run death only",
                            "death_only",
                            _QMB.ActionRole,
                        ),
                    ]
                choice = prompt_overwrite_single(
                    self,
                    "Overwrite Existing Features?",
                    existing,
                    extra_buttons=extra,
                )
                if choice == "cancel" or choice == "skip":
                    self.log(f"Feature extraction for {self.cell_type} cancelled.")
                    return
                if choice == "death_only":
                    death_only = True
                else:
                    overwrite = True
            else:
                overwrite = True

        # Persist + snapshot widget state on the Qt thread.
        self._persist()
        params_snapshot = self._collect_params()
        cell_type = self.cell_type

        if death_only:
            def _do_death_only(progress_cb=None):
                return self._run_death_only_for(cell_type, params=params_snapshot)

            def _on_done(_r):
                self.log(f"\u2705 {cell_type} death re-run finished.")
                _notify_post_extraction(self)
                notify_results_changed(self)

            def _on_failed(err: str):
                self.log(f"Error during death re-run: {err}")
                notify_results_changed(self)

            self._bg.run(
                fn=_do_death_only,
                desc=f"Death re-run \u2014 {cell_type}\u2026",
                progress_row=self.tab_progress_row,
                buttons=[self.btn_run],
                viewer=self.viewer,
                on_done=_on_done,
                on_failed=_on_failed,
                inject_progress=False,
                indeterminate=True,
            )
            return

        def _do_extraction(progress_cb=None):
            return self._run_feature_extraction_for(
                cell_type, overwrite=overwrite,
                params=params_snapshot, progress_cb=progress_cb,
            )

        def _on_done(_r):
            self.log(f"✅ {cell_type} feature extraction finished.")
            _notify_post_extraction(self)
            notify_results_changed(self)

        def _on_failed(err: str):
            self.log(f"Error during feature extraction: {err}")
            _QMB.critical(
                self, "Feature Extraction Error",
                f"Feature extraction for {cell_type} failed:\n\n{err}",
            )
            notify_results_changed(self)

        self._bg.run(
            fn=_do_extraction,
            desc=f"Feature extraction \u2014 {cell_type}\u2026",
            progress_row=self.tab_progress_row,
            buttons=[self.btn_run],
            viewer=self.viewer,
            on_done=_on_done,
            on_failed=_on_failed,
        )

    # ── Non-organoid Dead Threshold Preview ──────────────────────────────────
    # -- Dead Threshold Preview (all panel types) -------------------------
    def _on_preview_dead_clicked(self):
        """Load raw channels, dead mask, and ALL cell-type segments for the
        selected sample, then show a green/red dead-alive overlay.

        For organoid panels the overlay uses the shared threshold; adjusting
        the spinner in any organoid tab updates all organoid panels live.
        """
        try:
            viewer = self.viewer
            if viewer is None:
                self.log("\u26a0\ufe0f No viewer available for dead threshold preview.")
                return

            if viewer.layers:
                from qtpy.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "Warning",
                    "This will remove current layers in the viewer. Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            md = self.metadata_loader.metadata
            if md is None or md.empty:
                self.log("\u26a0\ufe0f No metadata loaded.")
                return

            # Sample selector
            sample_name = (
                self.preview_sample_combo.currentText()
                if hasattr(self, "preview_sample_combo")
                else ""
            )
            if not sample_name:
                self.log("\u26a0\ufe0f No sample selected in the preview combo.")
                return

            sample_rows = md[md["sample_name"] == sample_name]
            if sample_rows.empty:
                self.log(f"\u26a0\ufe0f Sample '{sample_name}' not in metadata.")
                return
            sample_row = sample_rows.iloc[0]

            output_dir = Path(self.metadata_loader.output_dir)
            self.log(f"Loading preview \u2014 {self.cell_type} / {sample_name}\u2026")

            import datetime
            def _ts():
                return datetime.datetime.now().strftime("%H:%M:%S")

            print(f"\n[{_ts()}] [Preview] {'='*46}")
            print(f"[{_ts()}] [Preview] Dead threshold preview: {self.cell_type} / {sample_name}")
            print(f"[{_ts()}] [Preview] {'='*46}")

            # Resolve dead mask (with diagnostic logging) \u2014 kept lazy (dask)
            print(f"[{_ts()}] [Preview] Step 1/5 \u2014 Resolving dead mask...")
            dead_arr, dead_method, tried_paths = _resolve_dead_mask(
                sample_row, output_dir, log_fn=self.log
            )
            if dead_arr is None:
                print(f"[{_ts()}] [Preview] \u274c Dead mask not found.")
                self.log(
                    "\u26a0\ufe0f Dead mask not found. Paths tried:\n"
                    + "\n".join(f"    {p}" for p in tried_paths)
                    + "\nRun dead mask segmentation first (Segmentation tab)."
                )
                return
            print(f"[{_ts()}] [Preview]   Dead mask loaded (method={dead_method}, shape={dead_arr.shape}).")
            if dead_method == "raw":
                self.log(
                    "\u2139\ufe0f Dead mask estimated from raw dead channel (Otsu). "
                    "Run dedicated dead mask segmentation for best results."
                )

            # Ensure dead mask is at least 4-D for consistent frame indexing
            import dask.array as da
            if dead_arr.ndim == 2:
                dead_arr = dead_arr[np.newaxis, np.newaxis, ...]
                if not isinstance(dead_arr, da.Array):
                    dead_arr = da.from_array(dead_arr)
            elif dead_arr.ndim == 3:
                dead_arr = dead_arr[np.newaxis, ...]
                if not isinstance(dead_arr, da.Array):
                    dead_arr = da.from_array(dead_arr)

            # Load raw image (all channels, lazy)
            print(f"[{_ts()}] [Preview] Step 2/5 \u2014 Loading raw image (lazy)...")
            raw_dask = _load_raw_dask(sample_row, output_dir)
            if raw_dask is not None:
                print(f"[{_ts()}] [Preview]   Raw image shape: {raw_dask.shape}")
            else:
                print(f"[{_ts()}] [Preview]   Raw image not found \u2014 skipping channel layers.")

            # Collect ALL available segments \u2014 lazy (dask) for per-frame access
            all_types = getattr(self, "all_cell_types", [self.cell_type])
            print(
                f"[{_ts()}] [Preview] Step 3/5 \u2014 Loading segments (lazy) for "
                f"{len(all_types)} type(s): {all_types}..."
            )
            segs_dict, seg_sources = _load_all_segments_for_sample(
                sample_row, all_types, output_dir, lazy=True
            )
            for ct, arr in segs_dict.items():
                print(f"[{_ts()}] [Preview]   {ct}: shape={arr.shape}, source={seg_sources.get(ct,'?')}")
            if self.cell_type not in segs_dict:
                print(f"[{_ts()}] [Preview] \u274c No segments for '{self.cell_type}' \u2014 aborting.")
                self.log(
                    f"\u26a0\ufe0f No segments found for '{self.cell_type}' "
                    f"in '{sample_name}'. Run segmentation first."
                )
                return

            # Build immune segments dict for per-frame dead mask cleaning.
            # Shares its detection logic with feature extraction via
            # resolve_immune_track_paths_for_sample (keyed off this specific
            # sample row, same as run_feature_extraction) so the preview and
            # the real pipeline can never disagree about which loaded types
            # count as immune.
            from behav3d.core.metadata import resolve_immune_track_paths_for_sample
            try:
                immune_track_paths = resolve_immune_track_paths_for_sample(sample_row)
            except Exception as exc:
                self.log(f"⚠️ [Preview] Immune-type detection failed: {exc}")
                immune_track_paths = {}
            if self.cell_type in immune_track_paths:
                # Mirrors exclude_immune_from_dead in timepoint_features.py:
                # don't clean a type's own dead mask under itself.
                immune_segs_for_cleaning = {}
            else:
                immune_segs_for_cleaning = {
                    ct: segs_dict[ct]
                    for ct in immune_track_paths
                    if ct in segs_dict
                }
            if immune_segs_for_cleaning:
                self.log(
                    f"  [Preview] Dead mask will be cleaned per-frame under immune "
                    f"segments {list(immune_segs_for_cleaning.keys())}"
                )
            elif immune_track_paths and self.cell_type not in immune_track_paths:
                missing = [ct for ct in immune_track_paths if ct not in segs_dict]
                self.log(
                    f"  ℹ️ [Preview] Detected immune type(s) {missing} in "
                    "metadata, but their segments weren't loaded for this "
                    "preview — dead mask will NOT be cleaned under them "
                    "(feature extraction will still clean under them if "
                    "loaded there)."
                )
            elif not immune_track_paths:
                self.log(
                    "  ℹ️ [Preview] No immune cell types detected for "
                    "this sample — dead mask will not be cleaned."
                )

            # Remove previous preview layers (preserves non-preview layers)
            print(f"[{_ts()}] [Preview] Clearing preview layers...")
            self._cleanup_preview()
            # Always tear down whichever dims listener — this panel's own,
            # another panel's, the shared organoid one, or a listener left
            # live by a different tab entirely (State/Track Classification,
            # Feature Backprojection) — is currently active before clearing
            # layers. A stale listener from whatever preview was loaded
            # previously would otherwise fire on the dims-range change
            # caused by clearing/re-adding layers below, recompute against
            # its now-stale cache, and either write into layers that were
            # just removed (crash) or silently clobber the new preview's
            # "Dead/Alive" / "% Dead Mask" layers.
            _disconnect_any_active_preview_dims(self.viewer)
            disconnect_all_preview_dims_listeners(self.viewer)
            # Stop any running dims animation before clearing layers so napari
            # doesn't try to use slider widgets it is about to destroy.
            from behav3d.napari._visualization import _is_addable_layer_data
            from behav3d.napari._preview_dims import stop_dim_playback
            stop_dim_playback(self.viewer)
            clear_viewer_layers(self.viewer)

            # Raw channels
            if raw_dask is not None:
                n_ch = raw_dask.shape[1] if raw_dask.ndim >= 5 else 1
                _add_channel_layers(viewer, raw_dask, sample_name)
                self.log(f"  Added {n_ch} raw channel(s).")
            else:
                self.log("  \u26a0\ufe0f Could not load raw image.")

            # Dead mask display layer (raw dask \u2014 napari renders lazily)
            if _is_addable_layer_data(dead_arr):
                viewer.add_image(
                    dead_arr,
                    name=f"{_PREVIEW_PREFIX} Dead Mask",
                    colormap="magenta",
                    blending="additive",
                    opacity=0.4,
                    visible=False,
                )
            else:
                self.log(
                    f"  \u26a0\ufe0f Skipping dead-mask display: unexpected/degenerate "
                    f"shape {getattr(dead_arr, 'shape', '?')}"
                )

            # All segment layers (this type full opacity, others lighter)
            primary_seg = segs_dict[self.cell_type]
            for ct, seg_arr in segs_dict.items():
                is_primary = (ct == self.cell_type)
                viewer.add_labels(
                    seg_arr,
                    name=f"{_PREVIEW_PREFIX} {ct} segments",
                    opacity=0.7 if is_primary else 0.3,
                    visible=False,
                )
            self.log(
                "  Segments: "
                + ", ".join(
                    (
                        f"{ct}[primary,{seg_sources.get(ct, 'untracked')}]"
                        if ct == self.cell_type
                        else f"{ct}[{seg_sources.get(ct, 'untracked')}]"
                    )
                    for ct in segs_dict
                )
            )

            # Store for per-frame organoid merge (no upfront materialization)
            if self._is_organoid and self._org_cell_types:
                org_names = [ct for ct in self._org_cell_types if ct in segs_dict]
                print(
                    f"[{_ts()}] [Preview] Step 4/5 \u2014 "
                    f"{len(org_names)} organoid type(s) will be merged per-frame."
                )
                self.log(
                    f"  Dead/Alive overlay merges {len(org_names)} organoid type(s): "
                    + ", ".join(org_names)
                )
            else:
                print(f"[{_ts()}] [Preview] Step 4/5 \u2014 Single type ({self.cell_type}), no merge needed.")
            self._preview_label_type_map = {}

            # Compute dead/alive overlay (on-demand: current frame only)
            thr = self._get_threshold()
            print(
                f"[{_ts()}] [Preview] Step 5/5 \u2014 Computing Dead/Alive overlay "
                f"on-demand for current frame (threshold={thr:.2f}%)..."
            )

            # Cache the lazy source arrays + per-frame cleaning info
            self._preview_seg_t = primary_seg
            self._preview_segs_dict = segs_dict
            self._preview_immune_segs = immune_segs_for_cleaning
            self._preview_dead_t = dead_arr
            self._preview_overlay_arr = None
            self._preview_pct_overlay_arr = None
            self._preview_label_arr = None
            self._preview_pct_maps_by_frame = {}
            self._preview_stats_by_frame = {}
            self._preview_computed_frames = set()
            self._preview_pct_computed_frames = set()
            self._preview_current_thr = None

            self._preview_is_timelapse = primary_seg.ndim >= 4

            # Tear down any previous time-slider listener and create the
            # layer/compute the current frame.
            self._disconnect_preview_dims()
            current_frame = self._current_viewer_frame() if self._preview_is_timelapse else 0
            self._refresh_preview_dead_layers(thr, frame_idx=current_frame)

            # Organoid panels fill the shared tab cache too
            if self._is_organoid and self._org_preview_cache is not None:
                self._org_preview_cache["seg_t"] = primary_seg
                self._org_preview_cache["segs_dict"] = segs_dict
                self._org_preview_cache["immune_segs"] = immune_segs_for_cleaning
                self._org_preview_cache["dead_t"] = dead_arr
                self._org_preview_cache["overlay_arr"] = self._preview_overlay_arr
                self._org_preview_cache["pct_overlay_arr"] = self._preview_pct_overlay_arr
                self._org_preview_cache["label_arr"] = self._preview_label_arr
                self._org_preview_cache["pct_maps_by_frame"] = self._preview_pct_maps_by_frame
                self._org_preview_cache["stats_by_frame"] = self._preview_stats_by_frame
                self._org_preview_cache["computed_frames"] = self._preview_computed_frames
                self._org_preview_cache["pct_computed_frames"] = self._preview_pct_computed_frames
                self._org_preview_cache["current_thr"] = self._preview_current_thr
                self._org_preview_cache["current_frame"] = self._preview_current_frame
                self._org_preview_cache["label_type_map"] = self._preview_label_type_map
                self._org_preview_cache["is_timelapse"] = self._preview_is_timelapse

            # Connect time-slider listener for on-demand frame recompute.
            # Organoid panels delegate the listener to FeatureExtractionTab
            # (a single shared listener covers all organoid panels).
            if self._preview_is_timelapse and not self._is_organoid:
                self._connect_preview_dims()
            elif self._is_organoid:
                pt = self.parent()
                while pt and not hasattr(pt, "_connect_org_preview_dims"):
                    pt = pt.parent()
                if pt is not None:
                    if self._preview_is_timelapse:
                        pt._connect_org_preview_dims()
                    # Mirror this panel's freshly-loaded preview state onto
                    # every other organoid panel so they share the same
                    # per-frame caches and hover tooltip data.
                    pt._propagate_org_preview_to_panels()

            # Wire non-organoid spinner -> live overlay (once)
            if not self._preview_connected and self.spin_dead_threshold is not None:
                if not self._is_organoid:
                    self.spin_dead_threshold.valueChanged.connect(
                        self._on_threshold_spin_changed
                    )
                self._preview_connected = True

            self.log(
                f"\u2705 Preview loaded (on-demand per-frame). "
                f"Adjust threshold (currently {thr:.2f}%) and use the viewer time slider; "
                f"each frame is computed when you visit it."
            )

        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            self.log(f"\u274c Error loading dead threshold preview: {exc}")


    def _on_threshold_spin_changed(self, value):
        """Live-update the overlay (current frame only) for non-organoid panels."""
        viewer = self.viewer
        if (
            viewer is None
            or self._preview_seg_t is None
            or self._preview_dead_t is None
        ):
            return
        # Threshold change invalidates every cached frame; recompute the
        # one the user is currently viewing.
        self._refresh_preview_dead_layers(value)



# ═══════════════════════════════════════════════════════════════════════════
# ActiveKillingPanel — Extended analysis for immune cells
# ═══════════════════════════════════════════════════════════════════════════
def _render_killing_event_gifs(
    metadata,
    output_dir,
    immune_type,
    target_types,
    n_top=5,
    log_fn=None,
):
    """Render the top-N killers' largest attributed death events as GIFs.

    Thin wrapper over :func:`behav3d.analysis.killing_figures.render_killing_event_gifs`
    (shared with the notebook gallery), resolving the results subfolder the
    same way the run does.
    """
    from behav3d.analysis.killing_figures import render_killing_event_gifs

    subfolder = (
        "combined" if len(target_types) > 1
        else (target_types[0] if target_types else immune_type)
    )
    results_dir = Path(output_dir) / "analysis" / immune_type / "active_killing" / subfolder
    return render_killing_event_gifs(
        metadata, output_dir, immune_type, results_dir, top_n=int(n_top), log_fn=log_fn,
    )


def _derived_min_patch_volume_um3(diameter_um: float) -> float:
    """Default size floor: a quarter of one cell's volume (dye may fill a dying cell only partly)."""
    import math
    return 0.25 * (4.0 / 3.0) * math.pi * (float(diameter_um) / 2.0) ** 3


class ActiveKillingPanel(QWidget):
    """
    Extended analysis: Active Killing for immune (effector) cell types.

    Detects localised **death events** -- new, connected patches in the
    annotated dead mask inside a target track -- and attributes each one's
    single unit of killing credit to the effectors that touched that target
    within the causal window and sit within the attribution radius of the
    patch. Total credit equals the number of attributed death events, so it
    does not inflate with effector density.

    Runs AFTER baseline feature extraction (needs the effector's
    combined_track_features CSV with contact columns) and needs the dead mask
    and tracked-label zarrs.

    Output
    ------
    CSVs + figures -> analysis/<immune>/active_killing/<target|combined>/
    Viewer         -> the top-N killers as Points layers at their credited
                      timepoints (size = credit), and a death-patch preview.
    """

    _LAYER_PREFIX = "[Active Killing]"

    def __init__(
        self,
        immune_types: list,
        metadata_loader,
        viewer=None,
        log_callback=None,
        queue_callback=None,
        parent=None,
        tab_progress_row=None,
    ):
        super().__init__(parent)
        self.immune_types = list(immune_types)
        self.metadata_loader = metadata_loader
        self.viewer = viewer
        self.log = log_callback or (lambda m: None)
        self._queue_callback = queue_callback
        self.tab_progress_row = tab_progress_row
        from behav3d.core.metadata import (
            detect_organoid_types_from_metadata,
            detect_other_cell_types_from_metadata,
        )
        md = getattr(self.metadata_loader, "metadata", None)
        self.target_types = (
            detect_organoid_types_from_metadata(md) + detect_other_cell_types_from_metadata(md)
            if md is not None else []
        )
        self._bg = BackgroundOperation(self)
        from behav3d.widgets.utils import migrate_active_killing_config
        saved_params = getattr(self.metadata_loader, "behav3d_parameters", None) or {}
        self._saved_cfg = migrate_active_killing_config(
            dict(saved_params.get("active_killing", {}) or {}), log_fn=self.log,
        )
        self._syncing_volume = False
        self._init_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        desc = QLabel(
            "Finds NEW death patches in the dead mask inside each target and gives each "
            "one exactly one unit of killing credit, shared among the effectors that "
            "touched that target shortly before and sit next to the patch. Total credit "
            "= number of attributed death events, so it does not grow with effector density.\n"
            "⚠️  Needs the effector's feature CSV (with contact), the dead mask and "
            "the tracked target labels."
        )
        desc.setWordWrap(True)
        desc.setMinimumWidth(0)
        desc.setStyleSheet("color: #90A4AE; font-size: 10px; padding: 2px 0;")
        desc_row = make_help_row(
            desc,
            "Active Killing - How It Works",
            "1. DEATH EVENTS (per target, no immune data involved)\n"
            "A death event is the appearance of a NEW connected patch in the\n"
            "annotated dead mask inside one tracked target - roughly one cell\n"
            "dying. Death is read from the dead mask only; the raw death channel\n"
            "is never re-thresholded. A patch that keeps growing, dims and comes\n"
            "back, or merges with another is still ONE event, and death already\n"
            "present when a target first appears is not new. Detection is cached\n"
            "per target type and reused until its inputs change.\n\n"
            "2. CANDIDATES\n"
            "An effector is a candidate killer of an event if it touched that\n"
            "target (Feature Extraction's Contact Threshold) within the causal\n"
            "window before the death, AND its surface is within the attribution\n"
            "radius of the death patch.\n\n"
            "3. CREDIT\n"
            "Each event carries exactly ONE unit of credit, shared among its\n"
            "candidates (longer, more recent and closer contacts earn more).\n"
            "An event with no candidate is 'unattributed' (background) death.\n"
            "So total credit = number of attributed deaths: extra bystander\n"
            "effectors split a kill, they never add one.\n\n"
            "PREREQUISITES\n"
            " - Feature Extraction for the immune type, with Contact enabled\n"
            " - Dead-mask segmentation and tracked labels for the targets\n\n"
            "SUGGESTED ORDER\n"
            " 1. Set the target cell diameter and check it with 'Preview death\n"
            "    patches'.\n"
            " 2. Set the causal window for your target biology.\n"
            " 3. Run Active Killing.\n"
            " 4. Optionally 'Calibrate radius & validate', then re-run.\n\n"
            "ADVANCED\n"
            "Derived values and literature constants (e.g. damage_tau_min,\n"
            "hit_cap, patch_link_radius_um) are not in the GUI. Override them\n"
            "in behav3d_parameters.yml under active_killing.advanced; any\n"
            "override is recorded in active_killing_run_params.json.\n\n"
            "Outputs of the previous algorithm are not comparable and are\n"
            "refused downstream until Active Killing is re-run.",
        )
        layout.addLayout(desc_row)

        imm_row = QHBoxLayout()
        imm_row.addWidget(QLabel("Immune cell type:"))
        self.immune_combo = QComboBox()
        self.immune_combo.setMaximumWidth(240)
        if self.immune_types:
            self.immune_combo.addItems(self.immune_types)
        else:
            self.immune_combo.addItem("(no immune types detected)")
            self.immune_combo.setEnabled(False)
        self.immune_combo.currentTextChanged.connect(self._validate)
        imm_row.addWidget(self.immune_combo, stretch=1)
        imm_row.addWidget(HelpButton(
            "Immune Cell Type",
            "The effector population whose cells can be credited with kills.\n\n"
            "Its feature CSV must contain the contact column 'touching_{target}s'\n"
            "(which targets each effector touches), so run Feature Extraction\n"
            "for it with Contact enabled first.\n\n"
            "Results go to analysis/<immune>/active_killing/. Run once per\n"
            "immune type; the cached death events are shared between them.",
        ))
        layout.addLayout(imm_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target cell types:"))
        self.target_list = QListWidget()
        self.target_list.setSelectionMode(QListWidget.MultiSelection)
        self.target_list.setMaximumHeight(60)
        if self.target_types:
            self.target_list.addItems(self.target_types)
            saved_targets = self._saved_cfg.get("target_types")
            saved_targets = set(saved_targets) if saved_targets else None
            for i in range(self.target_list.count()):
                item = self.target_list.item(i)
                item.setSelected(item.text() in saved_targets if saved_targets else True)
        else:
            self.target_list.addItem("(no targets detected)")
            self.target_list.setEnabled(False)
        self.target_list.itemSelectionChanged.connect(self._validate)
        target_row.addWidget(self.target_list, stretch=1)
        target_row.addWidget(HelpButton(
            "Target Cell Types",
            "The cell types whose deaths are searched for and attributed\n"
            "(organoids and other non-immune types from the metadata).\n\n"
            "Each selected target needs its tracked labels and the dead mask.\n"
            "Death events are detected per target type. Results go to a folder\n"
            "named after the target, or 'combined' when several are selected.\n\n"
            "Death-patch preview uses the first selected target.",
        ), alignment=Qt.AlignTop)
        layout.addLayout(target_row)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setMinimumWidth(0)
        self.validation_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.validation_label)

        # ── Parameters: the only three analysis choices ─────────────────────
        params_group = QGroupBox("Parameters")
        params_form = QFormLayout(params_group)
        params_form.setContentsMargins(6, 6, 6, 6)
        params_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        adv = dict(self._saved_cfg.get("advanced") or {})
        self.spin_cell_diameter = QDoubleSpinBox()
        self.spin_cell_diameter.setRange(1.0, 200.0)
        self.spin_cell_diameter.setDecimals(1)
        self.spin_cell_diameter.setSingleStep(0.5)
        self.spin_cell_diameter.setSuffix(" µm")
        self.spin_cell_diameter.setMaximumWidth(100)
        self.spin_cell_diameter.setValue(float(self._saved_cfg.get("target_cell_diameter_um", 10.0)))
        params_form.addRow(
            "Target cell diameter:",
            make_help_row(
                self.spin_cell_diameter,
                "Target Cell Diameter",
                "Diameter of ONE target cell (not the organoid). This sets the death\n"
                "threshold: a new dead patch must reach a quarter of one cell's volume,\n"
                "0.25 x (4/3)pi(d/2)^3, to count as a death event (dye may fill a dying\n"
                "cell only partly). It also sets how close two dead fragments must be\n"
                "to count as the same dying cell.\n\n"
                "Check it with 'Preview death patches' on a frame you trust: the patches\n"
                "it accepts should be cells you agree are dying. Lower it if real deaths\n"
                "are missed; raise it if specks are counted.",
            ),
        )

        self.spin_min_patch_volume = QDoubleSpinBox()
        self.spin_min_patch_volume.setRange(0.1, 1.0e7)
        self.spin_min_patch_volume.setDecimals(1)
        self.spin_min_patch_volume.setSuffix(" µm³")
        self.spin_min_patch_volume.setMaximumWidth(120)
        override = adv.get("min_patch_volume_um3")
        self.spin_min_patch_volume.setValue(
            float(override) if override is not None
            else _derived_min_patch_volume_um3(self.spin_cell_diameter.value())
        )
        self._volume_overridden = override is not None
        params_form.addRow(
            "  = min death patch:",
            make_help_row(
                self.spin_min_patch_volume,
                "Minimum Death-Patch Volume",
                "The same threshold as the diameter above, in volume. Filled in from\n"
                "the diameter (10 µm -> ~131 µm³); type a value here to set the\n"
                "threshold directly instead. Changing the diameter afterwards\n"
                "recomputes it.\n\n"
                "The hint below converts it to voxels for this dataset. Below ~20\n"
                "voxels the threshold is weaker than mask noise; above ~5000 only\n"
                "catastrophic death is detected.\n\n"
                "This is the successor of the old absolute death-signal threshold, but\n"
                "measured on one connected NEW dead patch instead of the whole\n"
                "organoid's dead-pixel count.",
            ),
        )
        self.spin_cell_diameter.valueChanged.connect(self._on_diameter_changed)
        self.spin_min_patch_volume.valueChanged.connect(self._on_volume_edited)

        self.spin_causal_window = QDoubleSpinBox()
        self.spin_causal_window.setRange(1.0, 1440.0)
        self.spin_causal_window.setDecimals(0)
        self.spin_causal_window.setSingleStep(10.0)
        self.spin_causal_window.setSuffix(" min")
        self.spin_causal_window.setMaximumWidth(110)
        self.spin_causal_window.setValue(float(self._saved_cfg.get("causal_window_min", 120.0)))
        self.spin_causal_window.valueChanged.connect(lambda _: self._update_param_hints())
        params_form.addRow(
            "Causal window:",
            make_help_row(
                self.spin_causal_window,
                "Causal Window",
                "How long after a contact a death can still be credited to it.\n\n"
                "Contact-to-apoptosis lags are long for solid tumours (1.8 +/- 1.5 h\n"
                "reported for melanoma) and short for haematologic targets (5-25 min):\n"
                "~120 min for organoids / carcinoma, ~30 min for haematologic targets.\n\n"
                "Inside the window, credit still favours recent contact: contact time\n"
                "is weighted by exp(-lag / 82 min), from the published ~57 min\n"
                "half-life of sublethal damage.\n\n"
                "Given in minutes on purpose - the frame equivalent (shown below)\n"
                "depends on your imaging interval.",
            ),
        )

        self.spin_attr_radius = QDoubleSpinBox()
        self.spin_attr_radius.setRange(0.5, 200.0)
        self.spin_attr_radius.setDecimals(1)
        self.spin_attr_radius.setSingleStep(0.5)
        self.spin_attr_radius.setSuffix(" µm")
        self.spin_attr_radius.setMaximumWidth(100)
        self.spin_attr_radius.setValue(float(self._saved_cfg.get("attribution_radius_um", 15.0)))
        self.spin_attr_radius.valueChanged.connect(lambda _: self._update_param_hints())
        params_form.addRow(
            "Attribution radius:",
            make_help_row(
                self.spin_attr_radius,
                "Attribution Radius",
                "Maximum surface-to-surface distance between an effector and a death\n"
                "patch for the effector to be a candidate killer. Candidates must ALSO\n"
                "have touched that target (Feature Extraction's Contact Threshold)\n"
                "within the causal window.\n\n"
                "Default 15 µm (effector radius + one target cell + segmentation error).\n"
                "Closer candidates get a larger share: credit is weighted by\n"
                "exp(-distance / (radius / 3)).\n\n"
                "Rather than guessing, use 'Calibrate radius & validate' below: it\n"
                "measures how often an effector would sit this close to a death by\n"
                "chance, and fills in the largest radius with an FDR <= 5%.\n\n"
                "This is not the Contact Threshold: that one decides whether an\n"
                "effector touches the target at all; this one whether it is next to\n"
                "this particular dead patch. The hint below warns when they differ\n"
                "by more than 2x.",
            ),
        )

        self.param_hint_label = QLabel("")
        self.param_hint_label.setWordWrap(True)
        self.param_hint_label.setMinimumWidth(0)
        self.param_hint_label.setStyleSheet("color: #90A4AE; font-size: 10px;")
        params_form.addRow("", self.param_hint_label)

        self.btn_calibrate = QPushButton("\U0001f4cf  Calibrate radius && validate")
        self.btn_calibrate.setStyleSheet(
            "QPushButton { background: #37474F; color: white; padding: 4px 10px; "
            "border-radius: 3px; font-size: 11px; } QPushButton:hover { background: #546E7A; }"
        )
        self.btn_calibrate.setToolTip(
            "Optional, slower check of the attribution itself (run after one Active Killing run):\n\n"
            " - Density invariance: re-attributes on random subsets of effector tracks.\n"
            "   Death events and credit per death must not change; the previous\n"
            "   algorithm's verdicts per death grow with effector density.\n"
            " - Radius calibration: rotates each death patch inside its organoid to see\n"
            "   how often an effector would be nearby by chance, and picks the largest\n"
            "   radius with an empirical FDR <= 5%. If none, it says so instead.\n\n"
            "Writes .../active_killing/<target>/validation/. A calibrated radius is\n"
            "applied to the field above and saved."
        )
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.calibration_label = QLabel(self._calibration_text())
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setStyleSheet("color: #90A4AE; font-size: 10px;")
        cal_box = QWidget()
        cal_row = QVBoxLayout(cal_box)
        cal_row.setContentsMargins(0, 0, 0, 0)
        cal_row.addLayout(make_help_row(
            self.btn_calibrate,
            "Calibrate Radius & Validate",
            "Optional, slower check that attribution means something in THIS\n"
            "dataset. Run it after one Active Killing run, with the same settings.\n\n"
            "Radius calibration (FDR): each death patch is rotated to random\n"
            "positions inside its own target, keeping its size, depth and the\n"
            "effectors exactly as they are. How often an effector still lands\n"
            "within r of the rotated patch is the chance rate. The empirical FDR\n"
            "at r is (chance + 1) / (observed + 1), and the calibrated radius is\n"
            "the largest r with FDR <= 5%.\n\n"
            "If no radius reaches 5%, nothing is applied and the read-out below\n"
            "gives the best FDR and where it was reached: attribution cannot be\n"
            "told apart from chance at that effector density. With few death\n"
            "events the FDR cannot fall below 1 / (N + 1).\n\n"
            "Density invariance: re-attributes on random subsets of effector\n"
            "tracks. Death events and credit per attributed death must stay the\n"
            "same; the previous algorithm's verdicts per death grow with density.\n\n"
            "Writes .../active_killing/<target>/validation/. A calibrated radius\n"
            "is put in the field above and saved; re-run Active Killing to use it.",
        ))
        cal_row.addWidget(self.calibration_label)
        params_form.addRow("", cal_box)

        layout.addWidget(params_group)

        # ── Viewer preview ─────────────────────────────────────────────────
        viewer_group = QGroupBox("Viewer Preview")
        viewer_form = QFormLayout(viewer_group)
        viewer_form.setContentsMargins(6, 6, 6, 6)

        self.btn_preview_patches = QPushButton("\U0001f50d  Preview death patches (current frame)")
        self.btn_preview_patches.setStyleSheet(
            "QPushButton { background: #37474F; color: white; padding: 5px 10px; "
            "border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background: #546E7A; }"
        )
        self.btn_preview_patches.setToolTip(
            "Adds the candidate NEW death patches at the viewer's current frame as a\n"
            "labels layer, and logs each patch's volume and whether it passes the\n"
            "death threshold. Use it to check the target cell diameter.\n\n"
            "Approximation: compares with the previous frame only (the analysis uses\n"
            "the whole history), so a patch still growing may appear here."
        )
        self.btn_preview_patches.clicked.connect(self._on_preview_patches_clicked)
        viewer_form.addRow("", make_help_row(
            self.btn_preview_patches,
            "Preview Death Patches",
            "The quickest way to set the target cell diameter.\n\n"
            "Adds the candidate NEW death patches at the viewer's current frame\n"
            "as a labels layer, and logs each patch's volume and whether it passes\n"
            "the death threshold. Move to a frame where you know cells are dying,\n"
            "click, and check that the accepted patches are the cells you would\n"
            "call dying. Change the diameter and click again to compare.\n\n"
            "Uses the first sample in the metadata and the first selected target.\n\n"
            "Approximation: it compares with the previous frame only, while the\n"
            "analysis uses the whole history, so a patch that is still growing\n"
            "can appear here without being a new event in the run.",
        ))

        self.spin_top_n = QSpinBox()
        self.spin_top_n.setRange(1, 50)
        self.spin_top_n.setValue(5)
        self.spin_top_n.setMaximumWidth(70)
        viewer_form.addRow(
            "Top-N killers:",
            make_help_row(
                self.spin_top_n,
                "Top-N Killers",
                "Number of top effectors (by attributed kill credit) used for the\n"
                "viewer points, the killing GIFs and the swimmer plot - one shared\n"
                "ranking, so all three show the same cells.",
            ),
        )

        self.btn_load_viewer = QPushButton("\U0001f441  Load Top Killers in Viewer")
        self.btn_load_viewer.setStyleSheet(
            "QPushButton { background: #37474F; color: white; padding: 5px 10px; "
            "border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background: #546E7A; }"
        )
        self.btn_load_viewer.setToolTip(
            "Loads the top-N killers as colour-coded Points layers at every timepoint\n"
            "credited with an attributed death event (point size = credit).\n"
            "Requires Active Killing Analysis to have been run at least once."
        )
        self.btn_load_viewer.clicked.connect(self._on_load_viewer_clicked)
        viewer_form.addRow("", make_help_row(
            self.btn_load_viewer,
            "Load Top Killers in Viewer",
            "Adds one Points layer per top-N killer (ranked by attributed kill\n"
            "credit), placed at every timepoint where that effector was credited\n"
            "with a death. Point size = credit, so a shared kill shows smaller.\n\n"
            "Credit sits on the effector's last contact frame before the death\n"
            "onset, so the point marks where it was, not where the patch appears.\n\n"
            "Needs a completed Active Killing run for the selected immune type.",
        ))

        self.btn_export_gifs = QPushButton("\U0001f39e  Export Killing GIFs")
        self.btn_export_gifs.setStyleSheet(
            "QPushButton { background: #37474F; color: white; padding: 5px 10px; "
            "border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background: #546E7A; }"
        )
        self.btn_export_gifs.setToolTip(
            "One GIF per top-N killer, centred on its largest attributed death event:\n"
            "raw signal, dead mask in red, the attributed death patch in yellow and\n"
            "the killer in purple, from its last contact to after the death onset.\n\n"
            "Saved to analysis/<immune>/active_killing/<target>/gallery/<sample>/."
        )
        self.btn_export_gifs.clicked.connect(self._on_export_gifs_clicked)
        viewer_form.addRow("", make_help_row(
            self.btn_export_gifs,
            "Export Killing GIFs",
            "Writes one GIF per top-N killer, centred on its largest attributed\n"
            "death event, from its last contact to after the death onset.\n\n"
            "Colours: raw signal in grey, dead mask in red, the attributed death\n"
            "patch in yellow, the killer in purple.\n\n"
            "Uses the same top-N ranking as the viewer points and the swimmer\n"
            "plot. Saved to analysis/<immune>/active_killing/<target>/gallery/\n"
            "<sample>/. Needs a completed Active Killing run.",
        ))
        layout.addWidget(viewer_group)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.btn_run = QPushButton("▶  Run Active Killing Analysis")
        self.btn_run.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 8px; font-size: 13px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        action_row.addWidget(self.btn_run, stretch=1)

        self.btn_queue = QPushButton("+\U0001f6d2")
        self.btn_queue.setFixedSize(40, 34)
        self.btn_queue.setToolTip("Add Active Killing Analysis to Processing Queue")
        self.btn_queue.setStyleSheet(
            "QPushButton { background: #1a1a2e; color: #ffc107; border: 1px solid #ffc107; "
            "border-radius: 4px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background: #ffc107; color: #1a1a2e; }"
            "QPushButton:disabled { background: #2a2a2a; color: #666; border: 1px solid #555; }"
        )
        self.btn_queue.clicked.connect(self._on_queue_clicked)
        action_row.addWidget(self.btn_queue)
        layout.addLayout(action_row)
        layout.addStretch()

        self._validate()
        self._update_param_hints()

    # ── Parameter helpers ────────────────────────────────────────────────────
    def _on_diameter_changed(self, value):
        if not self._volume_overridden:
            self._syncing_volume = True
            try:
                self.spin_min_patch_volume.setValue(_derived_min_patch_volume_um3(value))
            finally:
                self._syncing_volume = False
        self._update_param_hints()

    def _on_volume_edited(self, value):
        if self._syncing_volume:
            return
        derived = _derived_min_patch_volume_um3(self.spin_cell_diameter.value())
        self._volume_overridden = abs(float(value) - derived) > 0.05
        self._update_param_hints()

    def _voxel_and_frame(self):
        from behav3d.core.utils import minutes_per_frame_from_metadata, resolution_from_metadata
        md = getattr(self.metadata_loader, "metadata", None)
        xy, z, ok_res = resolution_from_metadata(md)
        mpf, ok_t = minutes_per_frame_from_metadata(md)
        return (xy, z, ok_res), (mpf, ok_t)

    def _update_param_hints(self):
        import math
        (xy, z, ok_res), (mpf, ok_t) = self._voxel_and_frame()
        parts = []
        vol = float(self.spin_min_patch_volume.value())
        if ok_res:
            n_vox = vol / (xy * xy * z)
            txt = f"Death threshold ≈ {vol:.0f} µm³ ≈ {n_vox:.0f} voxels"
            if n_vox < 20:
                txt += " ⚠️ below ~20 voxels the threshold is weaker than mask noise"
            elif n_vox > 5000:
                txt += " ⚠️ above ~5000 voxels only catastrophic death is detected"
            if z / xy > 4:
                txt += f" (z/xy = {z / xy:.1f}: z sensitivity is coarser)"
            parts.append(txt)
        if self._volume_overridden:
            parts.append("Death threshold set directly (overrides the diameter).")
        if ok_t and mpf > 0:
            parts.append(f"Causal window = {math.ceil(self.spin_causal_window.value() / mpf):.0f} frames "
                         f"at {mpf:g} min/frame")
        ct = self._contact_threshold_um()
        r = float(self.spin_attr_radius.value())
        if ct is not None and ct > 0 and (r / ct > 2 or ct / r > 2):
            parts.append(
                f"ℹ️ Contact Threshold (Feature Extraction) is {ct:g} µm vs radius {r:g} µm: "
                "candidates must satisfy both, so the smaller one limits who can be credited."
            )
        self.param_hint_label.setText("\n".join(parts))

    def _calibration_text(self) -> str:
        cal = self._saved_cfg.get("calibration") or {}
        if not cal:
            return "Radius not calibrated (default / user-set)."
        if cal.get("calibrated_radius_um") is None:
            best, at = cal.get("best_fdr"), cal.get("radius_at_best_fdr_um")
            best_txt = f"best FDR {best:.3f} at {at:g} µm" if best is not None and at is not None else "no estimate"
            return (f"Last calibration: no radius reached FDR <= {cal.get('alpha', 0.05):g} "
                    f"({best_txt}, {cal.get('n_informative_events')} events); nothing applied.")
        return (f"Calibrated: {cal['calibrated_radius_um']:g} \u00b5m, empirical FDR "
                f"{cal.get('fdr_at_radius', float('nan')):.3f} ({cal.get('n_informative_events')} events).")

    def _on_calibrate_clicked(self):
        """Run density-invariance + FDR radius calibration in the background."""
        self._validate()
        if not self.btn_run.isEnabled():
            return
        if self._bg.is_running():
            self.log("\u26a0\ufe0f An operation is already in progress.")
            return
        immune = self._get_immune_type()
        targets = self._get_selected_targets()
        params = self._collect_params()
        md = self.metadata_loader.metadata
        out_dir = self.metadata_loader.output_dir

        def _do(progress_cb=None):
            from behav3d.analysis.killing_validation import run_killing_validation
            return run_killing_validation(
                md, out_dir, immune, targets,
                target_cell_diameter_um=params["target_cell_diameter_um"],
                causal_window_min=params["causal_window_min"],
                attribution_radius_um=params["attribution_radius_um"],
                advanced=params.get("advanced") or None, log_fn=self.log,
            )

        def _on_done(res):
            self.btn_calibrate.setText("\U0001f4cf  Calibrate radius && validate")
            record = {k: res.get(k) for k in (
                "calibrated_radius_um", "fdr_at_radius", "alpha", "best_fdr", "radius_at_best_fdr_um",
                "fdr_floor", "n_informative_events", "requested_radius_um")}
            params_all = self.metadata_loader.behav3d_parameters
            if params_all is not None:
                params_all.setdefault("active_killing", {})["calibration"] = record
            self._saved_cfg["calibration"] = record
            if res.get("calibrated_radius_um") is not None:
                self.spin_attr_radius.setValue(float(res["calibrated_radius_um"]))
                self.log(f"\u2705 Attribution radius calibrated to {res['calibrated_radius_um']:g} \u00b5m "
                         f"(empirical FDR {res['fdr_at_radius']:.3f}); re-run Active Killing to apply it.")
            self._persist()
            self.calibration_label.setText(self._calibration_text())
            self.log(f"Validation outputs: {res['paths']['dir']}")

        def _on_failed(err):
            self.btn_calibrate.setText("\U0001f4cf  Calibrate radius && validate")
            self.log(f"\u274c Calibration failed: {err}")

        self.btn_calibrate.setText("\u23f3 Calibrating\u2026")
        self._bg.run(
            fn=_do, desc=f"Active Killing calibration \u2014 {immune}\u2026",
            progress_row=self.tab_progress_row, buttons=[self.btn_calibrate, self.btn_run],
            viewer=self.viewer, on_done=_on_done, on_failed=_on_failed,
            inject_progress=False, indeterminate=True,
        )

    def _contact_threshold_um(self):
        params = getattr(self.metadata_loader, "behav3d_parameters", None) or {}
        try:
            v = params.get("features", {}).get("immune", {}).get("contact_threshold")
            return float(v) if v is not None else None
        except Exception:
            return None

    # ── Small helpers ────────────────────────────────────────────────────────
    def _get_immune_type(self) -> str:
        return self.immune_combo.currentText()

    def _feature_csv_path(self, immune_type: str) -> Path:
        # Active Killing always reads the RAW (unfiltered) features.
        out = Path(self.metadata_loader.output_dir)
        return out / "analysis" / immune_type / "track_features" / f"BEHAV3D_{immune_type}_combined_track_features.csv"

    def _get_selected_targets(self) -> list:
        if not self.target_list.isEnabled():
            return []
        return [item.text() for item in self.target_list.selectedItems()]

    def _active_killing_dir(self, immune_type: str) -> Path:
        return Path(self.metadata_loader.output_dir) / "analysis" / immune_type / "active_killing"

    def set_queue_callback(self, callback):
        """Set the callback used by the cart button to enqueue this analysis."""
        self._queue_callback = callback
        if hasattr(self, "btn_queue"):
            self.btn_queue.setEnabled(callback is not None and self.btn_run.isEnabled())

    def get_queue_params(self) -> dict:
        """Snapshot the current Active Killing analysis settings for the queue."""
        return {"immune_type": self._get_immune_type(), **self._collect_params()}

    def _on_queue_clicked(self):
        if self._queue_callback is None:
            self.log("⚠️ Active Killing queue is not connected.")
            return
        self._validate()
        if not self.btn_run.isEnabled():
            return
        self._persist()
        self._queue_callback()

    def _set_invalid(self, text):
        self.validation_label.setText(text)
        self.validation_label.setStyleSheet("color: #E57373; font-size: 10px;")
        self.btn_run.setEnabled(False)
        if hasattr(self, "btn_queue"):
            self.btn_queue.setEnabled(False)

    def _validate(self):
        immune = self._get_immune_type()
        targets = self._get_selected_targets()
        if not immune or "(no immune" in immune:
            return self._set_invalid("⚠️ No immune cell types detected in metadata.")
        if not targets:
            return self._set_invalid("⚠️ No target cell types selected.")
        csv = self._feature_csv_path(immune)
        if not csv.exists():
            return self._set_invalid(
                f"⚠️ {immune} feature CSV not found — run feature extraction first.\n"
                f"Expected: .../analysis/{immune}/track_features/{csv.name}"
            )
        try:
            header_cols = set(pd.read_csv(csv, nrows=0).columns)
        except Exception:
            header_cols = None
        expected = {f"touching_{t}s" for t in targets}
        if header_cols is not None and not (expected & header_cols):
            return self._set_invalid(
                f"⚠️ {csv.name} has no contact columns for {', '.join(targets)} "
                f"(expected {', '.join(sorted(expected))}). Enable 'contact' in Feature "
                f"Extraction for {immune} and re-run it."
            )
        md = getattr(self.metadata_loader, "metadata", None)
        if md is not None and len(md):
            from behav3d.features.death_events import resolve_dead_mask_path
            missing = [str(r["sample_name"]) for _, r in md.iterrows()
                       if resolve_dead_mask_path(r, self.metadata_loader.output_dir)[0] is None]
            if len(missing) == len(md):
                return self._set_invalid(
                    "⚠️ No dead mask found for any sample. Active Killing reads death from "
                    "the annotated dead mask — run dead-mask segmentation first."
                )
        self.validation_label.setText(f"✓ Ready  —  using: {csv.name}")
        self.validation_label.setStyleSheet("color: #66BB6A; font-size: 10px;")
        self.btn_run.setEnabled(True)
        if hasattr(self, "btn_queue"):
            self.btn_queue.setEnabled(self._queue_callback is not None)

    def _collect_params(self) -> dict:
        advanced = dict(self._saved_cfg.get("advanced") or {})
        if self._volume_overridden:
            advanced["min_patch_volume_um3"] = float(self.spin_min_patch_volume.value())
        else:
            advanced.pop("min_patch_volume_um3", None)
        return {
            "target_cell_diameter_um": float(self.spin_cell_diameter.value()),
            "causal_window_min": float(self.spin_causal_window.value()),
            "attribution_radius_um": float(self.spin_attr_radius.value()),
            "advanced": advanced,
            "target_types": self._get_selected_targets(),
        }

    def _persist(self):
        """Write the current Active Killing settings into behav3d_parameters.yml."""
        params = self.metadata_loader.behav3d_parameters
        if params is None:
            return
        from behav3d.widgets.utils import migrate_active_killing_config
        cfg = migrate_active_killing_config(dict(params.get("active_killing", {}) or {}), log_fn=None)
        cfg.update(self._collect_params())
        params["active_killing"] = cfg
        self._saved_cfg = dict(cfg)
        out_dir = self.metadata_loader.output_dir
        if out_dir:
            try:
                from behav3d.io.parameters import save_params
                save_params(params, out_dir)
            except Exception as e:
                self.log(f"Warning: Could not save Active Killing parameters: {e}")

    def refresh_immune_types(self, immune_types: list):
        """Update the dropdown when metadata is reloaded."""
        self.immune_types = list(immune_types)
        current = self.immune_combo.currentText()
        self.immune_combo.blockSignals(True)
        self.immune_combo.clear()
        if immune_types:
            self.immune_combo.addItems(immune_types)
            if current in immune_types:
                self.immune_combo.setCurrentText(current)
            self.immune_combo.setEnabled(True)
        else:
            self.immune_combo.addItem("(no immune types detected)")
            self.immune_combo.setEnabled(False)
        self.immune_combo.blockSignals(False)
        self._validate()
        self._update_param_hints()

    # ── Run ──────────────────────────────────────────────────────────────────
    def _run_all_targets(self, immune, targets, params, progress_cb=None):
        """Run per target (and combined when >1 target); returns (stats, last_subfolder)."""
        from behav3d.features.advanced_timepoint_features import run_active_killing_analysis

        md = self.metadata_loader.metadata
        output_dir_str = str(self.metadata_loader.output_dir)
        jobs = [([t], t) for t in targets]
        if len(targets) > 1:
            jobs.append((list(targets), "combined"))
        stats = {}
        for i, (t_list, subfolder) in enumerate(jobs):
            self.log(f"--- Active Killing: {immune} vs {', '.join(t_list)} -> {subfolder}/ ---")

            def _cb(done, total, label, _i=i):
                # The progress signal is (int, int, str): report per-mille.
                if progress_cb is not None:
                    frac = (_i + float(done) / max(float(total), 1.0)) / len(jobs)
                    progress_cb(int(round(1000 * min(max(frac, 0.0), 1.0))), 1000, str(label))

            _, _, stats = run_active_killing_analysis(
                metadata=md, output_dir=output_dir_str, immune_cell_type=immune, target_cell_types=t_list,
                target_cell_diameter_um=params["target_cell_diameter_um"],
                causal_window_min=params["causal_window_min"],
                attribution_radius_um=params["attribution_radius_um"],
                advanced=params.get("advanced") or None, save_results=True, output_subfolder=subfolder,
                progress_cb=_cb, top_n_killers=int(self.spin_top_n.value()),
            )
        return stats, jobs[-1][1]

    def _report(self, stats, results_dir, interactive=True):
        n_att = int(stats["n_attributed"])
        n_ev = int(stats["n_death_events"])
        conv = stats["conversion_rate"]
        conv_txt = f"{conv:.2f}" if conv == conv else "n/a"
        self.log(
            f"✅ Active Killing complete — {n_att}/{n_ev} death events attributed, "
            f"conversion rate {conv_txt} (attributed deaths per contact event), "
            f"{stats['total_kill_credit']:.2f} kill credit."
        )
        if stats.get("figures_error"):
            self.log(f"⚠️ Figures failed: {stats['figures_error']} (result tables are complete).")
        stale_filtered = stats.get("filtering_needs_rerun_for") or []
        if stale_filtered:
            self.log(
                f"⚠️ Re-run Filtering for {', '.join(stale_filtered)} — its filtered CSV "
                "was built before this run and doesn't include these results yet."
            )
        if interactive:
            self._offer_open_folder(results_dir, stale_filtered_cell_types=stale_filtered)

    def _on_run_clicked(self):
        """Run Active Killing in the background with a per-frame progress bar."""
        self._validate()
        if not self.btn_run.isEnabled():
            return
        if self._bg.is_running():
            self.log("⚠️ An active killing run is already in progress.")
            return
        self._persist()
        immune = self._get_immune_type()
        targets = self._get_selected_targets()
        params = self._collect_params()
        self.btn_run.setText("⏳ Running…")
        self.log(
            f"▶ Active Killing: {immune} vs {targets}  (cell diameter "
            f"{params['target_cell_diameter_um']:g} µm, causal window {params['causal_window_min']:g} min, "
            f"radius {params['attribution_radius_um']:g} µm)…"
        )

        def _do(progress_cb=None):
            return self._run_all_targets(immune, targets, params, progress_cb=progress_cb)

        def _on_done(result):
            stats, subfolder = result
            self.btn_run.setText("▶  Run Active Killing Analysis")
            self._report(stats, self._active_killing_dir(immune) / subfolder, interactive=True)
            notify_results_changed(self)

        def _on_failed(err: str):
            self.log(f"❌ Active Killing Analysis error: {err}")
            self.btn_run.setText("▶  Run Active Killing Analysis")

        self._bg.run(
            fn=_do,
            desc=f"Active Killing — {immune}…",
            progress_row=self.tab_progress_row,
            buttons=[self.btn_run],
            viewer=self.viewer,
            on_done=_on_done,
            on_failed=_on_failed,
            inject_progress=True,
            indeterminate=False,
        )

    def run_analysis(self, interactive: bool = True, extra_callbacks=None):
        """Synchronous Active Killing used by the processing queue.

        Kept blocking on purpose: the queue iterates steps serially and relies
        on each step completing before moving on. ``extra_callbacks`` is the
        queue's chaining hook fired once the analysis finishes.
        """
        self._validate()
        if not self.btn_run.isEnabled():
            fire_extra_callback(extra_callbacks, "on_failed", "run button disabled")
            return
        immune = self._get_immune_type()
        params = self._collect_params()
        targets = params.get("target_types") or self.target_types
        if not targets:
            self.log("⚠️ No target types detected — cannot run active killing analysis.")
            fire_extra_callback(extra_callbacks, "on_failed", "no target types")
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Running…")
        if hasattr(self, "btn_queue"):
            self.btn_queue.setEnabled(False)
        try:
            stats, subfolder = self._run_all_targets(immune, targets, params)
            self._report(stats, self._active_killing_dir(immune) / subfolder, interactive=interactive)
            fire_extra_callback(extra_callbacks, "on_done", (None, None, stats, subfolder))
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            self.log(f"❌ Active Killing Analysis error: {e}")
            fire_extra_callback(extra_callbacks, "on_failed", str(e))
        finally:
            self.btn_run.setEnabled(True)
            if hasattr(self, "btn_queue"):
                self.btn_queue.setEnabled(self._queue_callback is not None and self.btn_run.isEnabled())
            self.btn_run.setText("▶  Run Active Killing Analysis")
            notify_results_changed(self)

    # ── Folder open popup ──────────────────────────────────────────────────────
    def _offer_open_folder(self, results_dir: Path, stale_filtered_cell_types=None):
        """Offer to open the output folder in the OS file manager."""
        box = QMessageBox(self)
        box.setWindowTitle("Active Killing Analysis Complete")
        text = (
            "✅  Active Killing Analysis finished!\n\n"
            f"Outputs saved to:\n{results_dir}\n\n"
        )
        if stale_filtered_cell_types:
            text += (
                "⚠️ Re-run Filtering for "
                f"{', '.join(sorted(stale_filtered_cell_types))} — its filtered CSV "
                "was built before this run and doesn't include these results yet.\n\n"
            )
        text += "Open output folder in file manager?"
        box.setText(text)
        btn_open = box.addButton("Open Folder", QMessageBox.AcceptRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() == btn_open:
            import subprocess
            import sys as _sys
            try:
                if _sys.platform == "win32":
                    subprocess.Popen(["explorer", str(results_dir)])
                elif _sys.platform == "darwin":
                    subprocess.Popen(["open", str(results_dir)])
                else:
                    subprocess.Popen(["xdg-open", str(results_dir)])
            except Exception as e:
                self.log(f"Could not open folder: {e}")

    # ── Viewer ─────────────────────────────────────────────────────────────────
    def _results_dir_for_viewer(self, immune):
        targets = self._get_selected_targets()
        sub = "combined" if len(targets) > 1 else (targets[0] if targets else None)
        base = self._active_killing_dir(immune)
        if sub and (base / sub).exists():
            return base / sub
        found = sorted(base.glob(f"*/per_effector_killing_{immune}.csv"), key=lambda p: p.stat().st_mtime)
        return found[-1].parent if found else None

    def _on_preview_patches_clicked(self):
        """Show candidate new-death patches at the current viewer frame."""
        if self.viewer is None:
            self.log("⚠️ No viewer available.")
            return
        md = getattr(self.metadata_loader, "metadata", None)
        targets = self._get_selected_targets()
        if md is None or not len(md) or not targets:
            self.log("⚠️ Load metadata and select a target type first.")
            return
        from behav3d.core.metadata import resolve_immune_track_paths_for_sample
        from behav3d.features.death_events import (
            DeathEventParams, preview_new_death_patches, resolve_dead_mask_path,
            resolve_tracks_image_path, voxel_spacing_from_sample)
        try:
            t = int(self.viewer.dims.current_step[0]) if self.viewer.dims.ndim >= 4 else 0
        except Exception:
            t = 0
        row = md.iloc[0]
        target = targets[0]
        tpath = resolve_tracks_image_path(row, target)
        dpath, tried = resolve_dead_mask_path(row, self.metadata_loader.output_dir)
        if tpath is None or not tpath.exists() or dpath is None:
            self.log(f"⚠️ Preview needs the tracked {target} labels and the dead mask for "
                     f"{row['sample_name']} (dead mask tried: {tried}).")
            return
        immune_paths = {k: v for k, v in resolve_immune_track_paths_for_sample(row).items()
                        if Path(v).exists() and k != target}
        adv = self._collect_params()["advanced"]
        dparams = DeathEventParams(
            target_cell_diameter_um=float(self.spin_cell_diameter.value()),
            **{k: v for k, v in adv.items() if k in DeathEventParams.__dataclass_fields__
               and k != "target_cell_diameter_um"},
        )
        spacing = voxel_spacing_from_sample(row)
        try:
            labels, table = preview_new_death_patches(
                target_tracks_path=tpath, dead_mask_path=dpath, immune_tracks_paths=immune_paths,
                t=t, voxel_spacing=spacing, params=dparams)
        except Exception as e:
            self.log(f"❌ Death-patch preview failed: {e}")
            return
        name = f"{self._LAYER_PREFIX} death patches t={t}"
        for layer in [l for l in list(self.viewer.layers) if l.name.startswith(f"{self._LAYER_PREFIX} death patches")]:
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
        keep = labels.copy()
        if not table.empty:
            fail = table.loc[~table["passes_floor"], "patch"].to_numpy()
            keep[np.isin(keep, fail)] = 0
        self.viewer.add_labels(keep, name=name, scale=spacing, opacity=0.8)
        n_pass = int(table["passes_floor"].sum()) if not table.empty else 0
        self.log(f"\U0001f50d {row['sample_name']} · {target} · t={t}: {len(table)} candidate new-death "
                 f"patches, {n_pass} pass the {dparams.resolved(spacing).min_patch_volume_um3:.0f} µm³ "
                 f"threshold (shown).")
        for _, r in table.sort_values("volume_um3", ascending=False).head(15).iterrows():
            mark = "✓" if r["passes_floor"] else "✗ below threshold"
            self.log(f"   patch {int(r['patch'])}: target #{int(r['target_track_id'])}, "
                     f"{r['volume_um3']:.0f} µm³ {mark}")

    def _on_load_viewer_clicked(self):
        """Load the top-N killers as Points layers at their credited timepoints (size = credit)."""
        if self.viewer is None:
            self.log("⚠️ No viewer available.")
            return
        immune = self._get_immune_type()
        if not immune or "(no immune" in immune:
            self.log("⚠️ No immune type selected.")
            return
        from behav3d.features.advanced_timepoint_features import find_advanced_features_csv
        from behav3d.features.kill_attribution import rank_top_killers
        try:
            advanced_path = find_advanced_features_csv(self.metadata_loader.output_dir, immune)
        except Exception as e:
            self.log(f"⚠️ {e}")
            return
        results_dir = self._results_dir_for_viewer(immune)
        if advanced_path is None or results_dir is None:
            self.log("⚠️ No active-killing results found — run Active Killing Analysis first.")
            return
        try:
            df = pd.read_csv(advanced_path)
            eff = pd.read_csv(results_dir / f"per_effector_killing_{immune}.csv")
            top = rank_top_killers(eff, int(self.spin_top_n.value()))
            if top.empty:
                self.log("ℹ️ No attributed kills — nothing to display.")
                return
            for l in [l for l in list(self.viewer.layers) if l.name.startswith(self._LAYER_PREFIX)
                      and "death patches" not in l.name]:
                try:
                    self.viewer.layers.remove(l)
                except Exception:
                    pass
            _palette = [
                [1.00, 0.18, 0.18, 0.90], [1.00, 0.55, 0.00, 0.90], [0.93, 0.83, 0.00, 0.90],
                [0.13, 0.70, 0.27, 0.90], [0.13, 0.47, 0.90, 0.90], [0.60, 0.13, 0.90, 0.90],
                [0.90, 0.13, 0.54, 0.90], [0.00, 0.84, 0.84, 0.90],
            ]
            killing = df[df["is_active_killing"].astype(bool)]
            n_loaded = 0
            for i, (_, k) in enumerate(top.iterrows()):
                rows = killing[(killing["sample_name"].astype(str) == str(k["sample_name"]))
                               & (killing["TrackID"] == k["immune_track_id"])].sort_values("position_t")
                if rows.empty:
                    continue
                coords = rows[["position_t", "position_z", "position_y", "position_x"]].to_numpy(dtype=float)
                sizes = 6 + 10 * rows["kill_credit"].to_numpy(dtype=float)
                self.viewer.add_points(
                    coords,
                    name=(f"{self._LAYER_PREFIX} {immune} #{int(k['immune_track_id'])} "
                          f"({k['sample_name']}, {float(k['kills_attributed']):.2f} kill credit)"),
                    face_color=[_palette[i % len(_palette)]], size=sizes, symbol="disc",
                    opacity=0.85, out_of_slice_display=True,
                )
                n_loaded += 1
            self.log(f"✅ Loaded {n_loaded} top killer(s); points mark credited timepoints, size = credit.")
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            self.log(f"❌ Error loading top killers in viewer: {e}")

    def _on_export_gifs_clicked(self):
        """Render the top-N killers' largest attributed death events as GIFs (background)."""
        if self._bg.is_running():
            self.log("⚠️ An operation is already in progress.")
            return
        immune = self._get_immune_type()
        if not immune or "(no immune" in immune:
            self.log("⚠️ No immune type selected.")
            return
        metadata = self.metadata_loader.metadata
        output_dir = self.metadata_loader.output_dir
        targets = self._get_selected_targets()
        n_top = int(self.spin_top_n.value())

        def _do_export(progress_cb=None):
            return _render_killing_event_gifs(
                metadata=metadata, output_dir=output_dir, immune_type=immune,
                target_types=targets, n_top=n_top, log_fn=self.log,
            )

        def _on_done(paths):
            self.btn_export_gifs.setText("\U0001f39e  Export Killing GIFs")
            if paths:
                self.log(f"✅ Wrote {len(paths)} killing GIF(s) to the gallery folder.")
                try:
                    self._offer_open_folder(Path(paths[0]).parent.parent)
                except Exception:
                    pass
            else:
                self.log("ℹ️ No killing GIFs produced (no attributed kills, or the images are missing).")

        def _on_failed(err):
            self.btn_export_gifs.setText("\U0001f39e  Export Killing GIFs")
            self.log(f"❌ Killing GIF export error: {err}")

        self._bg.run(
            fn=_do_export,
            desc=f"Killing GIFs — {immune}…",
            progress_row=self.tab_progress_row,
            buttons=[self.btn_export_gifs],
            viewer=self.viewer,
            on_done=_on_done,
            on_failed=_on_failed,
            inject_progress=False,
            indeterminate=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
class FeatureExtractionTab(QWidget):
    def __init__(self, viewer=None, metadata_loader=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.metadata_loader = metadata_loader
        self.panels: dict[str, CellTypeFeaturePanel] = {}
        self._queue_panel = None

        # Background-execution infrastructure for batch + active-killing.
        self._bg = BackgroundOperation(self)

        # Track which cell types are organoids (needed for sync)
        self._org_types: list = []

        # Shared preview cache for organoid panels (all org panels share same ref).
        # Per-frame caches are populated lazily as the user navigates timepoints.
        self._org_preview_cache: dict = {
            "seg_t": None,
            "dead_t": None,
            "overlay_arr": None,
            "pct_overlay_arr": None,
            "label_arr": None,
            "stats_by_frame": {},
            "pct_maps_by_frame": {},
            "computed_frames": set(),
            "current_thr": None,
            "label_type_map": None,
            "is_timelapse": False,
        }

        # Single shared napari time-slider listener for ALL organoid panels.
        # Attached the first time any organoid panel loads a preview.
        self._org_preview_dims_callback = None

        self._init_ui()

        if hasattr(self.metadata_loader, "metadata_loaded"):
            self.metadata_loader.metadata_loaded.connect(self._on_metadata_updated)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Vertical splitter: tab content on top, shared Results panel
        # underneath (re-scans the same output directory used by the
        # other BEHAV3D tabs).
        self.splitter = QSplitter(Qt.Vertical, self)
        self.splitter.setChildrenCollapsible(True)
        outer.addWidget(self.splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(0)
        self.splitter.addWidget(scroll)

        content = QWidget()
        content.setMinimumWidth(0)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        scroll.setWidget(content)

        self.results_panel = ResultsPanel(
            viewer=self.viewer,
            metadata_loader=self.metadata_loader,
            parent=self,
        )
        self.splitter.addWidget(self.results_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, True)
        self.splitter.setSizes([600, 400])

        # ── Global Death Classification group (Organoids only) ─────────────

        # Note: cell-type grouping now lives in the Analysis tab (after
        # Filtering) — see AnalysisTab in _analysis.py. It merges already
        # *filtered* populations, so it never needs to appear here.

        # ── Per-cell-type sub-tabs ─────────────────────────────────────────
        self.cell_tabs = QTabWidget()
        self.cell_tabs.setTabPosition(QTabWidget.West)
        layout.addWidget(self.cell_tabs)
        reset_scroll_on_page_change(self.cell_tabs)

        # ── Global Run + Queue ─────────────────────────────────────────────
        self.btn_run_batch = QPushButton(
            "Run Batch Feature Extraction (All Cell Types)"
        )
        self.btn_run_batch.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 10px; font-size: 14px;"
        )
        self.btn_run_batch.clicked.connect(self._on_run_batch_clicked)

        self.btn_queue_feature = QPushButton("+🛒")
        self.btn_queue_feature.setFixedSize(36, 32)
        self.btn_queue_feature.setToolTip("Add Feature Extraction to Processing Queue")
        self.btn_queue_feature.setStyleSheet(
            "QPushButton { background: #1a1a2e; color: #ffc107; border: 1px solid #ffc107; "
            "border-radius: 4px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: #ffc107; color: #1a1a2e; }"
        )

        batch_row = QHBoxLayout()
        batch_row.setSpacing(4)
        batch_row.addWidget(self.btn_run_batch, stretch=1)
        batch_row.addWidget(self.btn_queue_feature)
        layout.addLayout(batch_row)

        self.btn_run_batch.setVisible(False)
        self.btn_queue_feature.setVisible(False)

        # Shared progress row fed by every Run button on this tab.
        self.progress_row = ProgressBarRow()
        layout.addWidget(self.progress_row)

        # -- Active Killing Extended Analysis (collapsible) --------------------------
        # Panel body is hidden by default; user clicks the toggle to expand.
        self._ak_toggle_label_base = (
            "\U0001f9ec  Extended Analysis \u2014 Active Killing (Immune Cells)"
        )
        self._ak_toggle_btn = QPushButton(
            f"\u25b6  {self._ak_toggle_label_base}"
        )
        self._ak_toggle_btn.setCheckable(True)
        self._ak_toggle_btn.setChecked(False)
        self._ak_toggle_btn.setStyleSheet(
            "QPushButton { background: #2c1810; color: #e74c3c; font-weight: bold; "
            "border: 1px solid #c0392b; border-radius: 4px; padding: 6px 10px; "
            "text-align: left; } "
            "QPushButton:checked { background: #3d1f15; } "
            "QPushButton:hover { background: #3d1f15; }"
        )
        self._ak_toggle_btn.clicked.connect(self._on_ak_toggle)
        self._ak_toggle_btn.setVisible(False)
        layout.addWidget(self._ak_toggle_btn)

        self._ak_body = QWidget()
        self._ak_body.setVisible(False)
        self._ak_inner_layout = QVBoxLayout(self._ak_body)
        self._ak_inner_layout.setContentsMargins(4, 4, 4, 4)
        self.active_killing_panel = None  # created in _rebuild_tabs when immune types are known
        layout.addWidget(self._ak_body)

        # Alias kept for _rebuild_tabs compat
        self._active_killing_group = self._ak_body


        # ── Log ────────────────────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_box)

        # Placeholder shown before metadata is loaded
        self._placeholder = QLabel(
            "Load metadata in the Data Preparation tab to see feature extraction options."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        self.cell_tabs.addTab(self._placeholder, "—")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _log(self, msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def _on_metadata_updated(self):
        self._log("Metadata updated — refreshing feature extraction tabs…")
        self._rebuild_tabs()

    def request_tab_exit(self) -> bool:
        """Block tab switching while a background feature run is in flight."""
        from qtpy.QtWidgets import QMessageBox

        if self._bg.is_running():
            QMessageBox.information(
                self,
                "Operation in progress",
                "A feature extraction run is still in progress. Please "
                "wait for it to finish before switching tabs.",
            )
            return False
        for panel in self.panels.values():
            bg = getattr(panel, "_bg", None)
            if bg is not None and bg.is_running():
                QMessageBox.information(
                    self,
                    "Operation in progress",
                    "A feature extraction run is still in progress. "
                    "Please wait for it to finish before switching tabs.",
                )
                return False
        ak = getattr(self, "active_killing_panel", None)
        if ak is not None:
            bg = getattr(ak, "_bg", None)
            if bg is not None and bg.is_running():
                QMessageBox.information(
                    self,
                    "Operation in progress",
                    "An active-killing analysis is still in progress. "
                    "Please wait for it to finish before switching tabs.",
                )
                return False

        # All guards passed — clean up preview state before leaving
        self._disconnect_org_preview_dims()
        self._org_preview_cache.update({
            "seg_t": None,
            "dead_t": None,
            "segs_dict": {},
            "immune_segs": {},
            "overlay_arr": None,
            "pct_overlay_arr": None,
            "label_arr": None,
            "stats_by_frame": {},
            "pct_maps_by_frame": {},
            "computed_frames": set(),
            "pct_computed_frames": set(),
            "current_thr": None,
            "label_type_map": None,
            "is_timelapse": False,
        })
        for panel in self.panels.values():
            panel._cleanup_preview()
        disconnect_all_preview_dims_listeners(self.viewer)
        clear_viewer_layers(self.viewer)
        self._log("Cleaned up viewer layers.")

        return True

    def _detect_cell_types(self):
        """Return (organoid, immune, other) suitable for feature extraction.

        Mirrors :func:`behav3d.widgets.analysis._detect_downstream_cell_types`:
        per-channel multicolor inputs (``*_N_multicolor``) are stripped because
        feature extraction operates on the aggregated outputs (``*_merged`` or
        ``*_grouped``) emitted by the multicolor merging step. Merged /
        grouped names returned by the metadata helpers are preserved as-is.
        """
        from behav3d.core.metadata import (
            detect_organoid_types_from_metadata,
            detect_immune_cell_types_from_metadata,
            detect_other_cell_types_from_metadata,
            filter_multicolor_inputs,
        )
        md = self.metadata_loader.metadata
        if md is None:
            return [], [], []
        return (
            filter_multicolor_inputs(detect_organoid_types_from_metadata(md)),
            filter_multicolor_inputs(detect_immune_cell_types_from_metadata(md)),
            filter_multicolor_inputs(detect_other_cell_types_from_metadata(md)),
        )

    def _rebuild_tabs(self):
        self.cell_tabs.clear()
        self.panels.clear()

        org, imm, oth = self._detect_cell_types()
        all_types = org + imm + oth
        self._org_types = list(org)

        # Dead channel detection is handled per-panel (CellTypeFeaturePanel).
        # Organoid threshold sync handled via _notify_organoid_threshold_changed.

        if not all_types:
            self.cell_tabs.addTab(self._placeholder, "—")
            self.btn_run_batch.setVisible(False)
            self.btn_queue_feature.setVisible(False)
            self._ak_toggle_btn.setVisible(False)
            self._ak_toggle_btn.blockSignals(True)
            self._ak_toggle_btn.setChecked(False)
            self._ak_toggle_btn.blockSignals(False)
            self._set_ak_collapsible_state(False)
            return

        self.btn_run_batch.setVisible(True)
        self.btn_queue_feature.setVisible(True)

        color_map = {"organoid": "🟣", "immune": "🔵", "other": "🟡"}
        for ct in org:
            self._add_panel(ct, "organoid", all_types, org, color_map, is_organoid=True)
        for ct in imm:
            self._add_panel(ct, "immune", all_types, imm, color_map, is_organoid=False)
        for ct in oth:
            self._add_panel(ct, "other", all_types, oth, color_map, is_organoid=False)

        # -- Refresh / create Active Killing panel ----------------------------
        if imm:
            if self.active_killing_panel is None:
                # First load: create and embed the widget
                self.active_killing_panel = ActiveKillingPanel(
                    immune_types=list(imm),
                    metadata_loader=self.metadata_loader,
                    viewer=self.viewer,
                    log_callback=self._log,
                    tab_progress_row=self.progress_row,
                )
                self._ak_inner_layout.addWidget(self.active_killing_panel)
            else:
                # Subsequent metadata reloads: refresh state in-place
                self.active_killing_panel.metadata_loader = self.metadata_loader
                self.active_killing_panel.viewer = self.viewer
                self.active_killing_panel.refresh_immune_types(list(imm))
            self.active_killing_panel.set_queue_callback(
                self._queue_active_killing if self._queue_panel is not None else None
            )
            self._ak_toggle_btn.setVisible(True)
            self._set_ak_collapsible_state(self._ak_toggle_btn.isChecked())
        else:
            self._ak_toggle_btn.setVisible(False)
            self._ak_toggle_btn.blockSignals(True)
            self._ak_toggle_btn.setChecked(False)
            self._ak_toggle_btn.blockSignals(False)
            self._set_ak_collapsible_state(False)

    def _add_panel(self, ct, category, all_types, cat_types, color_map, is_organoid: bool):
        panel = CellTypeFeaturePanel(
            cell_type=ct,
            category=category,
            metadata_loader=self.metadata_loader,
            all_cell_types=all_types,
            category_types=cat_types,
            log_callback=self._log,
            is_organoid=is_organoid,
            threshold_getter=None,
            tab_progress_row=self.progress_row,
        )
        panel.viewer = self.viewer
        if is_organoid:
            panel._org_preview_cache = self._org_preview_cache
            panel._org_cell_types = list(self._org_types)
        self.panels[ct] = panel
        icon = color_map.get(category, "")
        self.cell_tabs.addTab(panel, f"{icon} {ct}")

    def set_queue_panel(self, queue_panel):
        """Attach the processing queue so Active Killing can enqueue itself."""
        self._queue_panel = queue_panel
        if self.active_killing_panel is not None:
            self.active_killing_panel.set_queue_callback(self._queue_active_killing)

    def _queue_active_killing(self):
        """Enqueue the currently selected Active Killing analysis."""
        if self._queue_panel is None or self.active_killing_panel is None:
            return
        from behav3d.napari._queue import StepType
        params = self.active_killing_panel.get_queue_params()
        self._queue_panel.add_step(StepType.ACTIVE_KILLING, params=params)

    def _set_ak_collapsible_state(self, checked: bool):
        """Keep Active Killing toggle text and body visibility in sync."""
        self._ak_body.setVisible(bool(checked))
        arrow = "\u25bc" if checked else "\u25b6"
        self._ak_toggle_btn.setText(f"{arrow}  {self._ak_toggle_label_base}")

    # ── Global (Organoid) Dead Threshold Preview ─────────────────────────────
    def _on_ak_toggle(self, checked: bool):
        """Toggle body visibility of the Active Killing collapsible section."""
        self._set_ak_collapsible_state(checked)

    # ── Shared organoid preview helpers ─────────────────────────────────────
    def _current_viewer_frame(self) -> int:
        viewer = self.viewer
        if viewer is None:
            return 0
        try:
            return int(viewer.dims.current_step[0])
        except Exception:
            return 0

    def _disconnect_org_preview_dims(self):
        viewer = self.viewer
        if viewer is None or self._org_preview_dims_callback is None:
            self._org_preview_dims_callback = None
            return
        try:
            viewer.dims.events.current_step.disconnect(self._org_preview_dims_callback)
        except Exception:
            pass
        if _ACTIVE_PREVIEW_DIMS.get(id(viewer)) is self._org_preview_dims_callback:
            _ACTIVE_PREVIEW_DIMS.pop(id(viewer), None)
        unregister_preview_dims_listener(viewer, self)
        self._org_preview_dims_callback = None

    def _connect_org_preview_dims(self):
        """Attach a single shared dims listener that triggers an on-demand
        recompute of the Dead/Alive overlay (current frame only) for the
        organoid preview path. Safe to call multiple times — any previous
        listener is dropped first."""
        viewer = self.viewer
        if viewer is None:
            return
        self._disconnect_org_preview_dims()
        _disconnect_any_active_preview_dims(viewer)

        def _on_step(*_):
            cache = self._org_preview_cache
            if (
                cache.get("seg_t") is None
                or cache.get("dead_t") is None
                or not cache.get("is_timelapse")
            ):
                return
            self._refresh_org_preview_for_current_frame()

        try:
            viewer.dims.events.current_step.connect(_on_step)
            self._org_preview_dims_callback = _on_step
            _ACTIVE_PREVIEW_DIMS[id(viewer)] = _on_step
            register_preview_dims_listener(viewer, self, _on_step)
        except Exception:
            self._org_preview_dims_callback = None

    def _invalidate_org_preview_cache(self):
        """Drop per-frame caches on threshold change."""
        cache = self._org_preview_cache
        cache["computed_frames"] = set()
        cache["pct_maps_by_frame"] = {}
        cache["stats_by_frame"] = {}
        cache["overlay_arr"] = None
        cache["pct_overlay_arr"] = None
        cache["label_arr"] = None
        for ct, panel in self.panels.items():
            if ct in self._org_types:
                panel._preview_computed_frames = cache["computed_frames"]
                panel._preview_pct_maps_by_frame = cache["pct_maps_by_frame"]
                panel._preview_stats_by_frame = cache["stats_by_frame"]
                panel._preview_overlay_arr = None
                panel._preview_pct_overlay_arr = None
                panel._preview_label_arr = None
                panel._preview_current_frame = None

    def _refresh_org_preview_for_current_frame(self, thr: float | None = None):
        """Recompute the shared organoid Dead/Alive overlay for the
        currently displayed timepoint (only). Materializes data lazily
        per-frame. Propagates the resulting per-frame caches to every
        organoid panel so hover tooltips stay in sync."""
        viewer = self.viewer
        cache = self._org_preview_cache
        if (
            viewer is None
            or cache.get("seg_t") is None
            or cache.get("dead_t") is None
        ):
            return

        if thr is None:
            thr_val = None
            for ct, panel in self.panels.items():
                if ct in self._org_types and panel.spin_dead_threshold is not None:
                    thr_val = round(float(panel.spin_dead_threshold.value()), 2)
                    break
            if thr_val is None:
                return
            thr = thr_val
        else:
            thr = round(float(thr), 2)

        cached_thr = cache.get("current_thr")
        if cached_thr is None or thr != cached_thr:
            self._invalidate_org_preview_cache()
            cache["current_thr"] = thr

        is_timelapse = bool(cache.get("is_timelapse"))
        frame_idx = self._current_viewer_frame() if is_timelapse else 0

        if frame_idx in cache["computed_frames"]:
            try:
                viewer.layers[f"{_PREVIEW_PREFIX} Dead/Alive"].refresh()
            except Exception:
                pass
            try:
                viewer.layers[f"{_PREVIEW_PREFIX} % Dead Mask"].refresh()
            except Exception:
                pass
            self._propagate_org_preview_to_panels()
            return

        # Materialize this frame: per-frame immune cleaning + org merge
        segs_dict = cache.get("segs_dict", {})
        immune_segs = cache.get("immune_segs", {})
        dead_data = cache["dead_t"]
        org_types = getattr(self, "_org_types", [])

        dead_vol = np.asarray(dead_data[frame_idx])
        if dead_vol.ndim > 3:
            dead_vol = dead_vol[0]
        if immune_segs:
            dead_vol = _clean_dead_frame(dead_vol, immune_segs, frame_idx)

        seg_vol, label_type_map = _merge_org_segments_frame(
            segs_dict, org_types, frame_idx
        )
        if seg_vol is None:
            seg_vol = np.asarray(cache["seg_t"][frame_idx])
            if seg_vol.ndim > 3:
                seg_vol = seg_vol[0]
            label_type_map = {}
        if dead_vol.ndim < seg_vol.ndim:
            dead_vol = np.broadcast_to(dead_vol, seg_vol.shape)
        if label_type_map:
            cache["label_type_map"] = label_type_map

        overlay_t, pct_t, frame_stats = _overlay_for_volume(
            seg_vol, dead_vol, thr, frame_label=f" t={frame_idx}",
        )

        # Store single-frame (Z,Y,X) arrays and replace layer data. ``seg_vol``
        # is the merged label-id array (see ``_merge_org_segments_frame``)
        # already used to build ``overlay_t``/``pct_t``/``label_type_map``.
        cache["overlay_arr"] = overlay_t
        cache["pct_overlay_arr"] = pct_t
        cache["label_arr"] = seg_vol

        cmap = _dead_alive_colormap()
        da_name = f"{_PREVIEW_PREFIX} Dead/Alive"
        try:
            layer = viewer.layers[da_name]
            layer.data = overlay_t
            layer.opacity = 1.0
            if cmap is not None:
                layer.colormap = cmap
            else:
                _apply_dead_alive_colors(layer)
            layer.refresh()
        except (KeyError, ValueError):
            kw = dict(name=da_name, opacity=1.0)
            if cmap is not None:
                kw["colormap"] = cmap
            layer = viewer.add_labels(overlay_t, **kw)
            if cmap is None:
                _apply_dead_alive_colors(layer)

        pct_name = f"{_PREVIEW_PREFIX} % Dead Mask"
        try:
            pct_layer = viewer.layers[pct_name]
            pct_layer.data = pct_t
            pct_layer.refresh()
        except (KeyError, ValueError):
            viewer.add_image(
                pct_t, name=pct_name, colormap="inferno",
                contrast_limits=(0, 100), blending="translucent", opacity=0.7,
                visible=False,
            )

        cache["stats_by_frame"][frame_idx] = frame_stats
        cache["pct_maps_by_frame"][frame_idx] = _build_dead_pct_map(frame_stats)
        cache["computed_frames"].add(frame_idx)
        cache["current_frame"] = frame_idx

        self._propagate_org_preview_to_panels()

    def _propagate_org_preview_to_panels(self):
        """Mirror the shared cache state onto every organoid panel and
        re-attach the hover tooltip so they stay in sync."""
        cache = self._org_preview_cache
        label_type_map = cache.get("label_type_map") or {}
        for ct, panel in self.panels.items():
            if ct in self._org_types and panel._preview_seg_t is not None:
                panel._preview_overlay_arr = cache.get("overlay_arr")
                panel._preview_pct_overlay_arr = cache.get("pct_overlay_arr")
                panel._preview_label_arr = cache.get("label_arr")
                panel._preview_current_frame = cache.get("current_frame")
                panel._preview_stats_by_frame = cache["stats_by_frame"]
                panel._preview_pct_maps_by_frame = cache["pct_maps_by_frame"]
                panel._preview_computed_frames = cache["computed_frames"]
                panel._preview_current_thr = cache.get("current_thr")
                panel._preview_label_type_map = label_type_map
                panel._preview_is_timelapse = bool(cache.get("is_timelapse"))
                panel._preview_segs_dict = cache.get("segs_dict", {})
                panel._preview_immune_segs = cache.get("immune_segs", {})
                panel._attach_preview_dead_hover()

    def _notify_organoid_threshold_changed(self, source_ct: str, value: float):
        """Called by an organoid panel when its threshold spinner changes.

        1. Syncs all other organoid panels' spinners.
        2. Triggers an on-demand single-frame Dead/Alive overlay refresh.
        """
        for ct, panel in self.panels.items():
            if ct != source_ct and ct in self._org_types:
                if panel.spin_dead_threshold is not None:
                    panel.spin_dead_threshold.blockSignals(True)
                    panel.spin_dead_threshold.setValue(value)
                    panel.spin_dead_threshold.blockSignals(False)

        viewer = self.viewer
        cache = self._org_preview_cache
        if (
            viewer is not None
            and cache.get("seg_t") is not None
            and cache.get("dead_t") is not None
        ):
            self._refresh_org_preview_for_current_frame(float(value))

    def _on_run_batch_clicked(self):
        """User-triggered batch run — asynchronous."""
        self.run_batch_feature_extraction(interactive=True, block=False)

    def run_batch_feature_extraction(self, interactive=True, skip_existing=False,
                                     block=True, extra_callbacks=None):
        """Run feature extraction for all cell types sequentially.

        ``block=True`` (default, queue) runs synchronously.  ``block=False``
        (GUI) moves the per-cell-type loop to a background worker and
        shows a determinate progress bar scaled at cell-type granularity
        with sample-level updates rendered as labels.

        ``extra_callbacks`` is the queue's chaining hook
        (``{"on_done": cb, "on_failed": cb}``).
        """
        from behav3d.napari._overwrite_prompt import prompt_overwrite_batch
        from qtpy.QtWidgets import QMessageBox as _QMB

        if not self.panels:
            self._log("No cell type panels available.")
            fire_extra_callback(extra_callbacks, "on_failed", "no cell type panels")
            return

        if not block and self._bg.is_running():
            self._log("⚠️ A batch feature-extraction run is already in progress.")
            fire_extra_callback(extra_callbacks, "on_failed", "already running")
            return

        # Persist global organoid threshold before any panel runs
        self._sync_global_threshold_to_params()

        if self.active_killing_panel is not None:
            self.active_killing_panel.set_queue_callback(
                self._queue_active_killing if self._queue_panel is not None else None
            )

        total = len(self.panels)
        self._log(f"Starting batch feature extraction for {total} cell type(s)…")

        all_cts = list(self.panels.keys())
        existing = []
        existing_cts = set()
        out_dir = Path(self.metadata_loader.output_dir)
        for ct in all_cts:
            feat_dir = out_dir / "analysis" / ct / "track_features"
            combined = feat_dir / f"BEHAV3D_{ct}_combined_track_features.csv"
            if combined.exists():
                existing.append(f"{ct} feature data ({combined.name})")
                existing_cts.add(ct)

        changed_cts = [
            ct for ct in all_cts
            if ct in existing_cts and self.panels[ct]._threshold_changed()
        ]

        skip_existing_flag = skip_existing
        overwrite = not skip_existing
        death_only_cts: set[str] = set()
        if existing:
            if interactive:
                extra = None
                if changed_cts:
                    extra = [
                        (
                            "Re-run death only (changed thresholds)",
                            "death_only",
                            _QMB.ActionRole,
                        ),
                    ]
                choice = prompt_overwrite_batch(
                    self,
                    "Overwrite Existing Features?",
                    existing,
                    extra_buttons=extra,
                )
                if choice == "cancel":
                    self._log("Batch feature extraction cancelled.")
                    fire_extra_callback(extra_callbacks, "on_failed", "cancelled")
                    return
                if choice == "death_only":
                    death_only_cts = set(changed_cts)
                    skip_existing_flag = False
                    overwrite = False
                else:
                    skip_existing_flag = choice == "skip"
                    overwrite = not skip_existing_flag
            else:
                overwrite = True

        # Persist + snapshot every panel's Qt widget state on the Qt
        # thread — the worker must not read widgets.
        panel_params: dict[str, dict] = {}
        for ct, panel in self.panels.items():
            panel._persist()
            panel_params[ct] = panel._collect_params()

        SCALE = 100  # sub-units per cell-type step (for ETA granularity)

        def _do_batch(progress_cb=None):
            for i, (ct, panel) in enumerate(self.panels.items()):
                step_label_prefix = f"[{i + 1}/{total}] {ct}"

                def _make_step_cb(step_idx, label_prefix):
                    def _cb(curr, sub_total, label):
                        if progress_cb is None:
                            return
                        if sub_total and sub_total > 0:
                            frac = min(max(curr / sub_total, 0.0), 1.0)
                        else:
                            frac = 0.0
                        try:
                            progress_cb(
                                step_idx * SCALE + int(frac * SCALE),
                                total * SCALE,
                                f"{label_prefix}: {label}" if label else label_prefix,
                            )
                        except Exception:
                            pass
                    return _cb

                if death_only_cts:
                    if ct in death_only_cts:
                        self._log(f"--- [{i + 1}/{total}] Death-only re-run: {ct} ---")
                        panel._run_death_only_for(ct, params=panel_params[ct])
                        self._log(f"Done (death only): {ct}")
                    else:
                        self._log(
                            f"--- [{i + 1}/{total}] Skipping {ct} "
                            "(threshold unchanged) ---"
                        )
                    if progress_cb is not None:
                        try:
                            progress_cb((i + 1) * SCALE, total * SCALE, step_label_prefix)
                        except Exception:
                            pass
                    continue
                if skip_existing_flag and ct in existing_cts:
                    self._log(f"--- [{i + 1}/{total}] Skipping {ct} (existing data) ---")
                    if progress_cb is not None:
                        try:
                            progress_cb((i + 1) * SCALE, total * SCALE, step_label_prefix)
                        except Exception:
                            pass
                    continue
                self._log(f"--- [{i + 1}/{total}] Feature extraction: {ct} ---")
                panel._run_feature_extraction_for(
                    ct,
                    overwrite=overwrite,
                    params=panel_params[ct],
                    progress_cb=_make_step_cb(i, step_label_prefix),
                )
                self._log(f"Done: {ct}")

            self._log("✅ Batch feature extraction finished.")

        if block:
            try:
                _do_batch(progress_cb=None)
                if interactive:
                    _notify_post_extraction(self)
                fire_extra_callback(extra_callbacks, "on_done", None)
            except Exception as e:
                traceback.print_exc()
                self._log(f"❌ Batch feature extraction error: {e}")
                fire_extra_callback(extra_callbacks, "on_failed", str(e))
            finally:
                try:
                    self.results_panel.refresh()
                except Exception:
                    traceback.print_exc()
            return

        def _on_done(result):
            if interactive:
                _notify_post_extraction(self)
            try:
                self.results_panel.refresh()
            except Exception:
                traceback.print_exc()
            fire_extra_callback(extra_callbacks, "on_done", result)

        def _on_failed(err: str):
            self._log(f"❌ Batch feature extraction error: {err}")
            _QMB.critical(
                self, "Batch Feature Extraction Error",
                f"Batch feature extraction failed:\n\n{err}",
            )
            try:
                self.results_panel.refresh()
            except Exception:
                traceback.print_exc()
            fire_extra_callback(extra_callbacks, "on_failed", err)

        self._bg.run(
            fn=_do_batch,
            desc=f"Batch feature extraction ({total} cell types)\u2026",
            progress_row=self.progress_row,
            buttons=[self.btn_run_batch],
            viewer=self.viewer,
            on_done=_on_done,
            on_failed=_on_failed,
        )

    def _sync_global_threshold_to_params(self):
        """Write the organoid dead threshold into behav3d_parameters
        for every organoid cell type, then save to YAML."""
        # Read from first available organoid panel spinner (percent, as shown in the UI)
        thr = 0.0
        for ct in self._org_types:
            if ct in self.panels and self.panels[ct].spin_dead_threshold is not None:
                thr = float(self.panels[ct].spin_dead_threshold.value())
                break
        # Persist as a fraction (0.0-1.0), matching ``percentage_dead_mask``'s scale.
        thr_val = round(thr / 100.0, 4) if thr > 0 else None
        params = self.metadata_loader.behav3d_parameters
        features = params.setdefault("features", {})
        for ct in self._org_types:
            ct_cfg = features.setdefault(ct, {})
            ct_cfg["dead_mask_percentage_threshold"] = thr_val

        out_dir = self.metadata_loader.output_dir
        if out_dir:
            params_path = Path(out_dir) / "behav3d_parameters.yml"
            try:
                with open(params_path, "w") as f:
                    yaml.safe_dump(params, f, sort_keys=False)
            except Exception as e:
                self._log(f"Warning: Could not save parameters: {e}")

    def get_queue_params(self) -> dict:
        """Collect feature extraction params for all panels (used by queue)."""
        # Make sure organoid threshold is synced before snapshotting
        self._sync_global_threshold_to_params()
        return {ct: panel._collect_params() for ct, panel in self.panels.items()}

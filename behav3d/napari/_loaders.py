"""Shared napari-layer loaders used by both the Visualization tab and the
notebook entry point of the manual-correction editor.

These helpers were extracted from
``behav3d.napari._visualization.VisualizationTab`` so that the
``TrackedSegmentEditingPanel`` (notebook) can spin up a fresh napari window
with the same exact layers — raw channels, tracked segments and tracks only,
**without** the regular untracked segments — without duplicating dataset-
loading logic.

All functions accept a napari ``viewer`` and a metadata ``row`` (a
``pd.Series``) and return the list of layer names they added.  Errors are
swallowed and reported through the optional ``log`` callback so that a
missing optional file does not abort the whole load.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import dask.array as da
import pandas as pd

from behav3d.core.metadata import is_multicolor_celltype
from behav3d.napari._track_colors import sync_tracks_colors_to_labels


_CHANNEL_COLORS = ["cyan", "yellow", "green", "red", "blue", "magenta"]


def _noop(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Axis helpers shared by every raw/label/track viewer loader
# ---------------------------------------------------------------------------
def _to_tczyx(img, dim_order: str):
    """Return ``img`` as 5D TCZYX, or ``None`` when its layout is unclear.

    Converted zarrs are already TCZYX. Other arrays are reordered by the
    letters of ``dim_order`` (when it matches ``img.ndim``) and missing
    T/C/Z axes are added as singletons.
    """
    order = str(dim_order or "").strip().upper() or "TCZYX"
    if img.ndim == 5 and order == "TCZYX":
        return img
    if len(order) != img.ndim or len(set(order)) != len(order) or set(order) - set("TCZYX"):
        return img if img.ndim == 5 else None
    present = "".join(ax for ax in "TCZYX" if ax in order)
    if present != order:
        img = da.transpose(img, [order.index(ax) for ax in present])
    index = tuple(slice(None) if ax in present else None for ax in "TCZYX")
    return img[index]


def iter_raw_channels(img, dim_order: str = "TCZYX", squeeze_z: bool = False):
    """Yield ``(channel_index, array)`` for every channel of a raw image.

    Each array is TZYX, or TYX when ``squeeze_z`` is set and the image has a
    single Z plane (2D experiments are stored as TCZYX with Z=1). An image
    whose layout cannot be worked out is yielded whole as channel 0.
    """
    tczyx = _to_tczyx(img, dim_order)
    if tczyx is None:
        yield 0, img
        return
    for c in range(int(tczyx.shape[1])):
        ch = tczyx[:, c]
        if squeeze_z and ch.shape[1] == 1:
            ch = ch[:, 0]
        yield c, ch


def squeeze_label_z(labels):
    """TZYX labels with a single Z plane -> TYX (2D experiments)."""
    if getattr(labels, "ndim", 0) == 4 and labels.shape[1] == 1:
        return labels[:, 0]
    return labels


def tracks_layer_data(tracks_df: pd.DataFrame, is_2d: bool = False):
    """``(N, 4|5)`` array for ``viewer.add_tracks`` from a tracks CSV.

    2D experiments drop ``pixel_position_z`` (always 0) so the tracks line
    up with TYX image and label layers.
    """
    cols = ["TrackID", "position_t", "pixel_position_y", "pixel_position_x"]
    if not is_2d and "pixel_position_z" in tracks_df.columns:
        cols.insert(2, "pixel_position_z")
    return tracks_df[cols].values


def configure_viewer_for_dimensionality(viewer, is_2d: bool) -> None:
    """2D experiments: stay in 2D display and label the axes t/y/x.

    The napari 2D/3D view button is disabled for 2D data, where the 3D
    view would render time as depth. Best effort: the button lives in
    private napari Qt internals.
    """
    try:
        if is_2d:
            viewer.dims.ndisplay = 2
            if viewer.dims.ndim == 3:
                viewer.dims.axis_labels = ("t", "y", "x")
    except Exception:
        pass
    try:
        qt_viewer = getattr(viewer.window, "_qt_viewer", None) or getattr(viewer.window, "qt_viewer", None)
        button = qt_viewer.viewerButtons.ndisplayButton
        button.setEnabled(not is_2d)
        button.setToolTip(
            "3D view is disabled for 2D experiments" if is_2d else "Toggle 2D / 3D view"
        )
    except Exception:
        pass


def detect_cell_type_columns(row: pd.Series) -> Dict[str, str]:
    """Return ``{cell_type_name: prefix}`` for all detected cell types in a row.

    Mirrors ``VisualizationTab._detect_cell_type_columns`` so that the
    notebook helper sees the same set of cell types as the napari plugin.
    """
    pattern = re.compile(r"^(or|im|ot)_(.+?)_line_condition$")
    out: Dict[str, str] = {}
    for col in row.index:
        m = pattern.match(col)
        if m:
            prefix, ct_name = m.group(1), m.group(2)
            out[ct_name] = prefix
            
    merged_pattern = re.compile(r"^(.+?)_(segments_image_path|tracks_image_path|tracks_csv_path)$")
    for col in row.index:
        if col.startswith(("or_", "im_", "ot_")):
            continue
        m = merged_pattern.match(col)
        if m:
            ct_name = m.group(1)
            if ct_name.endswith("_merged") or ct_name.endswith("_grouped"):
                if ct_name not in out:
                    out[ct_name] = ""
                    
    return out


def cell_types_with_tracked_segments(
    row: pd.Series, include_multicolor: bool = False
) -> List[str]:
    """Return cell-type names that have a non-empty ``*_tracks_image_path``.

    Per-channel multicolor types (``*_N_multicolor``) are excluded by default
    because the merged/grouped output is the usual edit target.  Pass
    ``include_multicolor=True`` to also list the individual channels — used when
    the user explicitly chooses to edit the independent tracks and rebuild the
    merged output afterwards.
    """
    out: List[str] = []
    for ct_name, prefix in detect_cell_type_columns(row).items():
        if is_multicolor_celltype(ct_name) and not include_multicolor:
            continue  # skip individual channels — only merged/grouped is editable
        col = f"{prefix}_{ct_name}_tracks_image_path" if prefix else f"{ct_name}_tracks_image_path"
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            p = Path(str(v).strip())
            if p.exists():
                out.append(ct_name)
    return out


def tracked_segments_paths_for(row: pd.Series, cell_type: str) -> Optional[Dict[str, Path]]:
    """Return ``{"zarr": ..., "csv": ..., "prefix": ...}`` for the given
    cell type, or ``None`` if no tracked image is configured."""
    for prefix in ("or", "im", "ot"):
        zcol = f"{prefix}_{cell_type}_tracks_image_path"
        ccol = f"{prefix}_{cell_type}_tracks_csv_path"
        v = row.get(zcol)
        if pd.notna(v) and str(v).strip():
            return {
                "zarr": Path(str(v).strip()),
                "csv": Path(str(row.get(ccol)).strip())
                if pd.notna(row.get(ccol)) and str(row.get(ccol)).strip()
                else None,
                "prefix": prefix,
            }
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_raw_into_viewer(
    viewer,
    sample_name: str,
    row: pd.Series,
    output_dir: str = "",
    log: Callable[[str], None] = _noop,
    display_settings: Optional[Dict] = None,
) -> List[str]:
    """Load the raw image as one Image layer per channel.  Returns layer names."""
    from behav3d.io.formats.zarr import load_zarr

    raw_path = str(row.get("raw_image_path", "")).strip()
    if not raw_path:
        log(f"  ⚠️ No raw_image_path for {sample_name}")
        return []

    raw_p = Path(raw_path)
    candidates = [
        raw_p if raw_p.suffix == ".zarr" else None,
        Path(output_dir, f"{sample_name}.zarr") if output_dir else None,
        raw_p.with_suffix(".zarr"),
    ]
    dask_img = None
    for zp in candidates:
        if zp and zp.exists():
            try:
                dask_img = load_zarr(zp)
                log(f"  📂 Loaded raw zarr: {zp}  shape={dask_img.shape}")
                break
            except Exception as exc:
                log(f"  ⚠️ Error loading {zp}: {exc}")
    if dask_img is None and raw_p.exists():
        try:
            from behav3d.io.images import load_image

            img = load_image(raw_p)
            if not isinstance(img, da.Array):
                img = da.from_array(img, chunks=(1,) + img.shape[1:])
            dask_img = img
            log(f"  📂 Loaded raw image: {raw_p}  shape={dask_img.shape}")
        except Exception as exc:
            log(f"  ❌ Could not load raw image: {exc}")
            return []
    if dask_img is None:
        log(f"  ⚠️ Raw image not found: {raw_path}")
        return []

    dim_order = str(row.get("dimension_order", "TCZYX")).strip() or "TCZYX"
    saved_channels = display_settings or {}
    added: List[str] = []
    # squeeze_z stays off: the manual-correction editor that uses these
    # loaders still works on TZYX labels (2D editing is not supported yet).
    for c, ch_data in iter_raw_channels(dask_img, dim_order, squeeze_z=False):
        saved = saved_channels.get(c) or saved_channels.get(str(c))
        color = (saved["colormap"] if saved and "colormap" in saved
                 else _CHANNEL_COLORS[c % len(_CHANNEL_COLORS)])
        name = f"{sample_name} – Ch{c}"
        add_kwargs = dict(
            name=name, colormap=color, blending="additive", visible=True,
        )
        if saved and "contrast_limits" in saved:
            add_kwargs["contrast_limits"] = tuple(saved["contrast_limits"])
        viewer.add_image(ch_data, **add_kwargs)
        log(f"    + Image layer: {name}  ({color})")
        added.append(name)
    return added


def load_tracked_segments_into_viewer(
    viewer,
    sample_name: str,
    row: pd.Series,
    log: Callable[[str], None] = _noop,
    only_cell_type: Optional[str] = None,
    as_numpy: bool = False,
) -> List[str]:
    """Load only the *tracked* segments (one Labels layer per cell type).

    Parameters
    ----------
    only_cell_type:
        If given, restrict the load to the matching cell type — used by the
        manual-correction editor when it wants exactly one layer.
    as_numpy:
        If True, eagerly materialise the labels array as ``np.ndarray``.
        The editor needs this so per-frame mutations work in-place.
    """
    import numpy as np
    from behav3d.io.formats.zarr import load_zarr

    added: List[str] = []
    for ct_name, prefix in detect_cell_type_columns(row).items():
        if only_cell_type is not None and ct_name != only_cell_type:
            continue
        col = f"{prefix}_{ct_name}_tracks_image_path" if prefix else f"{ct_name}_tracks_image_path"
        val = row.get(col)
        if pd.isna(val) or not str(val).strip():
            continue
        p = Path(str(val).strip())
        if not p.exists():
            log(f"    - Skipping {ct_name} tracked segments: file not found ({p})")
            continue
        try:
            if p.suffix == ".zarr":
                seg = load_zarr(p)
            else:
                from behav3d.io.images import load_image

                seg = load_image(p)
                if not isinstance(seg, da.Array):
                    seg = da.from_array(seg, chunks=(1,) + seg.shape[1:])
            if as_numpy:
                log(
                    f"    Materialising tracked segments for editing "
                    f"(shape={tuple(seg.shape)}, dtype={seg.dtype})…"
                )
                seg = np.asarray(seg)
            name = f"{sample_name} – {ct_name} tracked segments"
            # A multicolor channel is normally hidden, but when it is the
            # explicitly requested layer (editing an independent channel) it
            # must be visible.
            explicit = only_cell_type is not None and ct_name == only_cell_type
            viewer.add_labels(
                seg,
                name=name,
                visible=explicit or not is_multicolor_celltype(ct_name),
            )
            log(f"    + Tracked Labels layer: {name}")
            added.append(name)
        except Exception as exc:
            log(f"    ⚠️ Could not load tracked segments for {ct_name}: {exc}")
    return added


def load_tracks_into_viewer(
    viewer,
    sample_name: str,
    row: pd.Series,
    visible: bool = False,
    log: Callable[[str], None] = _noop,
    only_cell_type: Optional[str] = None,
) -> List[str]:
    """Load the per-cell-type Tracks layers (CSV-derived)."""
    added: List[str] = []
    for ct_name, prefix in detect_cell_type_columns(row).items():
        if only_cell_type is not None and ct_name != only_cell_type:
            continue
        col = f"{prefix}_{ct_name}_tracks_csv_path" if prefix else f"{ct_name}_tracks_csv_path"
        val = row.get(col)
        if pd.isna(val) or not str(val).strip():
            continue
        p = Path(str(val).strip())
        if not p.exists():
            log(f"    - Skipping {ct_name} tracks: file not found ({p})")
            continue
        try:
            df = pd.read_csv(p)
            required = {"TrackID", "position_t", "pixel_position_y", "pixel_position_x"}
            if not required.issubset(set(df.columns)):
                log(
                    f"    ⚠️ Tracks CSV for {ct_name} missing columns "
                    f"{required - set(df.columns)}"
                )
                continue
            if "pixel_position_z" in df.columns:
                arr = df[
                    ["TrackID", "position_t", "pixel_position_z",
                     "pixel_position_y", "pixel_position_x"]
                ].values
            else:
                arr = df[
                    ["TrackID", "position_t", "pixel_position_y", "pixel_position_x"]
                ].values
            name = f"{sample_name} – {ct_name} tracks"
            viewer.add_tracks(arr, name=name, visible=visible)
            log(f"    + Tracks layer: {name}")
            added.append(name)

            # Color each track the same as its cell's tracked-segments
            # label, instead of napari's default track_id gradient.
            seg_layer_name = f"{sample_name} – {ct_name} tracked segments"
            if seg_layer_name in viewer.layers:
                sync_tracks_colors_to_labels(
                    viewer.layers[name], viewer.layers[seg_layer_name]
                )
        except Exception as exc:
            log(f"    ⚠️ Could not load tracks for {ct_name}: {exc}")
    return added

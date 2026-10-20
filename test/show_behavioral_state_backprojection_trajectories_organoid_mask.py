from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scanpy as sc
import napari

from behav3d.analysis.behavior.state.visualization.backprojection import (
    _align_labels_to_raw_shape_for_view,
    _ensure_behavioral_state_backprojection_for_sample,
    _extract_state_code_color_map,
    _extract_state_label_map,
    _resolve_behavioral_states_h5ad_path,
    _resolve_raw_image_path,
)
from behav3d.analysis.behavior.track.visualization.backprojection import (
    add_track_cluster_trajectory_layers,
)
from behav3d.analysis.behavior.track.visualization.plots.exemplar_coordinate_utils import (
    merge_coordinate_columns_into_obs,
    resolve_exemplar_positions_csv_path,
)
from behav3d.analysis.organoid_analysis import resolve_organoid_seg_path
from behav3d.io.images import load_image

# Behavioral-state trajectory backprojection (see
# show_behavioral_state_backprojection_trajectories.py) plus an organoid mask layer.
# Self-contained on purpose: the sibling scripts run an interactive "# %%" cell at
# import time, so they cannot be imported from here.


# %%
# Interactive configuration
INTERACTIVE_BEHAV3D_FOLDER = "/Users/s.deblank-3/Documents/LowDensity_Multicolor_Maria"
INTERACTIVE_SAMPLE_NAME = "BHVD_SB1_Exp012_Img001"
INTERACTIVE_CELL_TYPE = "tcells_merged"
INTERACTIVE_STATE_COL = "full_behavioral_state"
# Organoid type whose mask is shown (e.g. "organoid"), or "all_organoids" for the
# merged mask written by the all-organoids tracking step.
INTERACTIVE_ORGANOID_TYPE = "organoid"
INTERACTIVE_ORGANOID_MASK_PATH = None  # optional explicit override
# Raw image lives outside this folder (metadata.csv points to LowDensity_MultiColor).
INTERACTIVE_RAW_IMAGE_PATH = None  # None = use the path from metadata.csv
INTERACTIVE_MASK_OPACITY = 0.3
INTERACTIVE_MASK_COLOR = "white"
INTERACTIVE_MIN_RUN_LENGTH = 1
INTERACTIVE_TAIL_LENGTH = None  # None = each state layer keeps its full time span
INTERACTIVE_VERBOSE = True

_PIXEL_POSITION_TRIPLETS = (
    ("pixel_position_z", "pixel_position_y", "pixel_position_x"),
    ("pixel_position_y", "pixel_position_x"),
)


# %%
# Input resolution
def _resolve_raw_path(output_dir: Path, sample_name: str, override=None, verbose=True):
    if override is not None:
        return Path(override)
    path = _resolve_raw_image_path(output_dir=output_dir, sample_name=sample_name, verbose=verbose)
    if path is not None and Path(path).exists():
        return Path(path)
    metadata_path = Path(output_dir, "metadata.csv")
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        rows = metadata[metadata["sample_name"].astype(str) == sample_name]
        if len(rows) > 0 and Path(str(rows["raw_image_path"].iloc[0])).exists():
            return Path(str(rows["raw_image_path"].iloc[0]))
    raise FileNotFoundError(f"Could not find raw image for sample '{sample_name}'.")


def _resolve_organoid_mask_path(output_dir: Path, sample_name: str, organoid_type: str, override=None) -> Path:
    if override is not None:
        path = Path(override)
    else:
        metadata_path = Path(output_dir, "metadata.csv")
        sample_metadata = None
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            sample_metadata = metadata[metadata["sample_name"].astype(str) == sample_name]
        path = resolve_organoid_seg_path(
            sample_metadata=sample_metadata,
            org_type=organoid_type,
            img_outdir=Path(output_dir, "images", sample_name),
            sample_name=sample_name,
        )
    if not path.exists():
        raise FileNotFoundError(f"Could not find organoid mask for sample '{sample_name}' at '{path}'.")
    return path


def _load_sample_obs_with_pixels(output_dir: Path, cell_type: str, sample_name: str) -> pd.DataFrame:
    h5ad_path = _resolve_behavioral_states_h5ad_path(output_dir=output_dir, cell_type=cell_type)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Full behavioral-state h5ad is missing: '{h5ad_path}'.")
    adata = sc.read_h5ad(h5ad_path)
    obs_cols = set(adata.obs.columns)
    if not any(set(triplet).issubset(obs_cols) for triplet in _PIXEL_POSITION_TRIPLETS):
        merge_coordinate_columns_into_obs(
            adata=adata,
            positions_csv_path=resolve_exemplar_positions_csv_path(output_dir=output_dir, cell_type=cell_type),
            sample_col="sample_name",
            track_col="TrackID",
            time_col="position_t",
        )
    return adata.obs[adata.obs["sample_name"].astype("string") == sample_name].copy()


# %%
# Contiguous same-state trajectory segments
def prepare_state_trajectory_data(sample_obs, state_col, min_run_length=1, track_col="TrackID", time_col="position_t"):
    """state label -> [N, 4|5] array of [segment_id, t, (z), y, x] for add_tracks.

    A segment is a run of consecutive frames of one track sharing one state."""
    pos = next((c for c in _PIXEL_POSITION_TRIPLETS if all(k in sample_obs.columns for k in c)), None)
    if pos is None:
        raise ValueError("sample_obs has no pixel_position_[z]/y/x columns.")
    obs = sample_obs[[track_col, time_col, state_col, *pos]].copy()
    obs[state_col] = obs[state_col].astype("string").str.strip()
    obs = obs.dropna(subset=[track_col, time_col, *pos])
    obs = obs[obs[state_col].notna() & (obs[state_col].str.len() > 0)]
    if obs.empty:
        return {}
    obs = obs.sort_values([track_col, time_col], kind="mergesort").reset_index(drop=True)

    tr, t, st = obs[track_col].to_numpy(), obs[time_col].to_numpy(), obs[state_col].to_numpy()
    continues = np.zeros(len(obs), dtype=bool)
    continues[1:] = (tr[1:] == tr[:-1]) & (st[1:] == st[:-1]) & (t[1:] - t[:-1] == 1)
    obs["__seg"] = np.cumsum(~continues) - 1
    if min_run_length > 1:
        obs = obs[obs.groupby("__seg")[time_col].transform("size") >= min_run_length]

    cols = ["__seg", time_col, *pos]
    return {
        str(label): group[cols].to_numpy(dtype=np.float64, copy=True)
        for label, group in obs.groupby(state_col, observed=True, sort=False)
    }


# %%
# Payload / viewer
def build_payload(
    sample_name,
    output_dir,
    cell_type,
    state_col="full_behavioral_state",
    organoid_type="all_organoids",
    organoid_mask_path=None,
    raw_image_path=None,
    min_run_length=1,
    tail_length=None,
    verbose=True,
):
    output_dir = Path(output_dir)
    raw_path = _resolve_raw_path(output_dir, sample_name, raw_image_path, verbose)
    raw_img = load_image(raw_path)
    if raw_img.ndim != 5:
        raise ValueError(f"Raw image must have shape (T, C, Z, Y, X), got {tuple(raw_img.shape)}.")
    raw_channels = [raw_img[:, ch, ...] for ch in range(raw_img.shape[1])]

    state_path = _ensure_behavioral_state_backprojection_for_sample(
        sample_name=sample_name,
        output_dir=output_dir,
        cell_type=cell_type,
        state_col=state_col,
        track_col="TrackID",
        time_col="position_t",
        sample_col="sample_name",
        background_value=0,
        enforce_time_coverage=True,
        refresh_if_stale=True,
        verbose=verbose,
    )
    state_img_view = _align_labels_to_raw_shape_for_view(
        load_image(state_path), raw_channels[0], layer_name="behavioral_state_class", verbose=verbose
    )

    mask_path = _resolve_organoid_mask_path(output_dir, sample_name, organoid_type, organoid_mask_path)
    organoid_mask_view = (
        _align_labels_to_raw_shape_for_view(
            load_image(mask_path), raw_channels[0], layer_name="organoid_mask", verbose=verbose
        )
        != 0
    ).astype(np.uint8)

    sample_obs = _load_sample_obs_with_pixels(output_dir, cell_type, sample_name)
    return {
        "sample_name": sample_name,
        "state_col": state_col,
        "raw_channels": raw_channels,
        "state_img_view": state_img_view,
        "organoid_mask_view": organoid_mask_view,
        "mask_path": Path(mask_path),
        "label_map": _extract_state_label_map(state_path),
        "code_colors": _extract_state_code_color_map(state_path),
        "trajectory_data": prepare_state_trajectory_data(sample_obs, state_col, min_run_length),
        "tail_length": tail_length,
    }


def launch_viewer(payload, mask_opacity=0.3, mask_color="white", run=True):
    viewer = napari.Viewer()
    for ch, channel_img in enumerate(payload["raw_channels"]):
        viewer.add_image(channel_img, name=f"raw_ch{ch}")
    viewer.add_labels(payload["state_img_view"], name="behavioral_state_class", visible=False)

    # Binary mask: label 1 gets a single flat color so the organoids read as one region.
    viewer.add_labels(
        payload["organoid_mask_view"],
        name="organoid_mask",
        opacity=float(mask_opacity),
        blending="translucent",
    ).colormap = napari.utils.DirectLabelColormap(
        color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: mask_color}
    )

    layers = add_track_cluster_trajectory_layers(
        viewer,
        trajectory_data=payload["trajectory_data"],
        code_colors=payload["code_colors"],
        label_map=payload["label_map"],
        output_col=payload["state_col"],
        tail_length=payload["tail_length"],
        visible=True,
    )
    for layer in layers:
        layer.blending = "opaque"
    print(
        f"Sample '{payload['sample_name']}': {len(layers)} state-trajectory layer(s), "
        f"organoid mask from '{payload['mask_path'].name}'."
    )
    if run:
        napari.run()
    return viewer


# %%
# Interactive usage: run this cell to build the payload, then the next to open napari.
payload = build_payload(
    sample_name=INTERACTIVE_SAMPLE_NAME,
    output_dir=Path(INTERACTIVE_BEHAV3D_FOLDER).expanduser(),
    cell_type=INTERACTIVE_CELL_TYPE,
    state_col=INTERACTIVE_STATE_COL,
    organoid_type=INTERACTIVE_ORGANOID_TYPE,
    organoid_mask_path=INTERACTIVE_ORGANOID_MASK_PATH,
    raw_image_path=INTERACTIVE_RAW_IMAGE_PATH,
    min_run_length=INTERACTIVE_MIN_RUN_LENGTH,
    tail_length=INTERACTIVE_TAIL_LENGTH,
    verbose=INTERACTIVE_VERBOSE,
)


# %%
viewer = launch_viewer(
    payload,
    mask_opacity=INTERACTIVE_MASK_OPACITY,
    mask_color=INTERACTIVE_MASK_COLOR,
    run=True,
)

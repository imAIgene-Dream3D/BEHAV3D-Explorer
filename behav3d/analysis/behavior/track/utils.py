from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from behav3d.analysis.behavior.utils import _sanitize_filename_token


def _winfo(prefix, message):
    print(f"[{prefix}] INFO {message}")


def _ordered_unique(values):
    out = []
    seen = set()
    for value in list(values or []):
        text = str(value).strip()
        if text == "" or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


# Canonical shared output folder for every track-classification method (DTW
# distance clustering, bouts/proportion feature clustering, classifier-apply,
# and the legacy "original BEHAV3D" feature-DTW method).
TRACK_TRAJECTORIES_SUBDIR_NAME = "behavioral_trajectories"
# Older projects may still have this misspelled folder on disk (missing the
# second "i" in "behavioral"); it is only ever read as a fallback, never written.
_LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME = "behavorial_trajectories"


@dataclass(frozen=True)
class TrackPaths:
    output_dir: Path
    analysis_outdir: Path
    state_outdir: Path
    outfolder: Path
    clustering_outfolder: Path
    behavior_proportions_outfolder: Path
    behavior_comparisons_outfolder: Path
    example_tracks_outfolder: Path
    example_tracks_backprojection_outfolder: Path
    classification_outfolder: Path
    original_behav3d_outfolder: Path


def _peek_track_outfolder(output_dir, cell_type, *, output_subdir_name=None):
    """Read-only lookup of the track-classification outfolder — no directories
    are created. Applies the same canonical-name-with-legacy-fallback logic as
    `_resolve_track_paths`, so UI code doing existence checks (e.g. enabling a
    "view results" button, autofilling a path on tab switch) can be called
    freely/often without side effects, while getting the same folder that
    `_resolve_track_paths` would resolve to for reading/writing.
    """
    if cell_type is None or len(str(cell_type).strip()) == 0:
        raise ValueError("cell_type is required.")

    root = Path(output_dir).expanduser()
    analysis_outdir = root / "analysis" / str(cell_type)

    if output_subdir_name is None:
        outfolder = analysis_outdir / TRACK_TRAJECTORIES_SUBDIR_NAME
        if not outfolder.exists():
            legacy_outfolder = analysis_outdir / _LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME
            if legacy_outfolder.exists():
                outfolder = legacy_outfolder
    else:
        outfolder = analysis_outdir / str(output_subdir_name)
    return outfolder


def _resolve_track_paths(output_dir, cell_type, *, output_subdir_name=None):
    """Resolve canonical track-classification paths under analysis/<cell_type>/.

    Shared by every track-classification method so they all write into the same
    folder layout regardless of which clustering basis (DTW vs. bouts) or which
    step (clustering, classifier-apply, legacy feature-DTW) produced them.

    When `output_subdir_name` is not given, resolves to the canonical
    "behavioral_trajectories" folder — falling back to the legacy misspelled
    "behavorial_trajectories" folder only if that is the sole one already
    present on disk (older projects), without renaming or migrating anything.
    Callers that need a different/nested subfolder (e.g. the legacy method's
    own raw-output staging dir) can pass an explicit `output_subdir_name`,
    which bypasses the fallback entirely.
    """
    if cell_type is None or len(str(cell_type).strip()) == 0:
        raise ValueError("cell_type is required.")

    root = Path(output_dir).expanduser()
    analysis_outdir = root / "analysis" / str(cell_type)
    analysis_outdir.mkdir(parents=True, exist_ok=True)

    state_outdir = analysis_outdir / "behavioral_states"

    outfolder = _peek_track_outfolder(root, cell_type, output_subdir_name=output_subdir_name)
    outfolder.mkdir(parents=True, exist_ok=True)

    example_tracks_outfolder = outfolder / "example_tracks"

    return TrackPaths(
        output_dir=root,
        analysis_outdir=analysis_outdir,
        state_outdir=state_outdir,
        outfolder=outfolder,
        clustering_outfolder=outfolder / "clustering",
        behavior_proportions_outfolder=outfolder / "behavior_proportions",
        behavior_comparisons_outfolder=outfolder / "behavior_comparisons",
        example_tracks_outfolder=example_tracks_outfolder,
        example_tracks_backprojection_outfolder=example_tracks_outfolder / "backprojection",
        classification_outfolder=outfolder / "classification",
        original_behav3d_outfolder=outfolder / "original_behav3d",
    )


def get_dtaidistance_track_trajectories_filename(cell_type):
    cell_token = _sanitize_filename_token(cell_type, fallback="cell")
    return f"BEHAV3D_{cell_token}_behavioral_trajectories.h5ad"


def _default_behavioral_states_path(output_dir, cell_type):
    return (
        Path(output_dir).expanduser()
        / "analysis"
        / str(cell_type)
        / "behavioral_states"
        / f"BEHAV3D_{cell_type}_behavioral_states.h5ad"
    )


def _resolve_optional_int(value):
    text = str(value).strip()
    return None if text == "" else int(text)


def _resolve_optional_float(value):
    text = str(value).strip()
    return None if text == "" else float(text)


def _filter_tracks_for_dtaidistance(
    adata,
    *,
    groupby_cols=("sample_name", "TrackID"),
    time_col="position_t",
    trajectory_size=None,
    min_length=None,
    trim_mode="last",
    split_long_tracks=False,
    window_col="trajectory_window_id",
):
    """Filter trajectories for one-hot dtaidistance clustering.

    When ``split_long_tracks`` is enabled, over-length tracks are divided into
    fixed-size, non-overlapping windows while keeping the original ``TrackID``.
    The added ``window_col`` can be included in downstream grouping to treat
    each window as an independent analysis trajectory.
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    missing = [col for col in groupby_cols if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing groupby_cols in adata.obs: {missing}")
    if str(time_col) not in adata.obs.columns:
        raise KeyError(f"'{time_col}' not found in adata.obs.")
    if str(window_col) in groupby_cols:
        raise ValueError("window_col must not already be present in groupby_cols.")

    obs = adata.obs.copy()
    obs["_orig_idx"] = np.arange(len(obs))
    obs = obs.sort_values(groupby_cols + [str(time_col), "_orig_idx"])

    if min_length is not None:
        group_sizes = obs.groupby(groupby_cols, observed=True).size()
        keep_groups = group_sizes[group_sizes >= int(min_length)].index
        obs["_keep"] = obs.set_index(groupby_cols).index.isin(keep_groups)
        obs = obs.loc[obs["_keep"]].copy()

    if trajectory_size is not None:
        trajectory_size = int(trajectory_size)
        if trajectory_size <= 0:
            raise ValueError("trajectory_size must be positive when supplied.")
        trim_mode = str(trim_mode).strip().lower()
        if trim_mode not in {"first", "last"}:
            raise ValueError("trim_mode must be 'first' or 'last'.")
        obs["_rank"] = obs.groupby(groupby_cols, observed=True).cumcount()
        sizes = obs.groupby(groupby_cols, observed=True)["_rank"].transform("max") + 1
        if bool(split_long_tracks):
            n_windows = sizes // trajectory_size
            full_window_rows = n_windows * trajectory_size
            if trim_mode == "first":
                obs["_window_rank"] = obs["_rank"]
            else:
                obs["_window_rank"] = obs["_rank"] - (sizes - full_window_rows)
            obs = obs.loc[(obs["_window_rank"] >= 0) & (obs["_window_rank"] < full_window_rows)].copy()
            obs[window_col] = (obs["_window_rank"] // trajectory_size).astype(int)
        elif trim_mode == "first":
            obs = obs.loc[obs["_rank"] < trajectory_size]
        else:
            obs = obs.loc[obs["_rank"] >= (sizes - trajectory_size)]

    idx = obs.index
    adata_out = adata[idx].copy()
    if bool(split_long_tracks) and trajectory_size is not None:
        adata_out.obs[window_col] = obs.loc[idx, window_col].to_numpy()
    for col in ["_orig_idx", "_keep", "_rank", "_window_rank"]:
        if col in adata_out.obs.columns:
            adata_out.obs.drop(columns=[col], inplace=True)
    return adata_out


def _build_identity_cluster_mapping_from_obs(obs, cluster_col="ClusterID"):
    if cluster_col not in obs.columns:
        return {}
    values = pd.Series(obs[cluster_col]).dropna().astype(str).unique().tolist()
    values = sorted(values, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
    return {str(value): str(value) for value in values}

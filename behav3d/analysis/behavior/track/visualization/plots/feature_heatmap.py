"""Cluster x feature summary heatmap for trajectory (track) clusters.

For every track, each selected per-timepoint feature is averaged over the
timepoints that went into its trajectory (``position_t_min``..``position_t_max``
of the track model, so trimmed / windowed tracks only use their own window).
Clusters are then summarised as the mean of their tracks' means, so every track
weighs the same regardless of length.

Values come from the per-timepoint track-features CSV (all raw features). Features
that only exist in the behavioral-state adata (e.g. windowed state features) are
taken from its ``X`` / numeric ``obs`` instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb

from behav3d.analysis.behavior.state.utils import (
    _apply_state_order,
    _get_classification_state_colors,
    _get_classification_state_order,
    _normalize_label_color_map,
    _resolve_positions_csv_path,
)
from behav3d.analysis.behavior.track.utils import _default_behavioral_states_path, _winfo
from behav3d.analysis.behavior.utils import _mixed_label_sort_key

ID_COLS = ("sample_name", "TrackID")
TIME_COL = "position_t"
FEATURE_HEATMAP_SUBDIR = "feature_heatmaps"

# Identifier / coordinate columns that are numeric but never meaningful features.
_NON_FEATURE_COLS = {
    "TrackID", "position_t", "exp_nr", "well", "SegmentID", "ID", "Unnamed: 0",
    "position_t_min", "position_t_max", "n_timepoints", "trajectory_window_id",
}


def _track_id_key(values):
    """TrackIDs as comparable strings: 7, 7.0 and "7" all become "7"."""
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    is_int = num.notna() & np.isfinite(num) & (num == np.round(num))
    out = s.astype(str).str.strip()
    out[is_int] = num[is_int].astype(np.int64).astype(str)
    return out.to_numpy()


def _as_list(value):
    # Lists round-trip through h5ad as numpy arrays.
    return [] if value is None else [str(v) for v in np.atleast_1d(np.asarray(value, dtype=object)).tolist()]


def _state_file_feature_info(state_path):
    """Cheaply read (without loading the matrix) the state adata's feature names.

    Returns ``(x_features, numeric_obs_cols, state_features)`` where
    ``state_features`` are the continuous + binary columns the states were built on.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        return [], [], []
    import h5py

    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem

    x_features, obs_numeric, state_features = [], [], []
    with h5py.File(state_path, "r") as f:
        if "var" in f:
            x_features = [str(v) for v in read_elem(f["var"]).index]
        obs = f.get("obs")
        if obs is not None:
            for key in obs.keys():
                node = obs[key]
                # Categorical columns are groups; plain numeric columns are datasets.
                if isinstance(node, h5py.Dataset) and node.dtype.kind in "biuf":
                    obs_numeric.append(str(key))
        pre = f.get("uns/preprocessing")
        if pre is not None:
            for key in ("continuous_feature_cols", "binary_cols_to_merge"):
                if key in pre:
                    state_features.extend(_as_list(read_elem(pre[key])))
    return x_features, obs_numeric, state_features


def _per_timepoint_path(output_dir, cell_type, track_adata=None):
    """The per-timepoint h5ad the trajectory model was built from (behavioral-state
    file, or the features-only trajectory-features file); default: state file."""
    if track_adata is not None:
        meta = track_adata.uns.get("dtai_trajectory_clustering", {}) or {}
        src = meta.get("source_adata_full_path")
        if src and Path(str(src)).exists():
            return Path(str(src))
    return _default_behavioral_states_path(output_dir, cell_type)


def list_track_feature_candidates(output_dir, cell_type, track_adata=None):
    """Numeric per-timepoint columns that can be shown in the heatmap.

    Returns ``(candidates, defaults)``. ``defaults`` are the features the
    trajectory model was clustered on when known ('dtw features' basis),
    otherwise the features the behavioral states were built from.
    """
    candidates = []
    try:
        csv_path = _resolve_positions_csv_path(output_dir=output_dir, cell_type=cell_type)
        head = pd.read_csv(csv_path, nrows=200, low_memory=False)
        candidates = [
            str(c) for c in head.columns
            if pd.api.types.is_numeric_dtype(head[c]) and str(c) not in _NON_FEATURE_COLS
            and not str(c).startswith(("position_", "pixel_", "_"))
        ]
    except Exception:
        pass
    x_features, obs_numeric, state_features = _state_file_feature_info(
        _per_timepoint_path(output_dir, cell_type, track_adata)
    )
    for col in list(x_features) + list(obs_numeric):
        if col not in candidates and col not in _NON_FEATURE_COLS and not col.startswith("_"):
            candidates.append(col)

    defaults = []
    if track_adata is not None:
        meta = track_adata.uns.get("dtai_trajectory_clustering", {}) or {}
        defaults = _as_list(meta.get("continuous_feature_cols")) + _as_list(meta.get("binary_feature_cols"))
    if not defaults:
        defaults = list(state_features)
    defaults = [c for c in dict.fromkeys(defaults) if c in candidates]
    return candidates, defaults


def _load_timepoint_features(output_dir, cell_type, features, track_adata=None, fallback_adata=None):
    """Per-timepoint DataFrame with ID_COLS, TIME_COL and the requested features."""
    features = [str(f) for f in features]
    keep = list(ID_COLS) + [TIME_COL]
    frames = []
    found = set()
    try:
        csv_path = _resolve_positions_csv_path(output_dir=output_dir, cell_type=cell_type)
        header = pd.read_csv(csv_path, nrows=0).columns
        csv_feats = [f for f in features if f in header]
        if csv_feats:
            df = pd.read_csv(csv_path, usecols=keep + csv_feats, low_memory=False)
            frames.append(df)
            found.update(csv_feats)
    except FileNotFoundError:
        pass

    missing = [f for f in features if f not in found]
    if missing:
        a = fallback_adata
        if a is None:
            state_path = _per_timepoint_path(output_dir, cell_type, track_adata)
            if state_path.exists():
                import anndata as ad

                a = ad.read_h5ad(state_path)
        if a is not None:
            cols = {}
            for f in missing:
                if f in a.var_names:
                    x = a[:, f].X
                    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
                    cols[f] = np.asarray(x, dtype=float).ravel()
                elif f in a.obs.columns:
                    cols[f] = pd.to_numeric(a.obs[f], errors="coerce").to_numpy(dtype=float)
            if cols:
                df_state = a.obs[keep].reset_index(drop=True).copy()
                for f, v in cols.items():
                    df_state[f] = v
                frames.append(df_state)
                found.update(cols)

    still_missing = [f for f in features if f not in found]
    if still_missing:
        raise KeyError(f"Features not found in the track-features CSV or the behavioral-state adata: {still_missing}")

    out = None
    for df in frames:
        df = df.copy()
        df["sample_name"] = df["sample_name"].astype(str).str.strip()
        df["TrackID"] = _track_id_key(df["TrackID"])
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
        df = df.drop_duplicates(subset=keep)
        out = df if out is None else out.merge(df, on=keep, how="outer")
    for f in features:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    return out


def compute_track_feature_means(track_adata, df_timepoints, features, *, cluster_key):
    """One row per track (same order as ``track_adata.obs``): cluster + feature means."""
    obs = track_adata.obs.copy()
    missing_ids = [c for c in ID_COLS if c not in obs.columns]
    if missing_ids:
        raise KeyError(f"Track model is missing identifier columns {missing_ids}.")
    tracks = pd.DataFrame({
        "_track_row": np.arange(len(obs)),
        "sample_name": obs["sample_name"].astype(str).str.strip().to_numpy(),
        "TrackID": _track_id_key(obs["TrackID"]),
        cluster_key: obs[cluster_key].astype(str).to_numpy(),
    })
    has_window = "position_t_min" in obs.columns and "position_t_max" in obs.columns
    if has_window:
        tracks["_tmin"] = pd.to_numeric(obs["position_t_min"], errors="coerce").to_numpy()
        tracks["_tmax"] = pd.to_numeric(obs["position_t_max"], errors="coerce").to_numpy()

    merged = tracks.merge(df_timepoints, on=list(ID_COLS), how="inner")
    if has_window:
        t = pd.to_numeric(merged[TIME_COL], errors="coerce")
        merged = merged[(t >= merged["_tmin"]) & (t <= merged["_tmax"])]
    means = merged.groupby("_track_row")[list(features)].mean()
    result = tracks.set_index("_track_row")[["sample_name", "TrackID", cluster_key]].join(means)
    return result.reset_index(drop=True)


def _cluster_order_and_colors(track_adata, cluster_key, labels):
    order = _apply_state_order(
        sorted(pd.Series(labels).astype(str).unique().tolist(), key=_mixed_label_sort_key),
        _get_classification_state_order(track_adata, cluster_key),
    )
    colors = _normalize_label_color_map(order, colors=_get_classification_state_colors(track_adata, cluster_key))
    return order, colors


def _summarize_and_plot(
    per_unit,
    features,
    *,
    group_col,
    order,
    colors,
    scaling,
    outfolder,
    title,
    unit_label,
    csv_prefix="feature_heatmap",
    annotate=True,
):
    """Shared core of the cluster/state x feature heatmaps.

    ``per_unit`` has one row per averaging unit (a track, a track-state pair or a
    timepoint) with ``group_col`` and the feature columns. Group means are the mean
    of their units. zscore colours: features standardized across all units, then
    averaged per group. minmax colours: group means rescaled 0-1 per feature.
    """
    counts = per_unit[group_col].value_counts().reindex(order).fillna(0).astype(int)
    raw_means = per_unit.groupby(group_col)[features].mean().reindex(order)

    if scaling == "zscore":
        vals = per_unit[features]
        std = vals.std(ddof=0).replace(0, np.nan)
        z = (vals - vals.mean()) / std
        z[group_col] = per_unit[group_col]
        color_vals = z.groupby(group_col)[features].mean().reindex(order)
        arr = color_vals.to_numpy(dtype=float)
        lim = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
        lim = max(lim, 1e-9)
        cmap, vmin, vmax = "RdBu_r", -lim, lim
        cbar_label = f"mean z-score (0 = average {unit_label})"
    else:
        lo, hi = raw_means.min(), raw_means.max()
        color_vals = (raw_means - lo) / (hi - lo).replace(0, np.nan)
        cmap, vmin, vmax, cbar_label = "viridis", 0.0, 1.0, "group mean, min-max per feature"

    # rows = features, columns = groups
    mat = color_vals.T.to_numpy(dtype=float)
    raw_mat = raw_means.T.to_numpy(dtype=float)
    n_feat, n_grp = mat.shape

    outfolder = Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    csv_path = outfolder / f"{csv_prefix}_cluster_means.csv"
    table = raw_means.add_suffix("_mean").join(color_vals.add_suffix(f"_{scaling}"))
    table.insert(0, f"n_{unit_label}s", counts)
    table.index.name = group_col
    table.to_csv(csv_path)

    pdf_path = outfolder / f"{csv_prefix}.pdf"
    fig_w = min(max(4.5, 1.1 * n_grp + 3.5), 24)
    fig_h = min(max(3.5, 0.42 * n_feat + 2.2), 40)
    with PdfPages(pdf_path) as pdf:
        fig, (ax_bar, ax) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [0.35, max(1.0, 0.42 * n_feat)]}, sharex=True,
        )
        ax_bar.imshow([[to_rgb(colors[c]) for c in order]], aspect="auto")
        ax_bar.set_yticks([])
        ax_bar.set_title(f"{title}\ncolour: {cbar_label}", fontsize=9)
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xticks(range(n_grp))
        # Angled so long cluster/state names don't overlap.
        ax.set_xticklabels(
            [f"{c} (n={counts[c]})" for c in order],
            fontsize=8, rotation=45, ha="right", rotation_mode="anchor",
        )
        ax.set_xlabel(str(group_col))
        if annotate and n_feat * n_grp <= 600:
            for i in range(n_feat):
                for j in range(n_grp):
                    v = raw_mat[i, j]
                    if not np.isfinite(v):
                        continue
                    shade = mat[i, j]
                    dark = np.isfinite(shade) and (
                        abs(shade) > 0.6 * vmax if scaling == "zscore" else shade < 0.35
                    )
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7,
                            color="white" if dark else "black")
        cbar = fig.colorbar(im, ax=[ax_bar, ax], fraction=0.04, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return pdf_path, csv_path


def _validate_heatmap_args(features, scaling):
    features = [str(f) for f in dict.fromkeys(features)]
    if len(features) == 0:
        raise ValueError("Select at least one feature for the heatmap.")
    scaling = str(scaling).strip().lower()
    if scaling not in {"zscore", "minmax"}:
        raise ValueError("scaling must be 'zscore' or 'minmax'.")
    return features, scaling


def save_track_feature_heatmap(
    track_adata,
    output_dir,
    cell_type,
    features,
    *,
    cluster_key="ClusterID",
    scaling="zscore",
    annotate=True,
    outfolder=None,
    verbose=True,
):
    """Write the trajectory-cluster x feature heatmap (PDF) and its values (CSV).

    Each track's features are averaged over its trajectory window, then per cluster.
    See ``_summarize_and_plot`` for the colour scalings; cell labels always show the
    unscaled cluster mean.
    """
    features, scaling = _validate_heatmap_args(features, scaling)
    if cluster_key not in track_adata.obs.columns:
        raise KeyError(f"Cluster column '{cluster_key}' not found in the track model.")
    if outfolder is None:
        outfolder = (
            Path(output_dir).expanduser() / "analysis" / str(cell_type) / "behavorial_trajectories"
            / FEATURE_HEATMAP_SUBDIR
        )

    if verbose:
        _winfo("feature-heatmap", f"loading {len(features)} feature(s) for {track_adata.n_obs} tracks")
    df_tp = _load_timepoint_features(output_dir, cell_type, features, track_adata=track_adata)
    per_track = compute_track_feature_means(track_adata, df_tp, features, cluster_key=cluster_key)
    if int(per_track[features].notna().any(axis=1).sum()) == 0:
        raise ValueError("No timepoints matched the tracks of the clustering model (check sample_name / TrackID).")

    order, colors = _cluster_order_and_colors(track_adata, cluster_key, per_track[cluster_key])
    pdf_path, csv_path = _summarize_and_plot(
        per_track, features,
        group_col=cluster_key, order=order, colors=colors, scaling=scaling, outfolder=outfolder,
        title=(
            f"Trajectory clusters x features ({cell_type})\n"
            "per-track means over each trajectory, then mean per cluster"
        ),
        unit_label="track", annotate=annotate,
    )
    per_track_csv = Path(outfolder) / "feature_heatmap_track_means.csv"
    per_track.to_csv(per_track_csv, index=False)
    if verbose:
        _winfo("feature-heatmap", f"saved {pdf_path}")
    return {"feature_heatmap_pdf": str(pdf_path), "cluster_means_csv": str(csv_path),
            "track_means_csv": str(per_track_csv)}


# ── Behavioral states ────────────────────────────────────────────────────────

STATE_FEATURE_HEATMAP_SUBDIR = "state_feature_heatmaps"

# State-column choices -> obs columns that may hold them (first present wins).
STATE_COLUMN_ALIASES = {
    "behavioral_state": ("behavioral_state", "full_behavioral_cluster"),
    "full_behavioral_cluster": ("full_behavioral_cluster", "behavioral_state"),
    "intrinsic_behavioral_cluster": ("intrinsic_behavioral_cluster", "hmm_intrinsic_behavioral_state"),
    "hmm_intrinsic_behavioral_state": ("hmm_intrinsic_behavioral_state", "intrinsic_behavioral_cluster"),
}


def resolve_state_column(adata_states, state_col):
    for col in STATE_COLUMN_ALIASES.get(str(state_col), (str(state_col),)):
        if col in adata_states.obs.columns:
            return col
    raise KeyError(f"State column '{state_col}' not found in the behavioral-state adata.")


def compute_state_feature_units(adata_states, df_timepoints, features, *, state_col, weighting="timepoints"):
    """Averaging units for the state heatmap.

    weighting="timepoints": one row per timepoint (every timepoint in a state counts).
    weighting="tracks": one row per (track, state) pair - the track's mean over its
    timepoints in that state - so each track counts once per state and long tracks
    don't dominate.
    """
    obs = adata_states.obs
    missing = [c for c in list(ID_COLS) + [TIME_COL, state_col] if c not in obs.columns]
    if missing:
        raise KeyError(f"Behavioral-state adata is missing columns {missing}.")
    labels = obs[state_col].astype("string").str.strip()
    tp = pd.DataFrame({
        "sample_name": obs["sample_name"].astype(str).str.strip().to_numpy(),
        "TrackID": _track_id_key(obs["TrackID"]),
        TIME_COL: pd.to_numeric(obs[TIME_COL], errors="coerce").to_numpy(),
        state_col: labels.to_numpy(),
    })
    tp = tp[tp[state_col].notna() & (tp[state_col] != "")]
    tp = tp.drop_duplicates(subset=list(ID_COLS) + [TIME_COL])
    merged = tp.merge(df_timepoints, on=list(ID_COLS) + [TIME_COL], how="left")
    if str(weighting) == "tracks":
        return (
            merged.groupby(["sample_name", "TrackID", state_col], sort=False)[list(features)]
            .mean().reset_index()
        )
    if str(weighting) != "timepoints":
        raise ValueError("weighting must be 'timepoints' or 'tracks'.")
    return merged


def save_state_feature_heatmap(
    adata_states,
    output_dir,
    cell_type,
    features,
    *,
    state_col="behavioral_state",
    weighting="timepoints",
    scaling="zscore",
    annotate=True,
    outfolder=None,
    verbose=True,
):
    """Write the behavioral-state x feature heatmap (PDF) and its values (CSV).

    Feature values come from the track-features CSV (raw units); features that only
    exist in the state adata (e.g. window features) are taken from it.
    """
    features, scaling = _validate_heatmap_args(features, scaling)
    state_col = resolve_state_column(adata_states, state_col)
    if outfolder is None:
        outfolder = (
            Path(output_dir).expanduser() / "analysis" / str(cell_type) / "behavioral_states"
            / STATE_FEATURE_HEATMAP_SUBDIR
        )

    if verbose:
        _winfo("feature-heatmap", f"loading {len(features)} feature(s) for {adata_states.n_obs} timepoints")
    df_tp = _load_timepoint_features(output_dir, cell_type, features, fallback_adata=adata_states)
    units = compute_state_feature_units(adata_states, df_tp, features, state_col=state_col, weighting=weighting)
    if len(units) == 0 or int(units[features].notna().any(axis=1).sum()) == 0:
        raise ValueError(
            "No feature values matched the behavioral-state timepoints (check sample_name / TrackID / position_t)."
        )

    order, colors = _cluster_order_and_colors(adata_states, state_col, units[state_col])
    unit_label = "track" if weighting == "tracks" else "timepoint"
    how = (
        "per-track means within each state, then mean per state"
        if weighting == "tracks" else "mean over all timepoints in each state"
    )
    pdf_path, csv_path = _summarize_and_plot(
        units, features,
        group_col=state_col, order=order, colors=colors, scaling=scaling, outfolder=outfolder,
        title=f"Behavioral states x features ({cell_type})\n{how}",
        unit_label=unit_label, csv_prefix=f"state_feature_heatmap_{state_col}", annotate=annotate,
    )
    if verbose:
        _winfo("feature-heatmap", f"saved {pdf_path}")
    return {"feature_heatmap_pdf": str(pdf_path), "cluster_means_csv": str(csv_path)}



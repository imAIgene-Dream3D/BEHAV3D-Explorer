"""Per-timepoint feature matrix for trajectory clustering without behavioral states.

``build_trajectory_feature_adata`` reads the track-features CSV and applies the
same preprocessing the HMM state classification offers (optional window
features, rolling smoothing, quantile capping, log1p scaling, z-scaling), but
fits no state model. The result is saved as an h5ad shaped like the
behavioral-state adata (``X`` = continuous features, ``obs`` = ids, binary
columns, coordinates, metadata), so the feature-based DTW trajectory clustering
and its downstream plots / backprojection can read it in place of the state file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from behav3d.core.h5_access import write_adata
from behav3d.core.anndata import df_to_adata
from behav3d.analysis.behavior.state.classification import (
    HMM_OPTIONAL_WINDOW_FEATURES,
    _compute_optional_window_feature_frame,
    _dedupe_preserve_order,
    _merge_optional_window_features,
    _normalize_selection,
    _smooth_timepoint_features,
)
from behav3d.analysis.behavior.state.utils import (
    _apply_log1p_to_feature_matrix,
    _normalize_log_scale_feature_selectors,
    _require_columns,
    _resolve_log_scale_feature_cols,
    _resolve_positions_csv_path,
    cap_values_to_quantile,
)
from behav3d.analysis.behavior.track.utils import _winfo

ID_COLS = ("sample_name", "TrackID")
TIME_COL = "position_t"
TRAJECTORY_FEATURES_SUBDIR = "trajectory_features"
WINDOW_FEATURES = tuple(HMM_OPTIONAL_WINDOW_FEATURES)

_COORD_COLS = (
    "position_x", "position_y", "position_z",
    "pixel_position_x", "pixel_position_y", "pixel_position_z",
)
_META_COLS = ("origin_cell_type", "well", "exp_nr")


def _drop_none(value):
    """h5ad cannot store None; drop such entries (recursively)."""
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    return value


def trajectory_feature_adata_path(output_dir, cell_type):
    return (
        Path(output_dir).expanduser() / "analysis" / str(cell_type) / TRAJECTORY_FEATURES_SUBDIR
        / f"BEHAV3D_{cell_type}_trajectory_features.h5ad"
    )


def build_trajectory_feature_adata(
    output_dir,
    cell_type,
    features,
    *,
    binary_features=None,
    additional_window_features=None,
    window_features_window=5,
    feature_smoothing_window=1,
    lower_quantile_cap=None,
    upper_quantile_cap=None,
    log_scale_features=None,
    start_offset=0,
    df_positions=None,
    save=True,
    verbose=True,
):
    """Build (and by default save) the per-timepoint trajectory feature adata.

    features : continuous per-timepoint columns of the track-features CSV.
    binary_features : 0/1 columns (e.g. contacts); kept unscaled in ``obs`` and
        standardized later by the DTW step (see ``resolve_state_feature_matrix``).
    additional_window_features : any of net_displacement / straightness /
        mean_square_displacement, computed over ``window_features_window`` frames.
    start_offset : drop each track's first N timepoints (e.g. the first frame,
        whose motion features are a fabricated 0).
    """
    raw_features = _dedupe_preserve_order(_normalize_selection(features))
    window_features = _dedupe_preserve_order(additional_window_features)
    unknown = [f for f in window_features if f not in WINDOW_FEATURES]
    if unknown:
        raise ValueError(f"Unsupported window features {unknown}; supported: {list(WINDOW_FEATURES)}")
    kept_features = _dedupe_preserve_order(list(raw_features) + list(window_features))
    binary_cols = [c for c in _dedupe_preserve_order(_normalize_selection(binary_features)) if c not in kept_features]
    if len(kept_features) == 0 and len(binary_cols) == 0:
        raise ValueError("Select at least one feature for trajectory clustering.")

    sort_cols = list(ID_COLS) + [TIME_COL]
    csv_path = None
    if df_positions is None:
        csv_path = _resolve_positions_csv_path(output_dir=output_dir, cell_type=cell_type)
        if verbose:
            _winfo("trajectory-features", f"loading {csv_path}")
        df_positions = pd.read_csv(csv_path, low_memory=False)
    required = sort_cols + raw_features + binary_cols
    if window_features:
        required += ["position_x", "position_y", "position_z"]
    _require_columns(df_positions, _dedupe_preserve_order(required), "trajectory feature preparation")

    df = df_positions.sort_values(sort_cols, kind="mergesort").copy()
    df = df.drop_duplicates(subset=sort_cols, keep="first")
    if int(start_offset) > 0:
        rank = df.groupby(list(ID_COLS), sort=False, observed=False).cumcount()
        df = df.loc[rank >= int(start_offset)].copy()

    window_size_used = None
    if window_features:
        df_window, window_size_used = _compute_optional_window_feature_frame(
            df,
            window_features=window_features,
            time_col=TIME_COL,
            id_cols=ID_COLS,
            window_size=window_features_window,
            verbose=verbose,
        )
        df = _merge_optional_window_features(df, df_window, join_cols=sort_cols)

    df = _smooth_timepoint_features(
        df, raw_features, id_cols=ID_COLS, time_col=TIME_COL,
        window=int(feature_smoothing_window), min_periods=1,
    )
    cap_limits = {}
    if kept_features and (lower_quantile_cap is not None or upper_quantile_cap is not None):
        df, cap_limits = cap_values_to_quantile(
            df, kept_features,
            lower_quantile=lower_quantile_cap, upper_quantile=upper_quantile_cap,
            return_limits=True,
        )

    numeric = df[kept_features].apply(pd.to_numeric, errors="coerce") if kept_features else pd.DataFrame(index=df.index)
    valid = numeric.notna().all(axis=1) if kept_features else pd.Series(True, index=df.index)
    n_dropped = int((~valid).sum())
    if n_dropped and verbose:
        _winfo("trajectory-features", f"dropping {n_dropped} timepoints with missing feature values")
    df = df.loc[valid].copy()
    if len(df) == 0:
        raise ValueError("No timepoints left after removing rows with missing feature values.")

    log_cols = []
    X = numeric.loc[valid, kept_features].to_numpy(dtype=float, copy=True) if kept_features else np.empty((len(df), 0))
    if kept_features:
        log_params = _resolve_log_scale_feature_cols(
            kept_features, _normalize_log_scale_feature_selectors(log_scale_features)
        )
        if log_params["unresolved_features"]:
            raise ValueError(f"Log-scale features not among the selected features: {log_params['unresolved_features']}")
        log_cols = list(log_params["resolved_feature_cols"])
        if log_cols:
            X = _apply_log1p_to_feature_matrix(
                X, feature_cols=kept_features, resolved_feature_cols=log_cols, inplace=True
            )
        X = StandardScaler().fit_transform(X)
        df.loc[:, kept_features] = X

    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    meta_cols = [c for c in _META_COLS if c in df.columns] + [c for c in df.columns if str(c).endswith("_line_condition")]
    coord_cols = [c for c in _COORD_COLS if c in df.columns and c not in kept_features]
    obs_cols = _dedupe_preserve_order(sort_cols + binary_cols + coord_cols + meta_cols)
    adata = df_to_adata(df, feature_cols=kept_features, obs_cols=obs_cols)
    adata.X = np.asarray(adata.X, dtype=float)

    adata.uns["preprocessing"] = _drop_none({
        "source": "trajectory_features",
        "raw_feature_cols": list(raw_features),
        "continuous_feature_cols": list(kept_features),
        "additional_window_features": list(window_features),
        "window_features_window": None if window_size_used is None else int(window_size_used),
        "binary_cols_to_merge": list(binary_cols),
        "feature_smoothing_window": int(feature_smoothing_window),
        "lower_quantile_cap": None if lower_quantile_cap is None else float(lower_quantile_cap),
        "upper_quantile_cap": None if upper_quantile_cap is None else float(upper_quantile_cap),
        "log_scaled_feature_cols": list(log_cols),
        "start_offset": int(start_offset),
        "n_timepoints_dropped_nan": n_dropped,
        "positions_csv_path": None if csv_path is None else str(csv_path),
        "quantile_feature_limits": {str(k): dict(v) for k, v in dict(cap_limits).items()},
    })

    if save:
        path = trajectory_feature_adata_path(output_dir, cell_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_adata(adata, path, compression="gzip")
        if verbose:
            _winfo(
                "trajectory-features",
                f"saved {path} | timepoints={adata.n_obs} | continuous={kept_features} | binary={binary_cols}",
            )
    return adata

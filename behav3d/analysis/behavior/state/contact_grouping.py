"""State-native contact-column comparison.

Unlike the track-side `behavior.track.contact_grouping` (which restricts to each track's
*classified DTW window* and applies a min-contiguous-bout threshold), every row of a
behavioral-states adata already carries a per-timepoint state label — so comparing contact
across several organoid-contact columns (e.g. `healthy_organoid_contact` vs.
`tumor_organoid_contact`) needs no window/bout-length concept here: each selected contact
column is merged directly from the per-timepoint tracks CSV onto the states adata's `.obs`.
"""
import pandas as pd

from behav3d.analysis.behavior.track.contact_grouping import _normalize_id_column, _missing_keys


def compute_state_contact_type_values(
    df_timepoints,
    adata_states,
    *,
    contact_cols,
    state_col,
    time_col="position_t",
    groupby_cols=("sample_name", "TrackID"),
    verbose=False,
):
    """Long-form per-(track, state, contact column) mean contact fraction, for a purely
    descriptive per-state mean +/- SEM comparison across several organoid-contact columns —
    no significance test is computed here or downstream.

    Each track contributes one row per state it visited per selected contact column (its own
    mean fraction of that state's timepoints with contact=True), so the per-state aggregate
    error bars reflect between-track variation rather than per-frame pseudoreplication.

    Returns a long dataframe with columns `[*groupby_cols, state_col, "contact_col",
    "mean_fraction", "n_timepoints"]`.
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    time_col = str(time_col)
    state_col = str(state_col)
    contact_cols = [str(c) for c in list(contact_cols)]
    if not contact_cols:
        raise ValueError("contact_cols must contain at least one contact column.")

    if state_col not in adata_states.obs.columns:
        raise KeyError(f"state_col={state_col!r} not found in adata_states.obs.")
    missing_obs = [c for c in groupby_cols + [time_col] if c not in adata_states.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required columns in adata_states.obs: {missing_obs}")
    missing_tp = [c for c in groupby_cols + [time_col] + contact_cols if c not in df_timepoints.columns]
    if missing_tp:
        raise KeyError(f"Missing required columns in df_timepoints: {missing_tp}")

    states = adata_states.obs[groupby_cols + [time_col, state_col]].copy()
    for col in groupby_cols:
        states[col] = _normalize_id_column(states[col])
    states[time_col] = pd.to_numeric(states[time_col], errors="coerce").round()

    contact = df_timepoints[groupby_cols + [time_col] + contact_cols].copy()
    for col in groupby_cols:
        contact[col] = _normalize_id_column(contact[col])
    contact[time_col] = pd.to_numeric(contact[time_col], errors="coerce").round()
    for contact_col in contact_cols:
        contact[contact_col] = pd.to_numeric(contact[contact_col], errors="coerce").fillna(0).astype(bool)

    merged = states.merge(contact, on=groupby_cols + [time_col], how="inner")

    missing = _missing_keys(states[groupby_cols].drop_duplicates(), merged[groupby_cols].drop_duplicates(), groupby_cols)
    if missing:
        raise ValueError(
            f"{len(missing)} track(s) present in adata_states.obs have no matching timepoints at "
            f"all in df_timepoints (matched on {groupby_cols}) — the per-timepoint contact CSV no "
            f"longer matches the behavioral-states h5ad. Re-run State Classification to regenerate "
            f"it from the current filtered CSV before running this analysis. Missing example "
            f"track key(s) (first 10): {sorted(missing)[:10]}"
        )

    long_rows = []
    for contact_col in contact_cols:
        stats = (
            merged.groupby(groupby_cols + [state_col], sort=False, observed=True)[contact_col]
            .agg(mean_fraction="mean", n_timepoints="size")
            .reset_index()
        )
        stats["contact_col"] = contact_col
        long_rows.append(stats)

    long_df = pd.concat(long_rows, ignore_index=True)
    if verbose:
        print(
            f"State contact-type values: {len(long_df)} (track, state, contact column) row(s) "
            f"across {len(contact_cols)} column(s)."
        )
    return long_df[groupby_cols + [state_col, "contact_col", "mean_fraction", "n_timepoints"]]

"""State-native contact-column comparison.

Unlike the track-side `behavior.track.contact_grouping` (which restricts to each track's
*classified DTW window* and applies a min-contiguous-bout threshold), every row of a
behavioral-states adata already carries a per-timepoint state label — so comparing behavior
across several organoid-contact columns (e.g. `healthy_organoid_contact` vs.
`tumor_organoid_contact`) needs no window/bout-length concept here: each selected contact
column is merged directly from the per-timepoint tracks CSV onto the states adata's `.obs`.
"""
import pandas as pd

from behav3d.analysis.behavior.track.contact_grouping import _normalize_id_column, _missing_keys
from behav3d.analysis.behavior.utils import _mixed_label_sort_key


def _merge_state_timepoints_with_contact(
    df_timepoints,
    adata_states,
    *,
    contact_cols,
    state_col,
    time_col,
    groupby_cols,
):
    """Validate + merge per-timepoint state labels (`adata_states.obs`) with the selected
    boolean `contact_cols` from `df_timepoints`, on `groupby_cols + time_col` (after
    normalizing id columns via `_normalize_id_column`). Raises if any track present in
    `adata_states.obs` has zero matching timepoints in `df_timepoints` — a sign the
    per-timepoint contact CSV no longer matches the behavioral-states h5ad.

    Returns the merged DataFrame with columns `[*groupby_cols, time_col, state_col,
    *contact_cols]` (contact_cols coerced to bool).
    """
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
    return merged


def compute_state_contact_type_cluster_proportions(
    df_timepoints,
    adata_states,
    *,
    contact_cols,
    state_col,
    time_col="position_t",
    groupby_cols=("sample_name", "TrackID"),
    verbose=False,
):
    """Per-(track, contact_col) state-composition proportions, restricted to each track's
    timepoints where that contact_col is True — i.e. "for T cells with organoid contact,
    what's the proportion of behavioral states while touching organoid type X", feeding a
    Welch's-t-test cluster-size-difference + stacked-composition comparison across 2+
    contact_cols (generalizes to N organoid/contact types, not just 2).

    A track touching multiple selected contact_cols contributes one independent unit row per
    contact_col it's positive for — not mutually exclusive. A timepoint where two contact_cols
    are simultaneously True counts toward both groups' state distributions.

    Returns
    -------
    per_unit_df : pd.DataFrame
        index = unit key "<groupby_cols joined by '|'>||<contact_col>", columns=state_order,
        values=proportion (each row sums to 1). Empty if no contact_col has any positive
        timepoint anywhere.
    unit_metadata : pd.DataFrame
        Same index, single column "contact_col".
    state_order : list[str]
        States observed anywhere in the contact-positive subset, mixed-label sorted (numeric
        labels sort numerically, others lexicographically).
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    contact_cols = [str(c) for c in list(contact_cols)]
    state_col = str(state_col)
    if not contact_cols:
        raise ValueError("contact_cols must contain at least one contact column.")

    merged = _merge_state_timepoints_with_contact(
        df_timepoints, adata_states, contact_cols=contact_cols, state_col=state_col,
        time_col=time_col, groupby_cols=groupby_cols,
    )
    merged[state_col] = merged[state_col].astype(str)

    frames = []
    for contact_col in contact_cols:
        sub = merged.loc[merged[contact_col]].copy()
        if len(sub) == 0:
            if verbose:
                print(f"  Note: '{contact_col}' has no positive timepoints — excluded from the comparison.")
            continue
        sub["_contact_col"] = contact_col
        frames.append(sub)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(columns=["contact_col"]), []

    stacked = pd.concat(frames, ignore_index=True)
    stacked["_unit_key"] = (
        stacked[groupby_cols].astype(str).agg("|".join, axis=1) + "||" + stacked["_contact_col"]
    )
    state_order = sorted(stacked[state_col].dropna().unique().tolist(), key=_mixed_label_sort_key)
    unit_order = stacked["_unit_key"].drop_duplicates().tolist()

    per_unit_df = pd.crosstab(stacked["_unit_key"], stacked[state_col], normalize="index").reindex(
        index=unit_order, columns=state_order, fill_value=0.0,
    )
    unit_metadata = (
        stacked.drop_duplicates(subset=["_unit_key"])[["_unit_key", "_contact_col"]]
        .set_index("_unit_key")
        .rename(columns={"_contact_col": "contact_col"})
    )

    if verbose:
        print(
            f"State contact-type cluster proportions: {len(per_unit_df)} (track, contact "
            f"column) unit(s) across {len(contact_cols)} column(s), {len(state_order)} state(s)."
        )
    return per_unit_df, unit_metadata, state_order

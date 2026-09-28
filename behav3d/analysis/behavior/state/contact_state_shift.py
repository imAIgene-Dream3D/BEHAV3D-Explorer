import numpy as np
import pandas as pd

from behav3d.features.state_descriptive_features import rle_encode
from behav3d.analysis.behavior.track.contact_grouping import (
    _normalize_id_column,
    _assert_tracks_have_window_data,
    _missing_keys,
)

BEFORE_LABEL = "before"
AFTER_LABEL = "after"


def _first_qualifying_bout_bounds(times, is_contact, min_bout_length):
    """First contiguous run of truthy ``is_contact`` (co-sorted with ``times`` by time) whose
    length is >= ``min_bout_length``. Returns ``(start_t, end_t)`` (inclusive) of that run, or
    ``(None, None)`` if no run qualifies. Extends ``contact_grouping._max_true_run_length`` (which
    only returns the longest run's *length*) by preserving the *position* of the first qualifying
    run instead.
    """
    runs = rle_encode(list(is_contact))
    pos = 0
    for value, length in runs:
        if bool(value) and length >= min_bout_length:
            return times[pos], times[pos + length - 1]
        pos += length
    return None, None


def _sample_null_reference_relative_positions(n, *, reference_relative_positions, seed):
    """Draw ``n`` relative split-points in [0, 1] with replacement from the empirical
    distribution of real contact-bout-start relative positions (``reference_relative_positions``),
    for use as a timing-matched null/control split point on no-contact tracks. Falls back to
    Uniform(0, 1) when there are no real bouts to draw from (e.g. zero contact tracks in the run).
    """
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference_relative_positions, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return rng.uniform(0.0, 1.0, size=int(n))
    return rng.choice(ref, size=int(n), replace=True)


def compute_contact_bout_windows(
    df_timepoints,
    adata_states,
    *,
    contact_col,
    min_bout_length,
    window_mode="fixed",
    fixed_window_length=10,
    min_window_timepoints=3,
    time_col="position_t",
    groupby_cols=("sample_name", "TrackID"),
    null_seed=0,
    verbose=False,
):
    """Per state-classified track, locate the before/after windows flanking its first
    sufficiently long contact bout (or, for tracks with no qualifying bout, a timing-matched
    synthetic reference split point), for use in a before-vs-after behavioral-state-shift
    comparison.

    Unlike a track-DTW classification (which restricts to a *classified window* that may be a
    trimmed subset of the track), this needs no track-DTW classification at all: each track's
    window bounds are simply its own full timepoint span in ``df_timepoints`` (min/max of
    ``time_col``), and ``adata_states.obs`` is only used to know which ``(sample_name, TrackID)``
    pairs are in scope (i.e. state-classified).

    ``min_bout_length`` is the same threshold ``contact_grouping.compute_track_contact_features``
    uses for the contact/no_contact grouping — it is passed through here (not re-derived) so both
    analyses agree on what counts as "a real contact bout" for a given run.

    For each state-classified track:

    - If it has a bout of ``contact_col`` truthy values >= ``min_bout_length`` timepoints long, the
      *first* such bout's ``(start_t, end_t)`` is used (via ``_first_qualifying_bout_bounds``) and
      the track is labeled ``"contact"``.
    - Otherwise the track is labeled ``"no_contact"`` and given a synthetic zero-length reference
      point ``bout_start_t == bout_end_t == t_ref`` (flagged ``is_null_reference=True``), where
      ``t_ref``'s relative position within the track's span is drawn from the empirical
      distribution of real bout-start relative positions observed elsewhere in this run (seeded by
      ``null_seed`` for reproducibility) — a timing-matched null rather than e.g. always the track
      midpoint, so any track-wide temporal trend unrelated to contact doesn't bias the comparison.

    Given ``window_mode`` ("fixed" or "full"):

    - "fixed": ``before = [max(position_t_min, bout_start_t - fixed_window_length), bout_start_t)``,
      ``after = (bout_end_t, min(position_t_max, bout_end_t + fixed_window_length)]``.
    - "full": ``before = [position_t_min, bout_start_t)``, ``after = (bout_end_t, position_t_max]``.

    Tracks where either window ends up with fewer than ``min_window_timepoints`` timepoints are
    marked ``excluded=True`` with an ``excluded_reason`` in {"before_too_short", "after_too_short",
    "both_too_short"}.

    Returns a DataFrame indexed by ``groupby_cols`` with columns: ``contact_group``,
    ``bout_start_t``, ``bout_end_t``, ``is_null_reference``, ``before_start_t``, ``before_end_t``,
    ``after_start_t``, ``after_end_t``, ``before_n_timepoints``, ``after_n_timepoints``,
    ``excluded``, ``excluded_reason``.
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    time_col = str(time_col)
    contact_col = str(contact_col)
    min_bout_length = int(min_bout_length)
    window_mode = str(window_mode).strip().lower()
    if window_mode not in ("fixed", "full"):
        raise ValueError(f"window_mode must be 'fixed' or 'full', got {window_mode!r}.")
    fixed_window_length = int(fixed_window_length)
    min_window_timepoints = int(min_window_timepoints)
    key_cols = groupby_cols

    missing = [c for c in groupby_cols + [time_col, contact_col] if c not in df_timepoints.columns]
    if missing:
        raise KeyError(f"Missing required columns in df_timepoints: {missing}")
    missing_obs = [c for c in groupby_cols if c not in adata_states.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required columns in adata_states.obs: {missing_obs}")

    classified_tracks = adata_states.obs[groupby_cols].drop_duplicates(subset=key_cols).copy()
    for col in groupby_cols:
        classified_tracks[col] = _normalize_id_column(classified_tracks[col])

    df = df_timepoints[groupby_cols + [time_col, contact_col]].copy()
    for col in groupby_cols:
        df[col] = _normalize_id_column(df[col])
    df[contact_col] = pd.to_numeric(df[contact_col], errors="coerce").fillna(0).astype(bool)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    span = (
        df.groupby(key_cols, sort=False, observed=True)[time_col]
        .agg(position_t_min="min", position_t_max="max")
        .reset_index()
    )
    windows = classified_tracks.merge(span, on=key_cols, how="left")

    # position_t_min/max were derived from df itself, so every row of df for a classified track
    # always satisfies [position_t_min, position_t_max] trivially -- no time-based filtering is
    # needed here (unlike the track-DTW-classified-window case this replaces).
    matched = windows.merge(df, on=key_cols, how="inner")

    _assert_tracks_have_window_data(
        windows, matched, key_cols=key_cols,
        h5ad_label="adata_states.obs (behavioral states h5ad)",
    )

    rows = []
    for key_vals, group in matched.groupby(key_cols, sort=False, observed=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        group = group.sort_values(time_col, kind="stable")
        times = group[time_col].to_numpy()
        is_contact = group[contact_col].to_numpy()
        t_min = float(group["position_t_min"].iloc[0])
        t_max = float(group["position_t_max"].iloc[0])

        bout_start_t, bout_end_t = _first_qualifying_bout_bounds(times, is_contact, min_bout_length)
        row = dict(zip(key_cols, key_vals))
        row["_times"] = times
        row["_t_min"] = t_min
        row["_t_max"] = t_max
        if bout_start_t is not None:
            row["contact_group"] = "contact"
            row["bout_start_t"] = float(bout_start_t)
            row["bout_end_t"] = float(bout_end_t)
            row["is_null_reference"] = False
            row["_relative_position"] = (
                (float(bout_start_t) - t_min) / (t_max - t_min) if t_max > t_min else 0.0
            )
        else:
            row["contact_group"] = "no_contact"
            row["bout_start_t"] = None
            row["bout_end_t"] = None
            row["is_null_reference"] = True
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        cols = key_cols + [
            "contact_group", "bout_start_t", "bout_end_t", "is_null_reference",
            "before_start_t", "before_end_t", "after_start_t", "after_end_t",
            "before_n_timepoints", "after_n_timepoints", "excluded", "excluded_reason",
        ]
        return pd.DataFrame(columns=cols).set_index(key_cols)

    real_relative_positions = out.loc[out["contact_group"] == "contact", "_relative_position"].to_numpy(dtype=float)
    null_mask = out["contact_group"] == "no_contact"
    n_null = int(null_mask.sum())
    if n_null > 0:
        sampled_rel = _sample_null_reference_relative_positions(
            n_null, reference_relative_positions=real_relative_positions, seed=null_seed,
        )
        null_rows = out.loc[null_mask]
        t_ref = null_rows["_t_min"].to_numpy(dtype=float) + sampled_rel * (
            null_rows["_t_max"].to_numpy(dtype=float) - null_rows["_t_min"].to_numpy(dtype=float)
        )
        t_ref = np.clip(np.round(t_ref), null_rows["_t_min"].to_numpy(dtype=float), null_rows["_t_max"].to_numpy(dtype=float))
        out.loc[null_mask, "bout_start_t"] = t_ref
        out.loc[null_mask, "bout_end_t"] = t_ref

    before_starts, before_ends, after_starts, after_ends = [], [], [], []
    before_counts, after_counts = [], []
    excluded, excluded_reason = [], []
    for _, r in out.iterrows():
        t_min, t_max = r["_t_min"], r["_t_max"]
        bout_start_t, bout_end_t = float(r["bout_start_t"]), float(r["bout_end_t"])
        if window_mode == "fixed":
            b_start = max(t_min, bout_start_t - fixed_window_length)
            b_end = bout_start_t
            a_start = bout_end_t
            a_end = min(t_max, bout_end_t + fixed_window_length)
        else:
            b_start = t_min
            b_end = bout_start_t
            a_start = bout_end_t
            a_end = t_max

        times = r["_times"]
        n_before = int(np.sum((times >= b_start) & (times < b_end)))
        n_after = int(np.sum((times > a_start) & (times <= a_end)))

        before_starts.append(b_start)
        before_ends.append(b_end)
        after_starts.append(a_start)
        after_ends.append(a_end)
        before_counts.append(n_before)
        after_counts.append(n_after)

        before_short = n_before < min_window_timepoints
        after_short = n_after < min_window_timepoints
        if before_short and after_short:
            excluded.append(True)
            excluded_reason.append("both_too_short")
        elif before_short:
            excluded.append(True)
            excluded_reason.append("before_too_short")
        elif after_short:
            excluded.append(True)
            excluded_reason.append("after_too_short")
        else:
            excluded.append(False)
            excluded_reason.append(None)

    out["before_start_t"] = before_starts
    out["before_end_t"] = before_ends
    out["after_start_t"] = after_starts
    out["after_end_t"] = after_ends
    out["before_n_timepoints"] = before_counts
    out["after_n_timepoints"] = after_counts
    out["excluded"] = excluded
    out["excluded_reason"] = excluded_reason

    out = out.drop(columns=["_times", "_t_min", "_t_max", "_relative_position"], errors="ignore")

    if verbose:
        n_contact = int((out["contact_group"] == "contact").sum())
        n_no_contact = int((out["contact_group"] == "no_contact").sum())
        n_excluded = int(out["excluded"].sum())
        reason_counts = out.loc[out["excluded"], "excluded_reason"].value_counts().to_dict()
        print(
            f"Contact bout windows for '{contact_col}' (min_bout_length={min_bout_length}, "
            f"window_mode={window_mode}): {n_contact} contact / {n_no_contact} no_contact tracks; "
            f"{n_excluded} excluded ({reason_counts})."
        )

    return out.set_index(key_cols)


def compute_state_shift_features(
    df_timepoints,
    adata_states,
    *,
    contact_col,
    min_bout_length,
    state_col,
    window_mode="fixed",
    fixed_window_length=10,
    min_window_timepoints=3,
    time_col="position_t",
    groupby_cols=("sample_name", "TrackID"),
    null_seed=0,
    verbose=False,
):
    """Join the before/after contact-bout windows (``compute_contact_bout_windows``) with the
    per-timepoint behavioral-state labels in ``adata_states.obs`` (a per-``(sample_name, TrackID,
    position_t)`` AnnData, e.g. the ``BEHAV3D_{cell_type}_behavioral_states.h5ad`` output of state
    classification), producing a long per-timepoint table tagged with each track's `contact_group`
    and `before`/`after` `period`.

    Returns a dict:
    - "track_windows": the DataFrame from ``compute_contact_bout_windows`` (all tracks, including
      excluded ones, for transparency/CSV export).
    - "state_timepoints": long DataFrame, one row per (track, period, timepoint), with columns
      ``[*groupby_cols, "period", time_col, state_col, "contact_group", "relative_t"]``, where
      ``relative_t`` is signed time-from-the-bout (<=0 before the bout start, >=0 after the bout
      end), for plotting a continuous composition-over-time view aligned across tracks.
    - "n_contact_tracks", "n_no_contact_tracks", "n_excluded_tracks": ints for logging/tests.
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    time_col = str(time_col)
    state_col = str(state_col)

    if state_col not in adata_states.obs.columns:
        raise KeyError(f"Missing state column '{state_col}' in adata_states.obs.")
    missing = [c for c in groupby_cols + [time_col] if c not in adata_states.obs.columns]
    if missing:
        raise KeyError(f"Missing required columns in adata_states.obs: {missing}")

    track_windows = compute_contact_bout_windows(
        df_timepoints,
        adata_states,
        contact_col=contact_col,
        min_bout_length=min_bout_length,
        window_mode=window_mode,
        fixed_window_length=fixed_window_length,
        min_window_timepoints=min_window_timepoints,
        time_col=time_col,
        groupby_cols=groupby_cols,
        null_seed=null_seed,
        verbose=verbose,
    )

    states = adata_states.obs[groupby_cols + [time_col, state_col]].copy()
    for col in groupby_cols:
        states[col] = _normalize_id_column(states[col])
    states[time_col] = pd.to_numeric(states[time_col], errors="coerce").round()

    usable = track_windows[~track_windows["excluded"]].reset_index()
    join_cols = groupby_cols

    merged = states.merge(usable, on=join_cols, how="inner", suffixes=("", "_window"))

    before_mask = (merged[time_col] >= merged["before_start_t"]) & (merged[time_col] < merged["before_end_t"])
    after_mask = (merged[time_col] > merged["after_start_t"]) & (merged[time_col] <= merged["after_end_t"])

    before_rows = merged.loc[before_mask, join_cols + [time_col, state_col, "contact_group"]].copy()
    before_rows["period"] = BEFORE_LABEL
    before_rows["relative_t"] = merged.loc[before_mask, time_col] - merged.loc[before_mask, "bout_start_t"]
    after_rows = merged.loc[after_mask, join_cols + [time_col, state_col, "contact_group"]].copy()
    after_rows["period"] = AFTER_LABEL
    after_rows["relative_t"] = merged.loc[after_mask, time_col] - merged.loc[after_mask, "bout_end_t"]

    state_timepoints = pd.concat([before_rows, after_rows], ignore_index=True)

    n_contact_tracks = int((usable["contact_group"] == "contact").sum())
    n_no_contact_tracks = int((usable["contact_group"] == "no_contact").sum())
    n_excluded_tracks = int(track_windows["excluded"].sum())

    if verbose:
        print(
            f"State-shift features for '{contact_col}' / state_col='{state_col}': "
            f"{n_contact_tracks} contact / {n_no_contact_tracks} no_contact tracks usable "
            f"({n_excluded_tracks} excluded); {len(state_timepoints)} before/after timepoint rows."
        )

    return {
        "track_windows": track_windows,
        "state_timepoints": state_timepoints,
        "n_contact_tracks": n_contact_tracks,
        "n_no_contact_tracks": n_no_contact_tracks,
        "n_excluded_tracks": n_excluded_tracks,
    }


def summarize_state_shift_track_fractions(
    state_timepoints_df,
    *,
    groupby_cols,
    state_col,
    state_order,
    period_col="period",
):
    """Per-track, per-period fraction of timepoints spent in each state — the per-track summary
    the before/after Welch's-t-test diff-bar panel needs (each track is exactly one independent
    observation), as opposed to the per-timepoint pooling the stacked-composition panel uses.

    Returns a DataFrame indexed by ``(*groupby_cols, period_col)`` with one column per
    ``state_order`` entry (values in [0, 1], missing states filled with 0.0).
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    state_order = [str(s) for s in list(state_order)]
    key_cols = groupby_cols + [period_col]

    if len(state_timepoints_df) == 0:
        return pd.DataFrame(columns=state_order)

    counts = (
        state_timepoints_df
        .groupby(key_cols, sort=False, observed=True)[state_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    counts = counts.reindex(columns=state_order, fill_value=0.0)
    return counts

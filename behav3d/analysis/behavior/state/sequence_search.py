import numpy as np
import pandas as pd

from behav3d.analysis.behavior.state.visualization.backprojection import (
    _validate_required_obs_columns,
)


_SEQUENCE_MATCH_RESULT_COLUMNS = (
    "sample_name",
    "TrackID",
    "start_time",
    "end_time",
    "n_states_matched",
)


def _collapse_into_bouts(states, times):
    """Collapse consecutive identical states into ``(state, start_t, end_t)`` bouts."""
    bouts = []
    for state, t in zip(states, times):
        if bouts and bouts[-1][0] == state:
            prev_state, start_t, _ = bouts[-1]
            bouts[-1] = (prev_state, start_t, t)
        else:
            bouts.append((state, t, t))
    return bouts


def _match_contiguous_bouts(bouts, target, require_track_start, require_track_end):
    """Sliding-window search for `target` as directly consecutive bouts.
    Anchors restrict which window start positions are even considered."""
    n = len(target)
    positions = range(len(bouts) - n + 1)
    if require_track_start:
        positions = [p for p in positions if p == 0]
    if require_track_end:
        positions = [p for p in positions if p == len(bouts) - n]
    for i in positions:
        if [bouts[i + k][0] for k in range(n)] == target:
            return (bouts[i][1], bouts[i + n - 1][2])
    return None


def _match_subsequence_bouts(bouts, target, require_track_start, require_track_end):
    """Greedy earliest-occurrence subsequence search: matches target[:-1] as
    early as possible, then resolves the final target state depending on
    ``require_track_end``. Greedy-earliest selection for a prefix never
    forecloses a later valid completion (a standard subsequence-matching
    property), so this stays correct while anchoring either end."""
    n = len(target)
    if n == 1:
        if require_track_start:
            idx = 0 if bouts[0][0] == target[0] else None
        elif require_track_end:
            idx = len(bouts) - 1 if bouts[-1][0] == target[0] else None
        else:
            idx = next((i for i, b in enumerate(bouts) if b[0] == target[0]), None)
        return None if idx is None else (bouts[idx][1], bouts[idx][2])

    if require_track_start and bouts[0][0] != target[0]:
        return None
    if require_track_end and bouts[-1][0] != target[-1]:
        return None

    pointer = 0
    prefix_start_idx = None
    last_prefix_idx = None
    for idx, bout in enumerate(bouts):
        if pointer >= n - 1:
            break
        if bout[0] == target[pointer]:
            if pointer == 0:
                prefix_start_idx = idx
            last_prefix_idx = idx
            pointer += 1
    if pointer < n - 1:
        return None

    if require_track_end:
        end_idx = len(bouts) - 1
        if end_idx <= last_prefix_idx:
            return None
    else:
        end_idx = next(
            (i for i in range(last_prefix_idx + 1, len(bouts)) if bouts[i][0] == target[-1]),
            None,
        )
        if end_idx is None:
            return None

    return (bouts[prefix_start_idx][1], bouts[end_idx][2])


def find_tracks_matching_state_sequence(
    adata,
    target_states,
    state_col,
    match_mode="subsequence",
    sample_col="sample_name",
    track_col="TrackID",
    time_col="position_t",
    require_track_start=False,
    require_track_end=False,
):
    """Find each track's earliest occurrence of an ordered state-bout sequence.

    ``match_mode``:
      - "subsequence" (default): ``target_states`` must appear in that order
        somewhere in the track; other states may occur in between.
      - "contiguous": ``target_states`` must appear as directly consecutive
        bouts (no other state bout in between).

    ``require_track_start``/``require_track_end``: if set, the first/last
    target state must be the track's actual first/last bout.

    Accepts any object with an ``.obs`` DataFrame (an ``AnnData`` or a thin
    wrapper around one) covering one or many samples — searches every
    ``(sample_col, track_col)`` group found in it.

    Returns a DataFrame with one row per matching track (its earliest match
    only): sample_name, TrackID, start_time, end_time, n_states_matched.
    """
    if match_mode not in ("contiguous", "subsequence"):
        raise ValueError(f"match_mode must be 'contiguous' or 'subsequence', got {match_mode!r}")

    target = [str(x).strip() for x in (target_states or []) if str(x).strip() != ""]
    if len(target) == 0:
        return pd.DataFrame(columns=_SEQUENCE_MATCH_RESULT_COLUMNS)

    _validate_required_obs_columns(
        adata.obs, required_cols=[sample_col, track_col, time_col, state_col]
    )

    work = adata.obs[[sample_col, track_col, time_col, state_col]].copy()
    work["_sample"] = work[sample_col].astype("string").str.strip()
    work["_track"] = pd.to_numeric(work[track_col], errors="coerce")
    work["_time"] = pd.to_numeric(work[time_col], errors="coerce")
    work["_state"] = work[state_col].astype("string").str.strip()
    work = work.dropna(subset=["_sample", "_track", "_time"])
    # Unassigned timepoints are gaps, not a "nan" bout.
    work = work[(work["_sample"] != "") & work["_state"].notna() & (work["_state"] != "")]
    if len(work) == 0:
        return pd.DataFrame(columns=_SEQUENCE_MATCH_RESULT_COLUMNS)
    work["_track"] = work["_track"].astype(np.int64)
    work["_time"] = work["_time"].astype(np.int64)
    work = work.sort_values(["_sample", "_track", "_time"], kind="mergesort")

    matcher = _match_contiguous_bouts if match_mode == "contiguous" else _match_subsequence_bouts

    rows = []
    for (sample, track), g in work.groupby(["_sample", "_track"], sort=False, observed=True):
        bouts = _collapse_into_bouts(g["_state"].tolist(), g["_time"].tolist())
        if len(bouts) < len(target):
            continue
        match = matcher(bouts, target, require_track_start, require_track_end)
        if match is None:
            continue
        rows.append(
            {
                "sample_name": str(sample),
                "TrackID": int(track),
                "start_time": int(match[0]),
                "end_time": int(match[1]),
                "n_states_matched": len(target),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=_SEQUENCE_MATCH_RESULT_COLUMNS)
    out = pd.DataFrame(rows, columns=_SEQUENCE_MATCH_RESULT_COLUMNS)
    return out.sort_values(["sample_name", "TrackID"], kind="mergesort").reset_index(drop=True)

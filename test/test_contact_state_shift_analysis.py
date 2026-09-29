import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.state.contact_state_shift import (
    _first_qualifying_bout_bounds,
    compute_contact_bout_windows,
    compute_state_shift_features,
    BEFORE_LABEL,
)
from behav3d.analysis.behavior.track.contact_grouping import compute_track_contact_features
from behav3d.analysis.behavior.state.visualization.plots.contact_state_shift_report import (
    save_state_contact_shift_report,
)

_N_TIMEPOINTS = 40
_MIN_BOUT_LENGTH = 5
_STATE_COL = "full_behavioral_state"

# (TrackID, contact_group, bout_start) — bout occupies [bout_start, bout_start + _MIN_BOUT_LENGTH - 1]
# for "contact" tracks; "no_contact" tracks never have any contact timepoints.
_TRACK_SPECS = [
    (0, "contact", 10),
    (1, "contact", 20),
    (2, "contact", 30),
    (3, "no_contact", None),
    (4, "no_contact", None),
    (5, "no_contact", None),
]


def _state_for_contact_track(t, bout_start):
    bout_end = bout_start + _MIN_BOUT_LENGTH - 1
    if t < bout_start:
        return "static"
    if t <= bout_end:
        return "scanning"
    return "engaging"


def _build_df_timepoints(track_id_dtype=int):
    rows = []
    for track_id, group, bout_start in _TRACK_SPECS:
        contact = np.zeros(_N_TIMEPOINTS, dtype=int)
        if group == "contact":
            bout_end = bout_start + _MIN_BOUT_LENGTH - 1
            contact[bout_start : bout_end + 1] = 1
        for t in range(_N_TIMEPOINTS):
            rows.append({
                "sample_name": "sample_1",
                "TrackID": track_id_dtype(track_id),
                "position_t": t,
                "macro_contact": int(contact[t]),
            })
    return pd.DataFrame(rows)


def _build_adata_states(track_id_dtype=int):
    rows = []
    for track_id, group, bout_start in _TRACK_SPECS:
        for t in range(_N_TIMEPOINTS):
            if group == "contact":
                state = _state_for_contact_track(t, bout_start)
            else:
                state = "static"
            rows.append({
                "sample_name": "sample_1",
                "TrackID": track_id_dtype(track_id),
                "position_t": t,
                _STATE_COL: state,
            })
    obs = pd.DataFrame(rows)
    obs.index = [str(i) for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _build_adata_tracks_for_shared_grouping_check(track_id_dtype=int):
    """Minimal adata_tracks, used only by `compute_track_contact_features` (the track-side
    contact grouping this module's bout detection must agree with) — unrelated to the
    state-native `compute_contact_bout_windows` under test here."""
    obs = pd.DataFrame(
        {
            "sample_name": ["sample_1"] * len(_TRACK_SPECS),
            "TrackID": [track_id_dtype(t[0]) for t in _TRACK_SPECS],
            "position_t_min": 0,
            "position_t_max": _N_TIMEPOINTS - 1,
        }
    )
    obs.index = [str(t[0]) for t in _TRACK_SPECS]
    return ad.AnnData(X=np.zeros((len(_TRACK_SPECS), 1)), obs=obs)


def test_first_qualifying_bout_uses_earliest_run():
    times = list(range(20))
    # Two qualifying runs (length >= 5): [2..6] and [12..17]; must return the first.
    is_contact = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    start_t, end_t = _first_qualifying_bout_bounds(times, is_contact, min_bout_length=5)
    assert (start_t, end_t) == (2, 6)


def test_first_qualifying_bout_skips_short_runs():
    times = list(range(10))
    is_contact = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0]  # first run length 2 (too short), second length 5
    start_t, end_t = _first_qualifying_bout_bounds(times, is_contact, min_bout_length=5)
    assert (start_t, end_t) == (3, 7)


def test_first_qualifying_bout_none_when_no_run_qualifies():
    times = list(range(10))
    is_contact = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    start_t, end_t = _first_qualifying_bout_bounds(times, is_contact, min_bout_length=5)
    assert (start_t, end_t) == (None, None)


def test_shared_min_bout_length_matches_existing_contact_grouping():
    """The contact/no_contact split produced here must agree with
    contact_grouping.compute_track_contact_features for the same min_bout_length, since both
    analyses are meant to share one definition of 'a real contact bout'."""
    adata_states = _build_adata_states()
    adata_tracks = _build_adata_tracks_for_shared_grouping_check()
    df_timepoints = _build_df_timepoints()

    windows = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5,
    )
    existing = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
    )

    for track_id, group, _bout_start in _TRACK_SPECS:
        key = ("sample_1", str(track_id))
        assert windows.loc[key, "contact_group"] == group
        assert existing.loc[key, "macro_contact_group"] == group


def test_fixed_window_mode_bounds():
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()

    windows = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5,
    )
    row = windows.loc[("sample_1", "0")]  # bout_start=10, bout_end=14
    assert row["before_start_t"] == 5
    assert row["before_end_t"] == 10
    assert row["after_start_t"] == 14
    assert row["after_end_t"] == 19
    assert row["before_n_timepoints"] == 5
    assert row["after_n_timepoints"] == 5
    assert not row["excluded"]


def test_full_window_mode_bounds():
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()

    windows = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="full",
    )
    row = windows.loc[("sample_1", "0")]  # bout_start=10, bout_end=14, track span [0, 39]
    assert row["before_start_t"] == 0
    assert row["before_end_t"] == 10
    assert row["after_start_t"] == 14
    assert row["after_end_t"] == 39
    assert row["before_n_timepoints"] == 10
    assert row["after_n_timepoints"] == 25


def test_min_window_timepoints_excludes_short_windows():
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()

    # bout_start=10 with fixed_window_length=20 -> before window would need [-10, 10), but is
    # clipped to [0, 10) => only 10 timepoints, which is fine; force a short window instead by
    # requiring more timepoints than available before t=10.
    windows = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5, min_window_timepoints=6,
    )
    row = windows.loc[("sample_1", "0")]  # before/after windows both have exactly 5 timepoints < 6
    assert row["excluded"]
    assert row["excluded_reason"] == "both_too_short"


def test_null_reference_reproducible_with_seed():
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()

    windows_a = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5, null_seed=42,
    )
    windows_b = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5, null_seed=42,
    )
    windows_c = compute_contact_bout_windows(
        df_timepoints, adata_states, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        window_mode="fixed", fixed_window_length=5, null_seed=7,
    )

    no_contact_keys = [("sample_1", str(t[0])) for t in _TRACK_SPECS if t[1] == "no_contact"]
    bout_starts_a = windows_a.loc[no_contact_keys, "bout_start_t"]
    bout_starts_b = windows_b.loc[no_contact_keys, "bout_start_t"]
    bout_starts_c = windows_c.loc[no_contact_keys, "bout_start_t"]

    assert bout_starts_a.tolist() == bout_starts_b.tolist()
    assert bout_starts_a.tolist() != bout_starts_c.tolist()


def test_state_shift_features_and_report(tmp_path):
    df_timepoints = _build_df_timepoints()
    adata_states = _build_adata_states()

    result = save_state_contact_shift_report(
        df_timepoints, adata_states, tmp_path,
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        state_col=_STATE_COL,
        window_mode="fixed",
        fixed_window_length=5,
        min_window_timepoints=3,
        verbose=False,
    )

    assert result["n_contact_tracks"] == 3
    assert result["n_no_contact_tracks"] == 3
    assert result["n_excluded_tracks"] == 0
    assert Path(result["pdf_path"]).exists()
    assert Path(result["track_windows_csv"]).exists()

    diff_csv = pd.read_csv(result["diff_bars_csv"])
    contact_rows = diff_csv[diff_csv["contact_group"] == "contact"].set_index("state")
    # Contact tracks: fully "static" before the bout, fully "engaging" after it.
    assert contact_rows.loc["static", "diff"] == pytest.approx(-1.0)
    assert contact_rows.loc["engaging", "diff"] == pytest.approx(1.0)
    assert contact_rows.loc["scanning", "diff"] == pytest.approx(0.0)

    no_contact_rows = diff_csv[diff_csv["contact_group"] == "no_contact"].set_index("state")
    # No-contact tracks: constant "static" the entire time -> no shift, regardless of where the
    # null reference split point landed.
    assert no_contact_rows.loc["static", "diff"] == pytest.approx(0.0)
    assert no_contact_rows.loc["engaging", "diff"] == pytest.approx(0.0)

    stacked_csv = pd.read_csv(result["stacked_composition_csv"])
    contact_before = stacked_csv[
        (stacked_csv["contact_group"] == "contact") & (stacked_csv["period"] == BEFORE_LABEL)
    ].set_index("state")
    assert contact_before.loc["static", "proportion"] == pytest.approx(1.0)


def test_state_shift_report_groups_by_extra_condition_column(tmp_path):
    """`extra_group_cols` (metadata columns merged into `adata_states.obs`, e.g. by the widget's
    "Group per page" multi-select) must write one 2x2-layout page per unique combination of their
    values, labeled in its title, and tag the CSV rows with those columns — restricted to the
    tracks in that combination rather than pooling the whole dataset into a single page."""
    pypdf = pytest.importorskip("pypdf")

    conditions = {"sample_1": "control", "sample_2": "treated"}
    rows_time, rows_states = [], []
    for sample_name in conditions:
        for track_id, group, bout_start in _TRACK_SPECS:
            contact = np.zeros(_N_TIMEPOINTS, dtype=int)
            if group == "contact":
                bout_end = bout_start + _MIN_BOUT_LENGTH - 1
                contact[bout_start : bout_end + 1] = 1
            for t in range(_N_TIMEPOINTS):
                rows_time.append({
                    "sample_name": sample_name, "TrackID": track_id, "position_t": t,
                    "macro_contact": int(contact[t]),
                })
                state = _state_for_contact_track(t, bout_start) if group == "contact" else "static"
                rows_states.append({
                    "sample_name": sample_name, "TrackID": track_id, "position_t": t,
                    _STATE_COL: state, "condition": conditions[sample_name],
                })
    df_timepoints = pd.DataFrame(rows_time)
    obs = pd.DataFrame(rows_states)
    obs.index = [str(i) for i in range(len(obs))]
    adata_states = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)

    result = save_state_contact_shift_report(
        df_timepoints, adata_states, tmp_path,
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        state_col=_STATE_COL,
        window_mode="fixed",
        fixed_window_length=5,
        min_window_timepoints=3,
        extra_group_cols=["condition"],
        verbose=False,
    )

    reader = pypdf.PdfReader(result["pdf_path"])
    assert len(reader.pages) == 2
    page_texts = [page.extract_text() or "" for page in reader.pages]
    assert any("condition=control" in text for text in page_texts)
    assert any("condition=treated" in text for text in page_texts)

    diff_csv = pd.read_csv(result["diff_bars_csv"])
    assert set(diff_csv["condition"].unique()) == {"control", "treated"}
    stacked_csv = pd.read_csv(result["stacked_composition_csv"])
    assert set(stacked_csv["condition"].unique()) == {"control", "treated"}
    windows_csv = pd.read_csv(result["track_windows_csv"])
    assert set(windows_csv["condition"].unique()) == {"control", "treated"}


def test_track_missing_from_csv_raises():
    """A track present in adata_states.obs but entirely absent from df_timepoints (stale filtered
    CSV) must raise a hard error instead of being silently dropped from the output."""
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()
    # Drop all rows for TrackID 0 from the CSV, simulating a re-run of filtering that removed it.
    df_timepoints = df_timepoints[df_timepoints["TrackID"] != 0]

    with pytest.raises(ValueError, match="have no matching timepoints in the filtered track-features CSV"):
        compute_contact_bout_windows(
            df_timepoints, adata_states, contact_col="macro_contact",
            min_bout_length=_MIN_BOUT_LENGTH, window_mode="fixed", fixed_window_length=5,
        )


def test_track_present_in_csv_but_shifted_out_of_range_raises():
    """A track present in df_timepoints but entirely shifted outside any plausible overlap with
    adata_states.obs's own timepoints for that track must also raise — since a state-classified
    track's window bounds are derived directly from df_timepoints itself, this specifically
    covers a track whose CSV rows use a completely different position_t range than what
    adata_states.obs expects it to have (e.g. after a re-run desynced the two sources)."""
    adata_states = _build_adata_states()
    df_timepoints = _build_df_timepoints()
    # Drop TrackID 0 from the CSV entirely (same failure mode as above, reached via a different
    # starting scenario description) to keep this a hard, unambiguous desync.
    df_timepoints = df_timepoints[df_timepoints["TrackID"] != 0]

    with pytest.raises(ValueError, match="have no matching timepoints in the filtered track-features CSV"):
        compute_state_shift_features(
            df_timepoints, adata_states, contact_col="macro_contact",
            min_bout_length=_MIN_BOUT_LENGTH, state_col=_STATE_COL,
            window_mode="fixed", fixed_window_length=5,
        )


def test_dtype_drift_across_both_sources():
    """TrackID as int in the states adata vs. '5.0'-style string in a re-read CSV — the join must
    still succeed for both sources."""
    df_timepoints = _build_df_timepoints(track_id_dtype=lambda x: f"{float(x)}")
    adata_states = _build_adata_states(track_id_dtype=int)

    features = compute_state_shift_features(
        df_timepoints, adata_states,
        contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH, state_col=_STATE_COL,
        window_mode="fixed", fixed_window_length=5,
    )
    assert features["n_contact_tracks"] == 3
    assert features["n_no_contact_tracks"] == 3
    assert len(features["state_timepoints"]) > 0

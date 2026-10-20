import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    merge_track_contact_features_into_obs,
    build_target_class_lookup_from_state_adata,
    build_target_class_lookup_from_track_adata,
    compute_track_contact_target_class_features,
    merge_track_target_class_group_into_obs,
    subset_to_target_period,
    build_target_windows_from_track_adata,
    build_target_windows_from_state_adata,
    compute_target_window_contact,
    contact_col_target_cell_type,
    touching_column_name,
)

_N_TIMEPOINTS = 20
_MIN_BOUT_LENGTH = 5


def test_contact_col_target_cell_type():
    assert contact_col_target_cell_type("macro_contact") == "macro"
    assert contact_col_target_cell_type("macro_contact_on_distance") == "macro"
    with pytest.raises(ValueError):
        contact_col_target_cell_type("macro_distance")


def test_touching_column_name():
    assert touching_column_name("macro") == "touching_macros"


def _build_reference_tracks():
    """4 T cell tracks: 0 contacts only round macros, 1 only elongated, 2 both (switching
    mid-bout, so the *overall* min-bout-length is met by a run that spans two different target
    classes), 3 never contacts."""
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * 4,
            "TrackID": [0, 1, 2, 3],
            "position_t_min": 0,
            "position_t_max": _N_TIMEPOINTS - 1,
        }
    )
    obs.index = [str(i) for i in range(4)]
    return ad.AnnData(X=np.zeros((4, 1)), obs=obs)


def _build_reference_timepoints():
    rows = []

    def add(track_id, contact_flags, touching):
        for t in range(_N_TIMEPOINTS):
            rows.append(
                {
                    "sample_name": "s1",
                    "TrackID": track_id,
                    "position_t": t,
                    "macro_contact": int(contact_flags[t]),
                    "touching_macros": touching[t],
                }
            )

    add(0, [1] * 6 + [0] * (_N_TIMEPOINTS - 6), ["100"] * 6 + [""] * (_N_TIMEPOINTS - 6))
    add(1, [1] * 6 + [0] * (_N_TIMEPOINTS - 6), ["101"] * 6 + [""] * (_N_TIMEPOINTS - 6))
    # 3 timepoints touching macro 100 (round), immediately followed by 3 touching macro 101
    # (elongated) — a single contiguous 6-timepoint "any contact" bout (>= min_bout_length),
    # but neither class alone reaches the threshold on its own.
    add(
        2,
        [1] * 6 + [0] * (_N_TIMEPOINTS - 6),
        ["100"] * 3 + ["101"] * 3 + [""] * (_N_TIMEPOINTS - 6),
    )
    add(3, [0] * _N_TIMEPOINTS, [""] * _N_TIMEPOINTS)
    return pd.DataFrame(rows)


def _build_target_states_adata():
    """Target (macro) per-timepoint states: TrackID 100 = "round" always, 101 = "elongated"."""
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * (2 * _N_TIMEPOINTS),
            "TrackID": [100] * _N_TIMEPOINTS + [101] * _N_TIMEPOINTS,
            "position_t": list(range(_N_TIMEPOINTS)) * 2,
            "full_behavioral_state": ["round"] * _N_TIMEPOINTS + ["elongated"] * _N_TIMEPOINTS,
        }
    )
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _build_target_track_classification_adata():
    """Target (macro) whole-track classification: TrackID 100 = "round", 101 = "elongated"."""
    obs = pd.DataFrame(
        {
            "sample_name": ["s1", "s1"],
            "TrackID": [100, 101],
            "ClusterID": ["round", "elongated"],
        }
    )
    obs.index = ["0", "1"]
    return ad.AnnData(X=np.zeros((2, 1)), obs=obs)


def _run_overall_contact_grouping(adata_tracks, df_timepoints):
    features = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
    )
    merge_track_contact_features_into_obs(
        adata_tracks, features, contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
    )


def _bucket_by_track_id(adata_tracks, col):
    return adata_tracks.obs.set_index("TrackID")[col].to_dict()


@pytest.mark.parametrize("time_varying", [True, False])
def test_target_class_group_bucketing(time_varying):
    adata_tracks = _build_reference_tracks()
    df_timepoints = _build_reference_timepoints()
    _run_overall_contact_grouping(adata_tracks, df_timepoints)

    if time_varying:
        adata_states = _build_target_states_adata()
        lookup = build_target_class_lookup_from_state_adata(adata_states, state_col="full_behavioral_state")
    else:
        adata_target_tracks = _build_target_track_classification_adata()
        lookup = build_target_class_lookup_from_track_adata(adata_target_tracks, class_col="ClusterID")

    long_df, group_df = compute_track_contact_target_class_features(
        df_timepoints, adata_tracks, lookup,
        contact_col="macro_contact", touching_col="touching_macros", time_varying=time_varying,
    )
    merge_track_target_class_group_into_obs(adata_tracks, group_df, contact_col="macro_contact")

    buckets = _bucket_by_track_id(adata_tracks, "macro_contact_target_class_group")
    assert buckets["0"] == "round"
    assert buckets["1"] == "elongated"
    assert buckets["2"] == "Multiple classes"
    assert buckets["3"] == "no_contact"

    # Track 2 touched both classes -> 2 rows in the long-format per-class table; tracks 0/1 -> 1 each.
    long_flat = long_df.reset_index()
    assert set(long_flat.loc[long_flat["TrackID"] == "0", "target_class"]) == {"round"}
    assert set(long_flat.loc[long_flat["TrackID"] == "1", "target_class"]) == {"elongated"}
    assert set(long_flat.loc[long_flat["TrackID"] == "2", "target_class"]) == {"round", "elongated"}
    assert len(long_flat.loc[long_flat["TrackID"] == "3"]) == 0

    # Track 0 touched macro 100 (round) at every one of its 6 contact timepoints, contiguously.
    row0 = long_flat[(long_flat["TrackID"] == "0") & (long_flat["target_class"] == "round")].iloc[0]
    assert row0["macro_contact_class_mean_fraction"] == pytest.approx(6 / _N_TIMEPOINTS)
    assert row0["macro_contact_class_max_bout_length"] == 6

    # Track 2 split its 6-timepoint bout 3/3 between round and elongated.
    row2_round = long_flat[(long_flat["TrackID"] == "2") & (long_flat["target_class"] == "round")].iloc[0]
    row2_elong = long_flat[(long_flat["TrackID"] == "2") & (long_flat["target_class"] == "elongated")].iloc[0]
    assert row2_round["macro_contact_class_max_bout_length"] == 3
    assert row2_elong["macro_contact_class_max_bout_length"] == 3


def test_missing_touching_column_raises_clear_error():
    adata_tracks = _build_reference_tracks()
    df_timepoints = _build_reference_timepoints().drop(columns=["touching_macros"])
    _run_overall_contact_grouping(adata_tracks, df_timepoints)

    adata_states = _build_target_states_adata()
    lookup = build_target_class_lookup_from_state_adata(adata_states, state_col="full_behavioral_state")

    with pytest.raises(KeyError, match="touching_macros"):
        compute_track_contact_target_class_features(
            df_timepoints, adata_tracks, lookup,
            contact_col="macro_contact", touching_col="touching_macros", time_varying=True,
        )


def test_build_target_class_lookup_from_state_adata_missing_column_raises():
    adata_states = _build_target_states_adata()
    with pytest.raises(KeyError):
        build_target_class_lookup_from_state_adata(adata_states, state_col="does_not_exist")


def test_build_target_class_lookup_from_track_adata_missing_column_raises():
    adata_target_tracks = _build_target_track_classification_adata()
    with pytest.raises(KeyError):
        build_target_class_lookup_from_track_adata(adata_target_tracks, class_col="does_not_exist")


def _build_windowed_tracks():
    """T cell trajectory windows (t = 0-29 recorded): track 0 has two windows inside the target
    classification period (0-19), track 1's window crosses its end, track 2's lies after it."""
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * 4,
            "TrackID": [0, 0, 1, 2],
            "trajectory_window_id": [0, 1, 0, 0],
            "position_t_min": [0, 10, 15, 20],
            "position_t_max": [9, 19, 24, 29],
        }
    )
    obs.index = [str(i) for i in range(4)]
    return ad.AnnData(X=np.zeros((4, 1)), obs=obs)


def _build_windowed_timepoints():
    """Every T cell touches macro 100 at the first 3 timepoints of each of its windows."""
    contact_starts = {0: [0, 10], 1: [15], 2: [20]}
    rows = []
    for track_id, starts in contact_starts.items():
        touched = {t for start in starts for t in range(start, start + 3)}
        for t in range(30):
            rows.append(
                {
                    "sample_name": "s1",
                    "TrackID": track_id,
                    "position_t": t,
                    "macro_contact": int(t in touched),
                    "touching_macros": "100" if t in touched else "",
                }
            )
    return pd.DataFrame(rows)


def _build_windowed_target_track_adata():
    """Macro 100 is "round" in its window t = 0-9 and "elongated" in t = 10-19; nothing after."""
    obs = pd.DataFrame(
        {
            "sample_name": ["s1", "s1"],
            "TrackID": [100, 100],
            "trajectory_window_id": [0, 1],
            "position_t_min": [0, 10],
            "position_t_max": [9, 19],
            "ClusterID": ["round", "elongated"],
        }
    )
    obs.index = ["0", "1"]
    return ad.AnnData(X=np.zeros((2, 1)), obs=obs)


def test_track_lookup_with_windows_is_time_varying():
    lookup = build_target_class_lookup_from_track_adata(
        _build_windowed_target_track_adata(), class_col="ClusterID",
    )
    assert "position_t" in lookup.columns
    assert len(lookup) == 20
    by_t = lookup.set_index("position_t")["target_class"]
    assert by_t[5] == "round"
    assert by_t[15] == "elongated"


@pytest.mark.parametrize("source", ["track", "state"])
def test_target_class_restricted_to_target_period(source):
    adata_tracks = _build_windowed_tracks()
    df_timepoints = _build_windowed_timepoints()
    features = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col="macro_contact", min_bout_length=3,
    )
    merge_track_contact_features_into_obs(
        adata_tracks, features, contact_col="macro_contact", min_bout_length=3,
    )

    if source == "track":
        lookup = build_target_class_lookup_from_track_adata(
            _build_windowed_target_track_adata(), class_col="ClusterID",
        )
        expected_second_window_class = "elongated"
    else:
        obs = pd.DataFrame(
            {"sample_name": "s1", "TrackID": 100, "position_t": range(20), "state": "round"}
        )
        lookup = build_target_class_lookup_from_state_adata(
            ad.AnnData(X=np.zeros((20, 1)), obs=obs), state_col="state",
        )
        expected_second_window_class = "round"

    # time_varying=False is overridden: the lookup has a time column.
    long_df, group_df = compute_track_contact_target_class_features(
        df_timepoints, adata_tracks, lookup,
        contact_col="macro_contact", touching_col="touching_macros", time_varying=False,
    )

    long_flat = long_df.reset_index()
    assert set(long_flat["TrackID"]) == {"0"}
    by_window = long_flat.set_index("trajectory_window_id")
    assert by_window.loc["0", "target_class"] == "round"
    assert by_window.loc["1", "target_class"] == expected_second_window_class
    assert by_window.loc["0", "macro_contact_class_mean_fraction"] == pytest.approx(3 / 10)

    merge_track_target_class_group_into_obs(adata_tracks, group_df, contact_col="macro_contact")
    labels = adata_tracks.obs.set_index(["TrackID", "trajectory_window_id"])[
        "macro_contact_target_class_group"
    ].astype(str)
    assert labels[("1", "0")] == "outside_target_period"
    assert labels[("2", "0")] == "outside_target_period"

    subset = subset_to_target_period(adata_tracks, contact_col="macro_contact")
    assert len(subset) == 2
    assert "outside_target_period" not in set(subset.obs["macro_contact_target_class_group"].astype(str))


def test_target_period_restriction_can_be_disabled():
    adata_tracks = _build_windowed_tracks()
    df_timepoints = _build_windowed_timepoints()
    features = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col="macro_contact", min_bout_length=3,
    )
    merge_track_contact_features_into_obs(
        adata_tracks, features, contact_col="macro_contact", min_bout_length=3,
    )
    lookup = build_target_class_lookup_from_track_adata(
        _build_windowed_target_track_adata(), class_col="ClusterID",
    )
    long_df, group_df = compute_track_contact_target_class_features(
        df_timepoints, adata_tracks, lookup,
        contact_col="macro_contact", touching_col="touching_macros", time_varying=True,
        restrict_to_target_period=False,
    )
    # Track 1 touched macro 100 at t = 15-17 (classified); track 2 at t = 20-22 (not classified).
    assert set(long_df.reset_index()["TrackID"]) == {"0", "1"}
    assert "outside_target_period" not in set(group_df["macro_contact_target_class_group"].astype(str))


def _source_timepoints(touches, n_timepoints=30):
    """Per-timepoint source-cell rows; ``touches`` maps source TrackID -> {t: "comma,ids"}."""
    rows = []
    for track_id, by_t in touches.items():
        for t in range(n_timepoints):
            rows.append({
                "sample_name": "s1", "TrackID": track_id, "position_t": t,
                "macro_contact": int(t in by_t), "touching_macros": by_t.get(t, ""),
            })
    return pd.DataFrame(rows)


def _window_row(windows, track_id, window_id):
    return windows[(windows["TrackID"] == str(track_id)) & (windows["window_id"] == str(window_id))].iloc[0]


def test_target_window_contact_counts_any_source_cell_once():
    # Macro 100 (two windows of different class), touched by source cells 0 and 1 with overlap in
    # window 0; macro 101 never touched; macro 102's window extends past the source data (t <= 29).
    target = _build_windowed_target_track_adata()
    extra = pd.DataFrame({
        "sample_name": ["s1", "s1"], "TrackID": [101, 102], "trajectory_window_id": [0, 0],
        "position_t_min": [0, 20], "position_t_max": [9, 39], "ClusterID": ["round", "round"],
    })
    obs = pd.concat([target.obs, extra], ignore_index=True)
    obs.index = obs.index.astype(str)
    windows = build_target_windows_from_track_adata(ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs))
    assert len(windows) == 4

    df_timepoints = _source_timepoints({
        0: {t: "100" for t in range(2, 6)},                      # t 2-5
        1: {**{t: "100" for t in range(4, 8)}, 12: "100,101"},   # t 4-7, plus t 12
    })
    result = compute_target_window_contact(df_timepoints, windows, touching_col="touching_macros")

    assert set(zip(result["TrackID"], result["window_id"])) == {("100", "0"), ("100", "1"), ("101", "0")}
    w0 = _window_row(result, 100, 0)
    assert w0["target_class"] == "round"
    assert w0["n_touched_timepoints"] == 6          # t 2-7, overlap 4-5 counted once
    assert w0["duration_timepoints"] == 6
    assert w0["duration_fraction"] == pytest.approx(6 / 10)
    w1 = _window_row(result, 100, 1)
    assert w1["target_class"] == "elongated"
    assert w1["n_touched_timepoints"] == 1
    # Macro 101 is touched at t = 12 only, outside its window (t 0-9).
    assert not bool(_window_row(result, 101, 0)["touched"])


def test_target_window_contact_longest_run_spans_gaps():
    windows = build_target_windows_from_track_adata(_build_windowed_target_track_adata())
    df_timepoints = _source_timepoints({0: {0: "100", 1: "100", 4: "100", 5: "100", 6: "100"}})
    w0 = _window_row(compute_target_window_contact(df_timepoints, windows, touching_col="touching_macros"), 100, 0)
    assert w0["n_touched_timepoints"] == 5
    assert w0["duration_timepoints"] == 3


def test_target_windows_from_state_adata_are_same_state_segments():
    states = ["round"] * 4 + ["elongated"] * 3 + ["round"] * 3
    times = list(range(7)) + [8, 9, 10]   # t = 7 missing -> the last "round" run is its own segment
    obs = pd.DataFrame({"sample_name": "s1", "TrackID": 100, "position_t": times, "state": states})
    windows = build_target_windows_from_state_adata(ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs), state_col="state")
    assert windows[["target_class", "position_t_min", "position_t_max"]].values.tolist() == [
        ["round", 0, 3], ["elongated", 4, 6], ["round", 8, 10],
    ]
    assert windows["window_id"].tolist() == ["0", "1", "2"]

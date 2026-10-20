import anndata as ad
import pandas as pd
import pytest

from behav3d.analysis.behavior.state.sequence_search import (
    find_tracks_matching_state_sequence,
)

_STATE_COL = "full_behavioral_state"


def _obs_from_tracks(track_states):
    """``track_states``: {(sample, track_id): [state_per_t, ...]} -> obs DataFrame."""
    rows = []
    for (sample, track_id), states in track_states.items():
        for t, s in enumerate(states):
            rows.append(
                {
                    "sample_name": sample,
                    "TrackID": track_id,
                    "position_t": t,
                    _STATE_COL: s,
                }
            )
    return pd.DataFrame(rows)


def _adata(track_states):
    return ad.AnnData(obs=_obs_from_tracks(track_states))


def _find(adata, target, **kwargs):
    return find_tracks_matching_state_sequence(
        adata, target_states=target, state_col=_STATE_COL, **kwargs
    )


def test_contiguous_match():
    adata = _adata(
        {("s1", 1): ["static", "static", "scanner", "scanner", "engager"]}
    )
    out = _find(adata, ["static", "scanner", "engager"], match_mode="contiguous")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sample_name"] == "s1"
    assert row["TrackID"] == 1
    assert row["start_time"] == 0
    assert row["end_time"] == 4


def test_contiguous_fails_with_gap():
    adata = _adata({("s1", 1): ["static", "scanner", "idle", "engager"]})
    out = _find(adata, ["static", "scanner", "engager"], match_mode="contiguous")
    assert len(out) == 0


def test_subsequence_matches_with_gap():
    adata = _adata({("s1", 1): ["static", "scanner", "idle", "engager"]})
    out = _find(adata, ["static", "scanner", "engager"], match_mode="subsequence")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["start_time"] == 0
    assert row["end_time"] == 3


def test_require_track_start_true():
    adata = _adata({("s1", 1): ["idle", "static", "scanner", "engager"]})
    out_anchored = _find(
        adata,
        ["static", "scanner", "engager"],
        match_mode="subsequence",
        require_track_start=True,
    )
    assert len(out_anchored) == 0

    adata2 = _adata({("s1", 1): ["static", "scanner", "engager"]})
    out_ok = _find(
        adata2,
        ["static", "scanner", "engager"],
        match_mode="subsequence",
        require_track_start=True,
    )
    assert len(out_ok) == 1


def test_require_track_end_true():
    adata = _adata({("s1", 1): ["static", "scanner", "engager", "idle"]})
    out_anchored = _find(
        adata,
        ["static", "scanner", "engager"],
        match_mode="subsequence",
        require_track_end=True,
    )
    assert len(out_anchored) == 0

    adata2 = _adata({("s1", 1): ["static", "scanner", "engager"]})
    out_ok = _find(
        adata2,
        ["static", "scanner", "engager"],
        match_mode="subsequence",
        require_track_end=True,
    )
    assert len(out_ok) == 1


def test_require_track_start_and_end_both():
    track_states = {
        ("s1", 1): ["idle", "static", "scanner", "engager"],
        ("s1", 2): ["static", "scanner", "engager"],
        ("s1", 3): ["static", "scanner", "engager", "idle"],
    }
    adata = _adata(track_states)
    out = _find(
        adata,
        ["static", "scanner", "engager"],
        match_mode="subsequence",
        require_track_start=True,
        require_track_end=True,
    )
    assert len(out) == 1
    assert out.iloc[0]["TrackID"] == 2


def test_no_match_wrong_order():
    adata = _adata({("s1", 1): ["engager", "scanner", "static"]})
    out = _find(adata, ["static", "scanner", "engager"], match_mode="subsequence")
    assert len(out) == 0
    assert list(out.columns) == [
        "sample_name",
        "TrackID",
        "start_time",
        "end_time",
        "n_states_matched",
    ]


def test_single_state_target():
    adata = _adata({("s1", 1): ["idle", "scanner", "idle"]})
    out_contig = _find(adata, ["scanner"], match_mode="contiguous")
    out_sub = _find(adata, ["scanner"], match_mode="subsequence")
    for out in (out_contig, out_sub):
        assert len(out) == 1
        assert out.iloc[0]["start_time"] == 1
        assert out.iloc[0]["end_time"] == 1


def test_earliest_match_only_returned():
    adata = _adata(
        {
            ("s1", 1): [
                "static", "scanner", "engager",
                "idle",
                "static", "scanner", "engager",
            ]
        }
    )
    out = _find(adata, ["static", "scanner", "engager"], match_mode="contiguous")
    assert len(out) == 1
    assert out.iloc[0]["start_time"] == 0
    assert out.iloc[0]["end_time"] == 2


def test_nan_gap_does_not_create_phantom_bout():
    obs = _obs_from_tracks({("s1", 1): ["static", "scanner", "engager"]})
    obs.loc[1, _STATE_COL] = None
    adata = ad.AnnData(obs=obs)
    out = _find(adata, ["static", "engager"], match_mode="contiguous")
    assert len(out) == 1
    assert out.iloc[0]["start_time"] == 0
    assert out.iloc[0]["end_time"] == 2


def test_multiple_samples_same_track_id_kept_distinct():
    adata = _adata(
        {
            ("s1", 1): ["static", "scanner", "engager"],
            ("s2", 1): ["engager", "scanner", "static"],
        }
    )
    out = _find(adata, ["static", "scanner", "engager"], match_mode="subsequence")
    assert len(out) == 1
    assert out.iloc[0]["sample_name"] == "s1"


def test_empty_target_states_returns_empty_frame():
    adata = _adata({("s1", 1): ["static", "scanner", "engager"]})
    out = _find(adata, [], match_mode="subsequence")
    assert len(out) == 0
    assert list(out.columns) == [
        "sample_name",
        "TrackID",
        "start_time",
        "end_time",
        "n_states_matched",
    ]


def test_missing_required_column_raises():
    obs = _obs_from_tracks({("s1", 1): ["static", "scanner", "engager"]})
    obs = obs.drop(columns=[_STATE_COL])
    adata = ad.AnnData(obs=obs)
    with pytest.raises(ValueError):
        _find(adata, ["static", "scanner"], match_mode="subsequence")

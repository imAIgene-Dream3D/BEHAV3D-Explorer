import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.track.contact_grouping import (
    build_target_class_lookup_from_track_adata,
    touching_column_name,
)
from behav3d.analysis.behavior.track.visualization.plots.contact_duration_report import (
    save_track_contact_duration_comparison,
    _build_comparisons,
    _compute_per_sample_long_contact_pct,
    _LONG_CONTACT_BUCKET_COL,
)

_N_TIMEPOINTS = 30
_MIN_BOUT_LENGTH = 3

# (sample, class, macro_track_id, [bout lengths for its T-cell tracks]) -- s4 has no "plastic"
# macrophage at all, so paired-by-sample comparisons involving "plastic" must drop s4.
_TOUCH_SPEC = [
    ("s1", "round", 100, [5, 7]),
    ("s2", "round", 100, [6, 8]),
    ("s3", "round", 100, [4, 6]),
    ("s4", "round", 100, [5, 7]),
    ("s1", "elongated", 101, [9, 11]),
    ("s2", "elongated", 101, [10, 12]),
    ("s3", "elongated", 101, [8, 10]),
    ("s4", "elongated", 101, [9, 13]),
    ("s1", "plastic", 102, [13, 15]),
    ("s2", "plastic", 102, [14, 16]),
    ("s3", "plastic", 102, [12, 14]),
]
_SAMPLES = ["s1", "s2", "s3", "s4"]
_MACRO_CLASS_BY_ID = {100: "round", 101: "elongated", 102: "plastic"}


def _build_adata_tracks():
    rows = []
    track_id = 0
    for sample, _cls, _macro_id, bout_lengths in _TOUCH_SPEC:
        for _bout_len in bout_lengths:
            rows.append({
                "sample_name": sample, "TrackID": track_id,
                "position_t_min": 0, "position_t_max": _N_TIMEPOINTS - 1,
            })
            track_id += 1
    obs = pd.DataFrame(rows)
    obs.index = [str(i) for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _build_df_timepoints():
    rows = []
    track_id = 0
    for sample, _cls, macro_id, bout_lengths in _TOUCH_SPEC:
        for bout_len in bout_lengths:
            contact = [1] * bout_len + [0] * (_N_TIMEPOINTS - bout_len)
            touching = [str(macro_id)] * bout_len + [""] * (_N_TIMEPOINTS - bout_len)
            for t in range(_N_TIMEPOINTS):
                rows.append({
                    "sample_name": sample, "TrackID": track_id, "position_t": t,
                    "macro_contact": contact[t], "touching_macros": touching[t],
                })
            track_id += 1
    return pd.DataFrame(rows)


def _build_target_track_adata():
    """One row per (sample, macrophage TrackID) touched anywhere in _TOUCH_SPEC, classified by
    its morphology class -- s4 simply never appears with macro_id=102 ("plastic")."""
    rows = []
    for sample in _SAMPLES:
        macro_ids = sorted({macro_id for s, _c, macro_id, _b in _TOUCH_SPEC if s == sample})
        for macro_id in macro_ids:
            rows.append({
                "sample_name": sample, "TrackID": macro_id, "ClusterID": _MACRO_CLASS_BY_ID[macro_id],
            })
    obs = pd.DataFrame(rows)
    obs.index = [str(i) for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _common_kwargs():
    adata_target = _build_target_track_adata()
    lookup = build_target_class_lookup_from_track_adata(adata_target, class_col="ClusterID")
    return dict(
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        target_class_lookup=lookup,
        touching_col=touching_column_name("macro"),
        time_varying=False,
        target_cell_type_label="macro",
    )


def test_build_comparisons_pairwise_plus_one_vs_rest():
    comparisons = _build_comparisons(["round", "elongated", "plastic"])
    assert len(comparisons) == 6
    pairwise = {(a, b) for a, classes_a, b, classes_b in comparisons if len(classes_a) == 1 and len(classes_b) == 1}
    assert pairwise == {("round", "elongated"), ("round", "plastic"), ("elongated", "plastic")}
    one_vs_rest = [row for row in comparisons if len(row[1]) > 1 or len(row[3]) > 1]
    assert len(one_vs_rest) == 3
    for label_a, classes_a, label_b, classes_b in one_vs_rest:
        assert classes_a == [label_a]
        assert set(classes_b) == {"round", "elongated", "plastic"} - {label_a}

    # 2 classes -> just the one pairwise entry, no duplicate "vs rest".
    assert len(_build_comparisons(["round", "elongated"])) == 1


def _find_row(csv, label_a, label_b):
    """Look up the comparison row for {label_a, label_b} regardless of which side
    ``_build_comparisons`` (sorted by ``_mixed_label_sort_key``) put in group_a vs group_b."""
    match = csv[
        ((csv["group_a"] == label_a) & (csv["group_b"] == label_b))
        | ((csv["group_a"] == label_b) & (csv["group_b"] == label_a))
    ]
    assert len(match) == 1, f"expected exactly one {label_a} vs {label_b} row, found {len(match)}"
    return match.iloc[0]


def _n_for(row, label):
    return row["n_a"] if row["group_a"] == label else row["n_b"]


def test_welch_mode_pdf_csv_and_group_sizes(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", minutes_per_frame=2.0, comparisons_per_page=4, verbose=False,
        **_common_kwargs(),
    )

    assert result["n_comparisons"] == 6
    # 6 comparisons, 4 per page -> 2 box pages, each followed by its "connected by sample"
    # companion page -> 4 pages total.
    assert result["n_pages"] == 4
    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()

    csv = pd.read_csv(result["csv_path"])
    assert len(csv) == 6
    assert set(csv["test_mode"]) == {"welch"}
    assert csv["pairing_col"].isna().all()

    round_vs_elong = _find_row(csv, "round", "elongated")
    # round touched by 2 tracks/sample x 4 samples = 8; elongated likewise 8.
    assert _n_for(round_vs_elong, "round") == 8
    assert _n_for(round_vs_elong, "elongated") == 8

    round_vs_plastic = _find_row(csv, "round", "plastic")
    # plastic only touched in s1-s3 -> 2 tracks x 3 samples = 6.
    assert _n_for(round_vs_plastic, "plastic") == 6

    # minutes = timepoints * minutes_per_frame for every row.
    assert csv["mean_a_minutes"].to_numpy() == pytest.approx(csv["mean_a_timepoints"].to_numpy() * 2.0)
    assert csv["mean_b_minutes"].to_numpy() == pytest.approx(csv["mean_b_timepoints"].to_numpy() * 2.0)


def test_main_comparison_includes_fraction_columns_and_stats(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", minutes_per_frame=2.0, verbose=False,
        **_common_kwargs(),
    )
    csv = pd.read_csv(result["csv_path"])
    for col in [
        "mean_a_fraction", "mean_b_fraction", "diff_fraction",
        "t_stat_fraction", "p_value_fraction", "stars_fraction",
    ]:
        assert col in csv.columns

    round_vs_elong = _find_row(csv, "round", "elongated")
    # Every track's classified window is _N_TIMEPOINTS long here, so duration_fraction is a
    # fixed rescale (1/_N_TIMEPOINTS) of duration_timepoints for every row -- hence
    # mean_*_fraction == mean_*_timepoints / _N_TIMEPOINTS, and Welch's t-stat/p-value (scale
    # invariant under a positive constant rescale) match the timepoints test exactly.
    assert round_vs_elong["mean_a_fraction"] == pytest.approx(round_vs_elong["mean_a_timepoints"] / _N_TIMEPOINTS)
    assert round_vs_elong["mean_b_fraction"] == pytest.approx(round_vs_elong["mean_b_timepoints"] / _N_TIMEPOINTS)
    assert round_vs_elong["t_stat_fraction"] == pytest.approx(round_vs_elong["t_stat"])
    assert round_vs_elong["p_value_fraction"] == pytest.approx(round_vs_elong["p_value"])
    assert round_vs_elong["stars_fraction"] == round_vs_elong["stars"]


def test_paired_mode_drops_incomplete_pairing_units(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="paired", pairing_col="sample_name", minutes_per_frame=None, verbose=False,
        **_common_kwargs(),
    )
    assert result["n_comparisons"] == 6
    assert result["pairing_col"] == "sample_name"

    csv = pd.read_csv(result["csv_path"])
    assert set(csv["test_mode"]) == {"paired"}
    assert (csv["pairing_col"] == "sample_name").all()
    # No time metadata given -> minutes columns are all NaN, but nothing errors.
    assert csv["mean_a_minutes"].isna().all()

    round_vs_elong = _find_row(csv, "round", "elongated")
    assert round_vs_elong["n_a"] == 4  # all 4 samples have both round and elongated touches
    assert round_vs_elong["n_b"] == 4

    round_vs_plastic = _find_row(csv, "round", "plastic")
    # s4 has no "plastic" touch -> dropped from the paired comparison.
    assert round_vs_plastic["n_a"] == 3
    assert round_vs_plastic["n_b"] == 3


def test_paired_mode_includes_fraction_columns(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="paired", pairing_col="sample_name", verbose=False,
        **_common_kwargs(),
    )
    csv = pd.read_csv(result["csv_path"])
    round_vs_elong = _find_row(csv, "round", "elongated")
    assert round_vs_elong["mean_a_fraction"] == pytest.approx(round_vs_elong["mean_a_timepoints"] / _N_TIMEPOINTS)
    assert np.isfinite(round_vs_elong["t_stat_fraction"])


def test_paired_mode_requires_pairing_col():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="pairing_col"):
        save_track_contact_duration_comparison(
            adata_tracks, df_timepoints, Path("unused"),
            test_mode="paired", pairing_col=None, **_common_kwargs(),
        )


def test_invalid_test_mode_raises():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="test_mode"):
        save_track_contact_duration_comparison(
            adata_tracks, df_timepoints, Path("unused"),
            test_mode="bogus", **_common_kwargs(),
        )


def test_long_contact_percentage_splits_long_vs_short(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    # Every track's classified window is _N_TIMEPOINTS long and each bout is a single contiguous
    # run starting at t=0, so duration_fraction == bout_length / _N_TIMEPOINTS exactly -- no time
    # metadata needed.
    # round bouts (2 per sample x 4 samples): s1=[5,7] s2=[6,8] s3=[4,6] s4=[5,7]
    # -> fraction*100 >= 7/30*100 selects the 7,8,7-timepoint bouts = 3 of 8 "long".
    long_contact_threshold = 100.0 * 7.0 / _N_TIMEPOINTS
    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", long_contact_threshold=long_contact_threshold, long_contact_unit="percent",
        verbose=False, **_common_kwargs(),
    )
    assert result["long_contact_threshold"] == pytest.approx(long_contact_threshold)
    assert result["long_contact_unit"] == "percent"
    assert Path(result["long_contact_csv_path"]).exists()

    long_csv = pd.read_csv(result["long_contact_csv_path"])
    assert set(long_csv["page_group"]) == {"(all)"}

    round_row = long_csv[long_csv["target_class"] == "round"].iloc[0]
    assert round_row["n_total"] == 8
    assert round_row["n_long_contact"] == 3
    assert round_row["n_short_contact"] == 5
    assert round_row["pct_long_contact"] == pytest.approx(3 / 8)
    assert round_row["long_contact_threshold"] == pytest.approx(long_contact_threshold)
    assert round_row["long_contact_unit"] == "percent"


def test_long_contact_minutes_uses_bout_length_not_fraction(tmp_path):
    """With unit='minutes', bucketing uses the longest sustained bout converted via
    ``minutes_per_frame`` -- not ``duration_fraction`` -- so it requires time metadata."""
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    # minutes_per_frame=2.0 -> round bouts (timepoints [5,7,6,8,4,6,5,7]) become minutes
    # [10,14,12,16,8,12,10,14]; threshold=13 selects the two 14- and one 16-minute bouts = 3 of 8.
    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", minutes_per_frame=2.0, long_contact_threshold=13.0,
        long_contact_unit="minutes", verbose=False, **_common_kwargs(),
    )
    long_csv = pd.read_csv(result["long_contact_csv_path"])
    round_row = long_csv[long_csv["target_class"] == "round"].iloc[0]
    assert round_row["n_long_contact"] == 3
    assert round_row["n_short_contact"] == 5
    assert round_row["long_contact_unit"] == "minutes"


def test_long_contact_seconds_and_hours_are_consistent_with_minutes(tmp_path):
    """Same threshold expressed in seconds/hours should select the same tracks as minutes."""

    def _n_long(threshold, unit, tmp_subdir):
        result = save_track_contact_duration_comparison(
            _build_adata_tracks(), _build_df_timepoints(), tmp_path / tmp_subdir,
            test_mode="welch", minutes_per_frame=2.0, verbose=False,
            long_contact_threshold=threshold, long_contact_unit=unit,
            **_common_kwargs(),
        )
        long_csv = pd.read_csv(result["long_contact_csv_path"])
        return int(long_csv[long_csv["target_class"] == "round"].iloc[0]["n_long_contact"])

    assert _n_long(13.0, "minutes", "min") == _n_long(13.0 * 60.0, "seconds", "sec")
    assert _n_long(13.0, "minutes", "min2") == _n_long(13.0 / 60.0, "hours", "hr")


def test_long_contact_non_percent_unit_requires_minutes_per_frame():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="minutes_per_frame"):
        save_track_contact_duration_comparison(
            adata_tracks, df_timepoints, Path("unused"),
            test_mode="welch", long_contact_threshold=5.0, long_contact_unit="minutes",
            minutes_per_frame=None, **_common_kwargs(),
        )


def test_long_contact_per_sample_pct_for_boxplot():
    # round bouts per sample (2 tracks/sample): s1=[5,7] s2=[6,8] s3=[4,6] s4=[5,7];
    # threshold=7 -> each sample has exactly 1 of its 2 round tracks "long", except s3 (0 of 2).
    duration_df = pd.DataFrame({
        "sample_name": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "target_class": ["round"] * 8,
        "duration_timepoints": [5, 7, 6, 8, 4, 6, 5, 7],
    })
    duration_df[_LONG_CONTACT_BUCKET_COL] = np.where(
        duration_df["duration_timepoints"] >= 7, "long_contact", "short_contact",
    )
    per_sample = _compute_per_sample_long_contact_pct(
        duration_df, sample_col="sample_name", class_order=["round"],
    )
    pct_by_sample = per_sample.set_index("sample_name")["pct_long_contact"].to_dict()
    assert pct_by_sample == {"s1": 50.0, "s2": 50.0, "s3": 0.0, "s4": 50.0}


def test_long_contact_percent_threshold_must_be_in_valid_range():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="long_contact_threshold"):
        save_track_contact_duration_comparison(
            adata_tracks, df_timepoints, Path("unused"),
            test_mode="welch", long_contact_threshold=150.0, long_contact_unit="percent",
            **_common_kwargs(),
        )


def test_long_contact_invalid_unit_raises():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="long_contact_unit"):
        save_track_contact_duration_comparison(
            adata_tracks, df_timepoints, Path("unused"),
            test_mode="welch", long_contact_threshold=5.0, long_contact_unit="fortnights",
            **_common_kwargs(),
        )


def test_long_contact_percent_threshold_matches_r_script_semantics(tmp_path):
    """Mirrors the originating R workflow's ``mean(tcell_contact == "True") > 0.25`` rule: a
    track is "long_contact" for a class once the *fraction* of its classified window spent
    touching that class clears the threshold -- independent of bout contiguity/length."""
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    # round bouts (2/sample x 4 samples): s1=[5,7] s2=[6,8] s3=[4,6] s4=[5,7] -> fractions
    # (bout/_N_TIMEPOINTS): [.167,.233,.200,.267,.133,.200,.167,.233] -- only the
    # 8-timepoint bout (0.267) clears 0.25 (25%).
    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", long_contact_threshold=25.0, long_contact_unit="percent", verbose=False,
        **_common_kwargs(),
    )
    long_csv = pd.read_csv(result["long_contact_csv_path"])
    round_row = long_csv[long_csv["target_class"] == "round"].iloc[0]
    assert round_row["n_total"] == 8
    assert round_row["n_long_contact"] == 1
    assert round_row["n_short_contact"] == 7
    assert round_row["long_contact_threshold"] == pytest.approx(25.0)


def test_paired_mode_multiple_pairing_cols_composite_key(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    # exp_nr is constant across samples here, so pairing on [sample_name, exp_nr] behaves
    # exactly like pairing on sample_name alone -- this just exercises the composite-key path.
    adata_tracks.obs["exp_nr"] = "run1"

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="paired", pairing_col=["sample_name", "exp_nr"], minutes_per_frame=None, verbose=False,
        **_common_kwargs(),
    )
    assert result["pairing_col"] == "sample_name + exp_nr"

    csv = pd.read_csv(result["csv_path"])
    assert (csv["pairing_col"] == "sample_name + exp_nr").all()
    round_vs_elong = _find_row(csv, "round", "elongated")
    assert round_vs_elong["n_a"] == 4
    assert round_vs_elong["n_b"] == 4


def test_connected_page_paired_csv_has_per_sample_rows(tmp_path):
    """The "connected by sample" companion page always pairs by ``sample_col`` (default
    "sample_name"), independent of ``test_mode`` -- so even in welch mode, every sample that
    touched both classes gets a row in the paired CSV."""
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", minutes_per_frame=2.0, verbose=False,
        **_common_kwargs(),
    )
    assert Path(result["paired_csv_path"]).exists()
    paired = pd.read_csv(result["paired_csv_path"])
    assert set(paired["page_group"]) == {"(all)"}

    round_vs_elong = paired[
        ((paired["group_a"] == "round") & (paired["group_b"] == "elongated"))
        | ((paired["group_a"] == "elongated") & (paired["group_b"] == "round"))
    ]
    # all 4 samples touched both round and elongated.
    assert set(round_vs_elong["sample_name"]) == set(_SAMPLES)
    assert len(round_vs_elong) == 4

    round_vs_plastic = paired[
        ((paired["group_a"] == "round") & (paired["group_b"] == "plastic"))
        | ((paired["group_a"] == "plastic") & (paired["group_b"] == "round"))
    ]
    # s4 never touched "plastic" -> dropped from the paired join, same as the paired t-test.
    assert set(round_vs_plastic["sample_name"]) == {"s1", "s2", "s3"}

    row = round_vs_elong[round_vs_elong["sample_name"] == "s1"].iloc[0]
    lo, hi = (row["mean_a_timepoints"], row["mean_b_timepoints"])
    # s1 round bouts [5,7] -> mean 6; elongated [9,11] -> mean 10 (order depends on a/b side).
    assert {round(lo), round(hi)} == {6, 10}
    assert row["mean_a_minutes"] == pytest.approx(row["mean_a_timepoints"] * 2.0)
    assert row["mean_b_minutes"] == pytest.approx(row["mean_b_timepoints"] * 2.0)


def test_duration_comparison_pages_include_connected_companion(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", comparisons_per_page=6, verbose=False,
        **_common_kwargs(),
    )
    # 6 comparisons all fit on one page -> 1 box page + 1 connected companion page.
    assert result["n_pages"] == 2


def test_long_contact_percentage_pages_and_per_sample_csv(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    long_contact_threshold = 100.0 * 7.0 / _N_TIMEPOINTS
    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        test_mode="welch", long_contact_threshold=long_contact_threshold, long_contact_unit="percent",
        verbose=False, **_common_kwargs(),
    )
    assert Path(result["long_contact_per_sample_csv_path"]).exists()
    per_sample = pd.read_csv(result["long_contact_per_sample_csv_path"])
    assert set(per_sample["page_group"]) == {"(all)"}
    assert set(per_sample.columns) == {"page_group", "sample_name", "target_class", "pct_long_contact"}

    # matches the direct _compute_per_sample_long_contact_pct check in
    # test_long_contact_per_sample_pct_for_boxplot: 1 of 2 round tracks "long" per sample except
    # s3 (0 of 2).
    round_pct = per_sample[per_sample["target_class"] == "round"].set_index("sample_name")["pct_long_contact"]
    assert round_pct.to_dict() == {"s1": 50.0, "s2": 50.0, "s3": 0.0, "s4": 50.0}

    # 1 box page + 1 connected companion page for the long-contact section, on top of the
    # duration-comparison section's own 2 pages (1 box + 1 connected, since all 6 comparisons fit
    # on the default comparisons_per_page).
    assert result["n_pages"] == 4


def test_fewer_than_two_classes_writes_placeholder(tmp_path):
    # Restrict the target-class lookup to just "round" so only one touched class remains.
    adata_target = _build_target_track_adata()
    adata_target = adata_target[adata_target.obs["ClusterID"] == "round"].copy()
    lookup = build_target_class_lookup_from_track_adata(adata_target, class_col="ClusterID")

    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_duration_comparison(
        adata_tracks, df_timepoints, tmp_path,
        contact_col="macro_contact", min_bout_length=_MIN_BOUT_LENGTH,
        target_class_lookup=lookup, touching_col=touching_column_name("macro"),
        time_varying=False, target_cell_type_label="macro",
        test_mode="welch",
    )
    assert result["n_comparisons"] == 0
    assert result["n_pages"] == 1
    assert Path(result["pdf_path"]).exists()
    csv = pd.read_csv(result["csv_path"])
    assert len(csv) == 0

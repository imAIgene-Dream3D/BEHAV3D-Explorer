"""
Killing by State report: per-state killing efficiency from Active Killing's values, with solo
and full-window filters, an optional window/threshold override, and no statistics.

Runs under pytest. Requires the `behav3d` conda env.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from behav3d.analysis.behavior.state.killing_by_state import (
    KillingSettings,
    build_killing_timepoints,
    count_other_cells_on_targets,
    filter_counts,
    parse_extracted_window,
    read_state_labels,
    recompute_killing_efficiency,
    resolve_killing_features_csv,
    resolve_killing_settings,
    summarize_killing_by_state,
)
from behav3d.analysis.behavior.state.visualization.plots.killing_by_state_report import (
    BAR_KEY,
    _fig_two_bar_plots,
    _settings_subtitle,
    save_state_killing_report,
)
from behav3d.features.advanced_timepoint_features import (
    analyze_active_killing_per_timepoint,
    identify_contact_events_global,
)

CT = "tcells"
ACTIVE_KILLING_CFG = {
    "observation_window": 5,
    "death_signal_column": "nr_dead_mask_pixels",
    "persistent_death_signal": True,
    "killing_threshold_multiplier": 1.5,
    "use_absolute_threshold": True,
    "absolute_killing_threshold": 10.0,
    "min_contact_duration": 1,
    "contact_column": "contact",
    "target_types": ["organoid"],
}


def _settings(window=5, **kwargs):
    defaults = dict(
        window=window, threshold_mode="absolute", threshold=10.0, target_types=("organoid",),
        death_signal_column="nr_dead_mask_pixels", min_contact_duration=1,
    )
    defaults.update(kwargs)
    return KillingSettings(**defaults)


def _row(track, t, target=None, sample="S1", contact=None, ke=0.0):
    return {
        "sample_name": sample, "TrackID": track, "position_t": t,
        "touching_organoids": None if target is None else str(target),
        "organoid_contact": bool(target is not None) if contact is None else contact,
        "killing_efficiency": ke, "is_active_killing": ke > 1,
        "death_signal_increase_5tp": ke * 10,
    }


def _adata(rows, state_col="full_behavioral_state"):
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    adata.uns["classification"] = {
        "state_order": {state_col: ["Static engager", "Protrusive engager"]},
        "state_colors": {state_col: {"Static engager": "#b05a89", "Protrusive engager": "#ff0500"}},
    }
    return adata


def _normalized(df):
    out = df.copy()
    out["sample_name"] = out["sample_name"].astype(str)
    out["TrackID"] = out["TrackID"].astype(str)
    out["position_t"] = out["position_t"].astype(float)
    return out


def _count(rows, table, window=5, targets=("organoid",)):
    return count_other_cells_on_targets(_normalized(rows), _normalized(table), list(targets), window)


# ---- solo logic ------------------------------------------------------------------------------


def test_other_cell_inside_window_counts_and_outside_does_not():
    table = pd.DataFrame([
        _row(1, 0, target=7),
        _row(2, 5, target=7),   # last timepoint of the window starting at t = 0
        _row(3, 6, target=7),   # just outside it
    ])
    query = pd.DataFrame([_row(1, 0, target=7)])
    assert _count(query, table).tolist() == [1]
    assert _count(query, table, window=6).tolist() == [2]
    assert _count(query, table, window=4).tolist() == [0]


def test_other_cell_before_the_timepoint_does_not_count():
    table = pd.DataFrame([_row(2, 0, target=7), _row(1, 1, target=7)])
    assert _count(pd.DataFrame([_row(1, 1, target=7)]), table).tolist() == [0]


def test_same_cell_is_never_its_own_neighbour_and_other_cell_counts_once():
    table = pd.DataFrame([
        _row(1, 0, target=7), _row(1, 1, target=7), _row(2, 1, target=7), _row(2, 2, target=7),
    ])
    assert _count(pd.DataFrame([_row(1, 0, target=7)]), table).tolist() == [1]


def test_any_touched_target_counts():
    table = pd.DataFrame([_row(1, 0, target="7, 8"), _row(2, 2, target=8), _row(3, 2, target=9)])
    assert _count(pd.DataFrame([_row(1, 0, target="7, 8")]), table).tolist() == [1]


def test_no_recorded_target_is_nan_not_solo():
    table = pd.DataFrame([_row(1, 0, target=7), _row(1, 1, target=None, contact=True)])
    result = _count(pd.DataFrame([_row(1, 1, target=None, contact=True)]), table)
    assert np.isnan(result[0])


def test_other_samples_do_not_count():
    table = pd.DataFrame([_row(1, 0, target=7), _row(2, 1, target=7, sample="S2")])
    assert _count(pd.DataFrame([_row(1, 0, target=7)]), table).tolist() == [0]


def test_same_id_on_another_target_type_is_another_target():
    table = pd.DataFrame([
        {"sample_name": "S1", "TrackID": 1, "position_t": 0, "touching_organoids": "7", "touching_tumors": None},
        {"sample_name": "S1", "TrackID": 2, "position_t": 1, "touching_organoids": None, "touching_tumors": "7"},
    ])
    query = table.iloc[[0]]
    assert _count(query, table, targets=("organoid", "tumor")).tolist() == [0]


# ---- timepoint table -------------------------------------------------------------------------


def _states_and_table():
    """Track 1: solo on target 7. Track 2: on target 8 with track 3 arriving later. Track 4: never
    in contact. Track 5: one contact timepoint in a mostly non-contact 'Dead' state."""
    table, states = [], []

    def add(track, t, target, state, ke=0.0):
        table.append(_row(track, t, target=target, ke=ke))
        states.append({"sample_name": "S1", "TrackID": track, "position_t": t,
                       "full_behavioral_state": state, "well": "A"})

    for t in range(10):
        add(1, t, 7, "Protrusive engager", ke=1.0)
    for t in range(10):
        add(2, t, 8, "Static engager", ke=0.5)
    add(3, 9, 8, "Static engager")
    for t in range(10):
        add(4, t, None, "Round jittering")
    add(5, 0, 9, "Dead", ke=0.25)
    for t in range(1, 10):
        add(5, t, None, "Dead")
    return _adata(states), pd.DataFrame(table)


def _build(**kwargs):
    adata, table = _states_and_table()
    defaults = dict(state_col="full_behavioral_state", settings=_settings(extracted_window=5), video_ends={"S1": 9})
    defaults.update(kwargs)
    return build_killing_timepoints(adata, table, **defaults)


def test_automatic_states_are_those_with_any_contact_timepoint():
    tp, observed = _build()
    assert set(observed) == {"Protrusive engager", "Static engager", "Dead"}
    assert "Round jittering" not in set(tp["state"])


def test_only_contact_timepoints_enter_even_for_a_mixed_state():
    tp, _ = _build(solo_only=False, full_windows_only=False)
    dead = tp[tp["state"] == "Dead"]
    assert dead["position_t"].tolist() == [0]
    assert dead["used"].tolist() == [True]


def test_manual_state_selection_leaves_out_other_states():
    tp, _ = _build(states=["Protrusive engager"])
    assert set(tp["state"]) == {"Protrusive engager"}


def test_solo_and_full_window_flags_and_toggles():
    tp, _ = _build(states=["Protrusive engager", "Static engager"])
    # Track 3 touches target 8 at t = 9, so track 2 is solo only while t + 5 < 9.
    assert tp[tp["TrackID"] == "2"].sort_values("position_t")["solo"].tolist() == [True] * 4 + [False] * 6
    assert tp[tp["TrackID"] == "1"]["solo"].all()
    # Video ends at 9, so full windows stop at t = 4.
    assert tp[tp["TrackID"] == "1"].sort_values("position_t")["full_window"].tolist() == [True] * 5 + [False] * 5

    counts = filter_counts(tp)
    assert counts["used"] == 5 + 4  # track 1: t 0-4, track 2: t 0-3

    no_solo, _ = _build(states=["Static engager"], solo_only=False)
    assert no_solo[no_solo["TrackID"] == "2"]["used"].sum() == 5
    no_window, _ = _build(states=["Static engager"], full_windows_only=False)
    assert no_window[no_window["TrackID"] == "2"]["used"].sum() == 4
    neither, _ = _build(states=["Static engager"], solo_only=False, full_windows_only=False)
    assert neither["used"].all()


def test_no_recorded_target_is_dropped_only_by_the_solo_filter():
    adata = _adata([{"sample_name": "S1", "TrackID": 1, "position_t": 0, "full_behavioral_state": "Static engager"}])
    table = pd.DataFrame([_row(1, 0, target=None, contact=True)])
    kwargs = dict(state_col="full_behavioral_state", settings=_settings(extracted_window=5), video_ends={"S1": 9})
    with_solo, _ = build_killing_timepoints(adata, table, **kwargs)
    assert with_solo["used"].tolist() == [False]
    without_solo, _ = build_killing_timepoints(adata, table, solo_only=False, **kwargs)
    assert without_solo["used"].tolist() == [True]


def test_csv_table_has_ids_for_a_downstream_model():
    tp, _ = _build()
    for col in ("cell_id", "cell_state_id", "used", "solo", "full_window", "n_other_cells", "video_end"):
        assert col in tp.columns
    assert tp.loc[tp["TrackID"] == "1", "cell_state_id"].iloc[0] == "S1_1_Protrusive engager"


def test_mismatched_tables_raise_instead_of_returning_nothing():
    adata, table = _states_and_table()
    table["sample_name"] = "other"
    with pytest.raises(ValueError, match="no longer belong together"):
        build_killing_timepoints(
            adata, table, state_col="full_behavioral_state", settings=_settings(), video_ends={"S1": 9},
        )


# ---- settings and warning text ---------------------------------------------------------------


def _extracted_output(tmp_path, threshold=10.0, varying=False):
    folder = tmp_path / "analysis" / CT / "active_killing" / "organoid"
    folder.mkdir(parents=True, exist_ok=True)
    values = [threshold, threshold + 1] if varying else [threshold, threshold]
    pd.DataFrame({"killing_threshold_used": values}).to_csv(
        folder / f"active_killing_per_timepoint_{CT}.csv", index=False
    )


def test_defaults_match_the_tab_and_show_no_difference(tmp_path):
    _extracted_output(tmp_path)
    settings = resolve_killing_settings(ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_5tp"])
    assert (settings.window, settings.threshold_mode, settings.threshold) == (5, "absolute", 10.0)
    assert settings.target_types == ("organoid",)
    assert settings.differences == ()
    assert not settings.needs_recompute


def test_window_override_is_reported_against_extracted(tmp_path):
    _extracted_output(tmp_path)
    settings = resolve_killing_settings(
        ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_5tp"], window_override=30,
    )
    assert settings.window == 30
    assert settings.differences == ("Window 30 tp differs from extracted features (5 tp).",)
    assert settings.needs_recompute


def test_threshold_override_is_reported_against_extracted(tmp_path):
    _extracted_output(tmp_path)
    settings = resolve_killing_settings(
        ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_5tp"], threshold_override=50.0,
    )
    assert settings.differences == ("Threshold 50 differs from extracted features (10).",)


def test_override_back_to_the_extracted_values_clears_the_difference(tmp_path):
    _extracted_output(tmp_path)
    settings = resolve_killing_settings(
        ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_5tp"],
        window_override=5, threshold_override=10.0,
    )
    assert settings.differences == ()


def test_tab_out_of_step_with_the_extracted_features_is_a_difference(tmp_path):
    _extracted_output(tmp_path, threshold=100.0)
    settings = resolve_killing_settings(ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_20tp"])
    assert len(settings.differences) == 2
    assert "Window 5 tp differs" in settings.differences[0]
    assert "Threshold 10 differs" in settings.differences[1]


def test_multiplier_mode_uses_the_tab_as_baseline(tmp_path):
    _extracted_output(tmp_path, varying=True)
    cfg = {**ACTIVE_KILLING_CFG, "use_absolute_threshold": False, "absolute_killing_threshold": None}
    settings = resolve_killing_settings(cfg, tmp_path, CT, ["death_signal_increase_5tp"])
    assert (settings.threshold_mode, settings.threshold) == ("multiplier", 1.5)
    assert settings.differences == ()
    changed = resolve_killing_settings(cfg, tmp_path, CT, ["death_signal_increase_5tp"], threshold_override=2.0)
    assert changed.differences == ("Threshold 2x differs from extracted features (1.5x).",)


def test_tab_mode_differing_from_extracted_mode_is_a_difference(tmp_path):
    _extracted_output(tmp_path, varying=True)
    settings = resolve_killing_settings(ACTIVE_KILLING_CFG, tmp_path, CT, ["death_signal_increase_5tp"])
    assert len(settings.differences) == 1
    assert "absolute" in settings.differences[0] and "multiplier" in settings.differences[0]


def test_parse_extracted_window():
    assert parse_extracted_window(["a", "death_signal_increase_20tp"]) == 20
    assert parse_extracted_window(["death_signal_increase"]) is None


def test_features_table_without_killing_columns_explains_what_to_run(tmp_path):
    csv = tmp_path / "features.csv"
    pd.DataFrame({"TrackID": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Run Active Killing first, then Filtering"):
        resolve_killing_features_csv(tmp_path, CT, csv)


# ---- recompute -------------------------------------------------------------------------------


def _write_raw_tracks(tmp_path):
    """Target 7's dead mask grows by 2 px per frame; tracks 1 and 2 touch it."""
    immune = [
        {"sample_name": "S1", "TrackID": track, "position_t": t,
         "organoid_contact": 1, "touching_organoids": "7"}
        for track in (1, 2) for t in range(0, 10)
    ]
    target = [
        {"sample_name": "S1", "TrackID": 7, "position_t": t, "nr_dead_mask_pixels": 2 * t}
        for t in range(0, 30)
    ]
    for ct, rows in ((CT, immune), ("organoid", target)):
        folder = tmp_path / "analysis" / ct / "track_features"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(folder / f"BEHAV3D_{ct}_combined_track_features.csv", index=False)
    return pd.DataFrame(immune), pd.DataFrame(target)


def test_recompute_at_the_same_settings_equals_active_killing(tmp_path):
    immune, target = _write_raw_tracks(tmp_path)
    settings = _settings(window=5, threshold=4.0)
    events = identify_contact_events_global(immune, ["organoid"], min_contact_duration=1, contact_column="contact")
    stored = analyze_active_killing_per_timepoint(
        immune, target, events, observation_window=5, death_signal_column="nr_dead_mask_pixels",
        absolute_killing_threshold=4.0, persistent_death_signal=True,
    )
    fresh = recompute_killing_efficiency(tmp_path, CT, settings)
    merged = stored.rename(columns={"immune_track_id": "TrackID"}).merge(
        fresh, on=["sample_name", "TrackID", "position_t"], suffixes=("_s", "_r"),
    )
    assert len(merged) == len(stored) == 20
    np.testing.assert_allclose(merged["killing_efficiency_s"], merged["killing_efficiency_r"])


def test_recompute_follows_window_and_threshold_and_writes_nothing(tmp_path):
    _write_raw_tracks(tmp_path)
    base = recompute_killing_efficiency(tmp_path, CT, _settings(window=5, threshold=4.0))
    # Increase over 5 frames is 10 px: efficiency 10 / 4.
    np.testing.assert_allclose(base["killing_efficiency"], 2.5, rtol=1e-6)
    wider = recompute_killing_efficiency(tmp_path, CT, _settings(window=10, threshold=4.0))
    np.testing.assert_allclose(wider["killing_efficiency"], 5.0, rtol=1e-6)
    stricter = recompute_killing_efficiency(tmp_path, CT, _settings(window=5, threshold=20.0))
    np.testing.assert_allclose(stricter["killing_efficiency"], 0.5, rtol=1e-6)
    assert not stricter["is_active_killing"].any()
    assert not (tmp_path / "analysis" / CT / "active_killing").exists()


def test_recomputed_values_replace_stored_ones_and_missing_contacts_get_zero():
    adata, table = _states_and_table()
    recomputed = pd.DataFrame({
        "sample_name": ["S1"], "TrackID": [1], "position_t": [0],
        "killing_efficiency": [9.0], "death_signal_increase": [90.0], "is_active_killing": [True],
    })
    tp, _ = build_killing_timepoints(
        adata, table, state_col="full_behavioral_state", settings=_settings(), video_ends={"S1": 9},
        states=["Protrusive engager"], recomputed=recomputed,
    )
    tp = tp.sort_values("position_t")
    assert tp["killing_efficiency"].tolist() == [9.0] + [0.0] * 9
    assert tp["is_active_killing"].tolist() == [True] + [False] * 9


# ---- summaries -------------------------------------------------------------------------------


def _used_timepoints():
    """Cell a: 3 timepoints of KE 1. Cell b: 1 timepoint of KE 0. Means differ by weighting."""
    rows = [("a", "S1", 1.0)] * 3 + [("b", "S2", 0.0)]
    return pd.DataFrame([
        {"sample_name": sample, "TrackID": cell, "position_t": i, "state": "X",
         "killing_efficiency": ke, "is_active_killing": ke > 0, "used": True,
         "cell_id": f"{sample}_{cell}", "cell_state_id": f"{sample}_{cell}_X"}
        for i, (cell, sample, ke) in enumerate(rows)
    ])


def test_cell_mean_counts_each_cell_once_and_timepoint_mean_each_timepoint():
    _, _, summary = summarize_killing_by_state(_used_timepoints(), ["X"])
    row = summary.iloc[0]
    assert row["cell_mean"] == pytest.approx(0.5)
    assert row["timepoint_mean"] == pytest.approx(0.75)
    assert (row["n_cells"], row["n_timepoints"], row["n_images"]) == (2, 4, 2)
    assert row["frac_active_timepoints"] == pytest.approx(0.75)


def test_state_without_used_timepoints_keeps_a_row_with_zero_counts():
    _, _, summary = summarize_killing_by_state(_used_timepoints(), ["X", "Y"])
    empty = summary[summary["state"] == "Y"].iloc[0]
    assert (empty["n_cells"], empty["n_timepoints"]) == (0, 0)
    assert pd.isna(empty["cell_mean"])


def test_unused_rows_stay_out_of_the_summaries():
    tp = _used_timepoints()
    tp.loc[tp["TrackID"] == "b", "used"] = False
    cell_state, _, summary = summarize_killing_by_state(tp, ["X"])
    assert summary.iloc[0]["cell_mean"] == pytest.approx(1.0)
    assert len(cell_state) == 1


# ---- report ----------------------------------------------------------------------------------


def _write_target_table(root):
    folder = root / "analysis" / "organoid" / "track_features"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_name": ["S1"], "TrackID": [7], "position_t": [9], "nr_dead_mask_pixels": [0],
    }).to_csv(folder / "BEHAV3D_organoid_combined_track_features.csv", index=False)


def _pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page[^s]", Path(path).read_bytes()))


def _report(tmp_path, **kwargs):
    _write_target_table(tmp_path)
    adata, table = _states_and_table()
    defaults = dict(
        output_dir=tmp_path, cell_type=CT, state_col="full_behavioral_state",
        settings=_settings(extracted_window=5),
    )
    defaults.update(kwargs)
    return save_state_killing_report(adata, table, tmp_path / "out", **defaults)


def test_report_writes_csvs_and_a_single_pdf_without_statistics(tmp_path):
    result = _report(tmp_path)
    folder = Path(result["report_dir"])
    assert folder == tmp_path / "out" / "killing_by_state"
    assert [f.name for f in folder.rglob("*.pdf")] == ["killing_by_state.pdf"]
    assert list(folder.rglob("*.png")) == []

    header = set()
    for csv in (folder / "csv").glob("*.csv"):
        header |= {c.lower() for c in pd.read_csv(csv, nrows=0).columns}
    assert len(list((folder / "csv").glob("*.csv"))) == 4
    assert not header & {"p", "pvalue", "p_value", "stars", "t", "f", "ci_low", "ci_high"}

    assert sorted(f.name for f in folder.iterdir() if f.is_file()) == ["killing_by_state.pdf"]


def test_rerun_replaces_the_previous_output_in_the_same_folder(tmp_path):
    first = _report(tmp_path, settings=_settings(extracted_window=5))
    keep = Path(first["report_dir"]) / "my_own_notes.txt"
    keep.write_text("mine")
    second = _report(tmp_path, solo_only=False, full_windows_only=False, states=["Static engager"])
    folder = Path(second["report_dir"])
    assert folder == Path(first["report_dir"])
    assert set(pd.read_csv(second["timepoints_csv"])["state"]) == {"Static engager"}
    assert [f.name for f in folder.rglob("*.pdf")] == ["killing_by_state.pdf"]
    assert pd.read_csv(second["timepoints_csv"])["used"].all()
    assert keep.read_text() == "mine"


def test_report_orders_states_as_the_reversed_stored_order(tmp_path):
    result = _report(tmp_path, states=["Static engager", "Protrusive engager"])
    assert result["state_order"] == ["Protrusive engager", "Static engager"]


def test_report_says_when_it_differs_from_extracted_features(tmp_path):
    _write_raw_tracks(tmp_path)
    settings = _settings(window=3, extracted_window=5)
    adata, table = _states_and_table()
    result = save_state_killing_report(
        adata, table, tmp_path / "out", output_dir=tmp_path, cell_type=CT,
        state_col="full_behavioral_state", settings=settings,
    )
    assert result["differences"] == ["Window 3 tp differs from extracted features (5 tp)."]
    assert "Recomputed: Window 3 tp differs" in _settings_subtitle(settings, True, True)
    assert not (tmp_path / "analysis" / CT / "active_killing").exists()


def test_bar_plots_show_means_with_sem_and_handle_sparse_states():
    cells = pd.DataFrame({
        "state": ["A", "A", "A", "B"],
        "mean_killing_efficiency": [1.0, 2.0, 6.0, 4.0],
    })
    timepoints = pd.DataFrame({
        "state": ["A", "A", "B", "B"],
        "killing_efficiency": [0.0, 10.0, 2.0, 4.0],
    })
    order = ["A", "B", "Empty"]
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "Empty": "#2ca02c"}
    fig = _fig_two_bar_plots(cells, timepoints, order, colors, "subtitle")
    try:
        cell_ax, timepoint_ax = fig.axes
        heights = [p.get_height() for p in cell_ax.patches]
        assert heights[:2] == pytest.approx([3.0, 4.0])
        assert np.isnan(heights[2])
        assert [p.get_height() for p in timepoint_ax.patches][:2] == pytest.approx([5.0, 3.0])

        error_bars = cell_ax.containers[-1].errorbar
        segments = error_bars.lines[2][0].get_segments()
        low, high = segments[0][:, 1]
        assert (high - low) / 2 == pytest.approx(pd.Series([1.0, 2.0, 6.0]).sem())
        assert segments[1].size == 0  # one value has no SEM, so no error bar is drawn
        assert [t.get_text().splitlines()[-1] for t in cell_ax.get_xticklabels()] == [
            "n = 3", "n = 1", "n = 0",
        ]
    finally:
        plt.close(fig)


def test_subtitle_explains_the_bars():
    assert BAR_KEY in _settings_subtitle(_settings(), True, True)


def test_report_repeats_per_group_and_adds_group_columns(tmp_path):
    _write_target_table(tmp_path)
    adata, table = _states_and_table()
    obs = adata.obs.copy()
    obs["well"] = np.where(obs["TrackID"].isin([1]), "A", "B")
    adata.obs = obs
    result = save_state_killing_report(
        adata, table, tmp_path / "out", output_dir=tmp_path, cell_type=CT,
        state_col="full_behavioral_state", settings=_settings(extracted_window=5),
        extra_group_cols=["well"],
    )
    folder = Path(result["report_dir"])
    assert [f.name for f in folder.rglob("*.pdf")] == ["killing_by_state.pdf"]
    assert list(folder.rglob("*.png")) == []
    assert _pdf_pages(folder / "killing_by_state.pdf") == 2
    summary = pd.read_csv(result["state_summary_csv"])
    assert set(summary["well"]) == {"A", "B"}
    assert "well" in pd.read_csv(result["timepoints_csv"], nrows=1).columns


def test_selected_state_without_usable_timepoints_does_not_break_the_plot(tmp_path):
    result = _report(tmp_path, states=["Protrusive engager", "Round jittering"])
    assert result["counts"]["used"] > 0
    assert _pdf_pages(result["pdf_path"]) == 1
    summary = pd.read_csv(result["state_summary_csv"]).set_index("state")
    assert summary.loc["Round jittering", "n_cells"] == 0


def test_report_without_usable_timepoints_writes_a_no_data_page(tmp_path):
    _write_target_table(tmp_path)
    result = _report(tmp_path, states=["Round jittering"])
    assert result["counts"]["used"] == 0
    assert Path(result["pdf_path"]).exists()
    assert Path(result["state_summary_csv"]).exists()


def test_read_state_labels_from_categorical_and_plain_columns(tmp_path):
    adata, _ = _states_and_table()
    adata.obs["full_behavioral_state"] = adata.obs["full_behavioral_state"].astype("category")
    path = tmp_path / "states.h5ad"
    adata.write_h5ad(path)
    assert set(read_state_labels(path, "full_behavioral_state")) == {
        "Protrusive engager", "Static engager", "Round jittering", "Dead",
    }
    assert read_state_labels(path, "missing") == []


# ---- GUI -------------------------------------------------------------------------------------

_APP = None


def _app():
    global _APP
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _state_tab(params, out_dir, cell_type=CT):
    import behav3d.napari._analysis  # noqa: F401  (imports _single_cell in the working order)
    from behav3d.napari._single_cell import StateClassificationSubTab
    loader = SimpleNamespace(output_dir=str(out_dir), behav3d_parameters=params, metadata=None)
    return StateClassificationSubTab(viewer=None, metadata_loader=loader, cell_type_getter=lambda: cell_type)


def _gui_project(tmp_path):
    """A project where Active Killing ran with window 5 and absolute threshold 10."""
    _extracted_output(tmp_path)
    adata, table = _states_and_table()
    states_dir = tmp_path / "analysis" / CT / "behavioral_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(states_dir / f"BEHAV3D_{CT}_behavioral_states.h5ad")
    features = tmp_path / "analysis" / CT / "track_features"
    features.mkdir(parents=True, exist_ok=True)
    table.to_csv(features / f"BEHAV3D_{CT}_combined_track_features_filtered.csv", index=False)
    return {"active_killing": dict(ACTIVE_KILLING_CFG)}


def test_gui_defaults_follow_the_tab_and_warning_stays_hidden(tmp_path):
    _app()
    tab = _state_tab(_gui_project(tmp_path), tmp_path)
    tab._refresh_state_killing_source()
    assert tab.chk_state_killing_solo.isChecked() and tab.chk_state_killing_full_windows.isChecked()
    assert not tab.chk_state_killing_override.isChecked()
    assert tab.spin_state_killing_window.value() == 5
    assert tab.spin_state_killing_threshold.value() == 10.0
    assert not tab.spin_state_killing_window.isEnabled()
    assert tab.label_state_killing_threshold.text() == "Threshold (absolute):"
    assert tab.btn_state_killing_report.isEnabled()
    assert tab.label_state_killing_warning.isHidden()
    assert tab.list_state_killing_states.count() == 4
    assert tab.list_state_killing_states.selectedItems() == []


def test_gui_warning_appears_and_clears_as_values_change(tmp_path):
    _app()
    tab = _state_tab(_gui_project(tmp_path), tmp_path)
    tab._refresh_state_killing_source()
    tab.chk_state_killing_override.setChecked(True)
    assert tab.spin_state_killing_window.isEnabled()
    assert tab.label_state_killing_warning.isHidden()

    tab.spin_state_killing_window.setValue(30)
    assert not tab.label_state_killing_warning.isHidden()
    text = tab.label_state_killing_warning.text()
    assert "Window 30 tp differs from extracted features (5 tp)." in text
    assert "recomputed for this report only" in text

    tab.spin_state_killing_threshold.setValue(50.0)
    assert "Threshold 50 differs from extracted features (10)." in tab.label_state_killing_warning.text()

    tab.spin_state_killing_window.setValue(5)
    tab.spin_state_killing_threshold.setValue(10.0)
    assert tab.label_state_killing_warning.isHidden()

    # Unticking goes back to the tab's values.
    tab.spin_state_killing_window.setValue(30)
    tab.chk_state_killing_override.setChecked(False)
    assert tab.spin_state_killing_window.value() == 5
    assert tab.label_state_killing_warning.isHidden()


@pytest.mark.parametrize("death_column, threshold, shown", [
    ("nr_dead_mask_pixels", 100.0, 100.0),
    ("percentage_dead_mask", 0.0125, 0.0125),
    ("mean_dead_dye", 12.3456, 12.3456),
])
def test_gui_override_does_not_round_the_threshold_away(tmp_path, death_column, threshold, shown):
    _app()
    params = _gui_project(tmp_path)
    _extracted_output(tmp_path, threshold=threshold)
    params["active_killing"].update(death_signal_column=death_column, absolute_killing_threshold=threshold)
    tab = _state_tab(params, tmp_path)
    tab._refresh_state_killing_source()
    tab.chk_state_killing_override.setChecked(True)
    assert tab.spin_state_killing_threshold.value() == pytest.approx(shown)
    assert tab.label_state_killing_warning.isHidden()


def test_gui_multiplier_threshold_keeps_the_tab_precision(tmp_path):
    _app()
    params = _gui_project(tmp_path)
    _extracted_output(tmp_path, varying=True)
    params["active_killing"].update(
        use_absolute_threshold=False, absolute_killing_threshold=None, killing_threshold_multiplier=1.25,
    )
    tab = _state_tab(params, tmp_path)
    tab._refresh_state_killing_source()
    tab.chk_state_killing_override.setChecked(True)
    assert tab.label_state_killing_threshold.text() == "Threshold (multiplier):"
    assert tab.spin_state_killing_threshold.value() == pytest.approx(1.25)
    assert tab.label_state_killing_warning.isHidden()


def test_gui_explains_missing_killing_features_and_blocks_the_button(tmp_path):
    _app()
    params = _gui_project(tmp_path)
    csv = tmp_path / "analysis" / CT / "track_features" / f"BEHAV3D_{CT}_combined_track_features_filtered.csv"
    pd.DataFrame({"sample_name": ["S1"], "TrackID": [1], "position_t": [0]}).to_csv(csv, index=False)
    tab = _state_tab(params, tmp_path)
    tab._refresh_state_killing_source()
    assert not tab.btn_state_killing_report.isEnabled()
    assert not tab.label_state_killing_warning.isHidden()
    assert "Run Active Killing first, then Filtering" in tab.label_state_killing_warning.text()


def test_gui_settings_are_saved_and_restored(tmp_path):
    _app()
    params = _gui_project(tmp_path)
    tab = _state_tab(params, tmp_path)
    tab._refresh_state_killing_source()
    tab.chk_state_killing_solo.setChecked(False)
    tab.chk_state_killing_override.setChecked(True)
    tab.spin_state_killing_window.setValue(30)
    for i in range(tab.list_state_killing_states.count()):
        item = tab.list_state_killing_states.item(i)
        item.setSelected(item.text() == "Dead")
    tab._persist_report_settings(CT, "killing_by_state")

    saved = yaml.safe_load((tmp_path / "behav3d_parameters.yml").read_text())
    report = saved["state_reports"][CT]["killing_by_state"]
    assert report["solo_only"] is False and report["override"] is True and report["window"] == 30
    assert report["states"] == ["Dead"]

    fresh = _state_tab(saved, tmp_path)
    fresh._queue_report_settings_restore(CT)
    fresh._refresh_state_killing_source()
    fresh._apply_pending_report_settings()
    assert not fresh.chk_state_killing_solo.isChecked()
    assert fresh.chk_state_killing_override.isChecked()
    assert fresh.spin_state_killing_window.value() == 30
    assert [i.text() for i in fresh.list_state_killing_states.selectedItems()] == ["Dead"]
    assert "Window 30 tp differs" in fresh.label_state_killing_warning.text()


def test_gui_view_button_opens_the_report_when_it_exists(tmp_path):
    _app()
    tab = _state_tab(_gui_project(tmp_path), tmp_path)
    assert tab._get_view_candidates("state_killing_report", CT) == []
    folder = tmp_path / "analysis" / CT / "behavioral_states" / "killing_by_state"
    folder.mkdir(parents=True)
    pdf = folder / "killing_by_state.pdf"
    pdf.write_bytes(b"%PDF")
    assert tab._get_view_candidates("state_killing_report", CT) == [("killing_by_state", pdf)]


def test_card_is_listed_and_opens_only_its_own_settings(tmp_path):
    _app()
    from behav3d.napari.analysis_guided_copy import STATE_REPORT_PIPELINES
    assert "state_killing" in [s["id"] for s in STATE_REPORT_PIPELINES]
    tab = _state_tab(_gui_project(tmp_path), tmp_path)
    tab._focus_pipeline("state_killing")
    assert not tab.grp_state_killing.isHidden()
    assert tab.grp_state_contact.isHidden()
    tab._focus_pipeline("state_contact")
    assert tab.grp_state_killing.isHidden()

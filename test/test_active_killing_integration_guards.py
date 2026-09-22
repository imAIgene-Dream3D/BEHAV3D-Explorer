"""End-to-end guards for the Active Killing outputs that downstream steps consume.

Each test pins a place where the pipeline could otherwise report a plausible
but wrong number: credit lost when merged onto the track table, NA holes,
dropped raw columns, or credit appearing on timepoints without contact.
"""
import json

import numpy as np
import pandas as pd
import pytest

from behav3d.features.advanced_timepoint_features import (
    KILL_COLUMN_DEFAULTS,
    find_advanced_features_csv,
    run_active_killing_analysis,
)
from behav3d.io.formats.zarr import save_as_zarr

T, Z, Y, X = 12, 20, 40, 40


def _sphere(center, radius):
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius ** 2


@pytest.fixture
def project(tmp_path):
    """Organoid 5 with one death patch from t=6; effectors 1 and 2 touching it at t=3..5
    on opposite sides, effector 3 never touching anything."""
    img = tmp_path / "images" / "s1"
    img.mkdir(parents=True)
    labels = np.zeros((T, Z, Y, X), np.uint16)
    labels[:] = _sphere((10, 20, 20), 9) * 5
    dead = np.zeros_like(labels)
    dead[6:, 8:12, 18:22, 18:22] = 1
    immune = np.zeros_like(labels)
    immune[:, 8:11, 18:21, 30:33] = 1
    immune[:, 8:11, 18:21, 7:10] = 2
    immune[:, 1:4, 1:4, 1:4] = 3
    paths = {k: img / f"s1_{k}.zarr" for k in ("organoid_tracked", "tcell_tracked", "mask_dead")}
    save_as_zarr(labels, paths["organoid_tracked"])
    save_as_zarr(immune, paths["tcell_tracked"])
    save_as_zarr(dead, paths["mask_dead"])
    md = pd.DataFrame([{
        "sample_name": "s1", "pixel_distance_xy": 1.0, "pixel_distance_z": 1.0, "distance_unit": "um",
        "time_interval": 2.0, "time_unit": "m", "dead_mask_path": str(paths["mask_dead"]),
        "or_organoid_tracks_image_path": str(paths["organoid_tracked"]),
        "im_tcell_tracks_image_path": str(paths["tcell_tracked"]),
    }])
    rows = []
    for tid in (1, 2, 3):
        for t in range(T):
            rows.append({"sample_name": "s1", "TrackID": tid, "position_t": t,
                         "touching_organoids": "5" if tid in (1, 2) and 3 <= t <= 5 else "",
                         "speed": float(tid + t), "volume": 27.0})
    raw = tmp_path / "analysis" / "tcell" / "track_features" / "BEHAV3D_tcell_combined_track_features.csv"
    raw.parent.mkdir(parents=True)
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(raw, index=False)
    cand, smp, stats = run_active_killing_analysis(
        metadata=md, output_dir=tmp_path, immune_cell_type="tcell", target_cell_types=["organoid"],
        target_cell_diameter_um=6.0, causal_window_min=120.0, attribution_radius_um=15.0,
        save_results=True, output_subfolder="organoid")
    adv = pd.read_csv(find_advanced_features_csv(tmp_path, "tcell"))
    return dict(tmp=tmp_path, cand=cand, smp=smp, stats=stats, adv=adv, raw=df_raw)


def test_credit_is_conserved_through_the_track_table(project):
    assert project["stats"]["n_attributed"] == 1
    assert project["cand"]["kill_credit"].sum() == pytest.approx(1.0)
    assert project["adv"]["kill_credit"].sum() == pytest.approx(1.0)
    assert project["stats"]["total_kill_credit"] == pytest.approx(1.0)


def test_advanced_table_has_no_holes_and_keeps_every_raw_column(project):
    adv, raw = project["adv"], project["raw"]
    assert len(adv) == len(raw)
    assert set(raw.columns) <= set(adv.columns)
    for col in KILL_COLUMN_DEFAULTS:
        assert col in adv.columns
        if col not in ("targeted_cell_type", "attribution_class"):
            assert adv[col].notna().all(), col
    pd.testing.assert_series_equal(adv["speed"], raw["speed"], check_names=False)


def test_active_killing_only_on_contact_timepoints(project):
    adv = project["adv"]
    killing = adv[adv["is_active_killing"]]
    assert set(killing["TrackID"]) == {1, 2}
    assert (killing["touching_organoids"].astype(str).str.contains("5")).all()
    assert not adv.loc[adv["TrackID"] == 3, "is_active_killing"].any()
    assert (killing["targeted_track_id"] == 5).all()
    assert (adv.loc[~adv["is_active_killing"], "targeted_track_id"] == -1).all()


def test_cumulative_credit_is_monotone_per_track(project):
    adv = project["adv"].sort_values(["TrackID", "position_t"])
    for _, g in adv.groupby("TrackID"):
        assert (np.diff(g["cum_kill_credit"].to_numpy()) >= -1e-12).all()
    last = adv.groupby("TrackID")["cum_kill_credit"].last()
    assert last.sum() == pytest.approx(1.0)


def test_summary_headline_and_run_params(project):
    row = project["smp"].iloc[0]
    # Two contact events (one per effector-target pair), one attributed death.
    assert row["n_contact_events"] == 2
    assert row["conversion_rate"] == pytest.approx(0.5)
    rp = json.loads((project["tmp"] / "analysis" / "tcell" / "active_killing" / "organoid"
                     / "active_killing_run_params.json").read_text())
    assert rp["death_event_params"]["target_cell_diameter_um"] == 6.0
    assert len(rp["upstream_paths"]) == 2


def test_rerun_reuses_the_death_event_cache(project, capsys):
    tmp = project["tmp"]
    md = pd.DataFrame([{
        "sample_name": "s1", "pixel_distance_xy": 1.0, "pixel_distance_z": 1.0, "distance_unit": "um",
        "time_interval": 2.0, "time_unit": "m",
        "dead_mask_path": str(tmp / "images" / "s1" / "s1_mask_dead.zarr"),
        "or_organoid_tracks_image_path": str(tmp / "images" / "s1" / "s1_organoid_tracked.zarr"),
        "im_tcell_tracks_image_path": str(tmp / "images" / "s1" / "s1_tcell_tracked.zarr"),
    }])
    # A different attribution radius must not re-detect death events...
    run_active_killing_analysis(metadata=md, output_dir=tmp, immune_cell_type="tcell", target_cell_types=["organoid"],
                                target_cell_diameter_um=6.0, attribution_radius_um=12.0, save_results=False)
    assert "reused 1 cached death events" in capsys.readouterr().out
    # ...but a different cell diameter (a detection parameter) must.
    run_active_killing_analysis(metadata=md, output_dir=tmp, immune_cell_type="tcell", target_cell_types=["organoid"],
                                target_cell_diameter_um=5.0, save_results=False)
    assert "detecting death events" in capsys.readouterr().out


# --- state classification: binary-group constraints ------------------------------

def _constraint_df(rows):
    return pd.DataFrame(rows, columns=["any_organoid_contact", "is_active_killing"])


def test_fresh_training_learns_killing_without_pixel_contact():
    """A credited distance-contact frame need not be pixel-adjacent, so
    'is_active_killing' alone is a real label now; training must allow it."""
    from behav3d.analysis.behavior.state.utils import (
        _assign_binary_group_labels, _infer_binary_group_constraints)
    df = _constraint_df([[1, 0], [1, 1], [0, 1], [0, 0]])
    cons = _infer_binary_group_constraints(df, list(df.columns))
    labels = _assign_binary_group_labels(df, list(df.columns), cons, enforce_binary_group_constraints=True)
    assert "is_active_killing" in set(labels)


def test_pre_rework_classifier_refuses_with_retrain_message():
    from behav3d.analysis.behavior.state.utils import (
        _assign_binary_group_labels, _infer_binary_group_constraints)
    df = _constraint_df([[1, 0], [1, 1], [0, 0]])
    cons = _infer_binary_group_constraints(df, list(df.columns))
    cons.pop("active_killing_schema_version")  # what an artifact trained before the rework carries
    with pytest.raises(ValueError, match="trained before the Active Killing rework"):
        _assign_binary_group_labels(df, list(df.columns), cons, enforce_binary_group_constraints=True)


def test_missing_binary_column_is_a_clear_error_not_keyerror():
    from behav3d.analysis.behavior.state.utils import _assign_binary_group_labels
    df = pd.DataFrame({"any_organoid_contact": [1, 0]})
    with pytest.raises(ValueError, match="missing from the data"):
        _assign_binary_group_labels(df, ["any_organoid_contact", "is_active_killing"])


def test_summarize_track_features_sums_kill_credit(project):
    """Per-track summaries must count kills, not average them."""
    import shutil
    from behav3d.analysis import summarize_track_features
    tmp = project["tmp"]
    src = find_advanced_features_csv(tmp, "tcell")
    dst = tmp / "analysis" / "tcell" / "track_features" / "BEHAV3D_tcell_combined_track_features_filtered.csv"
    shutil.copy(src, dst)
    summarize_track_features(output_dir=tmp, cell_type="tcell")
    out = next((tmp / "analysis" / "tcell").rglob("*summarized*.csv"), None)
    assert out is not None
    summ = pd.read_csv(out)
    assert summ["kills_attributed"].sum() == pytest.approx(1.0)
    assert summ.loc[summ["TrackID"] == 3, "kills_attributed"].iloc[0] == 0.0


def test_figures_and_their_backing_tables_are_written(project):
    plots = project["tmp"] / "analysis" / "tcell" / "active_killing" / "organoid" / "plots"
    assert "figures_error" not in project["stats"], project["stats"].get("figures_error")
    for name in ("death_event_fate_over_time", "attributed_fraction_by_condition", "attribution_funnel",
                 "patch_depth_attributed_vs_unattributed", "killing_concentration",
                 "serial_killing_and_timing", "organoid_swimmer_top5", "engagement_dose_response"):
        assert (plots / f"{name}.png").exists(), name
    funnel = pd.read_csv(plots / "attribution_funnel.csv")
    assert funnel["n_events"].tolist() == [1, 1, 1]
    swim = pd.read_csv(plots / "organoid_swimmer_top5_events.csv")
    assert set(swim["target_track_id"]) == {5}

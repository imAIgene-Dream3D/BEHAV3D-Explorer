"""Tests for per-item run planning and the tracking resume rules.

The scenario that motivated this: samples Img001-Img004 where only Img002 has
tracking saved.  "Skip" must keep Img002 and compute the other three, a sample
with only its tracked zarr must get just its csv rebuilt, and nothing may be
re-tracked unless the plan says so.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behav3d.core.run_plan import (
    ALL_ORGANOIDS,
    Action,
    PlanItem,
    RunPlan,
    Status,
    all_organoids_status,
    merge_plan_for_multicolor,
    resolve_action,
    scan_all_organoids_items,
    scan_tracking_items,
    tracking_paths,
    tracking_status,
)
from behav3d.io.formats.zarr import save_as_zarr
from behav3d.preprocessing.tracking import prepare_tracking_sample

CT = "Tcells"
SAMPLES = ["Img001", "Img002", "Img003", "Img004"]


def _labels(t=3, value=1):
    arr = np.zeros((t, 2, 8, 8), dtype=np.uint16)
    arr[:, :, 2:5, 2:5] = value
    return arr


def _write_zarr(path: Path, t=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_as_zarr(_labels(t), path)


def _write_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TrackID": [1]}).to_csv(path, index=False)


def _metadata(root: Path, samples=SAMPLES, with_segments=True):
    rows = []
    for sn in samples:
        row = {"sample_name": sn, "pixel_distance_xy": 0.5, "pixel_distance_z": 2.0}
        if with_segments:
            seg = root / "images" / sn / f"{sn}_{CT}_segments.zarr"
            _write_zarr(seg)
            row[f"im_{CT}_segments_image_path"] = str(seg)
        rows.append(row)
    return pd.DataFrame(rows)


def _track(root: Path, sn: str, zarr=True, csv=True, t=3):
    z, c = tracking_paths(root, sn, CT)
    if zarr:
        _write_zarr(z, t=t)
    if csv:
        _write_csv(c)


# ── status probe ─────────────────────────────────────────────────────────
def test_tracking_status_matrix(tmp_path):
    md = _metadata(tmp_path)
    seg = Path(md.loc[0, f"im_{CT}_segments_image_path"])

    assert tracking_status(tmp_path, "Img001", CT, seg).status is Status.MISSING

    _track(tmp_path, "Img001")
    assert tracking_status(tmp_path, "Img001", CT, seg).status is Status.COMPLETE

    _track(tmp_path, "Img002", csv=False)
    probe = tracking_status(tmp_path, "Img002", CT, seg)
    assert probe.status is Status.PARTIAL and probe.csv_only

    _track(tmp_path, "Img003", zarr=False)
    probe = tracking_status(tmp_path, "Img003", CT, seg)
    assert probe.status is Status.PARTIAL and not probe.csv_only

    # zarr shorter than the source segments -> not trusted, not csv-only
    _track(tmp_path, "Img004", t=2)
    probe = tracking_status(tmp_path, "Img004", CT, seg)
    assert probe.status is Status.PARTIAL and not probe.csv_only


# ── RunPlan ──────────────────────────────────────────────────────────────
def _items():
    return [
        PlanItem("Img001", CT, Status.MISSING),
        PlanItem("Img002", CT, Status.COMPLETE),
        PlanItem("Img003", CT, Status.PARTIAL),
    ]


def test_skip_existing_only_skips_complete():
    plan = RunPlan.skip_existing(_items())
    assert plan.action("Img001", CT) is Action.RUN
    assert plan.action("Img002", CT) is Action.SKIP
    assert plan.action("Img003", CT) is Action.RUN
    assert plan.any_work()


def test_custom_cannot_skip_incomplete_items():
    plan = RunPlan.custom(
        _items(),
        {("Img001", CT): Action.SKIP, ("Img002", CT): Action.OVERWRITE,
         ("Img003", CT): Action.SKIP},
    )
    assert plan.action("Img001", CT) is Action.RUN
    assert plan.action("Img003", CT) is Action.RUN
    assert plan.action("Img002", CT) is Action.OVERWRITE


def test_nothing_to_do_when_everything_complete():
    items = [PlanItem(s, CT, Status.COMPLETE) for s in SAMPLES]
    assert not RunPlan.skip_existing(items).any_work()
    assert RunPlan.overwrite_all(items).any_work()


def test_unknown_item_defaults_to_run_and_resolve_action():
    plan = RunPlan.skip_existing(_items())
    assert plan.action("ImgXYZ", CT) is Action.RUN
    assert resolve_action(None, True, "Img001", CT) is Action.OVERWRITE
    assert resolve_action(None, False, "Img001", CT) is Action.RUN
    assert resolve_action(plan, True, "Img002", CT) is Action.SKIP


def test_for_cell_type_filters():
    items = _items() + [PlanItem("Img002", "other", Status.COMPLETE)]
    plan = RunPlan.skip_existing(items).for_cell_type("other")
    assert plan.action("Img002", "other") is Action.SKIP
    assert not plan.any_work()


# ── prepare_tracking_sample (the shared backend decision) ────────────────
def _prepare(root, sn, **kw):
    z, c = tracking_paths(root, sn, CT)
    seg = root / "images" / sn / f"{sn}_{CT}_segments.zarr"
    return prepare_tracking_sample(
        sample_name=sn, cell_type=CT,
        tracked_img_outpath=z, tracked_csv_outpath=c, segments_path=seg,
        element_size_x=0.5, element_size_y=0.5, element_size_z=2.0,
        log=lambda *_: None, **kw,
    )


def test_prepare_complete_sample_is_kept(tmp_path):
    _metadata(tmp_path)
    _track(tmp_path, "Img002")
    assert _prepare(tmp_path, "Img002") is False
    assert _prepare(tmp_path, "Img002", overwrite=True) is True


def test_prepare_missing_sample_runs(tmp_path):
    _metadata(tmp_path)
    assert _prepare(tmp_path, "Img001") is True


def test_prepare_zarr_only_rebuilds_csv_without_retracking(tmp_path):
    _metadata(tmp_path)
    _track(tmp_path, "Img002", csv=False)
    assert _prepare(tmp_path, "Img002") is False
    _, csv = tracking_paths(tmp_path, "Img002", CT)
    df = pd.read_csv(csv)
    assert list(df.columns)[:3] == ["TrackID", "SegmentID", "position_t"]
    assert len(df) == 3  # one labelled object per timepoint
    # element sizes were applied: pixel z centroid 0.5 -> position_z 1.0
    assert df["position_z"].iloc[0] == pytest.approx(df["pixel_position_z"].iloc[0] * 2.0)


def test_prepare_plan_overrides_overwrite_flag(tmp_path):
    _metadata(tmp_path)
    _track(tmp_path, "Img002")
    items = [PlanItem("Img002", CT, Status.COMPLETE)]
    assert _prepare(tmp_path, "Img002", overwrite=True,
                    plan=RunPlan.skip_existing(items)) is False
    assert _prepare(tmp_path, "Img002", overwrite=False,
                    plan=RunPlan.overwrite_all(items)) is True


# ── run_propagation_tracking: the reported scenario ──────────────────────
def test_only_img002_tracked_skip_computes_the_rest(tmp_path, monkeypatch):
    from behav3d.preprocessing.tracking import propagation_tracking as pt

    md = _metadata(tmp_path)
    _track(tmp_path, "Img002")
    img002_zarr, img002_csv = tracking_paths(tmp_path, "Img002", CT)
    before = img002_csv.stat().st_mtime_ns

    called = []

    def fake_propagate(segments_path, tracked_img_outpath, tracked_csv_outpath, **kw):
        called.append(Path(tracked_img_outpath).name)
        _write_zarr(Path(tracked_img_outpath))
        _write_csv(Path(tracked_csv_outpath))

    monkeypatch.setattr(pt, "propagate_tracks", fake_propagate)

    items = scan_tracking_items(md, tmp_path, [CT])
    plan = RunPlan.skip_existing(items)
    pt.run_propagation_tracking(md, str(tmp_path), CT, plan=plan)

    assert sorted(called) == [f"{sn}_{CT}_tracked.zarr" for sn in ("Img001", "Img003", "Img004")]
    assert img002_csv.stat().st_mtime_ns == before
    assert all(i.status is Status.COMPLETE for i in scan_tracking_items(md, tmp_path, [CT]))
    for sn in SAMPLES:  # metadata registered for every sample, skipped ones included
        assert md.loc[md.sample_name == sn, f"im_{CT}_tracks_csv_path"].iloc[0]


def test_custom_plan_overwrites_only_the_chosen_sample(tmp_path, monkeypatch):
    from behav3d.preprocessing.tracking import propagation_tracking as pt

    md = _metadata(tmp_path)
    for sn in SAMPLES:
        _track(tmp_path, sn)

    called = []
    monkeypatch.setattr(
        pt, "propagate_tracks",
        lambda segments_path, tracked_img_outpath, tracked_csv_outpath, **kw:
            called.append(Path(tracked_img_outpath).name),
    )
    items = scan_tracking_items(md, tmp_path, [CT])
    plan = RunPlan.custom(
        items, {it.key: (Action.OVERWRITE if it.sample == "Img003" else Action.SKIP) for it in items}
    )
    pt.run_propagation_tracking(md, str(tmp_path), CT, plan=plan)
    assert called == [f"Img003_{CT}_tracked.zarr"]


def test_legacy_overwrite_flag_still_works(tmp_path, monkeypatch):
    from behav3d.preprocessing.tracking import propagation_tracking as pt

    md = _metadata(tmp_path, samples=["Img001", "Img002"])
    _track(tmp_path, "Img002")
    called = []
    monkeypatch.setattr(
        pt, "propagate_tracks",
        lambda segments_path, tracked_img_outpath, tracked_csv_outpath, **kw:
            called.append(Path(tracked_img_outpath).name),
    )
    pt.run_propagation_tracking(md, str(tmp_path), CT, overwrite=False)
    assert called == [f"Img001_{CT}_tracked.zarr"]
    called.clear()
    pt.run_propagation_tracking(md, str(tmp_path), CT, overwrite=True)
    assert len(called) == 2


# ── all-organoids group ──────────────────────────────────────────────────
ORGS = ["orgA", "orgB"]


def _org_metadata(root, samples=("Img001", "Img002")):
    rows = []
    for sn in samples:
        row = {"sample_name": sn, "pixel_distance_xy": 0.5, "pixel_distance_z": 2.0}
        for ct in ORGS:
            seg = root / "images" / sn / f"{sn}_{ct}_segments.zarr"
            _write_zarr(seg)
            row[f"or_{ct}_segments_image_path"] = str(seg)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_group(root, sn, split_csvs=True, combined_csv=True, split_zarrs=True):
    from behav3d.core.run_plan import all_organoids_paths

    paths = all_organoids_paths(root, sn, ORGS)
    _write_zarr(paths["combined_img"])
    if combined_csv:
        paths["combined_csv"].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"TrackID": [1, 2], "position_t": [0, 0], "organoid_type": ORGS}
        ).to_csv(paths["combined_csv"], index=False)
    for ct, p in paths["split"].items():
        if split_zarrs:
            _write_zarr(p["img"])
        if split_csvs:
            _write_csv(p["csv"])
    return paths


def test_all_organoids_status(tmp_path):
    md = _org_metadata(tmp_path)
    segs = {ct: Path(md.loc[0, f"or_{ct}_segments_image_path"]) for ct in ORGS}
    assert all_organoids_status(tmp_path, "Img001", ORGS, segs).status is Status.MISSING

    _write_group(tmp_path, "Img001")
    assert all_organoids_status(tmp_path, "Img001", ORGS, segs).status is Status.COMPLETE

    _write_group(tmp_path, "Img002", split_csvs=False)
    probe = all_organoids_status(tmp_path, "Img002", ORGS, segs)
    assert probe.status is Status.PARTIAL and probe.csv_only

    items = scan_all_organoids_items(md, tmp_path, ORGS)
    assert [(i.sample, i.cell_type, i.status) for i in items] == [
        ("Img001", ALL_ORGANOIDS, Status.COMPLETE),
        ("Img002", ALL_ORGANOIDS, Status.PARTIAL),
    ]


def test_all_organoids_missing_per_type_zarr_means_full_redo(tmp_path):
    md = _org_metadata(tmp_path, samples=("Img001",))
    segs = {ct: Path(md.loc[0, f"or_{ct}_segments_image_path"]) for ct in ORGS}
    paths = _write_group(tmp_path, "Img001")
    import shutil
    shutil.rmtree(paths["split"]["orgB"]["img"])
    probe = all_organoids_status(tmp_path, "Img001", ORGS, segs)
    assert probe.status is Status.PARTIAL and not probe.csv_only


def test_all_organoids_rebuilds_only_missing_split_csvs(tmp_path, monkeypatch):
    from behav3d.preprocessing.tracking import propagation_tracking_all_organoids as ao

    md = _org_metadata(tmp_path, samples=("Img001",))
    paths = _write_group(tmp_path, "Img001", split_csvs=False)
    monkeypatch.setattr(
        ao, "_initialize_first_timepoint",
        lambda *_: pytest.fail("tracking must not re-run"),
    )
    ao.run_propagation_tracking_all_organoids(
        md, str(tmp_path), plan=RunPlan.skip_existing(
            scan_all_organoids_items(md, tmp_path, ORGS)
        ),
    )
    for ct in ORGS:
        df = pd.read_csv(paths["split"][ct]["csv"])
        assert list(df["organoid_type"]) == [ct]


def test_all_organoids_complete_group_is_skipped(tmp_path, monkeypatch):
    from behav3d.preprocessing.tracking import propagation_tracking_all_organoids as ao

    md = _org_metadata(tmp_path, samples=("Img001",))
    _write_group(tmp_path, "Img001")
    monkeypatch.setattr(
        ao, "_initialize_first_timepoint",
        lambda *_: pytest.fail("tracking must not re-run"),
    )
    ao.run_propagation_tracking_all_organoids(
        md, str(tmp_path), plan=RunPlan.skip_existing(
            scan_all_organoids_items(md, tmp_path, ORGS)
        ),
    )


# ── multicolor merge plan ────────────────────────────────────────────────
def test_merge_rebuilt_when_a_channel_is_retracked():
    channels = ["ch1", "ch2"]
    items = (
        [PlanItem("S1", c, Status.COMPLETE) for c in channels]
        + [PlanItem("S1", "grp_merged", Status.COMPLETE)]
        + [PlanItem("S2", c, Status.COMPLETE) for c in channels]
        + [PlanItem("S2", "grp_merged", Status.COMPLETE)]
    )
    plan = RunPlan.custom(
        items,
        {**{it.key: Action.SKIP for it in items}, ("S1", "ch2"): Action.OVERWRITE},
    )
    merge = merge_plan_for_multicolor(plan, channels, "grp_merged")
    assert merge.action("S1", "grp_merged") is Action.RUN      # channel changed
    assert merge.action("S2", "grp_merged") is Action.SKIP     # untouched


def test_multicolor_merge_with_plan_does_not_raise_on_existing(tmp_path):
    from behav3d.preprocessing.tracking.multicolor_tracking_processing import (
        combine_multicolor_tracked_outputs,
    )

    md = pd.DataFrame([{"sample_name": "S1"}])
    chans = ["ch1", "ch2"]
    for i, c in enumerate(chans, start=1):
        z = tmp_path / "images" / "S1" / f"S1_{c}_tracked.zarr"
        arr = np.zeros((2, 1, 4, 4), dtype=np.uint16)
        arr[:, :, i - 1, i - 1] = 1
        z.parent.mkdir(parents=True, exist_ok=True)
        save_as_zarr(arr, z)
        csv = tmp_path / "trackdata" / "S1" / c / f"S1_{c}_tracks.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"TrackID": [1, 1], "SegmentID": [1, 1], "position_t": [0, 1]}).to_csv(csv, index=False)

    first = combine_multicolor_tracked_outputs(md, tmp_path, chans, "grp_merged")
    # Without a plan an existing merged output is still a conflict ...
    with pytest.raises(FileExistsError):
        combine_multicolor_tracked_outputs(md, tmp_path, chans, "grp_merged")
    # ... with a plan the per-sample decision is honoured.
    items = [PlanItem("S1", "grp_merged", Status.COMPLETE)]
    kept = combine_multicolor_tracked_outputs(
        md, tmp_path, chans, "grp_merged", plan=RunPlan.skip_existing(items)
    )
    assert kept.loc[0, "grp_merged_tracks_csv_path"] == first.loc[0, "grp_merged_tracks_csv_path"]
    rebuilt = combine_multicolor_tracked_outputs(
        md, tmp_path, chans, "grp_merged", plan=RunPlan.overwrite_all(items)
    )
    assert Path(rebuilt.loc[0, "grp_merged_tracks_csv_path"]).exists()


# ── GUI: the reported bug (Skip must run the missing samples) ────────────
def _stub_panel(tmp_path, md):
    from types import SimpleNamespace

    from behav3d.napari._tracking import CellTypeTrackingPanel

    ran = {}
    bg = SimpleNamespace(
        is_running=lambda: False,
        run=lambda fn, **kw: ran.update(fn=fn),
    )
    stub = SimpleNamespace(
        _bg=bg, cell_type=CT, log=lambda *_: None, _persist=lambda: None,
        _get_method_key=lambda: "propagation",
        collect_runtime_params=lambda: {"method": "propagation"},
        metadata_loader=SimpleNamespace(metadata=md, output_dir=str(tmp_path)),
        btn_run=None, tab_progress_row=None, viewer=None,
        _run_tracking_for=lambda *a, **kw: ran.update(call=(a, kw)),
    )
    stub._plan_tracking_run = lambda: CellTypeTrackingPanel._plan_tracking_run(stub)
    return CellTypeTrackingPanel, stub, ran


def test_single_panel_skip_runs_missing_samples(tmp_path, monkeypatch):
    md = _metadata(tmp_path)
    _track(tmp_path, "Img002")
    panel_cls, stub, ran = _stub_panel(tmp_path, md)

    from behav3d.napari import _overwrite_prompt as op

    seen = {}

    def fake_prompt(parent, title, items, **kw):
        seen["items"] = list(items)
        return "run", RunPlan.skip_existing(items)  # what the "Skip Existing" button returns

    monkeypatch.setattr(op, "prompt_run_plan", fake_prompt)
    panel_cls._on_run_clicked(stub)

    assert "fn" in ran, "Skip must start the run, not cancel it"
    assert {i.sample: i.status for i in seen["items"]} == {
        "Img001": Status.MISSING, "Img002": Status.COMPLETE,
        "Img003": Status.MISSING, "Img004": Status.MISSING,
    }
    ran["fn"]()
    plan = ran["call"][1]["plan"]
    assert [plan.action(sn, CT) for sn in SAMPLES] == [
        Action.RUN, Action.SKIP, Action.RUN, Action.RUN,
    ]
    assert ran["call"][1]["overwrite"] is False


def test_single_panel_cancel_and_nothing_to_do(tmp_path, monkeypatch):
    md = _metadata(tmp_path)
    for sn in SAMPLES:
        _track(tmp_path, sn)
    panel_cls, stub, ran = _stub_panel(tmp_path, md)

    from behav3d.napari import _overwrite_prompt as op

    monkeypatch.setattr(op, "prompt_run_plan", lambda p, t, items, **kw: ("cancel", None))
    panel_cls._on_run_clicked(stub)
    assert "fn" not in ran

    monkeypatch.setattr(
        op, "prompt_run_plan",
        lambda p, t, items, **kw: ("run", RunPlan.skip_existing(items)),
    )
    panel_cls._on_run_clicked(stub)
    assert "fn" not in ran, "everything complete + Skip -> nothing to do"


def test_no_prompt_when_nothing_exists(tmp_path, monkeypatch):
    md = _metadata(tmp_path)
    panel_cls, stub, ran = _stub_panel(tmp_path, md)

    from behav3d.napari import _overwrite_prompt as op

    monkeypatch.setattr(
        op, "prompt_run_plan",
        lambda *a, **kw: pytest.fail("no prompt expected when no output exists"),
    )
    panel_cls._on_run_clicked(stub)
    assert "fn" in ran


# ── GUI: the Customize dialog invariants ─────────────────────────────────
def test_run_plan_dialog_locks_incomplete_rows():
    from qtpy.QtWidgets import QApplication

    from behav3d.napari._overwrite_prompt import RunPlanDialog

    app = QApplication.instance() or QApplication([])  # noqa: F841
    dlg = RunPlanDialog(None, "t", _items())
    missing, complete, partial = dlg._combos
    assert not missing.isEnabled() and missing.count() == 1          # always run
    assert complete.isEnabled() and complete.currentText().startswith("Skip")
    assert partial.isEnabled() and [partial.itemText(i) for i in range(partial.count())] == [
        "Run missing", "Overwrite",
    ]                                                                 # never Skip

    dlg._set_all(overwrite=True)
    plan = dlg.plan()
    assert [plan.action(s, CT) for s in ("Img001", "Img002", "Img003")] == [
        Action.RUN, Action.OVERWRITE, Action.OVERWRITE,
    ]
    dlg._set_all(overwrite=False)
    plan = dlg.plan()
    assert [plan.action(s, CT) for s in ("Img001", "Img002", "Img003")] == [
        Action.RUN, Action.SKIP, Action.RUN,
    ]


# ── feature extraction ───────────────────────────────────────────────────
def _write_features(root, sn, with_combined=True):
    from behav3d.core.run_plan import feature_paths

    per_sample, combined = feature_paths(root, sn, CT)
    per_sample.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_name": [sn, sn], "TrackID": [1, 2], "relative_time": [0.0, 0.1], "x": [1, 2]}
    ).to_csv(per_sample, index=False)
    if with_combined:
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_text("placeholder")


def test_scan_feature_items(tmp_path):
    from behav3d.core.run_plan import scan_feature_items

    md = pd.DataFrame({"sample_name": SAMPLES})
    _write_features(tmp_path, "Img001")                       # complete
    _write_features(tmp_path, "Img002", with_combined=False)  # combined missing
    got = {i.sample: i.status for i in scan_feature_items(md, tmp_path, [CT])}
    # Img002 wrote its per-sample file but the combined file only exists after
    # Img001 wrote it above, so a shared combined file makes both complete.
    assert got["Img001"] is Status.COMPLETE
    assert got["Img002"] is Status.COMPLETE
    assert got["Img003"] is Status.PARTIAL   # combined exists, no per-sample file
    assert got["Img004"] is Status.PARTIAL


def test_scan_feature_items_nothing_on_disk(tmp_path):
    from behav3d.core.run_plan import scan_feature_items

    md = pd.DataFrame({"sample_name": SAMPLES})
    assert {i.status for i in scan_feature_items(md, tmp_path, [CT])} == {Status.MISSING}


def test_feature_extraction_skip_keeps_existing_samples_in_combined(tmp_path):
    from behav3d.core.run_plan import feature_paths, scan_feature_items
    from behav3d.features.timepoint_features import run_feature_extraction

    md = pd.DataFrame({"sample_name": ["Img001", "Img002"]})
    _write_features(tmp_path, "Img001")
    _write_features(tmp_path, "Img002")
    plan = RunPlan.skip_existing(scan_feature_items(md, tmp_path, [CT]))
    assert not plan.any_work()

    df = run_feature_extraction(md, output_dir=str(tmp_path), cell_type=CT, plan=plan)
    assert sorted(df["sample_name"].unique()) == ["Img001", "Img002"]
    combined = pd.read_csv(feature_paths(tmp_path, "Img001", CT)[1])
    assert sorted(combined["sample_name"].unique()) == ["Img001", "Img002"]


def test_feature_panel_skip_proceeds_when_something_is_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from behav3d.napari import _overwrite_prompt as op
    from behav3d.napari._feature_extraction import CellTypeFeaturePanel

    md = pd.DataFrame({"sample_name": SAMPLES})
    _write_features(tmp_path, "Img002")
    ran = {}
    stub = SimpleNamespace(
        _bg=SimpleNamespace(is_running=lambda: False, run=lambda fn, **kw: ran.update(fn=fn)),
        cell_type=CT, log=lambda *_: None, _persist=lambda: None,
        _collect_params=lambda: {}, _threshold_changed=lambda: False,
        metadata_loader=SimpleNamespace(metadata=md, output_dir=str(tmp_path)),
        btn_run=None, tab_progress_row=None, viewer=None,
        _run_feature_extraction_for=lambda *a, **kw: ran.update(call=kw),
    )
    monkeypatch.setattr(
        op, "prompt_run_plan",
        lambda p, t, items, **kw: ("run", RunPlan.skip_existing(items)),
    )
    CellTypeFeaturePanel._on_run_clicked(stub)
    assert "fn" in ran, "Skip must run the samples that have no features yet"
    ran["fn"]()
    plan = ran["call"]["plan"]
    assert plan.action("Img002", CT) is Action.SKIP
    assert plan.action("Img001", CT) is Action.RUN


# ── segmentation scans ───────────────────────────────────────────────────
def _write_seg(root, sn, ct, journal="complete", kind="segments"):
    from behav3d.core.run_plan import segmentation_output_path
    from behav3d.preprocessing.segmentation.segment_journal import (
        journal_path, new_journal, write_journal,
    )

    path = segmentation_output_path(root, sn, ct, kind)
    _write_zarr(path, t=3)
    if journal is None:
        return path
    j = new_journal("fp", (3, 2, 8, 8), "uint16", done=set())
    for t in (range(3) if journal == "complete" else range(1)):
        j["done"].append(t)
    write_journal(journal_path(path), j)
    return path


def test_scan_segmentation_items(tmp_path):
    from behav3d.core.run_plan import scan_segmentation_items

    md = pd.DataFrame({"sample_name": ["Img001", "Img002", "Img003", "Img004"]})
    _write_seg(tmp_path, "Img001", CT)                       # journal complete
    _write_seg(tmp_path, "Img002", CT, journal="partial")    # interrupted
    _write_seg(tmp_path, "Img003", CT, journal=None)         # legacy, no journal
    got = {i.sample: i.status for i in scan_segmentation_items(md, tmp_path, [CT])}
    assert got == {
        "Img001": Status.COMPLETE,
        "Img002": Status.PARTIAL,
        "Img003": Status.COMPLETE,   # predates journalling: treated as complete, like plan_output
        "Img004": Status.MISSING,
    }


def test_segmentation_status_requires_complete_mask_when_asked(tmp_path):
    from behav3d.core.run_plan import segmentation_status

    _write_seg(tmp_path, "Img001", CT)
    assert segmentation_status(tmp_path, "Img001", CT).status is Status.COMPLETE
    probe = segmentation_status(tmp_path, "Img001", CT, include_mask=True)
    assert probe.status is Status.PARTIAL and "mask" in probe.detail
    _write_seg(tmp_path, "Img001", CT, kind="mask")
    assert segmentation_status(tmp_path, "Img001", CT, include_mask=True).status is Status.COMPLETE


def test_dead_mask_uses_its_own_file_name(tmp_path):
    from behav3d.core.run_plan import scan_segmentation_items, segmentation_output_path

    md = pd.DataFrame({"sample_name": ["Img001"]})
    assert segmentation_output_path(tmp_path, "Img001", "dead").name == "Img001_mask_dead.zarr"
    _write_seg(tmp_path, "Img001", "dead")
    items = scan_segmentation_items(md, tmp_path, [], include_dead=True)
    assert [(i.cell_type, i.status) for i in items] == [("dead", Status.COMPLETE)]


def test_scan_pixelclassifier_items(tmp_path):
    from behav3d.core.run_plan import ALL_CELL_TYPES, scan_pixelclassifier_items

    md = pd.DataFrame({"sample_name": ["S1", "S2", "S3", "S4"]})
    for ct in ("a", "b"):
        _write_seg(tmp_path, "S1", ct)           # everything present
    _write_seg(tmp_path, "S2", "a")              # one of two cell types
    _write_seg(tmp_path, "S3", "a")
    (tmp_path / "images" / "S3" / ".seg_processing").write_text("")  # interrupted
    got = {i.sample: (i.cell_type, i.status) for i in scan_pixelclassifier_items(md, tmp_path, ["a", "b"])}
    assert got == {
        "S1": (ALL_CELL_TYPES, Status.COMPLETE),
        "S2": (ALL_CELL_TYPES, Status.PARTIAL),
        "S3": (ALL_CELL_TYPES, Status.PARTIAL),
        "S4": (ALL_CELL_TYPES, Status.MISSING),
    }


# ── queue cascade ────────────────────────────────────────────────────────
def test_cascade_forces_rerun_of_stale_downstream_items():
    from behav3d.core.run_plan import ALL_SAMPLES, cascade_plans

    seg = RunPlan.custom(
        [PlanItem("S1", "T", Status.COMPLETE, step="1. Segmentation"),
         PlanItem("S2", "T", Status.COMPLETE, step="1. Segmentation")],
        {("S1", "T"): Action.OVERWRITE, ("S2", "T"): Action.SKIP},
    )
    trk = RunPlan.skip_existing(
        [PlanItem("S1", "T", Status.COMPLETE, step="2. Tracking"),
         PlanItem("S2", "T", Status.COMPLETE, step="2. Tracking")]
    )
    feat = RunPlan.skip_existing(
        [PlanItem("S1", "T", Status.COMPLETE, step="3. Features"),
         PlanItem("S2", "T", Status.COMPLETE, step="3. Features")]
    )
    filt = RunPlan.skip_existing([PlanItem(ALL_SAMPLES, "T", Status.COMPLETE, step="4. Filtering")])
    plans, notes = cascade_plans([seg, trk, feat, filt])
    assert plans[0].action("S1", "T") is Action.OVERWRITE
    assert plans[1].action("S1", "T") is Action.OVERWRITE   # segments changed
    assert plans[1].action("S2", "T") is Action.SKIP        # untouched sample stays skipped
    assert plans[2].action("S1", "T") is Action.OVERWRITE   # tracking changed (chain)
    assert plans[2].action("S2", "T") is Action.SKIP
    assert plans[3].action(ALL_SAMPLES, "T") is Action.OVERWRITE
    assert len(notes) == 3


def test_cascade_through_all_organoids_group_and_multicolor_merge():
    from behav3d.core.run_plan import cascade_plans

    members = {ALL_ORGANOIDS: ["orgA", "orgB"], "grp_merged": ["ch1", "ch2"]}
    seg = RunPlan.custom(
        [PlanItem("S1", c, Status.COMPLETE) for c in ("orgA", "ch2")],
        {("S1", "orgA"): Action.OVERWRITE, ("S1", "ch2"): Action.OVERWRITE},
    )
    trk = RunPlan.skip_existing(
        [PlanItem("S1", ALL_ORGANOIDS, Status.COMPLETE), PlanItem("S1", "grp_merged", Status.COMPLETE)]
    )
    feat = RunPlan.skip_existing([PlanItem("S1", "orgB", Status.COMPLETE)])
    plans, _ = cascade_plans([seg, trk, feat], members)
    assert plans[1].action("S1", ALL_ORGANOIDS) is Action.OVERWRITE
    assert plans[1].action("S1", "grp_merged") is Action.OVERWRITE
    assert plans[2].action("S1", "orgB") is Action.OVERWRITE   # via the group


def test_cascade_leaves_independent_items_alone():
    from behav3d.core.run_plan import cascade_plans

    seg = RunPlan.custom(
        [PlanItem("S1", "T", Status.COMPLETE)], {("S1", "T"): Action.OVERWRITE}
    )
    trk = RunPlan.skip_existing([PlanItem("S1", "other", Status.COMPLETE)])
    plans, notes = cascade_plans([seg, trk])
    assert plans[1].action("S1", "other") is Action.SKIP and not notes


# ── queue planning ───────────────────────────────────────────────────────
def _queue_stub(step_labels):
    from types import SimpleNamespace

    steps = [SimpleNamespace(display_label=lbl) for lbl in step_labels]
    return SimpleNamespace(
        _steps=steps,
        tracking_tab=SimpleNamespace(plan_members=lambda: {}),
        _legacy_prompt=lambda coarse: False,
    )


def test_queue_skip_existing_builds_per_step_plans_with_cascade(monkeypatch):
    from behav3d.napari import _overwrite_prompt as op
    from behav3d.napari._queue import ProcessingQueuePanel

    stub = _queue_stub(["Segmentation", "Tracking"])
    seg_items = [PlanItem("S1", CT, Status.PARTIAL), PlanItem("S2", CT, Status.COMPLETE)]
    trk_items = [PlanItem("S1", CT, Status.COMPLETE), PlanItem("S2", CT, Status.COMPLETE)]

    def fake_prompt(parent, title, steps, **kw):
        # only existing items are shown; "skip" plans for every step
        assert all(it.status is not Status.MISSING for _k, _l, items in steps for it in items)
        return "skip", {k: RunPlan.skip_existing(items) for k, _l, items in steps}

    monkeypatch.setattr(op, "prompt_queue_plans", fake_prompt)
    skip, plans = ProcessingQueuePanel._build_step_plans(stub, {0: seg_items, 1: trk_items}, [])

    assert skip is True
    assert plans[0].action("S1", CT) is Action.RUN and plans[0].action("S2", CT) is Action.SKIP
    # S1's segments get recomputed, so S1's complete tracking is stale and re-runs;
    # S2 is untouched and stays skipped.
    assert plans[1].action("S1", CT) is Action.OVERWRITE
    assert plans[1].action("S2", CT) is Action.SKIP


def test_queue_missing_items_are_always_run_and_overwrite_all(monkeypatch):
    from behav3d.napari import _overwrite_prompt as op
    from behav3d.napari._queue import ProcessingQueuePanel

    stub = _queue_stub(["Tracking"])
    items = [PlanItem("S1", CT, Status.MISSING), PlanItem("S2", CT, Status.COMPLETE)]
    monkeypatch.setattr(
        op, "prompt_queue_plans",
        lambda parent, title, steps, **kw: ("overwrite", {k: RunPlan.overwrite_all(i) for k, _l, i in steps}),
    )
    skip, plans = ProcessingQueuePanel._build_step_plans(stub, {0: items}, [])
    assert skip is False
    assert plans[0].action("S1", CT) is Action.RUN        # nothing there to overwrite
    assert plans[0].action("S2", CT) is Action.OVERWRITE


def test_queue_cancel_and_nothing_existing(monkeypatch):
    from behav3d.napari import _overwrite_prompt as op
    from behav3d.napari._queue import ProcessingQueuePanel

    stub = _queue_stub(["Tracking"])
    items = [PlanItem("S1", CT, Status.COMPLETE)]
    monkeypatch.setattr(op, "prompt_queue_plans", lambda *a, **kw: ("cancel", {}))
    assert ProcessingQueuePanel._build_step_plans(stub, {0: items}, []) is None

    # nothing on disk and nothing coarse -> no dialog, steps just run
    monkeypatch.setattr(
        op, "prompt_queue_plans",
        lambda *a, **kw: pytest.fail("no prompt expected when nothing exists"),
    )
    missing = [PlanItem("S1", CT, Status.MISSING)]
    assert ProcessingQueuePanel._build_step_plans(stub, {0: missing}, []) == (False, {})


def test_queue_custom_choice_is_respected(monkeypatch):
    from behav3d.napari import _overwrite_prompt as op
    from behav3d.napari._queue import ProcessingQueuePanel

    stub = _queue_stub(["Tracking"])
    items = [PlanItem(s, CT, Status.COMPLETE) for s in ("S1", "S2", "S3")]
    chosen = RunPlan.custom(items, {("S2", CT): Action.OVERWRITE,
                                    ("S1", CT): Action.SKIP, ("S3", CT): Action.SKIP})
    monkeypatch.setattr(op, "prompt_queue_plans", lambda *a, **kw: ("custom", {0: chosen}))
    skip, plans = ProcessingQueuePanel._build_step_plans(stub, {0: items}, [])
    assert [plans[0].action(s, CT) for s in ("S1", "S2", "S3")] == [
        Action.SKIP, Action.OVERWRITE, Action.SKIP,
    ]


def test_queue_dialog_groups_plans_by_step():
    from qtpy.QtWidgets import QApplication

    from behav3d.napari._overwrite_prompt import RunPlanDialog

    app = QApplication.instance() or QApplication([])  # noqa: F841
    items = [PlanItem("S1", CT, Status.COMPLETE, step="1. A"),
             PlanItem("S1", CT, Status.COMPLETE, step="2. B")]
    dlg = RunPlanDialog(None, "t", items)
    dlg._combos[1].setCurrentIndex(1)  # overwrite step B only
    by_step = dlg.plans_by_step()
    assert by_step["1. A"].action("S1", CT) is Action.SKIP
    assert by_step["2. B"].action("S1", CT) is Action.OVERWRITE

"""Tests for behav3d.features.kill_attribution.

Scenes are synthetic zarr volumes run through the real detector, then
attributed. The central property: every attributed death event carries
exactly one unit of credit in total, however many effectors were touching.
These cases all fail on the pre-rework algorithm, which gave every contacting
effector an identical, full "active killing" verdict.
"""

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behav3d.features.death_events import DeathEventParams, detect_death_events_sample
from behav3d.features.kill_attribution import (
    AttributionParams,
    attribute_death_events,
    explode_contacts,
    identify_contact_events_per_target,
    per_timepoint_credit,
    summarize_per_effector,
    summarize_per_sample,
    summarize_per_target,
)
from behav3d.io.formats.zarr import save_as_zarr

T, Z, Y, X = 14, 20, 40, 80
SPACING = (1.0, 1.0, 1.0)
DT = 2.0
DPARAMS = DeathEventParams(target_cell_diameter_um=6.0)
APARAMS = AttributionParams(causal_window_min=120.0, attribution_radius_um=15.0)
SINFO = {"s1": {"voxel_spacing": SPACING, "minutes_per_frame": DT}}


@pytest.fixture
def case_dir():
    root = Path(__file__).parent / ".tmp_kill_attribution"
    d = root / uuid.uuid4().hex
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _sphere(center, radius):
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius ** 2


class Scene:
    """An organoid (label 5), optionally a second (label 7), effector cubes, contacts."""

    def __init__(self):
        self.labels = np.zeros((T, Z, Y, X), np.uint16)
        self.labels[:] = _sphere((10, 20, 20), 9) * 5
        self.dead = np.zeros_like(self.labels)
        self.immune = np.zeros_like(self.labels)
        self.rows = []

    def second_organoid(self):
        self.labels[:] = np.where(_sphere((10, 20, 60), 9), 7, self.labels[0])
        return self

    def patch(self, t_from, corner, size=4):
        z, y, x = corner
        self.dead[t_from:, z:z + size, y:y + size, x:x + size] = 1
        return self

    def effector(self, tid, corner, frames, touching="5", size=3):
        z, y, x = corner
        for t in frames:
            self.immune[t, z:z + size, y:y + size, x:x + size] = tid
        return self

    def contact(self, tid, frames, touching="5"):
        for t in frames:
            self.rows.append({"sample_name": "s1", "TrackID": tid, "position_t": t, "touching_organoids": touching})
        return self

    def run(self, case_dir):
        lp, dp, ip = case_dir / "l.zarr", case_dir / "d.zarr", case_dir / "i.zarr"
        save_as_zarr(self.labels, lp)
        save_as_zarr(self.dead, dp)
        save_as_zarr(self.immune, ip)
        ev, ts, patches = detect_death_events_sample(
            sample_name="s1", target_type="organoid", target_tracks_path=lp, dead_mask_path=dp,
            immune_tracks_paths={"tcell": ip}, voxel_spacing=SPACING, minutes_per_frame=DT, params=DPARAMS)
        # Every effector exists at every frame (as in a real track table),
        # touching only where declared.
        tids = sorted({r["TrackID"] for r in self.rows})
        seen = {(r["TrackID"], r["position_t"]) for r in self.rows}
        rows = list(self.rows) + [
            {"sample_name": "s1", "TrackID": tid, "position_t": t, "touching_organoids": ""}
            for tid in tids for t in range(T) if (tid, t) not in seen]
        df_imm = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["sample_name", "TrackID", "position_t", "touching_organoids"])
        contacts = explode_contacts(df_imm, ["organoid"])
        df_ce, contacts = identify_contact_events_per_target(contacts)
        cand, evs = attribute_death_events(
            ev, contacts, {("s1", "organoid"): patches}, ts, immune_type="tcell",
            immune_tracks_paths={"s1": ip}, sample_info=SINFO, params=APARAMS)
        return dict(events=ev, ts=ts, cand=cand, evs=evs, imm=df_imm, ce=df_ce)


# Patch sits at x 18..21; effectors 9 µm away on either side (x 30..32 / 7..9).
PATCH = (8, 18, 18)
LEFT, RIGHT = (8, 18, 7), (8, 18, 30)


def test_two_equidistant_effectors_split_one_credit(case_dir):
    s = Scene().patch(6, PATCH)
    s.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7))
    s.effector(2, LEFT, range(3, 7)).contact(2, range(3, 7))
    r = s.run(case_dir)
    assert len(r["events"]) == 1
    assert r["cand"]["kill_credit"].sum() == pytest.approx(1.0)
    assert r["cand"]["kill_credit"].tolist() == pytest.approx([0.5, 0.5])
    assert set(r["evs"]["attribution_class"]) == {"shared"}
    assert r["cand"]["is_nearest_effector"].sum() == 1


def test_four_effectors_still_one_credit_total(case_dir):
    s = Scene().patch(6, PATCH)
    for tid, corner in enumerate([RIGHT, LEFT, (8, 30, 18), (8, 6, 18)], start=1):
        s.effector(tid, corner, range(3, 7)).contact(tid, range(3, 7))
    r = s.run(case_dir)
    assert r["cand"]["kill_credit"].sum() == pytest.approx(1.0)
    assert len(r["cand"]) == 4


def test_long_contact_with_organoid_that_never_dies_earns_nothing(case_dir):
    s = Scene()
    s.effector(1, RIGHT, range(T)).contact(1, range(T))
    r = s.run(case_dir)
    assert r["events"].empty
    assert r["cand"].empty
    assert per_timepoint_credit(r["cand"]).empty


def test_sub_threshold_speck_earns_nothing(case_dir):
    s = Scene().patch(6, PATCH, size=2)  # 8 µm³ < ~28 µm³ floor
    s.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7))
    r = s.run(case_dir)
    assert r["events"].empty and r["cand"].empty


def test_contacting_but_distant_effector_is_unattributed(case_dir):
    s = Scene().patch(6, PATCH)
    s.effector(1, (8, 34, 30), range(3, 7)).contact(1, range(3, 7))  # ~16.6 µm > 15 µm radius
    r = s.run(case_dir)
    assert len(r["events"]) == 1
    assert r["cand"].empty
    assert r["evs"]["attribution_class"].iloc[0] in ("unattributed", "unattributed_truncated")


def test_death_before_contact_earns_nothing(case_dir):
    s = Scene().patch(4, PATCH)
    s.effector(1, RIGHT, range(8, 12)).contact(1, range(8, 12))
    r = s.run(case_dir)
    assert len(r["events"]) == 1
    assert r["cand"].empty


def test_bystander_does_not_change_event_count_or_total(case_dir):
    base = Scene().patch(6, PATCH)
    base.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7))
    (case_dir / "a").mkdir()
    (case_dir / "b").mkdir()
    r1 = base.run(case_dir / "a")
    s = Scene().patch(6, PATCH)
    s.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7))
    s.effector(2, LEFT, range(3, 7)).contact(2, range(3, 7))
    r2 = s.run(case_dir / "b")
    assert len(r1["events"]) == len(r2["events"]) == 1
    assert r1["cand"]["kill_credit"].sum() == pytest.approx(r2["cand"]["kill_credit"].sum()) == pytest.approx(1.0)


def test_serial_killer_across_two_organoids(case_dir):
    s = Scene().second_organoid().patch(6, PATCH).patch(10, (8, 18, 58))
    s.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7), touching="5")
    s.effector(1, (8, 18, 70), range(7, 11)).contact(1, range(7, 11), touching="7")
    r = s.run(case_dir)
    assert len(r["events"]) == 2
    eff = summarize_per_effector(r["cand"], r["imm"], r["ce"], SINFO)
    row = eff[eff["immune_track_id"] == 1].iloc[0]
    assert row["kills_attributed"] == pytest.approx(2.0)
    assert row["n_distinct_targets_killed"] == 2
    assert bool(row["is_serial_killer"])
    assert row["inter_kill_interval_min"] == pytest.approx(4 * DT)


def test_credit_lands_on_last_contact_frame_and_is_conserved(case_dir):
    s = Scene().patch(6, PATCH)
    s.effector(1, RIGHT, range(3, 6)).contact(1, range(3, 6))
    s.effector(2, LEFT, range(2, 5)).contact(2, range(2, 5))
    r = s.run(case_dir)
    pt = per_timepoint_credit(r["cand"])
    assert pt["kill_credit"].sum() == pytest.approx(1.0)
    got = dict(zip(pt["TrackID"], pt["position_t"]))
    assert got == {1: 5, 2: 4}
    # Every credited row is a real row of the effector table (so nothing is
    # lost when credit is merged back onto the track features).
    keys = set(zip(r["imm"]["TrackID"], r["imm"]["position_t"]))
    assert all((tid, t) in keys for tid, t in zip(pt["TrackID"], pt["position_t"]))
    # A longer, closer-in-time contact earns more than an earlier, shorter one.
    c = r["cand"].set_index("immune_track_id")["kill_credit"]
    assert c.loc[1] > c.loc[2]


def test_contact_events_are_per_target_pair():
    df = pd.DataFrame({
        "sample_name": ["s1"] * 4, "TrackID": [1] * 4, "position_t": [0, 1, 2, 3],
        "touching_organoids": ["5", "5,7", "7", "7"],
    })
    contacts = explode_contacts(df, ["organoid"])
    ev, contacts = identify_contact_events_per_target(contacts)
    assert len(ev) == 2  # (1,5) over t0-1 and (1,7) over t1-3; the old any-target grouping gave 1
    assert sorted(ev["contact_duration"]) == [2, 3]
    assert contacts.loc[contacts["position_t"] == 1, "n_targets"].tolist() == [2, 2]


def test_summaries_report_conversion_and_conservation(case_dir):
    s = Scene().patch(6, PATCH)
    s.effector(1, RIGHT, range(3, 7)).contact(1, range(3, 7))
    s.effector(2, LEFT, range(3, 7)).contact(2, range(3, 7))
    s.effector(3, (8, 34, 30), range(0, 3)).contact(3, range(0, 3))
    r = s.run(case_dir)
    eff = summarize_per_effector(r["cand"], r["imm"], r["ce"], SINFO)
    tgt = summarize_per_target(r["evs"], r["cand"], r["ts"], r["ce"], SINFO)
    smp = summarize_per_sample(r["evs"], r["cand"], eff, tgt, r["ce"], r["ts"], SINFO, APARAMS)
    row = smp.iloc[0]
    assert row["total_kill_credit"] == pytest.approx(row["n_attributed"]) == pytest.approx(1.0)
    assert row["n_contact_events"] == 3
    assert row["conversion_rate"] == pytest.approx(1 / 3)
    assert eff["kills_attributed"].sum() == pytest.approx(1.0)
    assert tgt.iloc[0]["n_death_events"] == 1

"""Synthetic-volume tests for behav3d.features.death_events.

Each case pins one property the Active Killing rework depends on. Volumes are
tiny (isotropic 1 µm voxels) and written as real zarr stores so the full
per-sample path (open handles, dead-mask cleaning, persistence) is exercised.
"""

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from behav3d.features.death_events import (
    DeathEventParams,
    _close_um,
    _dilate_um,
    detect_death_events_sample,
)
from behav3d.io.formats.zarr import save_as_zarr

T, Z, Y, X = 12, 20, 40, 40
SPACING = (1.0, 1.0, 1.0)
# d = 6 µm  ->  min patch volume ~28.3 µm³, closing 1 µm, link radius 2 µm.
PARAMS = DeathEventParams(target_cell_diameter_um=6.0)


@pytest.fixture
def case_dir():
    root = Path(__file__).parent / ".tmp_death_events"
    d = root / uuid.uuid4().hex
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _sphere(center, radius):
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius ** 2


def _cube(dead, t, corner, size):
    z, y, x = corner
    dead[t, z:z + size, y:y + size, x:x + size] = 1


def _run(case_dir, labels, dead, immune=None):
    lp, dp = case_dir / "labels.zarr", case_dir / "dead.zarr"
    save_as_zarr(labels.astype(np.uint16), lp)
    save_as_zarr(dead.astype(np.uint16), dp)
    imm = {}
    if immune is not None:
        ip = case_dir / "immune.zarr"
        save_as_zarr(immune.astype(np.uint16), ip)
        imm = {"tcell": ip}
    events, ts, patches = detect_death_events_sample(
        sample_name="s1", target_type="organoid", target_tracks_path=lp, dead_mask_path=dp,
        immune_tracks_paths=imm, voxel_spacing=SPACING, minutes_per_frame=2.0, params=PARAMS,
    )
    return events, ts, patches


def _static_organoid():
    labels = np.zeros((T, Z, Y, X), np.uint16)
    labels[:] = _sphere((10, 20, 20), 9) * 5
    return labels


def test_geometry_helpers_are_anisotropy_correct():
    m = np.zeros((9, 9, 9), bool)
    m[4, 4, 4] = True
    # 2 µm radius with 2 µm z spacing reaches one z-plane but two x/y voxels.
    d = _dilate_um(m, 2.0, (2.0, 1.0, 1.0))
    assert d[3, 4, 4] and d[5, 4, 4] and not d[2, 4, 4]
    assert d[4, 4, 2] and d[4, 4, 6] and not d[4, 4, 1]
    # Closing bridges a one-voxel gap between two blobs and is extensive (it
    # never removes an original voxel). A one-voxel-thin line is thinner than
    # the structuring ball and is correctly not bridged; real patches are blobs.
    g = np.zeros((7, 7, 11), bool)
    g[2:5, 2:5, 2:5] = True
    g[2:5, 2:5, 6:9] = True
    c = _close_um(g, 1.0, (1.0, 1.0, 1.0))
    assert c[3, 3, 5] and c[g].all()


def test_growing_patch_is_one_event(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for i, t in enumerate(range(3, T)):
        _cube(dead, t, (7, 17, 17), min(4 + i, 7))
    events, _, patches = _run(case_dir, labels, dead)
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t_onset"] == 3
    assert ev["target_track_id"] == 5
    assert ev["peak_volume_um3"] > ev["onset_volume_um3"]
    assert len(patches[int(ev["death_event_id"])]) == int(ev["onset_volume_um3"])


def test_death_present_at_first_sighting_is_not_new(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in range(T):
        _cube(dead, t, (7, 17, 17), 4)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 0


def test_translating_organoid_with_constant_patch_is_zero_events(case_dir):
    labels = np.zeros((T, Z, Y, X), np.uint16)
    dead = np.zeros_like(labels)
    for t in range(T):
        labels[t] = _sphere((10, 20, 12 + t), 8) * 5
        _cube(dead, t, (8, 18, 10 + t), 4)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 0


def test_single_frame_flicker_is_zero_events(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    _cube(dead, 5, (7, 17, 17), 5)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 0


def test_dim_then_rebrighten_is_one_event(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in (3, 4, 7, 8, 9, 10, 11):
        _cube(dead, t, (7, 17, 17), 4)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 1
    assert events.iloc[0]["t_onset"] == 3


def test_two_simultaneous_patches_are_two_events(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in range(3, T):
        _cube(dead, t, (8, 13, 13), 4)
        _cube(dead, t, (8, 23, 23), 4)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 2
    assert set(events["t_onset"]) == {3}


def test_patches_that_later_merge_stay_two_events(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in range(3, T):
        grow = min(t - 3, 4)
        dead[t, 8:12, 18:22, 12:16 + grow] = 1
        dead[t, 8:12, 18:22, 24 - grow:28] = 1
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 2
    assert (events["merged_into"] >= 0).sum() == 1


def test_patch_below_min_volume_is_dropped(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in range(3, T):
        _cube(dead, t, (8, 18, 18), 2)  # 8 µm³ < ~28 µm³ floor
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 0


def test_small_seed_that_grows_is_back_dated(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    _cube(dead, 3, (8, 18, 18), 2)
    for t in range(4, T):
        _cube(dead, t, (8, 18, 18), 4)
    events, _, _ = _run(case_dir, labels, dead)
    assert len(events) == 1
    assert events.iloc[0]["t_onset"] == 3
    assert events.iloc[0]["back_dated_frames"] == 1


def test_immune_segment_overlap_creates_no_patch(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    immune = np.zeros_like(labels)
    for t in range(3, T):
        _cube(dead, t, (7, 17, 17), 5)
        immune[t, 6:13, 16:23, 16:23] = 1  # effector sits on (and carries) the dye
    events, _, _ = _run(case_dir, labels, dead, immune)
    assert len(events) == 0


def test_timeseries_reports_raw_dead_volume(case_dir):
    labels = _static_organoid()
    dead = np.zeros_like(labels)
    for t in range(3, T):
        _cube(dead, t, (7, 17, 17), 4)
    _, ts, _ = _run(case_dir, labels, dead)
    last = ts[ts["position_t"] == T - 1].iloc[0]
    # The last frame cannot nucleate (no t+1), but its dead volume is still real.
    assert last["dead_volume_um3"] == 64.0

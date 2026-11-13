"""
Regression tests for the invasiveness cutout margin (INVASIVENESS_VERSION 3).

At a Contact Threshold of 0 um the contact cutout is the cell's exact bounding box, and
_surface_contact_percentage erodes with border_value=1, so every surface voxel on the
bounding-box faces used to be dropped. A blocky cell that fills its bounding box was left
with no surface at all and read 0 % even when fully engulfed by an organoid. The
invasiveness cutout now always has at least 1 voxel of margin; thresholds > 0 (whose
margin is already >= 2 voxels) are unchanged, and so are all contact columns.
"""
import json
import math

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from behav3d.features import timepoint_features as tf

SAMPLING = (4.655, 1.17, 1.17)  # z, y, x in um


def _run(monkeypatch, cells, orgs, threshold):
    """Run the real per-timepoint contact/invasiveness code on in-memory volumes."""
    volumes = {"cells": cells, "orgs": orgs}
    monkeypatch.setattr(tf, "load_image_timepoint", lambda path, t: volumes[path])
    args = (0, "cells", {"org": "orgs"}, {"im_t": "cells"}, {},
            SAMPLING[2], SAMPLING[1], SAMPLING[0], threshold, "im_t", True)
    return tf._calculate_contact_single_timepoint(args).set_index("TrackID")


def _organoid(shape):
    zz, yy, xx = np.indices(shape)
    return ((((zz - 6) * 4) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2) <= 25 ** 2).astype(np.uint16)


def test_blocky_engulfed_cell_is_fully_invasive(monkeypatch):
    shape = (12, 60, 60)
    cells = np.zeros(shape, np.uint16)
    cells[5:8, 27:33, 27:33] = 1      # 3x6x6 box: fills its own bounding box
    cells[5:7, 40:42, 40:42] = 2      # 2x2x2 blob
    df = _run(monkeypatch, cells, _organoid(shape), 0.0)
    for tid in (1, 2):
        assert df.loc[tid, "org_contact_on_distance"]
        assert df.loc[tid, "org_invasiveness_perc"] == pytest.approx(100.0)
        assert df.loc[tid, "org_invasiveness"]
        assert df.loc[tid, "any_org_invasiveness_perc"] == pytest.approx(100.0)


def test_rounded_engulfed_cell_unchanged(monkeypatch):
    shape = (12, 60, 60)
    zz, yy, xx = np.indices(shape)
    cells = ((((zz - 6) * 4) ** 2 <= 17) & ((yy - 30) ** 2 + (xx - 30) ** 2 <= 16)).astype(np.uint16)
    df = _run(monkeypatch, cells, _organoid(shape), 0.0)
    assert df.loc[1, "org_invasiveness_perc"] == pytest.approx(100.0)


def test_half_embedded_cube_is_half(monkeypatch):
    cells = np.zeros((10, 10, 10), np.uint16)
    cells[3:7, 3:7, 3:7] = 1
    orgs = np.zeros_like(cells)
    orgs[:, :, 5:] = 1
    monkeypatch.setattr(tf, "load_image_timepoint", lambda path, t: {"cells": cells, "orgs": orgs}[path])
    args = (0, "cells", {"org": "orgs"}, {"im_t": "cells"}, {}, 1.0, 1.0, 1.0, 0.0, "im_t", True)
    df = tf._calculate_contact_single_timepoint(args).set_index("TrackID")
    # 4x4x4 cube: 56 surface voxels, 28 of them at x >= 5.
    assert df.loc[1, "org_invasiveness_perc"] == pytest.approx(50.0)
    assert df.loc[1, "org_invasiveness"]


def test_no_contact_is_zero(monkeypatch):
    shape = (12, 60, 60)
    cells = np.zeros(shape, np.uint16)
    cells[5:8, 1:4, 1:4] = 1
    df = _run(monkeypatch, cells, _organoid(shape), 0.0)
    assert not df.loc[1, "org_contact_on_distance"]
    assert df.loc[1, "org_invasiveness_perc"] == 0.0
    assert not df.loc[1, "org_invasiveness"]


@pytest.mark.parametrize("threshold", [1.0, 3.5])
def test_positive_threshold_matches_contact_cutout(monkeypatch, threshold):
    """For thresholds > 0 the result equals the previous (contact-cutout) computation."""
    rng = np.random.default_rng(0)
    shape = (12, 40, 40)
    zz, yy, xx = np.indices(shape)
    orgs = ((((zz - 6) * 2) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) <= 9 ** 2).astype(np.uint16)
    cells = np.zeros(shape, np.uint16)
    for lab in range(1, 9):
        c = rng.integers([1, 2, 2], [11, 38, 38])
        cells[(((zz - c[0]) * 2) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= 9) & (cells == 0)] = lab
    df = _run(monkeypatch, cells, orgs, threshold)
    for lab in np.unique(cells)[1:]:
        locs = np.argwhere(cells == lab)
        lo, hi = locs.min(0), locs.max(0)
        ext = [2 * math.ceil(threshold / s) for s in SAMPLING]
        sl = tuple(slice(max(0, lo[i] - ext[i]), min(shape[i], hi[i] + ext[i] + 1)) for i in range(3))
        cut = cells[sl]
        contact = np.any(orgs[sl][distance_transform_edt(cut != lab, sampling=SAMPLING) <= threshold])
        expected = tf._surface_contact_percentage(cut == lab, orgs[sl], list(SAMPLING), threshold) if contact else 0.0
        assert df.loc[lab, "org_invasiveness_perc"] == pytest.approx(expected)


def test_contact_columns_unchanged_by_margin(monkeypatch):
    """The margin is for invasiveness only: a cell face-adjacent to an organoid (no
    overlap) keeps the pixel-contact result of the zero-margin contact cutout."""
    cells = np.zeros((8, 20, 20), np.uint16)
    cells[2:5, 5:10, 5:10] = 1
    orgs = np.zeros_like(cells)
    orgs[:, :, 10:] = 1
    monkeypatch.setattr(tf, "load_image_timepoint", lambda path, t: {"cells": cells, "orgs": orgs}[path])
    args = (0, "cells", {"org": "orgs"}, {"im_t": "cells"}, {}, 1.17, 1.17, 4.655, 0.0, "im_t", True)
    df = tf._calculate_contact_single_timepoint(args).set_index("TrackID")
    assert not df.loc[1, "org_contact"]
    assert not df.loc[1, "org_contact_on_distance"]
    assert df.loc[1, "org_invasiveness_perc"] == 0.0


def test_cached_contact_csv_from_previous_version_is_recomputed(tmp_path):
    csv = tmp_path / "contacts.csv"
    csv.write_text("TrackID,position_t\n")
    tf._contact_cache_meta_path(csv).write_text(json.dumps(
        {"contact_threshold": 0.0, "invasiveness_version": 2, "has_invasiveness": True}))
    usable, reason = tf._contact_cache_status(csv, 0.0, want_invasiveness=True)
    assert not usable and "invasiveness definition changed" in reason
    tf._write_contact_cache_meta(csv, 0.0, True)
    assert tf._contact_cache_status(csv, 0.0, want_invasiveness=True) == (True, None)

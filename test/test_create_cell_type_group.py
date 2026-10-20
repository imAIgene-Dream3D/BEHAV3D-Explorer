"""Correctness check for ``create_cell_type_group``'s TrackID remap.

Filtering (``filtering.py``) stamps every cell type's filtered CSV with an
``original_TrackID`` column so backprojection can find the pixels matching
a (possibly track-split) row. When multiple cell types are merged into a
group, ``TrackID`` is reassigned to a globally-unique value — this test
guards that ``original_TrackID`` is resynced too, since backprojection
prefers it over ``TrackID`` whenever it's present, and it must match the
merged group's own tracked image (which uses the new, merged ``TrackID``),
not each member's stale, pre-merge pixel labels.
"""
from pathlib import Path
import shutil
import uuid

import pandas as pd

from behav3d.analysis.grouping import (
    create_cell_type_group,
    filtered_track_features_csv,
)


def _case_dir(name):
    root = Path(__file__).resolve().parent / ".tmp_create_cell_type_group"
    root.mkdir(exist_ok=True)
    case_dir = root / f"{name}_{uuid.uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def _cleanup(path):
    if path.exists():
        shutil.rmtree(path)


def _write_filtered_csv(output_dir, cell_type, rows):
    path = filtered_track_features_csv(output_dir, cell_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_original_trackid_resynced_to_merged_trackid():
    case_dir = _case_dir("original_trackid")
    try:
        # Member A: no track-splitting, original_TrackID == TrackID for every row.
        rows_a = [
            {"TrackID": 1, "original_TrackID": 1, "sample_name": "s1", "position_t": 0},
            {"TrackID": 2, "original_TrackID": 2, "sample_name": "s1", "position_t": 0},
        ]
        _write_filtered_csv(case_dir, "tcell1", rows_a)

        # Member B: simulates a split long track — TrackID 2's rows keep
        # original_TrackID 1 (the pre-split id), diverging from TrackID.
        rows_b = [
            {"TrackID": 1, "original_TrackID": 1, "sample_name": "s1", "position_t": 0},
            {"TrackID": 2, "original_TrackID": 1, "sample_name": "s1", "position_t": 1},
        ]
        _write_filtered_csv(case_dir, "tcell2", rows_b)

        out_csv = create_cell_type_group(case_dir, "tcells", ["tcell1", "tcell2"])
        merged = pd.read_csv(out_csv)

        # The actual bug fix: original_TrackID must match the (remapped)
        # TrackID for every row, since that's what the merged tracked image's
        # pixels now use.
        assert (merged["original_TrackID"] == merged["TrackID"]).all()

        # TrackID stays globally unique across the merged frame.
        assert merged["TrackID"].is_unique

        # origin_TrackID (traceability column, no "al") still holds each
        # member's own pre-merge local TrackID, unaffected by the fix.
        origin_a = merged[merged["origin_cell_type"] == "tcell1"].sort_values("origin_TrackID")
        origin_b = merged[merged["origin_cell_type"] == "tcell2"].sort_values("origin_TrackID")
        assert list(origin_a["origin_TrackID"]) == [1, 2]
        assert list(origin_b["origin_TrackID"]) == [1, 2]
    finally:
        _cleanup(case_dir)


def test_merge_without_original_trackid_column_still_works():
    """Members without an original_TrackID column (e.g. never went through
    a Filtering run that stamps it) shouldn't error or gain the column."""
    case_dir = _case_dir("no_original_trackid")
    try:
        rows_a = [{"TrackID": 1, "sample_name": "s1", "position_t": 0}]
        rows_b = [{"TrackID": 1, "sample_name": "s1", "position_t": 0}]
        _write_filtered_csv(case_dir, "tcell1", rows_a)
        _write_filtered_csv(case_dir, "tcell2", rows_b)

        out_csv = create_cell_type_group(case_dir, "tcells", ["tcell1", "tcell2"])
        merged = pd.read_csv(out_csv)

        assert "original_TrackID" not in merged.columns
        assert merged["TrackID"].is_unique
    finally:
        _cleanup(case_dir)

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.state.visualization.plots.state_composition import (
    save_state_condition_comparison_report,
)
from behav3d.analysis.behavior.track.visualization.plots.reports import (
    save_track_condition_comparison_report,
)


# (sample_name, condition, position_t, ClusterID) -- one row per (track, frame).
_STATE_ROWS = [
    ("s1", "A", 0, "X"), ("s1", "A", 0, "X"), ("s1", "A", 0, "Y"),
    ("s1", "A", 1, "X"), ("s1", "A", 1, "X"), ("s1", "A", 1, "X"),
    ("s2", "A", 0, "X"), ("s2", "A", 0, "Y"), ("s2", "A", 0, "Y"),
    ("s2", "A", 1, "Y"), ("s2", "A", 1, "Y"), ("s2", "A", 1, "Y"),
    ("s3", "B", 0, "X"), ("s3", "B", 0, "Y"),
    ("s3", "B", 1, "X"), ("s3", "B", 1, "X"),
]

_STATE_ROWS_3COND = _STATE_ROWS + [
    ("s4", "C", 0, "X"), ("s4", "C", 0, "Y"),
    ("s4", "C", 1, "X"), ("s4", "C", 1, "Y"),
]


def _build_state_adata(rows=_STATE_ROWS):
    obs = pd.DataFrame(rows, columns=["sample_name", "condition", "position_t", "ClusterID"])
    obs.index = [f"row{i}" for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _build_track_adata(drop_time_cols=False):
    # s1: trackA1 (X) lives frames [0,2]; trackA2 (Y) lives only [0,1] (ends before frame 2).
    # s2: two X tracks, both live [0,2].
    # s3: trackB1 (Y) lives [0,2]; trackB2 (X) lives only frame [0,0].
    obs = pd.DataFrame(
        {
            "sample_name": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "condition": ["A", "A", "A", "A", "B", "B"],
            "ClusterID": ["X", "Y", "X", "X", "Y", "X"],
            "position_t_min": [0, 0, 0, 0, 0, 0],
            "position_t_max": [2, 1, 2, 2, 2, 0],
        },
        index=["A1", "A2", "A3", "A4", "B1", "B2"],
    )
    if drop_time_cols:
        obs = obs.drop(columns=["position_t_min", "position_t_max"])
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def test_state_over_time_page_appended_and_raw_csv_hand_computed(tmp_path):
    adata = _build_state_adata()
    out_pdf = tmp_path / "state_condition_comparison.pdf"

    result_off = save_state_condition_comparison_report(
        adata=adata,
        output_pdf_path=out_pdf,
        output_csv_path=out_pdf.with_suffix(".csv"),
        condition_col="condition",
        include_over_time=False,
        verbose=False,
    )
    pypdf = pytest.importorskip("pypdf")
    n_pages_off = len(pypdf.PdfReader(result_off["pdf_path"]).pages)
    assert result_off["over_time_raw_csv_path"] is None

    result_on = save_state_condition_comparison_report(
        adata=adata,
        output_pdf_path=out_pdf,
        output_csv_path=out_pdf.with_suffix(".csv"),
        condition_col="condition",
        include_over_time=True,
        verbose=False,
    )
    n_pages_on = len(pypdf.PdfReader(result_on["pdf_path"]).pages)
    assert n_pages_on > n_pages_off

    raw = pd.read_csv(result_on["over_time_raw_csv_path"])
    s1_f0_x = raw[(raw["comparison_unit"] == "s1 | A") & (raw["time"] == 0) & (raw["state_id"] == "X")]
    assert s1_f0_x["relative_proportion"].iloc[0] == pytest.approx(2 / 3)

    summary = pd.read_csv(result_on["over_time_summary_csv_path"])
    row_f0 = summary[(summary["condition"] == "A") & (summary["time"] == 0) & (summary["class"] == "X")].iloc[0]
    assert row_f0["mean"] == pytest.approx(0.5)
    assert row_f0["sem"] == pytest.approx(1 / 6)
    assert row_f0["n"] == 2

    row_f1 = summary[(summary["condition"] == "A") & (summary["time"] == 1) & (summary["class"] == "X")].iloc[0]
    assert row_f1["mean"] == pytest.approx(0.5)
    assert row_f1["sem"] == pytest.approx(0.5)


def test_track_over_time_page_appended_and_raw_csv_hand_computed(tmp_path):
    adata_tracks = _build_track_adata()
    out_dir = tmp_path / "track_report"

    result_off = save_track_condition_comparison_report(
        adata_tracks, out_dir, condition_col="condition", include_over_time=False, verbose=False,
    )
    pypdf = pytest.importorskip("pypdf")
    n_pages_off = len(pypdf.PdfReader(result_off["pdf_path"]).pages)

    result_on = save_track_condition_comparison_report(
        adata_tracks, out_dir, condition_col="condition", include_over_time=True, verbose=False,
    )
    n_pages_on = len(pypdf.PdfReader(result_on["pdf_path"]).pages)
    assert n_pages_on > n_pages_off

    raw = pd.read_csv(result_on["over_time_raw_csv_path"])
    # At frame 0/1, s1 has both trackA1 (X) and trackA2 (Y) alive -> X = 0.5.
    s1_f0_x = raw[(raw["comparison_unit"] == "s1 | A") & (raw["time"] == 0) & (raw["state_id"] == "X")]
    assert s1_f0_x["relative_proportion"].iloc[0] == pytest.approx(0.5)
    # At frame 2, trackA2 (Y)'s window [0,1] has ended -> only trackA1 (X) remains alive -> X = 1.0.
    s1_f2_x = raw[(raw["comparison_unit"] == "s1 | A") & (raw["time"] == 2) & (raw["state_id"] == "X")]
    assert s1_f2_x["relative_proportion"].iloc[0] == pytest.approx(1.0)

    summary = pd.read_csv(result_on["over_time_summary_csv_path"])
    row_f2 = summary[(summary["condition"] == "A") & (summary["time"] == 2) & (summary["class"] == "X")].iloc[0]
    assert row_f2["mean"] == pytest.approx(1.0)
    assert row_f2["sem"] == pytest.approx(0.0)
    assert row_f2["n"] == 2


def test_track_over_time_gracefully_skips_when_time_columns_missing(tmp_path):
    adata_tracks = _build_track_adata(drop_time_cols=True)
    out_dir = tmp_path / "track_report_no_time"

    result = save_track_condition_comparison_report(
        adata_tracks, out_dir, condition_col="condition", include_over_time=True, verbose=False,
    )

    assert result["over_time_raw_csv_path"] is None
    assert result["over_time_summary_csv_path"] is None
    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()


def test_over_time_degrades_to_frame_axis_when_minutes_unavailable(tmp_path):
    adata = _build_state_adata()
    out_pdf = tmp_path / "state_minutes_check.pdf"

    result = save_state_condition_comparison_report(
        adata=adata,
        output_pdf_path=out_pdf,
        output_csv_path=out_pdf.with_suffix(".csv"),
        condition_col="condition",
        include_over_time=True,
        minutes_per_frame=None,
        verbose=False,
    )

    raw = pd.read_csv(result["over_time_raw_csv_path"])
    assert "time_minutes" not in raw.columns
    summary = pd.read_csv(result["over_time_summary_csv_path"])
    assert set(summary["time_unit"].unique()) == {"frame"}

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(result["pdf_path"])
    texts = " ".join(page.extract_text() or "" for page in reader.pages)
    assert "minutes unavailable" in texts


def test_over_time_draws_one_line_per_condition_level_for_more_than_two_levels(tmp_path):
    adata = _build_state_adata(_STATE_ROWS_3COND)
    out_pdf = tmp_path / "state_3cond.pdf"

    result = save_state_condition_comparison_report(
        adata=adata,
        output_pdf_path=out_pdf,
        output_csv_path=out_pdf.with_suffix(".csv"),
        condition_col="condition",
        include_over_time=True,
        verbose=False,
    )
    summary = pd.read_csv(result["over_time_summary_csv_path"])
    assert set(summary["condition"].unique()) == {"A", "B", "C"}

    out_pdf_grouped = tmp_path / "state_3cond_grouped.pdf"
    result_grouped = save_state_condition_comparison_report(
        adata=adata,
        output_pdf_path=out_pdf_grouped,
        output_csv_path=out_pdf_grouped.with_suffix(".csv"),
        condition_col="condition",
        include_over_time=True,
        condition_groups={"A": "grp1", "B": "grp1", "C": "grp2"},
        verbose=False,
    )
    summary_grouped = pd.read_csv(result_grouped["over_time_summary_csv_path"])
    assert set(summary_grouped["condition"].unique()) == {"grp1", "grp2"}

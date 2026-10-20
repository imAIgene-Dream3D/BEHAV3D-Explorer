import importlib.util
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "behav3d/analysis/behavior/general/visualization/plots/proportion_bars.py"
)
SPEC = importlib.util.spec_from_file_location("proportion_bars", MODULE_PATH)
proportion_bars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proportion_bars)


def test_stacked_proportion_barh_grid_accepts_series_values():
    fig = proportion_bars.plot_page_stacked_proportion_barh_grid(
        {
            "sample_a": pd.Series({"rest": 0.25, "move": 0.75}),
            "sample_b": pd.Series({"rest": 0.5, "move": 0.5}),
        },
        row_order=["sample_a", "sample_b"],
        class_order=["rest", "move"],
        colors={"rest": "#4477AA", "move": "#EE6677"},
        title="Series-backed proportions",
    )

    assert fig is not None
    plt.close(fig)


def _build_condition_diff_inputs():
    samples = ["s1", "s2", "s3", "s4", "s5", "s6"]
    per_sample_class_props = pd.DataFrame(
        {
            "rest": [0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
            "move": [0.8, 0.7, 0.5, 0.4, 0.3, 0.2],
        },
        index=samples,
    )
    sample_metadata = pd.DataFrame(
        {
            "condition": ["None", "None", "M21", "M21", "M23", "M23"],
            "batch": ["b1", "b2", "b1", "b2", "b1", "b2"],
        },
        index=samples,
    )
    return per_sample_class_props, sample_metadata


def test_compute_condition_diff_stats_pairwise_orders_pairs_by_first_appearance():
    per_sample_class_props, sample_metadata = _build_condition_diff_inputs()

    out = proportion_bars.compute_condition_diff_stats_pairwise(
        per_sample_class_props,
        sample_metadata,
        class_order=["rest", "move"],
        condition_col="condition",
    )

    assert list(out.keys()) == ["(all)"]
    pairs = list(out["(all)"].keys())
    assert pairs == [("None", "M21"), ("None", "M23"), ("M21", "M23")]
    for diff_df in out["(all)"].values():
        assert list(diff_df["class"]) == ["rest", "move"]


def test_compute_condition_diff_stats_pairwise_splits_by_group_cols():
    per_sample_class_props, sample_metadata = _build_condition_diff_inputs()

    out = proportion_bars.compute_condition_diff_stats_pairwise(
        per_sample_class_props,
        sample_metadata,
        class_order=["rest", "move"],
        condition_col="condition",
        group_cols=["batch"],
    )

    assert set(out.keys()) == {"b1", "b2"}
    for pairs in out.values():
        assert list(pairs.keys()) == [("None", "M21"), ("None", "M23"), ("M21", "M23")]


def test_plot_condition_diff_grid_single_column_and_multi_group(tmp_path):
    per_sample_class_props, sample_metadata = _build_condition_diff_inputs()
    colors = {"rest": "#4477AA", "move": "#EE6677"}

    diff_stats = proportion_bars.compute_condition_diff_stats_pairwise(
        per_sample_class_props,
        sample_metadata,
        class_order=["rest", "move"],
        condition_col="condition",
    )
    out_pdf = tmp_path / "single_row.pdf"
    result = proportion_bars.plot_condition_diff_grid(
        diff_stats,
        class_order=["rest", "move"],
        colors=colors,
        title="Single-row comparison",
        out_pdf=out_pdf,
        out_csv=out_pdf.with_suffix(".csv"),
    )
    assert Path(result["pdf_path"]).exists()
    csv_out = pd.read_csv(result["csv_path"], keep_default_na=False)
    assert set(csv_out["group"].unique()) == {"(all)"}
    assert set(zip(csv_out["level_a"], csv_out["level_b"])) == {
        ("None", "M21"), ("None", "M23"), ("M21", "M23"),
    }

    diff_stats_grouped = proportion_bars.compute_condition_diff_stats_pairwise(
        per_sample_class_props,
        sample_metadata,
        class_order=["rest", "move"],
        condition_col="condition",
        group_cols=["batch"],
    )
    out_pdf_grouped = tmp_path / "grouped.pdf"
    result_grouped = proportion_bars.plot_condition_diff_grid(
        diff_stats_grouped,
        class_order=["rest", "move"],
        colors=colors,
        title="Grouped comparison",
        out_pdf=out_pdf_grouped,
        out_csv=out_pdf_grouped.with_suffix(".csv"),
    )
    assert Path(result_grouped["pdf_path"]).exists()
    csv_out_grouped = pd.read_csv(result_grouped["csv_path"])
    assert set(csv_out_grouped["group"].unique()) == {"b1", "b2"}


def test_compute_condition_time_series_stats_hand_computed_mean_sem():
    raw = pd.DataFrame({
        "condition": ["A", "A", "B", "B"],
        "time": [0, 0, 0, 0],
        "state_id": ["rest", "rest", "rest", "rest"],
        "relative_proportion": [0.2, 0.4, 0.6, 0.9],
    })

    out = proportion_bars.compute_condition_time_series_stats(
        raw, class_order=["rest"], condition_col="condition",
    )

    row_a = out[out["condition"] == "A"].iloc[0]
    row_b = out[out["condition"] == "B"].iloc[0]
    assert row_a["mean"] == pytest.approx(0.3)
    assert row_a["sem"] == pytest.approx(0.1)
    assert row_a["n"] == 2
    assert row_a["time_unit"] == "frame"
    assert row_b["mean"] == pytest.approx(0.75)
    assert row_b["sem"] == pytest.approx(0.15)


def test_compute_condition_time_series_stats_single_unit_sem_is_zero():
    raw = pd.DataFrame({
        "condition": ["A"],
        "time": [0],
        "state_id": ["rest"],
        "relative_proportion": [0.5],
    })

    out = proportion_bars.compute_condition_time_series_stats(
        raw, class_order=["rest"], condition_col="condition",
    )

    assert out.iloc[0]["sem"] == 0.0
    assert out.iloc[0]["n"] == 1


def test_build_condition_time_series_raw_table_adds_minutes_column_only_when_given():
    long_by_unit = pd.DataFrame({
        "comparison_unit": ["s1 | A", "s2 | B"],
        "time": [0, 10],
        "state_id": ["rest", "rest"],
        "relative_proportion": [0.5, 0.5],
    })
    unit_metadata = pd.DataFrame({"condition": ["A", "B"]}, index=["s1 | A", "s2 | B"])

    out_no_minutes = proportion_bars.build_condition_time_series_raw_table(
        long_by_unit, unit_metadata, condition_col="condition",
    )
    assert "time_minutes" not in out_no_minutes.columns

    out_with_minutes = proportion_bars.build_condition_time_series_raw_table(
        long_by_unit, unit_metadata, condition_col="condition", minutes_per_frame=2.0,
    )
    assert list(out_with_minutes["time_minutes"]) == [0.0, 20.0]


def test_plot_condition_time_series_grid_paginates_by_class(tmp_path):
    classes = [f"c{i}" for i in range(5)]
    rows = []
    for c in classes:
        rows.append({"group": "(all)", "condition": "A", "class": c, "time": 0, "time_unit": "frame", "mean": 0.5, "sem": 0.05, "n": 2})
        rows.append({"group": "(all)", "condition": "B", "class": c, "time": 0, "time_unit": "frame", "mean": 0.3, "sem": 0.05, "n": 2})
    stats_df = pd.DataFrame(rows)
    out_pdf = tmp_path / "dyn.pdf"

    result = proportion_bars.plot_condition_time_series_grid(
        stats_df,
        class_order=classes,
        condition_levels=["A", "B"],
        colors={"A": "#4477AA", "B": "#EE6677"},
        title="Dynamics",
        out_pdf=out_pdf,
        ncols=2,
        max_rows_per_page=2,
    )

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(result["pdf_path"])
    assert len(reader.pages) == 2  # 5 classes, 4 panels/page -> ceil(5/4) = 2 pages


def test_plot_condition_time_series_grid_multi_group_pages(tmp_path):
    classes = ["c0", "c1"]
    rows = []
    for grp in ["g1", "g2"]:
        for c in classes:
            rows.append({"group": grp, "condition": "A", "class": c, "time": 0, "time_unit": "frame", "mean": 0.5, "sem": 0.0, "n": 1})
    stats_df = pd.DataFrame(rows)
    out_pdf = tmp_path / "dyn_groups.pdf"

    result = proportion_bars.plot_condition_time_series_grid(
        stats_df,
        class_order=classes,
        condition_levels=["A"],
        colors={"A": "#4477AA"},
        title="Dynamics",
        out_pdf=out_pdf,
        ncols=2,
        max_rows_per_page=2,
    )

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(result["pdf_path"])
    assert len(reader.pages) == 2  # one page per group, both classes fit on each


def test_plot_condition_time_series_grid_empty_class_order_writes_placeholder(tmp_path):
    out_pdf = tmp_path / "empty.pdf"
    empty_stats = pd.DataFrame(columns=["group", "condition", "class", "time", "time_unit", "mean", "sem", "n"])

    result = proportion_bars.plot_condition_time_series_grid(
        empty_stats,
        class_order=[],
        condition_levels=["A"],
        colors={"A": "#4477AA"},
        title="Dynamics",
        out_pdf=out_pdf,
    )

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(result["pdf_path"])
    assert len(reader.pages) == 1

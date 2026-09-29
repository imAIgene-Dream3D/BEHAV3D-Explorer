import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.track.visualization.plots.contact_type_report import (
    save_track_contact_type_comparison,
)
from behav3d.analysis.behavior.state.visualization.plots.contact_type_report import (
    save_state_contact_type_comparison,
)

_N_TIMEPOINTS = 10
_MIN_BOUT_LENGTH = 5

# TrackID 0/1 -> cluster "A" (mostly healthy contact); TrackID 2/3 -> cluster "B" (mostly tumor
# contact). Each track is in contact for the whole window with exactly one organoid type.
_TRACKS = [
    (0, "A", "healthy_organoid_contact"),
    (1, "A", "healthy_organoid_contact"),
    (2, "B", "tumor_organoid_contact"),
    (3, "B", "tumor_organoid_contact"),
]


def _build_df_timepoints():
    rows = []
    for track_id, _cluster, contacted_col in _TRACKS:
        for t in range(_N_TIMEPOINTS):
            rows.append({
                "sample_name": "sample_1",
                "TrackID": track_id,
                "position_t": t,
                "healthy_organoid_contact": int(contacted_col == "healthy_organoid_contact"),
                "tumor_organoid_contact": int(contacted_col == "tumor_organoid_contact"),
            })
    return pd.DataFrame(rows)


def _build_adata_tracks():
    obs = pd.DataFrame(
        {
            "sample_name": ["sample_1"] * len(_TRACKS),
            "TrackID": [t[0] for t in _TRACKS],
            "ClusterID": [t[1] for t in _TRACKS],
            "position_t_min": 0,
            "position_t_max": _N_TIMEPOINTS - 1,
        }
    )
    obs.index = [str(t[0]) for t in _TRACKS]
    return ad.AnnData(X=np.zeros((len(_TRACKS), 1)), obs=obs)


def test_save_track_contact_type_comparison_requires_two_columns():
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()
    with pytest.raises(ValueError, match="at least 2 columns"):
        save_track_contact_type_comparison(
            adata_tracks, df_timepoints, "/tmp/unused",
            contact_cols=["healthy_organoid_contact"], min_bout_length=_MIN_BOUT_LENGTH,
        )


def test_save_track_contact_type_comparison_means_and_outputs(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_type_comparison(
        adata_tracks, df_timepoints, tmp_path,
        contact_cols=["healthy_organoid_contact", "tumor_organoid_contact"],
        min_bout_length=_MIN_BOUT_LENGTH,
        class_col="ClusterID",
        verbose=False,
    )

    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()
    assert result["class_order"] == ["A", "B"]

    long_df = pd.read_csv(result["csv_path"])
    means = long_df.groupby(["ClusterID", "contact_col"])["mean_fraction"].mean()
    assert means.loc[("A", "healthy_organoid_contact")] == pytest.approx(1.0)
    assert means.loc[("A", "tumor_organoid_contact")] == pytest.approx(0.0)
    assert means.loc[("B", "healthy_organoid_contact")] == pytest.approx(0.0)
    assert means.loc[("B", "tumor_organoid_contact")] == pytest.approx(1.0)

    # No significance-test columns should be computed for this purely descriptive comparison.
    assert not {"p_value", "chi2", "cramers_v", "stars"} & set(long_df.columns)


# TrackID 4 is positive for BOTH contact columns (with different timepoint subsets), so it
# contributes an independent unit row to each group with different state proportions — a
# regression check that units aren't collapsed to one row per track.
_MIXED_SPECS = {
    0: {"healthy": lambda t: True, "tumor": lambda t: False, "state": lambda t: "A" if t < 6 else "B"},
    1: {"healthy": lambda t: True, "tumor": lambda t: False, "state": lambda t: "A" if t < 8 else "B"},
    2: {"healthy": lambda t: False, "tumor": lambda t: True, "state": lambda t: "A" if t < 2 else "B"},
    3: {"healthy": lambda t: False, "tumor": lambda t: True, "state": lambda t: "A" if t < 4 else "B"},
    4: {"healthy": lambda t: True, "tumor": lambda t: t < 5, "state": lambda t: "A" if t < 5 else "B"},
}


def _build_df_timepoints_mixed():
    rows = [
        {
            "sample_name": "sample_1",
            "TrackID": track_id,
            "position_t": t,
            "healthy_organoid_contact": int(spec["healthy"](t)),
            "tumor_organoid_contact": int(spec["tumor"](t)),
        }
        for track_id, spec in _MIXED_SPECS.items() for t in range(_N_TIMEPOINTS)
    ]
    return pd.DataFrame(rows)


def _build_adata_states_mixed():
    rows = [
        {
            "sample_name": "sample_1",
            "TrackID": track_id,
            "position_t": t,
            "full_behavioral_state": spec["state"](t),
        }
        for track_id, spec in _MIXED_SPECS.items() for t in range(_N_TIMEPOINTS)
    ]
    obs = pd.DataFrame(rows)
    obs.index = [str(i) for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def test_save_state_contact_type_comparison_requires_two_columns():
    df_timepoints = _build_df_timepoints_mixed()
    adata_states = _build_adata_states_mixed()
    with pytest.raises(ValueError, match="at least 2 columns"):
        save_state_contact_type_comparison(
            adata_states, df_timepoints, "/tmp/unused",
            contact_cols=["healthy_organoid_contact"], state_col="full_behavioral_state",
        )


def test_save_state_contact_type_comparison_diff_and_stacked_outputs(tmp_path):
    df_timepoints = _build_df_timepoints_mixed()
    adata_states = _build_adata_states_mixed()

    result = save_state_contact_type_comparison(
        adata_states, df_timepoints, tmp_path,
        contact_cols=["healthy_organoid_contact", "tumor_organoid_contact"],
        state_col="full_behavioral_state",
        verbose=False,
    )

    assert Path(result["pdf_path"]).exists()
    assert Path(result["diff_csv_path"]).exists()
    assert Path(result["stacked_csv_path"]).exists()
    assert result["state_order"] == ["A", "B"]

    # healthy group = tracks 0, 1, 4 -> per-track proportion of state "A": 0.6, 0.8, 0.5
    # tumor group   = tracks 2, 3, 4 -> per-track proportion of state "A": 0.2, 0.4, 1.0
    # (track 4 contributes a DIFFERENT proportion to each group, from its own timepoints where
    # that contact column is True — the "not mutually exclusive, one unit per contact_col" check.)
    healthy_mean_a = (0.6 + 0.8 + 0.5) / 3
    healthy_mean_b = (0.4 + 0.2 + 0.5) / 3
    tumor_mean_a = (0.2 + 0.4 + 1.0) / 3
    tumor_mean_b = (0.8 + 0.6 + 0.0) / 3

    stacked = pd.read_csv(result["stacked_csv_path"]).set_index(["contact_col", "state"])["proportion"]
    assert stacked.loc[("healthy_organoid_contact", "A")] == pytest.approx(healthy_mean_a)
    assert stacked.loc[("healthy_organoid_contact", "B")] == pytest.approx(healthy_mean_b)
    assert stacked.loc[("tumor_organoid_contact", "A")] == pytest.approx(tumor_mean_a)
    assert stacked.loc[("tumor_organoid_contact", "B")] == pytest.approx(tumor_mean_b)

    diff_df = pd.read_csv(result["diff_csv_path"])
    row_a = diff_df[diff_df["class"] == "A"].iloc[0]
    assert row_a["mean_a"] == pytest.approx(healthy_mean_a)
    assert row_a["mean_b"] == pytest.approx(tumor_mean_a)
    assert row_a["diff"] == pytest.approx(tumor_mean_a - healthy_mean_a)
    assert row_a["n_a"] == 3 and row_a["n_b"] == 3

    # This is a Welch's-t-test cluster-size-difference comparison, not the old purely
    # descriptive mean-fraction bar chart — no chi-square columns should leak in.
    assert not {"chi2", "cramers_v"} & set(diff_df.columns)

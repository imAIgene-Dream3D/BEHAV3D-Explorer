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
from behav3d.analysis.behavior.state.contact_grouping import compute_state_contact_type_values
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


def _build_adata_states():
    rows = []
    for track_id, cluster, _contacted_col in _TRACKS:
        for t in range(_N_TIMEPOINTS):
            rows.append({
                "sample_name": "sample_1",
                "TrackID": track_id,
                "position_t": t,
                "full_behavioral_cluster": cluster,
            })
    obs = pd.DataFrame(rows)
    obs.index = [str(i) for i in range(len(obs))]
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


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


def test_compute_state_contact_type_values_per_state_per_track():
    df_timepoints = _build_df_timepoints()
    adata_states = _build_adata_states()

    long_df = compute_state_contact_type_values(
        df_timepoints, adata_states,
        contact_cols=["healthy_organoid_contact", "tumor_organoid_contact"],
        state_col="full_behavioral_cluster",
    )
    assert set(long_df.columns) == {
        "sample_name", "TrackID", "full_behavioral_cluster", "contact_col",
        "mean_fraction", "n_timepoints",
    }
    row = long_df[
        (long_df["TrackID"].astype(str) == "0") & (long_df["contact_col"] == "healthy_organoid_contact")
    ].iloc[0]
    assert row["mean_fraction"] == pytest.approx(1.0)
    assert row["n_timepoints"] == _N_TIMEPOINTS


def test_save_state_contact_type_comparison_means_and_outputs(tmp_path):
    df_timepoints = _build_df_timepoints()
    adata_states = _build_adata_states()

    result = save_state_contact_type_comparison(
        adata_states, df_timepoints, tmp_path,
        contact_cols=["healthy_organoid_contact", "tumor_organoid_contact"],
        state_col="full_behavioral_cluster",
        verbose=False,
    )

    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()
    assert result["state_order"] == ["A", "B"]

    long_df = pd.read_csv(result["csv_path"])
    means = long_df.groupby(["full_behavioral_cluster", "contact_col"])["mean_fraction"].mean()
    assert means.loc[("A", "healthy_organoid_contact")] == pytest.approx(1.0)
    assert means.loc[("B", "tumor_organoid_contact")] == pytest.approx(1.0)
    assert not {"p_value", "chi2", "cramers_v", "stars"} & set(long_df.columns)

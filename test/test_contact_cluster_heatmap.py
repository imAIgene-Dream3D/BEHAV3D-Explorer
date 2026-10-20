import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.track.visualization.plots.contact_cluster_heatmap import (
    save_track_contact_cluster_heatmap,
)

# Same fixture shape as test_contact_group_analysis.py: 4 samples x 4 tracks, alternating
# (ClusterID, has_contact_bout) patterns per sample so both clusters end up with a 50/50
# contact/no-contact split (8 tracks each) — enough to check per-cluster aggregation is correct
# without needing an unbalanced fixture.
_SAMPLES = ["sample_1", "sample_2", "sample_3", "sample_4"]
_CLUSTER_PATTERN_A = [("A", True), ("B", False), ("A", False), ("B", True)]
_CLUSTER_PATTERN_B = [("A", False), ("B", True), ("A", True), ("B", False)]
_N_TIMEPOINTS = 10
_MIN_BOUT_LENGTH = 5

_TRACKS = [
    (sample_name, cluster, has_contact)
    for i, sample_name in enumerate(_SAMPLES)
    for cluster, has_contact in (_CLUSTER_PATTERN_A if i % 2 == 0 else _CLUSTER_PATTERN_B)
]


def _build_adata_tracks():
    obs = pd.DataFrame(
        {
            "sample_name": [t[0] for t in _TRACKS],
            "TrackID": list(range(len(_TRACKS))),
            "ClusterID": [t[1] for t in _TRACKS],
            "position_t_min": 0,
            "position_t_max": _N_TIMEPOINTS - 1,
        }
    )
    obs.index = [str(i) for i in range(len(_TRACKS))]
    return ad.AnnData(X=np.zeros((len(_TRACKS), 1)), obs=obs)


def _build_df_timepoints():
    rows = []
    for track_id, (sample_name, _cluster, has_contact) in enumerate(_TRACKS):
        contact_values = np.zeros(_N_TIMEPOINTS, dtype=int)
        if has_contact:
            contact_values[: _MIN_BOUT_LENGTH + 1] = 1
        for t in range(_N_TIMEPOINTS):
            rows.append(
                {
                    "sample_name": sample_name,
                    "TrackID": track_id,
                    "position_t": t,
                    "macro_contact": int(contact_values[t]),
                }
            )
    return pd.DataFrame(rows)


def test_contact_cluster_heatmap_basic(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_cluster_heatmap(
        adata_tracks,
        df_timepoints,
        tmp_path,
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        class_col="ClusterID",
        verbose=False,
    )

    assert result["class_order"] == ["A", "B"]
    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()

    csv = pd.read_csv(result["csv_path"]).set_index("ClusterID")
    assert set(csv.index) == {"A", "B"}
    assert "mean_max_bout_minutes" not in csv.columns

    for cluster in ("A", "B"):
        assert csv.loc[cluster, "n_tracks"] == 8
        assert csv.loc[cluster, "n_contact"] == 4
        assert csv.loc[cluster, "pct_contact"] == pytest.approx(50.0)
        # Contact tracks have a 6-timepoint bout (min_bout_length + 1), no-contact tracks 0 —
        # a 50/50 split averages to 3.0 timepoints / 30% of the window.
        assert csv.loc[cluster, "mean_max_bout_timepoints"] == pytest.approx(3.0)
        assert csv.loc[cluster, "mean_fraction_pct"] == pytest.approx(30.0)


def test_contact_cluster_heatmap_minutes_per_frame(tmp_path):
    adata_tracks = _build_adata_tracks()
    df_timepoints = _build_df_timepoints()

    result = save_track_contact_cluster_heatmap(
        adata_tracks,
        df_timepoints,
        tmp_path,
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        class_col="ClusterID",
        minutes_per_frame=2.0,
        verbose=False,
    )

    csv = pd.read_csv(result["csv_path"]).set_index("ClusterID")
    assert "mean_max_bout_minutes" in csv.columns
    for cluster in ("A", "B"):
        assert csv.loc[cluster, "mean_max_bout_minutes"] == pytest.approx(6.0)


def test_contact_cluster_heatmap_no_clusters(tmp_path):
    obs = pd.DataFrame(
        {
            "sample_name": [],
            "TrackID": [],
            "ClusterID": [],
            "position_t_min": [],
            "position_t_max": [],
        }
    )
    adata_tracks = ad.AnnData(X=np.zeros((0, 1)), obs=obs)
    df_timepoints = pd.DataFrame(
        {"sample_name": [], "TrackID": [], "position_t": [], "macro_contact": []}
    )

    result = save_track_contact_cluster_heatmap(
        adata_tracks,
        df_timepoints,
        tmp_path,
        contact_col="macro_contact",
        min_bout_length=_MIN_BOUT_LENGTH,
        class_col="ClusterID",
        verbose=False,
    )

    assert result["class_order"] == []
    assert Path(result["pdf_path"]).exists()
    assert Path(result["csv_path"]).exists()

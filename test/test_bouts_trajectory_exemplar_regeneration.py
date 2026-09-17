from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.track.bouts import (
    run_state_based_analysis,
    save_bouts_exemplar_overview,
)
from behav3d.analysis.behavior.track.state_dtw import (
    _load_filtered_state_adata_for_model,
    save_dtaidistance_exemplar_overview,
)
from behav3d.analysis.behavior.track.visualization.plots.exemplar_track_per_cluster import (
    select_exemplar_tracks_by_cluster,
)


_STATE_CYCLE = ["move", "rest", "contact"]


def _make_behavioral_states_adata(n_tracks=4, track_len=200):
    rows = []
    for track_id in range(n_tracks):
        sample_name = f"sample_{track_id % 2}"
        for t in range(track_len):
            rows.append(
                {
                    "sample_name": sample_name,
                    "TrackID": track_id,
                    "position_t": t,
                    "position_x": float(t + 0.1 * track_id),
                    "position_y": float(np.sin(t / 5.0) + 0.2 * track_id),
                    "position_z": 0.0,
                    FULL_STATE_COL: _STATE_CYCLE[(t + track_id) % len(_STATE_CYCLE)],
                }
            )
    obs = pd.DataFrame(rows).reset_index(drop=True)
    obs[FULL_STATE_COL] = obs[FULL_STATE_COL].astype("category")
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _run_split_bouts_clustering(output_dir, states_path):
    return run_state_based_analysis(
        output_dir=output_dir,
        cell_type="tcell",
        adata_full_path=states_path,
        state_col=FULL_STATE_COL,
        behavioral_trajectory_size=50,
        split_long_tracks=True,
        use_bigrams=False,
        use_trigrams=False,
        drop_highly_correlated=False,
        drop_low_variance=False,
        do_pca=False,
        clustering_method="agglomerative",
        n_clusters=2,
        n_neighbors=5,
        plot_results=False,
        plot_exemplars=False,
        n_per_cluster=25,
        random_state=7,
        save_outputs=True,
        verbose=False,
    )


def test_bouts_clustering_stamps_full_reload_metadata_for_split_tracks(tmp_path):
    output_dir = Path(tmp_path) / "bouts_case"
    state_dir = output_dir / "analysis" / "tcell" / "behavioral_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    states_path = state_dir / "BEHAV3D_tcell_behavioral_states.h5ad"
    _make_behavioral_states_adata(n_tracks=4, track_len=200).write(states_path, compression="gzip")

    result = _run_split_bouts_clustering(output_dir, states_path)

    meta = result.uns["dtai_trajectory_clustering"]
    assert meta["method"] == "bouts_feature_clustering"
    assert meta["split_long_tracks"] is True
    assert meta["behavioral_trajectory_size"] == 50
    assert meta["min_track_length"] == 50
    assert meta["trajectory_window_col"] == "trajectory_window_id"
    assert meta["state_col"] == FULL_STATE_COL
    assert meta["source_adata_full_path"] == str(states_path)
    assert meta["n_per_cluster"] == 25
    assert meta["random_state"] == 7

    # Simulates what the "Create Diagnostics" / post-rename regeneration paths
    # do: reload the raw per-timepoint states and re-split them using only the
    # metadata stamped on the clustering result above (no re-clustering).
    adata_filt = _load_filtered_state_adata_for_model(
        result, str(output_dir), "tcell", verbose=False,
    )
    assert "trajectory_window_id" in adata_filt.obs.columns

    window_sizes = adata_filt.obs.groupby(
        ["sample_name", "TrackID", "trajectory_window_id"], observed=True
    ).size()
    assert (window_sizes == 50).all()

    windows_per_track = adata_filt.obs.groupby(
        ["sample_name", "TrackID"], observed=True
    )["trajectory_window_id"].nunique()
    assert (windows_per_track == 4).all()


def test_exemplar_overview_regenerates_with_multiple_tracklets_per_track(tmp_path):
    output_dir = Path(tmp_path) / "bouts_exemplar_case"
    state_dir = output_dir / "analysis" / "tcell" / "behavioral_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    states_path = state_dir / "BEHAV3D_tcell_behavioral_states.h5ad"
    _make_behavioral_states_adata(n_tracks=4, track_len=200).write(states_path, compression="gzip")

    result = _run_split_bouts_clustering(output_dir, states_path)

    overview_pdf = save_dtaidistance_exemplar_overview(
        result,
        output_dir=str(output_dir),
        cell_type="tcell",
        n_per_cluster=25,
        verbose=False,
    )
    assert Path(overview_pdf).exists()

    chosen, _ = select_exemplar_tracks_by_cluster(
        result, n_per_cluster=25, cluster_key="ClusterID", seed=0,
    )
    windows_per_track = chosen.groupby("TrackID")["trajectory_window_id"].nunique()
    assert windows_per_track.max() > 1, (
        "expected at least one TrackID to contribute more than one distinct "
        "tracklet (trajectory_window_id) to the exemplar selection"
    )


def test_bouts_native_exemplar_overview_regenerates_with_multiple_tracklets_per_track(tmp_path):
    """The napari rename/"Create Diagnostics" flows regenerate the bouts exemplar
    overview via `save_bouts_exemplar_overview` (reusing the same plotting logic
    bouts.py runs right after clustering), not `save_dtaidistance_exemplar_overview`.
    This mirrors `test_exemplar_overview_regenerates_with_multiple_tracklets_per_track`
    above, but exercises that function directly to make sure it also renders
    split tracks (multiple tracklets sharing one TrackID) without raising."""
    output_dir = Path(tmp_path) / "bouts_native_exemplar_case"
    state_dir = output_dir / "analysis" / "tcell" / "behavioral_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    states_path = state_dir / "BEHAV3D_tcell_behavioral_states.h5ad"
    _make_behavioral_states_adata(n_tracks=4, track_len=200).write(states_path, compression="gzip")

    result = _run_split_bouts_clustering(output_dir, states_path)

    overview_pdf = save_bouts_exemplar_overview(
        result,
        output_dir=str(output_dir),
        cell_type="tcell",
        n_per_cluster=25,
        verbose=False,
    )
    assert Path(overview_pdf).exists()
    assert Path(overview_pdf).parent.name == "clustering", (
        "expected the regenerated overview to land directly in the same "
        "clustering/ folder run_state_based_analysis itself writes to"
    )

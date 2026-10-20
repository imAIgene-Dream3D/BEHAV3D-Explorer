"""Regression tests for the clear_outputs=True default added to every
clustering entry point: rerunning clustering must wipe stale files left over
from a previous attempt (renamed clusters, changed cluster counts, etc.)
instead of leaving them to accumulate alongside new outputs.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from behav3d.analysis.behavior.state.classification import (
    _resolve_state_paths,
    run_hmm_state_clustering,
)
from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.track.bouts import run_state_based_analysis
from behav3d.analysis.behavior.track.feature_dtw import cluster_umap
from behav3d.analysis.behavior.track.utils import _resolve_track_paths


def _make_positions_df(n_tracks=6, track_len=15):
    rows = []
    for track_id in range(n_tracks):
        group_id = track_id % 3
        sample_name = f"sample_{track_id // 3}"
        for t in range(track_len):
            rows.append(
                {
                    "sample_name": sample_name,
                    "TrackID": track_id,
                    "position_t": t,
                    "position_x": float(t + 0.15 * track_id),
                    "position_y": float(group_id * 2.5 + np.sin(t / 3.0) + 0.05 * track_id),
                    "position_z": float((t % 4) * 0.4 + group_id * 0.2),
                    "speed": float(group_id + 0.25 * t + 0.03 * ((-1) ** t)),
                    "elongation": float(1.0 + group_id * 0.4 + 0.08 * np.cos(t / 2.0) + 0.01 * track_id),
                }
            )
    return pd.DataFrame(rows)


def _plant_stale_file(state_outdir):
    stale = Path(state_outdir) / "state_composition" / "stale_old_report.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")
    return stale


def test_run_hmm_state_clustering_clear_outputs_removes_stale_files(tmp_path):
    df_positions = _make_positions_df()
    output_dir = Path(tmp_path) / "hmm_clear"

    run_hmm_state_clustering(
        features=["speed", "elongation"],
        binary_features_to_group=[],
        output_dir=output_dir,
        cell_type="tcell",
        n_states=2,
        random_state=19,
        df_positions=df_positions,
        verbose=False,
    )
    state_paths = _resolve_state_paths(output_dir, "tcell")
    stale = _plant_stale_file(state_paths.state_outdir)
    assert stale.exists()

    run_hmm_state_clustering(
        features=["speed", "elongation"],
        binary_features_to_group=[],
        output_dir=output_dir,
        cell_type="tcell",
        n_states=2,
        random_state=19,
        df_positions=df_positions,
        verbose=False,
    )
    assert not stale.exists(), "rerunning HMM clustering must clear stale downstream outputs"


def test_run_hmm_state_clustering_clear_outputs_false_keeps_stale_files(tmp_path):
    df_positions = _make_positions_df()
    output_dir = Path(tmp_path) / "hmm_no_clear"

    run_hmm_state_clustering(
        features=["speed", "elongation"],
        binary_features_to_group=[],
        output_dir=output_dir,
        cell_type="tcell",
        n_states=2,
        random_state=19,
        df_positions=df_positions,
        verbose=False,
    )
    state_paths = _resolve_state_paths(output_dir, "tcell")
    stale = _plant_stale_file(state_paths.state_outdir)

    run_hmm_state_clustering(
        features=["speed", "elongation"],
        binary_features_to_group=[],
        output_dir=output_dir,
        cell_type="tcell",
        n_states=2,
        random_state=19,
        df_positions=df_positions,
        clear_outputs=False,
        verbose=False,
    )
    assert stale.exists(), "clear_outputs=False must be a real opt-out"


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


def _run_bouts_clustering(output_dir, states_path):
    return run_state_based_analysis(
        output_dir=output_dir,
        cell_type="tcell",
        adata_full_path=states_path,
        state_col=FULL_STATE_COL,
        behavioral_trajectory_size=50,
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


def test_run_state_based_analysis_clear_outputs_removes_stale_files(tmp_path):
    output_dir = Path(tmp_path) / "bouts_clear"
    state_dir = output_dir / "analysis" / "tcell" / "behavioral_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    states_path = state_dir / "BEHAV3D_tcell_behavioral_states.h5ad"
    _make_behavioral_states_adata().write(states_path, compression="gzip")

    _run_bouts_clustering(output_dir, states_path)
    paths = _resolve_track_paths(output_dir, "tcell")
    stale = paths.example_tracks_outfolder / "per_cluster" / "pdf" / "example_track_cluster_stale.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")
    assert stale.exists()

    _run_bouts_clustering(output_dir, states_path)
    assert not stale.exists(), "rerunning bouts trajectory clustering must clear stale outputs"


def test_run_state_based_analysis_clear_outputs_false_keeps_stale_files(tmp_path):
    output_dir = Path(tmp_path) / "bouts_no_clear"
    state_dir = output_dir / "analysis" / "tcell" / "behavioral_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    states_path = state_dir / "BEHAV3D_tcell_behavioral_states.h5ad"
    _make_behavioral_states_adata().write(states_path, compression="gzip")

    _run_bouts_clustering(output_dir, states_path)
    paths = _resolve_track_paths(output_dir, "tcell")
    stale = paths.example_tracks_outfolder / "per_cluster" / "pdf" / "example_track_cluster_stale.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")

    run_state_based_analysis(
        output_dir=output_dir,
        cell_type="tcell",
        adata_full_path=states_path,
        state_col=FULL_STATE_COL,
        behavioral_trajectory_size=50,
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
        clear_outputs=False,
        verbose=False,
    )
    assert stale.exists(), "clear_outputs=False must be a real opt-out"


def _make_cluster_umap_inputs(n_tracks=6):
    df_tracks_summarized = pd.DataFrame(
        {
            "sample_name": [f"sample_{i % 2}" for i in range(n_tracks)],
            "TrackID": list(range(n_tracks)),
            "track_length": [10] * n_tracks,
            "speed": np.random.rand(n_tracks),
        }
    )
    umap_embedding = pd.DataFrame(
        {
            "TrackID": list(range(n_tracks)),
            "UMAP1": np.random.rand(n_tracks),
            "UMAP2": np.random.rand(n_tracks),
        }
    )
    df_tracks = pd.DataFrame(
        {
            "sample_name": [f"sample_{i % 2}" for i in range(n_tracks) for _ in range(10)],
            "TrackID": [i for i in range(n_tracks) for _ in range(10)],
            "relative_time": list(range(10)) * n_tracks,
            "speed": np.random.rand(n_tracks * 10),
        }
    )
    return df_tracks, df_tracks_summarized, umap_embedding


def test_cluster_umap_clear_outputs_removes_stale_files(tmp_path):
    output_dir = Path(tmp_path) / "feature_dtw_clear"
    df_tracks, df_tracks_summarized, umap_embedding = _make_cluster_umap_inputs()

    cluster_umap(
        umap_embedding=umap_embedding,
        nr_of_clusters=2,
        df_tracks=df_tracks,
        df_tracks_summarized=df_tracks_summarized,
        random_state=0,
        output_dir=output_dir,
        cell_type="tcell",
        plot_results=False,
        plot_feature_cols=["speed"],
    )
    results_outdir = Path(output_dir, "analysis", "tcell", "results")
    stale = results_outdir / "stale_old_report.pdf"
    stale.write_text("stale")
    assert stale.exists()

    cluster_umap(
        umap_embedding=umap_embedding,
        nr_of_clusters=2,
        df_tracks=df_tracks,
        df_tracks_summarized=df_tracks_summarized,
        random_state=0,
        output_dir=output_dir,
        cell_type="tcell",
        plot_results=False,
        plot_feature_cols=["speed"],
    )
    assert not stale.exists(), "rerunning the legacy feature-DTW basis must clear stale outputs"


def test_cluster_umap_clear_outputs_false_keeps_stale_files(tmp_path):
    output_dir = Path(tmp_path) / "feature_dtw_no_clear"
    df_tracks, df_tracks_summarized, umap_embedding = _make_cluster_umap_inputs()

    cluster_umap(
        umap_embedding=umap_embedding,
        nr_of_clusters=2,
        df_tracks=df_tracks,
        df_tracks_summarized=df_tracks_summarized,
        random_state=0,
        output_dir=output_dir,
        cell_type="tcell",
        plot_results=False,
        plot_feature_cols=["speed"],
    )
    results_outdir = Path(output_dir, "analysis", "tcell", "results")
    stale = results_outdir / "stale_old_report.pdf"
    stale.write_text("stale")

    cluster_umap(
        umap_embedding=umap_embedding,
        nr_of_clusters=2,
        df_tracks=df_tracks,
        df_tracks_summarized=df_tracks_summarized,
        random_state=0,
        output_dir=output_dir,
        cell_type="tcell",
        plot_results=False,
        plot_feature_cols=["speed"],
        clear_outputs=False,
    )
    assert stale.exists(), "clear_outputs=False must be a real opt-out"

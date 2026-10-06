"""
Tests for the consolidated "Use features used in behavioral state clustering"
checkbox (Feature-based Track Classification): checking it must now reproduce
what the removed State-based "dtw features" Basis used to do -- DTW over the
behavioral-state model's own numeric features, reusing its *exact* saved
preprocessing/scaler (resolve_state_feature_matrix's "state_model_preprocessing"
path), rather than rebuilding a fresh adata from the filtered track-features
CSV and re-z-scoring it independently.

Two halves, since each proves only part of the guarantee:
  (a) backend numerical reproduction -- no Qt, fast/deterministic.
  (b) widget dispatch routing -- proves the checkbox actually reaches that
      backend call the way (a) exercises it, and that the checkbox/frame are
      hidden and force-unchecked once behavioral states are unavailable.

Runs under pytest *or* standalone: python test/test_track_state_features_preset.py
Requires the `behav3d` conda env (PyQt5/qtpy, anndata). No pytest-qt / .exec_()
needed -- mirrors test_assistant.py's QApplication.instance() or QApplication([])
pattern (see test_background_operation_teardown.py for the same convention).
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd

from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.track.dtw import resolve_state_feature_matrix
from behav3d.analysis.behavior.track.state_dtw import (
    run_categorical_dtaidistance_trajectory_clustering,
)

_STATE_CYCLE = ["move", "rest", "contact"]
_CONT_COLS = ["speed", "elongation"]
_BINARY_COL = "target_contact"


def _make_state_adata_with_scaler(n_tracks=6, track_len=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for track_id in range(n_tracks):
        sample_name = f"sample_{track_id % 2}"
        for t in range(track_len):
            rows.append({
                "sample_name": sample_name,
                "TrackID": track_id,
                "position_t": t,
                "position_x": float(t + 0.1 * track_id),
                "position_y": float(np.sin(t / 5.0) + 0.2 * track_id),
                "position_z": 0.0,
                FULL_STATE_COL: _STATE_CYCLE[(t + track_id) % len(_STATE_CYCLE)],
                _BINARY_COL: int((t + track_id) % 5 == 0),
            })
            X.append([
                5.0 + track_id + rng.normal(scale=0.5),
                1.0 + 0.1 * track_id + rng.normal(scale=0.05),
            ])
    obs = pd.DataFrame(rows).reset_index(drop=True)
    obs[FULL_STATE_COL] = obs[FULL_STATE_COL].astype("category")
    X = np.asarray(X, dtype=float)
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(_CONT_COLS, name="feature")),
    )
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    adata.uns["preprocessing"] = {
        "continuous_feature_cols": list(_CONT_COLS),
        "binary_cols_to_merge": [_BINARY_COL],
        # Deliberately offset from X's own mean/std -- a fake "saved HMM
        # scaler" -- so reuse vs. a fresh z-score produce visibly different
        # numbers, making the "was the literal scaler reused" assertion
        # meaningful rather than coincidentally true either way.
        "scaler": {"mean": (mean + 1.0).tolist(), "scale": (scale * 2.0).tolist()},
    }
    return adata


# ─────────────────────────────────────────────────────────────────────────
# (a) Backend: sequence_source="features" against the state h5ad reuses the
#     literal saved scaler, not a fresh z-score.
# ─────────────────────────────────────────────────────────────────────────

def test_state_feature_preset_reuses_literal_hmm_scaler_not_fresh_zscore(tmp_path):
    adata = _make_state_adata_with_scaler()
    states_path = tmp_path / "states.h5ad"
    adata.write(states_path, compression="gzip")

    # This is the exact call both the removed State-based "dtw features"
    # Basis and the new Feature-based checkbox path now make: sequence_source
    # ="features", no feature_cols/binary_cols override (defaults from
    # uns["preprocessing"]).
    result = run_categorical_dtaidistance_trajectory_clustering(
        output_dir=str(tmp_path),
        cell_type="tcell",
        adata_full_path=str(states_path),
        sequence_source="features",
        behavioral_trajectory_size=None,
        n_clusters=2,
        clustering_method="agglomerative",
        parallel=False,
        save_outputs=False,
        clear_outputs=False,
        plot_results=False,
        verbose=False,
        random_state=7,
    )
    meta = result.uns["dtai_trajectory_clustering"]
    assert meta["has_behavioral_states"] is True
    assert meta["feature_preprocessing"] == "state_model_preprocessing"
    assert sorted(meta["continuous_feature_cols"]) == sorted(_CONT_COLS)
    assert meta["binary_feature_cols"] == [_BINARY_COL]

    # Direct check that the reused scaler is the literal saved one.
    matrix_saved, _, _, source_saved = resolve_state_feature_matrix(adata)
    assert source_saved == "state_model_preprocessing"
    pre = adata.uns["preprocessing"]
    expected_cont = (
        adata[:, _CONT_COLS].X - np.asarray(pre["scaler"]["mean"])
    ) / np.asarray(pre["scaler"]["scale"])
    np.testing.assert_allclose(matrix_saved[:, :len(_CONT_COLS)], expected_cont, atol=1e-8)

    # Contrast: without a usable saved scaler, resolve_state_feature_matrix
    # falls back to a fresh z-score -- materially different numbers -- so the
    # reuse path above isn't just always producing the same matrix regardless.
    adata_no_scaler = adata.copy()
    del adata_no_scaler.uns["preprocessing"]["scaler"]
    matrix_zscore, _, _, source_zscore = resolve_state_feature_matrix(adata_no_scaler)
    assert source_zscore == "zscore"
    assert not np.allclose(
        matrix_saved[:, :len(_CONT_COLS)], matrix_zscore[:, :len(_CONT_COLS)], atol=1e-3
    )


# ─────────────────────────────────────────────────────────────────────────
# (b) Widget: the checkbox actually dispatches through that same call
#     pattern, and is hidden/force-unchecked when states are unavailable.
# ─────────────────────────────────────────────────────────────────────────

_APP = None


def _qt_app():
    global _APP
    # Priming behav3d.napari._analysis first resolves a circular import
    # between it and _single_cell (SingleCellTab <-> CollapsibleSection/
    # DualListGroupSelector) that otherwise breaks a standalone import of
    # TrackClassificationSubTab.
    import behav3d.napari._analysis  # noqa: F401
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump_until(condition, timeout_s=5.0):
    app = _qt_app()
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return condition()


def _make_tab(tmp_path, cell_type="tcell"):
    _qt_app()
    from behav3d.napari._single_cell import TrackClassificationSubTab
    fake_loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={})
    return TrackClassificationSubTab(
        viewer=None, metadata_loader=fake_loader, cell_type_getter=lambda: cell_type
    )


def _write_states_h5ad(tmp_path, cell_type="tcell"):
    adata = _make_state_adata_with_scaler()
    states_path = (
        tmp_path / "analysis" / cell_type / "behavioral_states"
        / f"BEHAV3D_{cell_type}_behavioral_states.h5ad"
    )
    states_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(states_path, compression="gzip")
    return states_path


def _write_filtered_tracks_csv(tmp_path, cell_type="tcell", n_tracks=4, track_len=20):
    """Minimal filtered track-features CSV so TrajectoryFeatureSelector.populate()
    has something to discover a real "speed" candidate from -- without this,
    the Feature-based (unchecked) path has no features to validate/select."""
    rng = np.random.default_rng(1)
    rows = []
    for track_id in range(n_tracks):
        for t in range(track_len):
            rows.append({
                "sample_name": f"sample_{track_id % 2}",
                "TrackID": track_id,
                "position_t": t,
                "position_x": float(t),
                "position_y": float(t),
                "position_z": 0.0,
                "speed": float(rng.normal(loc=5.0 + track_id, scale=0.5)),
            })
    csv_path = (
        tmp_path / "analysis" / cell_type / "track_features"
        / f"BEHAV3D_{cell_type}_combined_track_features_filtered.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_checkbox_checked_dispatches_state_h5ad_call_with_no_csv_rebuild(tmp_path):
    import behav3d.analysis.behavior.track.state_dtw as state_dtw_mod
    import behav3d.analysis.behavior.track.trajectory_features as traj_feat_mod

    _write_states_h5ad(tmp_path)
    tab = _make_tab(tmp_path)

    fake_result = ad.AnnData(X=np.zeros((1, 1)))
    fake_result.uns["dtai_trajectory_clustering"] = {"has_behavioral_states": False}
    mock_cluster = MagicMock(return_value=fake_result)
    mock_build = MagicMock(side_effect=AssertionError("should not build a features-only adata"))
    orig_cluster = state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering
    orig_build = traj_feat_mod.build_trajectory_feature_adata
    state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering = mock_cluster
    traj_feat_mod.build_trajectory_feature_adata = mock_build
    try:
        tab.combo_clustering_family.setCurrentIndex(
            tab.combo_clustering_family.findData("feature_based")
        )
        tab._apply_clustering_family_mode("feature_based")
        assert tab._behavioral_states_available() is True
        assert not tab._state_features_preset_frame.isHidden()

        tab.chk_use_state_features_preset.setChecked(True)
        tab._dispatch_track_cluster("tcell")
        assert _pump_until(lambda: mock_cluster.called)

        kwargs = mock_cluster.call_args.kwargs
        assert kwargs.get("sequence_source") == "features"
        assert "adata_full_path" not in kwargs
        mock_build.assert_not_called()
    finally:
        state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering = orig_cluster
        traj_feat_mod.build_trajectory_feature_adata = orig_build


def test_checkbox_unchecked_still_builds_features_only_adata(tmp_path):
    import behav3d.analysis.behavior.track.state_dtw as state_dtw_mod
    import behav3d.analysis.behavior.track.trajectory_features as traj_feat_mod

    _write_states_h5ad(tmp_path)
    _write_filtered_tracks_csv(tmp_path)
    tab = _make_tab(tmp_path)

    fake_result = ad.AnnData(X=np.zeros((1, 1)))
    fake_result.uns["dtai_trajectory_clustering"] = {"has_behavioral_states": False}
    mock_cluster = MagicMock(return_value=fake_result)
    mock_build = MagicMock(return_value=None)
    orig_cluster = state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering
    orig_build = traj_feat_mod.build_trajectory_feature_adata
    state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering = mock_cluster
    traj_feat_mod.build_trajectory_feature_adata = mock_build
    try:
        tab.combo_clustering_family.setCurrentIndex(
            tab.combo_clustering_family.findData("feature_based")
        )
        tab._apply_clustering_family_mode("feature_based")
        tab.chk_use_state_features_preset.setChecked(False)
        # populate() ran automatically above (CSV now exists). Its column scan
        # runs on a worker thread, so wait for the checkboxes to exist: then
        # "speed" is a real checkbox and set_params can actually select it.
        assert _pump_until(lambda: not tab.traj_feature_selector.is_populating())
        tab.traj_feature_selector.set_params({"features": ["speed"]})
        assert "speed" in tab.traj_feature_selector.params()["features"]

        tab._dispatch_track_cluster("tcell")
        assert _pump_until(lambda: mock_cluster.called)

        mock_build.assert_called_once()
        kwargs = mock_cluster.call_args.kwargs
        assert kwargs.get("sequence_source") == "features"
        assert "adata_full_path" in kwargs
        assert kwargs["adata_full_path"].endswith(".h5ad")
    finally:
        state_dtw_mod.run_categorical_dtaidistance_trajectory_clustering = orig_cluster
        traj_feat_mod.build_trajectory_feature_adata = orig_build


def test_states_unavailable_hides_and_force_unchecks_preset(tmp_path):
    # No states h5ad written for this cell type at all.
    tab = _make_tab(tmp_path, cell_type="other_ct")

    tab.combo_clustering_family.setCurrentIndex(
        tab.combo_clustering_family.findData("feature_based")
    )
    tab._apply_clustering_family_mode("feature_based")

    assert tab._behavioral_states_available() is False
    assert tab._state_features_preset_frame.isHidden() is True
    assert tab.chk_use_state_features_preset.isChecked() is False


def test_states_disappearing_after_being_checked_force_unchecks():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        states_path = _write_states_h5ad(tmp_path)
        tab = _make_tab(tmp_path)

        tab.combo_clustering_family.setCurrentIndex(
            tab.combo_clustering_family.findData("feature_based")
        )
        tab._apply_clustering_family_mode("feature_based")
        tab.chk_use_state_features_preset.setChecked(True)
        assert not tab._state_features_preset_frame.isHidden()

        states_path.unlink()
        assert tab._behavioral_states_available() is False

        tab._apply_clustering_family_mode("feature_based")

        assert tab._state_features_preset_frame.isHidden() is True
        assert tab.chk_use_state_features_preset.isChecked() is False


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        with tempfile.TemporaryDirectory() as td:
            if fn is test_states_disappearing_after_being_checked_force_unchecks:
                fn()
            else:
                fn(Path(td))
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")

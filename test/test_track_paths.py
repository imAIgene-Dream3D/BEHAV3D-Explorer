import inspect
from pathlib import Path

import pytest

from behav3d.analysis.behavior.track.utils import (
    TRACK_TRAJECTORIES_SUBDIR_NAME,
    _LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME,
    _peek_track_outfolder,
    _resolve_track_paths,
)


def test_resolve_track_paths_is_method_agnostic(tmp_path):
    """Calling the shared resolver twice (as DTW vs. bouts would) with the same
    output_dir/cell_type and no override must resolve to the identical folder
    layout — this is the core invariant the unification relies on."""
    paths_a = _resolve_track_paths(tmp_path, "tcell")
    paths_b = _resolve_track_paths(tmp_path, "tcell")

    assert paths_a.outfolder == paths_b.outfolder
    assert paths_a.clustering_outfolder == paths_b.clustering_outfolder
    assert paths_a.example_tracks_outfolder == paths_b.example_tracks_outfolder
    assert (
        paths_a.example_tracks_backprojection_outfolder
        == paths_b.example_tracks_backprojection_outfolder
    )
    assert paths_a.classification_outfolder == paths_b.classification_outfolder


def test_resolve_track_paths_uses_canonical_spelling_for_fresh_projects(tmp_path):
    paths = _resolve_track_paths(tmp_path, "tcell")
    assert paths.outfolder.name == TRACK_TRAJECTORIES_SUBDIR_NAME == "behavioral_trajectories"
    assert paths.outfolder.exists()


def test_resolve_track_paths_falls_back_to_legacy_misspelled_folder(tmp_path):
    """An older project that only has the misspelled folder on disk must still
    resolve to it, without any renaming/migration — no re-run required."""
    legacy = tmp_path / "analysis" / "tcell" / _LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("pre-existing project data")

    paths = _resolve_track_paths(tmp_path, "tcell")

    assert paths.outfolder == legacy
    assert (paths.outfolder / "marker.txt").exists()


def test_resolve_track_paths_prefers_canonical_when_both_exist(tmp_path):
    analysis_outdir = tmp_path / "analysis" / "tcell"
    legacy = analysis_outdir / _LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME
    canonical = analysis_outdir / TRACK_TRAJECTORIES_SUBDIR_NAME
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)

    paths = _resolve_track_paths(tmp_path, "tcell")

    assert paths.outfolder == canonical


def test_peek_track_outfolder_matches_resolve_but_creates_nothing(tmp_path):
    """UI existence checks (autofill, view-button enablement) must be able to
    call this on every refresh without creating directories as a side effect."""
    peeked = _peek_track_outfolder(tmp_path, "tcell")
    assert peeked == tmp_path / "analysis" / "tcell" / "behavioral_trajectories"
    assert not peeked.exists()
    assert not (tmp_path / "analysis").exists()

    resolved = _resolve_track_paths(tmp_path, "tcell")
    assert peeked == resolved.outfolder


def test_explicit_output_subdir_name_bypasses_fallback(tmp_path):
    """Callers that need a nested/different subfolder (e.g. the legacy
    original-BEHAV3D method) must not trigger the typo fallback."""
    legacy = tmp_path / "analysis" / "tcell" / _LEGACY_TRACK_TRAJECTORIES_SUBDIR_NAME
    legacy.mkdir(parents=True)

    paths = _resolve_track_paths(tmp_path, "tcell", output_subdir_name="some_other_subdir")

    assert paths.outfolder == tmp_path / "analysis" / "tcell" / "some_other_subdir"


def test_example_tracks_backprojection_is_nested():
    paths = _resolve_track_paths(Path("/tmp/does-not-need-to-exist"), "tcell")
    assert (
        paths.example_tracks_backprojection_outfolder
        == paths.example_tracks_outfolder / "backprojection"
    )


@pytest.mark.parametrize(
    "func_path",
    [
        "behav3d.analysis.behavior.track.bouts:apply_track_classifier_to_subtracks",
        "behav3d.analysis.behavior.track.bouts:run_state_based_analysis",
        "behav3d.analysis.behavior.track.bouts:train_track_classifier",
        "behav3d.analysis.behavior.track.state_dtw:run_categorical_dtaidistance_trajectory_clustering",
    ],
)
def test_track_methods_default_to_the_shared_canonical_folder(func_path):
    """Regression test for the confirmed napari "Apply classifier" bug: every
    track-classification entry point must defer to the shared resolver's own
    canonical default (output_subdir_name=None) rather than hardcoding its own
    folder name — two divergent hardcoded defaults is exactly what caused
    classifier-apply to silently write into a different folder than clustering.
    """
    module_name, func_name = func_path.split(":")
    module = __import__(module_name, fromlist=[func_name])
    func = getattr(module, func_name)
    sig = inspect.signature(func)
    assert sig.parameters["output_subdir_name"].default is None


def test_legacy_feature_dtw_internal_staging_default_is_unchanged():
    """The legacy method's own 'results' staging-dir default is a distinct,
    internal concept from the canonical outfolder and is intentionally left
    untouched by the unification."""
    from behav3d.analysis.behavior.track.feature_dtw import run_tcell_analysis

    sig = inspect.signature(run_tcell_analysis)
    assert sig.parameters["output_subdir_name"].default == "results"

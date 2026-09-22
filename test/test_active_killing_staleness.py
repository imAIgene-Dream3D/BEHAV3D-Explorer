"""
Tests for staleness detection across Feature Extraction -> Active Killing -> Filtering.

If Feature Extraction is rerun for a cell type, any previously-produced Active Killing
or Filtering output for that cell type was computed from track features that no longer
exist. See [[test_contact_group_analysis]] for the same "stale CSV" hard-error convention
already used for contact_grouping.py.

Active Killing always reads the RAW (unfiltered) combined_track_features.csv, never the
filtered one -- see the comment in run_active_killing_analysis(). Filtering, in turn,
reads Active Killing's advanced-features CSV as its own input when one exists (see
find_advanced_features_csv / filter_tracks' df_input_path), so Active Killing preferring
the filtered CSV would make the two steps each other's upstream dependency: a staleness
check on either side could deadlock, with Filtering refusing to run because Active
Killing looks stale and Active Killing refusing to run because Filtering looks stale.
Always starting Active Killing from raw breaks that cycle; instead, Active Killing warns
the caller (print + stats["filtering_needs_rerun_for"]) that any existing filtered CSV is
now behind and Filtering should be re-run to pick up its output.

Since the death-event rework, Active Killing also reads the dead-mask and tracked-label
zarrs and caches death events per target type. Those add upstream edges but no cycle
(nothing downstream writes them): an advanced CSV older than a mask it was built from is
stale, and so is one produced by the previous algorithm (schema_version < 2), whose
columns and density-inflated counts must never be consumed as if current.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from behav3d.features.advanced_timepoint_features import (
    ACTIVE_KILLING_SCHEMA_VERSION,
    RUN_PARAMS_FILENAME,
    ActiveKillingInputError,
    StaleDataError,
    find_advanced_features_csv,
    run_active_killing_analysis,
)
from behav3d.features.timepoint_features import run_feature_extraction
from behav3d.io.formats.zarr import save_as_zarr


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TrackID": [0]}).to_csv(path, index=False)
    os.utime(path, (mtime, mtime))


def _raw_csv_path(output_dir, cell_type):
    return output_dir / "analysis" / cell_type / "track_features" / f"BEHAV3D_{cell_type}_combined_track_features.csv"


def _filtered_csv_path(output_dir, cell_type):
    return output_dir / "analysis" / cell_type / "track_features" / f"BEHAV3D_{cell_type}_combined_track_features_filtered.csv"


def _advanced_csv_path(output_dir, cell_type, subfolder="organoid"):
    return output_dir / "analysis" / cell_type / "active_killing" / subfolder / f"BEHAV3D_{cell_type}_advanced_track_features.csv"


def _write_run_params(advanced_path, mtime, schema=ACTIVE_KILLING_SCHEMA_VERSION, upstream=()):
    p = advanced_path.parent / RUN_PARAMS_FILENAME
    p.write_text(json.dumps({"schema_version": schema, "upstream_paths": [str(u) for u in upstream]}))
    os.utime(p, (mtime, mtime))


def _track_csv(path, mtime, touching="5", extra=None):
    """Minimal effector track table: one track touching organoid 5 at t=0..2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sample_name": ["s1"] * 3, "TrackID": [1] * 3, "position_t": [0, 1, 2],
                       "touching_organoids": [touching] * 3})
    if extra:
        df = df.assign(**extra)
    df.to_csv(path, index=False)
    os.utime(path, (mtime, mtime))


def _project(tmp_path):
    """A tiny real project: metadata + organoid/effector tracked zarrs + a dead mask."""
    T, Z, Y, X = 3, 6, 16, 16
    img = tmp_path / "images" / "s1"
    img.mkdir(parents=True)
    labels = np.zeros((T, Z, Y, X), np.uint16)
    labels[:, 1:5, 3:12, 3:12] = 5
    immune = np.zeros_like(labels)
    immune[:, 1:4, 12:15, 5:8] = 1
    dead = np.zeros_like(labels)
    paths = {"org": img / "s1_organoid_tracked.zarr", "imm": img / "s1_tcell_tracked.zarr",
             "dead": img / "s1_mask_dead.zarr"}
    save_as_zarr(labels, paths["org"])
    save_as_zarr(immune, paths["imm"])
    save_as_zarr(dead, paths["dead"])
    md = pd.DataFrame([{
        "sample_name": "s1", "pixel_distance_xy": 1.0, "pixel_distance_z": 1.0, "distance_unit": "um",
        "time_interval": 2.0, "time_unit": "m", "dead_mask_path": str(paths["dead"]),
        "or_organoid_tracks_image_path": str(paths["org"]), "im_tcell_tracks_image_path": str(paths["imm"]),
    }])
    return md, paths


class TestFindAdvancedFeaturesCsv:
    def test_raises_when_advanced_csv_predates_raw(self, tmp_path):
        """Feature Extraction rerun after Active Killing must not let Filtering silently
        pick up the now-stale advanced-features CSV."""
        now = time.time()
        advanced = _advanced_csv_path(tmp_path, "tcell")
        _touch(advanced, now - 100)
        _write_run_params(advanced, now - 100)
        _touch(_raw_csv_path(tmp_path, "tcell"), now)

        with pytest.raises(StaleDataError, match="Feature Extraction was rerun"):
            find_advanced_features_csv(tmp_path, "tcell")

    def test_returns_path_when_advanced_csv_is_newer(self, tmp_path):
        now = time.time()
        advanced = _advanced_csv_path(tmp_path, "tcell")
        _touch(_raw_csv_path(tmp_path, "tcell"), now - 100)
        _touch(advanced, now)
        _write_run_params(advanced, now)

        assert find_advanced_features_csv(tmp_path, "tcell") == advanced

    def test_returns_path_when_no_raw_csv_exists_yet(self, tmp_path):
        advanced = _advanced_csv_path(tmp_path, "tcell")
        _touch(advanced, time.time())
        _write_run_params(advanced, time.time())

        assert find_advanced_features_csv(tmp_path, "tcell") == advanced

    def test_returns_none_when_no_active_killing_dir(self, tmp_path):
        assert find_advanced_features_csv(tmp_path, "tcell") is None

    def test_rejects_output_of_previous_algorithm(self, tmp_path):
        """A pre-rework advanced CSV has no run-params file (or schema 1): its columns
        and density-inflated counts must not be consumed as if they were current."""
        advanced = _advanced_csv_path(tmp_path, "tcell")
        _touch(advanced, time.time())
        with pytest.raises(StaleDataError, match="previous Active Killing algorithm"):
            find_advanced_features_csv(tmp_path, "tcell")
        _write_run_params(advanced, time.time(), schema=1)
        with pytest.raises(StaleDataError, match="previous Active Killing algorithm"):
            find_advanced_features_csv(tmp_path, "tcell")

    def test_raises_when_a_mask_was_regenerated_after_active_killing(self, tmp_path):
        now = time.time()
        advanced = _advanced_csv_path(tmp_path, "tcell")
        mask = tmp_path / "images" / "s1" / "s1_mask_dead.zarr"
        mask.mkdir(parents=True)
        os.utime(mask, (now, now))
        _touch(advanced, now - 100)
        _write_run_params(advanced, now - 100, upstream=[mask])
        with pytest.raises(StaleDataError, match="Segmentation / Tracking"):
            find_advanced_features_csv(tmp_path, "tcell")


class TestRunActiveKillingAnalysisAlwaysUsesRaw:
    """Active Killing must never read the filtered CSV -- staleness of the
    filtered CSV relative to raw (in either direction) is irrelevant to it
    and must never raise or change which file gets loaded."""

    @pytest.mark.parametrize("filtered_offset", [-100, +100])
    def test_immune_raw_used_regardless_of_filtered_mtime(self, tmp_path, capsys, filtered_offset):
        md, _ = _project(tmp_path)
        now = time.time()
        raw = _raw_csv_path(tmp_path, "tcell")
        _track_csv(raw, now)
        # The filtered CSV deliberately carries a different, contact-free table:
        # if it were ever read, the run would fail its contact validation.
        _track_csv(_filtered_csv_path(tmp_path, "tcell"), now + filtered_offset, touching="")

        run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                    target_cell_types=["organoid"], save_results=False)

        assert f"Loading effector tracks from {raw}" in capsys.readouterr().out

    def test_immune_filenotfound_when_only_filtered_exists(self, tmp_path):
        """No raw CSV means Active Killing has nothing to read, even though a
        filtered CSV is sitting right there -- it is never used as a fallback."""
        md, _ = _project(tmp_path)
        _track_csv(_filtered_csv_path(tmp_path, "tcell"), time.time())
        with pytest.raises(FileNotFoundError):
            run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                        target_cell_types=["organoid"], save_results=False)

    def test_immune_filenotfound_when_no_tracks_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_active_killing_analysis(metadata=pd.DataFrame(), output_dir=tmp_path, immune_cell_type="tcell",
                                        target_cell_types=["organoid"], save_results=False)


class TestRefusesInputsThatWouldSilentlyYieldZero:
    def test_missing_contact_columns_raise(self, tmp_path):
        md, _ = _project(tmp_path)
        p = _raw_csv_path(tmp_path, "tcell")
        p.parent.mkdir(parents=True)
        pd.DataFrame({"sample_name": ["s1"], "TrackID": [1], "position_t": [0]}).to_csv(p, index=False)
        with pytest.raises(ActiveKillingInputError, match="Enable 'contact'"):
            run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                        target_cell_types=["organoid"], save_results=False)

    def test_no_contact_anywhere_raises(self, tmp_path):
        md, _ = _project(tmp_path)
        _track_csv(_raw_csv_path(tmp_path, "tcell"), time.time(), touching="")
        with pytest.raises(ActiveKillingInputError, match="Contact Threshold"):
            run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                        target_cell_types=["organoid"], save_results=False)

    def test_missing_dead_mask_raises_instead_of_rethresholding(self, tmp_path):
        md, paths = _project(tmp_path)
        _track_csv(_raw_csv_path(tmp_path, "tcell"), time.time())
        import shutil
        shutil.rmtree(paths["dead"])
        with pytest.raises(FileNotFoundError, match="dead-mask segmentation"):
            run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                        target_cell_types=["organoid"], save_results=False)


class TestRunActiveKillingAnalysisRerunFilteringWarning:
    """After a run, Active Killing should tell the caller which cell types'
    filtered CSVs are now behind, both via a printed warning and via
    stats["filtering_needs_rerun_for"], so the GUI can surface a popup."""

    def test_warns_and_reports_stats_when_filtered_csv_exists(self, tmp_path, capsys):
        md, _ = _project(tmp_path)
        now = time.time()
        _track_csv(_raw_csv_path(tmp_path, "tcell"), now)
        _track_csv(_filtered_csv_path(tmp_path, "tcell"), now)

        _, _, stats = run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                                  target_cell_types=["organoid"], save_results=True)

        assert stats["filtering_needs_rerun_for"] == ["tcell"]
        out = capsys.readouterr().out
        assert "Re-run Filtering" in out and "tcell" in out

    def test_no_warning_when_no_filtered_csv_exists(self, tmp_path, capsys):
        md, _ = _project(tmp_path)
        _track_csv(_raw_csv_path(tmp_path, "tcell"), time.time())

        _, _, stats = run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                                  target_cell_types=["organoid"], save_results=True)

        assert stats["filtering_needs_rerun_for"] == []
        assert "Re-run Filtering" not in capsys.readouterr().out

    def test_fresh_output_is_findable_and_carries_schema(self, tmp_path):
        md, _ = _project(tmp_path)
        _track_csv(_raw_csv_path(tmp_path, "tcell"), time.time() - 10)
        run_active_killing_analysis(metadata=md, output_dir=tmp_path, immune_cell_type="tcell",
                                    target_cell_types=["organoid"], save_results=True, output_subfolder="organoid")
        found = find_advanced_features_csv(tmp_path, "tcell")
        assert found == _advanced_csv_path(tmp_path, "tcell")
        rp = json.loads((found.parent / RUN_PARAMS_FILENAME).read_text())
        assert rp["schema_version"] == ACTIVE_KILLING_SCHEMA_VERSION


class TestRunFeatureExtractionProactiveWarning:
    def test_warns_when_filtered_and_active_killing_outputs_already_exist(self, tmp_path, capsys):
        now = time.time()
        _touch(_filtered_csv_path(tmp_path, "tcell"), now - 100)
        _touch(_advanced_csv_path(tmp_path, "tcell"), now - 100)

        run_feature_extraction(metadata=pd.DataFrame(columns=["sample_name"]), output_dir=tmp_path, cell_type="tcell")

        out = capsys.readouterr().out
        assert "WARNING: Feature Extraction for 'tcell' was just rerun" in out
        assert "Filtering" in out
        assert "Active Killing" in out

    def test_no_warning_when_no_downstream_output_exists(self, tmp_path, capsys):
        run_feature_extraction(metadata=pd.DataFrame(columns=["sample_name"]), output_dir=tmp_path, cell_type="tcell")

        out = capsys.readouterr().out
        assert "WARNING: Feature Extraction" not in out

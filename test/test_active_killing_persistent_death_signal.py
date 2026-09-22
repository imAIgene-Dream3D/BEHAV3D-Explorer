"""
Tests for the ``persistent_death_signal`` option on Active Killing Analysis.

Segmentation of the dead mask is sometimes unstable: a target's dead-mask segment can be
included in one frame and dropped in the next, making the raw death signal dip and recover
instead of only rising as the cell actually dies. ``persistent_death_signal`` (default True)
reads each target's death signal as a running (cumulative) maximum over its own track so a
dip like this can't be misread as the target "un-dying". See
[[test_active_killing_contact_column]] for the broader Active Killing test conventions this
file follows.
"""
import os
import time

import pandas as pd

from behav3d.features.advanced_timepoint_features import (
    analyze_active_killing_per_timepoint,
    run_active_killing_analysis,
)

# The user's own example: a real rise (0 -> 200) followed by a segmentation-driven dip
# (200 -> 150 -> 100) that recovers (-> 200 -> 200 -> 250). Cumulative max reads this as
# 0,10,20,200,200,200,200,200,250.
_DIP_SERIES = [0, 10, 20, 200, 150, 100, 200, 200, 250]


def _target_df_with_dip():
    return pd.DataFrame({
        "sample_name": ["s1"] * len(_DIP_SERIES),
        "TrackID": [5] * len(_DIP_SERIES),
        "position_t": list(range(len(_DIP_SERIES))),
        "nr_dead_mask_pixels": _DIP_SERIES,
    })


def _immune_df():
    return pd.DataFrame({
        "sample_name": ["s1"],
        "TrackID": [0],
        "position_t": [0],
    })


def _contact_event(t_list):
    return pd.DataFrame([{
        "contact_event_id": 1,
        "sample_name": "s1",
        "immune_track_id": 0,
        "target_track_ids": "5",
        "contact_start_t": t_list[0],
        "contact_end_t": t_list[-1],
        "contact_duration": len(t_list),
        "contact_timepoints": t_list,
    }])


class TestAnalyzeActiveKillingPersistentDeathSignal:
    """t=4 -> t=5 (window=1) lands squarely inside the dip: raw death_at_start/end are
    150/100 (a spurious decrease), while the running max is flat at 200/200 -- the two
    modes disagree on the *sign* of death_signal_increase, isolating the fix."""

    def test_persistent_absorbs_the_dip(self):
        df_killing = analyze_active_killing_per_timepoint(
            df_immune_tracks=_immune_df(),
            df_target_tracks=_target_df_with_dip(),
            df_contact_events=_contact_event([4]),
            observation_window=1,
            death_signal_column="nr_dead_mask_pixels",
            absolute_killing_threshold=0.0,
            persistent_death_signal=True,
        )

        row = df_killing.iloc[0]
        assert row["death_signal_increase"] == 0.0
        assert row["is_active_killing"] == False

    def test_raw_signal_shows_a_spurious_decrease(self):
        df_killing = analyze_active_killing_per_timepoint(
            df_immune_tracks=_immune_df(),
            df_target_tracks=_target_df_with_dip(),
            df_contact_events=_contact_event([4]),
            observation_window=1,
            death_signal_column="nr_dead_mask_pixels",
            absolute_killing_threshold=0.0,
            persistent_death_signal=False,
        )

        row = df_killing.iloc[0]
        assert row["death_signal_increase"] == -50.0
        assert row["is_active_killing"] == False

    def test_default_is_persistent(self):
        """persistent_death_signal defaults to True when omitted."""
        df_killing = analyze_active_killing_per_timepoint(
            df_immune_tracks=_immune_df(),
            df_target_tracks=_target_df_with_dip(),
            df_contact_events=_contact_event([4]),
            observation_window=1,
            death_signal_column="nr_dead_mask_pixels",
            absolute_killing_threshold=0.0,
        )

        assert df_killing.iloc[0]["death_signal_increase"] == 0.0


def _touch_immune_csv(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_name": ["s1"] * 4,
        "TrackID": [0] * 4,
        "position_t": [0, 1, 2, 3],
        "organoid_contact": [True, True, True, True],
        "touching_organoids": ["5", "5", "5", "5"],
    }).to_csv(path, index=False)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _touch_organoid_csv(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_name": ["s1"] * 4,
        "TrackID": [5] * 4,
        "position_t": [0, 1, 2, 3],
        "mean_dead_dye": [10.0, 20.0, 15.0, 30.0],
    }).to_csv(path, index=False)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestRunActiveKillingAnalysisPersistentDeathSignal:
    """End-to-end: run_active_killing_analysis must forward persistent_death_signal and
    default to True, so the legacy ipywidgets panel (which never passes this kwarg)
    inherits the fix automatically."""

    def test_default_persistent_death_signal_is_true(self, tmp_path):
        now = time.time()
        _touch_immune_csv(
            tmp_path / "analysis" / "tcell" / "track_features" / "BEHAV3D_tcell_combined_track_features.csv", now,
        )
        _touch_organoid_csv(
            tmp_path / "analysis" / "organoid" / "track_features" / "BEHAV3D_organoid_combined_track_features.csv", now,
        )

        _, _, stats = run_active_killing_analysis(
            metadata=pd.DataFrame(),
            output_dir=tmp_path,
            immune_cell_type="tcell",
            target_cell_types=["organoid"],
            save_results=False,
        )

        assert stats["persistent_death_signal"] is True

    def test_persistent_death_signal_can_be_disabled(self, tmp_path):
        now = time.time()
        _touch_immune_csv(
            tmp_path / "analysis" / "tcell" / "track_features" / "BEHAV3D_tcell_combined_track_features.csv", now,
        )
        _touch_organoid_csv(
            tmp_path / "analysis" / "organoid" / "track_features" / "BEHAV3D_organoid_combined_track_features.csv", now,
        )

        _, _, stats = run_active_killing_analysis(
            metadata=pd.DataFrame(),
            output_dir=tmp_path,
            immune_cell_type="tcell",
            target_cell_types=["organoid"],
            persistent_death_signal=False,
            save_results=False,
        )

        assert stats["persistent_death_signal"] is False

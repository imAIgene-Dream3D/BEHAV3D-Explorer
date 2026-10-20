"""Regression test for the Sankey per-state-pair accumulation bug: rerunning
`save_state_transition_report` must clear `sankey_html/` and `sankey_pdf_pages/`
before writing, unconditionally, so files from a previous run with
renamed/removed states (or with the "include sankey pairs" toggle previously
on) never linger once the toggle is off or the state set has changed.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

from pathlib import Path

import anndata as ad
import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.state.visualization.plots.state_transitions import (
    save_state_transition_report,
)

_STATE_COL = "full_behavioral_state"


def _build_obs():
    return pd.DataFrame(
        {
            "sample_name": ["s1"] * 6,
            "TrackID": [1, 1, 1, 2, 2, 2],
            "position_t": [0, 1, 2, 0, 1, 2],
            _STATE_COL: ["zeta", "alpha", "mu", "zeta", "alpha", "mu"],
        }
    )


def test_save_state_transition_report_clears_stale_sankey_files_regardless_of_flag(tmp_path):
    obs = _build_obs()
    adata = ad.AnnData(obs=obs)

    stale_html = Path(tmp_path) / "sankey_html" / "sankey_OLDSTATE_to_OTHER.html"
    stale_pdf_page = Path(tmp_path) / "sankey_pdf_pages" / "sankey_OLDSTATE_to_OTHER.pdf"
    stale_html.parent.mkdir(parents=True, exist_ok=True)
    stale_pdf_page.parent.mkdir(parents=True, exist_ok=True)
    stale_html.write_text("stale")
    stale_pdf_page.write_text("stale")

    # include_sankey_pairs=False is the cheap, common case (no sankey rendering
    # happens at all) -- it must still clear leftovers from a prior run where
    # the flag was on or the state set has since changed/shrunk.
    save_state_transition_report(
        adata=adata,
        output_dir=tmp_path,
        state_col=_STATE_COL,
        id_cols=("sample_name", "TrackID"),
        time_col="position_t",
        include_ngram_rankings=False,
        include_sankey_pairs=False,
        verbose=False,
    )

    assert not stale_html.exists(), "stale sankey_html file must be cleared even when the flag is off"
    assert not stale_pdf_page.exists(), "stale sankey_pdf_pages file must be cleared even when the flag is off"


def _build_obs_for_states(states, n_tracks=6, track_len=9):
    rows = []
    for track_id in range(n_tracks):
        for t in range(track_len):
            rows.append(
                {
                    "sample_name": "s1",
                    "TrackID": track_id,
                    "position_t": t,
                    _STATE_COL: states[(t + track_id) % len(states)],
                }
            )
    return pd.DataFrame(rows)


def _sankey_pair_files(output_dir):
    files = []
    for sub in ("sankey_html", "sankey_pdf_pages"):
        d = Path(output_dir) / sub
        if d.exists():
            files.extend(p.name for p in d.glob("*"))
    return files


def test_save_state_transition_report_removes_stale_pairs_when_states_change(tmp_path):
    """End-to-end: after a first run over states {A,B,C} writes one Sankey file
    per state pair, rerunning over a different state set {A,D} (simulating a
    rename/reduction of states) must leave no file referencing the old B/C
    states -- only the state pairs than exist have to be described in files
    from the current run."""
    adata_abc = ad.AnnData(obs=_build_obs_for_states(["A", "B", "C"]))
    save_state_transition_report(
        adata=adata_abc,
        output_dir=tmp_path,
        state_col=_STATE_COL,
        id_cols=("sample_name", "TrackID"),
        time_col="position_t",
        include_ngram_rankings=False,
        include_sankey_pairs=True,
        sankey_min_count=1,
        verbose=False,
    )
    first_run_files = _sankey_pair_files(tmp_path)
    assert len(first_run_files) > 0
    assert any("_B_" in name or "_B." in name for name in first_run_files)
    assert any("_C_" in name or "_C." in name for name in first_run_files)

    adata_ad = ad.AnnData(obs=_build_obs_for_states(["A", "D"]))
    save_state_transition_report(
        adata=adata_ad,
        output_dir=tmp_path,
        state_col=_STATE_COL,
        id_cols=("sample_name", "TrackID"),
        time_col="position_t",
        include_ngram_rankings=False,
        include_sankey_pairs=True,
        sankey_min_count=1,
        verbose=False,
    )
    second_run_files = _sankey_pair_files(tmp_path)
    assert len(second_run_files) > 0
    assert not any("_B_" in name or "_B." in name for name in second_run_files), (
        "stale per-pair file referencing removed state 'B' must not survive a rerun"
    )
    assert not any("_C_" in name or "_C." in name for name in second_run_files), (
        "stale per-pair file referencing removed state 'C' must not survive a rerun"
    )
    assert any("_D_" in name or "_D." in name for name in second_run_files)

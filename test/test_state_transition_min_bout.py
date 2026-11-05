"""`min_bout_length` in `save_state_transition_report`: bouts shorter than the threshold are
merged into their neighbouring state before transitions are counted, missing states are left
alone, and the caller's AnnData is not modified.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

from behav3d.analysis.behavior.state.filtering import filter_short_state_runs
from behav3d.analysis.behavior.state.visualization.plots.state_transitions import (
    save_state_transition_report,
)

_STATE_COL = "full_behavioral_state"


def _build_adata(states):
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * len(states),
            "TrackID": [1] * len(states),
            "position_t": list(range(len(states))),
            _STATE_COL: states,
        }
    )
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs)


def _run_report(adata, tmp_path, min_bout_length):
    out = save_state_transition_report(
        adata=adata,
        output_dir=tmp_path,
        state_col=_STATE_COL,
        include_circular_diagram=False,
        include_ngram_rankings=False,
        include_sankey_pairs=False,
        min_bout_length=min_bout_length,
        verbose=False,
    )
    return pd.read_csv(out["transition_matrix_counts_csv"], index_col=0)


def test_min_bout_length_merges_short_bouts(tmp_path):
    states = list("AAAABCCCC")
    adata = _build_adata(states)

    counts_raw = _run_report(adata, tmp_path / "raw", min_bout_length=1)
    assert counts_raw.loc["A", "B"] == 1
    assert counts_raw.loc["B", "C"] == 1
    assert counts_raw.loc["A", "A"] == 3

    counts = _run_report(adata, tmp_path / "filtered", min_bout_length=2)
    assert "B" not in counts.index
    assert counts.loc["A", "C"] == 1
    assert counts.loc["A", "A"] == 4
    assert counts.loc["C", "C"] == 3

    # The caller's AnnData keeps its original states.
    assert adata.obs[_STATE_COL].tolist() == states


def test_filter_short_state_runs_leaves_missing_states_alone():
    adata = _build_adata(["A", "A", "A", np.nan, "B", "B", "B", "C", "B", "B"])
    out = filter_short_state_runs(adata, cluster_key=_STATE_COL, length_removed=1)
    result = out.obs[_STATE_COL].tolist()
    assert pd.isna(result[3])
    assert result[:3] == ["A", "A", "A"]
    assert result[4:] == ["B"] * 6

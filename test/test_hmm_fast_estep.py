"""The fast HMM E-step, EM progress bar and vectorised start-offset backfill must not change results."""
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM

from behav3d.analysis.behavior.state import hmm as hmm_mod
from behav3d.analysis.behavior.state.classification import (
    _backfill_start_offset_rows_from_applied_states,
    _backfill_start_offset_rows_from_raw_state,
    _format_hmm_raw_state_value_for_key,
)

RAW = "raw_state"
INTR = "intrinsic_state"
CONF = "conf"


def _synthetic_tracks(n_tracks=120, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(4, 40, n_tracks)
    centers = rng.normal(scale=3.0, size=(3, n_features))
    rows = []
    for tid, n in enumerate(lengths):
        state = rng.integers(0, 3)
        for t in range(n):
            if rng.random() < 0.1:
                state = rng.integers(0, 3)
            rows.append([f"s{tid % 3}", tid, t, *(centers[state] + rng.normal(size=n_features))])
    feats = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(rows, columns=["sample_name", "TrackID", "position_t", *feats]), feats


def _stock_fit(model_kwargs, X, lengths, sticky_init=None):
    m = GaussianHMM(**model_kwargs)
    if sticky_init is not None:
        m.startprob_, m.transmat_ = sticky_init
    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        m.fit(X, lengths=lengths)
    return m


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
@pytest.mark.parametrize("sticky", [False, True])
def test_fast_estep_matches_stock_hmmlearn(covariance_type, sticky):
    df, feats = _synthetic_tracks()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_out, model, _ = hmm_mod.run_hmm_state_classification(
            df, feats, n_states=3, covariance_type=covariance_type, n_iter=25, tol=1e-4,
            random_state=7, sticky=sticky, verbose=True, out_col_name="s",
        )
    df_sorted = df.sort_values(["sample_name", "TrackID", "position_t"], kind="mergesort")
    X = df_sorted[feats].to_numpy(dtype=float)
    lengths = df_sorted.groupby(["sample_name", "TrackID"], sort=False).size().tolist()
    kwargs = dict(n_components=3, covariance_type=covariance_type, n_iter=25, tol=1e-4,
                  random_state=7, verbose=False, min_covar=1e-3)
    sticky_init = None
    if sticky:
        prior = np.full((3, 3), 1.0)
        np.fill_diagonal(prior, 9.0)
        kwargs.update(params="tmc", init_params="mc", transmat_prior=prior)
        sticky_init = (np.full(3, 1 / 3), prior / prior.sum(axis=1, keepdims=True))
    ref = _stock_fit(kwargs, X, lengths, sticky_init)

    assert type(model) is GaussianHMM
    for attr in ("means_", "covars_", "transmat_", "startprob_"):
        np.testing.assert_allclose(getattr(model, attr), getattr(ref, attr), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(list(model.monitor_.history), list(ref.monitor_.history), rtol=1e-12)
    assert model.monitor_.iter == ref.monitor_.iter
    ref_states = ref.predict(X, lengths=lengths) + 1
    merged = df_sorted.merge(df_out, on=["sample_name", "TrackID", "position_t"], sort=False)
    np.testing.assert_array_equal(merged["s"].to_numpy(), ref_states)


def test_auto_mode_selection_and_plain_pickle():
    df, feats = _synthetic_tracks(n_tracks=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model, sel = hmm_mod.run_hmm_state_classification(
            df, feats, n_states="auto", k_min=1, k_max=3, n_iter=15, random_state=3, verbose=True,
        )
    assert list(sel["k"]) == [1, 2, 3]
    df_sorted = df.sort_values(["sample_name", "TrackID", "position_t"], kind="mergesort")
    X = df_sorted[feats].to_numpy(dtype=float)
    lengths = df_sorted.groupby(["sample_name", "TrackID"], sort=False).size().tolist()
    for k, logL in zip(sel["k"], sel["logL"]):
        ref = _stock_fit(dict(n_components=int(k), covariance_type="full", n_iter=15, tol=1e-3,
                              random_state=3, verbose=False, min_covar=1e-3), X, lengths)
        np.testing.assert_allclose(logL, ref.score(X, lengths=lengths), rtol=1e-10)
    # Deployment artifacts pickle the model: it must be a plain hmmlearn object without tqdm handles.
    assert type(model) is GaussianHMM
    assert type(model.monitor_) is ConvergenceMonitor
    # hmmlearn's one-line-per-EM-iteration stderr output stays off; the tqdm bar replaces it.
    assert model.verbose is False and model.monitor_.verbose is False
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X, lengths=lengths), model.predict(X, lengths=lengths))


# ---- start-offset backfill: vectorised helper vs. the previous per-track loops ----

def _backfill_frame(start_offset, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(40):
        n = int(rng.integers(1, 8))  # includes tracks no longer than the offset
        for t in range(n):
            rows.append(("a" if tid % 2 else "b", tid, t))
    df = pd.DataFrame(rows, columns=["sample_name", "TrackID", "position_t"])
    df.index = [f"r{i}" for i in range(len(df))]
    idx = df.groupby(["sample_name", "TrackID"]).cumcount()
    df["_start_offset_scored"] = idx >= start_offset
    df["_start_offset_skipped"] = ~df["_start_offset_scored"]
    raw = pd.Series(rng.integers(1, 5, len(df)), index=df.index, dtype="Int64")
    raw[~df["_start_offset_scored"]] = pd.NA
    raw[rng.random(len(df)) < 0.1] = pd.NA  # some scored rows without a state
    df[RAW] = raw
    df[INTR] = raw.astype("string")
    conf = pd.Series(rng.random(len(df)), index=df.index)
    conf[raw.isna().to_numpy()] = np.nan
    df[CONF] = conf
    return df


def _old_training_loop(df):
    for _, group_idx in df.groupby(["sample_name", "TrackID"], sort=False, observed=False).groups.items():
        group = df.loc[list(group_idx)]
        scored_group = group[group["_start_offset_scored"]]
        skipped_group = group[group["_start_offset_skipped"]]
        if len(scored_group) == 0 or len(skipped_group) == 0:
            continue
        first_raw = scored_group[RAW].iloc[0]
        if pd.isna(first_raw):
            continue
        df.loc[skipped_group.index, RAW] = first_raw
        first_intrinsic = _format_hmm_raw_state_value_for_key(first_raw, label=RAW)
        if first_intrinsic != "":
            df.loc[skipped_group.index, INTR] = first_intrinsic
    return df


def _old_apply_loop(obs):
    for _, group_idx in obs.groupby(["sample_name", "TrackID"], sort=False, observed=False).groups.items():
        group = obs.loc[list(group_idx)]
        scored_group = group[group["_start_offset_scored"]]
        skipped_group = group[group["_start_offset_skipped"]]
        if len(scored_group) == 0 or len(skipped_group) == 0:
            continue
        first_raw = scored_group[RAW].iloc[0]
        first_intrinsic = scored_group[INTR].iloc[0]
        first_conf = pd.to_numeric(pd.Series(scored_group[CONF]), errors="coerce").iloc[0]
        if not pd.isna(first_raw):
            obs.loc[skipped_group.index, RAW] = first_raw
        if not pd.isna(first_intrinsic):
            obs.loc[skipped_group.index, INTR] = first_intrinsic
        if np.isfinite(first_conf):
            obs.loc[skipped_group.index, CONF] = first_conf
    return obs


@pytest.mark.parametrize("start_offset", [1, 3])
@pytest.mark.parametrize("categorical_sample", [False, True])
def test_vectorised_backfill_matches_old_loops(start_offset, categorical_sample):
    base = _backfill_frame(start_offset)
    if categorical_sample:
        base["sample_name"] = base["sample_name"].astype("category")
    new_training = _backfill_start_offset_rows_from_raw_state(base.copy(), raw_col=RAW, intrinsic_col=INTR)
    pd.testing.assert_frame_equal(new_training, _old_training_loop(base.copy()))
    new_apply = _backfill_start_offset_rows_from_applied_states(
        base.copy(), raw_col=RAW, intrinsic_col=INTR, confidence_col=CONF
    )
    pd.testing.assert_frame_equal(new_apply, _old_apply_loop(base.copy()))

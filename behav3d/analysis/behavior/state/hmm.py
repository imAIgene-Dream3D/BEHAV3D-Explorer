import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

from behav3d.core.thread_limits import openmp_thread_cap

try:  # private hmmlearn API used by the fast E-step; fall back to stock fit if unavailable
    import hmmlearn as _hmmlearn
    from hmmlearn import _hmmc

    _FAST_ESTEP_SUPPORTED = str(_hmmlearn.__version__).startswith("0.3.")
except Exception:  # pragma: no cover - depends on installed hmmlearn
    _hmmc = None
    _FAST_ESTEP_SUPPORTED = False


class _FastGaussianHMM(GaussianHMM):
    """GaussianHMM whose E-step computes the emission log-likelihood once per iteration.

    Stock hmmlearn recomputes the Gaussian log-likelihood (K Cholesky factorisations plus
    K small triangular solves) separately for every track at every EM iteration. The
    log-likelihood of a row does not depend on the other rows, so computing it once for
    the whole matrix and slicing it per track gives the same per-row values. Everything
    else (forward/backward, posteriors, statistics accumulated per track in the same
    order) is identical to ``hmmlearn.base._AbstractHMM._do_estep``.

    Only used during ``fit``; afterwards the instance is turned back into a plain
    ``GaussianHMM`` so pickled deployment artifacts stay plain hmmlearn objects.
    """

    def _do_estep(self, X, lengths):
        if self.implementation != "log":
            return super()._do_estep(X, lengths)
        stats = self._initialize_sufficient_statistics()
        self._estep_begin()
        curr_logprob = 0
        log_frameprob_all = self._compute_log_likelihood(X)
        start = 0
        for n in lengths:
            stop = start + int(n)
            sub_X = X[start:stop]
            log_frameprob = log_frameprob_all[start:stop]
            start = stop
            logprob, fwdlattice = _hmmc.forward_log(self.startprob_, self.transmat_, log_frameprob)
            bwdlattice = _hmmc.backward_log(self.startprob_, self.transmat_, log_frameprob)
            posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
            self._accumulate_sufficient_statistics(
                stats, sub_X, log_frameprob, posteriors, fwdlattice, bwdlattice
            )
            curr_logprob += logprob
        return stats, curr_logprob


class _TqdmConvergenceMonitor(ConvergenceMonitor):
    """ConvergenceMonitor that also drives a tqdm bar (one tick per EM iteration).

    EM stops when the log-likelihood gain drops below ``tol`` or after ``n_iter``
    iterations, so ``total=n_iter`` is an upper bound and the bar ends early on
    convergence.
    """

    def __init__(self, tol, n_iter, verbose=False, *, desc="HMM fit"):
        super().__init__(tol, n_iter, verbose)
        self._desc = desc
        self._bar = None

    def _reset(self):
        super()._reset()
        self.close()
        self._bar = tqdm(total=int(self.n_iter), desc=self._desc, unit="iter")

    def report(self, log_prob):
        prev = self.history[-1] if self.history else None
        super().report(log_prob)
        if self._bar is None:
            return
        self._bar.update(1)
        postfix = {"logL": f"{log_prob:.4g}", "tol": f"{self.tol:.0e}"}
        if prev is not None:
            postfix["delta"] = f"{log_prob - prev:.3g}"
        self._bar.set_postfix(postfix, refresh=False)
        if self.converged:
            if self.iter < self.n_iter:
                self._bar.set_description_str(f"{self._desc} converged at iter {self.iter}")
            else:
                self._bar.set_description_str(f"{self._desc} stopped at n_iter (not converged)")
            self.close()

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _fit_hmm_model(m, X, lengths, *, progress_desc=None):
    """Fit ``m`` with the fast E-step and an optional tqdm bar; return a plain GaussianHMM."""
    monitor = None
    if progress_desc is not None:
        monitor = _TqdmConvergenceMonitor(m.tol, m.n_iter, False, desc=progress_desc)
        m.monitor_ = monitor
    if _FAST_ESTEP_SUPPORTED and type(m) is GaussianHMM:
        m.__class__ = _FastGaussianHMM
    try:
        # fit() starts with KMeans -> cap OpenMP threads (crash on >128 CPUs)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"), openmp_thread_cap():
            m.fit(X, lengths=lengths)
    finally:
        if type(m) is _FastGaussianHMM:
            m.__class__ = GaussianHMM
        if monitor is not None:
            monitor.close()
            plain = ConvergenceMonitor(monitor.tol, monitor.n_iter, m.verbose)
            plain.history = monitor.history
            plain.iter = monitor.iter
            m.monitor_ = plain
    return m


def run_hmm_state_classification(
    df_features,
    feature_cols,
    model=None,
    id_cols=("sample_name", "TrackID"),
    time_col="position_t",
    n_states="auto",
    k_min=1,
    k_max=8,
    covariance_type="full",
    n_iter=200,
    tol=1e-3,
    random_state=0,
    verbose=False,
    error_on_nans=True,
    out_col_name="hmm_state",
    # --- sticky knobs (optional) ---
    sticky=False,
    stickiness_kappa=8.0,
    transmat_alpha=1.0,
    min_covar=1e-3,
):
    """
    Fit a (Gaussian) HMM to track time series and assign a hidden state to each timepoint.

    If n_states="auto", choose number of states via BIC sweep from k_min..k_max.

    If sticky=True, apply a "sticky" Dirichlet prior to the transition matrix:
      prior_ij = alpha         (i != j)
      prior_ii = alpha + kappa (diagonal)

    Returns
    -------
    (df_with_states, fitted_model, selection_df)
        df_with_states is df_features plus a column out_col_name.
        fitted_model is the chosen global model.
        selection_df is a DataFrame with columns ["k","bic","logL"] if auto, otherwise None.
    """

    def _prep_X_and_lengths(df):
        req = set(list(id_cols) + [time_col] + list(feature_cols))
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"df_features missing columns: {missing}")

        df_sorted = df.sort_values(list(id_cols) + [time_col], kind="mergesort").copy()

        sequences = []
        lengths = []
        for _, g in df_sorted.groupby(list(id_cols), sort=False):
            X = g[list(feature_cols)].to_numpy(dtype=float)

            if error_on_nans and np.isnan(X).any():
                raise ValueError(
                    "NaNs found in feature matrix. "
                    "Please interpolate/impute before calling run_hmm_state_classification."
                )

            sequences.append(X)
            lengths.append(len(g))

        X_all = np.vstack(sequences) if sequences else np.empty((0, len(feature_cols)), dtype=float)
        return X_all, lengths, df_sorted

    def _num_params(K, D, cov_type, count_startprob=True):
        # startprob_ params: (K-1) if learned, else 0
        # transmat_ free params: K*(K-1) (row-normalized)
        # means: K*D
        # covars: depends on type
        if cov_type == "full":
            cov_params = K * (D * (D + 1) / 2)
        elif cov_type == "diag":
            cov_params = K * D
        elif cov_type == "spherical":
            cov_params = K
        elif cov_type == "tied":
            cov_params = (D * (D + 1) / 2)
        else:
            raise ValueError(f"Unknown covariance_type: {cov_type}")

        start_params = (K - 1) if count_startprob else 0
        trans_params = K * (K - 1)
        mean_params = K * D
        return start_params + trans_params + mean_params + cov_params

    def _bic(m, X, logL, count_startprob=True):
        N, D = X.shape
        K = m.n_components
        p = _num_params(K, D, m.covariance_type, count_startprob=count_startprob)
        return -2 * logL + p * np.log(N)

    def _make_sticky_prior(K, alpha, kappa):
        if alpha <= 0:
            raise ValueError("transmat_alpha must be > 0 for a valid Dirichlet prior.")
        if kappa < 0:
            raise ValueError("stickiness_kappa must be >= 0.")
        prior = np.full((K, K), float(alpha), dtype=float)
        np.fill_diagonal(prior, float(alpha) + float(kappa))
        return prior

    def _make_sticky_init_transmat(K, alpha, kappa):
        prior = _make_sticky_prior(K, alpha, kappa)
        return prior / prior.sum(axis=1, keepdims=True)

    def _build_model(K):
        if sticky:
            sticky_prior = _make_sticky_prior(K, transmat_alpha, stickiness_kappa)
            sticky_init = _make_sticky_init_transmat(K, transmat_alpha, stickiness_kappa)

            # We set startprob_ and transmat_ ourselves -> exclude 's' and 't' from init_params.
            # We do NOT want to train startprob_ -> exclude 's' from params.
            init_params = "mc"
            params = "tmc"

            m = GaussianHMM(
                n_components=K,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=False,  # no per-iteration stderr lines; progress is the tqdm bar
                params=params,
                init_params=init_params,
                transmat_prior=sticky_prior,
                min_covar=min_covar,
            )

            # Fixed uniform initial state distribution
            m.startprob_ = np.full(K, 1.0 / K, dtype=float)

            # Sticky starting point for transitions
            m.transmat_ = sticky_init
            return m

        # non-sticky
        return GaussianHMM(
            n_components=K,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=False,  # no per-iteration stderr lines; progress is the tqdm bar
            min_covar=min_covar,
        )

    def _print_fit_plan(k_values):
        if not verbose:
            return
        k_txt = str(k_values[0]) if len(k_values) == 1 else f"{k_values[0]}..{k_values[-1]}"
        print(
            f"[state-hmm] INFO HMM fit: {X_all.shape[0]} timepoints, {len(lengths)} tracks, "
            f"{X_all.shape[1]} features, k={k_txt} ({len(k_values)} model(s)); up to n_iter={int(n_iter)} "
            f"EM iterations per model, stops earlier once the logL gain < tol={float(tol):g}. "
            "The progress bar total is therefore an upper bound."
        )

    X_all, lengths, df_sorted = _prep_X_and_lengths(df_features)

    if X_all.shape[0] == 0:
        out = df_sorted.copy()
        out[out_col_name] = np.nan
        selection_df = None
        df_features_clean = df_features.drop(columns=[out_col_name], errors="ignore")
        df_out = df_features_clean.merge(
            out[list(id_cols) + [time_col, out_col_name]],
            on=list(id_cols) + [time_col],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        return df_out, model, selection_df

    selection_df = None

    if model is None:
        if isinstance(n_states, str):
            if n_states.lower() != "auto":
                raise ValueError("n_states must be int or 'auto'")
            if k_max < k_min:
                raise ValueError("k_max must be >= k_min")

            rows = []
            best_bic = np.inf
            best_model = None

            k_values = list(range(int(k_min), int(k_max) + 1))
            _print_fit_plan(k_values)
            for i_k, k in enumerate(k_values, start=1):
                m = _build_model(k)
                _fit_hmm_model(
                    m,
                    X_all,
                    lengths,
                    progress_desc=(f"HMM fit k={k} [{i_k}/{len(k_values)}]" if verbose else None),
                )
                if hasattr(m, "monitor_") and not m.monitor_.converged:
                    warnings.warn(
                        f"HMM (n_states={k}) stopped after reaching n_iter={n_iter} without converging. "
                        "The result may be suboptimal — consider increasing n_iter or relaxing tol.",
                        UserWarning,
                        stacklevel=2,
                    )

                # if sticky=True and startprob fixed, don't count startprob params in BIC
                count_startprob = not sticky
                logL_k = m.score(X_all, lengths=lengths)
                bic_k = _bic(m, X_all, logL_k, count_startprob=count_startprob)

                rows.append({"k": k, "bic": bic_k, "logL": logL_k})

                if bic_k < best_bic:
                    best_bic = bic_k
                    best_model = m

            model = best_model
            selection_df = pd.DataFrame(rows)
        else:
            model = _build_model(int(n_states))
            _print_fit_plan([int(n_states)])
            _fit_hmm_model(
                model,
                X_all,
                lengths,
                progress_desc=(f"HMM fit k={int(n_states)}" if verbose else None),
            )
            if hasattr(model, "monitor_") and not model.monitor_.converged:
                warnings.warn(
                    f"HMM (n_states={n_states}) stopped after reaching n_iter={n_iter} without converging. "
                    "The result may be suboptimal — consider increasing n_iter or relaxing tol.",
                    UserWarning,
                    stacklevel=2,
                )

    # ---------- decode ----------
    states_all = model.predict(X_all, lengths=lengths) + 1  # 1-based
    out = df_sorted.copy()
    out[out_col_name] = states_all

    state_map = out[list(id_cols) + [time_col, out_col_name]]
    df_features_clean = df_features.drop(columns=[out_col_name], errors="ignore")

    df_out = df_features_clean.merge(
        state_map,
        on=list(id_cols) + [time_col],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    return df_out, model, selection_df


def run_sticky_hmm_state_classification(
    df_features,
    feature_cols,
    model=None,
    id_cols=("sample_name", "TrackID"),
    time_col="position_t",
    n_states="auto",
    k_min=1,
    k_max=8,
    covariance_type="full",
    n_iter=200,
    tol=1e-3,
    random_state=0,
    verbose=False,
    error_on_nans=True,
    out_col_name="hmm_state",
    # sticky defaults
    stickiness_kappa=8.0,
    transmat_alpha=1.0,
    min_covar=1e-3,
):
    """
    Convenience wrapper around run_hmm_state_classification with sticky=True and
    default sticky hyperparameters.
    """
    return run_hmm_state_classification(
        df_features=df_features,
        feature_cols=feature_cols,
        model=model,
        id_cols=id_cols,
        time_col=time_col,
        n_states=n_states,
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        error_on_nans=error_on_nans,
        out_col_name=out_col_name,
        sticky=True,
        stickiness_kappa=stickiness_kappa,
        transmat_alpha=transmat_alpha,
        min_covar=min_covar,
    )

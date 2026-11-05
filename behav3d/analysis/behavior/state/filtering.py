import numpy as np
import pandas as pd


def filter_short_state_runs(
    adata,
    cluster_key,
    id_cols=("sample_name", "TrackID"),
    time_key="position_t",
    length_removed=1,
    new_key=None,
    inplace=False,
):
    """
    Progressive + ordered smoothing:
      - For each track, for k = 1..length_removed:
          scan left-to-right;
          whenever there is a run of length == k, replace it,
          then step back and re-check (so if following states are now continuous it doesnt overfilters).
          e.g. A A A A B A C C C k=1 -> A A A A A A C C C and not A A A A A A B C C C 
    """

    if not inplace:
        adata = adata.copy()

    df = adata.obs[list(id_cols) + [cluster_key, time_key]].copy()
    df = df.sort_values(list(id_cols) + [time_key])

    out = df[cluster_key].copy()

    def _smooth_exact_k_in_order(states, k):
        """Scan left-to-right; replace runs of exact length k immediately; re-check after each change."""
        n = len(states)
        if n == 0:
            return states

        i = 0
        while i < n:
            # find run [i:j)
            j = i + 1
            while j < n and states[j] == states[i]:
                j += 1
            run_len = j - i

            if run_len == k:
                prev_state = states[i - 1] if i > 0 else None
                next_state = states[j] if j < n else None

                if prev_state is None and next_state is None:
                    replacement = states[i]      # only run
                elif prev_state is None:
                    replacement = next_state     # start
                elif next_state is None:
                    replacement = prev_state     # end
                else:
                    # rule: if neighbors differ, take previous state
                    replacement = prev_state

                if replacement != states[i]:
                    states[i:j] = replacement
                    # step back to re-check merges around the edit
                    i = max(0, i - 1)
                    continue

            # move to next run
            i = j

        return states

    for _, sub in df.groupby(list(id_cols), sort=False, observed=True):
        idx = sub.index.to_numpy()
        states = sub[cluster_key].to_numpy().copy()
        # Missing states split the track into segments smoothed independently, so they are
        # never filled in or used as a replacement (NaN != NaN would make each a run of 1).
        missing = pd.isna(states)
        bounds = np.flatnonzero(np.diff(np.r_[0, (~missing).astype(np.int8), 0]))

        for start, stop in zip(bounds[::2], bounds[1::2]):
            seg = states[start:stop]
            for k in range(1, int(length_removed) + 1):
                seg = _smooth_exact_k_in_order(seg, k)
            states[start:stop] = seg

        out.loc[idx] = states

    if new_key is not None:
        adata.obs[new_key] = out
    else:
        adata.obs[cluster_key] = out

    return adata

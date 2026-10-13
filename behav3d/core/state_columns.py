"""Behavioral-state obs-column names and their resolvers.

Deliberately a leaf module: it depends on nothing but ``numpy`` and ``pandas``.
These names used to live in :mod:`behav3d.analysis.behavior.state.classification`,
whose imports pull in scanpy, seaborn, sklearn, hmmlearn and plotly. The napari
plugin only needs the column names and resolvers at plugin-load / metadata-load
time, so importing them from there put that whole stack on the Qt main thread.
They are re-exported from the old module for backwards compatibility (same
pattern as :mod:`behav3d.core.column_detection`).
"""
import numpy as np
import pandas as pd

INTRINSIC_STATE_COL = "hmm_intrinsic_behavioral_state"
HMM_INTRINSIC_RAW_STATE_COL = "hmm_intrinsic_behavioral_state_raw"
FULL_STATE_COL = "full_behavioral_state"

# Legacy obs-column names retired by the column consolidation; still checked as
# a fallback when loading files written before columns were consolidated.
_LEGACY_INTRINSIC_STATE_ALIASES = ("intrinsic_behavioral_cluster",)
_LEGACY_FULL_STATE_ALIASES = ("full_behavioral_cluster", "behavioral_state")


def _resolve_obs_column_with_legacy_fallback(adata, canonical_col, legacy_aliases=()):
    """Return the first of [canonical_col, *legacy_aliases] present in adata.obs.columns, or None.

    Lets read sites recognise files written before an obs column was renamed
    or consolidated, without forcing a rewrite of the file on read.
    """
    if adata is None or not hasattr(adata, "obs"):
        return None
    for col in (canonical_col, *legacy_aliases):
        if col in adata.obs.columns:
            return col
    return None


def resolve_intrinsic_state_col(adata):
    """Return the obs column holding intrinsic state labels, falling back to
    the pre-consolidation legacy column name for old files."""
    return _resolve_obs_column_with_legacy_fallback(
        adata, INTRINSIC_STATE_COL, _LEGACY_INTRINSIC_STATE_ALIASES
    )


def resolve_full_state_col(adata):
    """Return the obs column holding full (intrinsic x binary-group) state
    labels, falling back to pre-consolidation legacy column names (including
    the old value of FULL_STATE_COL itself, "behavioral_state") for old files."""
    return _resolve_obs_column_with_legacy_fallback(
        adata, FULL_STATE_COL, _LEGACY_FULL_STATE_ALIASES
    )


def _coerce_hmm_raw_state_series(raw_series, *, label):
    series = pd.Series(raw_series)
    stringified = series.astype("string").str.strip()
    empty_mask = stringified.isna() | (stringified == "")
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = (~empty_mask) & numeric.isna()
    if bool(invalid_mask.any()):
        bad_vals = sorted(set(stringified[invalid_mask].tolist()))
        raise ValueError(
            f"{label} contains non-numeric HMM raw state values: {bad_vals[:10]}"
        )
    numeric_valid = numeric[~numeric.isna()]
    if not numeric_valid.empty:
        fractional = np.mod(numeric_valid.astype(float), 1.0)
        non_integer_mask = ~np.isclose(fractional, 0.0)
        if bool(np.any(non_integer_mask)):
            bad_vals = sorted(set(numeric_valid[non_integer_mask].tolist()))
            raise ValueError(
                f"{label} contains non-integer HMM raw state values: {bad_vals[:10]}"
            )
    coerced = numeric.round(0).astype("Int64")
    coerced[empty_mask] = pd.NA
    coerced.index = series.index
    return coerced


def _format_hmm_raw_state_series_for_key(raw_series, *, label):
    raw_int = _coerce_hmm_raw_state_series(raw_series, label=label)
    return raw_int.astype("string").str.strip()

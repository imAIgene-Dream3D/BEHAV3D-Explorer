"""Value-based column classification for track/state feature CSVs.

Deliberately a leaf module: it depends on nothing but ``math`` and ``pandas``.
These helpers used to live in :mod:`behav3d.widgets.base_state_classification`,
whose imports pull in the whole clustering stack (scanpy -> umap -> pynndescent),
costing ~12 s of numba JIT compilation on first import. The napari Single Cell
tab calls them on every metadata load, so that cost landed on the Qt main thread
and froze the GUI. They are re-exported from the old module for backwards
compatibility.
"""
import math
import os
import threading

import pandas as pd


def normalize_binary_value(value, tol=1e-9):
    """Return 0/1 if ``value`` normalizes to a boolean, else ``None``.

    Accepts native bools, the strings true/false/t/f, and numeric 0/1.
    Shared by the notebook and napari binary-column detectors so both use
    identical semantics.
    """
    if isinstance(value, (bool, pd.BooleanDtype)):
        return 1 if bool(value) else 0
    if isinstance(value, str):
        sval = value.strip().lower()
        if sval in {"true", "t"}:
            return 1
        if sval in {"false", "f"}:
            return 0
        try:
            value = float(sval)
        except Exception:
            return None
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval):
        return None
    if abs(fval - 0.0) <= tol:
        return 0
    if abs(fval - 1.0) <= tol:
        return 1
    return None


def detect_binary_columns_from_csv(csv_path, cols, chunksize=50000, cancel_check=None):
    """Value-based binary-column detection over the *full* CSV.

    A column is binary only if every non-NA value normalizes (via
    :func:`normalize_binary_value`) to ``{0, 1}``. This replaces fragile
    dtype sniffing over a small row sample, which mis-classifies numeric
    columns as binary whenever the sampled rows happen to be NaN/blank.

    ``cancel_check``, when given, is a zero-arg callable polled once per
    chunk; when it returns true the scan stops early and returns whatever
    was classified so far. Lets a caller running this on a background
    thread (see ``behav3d.napari._background_runner.BackgroundOperation``)
    interrupt a long scan instead of blocking teardown indefinitely.
    """
    if csv_path is None or len(cols) == 0:
        return []

    states = {str(c): {"seen": set(), "invalid": False} for c in cols}
    try:
        for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize, low_memory=False):
            if cancel_check is not None and cancel_check():
                break
            for col in cols:
                st = states[str(col)]
                if st["invalid"]:
                    continue
                series = chunk[col].dropna()
                if len(series) == 0:
                    continue
                unique_vals = pd.unique(series)
                for raw in unique_vals:
                    norm = normalize_binary_value(raw)
                    if norm is None:
                        st["invalid"] = True
                        break
                    st["seen"].add(int(norm))
                    if len(st["seen"]) > 2:
                        st["invalid"] = True
                        break
    except Exception:
        return []

    return sorted(
        [
            c
            for c in cols
            if (not states[str(c)]["invalid"]) and (len(states[str(c)]["seen"]) > 0)
        ]
    )


def detect_non_numeric_columns_from_csv(csv_path, cols, chunksize=50000, cancel_check=None):
    """Value-based detection of columns unsuitable as continuous HMM features.

    A column is flagged if any non-NA value fails to parse as a finite float --
    e.g. free-text/categorical labels ("um", "27t") or comma-separated contact-ID
    lists ("45,46"). Such object-dtype columns silently poison the HMM
    observation matrix's dtype when selected as a feature: pandas keeps the
    column as ``object`` even after numeric coercion, so ``adata.X`` ends up
    object-dtype and anndata's h5ad writer -- assuming an object array must be
    strings -- crashes with "Can't implicitly convert non-string objects to
    strings" the moment it hits the actual float values. These columns must be
    excluded before they are ever offered as selectable timepoint features.

    ``cancel_check``, when given, is a zero-arg callable polled once per
    chunk; when it returns true the scan stops early and returns whatever
    was classified so far (see ``detect_binary_columns_from_csv``).
    """
    if csv_path is None or len(cols) == 0:
        return []

    invalid = {str(c): False for c in cols}
    try:
        for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize, low_memory=False):
            if cancel_check is not None and cancel_check():
                break
            for col in cols:
                key = str(col)
                if invalid[key]:
                    continue
                series = chunk[col].dropna()
                if len(series) == 0:
                    continue
                if pd.to_numeric(series, errors="coerce").isna().any():
                    invalid[key] = True
    except Exception:
        return []

    return sorted([c for c in cols if invalid[str(c)]])


# ---------------------------------------------------------------------------
# Single-pass, memoized classification
# ---------------------------------------------------------------------------
# The State tab and the Trajectory Feature selector both need "which columns are
# binary / non-numeric" for the same track-features CSV. Doing that as two
# separate full-CSV passes in each consumer meant four parses of a (multi-GB)
# file per refresh, repeated every time the tab was revisited. One pass that
# tracks both properties, memoized on the file's identity, makes it one parse per
# file version for everyone.
_CLASSIFY_CACHE = {}
_CLASSIFY_LOCK = threading.Lock()
_CLASSIFY_CACHE_MAX = 8


def _file_identity(csv_path):
    """``(path, mtime_ns, size)`` or ``None`` if the file can't be stat-ed."""
    try:
        st = os.stat(csv_path)
    except OSError:
        return None
    return (str(csv_path), st.st_mtime_ns, st.st_size)


def classify_feature_columns(csv_path, cols, chunksize=50000, cancel_check=None):
    """Classify ``cols`` of a track-features CSV in a single pass over the file.

    Returns ``{"bin_cols": [...], "non_numeric": [...], "feat_cols": [...]}``:

    * ``bin_cols``    - same result as :func:`detect_binary_columns_from_csv`;
    * ``non_numeric`` - non-binary columns that do not parse as finite floats
      (same result as :func:`detect_non_numeric_columns_from_csv` run over the
      non-binary columns);
    * ``feat_cols``   - the non-binary, numeric columns, in ``cols`` order.

    Results are memoized per ``(path, mtime, size, cols)``, so revisiting a tab
    does not re-parse an unchanged CSV. A scan stopped by ``cancel_check`` or an
    unreadable file is never memoized.
    """
    cols = list(cols)
    empty = {"bin_cols": [], "non_numeric": [], "feat_cols": list(cols)}
    if csv_path is None or len(cols) == 0:
        return {"bin_cols": [], "non_numeric": [], "feat_cols": []}

    identity = _file_identity(csv_path)
    key = None if identity is None else (identity, tuple(str(c) for c in cols))
    if key is not None:
        with _CLASSIFY_LOCK:
            hit = _CLASSIFY_CACHE.get(key)
        if hit is not None:
            return {k: list(v) for k, v in hit.items()}

    bin_state = {str(c): {"seen": set(), "invalid": False} for c in cols}
    non_numeric = {str(c): False for c in cols}
    try:
        for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize, low_memory=False):
            if cancel_check is not None and cancel_check():
                return empty
            for col in cols:
                name = str(col)
                bst = bin_state[name]
                if bst["invalid"] and non_numeric[name]:
                    continue
                series = chunk[col].dropna()
                if len(series) == 0:
                    continue
                if not bst["invalid"]:
                    for raw in pd.unique(series):
                        norm = normalize_binary_value(raw)
                        if norm is None:
                            bst["invalid"] = True
                            break
                        bst["seen"].add(int(norm))
                        if len(bst["seen"]) > 2:
                            bst["invalid"] = True
                            break
                if not non_numeric[name] and pd.to_numeric(series, errors="coerce").isna().any():
                    non_numeric[name] = True
    except Exception:
        return empty

    if cancel_check is not None and cancel_check():
        return empty

    bin_cols = sorted(
        c for c in cols
        if not bin_state[str(c)]["invalid"] and len(bin_state[str(c)]["seen"]) > 0
    )
    bin_set = set(bin_cols)
    candidates = [c for c in cols if c not in bin_set]
    result = {
        "bin_cols": bin_cols,
        "non_numeric": sorted(c for c in candidates if non_numeric[str(c)]),
        "feat_cols": [c for c in candidates if not non_numeric[str(c)]],
    }
    if key is not None:
        with _CLASSIFY_LOCK:
            if len(_CLASSIFY_CACHE) >= _CLASSIFY_CACHE_MAX:
                _CLASSIFY_CACHE.pop(next(iter(_CLASSIFY_CACHE)))
            _CLASSIFY_CACHE[key] = {k: list(v) for k, v in result.items()}
    return result

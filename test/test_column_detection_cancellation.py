"""
Cancellation tests for the chunked CSV column-detection scans.

These two functions are dispatched via ``BackgroundOperation`` from the
napari Single Cell tab (see ``behav3d/napari/_single_cell.py``). A long scan
needs a way to stop mid-CSV-read when ``BackgroundOperation.cancel()`` is
called at app-quit teardown, since Qt's ``thread.quit()`` alone can't
interrupt a scan that's already mid-flight inside a blocking call (see
``behav3d/napari/_background_runner.py``).

Runs under pytest *or* standalone: python test/test_column_detection_cancellation.py
Requires the `behav3d` conda env (pandas). No Qt needed.
"""
import os

import pandas as pd

from behav3d.core.column_detection import (
    detect_binary_columns_from_csv,
    detect_non_numeric_columns_from_csv,
)


def _write_csv(path, n_chunks, rows_per_chunk, bad_chunk_index):
    """Write a CSV where ``col_bin``/``col_text`` are well-behaved (binary /
    numeric-looking) everywhere except one row in ``bad_chunk_index``, which
    gets a value that would flip that column's classification if seen."""
    rows = []
    for chunk_idx in range(n_chunks):
        for i in range(rows_per_chunk):
            is_bad_row = chunk_idx == bad_chunk_index and i == 0
            rows.append({
                "col_bin": 5 if is_bad_row else (i % 2),
                "col_text": "notanumber" if is_bad_row else float(i),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _cancel_after_n_chunks(n):
    """Return a cancel_check that answers False for the first ``n`` calls,
    then True -- i.e. lets ``n`` chunks be processed before stopping."""
    calls = []

    def _check():
        calls.append(1)
        return len(calls) > n

    _check.calls = calls
    return _check


def test_detect_binary_columns_cancel_check_stops_before_invalidating_column(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    csv_path = os.path.join(str(d), "features.csv")
    # col_bin only becomes non-binary in chunk index 1.
    _write_csv(csv_path, n_chunks=3, rows_per_chunk=10, bad_chunk_index=1)

    # No cancellation: the scan reaches the bad chunk, col_bin is excluded.
    full = detect_binary_columns_from_csv(csv_path, ["col_bin"], chunksize=10)
    assert full == []

    # Cancel after the first (good) chunk, before the bad one is read:
    # col_bin is never invalidated, so it's still reported as binary.
    cancel_check = _cancel_after_n_chunks(1)
    early = detect_binary_columns_from_csv(
        csv_path, ["col_bin"], chunksize=10, cancel_check=cancel_check,
    )
    assert early == ["col_bin"]
    assert len(cancel_check.calls) == 2  # checked once per chunk until it fired


def test_detect_non_numeric_columns_cancel_check_stops_before_flagging_column(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    csv_path = os.path.join(str(d), "features.csv")
    _write_csv(csv_path, n_chunks=3, rows_per_chunk=10, bad_chunk_index=1)

    full = detect_non_numeric_columns_from_csv(csv_path, ["col_text"], chunksize=10)
    assert full == ["col_text"]

    cancel_check = _cancel_after_n_chunks(1)
    early = detect_non_numeric_columns_from_csv(
        csv_path, ["col_text"], chunksize=10, cancel_check=cancel_check,
    )
    assert early == []
    assert len(cancel_check.calls) == 2


def test_cancel_check_none_preserves_default_behavior(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    csv_path = os.path.join(str(d), "features.csv")
    _write_csv(csv_path, n_chunks=2, rows_per_chunk=10, bad_chunk_index=0)

    # cancel_check omitted entirely (today's exact call signature) behaves
    # identically to passing None explicitly -- this also regression-proofs
    # the pre-existing scan behavior, which had no test coverage before.
    assert (
        detect_binary_columns_from_csv(csv_path, ["col_bin"], chunksize=10)
        == detect_binary_columns_from_csv(csv_path, ["col_bin"], chunksize=10, cancel_check=None)
    )
    assert (
        detect_non_numeric_columns_from_csv(csv_path, ["col_text"], chunksize=10)
        == detect_non_numeric_columns_from_csv(csv_path, ["col_text"], chunksize=10, cancel_check=None)
    )
    assert detect_binary_columns_from_csv(csv_path, ["col_bin"], chunksize=10) == []
    assert detect_non_numeric_columns_from_csv(csv_path, ["col_text"], chunksize=10) == ["col_text"]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")

"""
Cellpose-SAM worker stderr regression guard.

The worker's stderr is UTF-8, but ``text=True`` without an encoding decodes it with
the locale's encoding (cp1252 on Western-locale Windows). tqdm's first partial block
is ``▏`` (``E2 96 8F``) and ``0x8F`` is undefined in cp1252, so the reader thread died
at 1.25% of any progress bar. With nothing draining the pipe the worker then blocked
on its next tqdm write, which hung the ~1.15 GB weight download forever with no error.

These tests fail by timing out, and only reproduce the decode failure on a cp1252
locale; on a UTF-8 locale ``test_undecodable_stderr_does_not_hang`` passes either way.

Runs under pytest *or* standalone: python test/test_cpsam_stderr.py
Requires the `behav3d` conda env.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from behav3d.preprocessing.segmentation import cellpose_sam_prediction as cps  # noqa: E402

TIMEOUT_S = 15

# Writes far more than a pipe buffer of tqdm-style refreshes. The watchdog stops a
# blocked stub from outliving a failed test.
STUB_WORKER = f"""
import json, os, sys, threading

threading.Timer({TIMEOUT_S + 5}, lambda: os._exit(3)).start()
for i in range(2000):
    sys.stderr.write(f"\\r  1%|\\u258f         | {{i}}M/1.15G [00:01<01:56, 10.5MB/s]")
    sys.stderr.flush()
print(json.dumps({{"event": "done", "n_done": 1}}), flush=True)
os._exit(0)
"""


def _run_stub_worker():
    """Run ``_run_worker`` against the stub; return ``(finished, error, events)``."""
    events, outcome = [], {}

    def target():
        try:
            cps._run_worker({}, Path(sys.executable), "auto", False, on_event=events.append)
        except Exception as exc:
            outcome["error"] = exc

    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "stub_worker.py"
        stub.write_text(STUB_WORKER, encoding="utf-8")

        original_worker = cps._WORKER
        original_encoding = os.environ.get("PYTHONIOENCODING")
        cps._WORKER = stub
        # What launch_napari.py sets for the napari process, which the worker inherits.
        os.environ["PYTHONIOENCODING"] = "utf-8"
        try:
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(TIMEOUT_S)
            finished = not thread.is_alive()
        finally:
            cps._WORKER = original_worker
            if original_encoding is None:
                os.environ.pop("PYTHONIOENCODING", None)
            else:
                os.environ["PYTHONIOENCODING"] = original_encoding

    return finished, outcome.get("error"), events


def test_undecodable_stderr_does_not_hang():
    original_stderr = sys.stderr
    # The worker's chatter would otherwise flood the test output.
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        finished, error, events = _run_stub_worker()
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    assert finished, "worker hung: its stderr stopped being drained"
    assert error is None, error
    assert [e["event"] for e in events] == ["done"]


def test_display_failure_does_not_hang():
    # sys.stderr is None under a windowless launch (pythonw), so echoing raises.
    original_stderr = sys.stderr
    sys.stderr = None
    try:
        finished, error, events = _run_stub_worker()
    finally:
        sys.stderr = original_stderr
    assert finished, "worker hung: a failed echo stopped stderr being drained"
    assert error is None, error
    assert [e["event"] for e in events] == ["done"]


if __name__ == "__main__":
    failures = 0
    for test in (test_undecodable_stderr_does_not_hang, test_display_failure_does_not_hang):
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    sys.exit(1 if failures else 0)

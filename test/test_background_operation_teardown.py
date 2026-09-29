"""
Teardown / cancellation tests for BackgroundOperation.

BackgroundOperation creates a QThread it manages itself (see
behav3d/napari/_background_runner.py); before cancel()/drain_zombies()
existed, the only way its QThread reference was ever released was via
thread.finished, so tearing down the owning widget while a scan was still
running raced Qt's own teardown and produced a hard
"QThread: Destroyed while thread is still running" crash. These tests
exercise the fix deterministically -- unlike the original crash, which
depended on GC/OS timing to reproduce.

Runs under pytest *or* standalone: python test/test_background_operation_teardown.py
Requires the `behav3d` conda env (PyQt5/qtpy). No pytest-qt / .exec_() needed --
mirrors test_assistant.py's QApplication.instance() or QApplication([]) pattern.
"""
import threading
import time

from qtpy.QtWidgets import QApplication, QWidget

from behav3d.napari._background_runner import BackgroundOperation

# Module-level reference: QApplication([]) must stay alive for the whole
# process, not just the function that created it -- an unreferenced
# QApplication is garbage-collected immediately, and any QWidget created
# after that aborts the process ("Must construct a QApplication before a
# QWidget").
_APP = None


def _qt_app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump_until(condition, timeout_s=5.0):
    """Process the Qt event loop until ``condition()`` is true or timeout."""
    app = _qt_app()
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return condition()


def _cooperative_loop_fn(cancel_check=None):
    while not (cancel_check and cancel_check()):
        time.sleep(0.005)
    return "stopped"


def _blocking_fn(block_event, cancel_check=None):
    block_event.wait()
    return "released"


def _quick_fn():
    return "done"


def test_cancel_cooperative_run_stops_within_timeout():
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)

    op.run(fn=_cooperative_loop_fn, inject_progress=False, inject_cancel_check=True)
    assert op.is_running()

    finished_cleanly = op.cancel(timeout_ms=2000)

    assert finished_cleanly is True
    assert op._zombie_threads == []
    assert op.is_running() is False


def test_cancel_uncooperative_run_times_out_and_parks_zombie():
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)
    block_event = threading.Event()

    op.run(fn=_blocking_fn, args=(block_event,), inject_progress=False, inject_cancel_check=True)
    assert op.is_running()

    finished_cleanly = op.cancel(timeout_ms=50)

    assert finished_cleanly is False
    assert len(op._zombie_threads) == 1
    # Freed for new work immediately, even though the zombie is still alive.
    assert op.is_running() is False

    # Let the real OS thread finish and clear the zombie so it doesn't leak
    # into other tests / the interpreter's own shutdown.
    block_event.set()
    assert _pump_until(lambda: op.drain_zombies(timeout_ms=50) == 0)
    assert op._zombie_threads == []


def test_second_run_after_uncooperative_timeout_is_not_corrupted_by_stale_zombie():
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)
    block_event = threading.Event()

    op.run(fn=_blocking_fn, args=(block_event,), inject_progress=False, inject_cancel_check=True)
    finished_cleanly = op.cancel(timeout_ms=50)
    assert finished_cleanly is False
    assert len(op._zombie_threads) == 1
    assert op.is_running() is False

    # A second run on the same instance must succeed immediately -- the
    # instance isn't permanently wedged by the abandoned zombie.
    results = []
    failures = []
    op.run(
        fn=_quick_fn,
        inject_progress=False,
        on_done=results.append,
        on_failed=failures.append,
    )
    assert _pump_until(lambda: bool(results or failures))
    assert results == ["done"]
    assert failures == []

    # Now let the *first* run's zombie actually finish. Its done/finished
    # signals were disconnected by cancel(), so its late completion must
    # not re-dispatch into the second run's callbacks or corrupt op's state.
    block_event.set()
    assert _pump_until(lambda: op.drain_zombies(timeout_ms=50) == 0)
    assert op._zombie_threads == []

    _pump_until(lambda: False, timeout_s=0.2)  # flush any stray queued events
    assert results == ["done"]
    assert failures == []
    assert op.is_running() is False


def test_is_running_reflects_thread_liveness_not_just_state_presence():
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)
    block_event = threading.Event()

    op.run(fn=_blocking_fn, args=(block_event,), inject_progress=False)
    assert op.is_running() is True
    assert op._state.thread.isRunning() is True

    block_event.set()
    assert _pump_until(lambda: not op.is_running())


def test_run_survives_bypassed_is_running_guard_without_crashing():
    """Regression test for a production crash: a run() call that somehow got
    past the is_running() guard at the top of run() (the exact trigger was
    never conclusively pinned down) while the previous run's OS thread was
    still genuinely alive. Before the defensive check added to run(), the
    unconditional ``self._state = state`` reassignment dropped the last
    Python reference to the still-running QThread there, producing an
    uncatchable "QThread: Destroyed while thread is still running" process
    abort. This forces exactly that condition deterministically -- via a
    monkeypatch rather than depending on the OS/GC timing that made the
    original crash hard to reproduce -- and asserts run() survives it by
    parking the orphaned thread instead of dropping it.
    """
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)
    block_event = threading.Event()

    op.run(fn=_blocking_fn, args=(block_event,), inject_progress=False)
    first_state = op._state
    assert first_state is not None and first_state.thread.isRunning()

    # Simulate is_running() incorrectly reporting False while self._state
    # still references the live thread.
    op.is_running = lambda: False
    try:
        op.run(fn=_quick_fn, inject_progress=False)
    finally:
        del op.is_running  # restore the bound method

    # No crash reaching this point is the primary assertion.
    assert first_state.thread in [t for t, _w in op._zombie_threads]
    assert op.is_running() is True  # the *new* run is now the active one

    block_event.set()
    assert _pump_until(lambda: op.drain_zombies(timeout_ms=50) == 0)
    assert op._zombie_threads == []


def test_run_parks_live_thread_when_qthread_isRunning_itself_misreports_false():
    """Regression test for the actual production failure mode.

    ``test_run_survives_bypassed_is_running_guard_without_crashing`` only
    fakes ``BackgroundOperation.is_running()`` -- the real ``QThread``'s own
    ``isRunning()`` (read again by run()'s park check) still truthfully
    reports the thread as alive there, so that test can pass even if the
    park check were dead code, since the two reads disagree.

    In production both the outer guard and the inner park check read the
    exact same ``QThread.isRunning()`` value, with nothing able to change
    it in between -- so if that one read is ever wrong, *both* checks are
    wrong identically. This test reproduces that by patching the QThread
    instance's own ``isRunning`` (not the BackgroundOperation wrapper), so
    both reads see the same misreported ``False`` while the OS thread is
    genuinely still blocked. Before the fix, this let ``self._state = state``
    drop the last reference to the live QThread -- the fatal crash. After
    the fix, the thread must be parked unconditionally, regardless of what
    ``isRunning()`` claims.
    """
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent)
    block_event = threading.Event()

    op.run(fn=_blocking_fn, args=(block_event,), inject_progress=False)
    first_state = op._state
    assert first_state is not None and first_state.thread.isRunning()

    # Patch the QThread instance itself: both op.is_running() and run()'s
    # internal park check read this same value.
    first_state.thread.isRunning = lambda: False

    op.run(fn=_quick_fn, inject_progress=False)

    # No crash reaching this point is the primary assertion: the still-
    # alive thread must have been parked, not dropped.
    assert first_state.thread in [t for t, _w in op._zombie_threads]
    assert op.is_running() is True  # the *new* run is now the active one

    block_event.set()
    # drain_zombies() uses QThread.wait() (a real blocking/pumped wait), not
    # isRunning(), so the patched isRunning() doesn't interfere here.
    assert _pump_until(lambda: op.drain_zombies(timeout_ms=50) == 0)
    assert op._zombie_threads == []
    del first_state.thread.isRunning  # restore the real bound method


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")

"""
Tests for the global "something is running" busy guard in
behav3d/napari/_background_runner.py.

Each tab owns its own BackgroundOperation, so before the guard nothing knew that
*another* tab was busy; starting a second heavy job (or switching tabs) while the
first ran caused crashes/freezes. These tests pin the contract:

* a non-silent run is refused while any *other* operation runs;
* ``silent`` (automatic scans), ``chained`` and queue-dispatched runs are exempt,
  and silent operations never count as "work to wait for";
* the Processing Queue being active gates click-level hooks (warn_if_busy) but not
  run() itself; queue_dispatch() lets a queue step start next to a lingering op;
* an instance may restart itself after a timed-out cancel() (existing contract,
  see test_background_operation_teardown.py), but another instance may not;
* ``busy_scope`` extends the guard to non-BackgroundOperation work.

Runs under pytest *or* standalone: python test/test_busy_guard.py
Requires the `behav3d` conda env (PyQt5/qtpy). The modal dialog is suppressed via
BEHAV3D_NO_BUSY_DIALOG so a blocked run never hangs the test on a QMessageBox.
"""
import os
import threading
import time

os.environ["BEHAV3D_NO_BUSY_DIALOG"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from behav3d.napari import _background_runner as br  # noqa: E402
from behav3d.napari._background_runner import BackgroundOperation  # noqa: E402

_APP = None


def _qt_app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump_until(condition, timeout_s=5.0):
    app = _qt_app()
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return condition()


def _blocking_fn(block_event, cancel_check=None):
    block_event.wait()
    return "released"


def _quick_fn():
    return "ran"


def _make_op(**kwargs):
    _qt_app()
    parent = QWidget()
    op = BackgroundOperation(parent, **kwargs)
    return parent, op


def _start_blocking(op, ev):
    op.run(fn=_blocking_fn, args=(ev,), inject_progress=False, desc="Long job")


def _finish(op, ev):
    ev.set()
    assert _pump_until(lambda: not op.is_running() and not br.any_background_work())


def test_idle_means_not_busy():
    _qt_app()
    assert br.any_background_work() is False
    assert br.warn_if_busy(None, "x") is False


def test_second_operation_is_blocked_while_first_runs():
    _pa, a = _make_op()
    _pb, b = _make_op()
    ev = threading.Event()
    _start_blocking(a, ev)
    try:
        assert br.any_background_work()
        assert "Long job" in br.background_work_descriptions()
        results = []
        b.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
        assert not b.is_running(), "second op must not start while the first runs"
        _pump_until(lambda: bool(results), timeout_s=0.3)
        assert results == []
    finally:
        _finish(a, ev)


def test_second_operation_runs_once_first_has_finished():
    _pa, a = _make_op()
    _pb, b = _make_op()
    ev = threading.Event()
    _start_blocking(a, ev)
    _finish(a, ev)
    results = []
    b.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
    assert _pump_until(lambda: bool(results))
    assert results == ["ran"]


def test_silent_operations_neither_block_nor_are_blocked():
    _pa, a = _make_op()
    _ps, silent = _make_op(silent=True)
    ev = threading.Event()

    # A silent op running does not make the app "busy" ...
    _start_blocking(silent, ev)
    try:
        assert br.any_background_work() is False
        results = []
        a.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
        assert _pump_until(lambda: bool(results))
    finally:
        _finish(silent, ev)

    # ... and is itself allowed to start while a real op runs.
    ev2 = threading.Event()
    _start_blocking(a, ev2)
    try:
        silent_results = []
        silent.run(fn=_quick_fn, inject_progress=False, on_done=silent_results.append)
        assert _pump_until(lambda: bool(silent_results))
    finally:
        _finish(a, ev2)


def test_chained_run_bypasses_the_guard():
    _pa, a = _make_op()
    _pb, b = _make_op()
    ev = threading.Event()
    _start_blocking(a, ev)
    try:
        results = []
        b.run(fn=_quick_fn, inject_progress=False, on_done=results.append, chained=True)
        assert _pump_until(lambda: bool(results))
    finally:
        _finish(a, ev)


def test_queue_active_gates_click_level_hooks_but_not_run_itself():
    _pa, a = _make_op()
    br.set_queue_active(True)
    try:
        assert "Processing queue" in br.background_work_descriptions()
        # Click-level hooks (tab switch, selectors, explicit warn_if_busy) are refused...
        assert br.warn_if_busy(None, "switching tabs") is True

        # ...but run() is not refused by the queue's mere active flag: queue steps
        # may start work after an event-loop hop outside queue_dispatch(), and
        # refusing those would hang the queue.
        results = []
        a.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
        assert _pump_until(lambda: bool(results))
    finally:
        br.set_queue_active(False)
    assert br.any_background_work() is False


def test_queue_dispatch_bypasses_guard_for_the_queues_own_steps():
    _pa, a = _make_op()
    _pb, b = _make_op()
    ev = threading.Event()
    _start_blocking(a, ev)  # e.g. the previous step's chained follow-up
    try:
        results = []
        with br.queue_dispatch():
            b.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
        assert _pump_until(lambda: bool(results))
    finally:
        _finish(a, ev)


def test_busy_scope_counts_as_busy():
    _pa, a = _make_op()
    with br.busy_scope("Loading metadata"):
        assert "Loading metadata" in br.background_work_descriptions()
        a.run(fn=_quick_fn, inject_progress=False)
        assert not a.is_running()
    assert br.any_background_work() is False


def test_own_zombie_does_not_block_restart_but_blocks_other_instances():
    _pa, a = _make_op()
    _pb, b = _make_op()
    ev = threading.Event()
    a.run(fn=_blocking_fn, args=(ev,), inject_progress=False, inject_cancel_check=True)
    try:
        # Uncooperative worker: cancel() times out and parks it as a zombie.
        assert a.cancel(timeout_ms=50) is False
        assert br.any_background_work(), "a still-running zombie is still work"

        # Another instance is refused...
        b.run(fn=_quick_fn, inject_progress=False)
        assert not b.is_running()

        # ...the owning instance may restart itself (documented contract).
        results = []
        a.run(fn=_quick_fn, inject_progress=False, on_done=results.append)
        assert _pump_until(lambda: bool(results))
    finally:
        ev.set()
        a.drain_zombies(timeout_ms=2000)
        _pump_until(lambda: not br.any_background_work())


def test_redispatch_runs_gui_only_methods_on_the_gui_thread():
    """Worker threads calling a widget's _log must not touch the QTextEdit
    themselves (Qt: 'Cannot queue arguments of type QTextCursor')."""
    from qtpy.QtCore import QThread
    from qtpy.QtWidgets import QTextEdit

    _qt_app()
    gui_thread = QThread.currentThread()

    class Logger(QWidget):
        def __init__(self):
            super().__init__()
            self.box = QTextEdit(self)
            self.threads = []

        def _log(self, msg):
            if br.redispatch_to_gui_thread(self._log, msg):
                return
            self.threads.append(QThread.currentThread())
            self.box.append(msg)

    w = Logger()
    _p, op = _make_op()

    def worker_logs():
        for i in range(25):
            w._log(f"line {i}")
        return "ok"

    done = []
    op.run(fn=worker_logs, inject_progress=False, on_done=done.append)
    assert _pump_until(lambda: bool(done))
    assert _pump_until(lambda: len(w.threads) == 25)
    assert all(t is gui_thread for t in w.threads), "log appends must run on the GUI thread"
    assert w.box.toPlainText().splitlines() == [f"line {i}" for i in range(25)]
    # And a direct GUI-thread call is not deferred.
    w._log("direct")
    assert w.threads[-1] is gui_thread and w.box.toPlainText().endswith("direct")


def test_change_notifier_fires_on_start_and_finish():
    _pa, a = _make_op()
    count = []
    notifier = br.busy_notifier()
    notifier.changed.connect(lambda: count.append(1))
    ev = threading.Event()
    _start_blocking(a, ev)
    n_after_start = len(count)
    assert n_after_start >= 1
    _finish(a, ev)
    # The terminal notification is emitted from the queued thread-finished slot.
    assert _pump_until(lambda: len(count) > n_after_start)


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)

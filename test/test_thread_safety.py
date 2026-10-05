"""
Thread-safety regression tests.

* ``_LatestOnlyLoader``: switching cell type A -> B while A's .h5ad is still loading
  used to deliver A's data as B's model (the in-flight load was simply not
  restarted). Only the latest request may be delivered.
* ``h5_access``: one process-wide lock serialises .h5ad reads/writes; the GUI
  thread must never block on it.

Runs under pytest *or* standalone: python test/test_thread_safety.py
Requires the `behav3d` conda env.
"""
import os
import threading
import time

os.environ.setdefault("BEHAV3D_NO_BUSY_DIALOG", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

import behav3d.napari._widget  # noqa: E402,F401  (resolves the _analysis/_single_cell import cycle)
from behav3d.core.h5_access import H5Busy, h5_access  # noqa: E402
from behav3d.napari._background_runner import BackgroundOperation  # noqa: E402
from behav3d.napari._single_cell import _LatestOnlyLoader  # noqa: E402

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


def _make_loader():
    _qt_app()
    parent = QWidget()
    bg = BackgroundOperation(parent, silent=True)
    return parent, bg, _LatestOnlyLoader(bg)


def test_stale_result_is_discarded_and_latest_request_wins():
    _parent, bg, loader = _make_loader()
    gate = threading.Event()
    delivered = []

    def load_a():
        gate.wait()
        return "DATA_A"

    def load_b():
        return "DATA_B"

    loader.request("A", load_a, lambda r: delivered.append(("A", r)), lambda e: delivered.append(("A!", e)))
    assert bg.is_running()
    # User switches cell type while A is still loading.
    loader.request("B", load_b, lambda r: delivered.append(("B", r)), lambda e: delivered.append(("B!", e)))
    gate.set()

    assert _pump_until(lambda: any(tag == "B" for tag, _ in delivered))
    assert _pump_until(lambda: not bg.is_running())
    assert delivered == [("B", "DATA_B")], f"A's stale data must never be delivered: {delivered}"


def test_same_path_requested_twice_loads_once():
    _parent, bg, loader = _make_loader()
    gate = threading.Event()
    calls = []

    def load():
        calls.append(1)
        gate.wait()
        return "X"

    out = []
    loader.request("A", load, out.append, out.append)
    loader.request("A", load, out.append, out.append)
    gate.set()
    assert _pump_until(lambda: out)
    assert _pump_until(lambda: not bg.is_running())
    assert out == ["X"] and len(calls) == 1


def test_invalidate_discards_an_in_flight_result():
    _parent, bg, loader = _make_loader()
    gate = threading.Event()
    out = []

    def load():
        gate.wait()
        return "STALE"

    loader.request("A", load, out.append, out.append)
    loader.invalidate()  # e.g. the new cell type has no file on disk
    gate.set()
    assert _pump_until(lambda: not bg.is_running())
    _qt_app().processEvents()
    assert out == []


def test_failure_of_latest_is_reported_but_not_of_superseded():
    _parent, bg, loader = _make_loader()
    gate = threading.Event()
    errs = []

    def bad_a():
        gate.wait()
        raise ValueError("a failed")

    def bad_b():
        raise ValueError("b failed")

    loader.request("A", bad_a, lambda r: None, lambda e: errs.append(("A", e)))
    loader.request("B", bad_b, lambda r: None, lambda e: errs.append(("B", e)))
    gate.set()
    assert _pump_until(lambda: any(t == "B" for t, _ in errs))
    assert _pump_until(lambda: not bg.is_running())
    assert [t for t, _ in errs] == ["B"]


def _hold(write, held, release):
    def run():
        with h5_access(write=write):
            held.set()
            release.wait()

    t = threading.Thread(target=run)
    t.start()
    assert held.wait(2)
    return t


def test_gui_side_read_fails_fast_while_a_writer_holds_the_lock():
    held, release = threading.Event(), threading.Event()
    t = _hold(True, held, release)
    try:
        raised = False
        try:
            with h5_access(blocking=False):
                pass
        except H5Busy:
            raised = True
        assert raised, "GUI-side access must fail fast instead of blocking"
    finally:
        release.set()
        t.join(2)
    with h5_access(blocking=False):  # free again
        pass


def test_readers_share_but_a_writer_is_exclusive():
    held, release = threading.Event(), threading.Event()
    t = _hold(False, held, release)  # a background reader is active
    try:
        with h5_access(blocking=False):  # a second reader is fine (e.g. GUI peek)
            pass
        raised = False
        try:
            with h5_access(blocking=False, write=True):  # a writer must wait
                pass
        except H5Busy:
            raised = True
        assert raised, "writer must not start while a reader is active"
    finally:
        release.set()
        t.join(2)
    with h5_access(blocking=False, write=True):
        pass


def test_writers_are_serialised():
    order = []
    started = threading.Event()

    def writer(name):
        with h5_access(write=True):
            order.append(f"{name}-start")
            started.set()
            time.sleep(0.15)
            order.append(f"{name}-end")

    t1 = threading.Thread(target=writer, args=("w1",))
    t2 = threading.Thread(target=writer, args=("w2",))
    t1.start()
    started.wait(2)
    t2.start()
    t1.join(3)
    t2.join(3)
    assert order in (["w1-start", "w1-end", "w2-start", "w2-end"],
                     ["w2-start", "w2-end", "w1-start", "w1-end"]), order


def test_waiting_writer_gets_in_and_is_not_starved_by_new_readers():
    held, release = threading.Event(), threading.Event()
    reader = _hold(False, held, release)
    wrote = threading.Event()

    def writer():
        with h5_access(write=True):
            wrote.set()

    w = threading.Thread(target=writer)
    w.start()
    time.sleep(0.1)  # writer is now waiting for the active reader
    try:
        raised = False
        try:
            with h5_access(blocking=False):  # new reader must queue behind the writer
                pass
        except H5Busy:
            raised = True
        assert raised, "new readers must yield to a waiting writer"
    finally:
        release.set()
        reader.join(2)
    assert wrote.wait(2), "writer never got the lock"
    w.join(2)


def test_nested_access_does_not_deadlock():
    with h5_access():  # nested read inside read
        with h5_access():
            pass
    with h5_access(write=True):  # read and write inside our own write
        with h5_access():
            pass
        with h5_access(write=True):
            pass
    # nested read must not wait behind a waiting writer (classic RW-lock deadlock)
    held, release = threading.Event(), threading.Event()
    done = threading.Event()

    def outer_reader():
        with h5_access():
            held.set()
            release.wait()
            with h5_access():  # nested while a writer is waiting
                done.set()

    r = threading.Thread(target=outer_reader)
    r.start()
    assert held.wait(2)
    # a writer that waits for the outer reader (acquire + release in its own thread)
    def writer():
        with h5_access(write=True):
            pass

    w = threading.Thread(target=writer)
    w.start()
    time.sleep(0.1)
    release.set()
    assert done.wait(2), "nested read deadlocked behind a waiting writer"
    r.join(2)
    w.join(2)


def test_workers_get_a_private_metadata_snapshot_and_publish_by_assignment():
    import pandas as pd

    from behav3d.napari._data_preparation import DataPreparationTab

    _qt_app()
    tab = DataPreparationTab()
    live = pd.DataFrame({"sample_name": ["s1", "s2"], "tcell_tracks_image_path": ["", ""]})
    tab.metadata = live
    assert tab.metadata is live, "the GUI thread keeps the live object"

    seen = {}

    def worker():
        md = tab.metadata
        seen["is_copy"] = md is not live
        md.at[0, "tcell_tracks_image_path"] = "/in/place/edit"   # what the backends do
        seen["live_during"] = live.at[0, "tcell_tracks_image_path"]
        tab.metadata = md                                        # publish by assignment

    t = threading.Thread(target=worker)
    t.start()
    t.join(3)

    assert seen["is_copy"], "a worker must receive a snapshot, not the shared table"
    assert seen["live_during"] == "", "in-place edits must not leak into the table the GUI reads"
    assert tab.metadata.at[0, "tcell_tracks_image_path"] == "/in/place/edit", "assignment publishes"
    assert live.at[0, "tcell_tracks_image_path"] == "", "the original object was never mutated"


def test_pump_events_excludes_user_input_but_delivers_queued_work():
    """Synchronous GUI-thread loops that pump events must not let a click or key
    press re-enter a half-finished handler, yet queued signals/timers still run.

    (Qt only classifies *window-system* events as user input, so the flag can't be
    demonstrated with synthetic ``postEvent`` input; assert the contract instead:
    the flag is passed, and queued signal delivery still happens.)
    """
    from qtpy.QtCore import QEventLoop, QObject, Qt, Signal

    from behav3d.core import qt_events

    app = _qt_app()

    class Emitter(QObject):
        ping = Signal()

    got = []
    em = Emitter()
    em.ping.connect(lambda: got.append(1), Qt.QueuedConnection)
    em.ping.emit()
    qt_events.pump_events()
    assert got == [1], "queued signals must still be delivered while pumping"

    calls = []

    class FakeApp:
        def processEvents(self, *flags):
            calls.append(flags)

    class FakeQApplication:
        @staticmethod
        def instance():
            return FakeApp()

    real = qt_events.QApplication
    qt_events.QApplication = FakeQApplication
    try:
        qt_events.pump_events()
    finally:
        qt_events.QApplication = real
    assert calls == [(QEventLoop.ExcludeUserInputEvents,)], calls
    assert app is not None


if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)

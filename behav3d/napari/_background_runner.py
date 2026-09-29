"""Shared background-execution helpers for the BEHAV3D napari plugin.

Mirrors the pattern established by
:mod:`behav3d.napari._segment_editor` (see ``_EditWorker`` /
``_run_operation_async``) and makes it reusable from the Segmentation,
Tracking, Feature Extraction, and Filtering tabs.

Three concerns are packaged here:

1. :class:`_AsyncWorker` – a generic ``QObject`` that executes an arbitrary
   callable in a ``QThread`` and exposes ``done`` / ``failed`` / ``finished``
   / ``progress`` Qt signals.  The callable is invoked with an optional
   ``progress_cb`` keyword which, when called by the backend, drives both
   the in-tab progress widget and the napari activity-dock entry.

2. :class:`BackgroundOperation` – per-owner-widget orchestrator.  Wires
   the worker up to a :class:`ProgressBarRow`, manages the napari
   activity-dock entry, disables/restores a set of buttons while running,
   and routes ``done`` / ``failed`` back to the Qt thread.

3. :class:`ProgressBarRow` – the tiny ``QProgressBar`` + status ``QLabel``
   row each tab embeds once and reuses across operations.  Has explicit
   ``start`` / ``set_busy`` / ``update`` / ``finish`` helpers so calling
   code stays terse.

A :class:`ThreadSafeLogger` helper is included so backend functions that
accept a ``log_callback=`` can keep writing to the GUI ``QTextEdit`` from
the worker thread without touching Qt widgets directly.

This module is imported only by the Qt-based napari plugin tabs.  It is
intentionally **not** referenced by ``behav3d/widgets/*`` (the notebook
ipywidgets layer) so notebook behaviour is unaffected.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from qtpy.QtCore import QObject, QThread, QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

try:
    from napari.utils import progress as _NapariProgress
except Exception:  # pragma: no cover - napari < 0.5 or import error fallback
    _NapariProgress = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


__all__ = [
    "ProgressBarRow",
    "BackgroundOperation",
    "ThreadSafeLogger",
    "make_activity_progress",
    "update_activity_progress",
    "close_activity_progress",
    "show_napari_activity_panel",
    "hide_napari_activity_panel",
    "fire_extra_callback",
]


def fire_extra_callback(extra_callbacks, key: str, *args) -> None:
    """Best-effort fire of ``extra_callbacks[key](*args)``.

    The processing queue plumbs ``extra_callbacks={"on_done": cb,
    "on_failed": cb, "progress": cb}`` through every Run-batch method.
    Each method invokes this helper at the same point as its existing
    on_done / on_failed wrappers.  Default ``None`` keeps GUI / notebook
    behaviour untouched; bad callbacks never propagate.
    """
    if not extra_callbacks:
        return
    cb = extra_callbacks.get(key)
    if cb is None:
        return
    try:
        cb(*args)
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Tab-side progress widget
# ---------------------------------------------------------------------------
class ProgressBarRow(QWidget):
    """Tiny progress bar + status label embedded in a tab.

    The widget is hidden by default; call :meth:`start` to make it visible
    in busy/indeterminate mode, :meth:`update` to switch to determinate
    mode, and :meth:`finish` to hide it again.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color:#555; font-size:11px; min-width:140px;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # indeterminate by default
        self.bar.setMaximumHeight(14)
        self.bar.setTextVisible(True)
        lay.addWidget(self.lbl)
        lay.addWidget(self.bar, stretch=1)
        self.setVisible(False)
        # Wall-clock when the current run started (set in start/set_busy).
        # Used to derive an ETA shown on the label during update().
        self._start_time: Optional[float] = None

    # -- public API ----------------------------------------------------
    def start(self, desc: str = "Running…") -> None:
        """Show in busy/indeterminate mode."""
        self._start_time = time.time()
        self.lbl.setText(desc)
        self.bar.setRange(0, 0)
        self.bar.setValue(0)
        self.setVisible(True)

    def set_busy(self, desc: Optional[str] = None) -> None:
        """Switch back to indeterminate (busy) mode."""
        if self._start_time is None:
            self._start_time = time.time()
        if desc is not None:
            self.lbl.setText(desc)
        self.bar.setRange(0, 0)
        self.bar.setValue(0)
        self.setVisible(True)

    def update(self, current: int, total: int, label: str = "") -> None:
        """Switch to determinate mode and update the bar.

        If ``total <= 0`` the bar stays indeterminate (only the label is
        updated).  The label always gets a ``" — NN% — ETA H:MM:SS"`` suffix
        when both ``current > 0`` and ``total > 0`` so the user sees live
        percentage and estimated time even when the backend only emits a
        bare description.
        """
        if total > 0:
            if self.bar.maximum() != total:
                self.bar.setRange(0, total)
            self.bar.setValue(max(0, min(current, total)))
        else:
            self.bar.setRange(0, 0)

        base_label = label if label else self.lbl.text().split(" — ")[0]
        suffix = self._format_progress_suffix(current, total)
        full_label = f"{base_label}{suffix}" if suffix else base_label
        if full_label:
            self.lbl.setText(full_label)
        self.setVisible(True)

    def finish(self) -> None:
        """Hide the row."""
        self.setVisible(False)
        self.lbl.setText("")
        self.bar.setRange(0, 0)
        self.bar.setValue(0)
        self._start_time = None

    # -- helpers -------------------------------------------------------
    def _format_progress_suffix(self, current: int, total: int) -> str:
        """Return ``" — NN% — ETA H:MM:SS"`` (or just ``" — NN%"``)."""
        if total <= 0 or current < 0:
            return ""
        pct = int(round(100.0 * current / total))
        pct = max(0, min(pct, 100))
        if self._start_time is None or current <= 0:
            return f" — {pct}%"
        elapsed = max(0.0, time.time() - self._start_time)
        if elapsed <= 0:
            return f" — {pct}%"
        rate = current / elapsed
        if rate <= 0:
            return f" — {pct}%"
        remaining_s = max(0.0, (total - current) / rate)
        return f" — {pct}% — ETA {self._format_eta(remaining_s)}"

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds as ``H:MM:SS`` (or ``M:SS`` when under one hour)."""
        s = int(round(seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# Thread-safe log adapter
# ---------------------------------------------------------------------------
class ThreadSafeLogger(QObject):
    """Bridge a worker-thread ``log_callback`` to a Qt-thread sink.

    Backend functions that accept ``log_callback=`` may emit log lines from
    the worker thread; :class:`~qtpy.QtWidgets.QTextEdit` is not
    thread-safe so we must marshal those calls back to the Qt thread via a
    signal/slot.

    The instance itself is callable, mirroring the original
    ``log_callback(msg)`` interface so it is a drop-in replacement:

    .. code-block:: python

        logger = ThreadSafeLogger(self.log)
        run_pixel_classifier_segmentation(..., log_callback=logger)
    """

    log = Signal(str)

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__()
        self._sink = sink
        # Queue-connect so calls cross-thread; if the sink is on the Qt
        # thread the slot fires there.
        self.log.connect(self._dispatch, Qt.QueuedConnection)

    def _dispatch(self, msg: str) -> None:
        try:
            self._sink(msg)
        except Exception:
            # Never let a broken log handler take down a worker thread.
            traceback.print_exc()

    def __call__(self, msg: str) -> None:
        try:
            self.log.emit(str(msg))
        except Exception:
            # Fallback: write directly if the signal failed for some reason.
            try:
                self._sink(str(msg))
            except Exception:
                traceback.print_exc()


# ---------------------------------------------------------------------------
# napari activity-dock helpers (lifted from _segment_editor.py)
# ---------------------------------------------------------------------------
def make_activity_progress(viewer, desc: str = "Running…"):
    """Create an indeterminate napari activity-dock progress bar.

    ``total=0`` initially renders as a spinning busy indicator until the
    real total is set on the first progress tick via
    :func:`update_activity_progress`.

    Returns ``None`` when napari's ``progress`` helper is unavailable, so
    callers can no-op gracefully.
    """
    if _NapariProgress is None:
        return None
    try:
        pbr = _NapariProgress(total=0, desc=desc)
        if viewer is not None:
            show_napari_activity_panel(viewer)
        return pbr
    except Exception:
        return None


def show_napari_activity_panel(viewer) -> None:
    """Best-effort attempt to make the napari activity panel visible.

    The internal attribute name for the activity dock varies across napari
    versions; we try each in turn and fall back to a generic dock-search.
    """
    try:
        win = viewer.window._qt_window
    except Exception:
        return
    for attr in (
        "_activity_dialog",
        "_qt_activity",
        "_activity_dock",
    ):
        obj = getattr(win, attr, None)
        if obj is not None and hasattr(obj, "show"):
            try:
                obj.show()
                obj.raise_()
            except Exception:
                pass
            return
    try:
        from qtpy.QtWidgets import QDialog, QDockWidget
        for child in win.findChildren((QDockWidget, QDialog)):
            if "activity" in (child.objectName() or "").lower():
                child.show()
                child.raise_()
                return
    except Exception:
        pass


def hide_napari_activity_panel(viewer) -> None:
    """Best-effort attempt to hide/collapse the napari activity panel."""
    try:
        win = viewer.window._qt_window
    except Exception:
        return
    for attr in (
        "_activity_dialog",
        "_qt_activity",
        "_activity_dock",
    ):
        obj = getattr(win, attr, None)
        if obj is not None and hasattr(obj, "hide"):
            try:
                obj.hide()
            except Exception:
                pass
            return
    try:
        from qtpy.QtWidgets import QDialog, QDockWidget
        for child in win.findChildren((QDockWidget, QDialog)):
            if "activity" in (child.objectName() or "").lower():
                child.hide()
                return
    except Exception:
        pass


def update_activity_progress(pbr, current: int, total: int) -> None:
    """Update an existing activity-dock progress entry.

    First call re-initialises the bar with the correct ``total`` via
    ``reset()`` so the Qt widget's maximum is set properly and the
    percentage / ETA is meaningful.  Subsequent calls use ``update(delta)``
    (the canonical tqdm API) which internally refreshes the Qt widget.
    """
    if pbr is None:
        return
    try:
        if total > 0 and pbr.total != total:
            try:
                pbr.reset(total=total)
            except Exception:
                pbr.total = total
                pbr.n = 0
        delta = current - pbr.n
        if delta > 0:
            pbr.update(delta)
        elif delta < 0:
            pbr.n = current
            pbr.refresh()
    except Exception:
        pass


def close_activity_progress(pbr, viewer=None, *, delay_ms: int = 1000) -> None:
    """Fill the bar to 100 %, wait ``delay_ms`` ms, then close the entry."""
    if pbr is None:
        return
    try:
        if pbr.total and pbr.total > 0 and pbr.n < pbr.total:
            pbr.update(pbr.total - pbr.n)
    except Exception:
        pass

    def _do_close():
        try:
            pbr.close()
        except Exception:
            pass
        if viewer is not None:
            hide_napari_activity_panel(viewer)

    QTimer.singleShot(max(0, int(delay_ms)), _do_close)


# ---------------------------------------------------------------------------
# Thread-wait helper
# ---------------------------------------------------------------------------
def _wait_pumping(thread: QThread, timeout_ms: int) -> bool:
    """Like ``QThread.wait()``, but also pumps the calling thread's Qt event
    loop while waiting.

    ``BackgroundOperation.run()`` connects ``worker.finished`` (emitted on
    the *worker* thread) to ``thread.quit`` (a slot on the ``QThread``
    object, which lives on whichever thread created it). Qt resolves that
    as a queued, cross-thread connection, so ``thread.quit()`` is only
    actually invoked once the *creating* thread's event loop processes that
    queued call -- which is also what lets the worker thread's default
    ``run()`` (``exec_()``) return. A plain, non-pumping ``QThread.wait()``
    called from the creating thread (e.g. from an ``aboutToQuit`` handler,
    which does not itself pump) would therefore block for the full timeout
    even after the worker has already finished its work — see the
    docstring of :meth:`BackgroundOperation.wait` for where this is used.
    """
    app = QApplication.instance()
    if app is None:
        # No event loop to pump (e.g. a non-GUI context) -- best effort,
        # a plain wait is at least correct when nothing needs dispatching.
        try:
            return bool(thread.wait(timeout_ms if timeout_ms >= 0 else 0x7FFFFFFF))
        except Exception:
            return False
    unlimited = timeout_ms < 0
    deadline = None if unlimited else (time.monotonic() + timeout_ms / 1000.0)
    try:
        while True:
            if unlimited:
                slice_ms = 20
            else:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    return bool(thread.wait(0))
                slice_ms = max(0, min(20, remaining_ms))
            if thread.wait(slice_ms):
                return True
            app.processEvents()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Generic worker
# ---------------------------------------------------------------------------
class _AsyncWorker(QObject):
    """Run ``fn(*args, **kwargs)`` in a background ``QThread``.

    Emits, in order:
        - ``progress(current, total, label)`` zero or more times,
        - either ``done(result)`` or ``failed(error_msg)`` exactly once,
        - ``finished()`` exactly once (no arguments) so the owning
          ``QThread`` can be told to quit via a clean signal connection.

    ``progress_cb`` is injected into ``kwargs`` so callers can either:
        - accept it and emit it themselves, or
        - have the backend ignore it (it defaults to ``None`` everywhere
          else, so the call is safe even if the backend doesn't know
          about it — see :class:`BackgroundOperation` for the inject-only
          path that omits the kwarg).
    """

    done = Signal(object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(int, int, str)

    def __init__(self, fn: Callable[..., Any], args: tuple, kwargs: dict,
                 inject_progress: bool = True,
                 cancel_event: Optional[threading.Event] = None) -> None:
        super().__init__()
        self._fn = fn
        self._args = tuple(args or ())
        self._kwargs = dict(kwargs or {})
        if inject_progress:
            # Bind ``progress_cb`` to the signal's emit method so the
            # backend can call it freely from the worker thread.  Qt's
            # auto-connection to the Qt thread is handled at the
            # ``progress.connect(...)`` site in :class:`BackgroundOperation`.
            self._kwargs.setdefault("progress_cb", self._emit_progress)
        if cancel_event is not None:
            # Same injection pattern as progress_cb: the backend opts in by
            # accepting a ``cancel_check`` kwarg and polling it periodically.
            self._kwargs.setdefault("cancel_check", cancel_event.is_set)

    def _emit_progress(self, current: int, total: int, label: str = "") -> None:
        try:
            self.progress.emit(int(current), int(total), str(label))
        except Exception:
            # Never let a progress emit kill the worker.
            pass

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.done.emit(result)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Background-operation orchestrator
# ---------------------------------------------------------------------------
@dataclass
class _RunState:
    thread: QThread
    worker: _AsyncWorker
    on_done: Optional[Callable[[Any], None]]
    on_failed: Optional[Callable[[str], None]]
    activity_progress: Any
    buttons: List[QWidget]
    progress_row: Optional[ProgressBarRow]
    finish_delay_ms: int
    viewer: Any = None
    pending_result: Optional[Tuple[bool, Any]] = None
    cancel_event: Optional[threading.Event] = None
    desc: str = ""
    fn_name: str = ""


class BackgroundOperation(QObject):
    """Run a backend callable in the background with progress feedback.

    One instance is intended to be owned by a single widget (tab or sub-
    panel) and re-used across runs.  Only one operation may be active at a
    time on a given instance; calling :meth:`run` again while a previous
    operation is in flight prints a message and returns without starting
    the new run, so callers get a clear signal rather than silently
    dropping the request.

    Typical usage from a Qt button slot::

        self._bg.run(
            fn=self._run_tracking_for,
            args=(self.cell_type,),
            kwargs={"overwrite": True},
            desc=f"Tracking {self.cell_type}…",
            progress_row=self.tab.progress_row,
            buttons=[self.btn_run],
            viewer=self.viewer,
            on_done=self._on_tracking_done,
            on_failed=self._on_tracking_failed,
        )

    ``fn`` is invoked with ``progress_cb`` automatically injected (unless
    ``inject_progress=False``).  ``progress_cb(current, total, label)``
    drives both the in-tab progress row and the napari activity-dock
    entry.

    External observers (e.g. the processing queue) can subscribe to the
    public :attr:`progress` Qt signal to receive the same
    ``(current, total, label)`` tuples without owning the underlying
    worker.
    """

    progress = Signal(int, int, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[_RunState] = None
        # Threads that didn't stop within a cancel() timeout. Kept
        # referenced (never dereferenced) so Qt can't destroy a QThread
        # while its OS thread is still alive -- see cancel()'s docstring.
        self._zombie_threads: List[Tuple[QThread, _AsyncWorker]] = []

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        st = self._state
        # ``st.thread.isRunning()`` reflects Qt's own view of whether the OS
        # thread is still alive, not just whether we haven't cleared
        # ``self._state`` yet. ``QThread.start()`` flips Qt's internal
        # "running" flag synchronously, before the OS thread body begins
        # executing, so there is no false-negative window immediately after
        # ``run()`` below calls ``thread.start()``.
        return st is not None and st.thread is not None and st.thread.isRunning()

    # ------------------------------------------------------------------
    def run(
        self,
        fn: Callable[..., Any],
        args: Sequence[Any] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        desc: str = "Running…",
        progress_row: Optional[ProgressBarRow] = None,
        buttons: Optional[Iterable[QWidget]] = None,
        viewer=None,
        on_done: Optional[Callable[[Any], None]] = None,
        on_failed: Optional[Callable[[str], None]] = None,
        inject_progress: bool = True,
        inject_cancel_check: bool = False,
        indeterminate: bool = False,
        finish_delay_ms: int = 600,
    ) -> None:
        """Kick off a background run.

        Parameters
        ----------
        fn:
            Callable executed on the worker thread.  It receives
            ``*args, **kwargs``; when ``inject_progress=True`` (default) a
            ``progress_cb`` kwarg is added so the backend can report
            progress.
        desc:
            Short label shown in the in-tab progress row and the napari
            activity-dock entry.
        progress_row:
            Optional :class:`ProgressBarRow` to drive.  Tabs typically
            instantiate one of these once and pass it here.
        buttons:
            Iterable of ``QWidget`` to disable while the operation runs.
            They are re-enabled on completion / failure (including via
            crash).
        viewer:
            ``napari.Viewer`` used to surface / hide the activity dock.
        on_done / on_failed:
            Callbacks invoked on the Qt thread when the operation
            finishes.  ``on_done`` receives the value returned by ``fn``.
        inject_progress:
            When ``False``, ``progress_cb`` is not added to ``kwargs``
            (use this for backends that take other progress keywords or
            none at all).
        inject_cancel_check:
            When ``True``, a ``cancel_check`` kwarg (a zero-arg callable
            returning ``bool``) is added to ``kwargs`` so ``fn`` can poll it
            and return early when :meth:`cancel` is called. Opt-in and
            ``False`` by default: most existing backends don't accept this
            kwarg, so defaulting to inject would break them.
        indeterminate:
            When ``True``, force the in-tab progress row into busy mode
            even when the worker emits progress (useful for backends that
            don't expose a per-sample loop).
        finish_delay_ms:
            How long the activity-dock bar lingers at 100 % before
            disappearing.
        """
        if self.is_running():
            print("BackgroundOperation already running, ignoring request.", flush=True)
            return

        # self._state may still reference a previous run. Note that
        # is_running() above reads the same live QThread.isRunning() flag
        # we'd have to re-read here to decide whether it's still alive --
        # a second read can't tell us anything the first one didn't, so
        # gating this park step on "isRunning() is True" makes it
        # unreachable whenever is_running() has already told us False.
        # Instead, always disconnect and park whatever old_state exists:
        # parking an already-finished thread is a harmless no-op, but
        # letting the plain ``self._state = state`` reassignment below drop
        # the last Python reference to a thread that's still genuinely
        # alive is exactly what produces the fatal "QThread: Destroyed
        # while thread is still running" crash (see cancel()'s docstring).
        # Do this before touching any of *this* run's buttons/progress
        # row/activity dock so cleaning up the orphaned run can't stomp UI
        # objects the caller is reusing for the new run.
        old_state = self._state
        if old_state is not None:
            still_alive = old_state.thread is not None and old_state.thread.isRunning()
            if still_alive:
                logger.warning(
                    "BackgroundOperation.run: previous run (fn=%s) still "
                    "alive when starting new run (fn=%s) -- parking old "
                    "thread as zombie instead of dropping the last "
                    "reference to it.",
                    old_state.fn_name, getattr(fn, "__name__", repr(fn)),
                )
            else:
                logger.info(
                    "BackgroundOperation.run: previous run (fn=%s) state "
                    "was not cleared before starting new run (fn=%s) -- "
                    "parking its thread defensively in case isRunning() "
                    "misreported it as finished.",
                    old_state.fn_name, getattr(fn, "__name__", repr(fn)),
                )
            if old_state.cancel_event is not None:
                old_state.cancel_event.set()
            if old_state.thread is not None:
                for signal, slot in (
                    (old_state.thread.finished, self._on_thread_finished),
                    (old_state.worker.done, self._on_done),
                    (old_state.worker.failed, self._on_failed),
                    (old_state.worker.progress, self._on_progress),
                ):
                    try:
                        signal.disconnect(slot)
                    except (TypeError, RuntimeError):
                        pass
            self._cleanup_ui()
            if old_state.thread is not None:
                self._zombie_threads.append((old_state.thread, old_state.worker))
            self._state = None

        kwargs = dict(kwargs or {})
        btn_list = [b for b in (buttons or []) if b is not None]

        # Disable buttons up-front on the Qt thread.
        for b in btn_list:
            try:
                b.setEnabled(False)
            except Exception:
                pass

        # Tab-side progress row.
        if progress_row is not None:
            if indeterminate:
                progress_row.set_busy(desc)
            else:
                progress_row.start(desc)

        # napari activity-dock entry.
        activity_progress = make_activity_progress(viewer, desc=desc)

        # Worker + thread.
        cancel_event = threading.Event() if inject_cancel_check else None
        worker = _AsyncWorker(fn, tuple(args), kwargs,
                              inject_progress=inject_progress,
                              cancel_event=cancel_event)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)

        fn_name = getattr(fn, "__name__", repr(fn))
        logger.info(
            "BackgroundOperation run start: owner=%r fn=%s desc=%r",
            self.parent(), fn_name, desc,
        )

        state = _RunState(
            thread=thread,
            worker=worker,
            on_done=on_done,
            on_failed=on_failed,
            activity_progress=activity_progress,
            buttons=btn_list,
            progress_row=progress_row,
            finish_delay_ms=finish_delay_ms,
            cancel_event=cancel_event,
            desc=desc,
            fn_name=fn_name,
            viewer=viewer,
        )
        self._state = state

        # Progress routing — both the tab row and the activity dock.
        if not indeterminate:
            worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        worker.done.connect(self._on_done, Qt.QueuedConnection)
        worker.failed.connect(self._on_failed, Qt.QueuedConnection)
        thread.finished.connect(self._on_thread_finished, Qt.QueuedConnection)

        thread.start()

    # ------------------------------------------------------------------
    def _on_progress(self, current: int, total: int, label: str) -> None:
        st = self._state
        if st is None:
            return
        # Tab row
        if st.progress_row is not None:
            try:
                st.progress_row.update(int(current), int(total), str(label))
            except Exception:
                pass
        # Activity dock
        try:
            update_activity_progress(st.activity_progress, int(current), int(total))
        except Exception:
            pass
        # Re-emit on the public signal so external observers (e.g. the
        # processing queue) receive the same tuple.
        try:
            self.progress.emit(int(current), int(total), str(label))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _on_done(self, result: Any) -> None:
        st = self._state
        if st is None:
            return
        # Close UI feedback now so the callback's own dialogs / view updates
        # don't overlap with the activity bar being torn down. The user
        # callback itself is deferred to ``_on_thread_finished`` so that
        # ``is_running()`` has already flipped False by the time it runs —
        # see that method's docstring for why.
        self._cleanup_ui()
        st.pending_result = (True, result)

    # ------------------------------------------------------------------
    def _on_failed(self, error: str) -> None:
        st = self._state
        if st is None:
            return
        self._cleanup_ui()
        st.pending_result = (False, error)

    # ------------------------------------------------------------------
    def _cleanup_ui(self) -> None:
        st = self._state
        if st is None:
            return
        # Re-enable buttons.
        for b in st.buttons:
            try:
                b.setEnabled(True)
            except Exception:
                pass
        # Finish the in-tab progress row.
        if st.progress_row is not None:
            try:
                st.progress_row.finish()
            except Exception:
                pass
        # Close the activity-dock entry AND hide the dock itself so the
        # panel collapses after the run finishes.
        try:
            close_activity_progress(
                st.activity_progress,
                viewer=st.viewer,
                delay_ms=st.finish_delay_ms,
            )
        except Exception:
            pass
        st.activity_progress = None

    # ------------------------------------------------------------------
    def _on_thread_finished(self) -> None:
        """Called once the worker's QThread has fully stopped.

        Safe place to drop the Python references; clearing earlier (inside
        ``_on_done``/``_on_failed``) would release the last reference to
        ``QThread`` while the OS thread is still alive — a recipe for
        ``QThread: Destroyed while thread is still running`` and the
        resulting hard crash.

        The user's ``on_done``/``on_failed`` callback is also dispatched
        from here (after ``self._state`` is cleared) rather than from
        ``_on_done``/``_on_failed`` directly. Those fire as soon as the
        worker emits its result, which is *before* this method runs, so
        calling the user callback there would let a callback that starts
        another run via ``self._bg.run(...)`` (e.g. a "Run All" chain)
        observe ``is_running()`` as still ``True`` and self-skip.
        """
        st = self._state
        self._state = None
        if st is not None:
            logger.info(
                "BackgroundOperation run finished: owner=%r fn=%s",
                self.parent(), st.fn_name,
            )
        # Also the safest point to reclaim any pyplot figure the backend
        # opened and never closed: the worker is gone, so nothing can be
        # mid-draw.
        try:
            from behav3d.napari._matplotlib_guard import (
                close_leftover_pyplot_figures,
            )

            close_leftover_pyplot_figures()
        except Exception:
            pass
        if st is not None and st.pending_result is not None:
            ok, payload = st.pending_result
            cb = st.on_done if ok else st.on_failed
            if cb is not None:
                try:
                    cb(payload)
                except Exception:
                    traceback.print_exc()

    # ------------------------------------------------------------------
    def wait(self, timeout_ms: int = -1) -> bool:
        """Block the current thread until the operation completes.

        Returns ``True`` if the worker thread terminated within the
        timeout, ``False`` otherwise.  Useful only in tests / shutdown
        paths — interactive code should rely on the ``on_done`` callback.
        Pumps the calling thread's Qt event loop while waiting -- see
        :func:`_wait_pumping` for why that's required.
        """
        st = self._state
        if st is None or st.thread is None:
            return True
        return _wait_pumping(st.thread, timeout_ms)

    # ------------------------------------------------------------------
    def cancel(self, timeout_ms: int = 500) -> bool:
        """Best-effort stop of the in-flight run, safe to call at teardown.

        Requests cooperative cancellation (if the run opted in via
        ``inject_cancel_check``), disconnects this instance's Qt signal
        handlers so a straggling worker's eventual completion can't corrupt
        a later, unrelated run, then waits up to ``timeout_ms`` for the
        thread to actually stop.

        If it doesn't stop in time, the ``(thread, worker)`` pair is parked
        in ``self._zombie_threads`` instead of being dropped — mirroring
        ``_segment_editor.py``'s ``_cleanup()`` — because dropping the last
        Python reference to a still-running ``QThread`` is exactly what
        produces ``QThread: Destroyed while thread is still running``.
        Either way, ``self._state`` is cleared so this instance is free to
        start new work immediately; the scan-style backends this is
        designed for are pure/read-only, so an abandoned zombie run has no
        shared mutable state to corrupt.

        Returns ``True`` if the thread stopped cleanly within the timeout.
        """
        st = self._state
        if st is None:
            return True

        logger.info(
            "BackgroundOperation cancel requested: owner=%r fn=%s",
            self.parent(), st.fn_name,
        )

        if st.cancel_event is not None:
            st.cancel_event.set()

        for signal, slot in (
            (st.thread.finished, self._on_thread_finished),
            (st.worker.done, self._on_done),
            (st.worker.failed, self._on_failed),
            (st.worker.progress, self._on_progress),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        finished_cleanly = _wait_pumping(st.thread, timeout_ms)

        self._cleanup_ui()

        if finished_cleanly:
            logger.info(
                "BackgroundOperation cancel: fn=%s stopped cleanly within %dms",
                st.fn_name, timeout_ms,
            )
        else:
            logger.warning(
                "BackgroundOperation cancel: fn=%s did not stop within %dms, "
                "parking as zombie thread",
                st.fn_name, timeout_ms,
            )
            self._zombie_threads.append((st.thread, st.worker))

        self._state = None
        return finished_cleanly

    # ------------------------------------------------------------------
    def drain_zombies(self, timeout_ms: int = 0) -> int:
        """Give parked zombie threads a further bounded wait.

        Removes any that have since finished. Returns the number still
        alive. Intended for a second, longer-timeout pass at app-quit time
        (see ``BEHAV3DWidget._shutdown_background_operations``), after the
        initial ``cancel()`` sweep.
        """
        still_alive: List[Tuple[QThread, _AsyncWorker]] = []
        for thread, worker in self._zombie_threads:
            finished = _wait_pumping(thread, timeout_ms)
            if finished:
                logger.info("BackgroundOperation zombie thread finished: %r", worker)
            else:
                still_alive.append((thread, worker))
        self._zombie_threads = still_alive
        if still_alive:
            logger.warning(
                "BackgroundOperation: %d zombie thread(s) still running after drain",
                len(still_alive),
            )
        return len(still_alive)

"""Process-wide coordination of ``.h5ad`` / HDF5 file access.

The plugin reads and writes the same state/track ``.h5ad`` files from several
places at once: background workers that load the whole AnnData (``_preload_bg``),
the data-consistency scan, report workers, the analysis backends that rewrite the
file after a clustering run, and short column peeks on the GUI thread. Reading a
file while another thread rewrites it is what is dangerous in HDF5 (corrupt reads,
"unable to open file", or a native crash); concurrent *reads* are fine. An earlier
attempt made the consistency check synchronous for exactly this reason (commit
5f2a317).

This is a readers-writer lock over *all* such files:

* ``h5_access()`` / ``h5_access(write=False)`` -- shared; any number of readers.
* ``h5_access(write=True)`` -- exclusive; waits for readers to leave and blocks new
  ones (a waiting writer is not starved).
* **The GUI thread must never block on it**: use ``blocking=False``, which raises
  :class:`H5Busy` instead of waiting. Every GUI-side call site already treats a
  failed read as "no information" (``except Exception``), so a contended peek just
  yields an empty hint and is retried on the next refresh. Workers simply wait.

Re-entrant for a thread that already holds the write lock (it may also read). A
thread that holds a read lock must not ask for the write lock (upgrade): that can
deadlock against another reader doing the same, and nothing here needs it.

Deliberately a leaf module (stdlib only) so any layer can import it without
pulling in the scientific stack.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class H5Busy(RuntimeError):
    """Raised by ``h5_access(blocking=False)`` when the lock cannot be taken now."""


class _RWLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = None          # thread ident of the current writer
        self._write_depth = 0
        self._writers_waiting = 0
        self._local = threading.local()   # per-thread read depth (re-entrant reads)

    def acquire(self, write: bool, blocking: bool, timeout: float) -> bool:
        me = threading.get_ident()
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        with self._cond:
            if self._writer == me:                      # nested inside our own write
                self._write_depth += 1
                return True
            if not write and getattr(self._local, "reads", 0) > 0:
                # Nested read: must not wait behind a waiting writer, which is itself
                # waiting for this thread's outer read -> deadlock.
                self._local.reads += 1
                return True

            def can_go() -> bool:
                if write:
                    return self._writer is None and self._readers == 0
                # Readers yield to a *waiting* writer so it cannot starve.
                return self._writer is None and self._writers_waiting == 0

            if not can_go():
                if not blocking:
                    return False
                if write:
                    self._writers_waiting += 1
                try:
                    while not can_go():
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            return False
                        self._cond.wait(remaining)
                finally:
                    if write:
                        self._writers_waiting -= 1
            if write:
                self._writer = me
                self._write_depth = 1
            else:
                self._readers += 1
                self._local.reads = 1
            return True

    def release(self, write: bool) -> None:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._write_depth -= 1
                if self._write_depth == 0:
                    self._writer = None
                    self._cond.notify_all()
                return
            self._local.reads -= 1
            if self._local.reads > 0:                   # still inside an outer read
                return
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()


_LOCK = _RWLock()


@contextmanager
def h5_access(blocking: bool = True, timeout: float = -1, write: bool = False):
    """Hold the HDF5 access lock (shared, or exclusive with ``write=True``).

    ``blocking=False`` raises :class:`H5Busy` immediately if it cannot be taken
    (use on the GUI thread). ``timeout`` (seconds, ``-1`` = forever) bounds a
    blocking wait.
    """
    if not _LOCK.acquire(write, blocking, timeout):
        raise H5Busy(
            "an .h5ad file is being written by another task" if not write
            else "an .h5ad file is in use by another task"
        )
    try:
        yield
    finally:
        _LOCK.release(write)


def h5_read():
    """Shared access for a read: waits on worker threads, never blocks the main
    (GUI) thread -- there it raises :class:`H5Busy` if a writer holds the lock."""
    return h5_access(blocking=threading.current_thread() is not threading.main_thread())


def write_adata(adata, path, **kwargs) -> None:
    """``adata.write(path, **kwargs)`` under the exclusive HDF5 lock."""
    with h5_access(write=True):
        adata.write(path, **kwargs)


def read_h5ad_locked(path, **kwargs):
    """``anndata.read_h5ad(path, **kwargs)`` under the shared HDF5 lock.

    Note: with ``backed="r"`` the returned object keeps the file open after the
    lock is released; prefer reading fully, or hold :func:`h5_access` yourself
    around the whole use of the backed object.
    """
    import anndata as ad

    with h5_access():
        return ad.read_h5ad(str(path), **kwargs)

"""Persist native/Qt-level crash diagnostics for the napari plugin.

Both ways the plugin is started (the ``run_behav3d_*`` wrapper scripts and a
plain ``napari`` + Plugins-menu launch) leave a native crash — a Qt
``qFatal()``/``qWarning()`` such as "QThread: Destroyed while thread is
still running", or a raw SIGSEGV/SIGABRT from a C extension — with nothing
persisted: the process just dies, and whatever it printed only ever lived in
a terminal window that may already be gone. Two gaps, closed here:

* Qt's own diagnostic messages (``qDebug``/``qWarning``/``qCritical``/
  ``qFatal``) go through ``qInstallMessageHandler``, not Python's
  ``logging``/``warnings`` — nothing in the app captured them before.
* A hard native crash doesn't raise a catchable Python exception, so
  ``try/except`` anywhere is powerless; the *only* way to learn what each
  thread's Python code was doing at that instant is ``faulthandler``, which
  installs a signal handler for exactly this (SIGSEGV/SIGABRT/SIGBUS/
  SIGILL/SIGFPE) and dumps every live thread's Python stack before the
  process dies.

Both are wired to the same timestamped log file under the ``logs/``
directory at the root of the BEHAV3D-Explorer checkout, so a single file
has the full picture: Qt's own warning trail leading up to a crash,
followed by the Python-level stack of every thread at the moment it
happened.

A third gap this closes: regular ``logging.getLogger(__name__)`` calls made
anywhere under ``behav3d.*`` did not reach this file either (nothing in the
app ever attached a handler to any logger) -- so a crash like the QThread
one above left no trail of *why* it happened (which widget/dock was being
torn down, whether a background operation was mid-run), only *that* it
happened. ``install_crash_diagnostics`` now also attaches a handler to the
``"behav3d"`` logger, so lifecycle breadcrumbs logged by e.g.
``behav3d.napari._background_runner`` land in the same file, in the same
timeline, as the Qt/faulthandler output above.

A fourth gap: Python exceptions raised inside Qt slots. Under ``napari.run()``
napari replaces ``sys.excepthook`` with its own handler that turns the
exception into an on-screen notification bubble and does *not* call the
previous hook, so such a traceback never reached this file (the bubble is
dismissed and the evidence is gone). Outside ``napari.run()`` PyQt5 >= 5.5
instead calls ``qFatal()`` (-> abort) on an unhandled slot exception unless a
Python ``sys.excepthook`` is installed. This module therefore (a) installs
logging ``sys.excepthook``/``threading.excepthook`` that chain to the previous
hooks, and (b) subscribes to napari's ``notification_manager`` so every
exception *and warning* napari surfaces is also written here, with traceback.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
# Separate child logger so slot exceptions / napari notifications are easy to
# grep for ("behav3d.crash") in the shared log file.
_crash_logger = logging.getLogger("behav3d.crash")

# behav3d/napari/_crash_diagnostics.py -> repo root is three parents up.
# Editable-installed (pip install -e .), so __file__ resolves to the actual
# checkout, not a copy under site-packages.
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_hooks_installed = False
_notifications_hooked = False

# Kept at module scope: faulthandler and the Qt message handler both need
# this file object to stay open and referenced for the lifetime of the
# process, not just for the duration of installation.
_log_file = None


def _qt_message_type_name(msg_type) -> str:
    from qtpy.QtCore import QtMsgType

    return {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }.get(msg_type, str(msg_type))


def _make_qt_message_handler(log_file, previous_handler):
    def _handler(msg_type, context, message) -> None:
        # Never let a broken handler or write take the message (or the
        # process) down; this must be at least as safe as doing nothing.
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_file.write(f"[{ts}] Qt {_qt_message_type_name(msg_type)}: {message}\n")
            log_file.flush()
        except Exception:
            pass
        if previous_handler is not None:
            # Chain to whatever was installed before us (e.g. vispy's own
            # handler, which filters a blacklist of known-noisy messages and
            # logs the rest) so on-screen/logging behaviour is preserved
            # regardless of *when* we're installed relative to napari's own
            # Qt setup — see install_crash_diagnostics's docstring.
            try:
                previous_handler(msg_type, context, message)
                return
            except Exception:
                pass
        # No previous handler (or it failed) — fall back to stderr, Qt's
        # own default behaviour.
        print(message, file=sys.stderr)

    return _handler


def _install_python_excepthooks() -> None:
    """Log uncaught Python exceptions (main thread and worker threads) to the
    crash log, chaining to whatever hook was active before.

    Besides the logging, having *a* Python ``sys.excepthook`` installed keeps
    PyQt5 from calling ``qFatal()`` on an unhandled exception in a slot when the
    app is not running under ``napari.run()`` (which installs its own hook).
    """
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    previous_sys_hook = sys.excepthook

    def _sys_hook(exc_type, exc_value, exc_tb):
        try:
            _crash_logger.error(
                "Uncaught exception on main thread:\n%s",
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
        except Exception:
            pass
        try:
            previous_sys_hook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _sys_hook

    previous_thread_hook = getattr(threading, "excepthook", None)
    if previous_thread_hook is not None:

        def _thread_hook(args):
            try:
                name = getattr(args.thread, "name", "?")
                _crash_logger.error(
                    "Uncaught exception in thread %r:\n%s",
                    name,
                    "".join(
                        traceback.format_exception(
                            args.exc_type, args.exc_value, args.exc_traceback
                        )
                    ),
                )
            except Exception:
                pass
            try:
                previous_thread_hook(args)
            except Exception:
                pass

        threading.excepthook = _thread_hook


def _log_napari_notification(notification) -> None:
    """Write one napari notification (exception or warning) to the crash log."""
    try:
        severity = str(getattr(notification, "severity", "")).upper() or "INFO"
        exc = getattr(notification, "exception", None)
        if exc is not None:
            body = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            _crash_logger.error("napari notification [%s]:\n%s", severity, body)
        else:
            _crash_logger.warning("napari notification [%s]: %s", severity, notification)
    except Exception:
        pass


def install_notification_logging() -> bool:
    """Subscribe to napari's notification manager so every exception/warning it
    turns into a bubble is also persisted. Idempotent; returns True once hooked.

    napari is imported lazily and a failure is swallowed: this must never stop
    the plugin from loading.
    """
    global _notifications_hooked
    if _notifications_hooked:
        return True
    try:
        from napari.utils.notifications import notification_manager

        notification_manager.notification_ready.connect(_log_napari_notification)
        _notifications_hooked = True
    except Exception:
        logger.debug("Could not hook napari notifications", exc_info=True)
    return _notifications_hooked


def _log_environment_summary() -> None:
    """One-time header so a crash log is interpretable on its own."""
    try:
        from importlib import metadata

        def _ver(dist):
            try:
                return metadata.version(dist)
            except Exception:
                return "n/a"

        pkgs = (
            "napari", "vispy", "PyQt5", "PyQt6", "qtpy", "superqt", "magicgui",
            "numpy", "pandas", "numba", "zarr", "anndata", "scanpy", "torch",
        )
        _crash_logger.info(
            "Environment: python=%s platform=%s | %s",
            platform.python_version(),
            platform.platform(),
            " ".join(f"{d}={_ver(d)}" for d in pkgs),
        )
        _crash_logger.info(
            "Env vars: NUMBA_THREADING_LAYER=%s NUMBA_NUM_THREADS=%s "
            "NAPARI_CATCH_ERRORS=%s KMP_DUPLICATE_LIB_OK=%s",
            os.environ.get("NUMBA_THREADING_LAYER"),
            os.environ.get("NUMBA_NUM_THREADS"),
            os.environ.get("NAPARI_CATCH_ERRORS"),
            os.environ.get("KMP_DUPLICATE_LIB_OK"),
        )
    except Exception:
        pass


def install_crash_diagnostics() -> None:
    """Enable faulthandler + a Qt message logger, both writing to a
    timestamped file under the repo's ``logs/`` directory.

    Several dependencies (vispy's Qt canvas backend, in particular) install
    their *own* ``qInstallMessageHandler`` as a side effect of being
    imported — which can happen after this runs, silently replacing this
    handler. To stay the active one, this re-installs itself (idempotent,
    chains to whichever handler was active) each time it's called; callers
    on a launch path where napari/vispy import later than this module
    should call it again once napari's Qt canvas machinery has definitely
    loaded (e.g. right before ``napari.run()``).

    Never raises.
    """
    global _log_file
    try:
        if _log_file is None:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = _LOG_DIR / f"napari_{ts}.log"
            _log_file = open(log_path, "a", buffering=1)
            faulthandler.enable(file=_log_file, all_threads=True)
            print(f"[BEHAV3D] Crash diagnostics: logging to {log_path}", flush=True)

            # Attach to the "behav3d" logger (not the root logger) so every
            # behav3d.* module's logging.getLogger(__name__) call propagates
            # up into this file automatically, with no per-file wiring.
            behav3d_logger = logging.getLogger("behav3d")
            file_handler = logging.StreamHandler(_log_file)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s.%(msecs)03d] %(name)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            behav3d_logger.addHandler(file_handler)
            behav3d_logger.setLevel(logging.INFO)

            _log_environment_summary()
            _install_python_excepthooks()

        # Idempotent, and safe to retry on every call: napari may not have been
        # importable yet the first time round on some launch paths.
        install_notification_logging()

        from qtpy.QtCore import qInstallMessageHandler

        previous_handler = qInstallMessageHandler(None)  # peek without changing it
        qInstallMessageHandler(_make_qt_message_handler(_log_file, previous_handler))
    except Exception:
        logger.debug("Could not install crash diagnostics", exc_info=True)

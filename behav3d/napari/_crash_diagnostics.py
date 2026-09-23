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

Both are wired to the same timestamped log file under ``~/.behav3d/logs/``
so a single file has the full picture: Qt's own warning trail leading up to
a crash, followed by the Python-level stack of every thread at the moment
it happened.
"""
from __future__ import annotations

import faulthandler
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_DIR = Path.home() / ".behav3d" / "logs"

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


def install_crash_diagnostics() -> None:
    """Enable faulthandler + a Qt message logger, both writing to a
    timestamped file under ``~/.behav3d/logs/``.

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

        from qtpy.QtCore import qInstallMessageHandler

        previous_handler = qInstallMessageHandler(None)  # peek without changing it
        qInstallMessageHandler(_make_qt_message_handler(_log_file, previous_handler))
    except Exception:
        logger.debug("Could not install crash diagnostics", exc_info=True)

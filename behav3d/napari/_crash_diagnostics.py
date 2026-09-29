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
"""
from __future__ import annotations

import faulthandler
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# behav3d/napari/_crash_diagnostics.py -> repo root is three parents up.
# Editable-installed (pip install -e .), so __file__ resolves to the actual
# checkout, not a copy under site-packages.
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

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

        from qtpy.QtCore import qInstallMessageHandler

        previous_handler = qInstallMessageHandler(None)  # peek without changing it
        qInstallMessageHandler(_make_qt_message_handler(_log_file, previous_handler))
    except Exception:
        logger.debug("Could not install crash diagnostics", exc_info=True)

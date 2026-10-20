"""Keep Matplotlib off the Qt GUI thread inside the napari plugin.

BEHAV3D's analysis backends are notebook-derived: they draw through ``pyplot``
and call ``plt.show()`` (``behav3d.analysis.organoid_analysis`` and friends).
The plugin runs those same functions in background ``QThread``s via
``behav3d.napari._background_runner``, and with Matplotlib's default Qt backend
every ``plt.figure`` / ``plt.subplots`` there builds a Qt figure-manager
*window* off the GUI thread. That is the source of both symptoms reported on
the Analysis tab:

* ``pyplot._warn_if_gui_out_of_main_thread`` warns "Starting a Matplotlib GUI
  outside of the main thread will likely fail" once per figure, and napari
  turns each warning into a notification bubble.
* The Qt window really was built on the wrong thread, so the process is fragile
  from then on — dismissing one of those bubbles takes napari down.

The plugin never needs pyplot's own windows. Figures shown to the user are
built as bare ``Figure`` objects (see ``build_feature_distribution_figure`` /
``build_track_length_distribution_figure`` in ``behav3d.widgets.utils``) and
embedded by :func:`behav3d.napari._pdf_view.show_matplotlib_figure`, which
attaches its own ``FigureCanvasQTAgg`` on the main thread regardless of the
pyplot backend; everything else is written to PDF/PNG. So we pin the process to
the non-interactive ``Agg`` backend, which never constructs a figure-manager
window and therefore never warns.

This is scoped to the napari application — it is installed from this package's
``__init__``, so importing ``behav3d`` in a notebook keeps the interactive
backend and inline ``plt.show()`` behaviour it has always had. An explicit
``MPLBACKEND`` in the environment is honoured and left alone.
"""
import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)


def _pin_agg_backend():
    """Select ``Agg`` unless the user pinned a backend via ``MPLBACKEND``."""
    if os.environ.get("MPLBACKEND"):
        logger.debug(
            "MPLBACKEND=%s set explicitly; leaving the Matplotlib backend alone",
            os.environ["MPLBACKEND"],
        )
        return

    # Covers the common case where Matplotlib has not been imported yet (the
    # analysis backends import it lazily), without paying for the import here.
    os.environ["MPLBACKEND"] = "Agg"

    # ...and the case where something already pulled it in with a Qt backend.
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is None:
        return
    try:
        if matplotlib.get_backend().lower() != "agg":
            # force=True also switches an already-imported pyplot. It closes any
            # open figure, which is harmless at plugin-load time.
            matplotlib.use("Agg", force=True)
    except Exception:
        logger.debug("Could not switch Matplotlib to the Agg backend", exc_info=True)


def _silence_leftover_show_warnings():
    """Drop the warnings the leftover notebook ``plt.show()`` calls produce.

    Under ``Agg``, ``plt.show()`` raises ``NonGuiException`` internally and
    ``backend_bases._Backend.show`` re-emits it as "FigureCanvasAgg is
    non-interactive, and thus cannot be shown" — that would just replace one
    notification storm with another.

    Both messages are emitted with ``_api.warn_external``, i.e. plain
    ``UserWarning``s, so a message filter is enough and it stays in place for
    the whole process. The GUI-thread one should no longer fire once ``Agg`` is
    pinned; it is filtered anyway so that a figure drawn under a backend we did
    not choose (a user switching backends from napari's console, say) cannot
    bring the crash back as a notification.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*is non-interactive, and thus cannot be shown.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Starting a Matplotlib GUI outside of the main thread.*",
        category=UserWarning,
    )


def install_matplotlib_guard():
    """Apply the Matplotlib workarounds. Safe to call more than once."""
    try:
        _pin_agg_backend()
        _silence_leftover_show_warnings()
    except Exception:
        # Never let this break loading the plugin; worst case the warnings and
        # the off-thread figure windows come back.
        logger.debug("Could not install the Matplotlib guard", exc_info=True)


def close_leftover_pyplot_figures():
    """Close pyplot-managed figures left behind by a background analysis run.

    Under ``Agg`` (as under a non-blocking Qt ``plt.show()``) a figure the
    backends forget to ``plt.close`` stays registered in pyplot's figure
    manager forever, so a long GUI session slowly accumulates them. Called from
    ``_background_runner`` once a worker thread has fully stopped.

    Only touches figures pyplot itself owns. The figures the plugin shows the
    user are built as bare ``Figure`` objects and handed to
    ``show_matplotlib_figure``, so they are never registered here and are not
    affected. No-op when pyplot was never imported.
    """
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return
    try:
        pyplot.close("all")
    except Exception:
        logger.debug("Could not close leftover pyplot figures", exc_info=True)

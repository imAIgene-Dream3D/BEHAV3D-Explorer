"""Runtime workarounds for napari bugs that affect BEHAV3D viewers.

Patches are applied to napari classes, so they take effect for every viewer in the
process, including layers that already exist. Installed from this package's
``__init__``.
"""
import functools
import logging

logger = logging.getLogger(__name__)

# Marks an already-patched callable so re-installing the patches is a no-op.
_GUARD_FLAG = "_behav3d_patch_guard"


def _patch_labels_thumbnail():
    """Skip Labels thumbnail updates whose cached slice has a stale ``ndisplay``.

    napari#7918. ``Layer._slice_dims`` always updates ``_slice_input`` (which
    carries ``ndisplay``), but ``_refresh_sync`` returns early for invisible
    layers, so a hidden layer keeps the slice it was last computed at. In 3D,
    ``Labels._update_thumbnail`` trusts ``_slice_input.ndisplay`` and max-projects
    that stale slice down to 1-D, which ``scipy.ndimage.zoom`` then rejects with
    "sequence argument must have length equal to input rank". Any opacity write
    reaches it, and napari's layer controls set opacity whenever they are built,
    so selecting a hidden Labels layer in 3D is enough to raise.

    Skipping matches napari's own policy of not slicing hidden layers: ``refresh``
    recomputes the thumbnail as soon as the layer is shown again.

    napari's own fix (PR #8251, 0.6.5) only guards ``_slice.empty``, so it still
    raises for a layer that was sliced while visible and then hidden. Comparing
    ``ndisplay`` covers that case too, and stays correct once we move off 0.5.6.
    """
    from napari.layers import Labels

    original = Labels._update_thumbnail
    if getattr(original, _GUARD_FLAG, False):
        return

    @functools.wraps(original)
    def _update_thumbnail(self):
        cached = getattr(getattr(self, "_slice", None), "slice_input", None)
        current = getattr(self, "_slice_input", None)
        if cached is not None and current is not None:
            if cached.ndisplay != current.ndisplay:
                return
        original(self)

    setattr(_update_thumbnail, _GUARD_FLAG, True)
    Labels._update_thumbnail = _update_thumbnail


def _patch_notification_timers():
    """Make a notification's self-destruct timer tolerate a deleted bubble.

    ``NapariQtNotification`` walks its siblings with
    ``parent().findChildren(...)`` when a bubble is shown, hovered or closed,
    and calls ``timer_start`` / ``timer_stop`` on what it finds. Bubbles are
    ``WA_DeleteOnClose``, so a sibling closed earlier in the same event-loop
    turn can still be listed while its C++ object is already gone; touching its
    timer then raises ``RuntimeError`` (PyQt) or takes the process down
    (PySide). The timer is purely cosmetic, so dropping the call is fine.
    """
    from napari._qt.dialogs.qt_notification import NapariQtNotification

    for name in ("timer_start", "timer_stop"):
        original = getattr(NapariQtNotification, name)
        if getattr(original, _GUARD_FLAG, False):
            continue

        @functools.wraps(original)
        def guarded(self, _original=original, _name=name):
            try:
                return _original(self)
            except RuntimeError:
                logger.debug(
                    "napari notification %s on a deleted bubble; ignoring",
                    _name,
                    exc_info=True,
                )
                return None

        setattr(guarded, _GUARD_FLAG, True)
        setattr(NapariQtNotification, name, guarded)


def _patch_notification_close():
    """Make dismissing a notification bubble survive a deleted sibling.

    Second line of defence behind :func:`_patch_notification_timers`: whatever
    else ``close`` trips over while poking its siblings, the bubble the user
    clicked must still close. Without this the dialog stays on screen (or the
    process dies) because the exception escapes before ``QDialog.close``.
    """
    from qtpy.QtWidgets import QDialog
    from napari._qt.dialogs.qt_notification import NapariQtNotification

    original = NapariQtNotification.close
    if getattr(original, _GUARD_FLAG, False):
        return

    @functools.wraps(original)
    def close(self):
        try:
            original(self)
        except RuntimeError:
            logger.debug(
                "napari notification close hit a deleted sibling; "
                "closing the bubble directly",
                exc_info=True,
            )
            # napari's close stops these first; it may not have got that far.
            for attr in ("timer", "opacity_anim", "geom_anim"):
                try:
                    getattr(self, attr).stop()
                except (AttributeError, RuntimeError):
                    pass
            QDialog.close(self)

    setattr(close, _GUARD_FLAG, True)
    NapariQtNotification.close = close


def install_napari_patches():
    """Apply all napari workarounds. Safe to call more than once."""
    try:
        _patch_labels_thumbnail()
    except Exception:
        # Never let a workaround break importing the plugin: if napari is absent
        # (headless behav3d use) or its internals moved, the bug just resurfaces.
        logger.debug("Could not install napari Labels thumbnail guard", exc_info=True)

    try:
        _patch_notification_timers()
        _patch_notification_close()
    except Exception:
        logger.debug("Could not install napari notification close guards", exc_info=True)

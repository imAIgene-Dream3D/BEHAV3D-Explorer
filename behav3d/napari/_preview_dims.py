"""Shared helpers for tabs that keep a live, viewer-time-slider-driven
preview layer on the single shared ``napari.Viewer``.

Several tabs (State Classification, Track Classification, Feature
Backprojection, Feature Extraction's Dead/Alive + organoid previews) connect
a callback to ``viewer.dims.events.current_step`` so their overlay layer
recomputes as the user scrubs the time slider. Because every tab shares the
*same* ``Viewer`` instance (see ``behav3d.napari._widget.BEHAV3DWidget``), a
listener left connected by one tab can fire while another tab is bulk-
mutating ``viewer.layers`` (e.g. ``viewer.layers.clear()``), reacting by
re-adding a layer reentrantly. That out-of-band mutation, happening mid-
iteration, desyncs napari's own vispy canvas bookkeeping and raises a
``KeyError`` deep inside ``_vispy/canvas.py::_reorder_layers``.

This module centralizes the "only known-live listeners should ever be
connected, and any of them can be dropped in one call" bookkeeping that
``_feature_extraction.py`` originally implemented for its own two panels
(see its former ``_ACTIVE_PREVIEW_DIMS``), generalized to cover every
listener owner across every tab.

Usage, per preview owner:

    from behav3d.napari._preview_dims import (
        register_preview_dims_listener,
        unregister_preview_dims_listener,
    )

    def _connect_dims_listener(self):
        ...
        self.viewer.dims.events.current_step.connect(_on_step)
        register_preview_dims_listener(self.viewer, self, _on_step)

    def _disconnect_dims_listener(self):
        ...
        self.viewer.dims.events.current_step.disconnect(self._dims_callback)
        unregister_preview_dims_listener(self.viewer, self)

Usage, before any bulk layer mutation (``viewer.layers.clear()`` or
equivalent):

    from behav3d.napari._preview_dims import (
        clear_viewer_layers,
        disconnect_all_preview_dims_listeners,
    )

    disconnect_all_preview_dims_listeners(self.viewer)
    clear_viewer_layers(self.viewer)

Always clear through ``clear_viewer_layers`` rather than
``viewer.layers.clear()`` — see that function for why a raising listener
otherwise leaves the viewer half-cleared.
"""
from __future__ import annotations

# Maps id(viewer) -> {owner_key: callback}. ``owner_key`` is typically the
# owning tab/panel instance (``self``), so a given owner can register at
# most one listener per viewer at a time, while distinct owners' listeners
# are tracked independently.
_ACTIVE_PREVIEW_DIMS: dict = {}


def register_preview_dims_listener(viewer, owner_key, callback) -> None:
    """Record ``callback`` as ``owner_key``'s live preview dims listener on
    ``viewer``. Does not connect it — the caller is expected to have already
    called ``viewer.dims.events.current_step.connect(callback)``."""
    if viewer is None:
        return
    _ACTIVE_PREVIEW_DIMS.setdefault(id(viewer), {})[owner_key] = callback


def unregister_preview_dims_listener(viewer, owner_key) -> None:
    """Drop the bookkeeping entry for ``owner_key`` on ``viewer``. Does not
    disconnect anything — the caller handles that."""
    if viewer is None:
        return
    owners = _ACTIVE_PREVIEW_DIMS.get(id(viewer))
    if owners is None:
        return
    owners.pop(owner_key, None)
    if not owners:
        _ACTIVE_PREVIEW_DIMS.pop(id(viewer), None)


def disconnect_all_preview_dims_listeners(viewer) -> None:
    """Disconnect and forget every preview dims listener currently
    registered on ``viewer``, regardless of which tab/panel owns it.

    Call this immediately before any ``viewer.layers.clear()`` (or other
    bulk layer replacement) so no stale listener can fire reentrantly while
    the layer list is mid-mutation.
    """
    if viewer is None:
        return
    owners = _ACTIVE_PREVIEW_DIMS.pop(id(viewer), None)
    if not owners:
        return
    for callback in owners.values():
        try:
            viewer.dims.events.current_step.disconnect(callback)
        except Exception:
            pass


def clear_viewer_layers(viewer, log=None) -> int:
    """Remove every layer without letting one bad layer abort the rest.

    ``viewer.layers.clear()`` is ``MutableSequence.clear()`` — a bare
    ``pop()`` loop. napari removes a layer from the list *before* it emits
    ``removed`` (see ``_evented_list.__delitem__``), so when any listener
    raises, that layer is already gone but the loop unwinds and every
    remaining layer stays in the viewer.

    The listener that raises in practice is napari's own
    ``QtLayerControlsContainer._remove``, which does ``self.widgets[layer]``
    and throws ``KeyError`` for any layer whose controls widget was never
    registered (its ``_add`` failed earlier, e.g. behind a swallowed
    exception). The half-cleared viewer then leaves napari's Qt layer-list
    model stale, ``LayerDelegate.paint`` raises on every repaint, and
    raising repeatedly inside Qt's paint path takes the whole process down
    with an access violation.

    Removing one layer at a time, each guarded, keeps a single problem layer
    from stopping the teardown. Returns the number of layers left behind.
    """
    try:
        layers = list(viewer.layers)
    except Exception:
        return 0

    failed = []
    for layer in layers:
        try:
            viewer.layers.remove(layer)
        except Exception as exc:
            # napari pops before it notifies, so the layer is usually already
            # out of the list and only the notification failed. Keep going.
            failed.append((getattr(layer, "name", "?"), exc))

    try:
        remaining = len(viewer.layers)
    except Exception:
        remaining = 0

    if failed or remaining:
        emit = log if callable(log) else print
        for name, exc in failed:
            try:
                emit(f"  Layer '{name}' raised while being removed: {exc}")
            except Exception:
                pass
        if remaining:
            try:
                emit(f"  {remaining} layer(s) could not be removed from the viewer.")
            except Exception:
                pass
    return remaining


def stop_dim_playback(viewer) -> None:
    """Best-effort stop of any running napari dims animation.

    Clearing/replacing layers while a 3-D movie is playing destroys the
    ``QtDimSliderWidget`` objects the animation thread still references,
    raising ``RuntimeError: wrapped C/C++ object … has been deleted``. We
    stop playback via the public ``QtDims.stop()`` and also quit the
    animation thread directly, since depending on the napari version either
    (or neither) attribute may be present while a movie is live.
    """
    try:
        qt_dims = viewer.window._qt_viewer.dims
    except Exception:
        return
    # Public stop first (handles the current napari playback implementation).
    try:
        qt_dims.stop()
    except Exception:
        pass
    # Fall back to tearing down the animation thread if it is still around.
    try:
        thread = getattr(qt_dims, "_animation_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait()
    except Exception:
        pass


def close_backprojection_legend_docks(viewer) -> None:
    """Close any open state/track backprojection legend or statebar dock.

    State and track backprojection (``StateClassificationSubTab``,
    ``TrackClassificationSubTab`` in ``_single_cell.py``) each add one or two
    plain napari dock widgets alongside the preview layers: a cluster-color
    mapping legend, and for tracks a click-driven statebar dock. Unlike the
    preview layers themselves (handled by ``clear_viewer_layers`` above),
    nothing tracked these docks well enough to remove a stale one before
    adding a fresh one, so re-running backprojection — or navigating to a
    different tab and back — used to stack duplicates indefinitely.

    Napari's ``Viewer`` is a pydantic ``EventedModel`` and rejects arbitrary
    attribute assignment, so callers tag each dock's inner ``QWidget`` with
    ``_behav3d_backprojection_legend_dock = True`` at creation time (same
    trick used by ``behav3d/napari/_pdf_view.py:_install_result_dock``)
    instead of storing a reference on the viewer or on the owning tab. This
    function scans ``viewer.window._dock_widgets`` (private napari API, same
    as ``_install_result_dock``) for that tag and removes every match, so it
    finds stale docks even if the tab instance that created them is no
    longer reachable.

    The track statebar dock also registers a mouse-click callback directly
    on a viewer layer (``clickable_layer.mouse_drag_callbacks``). That layer
    normally outlives the dock once this function can close the dock while
    leaving the backprojection layers in place (e.g. on tab navigation), so
    the callback is dropped here too — otherwise a later click would call
    into Qt widgets whose C++ side this removal already deleted.
    """
    if viewer is None:
        return
    try:
        docks = list(viewer.window._dock_widgets.values())
    except Exception:
        return
    for dock in docks:
        try:
            inner = dock.widget()
        except Exception:
            continue
        if not getattr(inner, "_behav3d_backprojection_legend_dock", False):
            continue
        statebar_info = getattr(inner, "_behav3d_track_statebar_dock", None)
        if isinstance(statebar_info, dict):
            layer = statebar_info.get("clickable_layer")
            callback = statebar_info.get("callback")
            if layer is not None and callback is not None:
                try:
                    layer.mouse_drag_callbacks.remove(callback)
                except Exception:
                    pass
        try:
            viewer.window.remove_dock_widget(dock)
        except Exception:
            pass

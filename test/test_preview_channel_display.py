"""Dead-threshold preview restores / records raw-channel display settings."""
from types import SimpleNamespace

import dask.array as da
import numpy as np

from behav3d.napari._feature_extraction import _add_channel_layers
from behav3d.napari._loaders import saved_channel_display, snapshot_channel_display


class _Event:
    def __init__(self):
        self.callbacks = []

    def connect(self, cb):
        self.callbacks.append(cb)


class _Layer:
    def __init__(self, name, colormap):
        self.name = name
        self.colormap = SimpleNamespace(name=colormap)
        self.contrast_limits = (0.0, 1.0)
        self.events = SimpleNamespace(contrast_limits=_Event(), colormap=_Event())


class _Viewer:
    def __init__(self):
        self.layers = []

    def add_image(self, data, name, colormap, **_kwargs):
        layer = _Layer(name, colormap)
        self.layers.append(layer)
        return layer


def _img(n_channels=2):
    return da.from_array(np.ones((2, n_channels, 1, 4, 4), dtype=np.uint16) * 100)


def test_uses_saved_colormap_and_contrast_limits():
    params = {"viewer_display": {"channels": {0: {"colormap": "red", "contrast_limits": [5.0, 50.0]}}}}
    viewer = _Viewer()

    _add_channel_layers(viewer, _img(), "s1", params=params)

    assert viewer.layers[0].colormap.name == "red"
    assert tuple(viewer.layers[0].contrast_limits) == (5.0, 50.0)


def test_string_keys_from_yaml_are_honoured():
    params = {"viewer_display": {"channels": {"1": {"colormap": "green", "contrast_limits": [1, 9]}}}}
    viewer = _Viewer()

    _add_channel_layers(viewer, _img(), "s1", params=params)

    assert viewer.layers[1].colormap.name == "green"


def test_falls_back_to_defaults_without_saved_settings():
    viewer = _Viewer()

    _add_channel_layers(viewer, _img(), "s1", params={})

    assert [l.colormap.name for l in viewer.layers] == ["cyan", "yellow"]
    assert tuple(viewer.layers[0].contrast_limits) == (0, 100.0)


def test_display_callback_connected_without_firing_on_restore():
    calls = []
    viewer = _Viewer()

    _add_channel_layers(viewer, _img(1), "s1", params={}, on_display_changed=calls.append)

    layer = viewer.layers[0]
    assert len(layer.events.contrast_limits.callbacks) == 1
    assert len(layer.events.colormap.callbacks) == 1
    assert calls == []  # restoring must not trigger a save


def test_snapshot_round_trips_preview_layer_names():
    viewer = _Viewer()
    _add_channel_layers(viewer, _img(), "s1", params={})
    viewer.layers[1].contrast_limits = (2.0, 70.0)
    params = {}

    snapshot_channel_display(viewer, params)

    saved = saved_channel_display(params, 1)
    assert saved == {"colormap": "yellow", "contrast_limits": [2.0, 70.0]}

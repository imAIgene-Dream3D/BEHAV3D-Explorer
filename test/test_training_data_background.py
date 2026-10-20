"""
Tests for the background "Generate Training Data" plumbing
(behav3d/napari/_training_data.py) and the Qt-free multicolor cleanup.

* ``run_training_data_job``: the builder runs off the GUI thread, ``apply`` runs on
  it, a failing builder / stale dataset / empty payload never reaches the viewer
  and always reports to the queue hook.
* ``add_layer_specs``: layers are added in order and the tag hooks fire.
* ``run_multicolor_cleanup``: ``touched_cell_types`` limits the families processed.

Runs under pytest *or* standalone. Requires the `behav3d` conda env (PyQt5/qtpy).
Mirrors test_background_operation_teardown.py's QApplication handling.
"""
import threading
import time

import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication, QWidget

from behav3d.napari._background_runner import BackgroundOperation
from behav3d.napari import _training_data as td

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


class _Page(QWidget):
    """Minimal stand-in for the pixel-classifier pages."""

    def __init__(self):
        _qt_app()
        super().__init__()
        self._bg = BackgroundOperation(self)
        self.viewer = None
        self.messages = []

    def log(self, msg):
        self.messages.append(str(msg))


def _hooks():
    events = []
    return events, {
        "on_done": lambda *_: events.append("done"),
        "on_failed": lambda err: events.append(("failed", err)),
    }


def test_builder_runs_off_gui_thread_and_apply_on_it():
    page = _Page()
    events, hooks = _hooks()
    seen = {}
    gui = threading.current_thread()

    def build_payload(snap, log, progress_cb):
        seen["builder_thread"] = threading.current_thread()
        progress_cb(1, 2, "half way")
        return {"value": snap["x"]}

    def apply(payload):
        seen["apply_thread"] = threading.current_thread()
        seen["payload"] = payload
        return True

    td.run_training_data_job(
        page, desc="t", builder=build_payload, snap={"x": 7}, apply=apply,
        extra_callbacks=hooks,
    )
    assert _pump_until(lambda: events)
    assert events == ["done"]
    assert seen["builder_thread"] is not gui
    assert seen["apply_thread"] is gui
    assert seen["payload"] == {"value": 7}


def test_failing_builder_reports_failure_and_never_applies():
    page = _Page()
    events, hooks = _hooks()
    applied = []

    def build_payload(snap, log, progress_cb):
        raise RuntimeError("disk on fire")

    td.run_training_data_job(
        page, desc="t", builder=build_payload, snap={}, apply=lambda p: applied.append(p),
        extra_callbacks=hooks,
    )
    assert _pump_until(lambda: events)
    assert events[0][0] == "failed" and "disk on fire" in events[0][1]
    assert applied == []
    assert any("disk on fire" in m for m in page.messages)


def test_stale_dataset_payload_is_dropped():
    page = _Page()
    events, hooks = _hooks()
    applied = []
    td.run_training_data_job(
        page, desc="t", builder=lambda s, l, p: {"ok": 1}, snap={},
        apply=lambda p: applied.append(p), extra_callbacks=hooks,
        stale_check=lambda: True,
    )
    assert _pump_until(lambda: events)
    assert events[0][0] == "failed"
    assert applied == []


def test_empty_payload_counts_as_failure_for_the_queue():
    page = _Page()
    events, hooks = _hooks()
    td.run_training_data_job(
        page, desc="t", builder=lambda s, l, p: {"empty": "nothing"}, snap={},
        apply=lambda p: False, extra_callbacks=hooks,
    )
    assert _pump_until(lambda: events)
    assert events == [("failed", "no training data")]


def test_apply_exception_is_logged_and_reported():
    page = _Page()
    events, hooks = _hooks()

    def apply(payload):
        raise ValueError("viewer exploded")

    td.run_training_data_job(
        page, desc="t", builder=lambda s, l, p: {}, snap={}, apply=apply,
        extra_callbacks=hooks,
    )
    assert _pump_until(lambda: events)
    assert events[0][0] == "failed"
    assert any("viewer exploded" in m for m in page.messages)


def test_block_mode_runs_inline_without_a_thread():
    page = _Page()
    events, hooks = _hooks()
    threads = []

    def build_payload(snap, log, progress_cb):
        threads.append(threading.current_thread())
        return {}

    td.run_training_data_job(
        page, desc="t", builder=build_payload, snap={}, apply=lambda p: True,
        extra_callbacks=hooks, block=True,
    )
    assert events == ["done"]  # no event loop needed
    assert threads == [threading.current_thread()]


def test_training_data_key_changes_with_dataset():
    class Loader:
        output_dir = "/out"
        metadata = pd.DataFrame({"sample_name": ["a", "b"]})

    k1 = td.training_data_key(Loader)
    Loader.metadata = pd.DataFrame({"sample_name": ["a", "c"]})
    assert td.training_data_key(Loader) != k1


class _FakeLayer:
    def __init__(self, kind, data, name, kwargs):
        self.kind, self.data, self.name, self.kwargs = kind, data, name, kwargs
        self.contrast_limits_range = None


class _FakeViewer:
    def __init__(self):
        self.layers = []

    def add_image(self, data, name, **kwargs):
        self.layers.append(_FakeLayer("image", data, name, kwargs))
        return self.layers[-1]

    def add_labels(self, data, name, **kwargs):
        self.layers.append(_FakeLayer("labels", data, name, kwargs))
        return self.layers[-1]


def test_add_layer_specs_order_ranges_and_hooks():
    viewer = _FakeViewer()
    configured, styled = [], []
    specs = [
        td._layer("image", "Channel 0", [1], tag="channel",
                  contrast_limits_range=(0, 9), colormap="cyan"),
        td._layer("labels", "User", [2], tag="user_labels", opacity=0.5),
        td._layer("labels", "Dead pred", [3], tag="dead_pred"),
        td._layer("labels", "Plain", [4]),
    ]
    added = td.add_layer_specs(
        viewer, specs,
        configure_user_labels=configured.append, style_dead_pred=styled.append,
    )
    assert [l.name for l in viewer.layers] == ["Channel 0", "User", "Dead pred", "Plain"]
    assert added[0].contrast_limits_range == (0, 9)
    assert viewer.layers[0].kwargs == {"colormap": "cyan"}
    assert configured == [viewer.layers[1]]
    assert styled == [viewer.layers[2]]


# ---------------------------------------------------------------------------
# Multicolor cleanup scoping
# ---------------------------------------------------------------------------
def _multicolor_metadata():
    return pd.DataFrame({
        "sample_name": ["s1"],
        "im_tcells_1_multicolor_segments_image_path": ["a"],
        "im_tcells_2_multicolor_segments_image_path": ["b"],
        "im_nk_segments_image_path": ["c"],
    })


@pytest.fixture
def cleanup_calls(monkeypatch):
    calls = []
    import behav3d.preprocessing.segmentation.multicolor_segment_processing as msp

    def fake_apply(metadata, output_dir, base_cell_type, n_channels, **kwargs):
        calls.append((base_cell_type, n_channels))
        return metadata

    monkeypatch.setattr(msp, "apply_multicolor_segment_correction_for_base", fake_apply)
    return calls


def test_cleanup_runs_family_when_a_member_was_touched(cleanup_calls):
    from behav3d.napari._segmentation import run_multicolor_cleanup

    md = _multicolor_metadata()
    run_multicolor_cleanup(md, "/out", lambda m: None,
                           touched_cell_types=["tcells_1_multicolor"])
    assert cleanup_calls == [("tcells", 2)]


def test_cleanup_skips_family_when_run_touched_other_types(cleanup_calls):
    from behav3d.napari._segmentation import run_multicolor_cleanup

    md = _multicolor_metadata()
    run_multicolor_cleanup(md, "/out", lambda m: None, touched_cell_types=["nk"])
    run_multicolor_cleanup(md, "/out", lambda m: None, touched_cell_types=["dead"])
    assert cleanup_calls == []


def test_cleanup_unrestricted_and_unified(cleanup_calls):
    from behav3d.napari._segmentation import run_multicolor_cleanup

    md = _multicolor_metadata()
    run_multicolor_cleanup(md, "/out", lambda m: None)
    assert cleanup_calls == [("tcells", 2)]
    cleanup_calls.clear()
    assert run_multicolor_cleanup(md, "/out", lambda m: None, skip_unified=True) is md
    assert cleanup_calls == []


# ---------------------------------------------------------------------------
# Saving user labels (snapshot on the GUI thread, write on a worker)
# ---------------------------------------------------------------------------
def test_label_item_copies_so_later_painting_cannot_tear_the_write(tmp_path):
    import numpy as np

    live = np.zeros((2, 3, 3), dtype=np.int16)
    item = td.label_item("x labels", tmp_path / "a.zarr", live)
    live[:] = 2  # the user keeps painting after the snapshot
    assert item["data"].max() == 0


def test_write_label_snapshot_roundtrip_and_overwrite(tmp_path):
    import numpy as np
    from behav3d.io.images import load_zarr

    path = tmp_path / "PixelClassifier_UserTcellLabels.zarr"
    msgs, extras = [], []
    first = [td.label_item("tcell labels", path, np.full((2, 4, 4), 1, dtype=np.int16))]
    assert td.write_label_snapshot(first, msgs.append) == 1
    second = [
        td.label_item("tcell labels", path, np.full((2, 4, 4), 2, dtype=np.int16)),
        {"label": "label map", "fn": lambda: extras.append("map")},
    ]
    assert td.write_label_snapshot(second, msgs.append) == 1
    assert int(np.asarray(load_zarr(path)).max()) == 2  # replaced, not merged
    assert extras == ["map"]
    assert any("All user labels saved" in m for m in msgs)


def test_extra_item_failure_does_not_abort_the_save(tmp_path):
    import numpy as np

    def boom():
        raise RuntimeError("sidecar failed")

    items = [
        {"label": "label map", "fn": boom},
        td.label_item("x", tmp_path / "x.zarr", np.zeros((1, 2, 2), dtype=np.int16)),
    ]
    assert td.write_label_snapshot(items, lambda m: None) == 1


def test_start_label_save_writes_in_background_and_reports_done(tmp_path):
    import numpy as np

    page = _Page()
    done = []
    threads = []
    real_write = td.write_label_snapshot

    def spy(items, log, progress_cb=None):
        threads.append(threading.current_thread())
        return real_write(items, log, progress_cb)

    td.write_label_snapshot = spy
    try:
        items = [td.label_item("x", tmp_path / "x.zarr", np.zeros((1, 2, 2), dtype=np.int16))]
        started = td.start_label_save(page._bg, items, page.log, on_done=lambda: done.append(1))
        assert started is True
        assert _pump_until(lambda: done)
    finally:
        td.write_label_snapshot = real_write
    assert threads and threads[0] is not threading.current_thread()
    assert (tmp_path / "x.zarr").exists()


def test_start_label_save_refuses_while_the_page_is_busy(tmp_path):
    page = _Page()
    gate = threading.Event()
    page._bg.run(fn=lambda progress_cb=None: gate.wait(5), desc="busy", inject_progress=True)
    try:
        assert td.start_label_save(page._bg, [], page.log) is False
        assert any("Another job is running" in m for m in page.messages)
    finally:
        gate.set()
        _pump_until(lambda: not page._bg.is_running())

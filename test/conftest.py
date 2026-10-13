"""Shared pytest fixtures for the BEHAV3D test-suite.

The napari plugin has a process-wide "something is running" guard
(``behav3d.napari._background_runner``): a non-silent ``BackgroundOperation`` is
refused while *any* other operation is still running. Tests build a fresh tab per
test but share the process, and a test that ends the moment its mock has been
called can leave that tab's worker thread still winding down -- which would make
the *next* test's first run get refused. Waiting for the app to go idle between
tests keeps each test independent, exactly as a real user pausing between clicks.
"""
import os
import time

import pytest

# Never let the "please wait" QMessageBox block a headless test run.
os.environ.setdefault("BEHAV3D_NO_BUSY_DIALOG", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _wait_for_background_work_to_settle():
    def _settle(timeout_s=10.0):
        try:
            from qtpy.QtWidgets import QApplication

            from behav3d.napari import _background_runner as br
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        deadline = time.monotonic() + timeout_s
        while br.any_background_work(include_queue=False) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    _settle()
    yield
    _settle()

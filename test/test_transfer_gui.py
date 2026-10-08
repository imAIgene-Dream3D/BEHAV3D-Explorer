"""Transfer menu in the Data Preparation tab: wiring, background run, busy guard.

Runs under pytest or standalone: python test/test_transfer_gui.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("BEHAV3D_NO_BUSY_DIALOG", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy.QtWidgets import QApplication, QToolButton  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _wait(cond, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        _APP.processEvents()
        time.sleep(0.01)
    _APP.processEvents()
    return cond()


def test_transfer_menu_runs_pack_in_background():
    from behav3d.io.transfer import pack_project
    from behav3d.napari import _background_runner as br
    from behav3d.napari._data_preparation import DataPreparationTab
    from test_transfer import _make_project

    tab = DataPreparationTab()
    buttons = [b for b in tab.findChildren(QToolButton) if b.text().startswith("Transfer")]
    assert len(buttons) == 1
    actions = [a.text() for a in buttons[0].menu().actions()]
    assert actions == ["Pack project for transfer…", "Verify a copied bundle…", "Import packed project…"]

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "projA"
        _make_project(src)
        results = []
        ctl = tab._transfer
        ctl._start("Packing project…", pack_project,
                   {"output_dir": str(src), "destination": str(Path(tmp) / "usb")}, results.append)
        assert br.any_background_work(include_queue=False)
        assert _wait(lambda: results)
        assert results[0]["ok"], results[0]
        assert (results[0]["result"] / "bundle.json").exists()
        assert _wait(lambda: not br.any_background_work(include_queue=False))
        assert ctl._progress_dialog is None and ctl._worker is None
        assert "Bundle ready" in tab.log.toPlainText()

        # Errors come back as results, never as exceptions in the GUI thread.
        ctl._start("Packing project…", pack_project,
                   {"output_dir": str(Path(tmp) / "missing"), "destination": str(Path(tmp) / "usb")},
                   results.append)
        assert _wait(lambda: len(results) == 2)
        assert not results[1]["ok"] and results[1]["kind"] == "transfer"
    tab.deleteLater()
    _APP.processEvents()


def test_progress_label_is_a_percentage_not_a_size():
    from behav3d.napari._data_preparation import DataPreparationTab

    tab = DataPreparationTab()
    labels = []

    class _Dlg:
        def setValue(self, v):
            pass

        def setLabelText(self, text):
            labels.append(text)

    tab._transfer._progress_dialog = _Dlg()
    gb = 1024 ** 3
    tab._transfer._on_progress(100 * gb, 300 * gb, "Writing: a.zarr")  # 3 passes over 100 GB of data
    tab._transfer._progress_dialog = None
    assert labels and labels[0].startswith("33.3%") and "GB" not in labels[0]
    tab.deleteLater()
    _APP.processEvents()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    test_transfer_menu_runs_pack_in_background()
    print("PASSED test_transfer_menu_runs_pack_in_background")

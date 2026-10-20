"""
Report settings (State / State Trajectory reports) are saved to
behav3d_parameters.yml when a report runs and restored on reload.

Runs under pytest *or* standalone: python test/test_report_settings_persistence.py
Requires the `behav3d` conda env (qtpy). Uses the QApplication.instance() or
QApplication([]) pattern of the other widget tests (no pytest-qt needed).
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

_APP = None


def _app():
    global _APP
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_widget_value_round_trip():
    _app()
    from qtpy.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QListWidget
    from behav3d.napari._analysis import DualListGroupSelector
    from behav3d.napari._single_cell import _apply_report_widget_value, _report_widget_value

    chk = QCheckBox()
    assert _apply_report_widget_value(chk, True) and _report_widget_value(chk) is True

    spin = QDoubleSpinBox()
    spin.setRange(0.0, 10.0)
    assert _apply_report_widget_value(spin, 2.5) and _report_widget_value(spin) == 2.5

    combo = QComboBox()
    combo.addItems(["(none)"])
    assert not _apply_report_widget_value(combo, "condition")  # option not listed yet
    combo.addItems(["condition"])
    assert _apply_report_widget_value(combo, "condition")
    assert _report_widget_value(combo) == "condition"

    lst = QListWidget()
    lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
    assert not _apply_report_widget_value(lst, ["b"])  # list still empty
    lst.addItems(["a", "b", "c"])
    assert _apply_report_widget_value(lst, ["c", "a"])
    assert _report_widget_value(lst) == ["a", "c"]

    sel = DualListGroupSelector()
    saved = {"group_1": ["x"], "group_2": ["y", "z"], "excluded": ["w"]}
    assert not _apply_report_widget_value(sel, saved)  # no levels yet
    sel.set_items(["w", "x", "y", "z"])
    assert _apply_report_widget_value(sel, saved)
    assert _report_widget_value(sel) == saved


def _track_tab(params, out_dir, cell_type="tcells"):
    from behav3d.napari._single_cell import TrackClassificationSubTab
    loader = SimpleNamespace(output_dir=str(out_dir), behav3d_parameters=params, metadata=None)
    return TrackClassificationSubTab(
        viewer=None, metadata_loader=loader, cell_type_getter=lambda: cell_type
    )


def test_track_report_settings_saved_and_restored():
    _app()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        params = {}
        tab = _track_tab(params, out)
        tab.spin_transanalysis_gamma.setValue(3.5)
        tab.chk_transanalysis_include_window_sankey.setChecked(False)
        tab.combo_long_contact_unit.setCurrentIndex(0)  # percent
        tab.spin_long_contact_threshold.setValue(40.0)
        tab.spin_n_per_cluster.setValue(4)
        tab._persist_report_settings("tcells", "transition", "contact_duration", "exemplars")

        saved = yaml.safe_load((out / "behav3d_parameters.yml").read_text())
        reports = saved["track_reports"]["tcells"]
        assert reports["transition"]["emphasis_gamma"] == 3.5
        assert reports["transition"]["include_window_sankey"] is False
        assert reports["contact_duration"]["long_contact_unit"] == "Percent of time window"
        assert reports["exemplars"]["n_per_cluster"] == 4

        # Re-saving the classification settings must not wipe the report settings.
        tab._persist_track_cfg("tcells")
        assert "transition" in params["track_reports"]["tcells"]

        # A fresh GUI on the saved parameters gets the settings back on reload.
        tab2 = _track_tab(saved, out)
        tab2._queue_report_settings_restore("tcells")
        tab2._apply_pending_report_settings()
        assert tab2.spin_transanalysis_gamma.value() == 3.5
        assert not tab2.chk_transanalysis_include_window_sankey.isChecked()
        assert tab2.combo_long_contact_unit.currentData() == "percent"
        assert tab2.spin_long_contact_threshold.value() == 40.0
        assert tab2.spin_n_per_cluster.value() == 4

        # Another cell type keeps the defaults.
        tab3 = _track_tab(saved, out, cell_type="macrophages")
        tab3._queue_report_settings_restore("macrophages")
        tab3._apply_pending_report_settings()
        assert tab3.spin_transanalysis_gamma.value() == 2.0


if __name__ == "__main__":
    test_widget_value_round_trip()
    test_track_report_settings_saved_and_restored()
    print("ok")

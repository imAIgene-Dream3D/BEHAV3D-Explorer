"""Modal "Suggest btrack preset" dialog.

Measures how far cells move and how crowded they are from the segmentation
(no tracking needed), asks two questions, and recommends a preset plus
parameter adjustments using :func:`btrack_config.suggest_preset`. The
measurements are read-only; only the two answers are inputs.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
    QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from behav3d.core.utils import minutes_per_frame_from_metadata
from behav3d.napari._widgets import HelpButton
from behav3d.preprocessing.tracking.btrack_config import (
    AMBIGUITY_THRESHOLD,
    list_presets,
    suggest_preset,
)

_FIELD_LABELS = {
    "max_search_radius": "Max search radius (µm)",
    "dist_thresh": "Gap-bridging distance (µm)",
    "time_thresh": "Gap-bridging frames",
}

_HELP = {
    "motion": (
        "Motion scale (D)",
        "How far cells travel between two frames.\n\n"
        "For every cell it takes the distance to the closest cell in the "
        "next frame, then reports the value that 95% of cells stay below. "
        "Larger means faster cells or a longer frame interval."),
    "crowding": (
        "Crowding (N)",
        "How close neighbouring cells sit within one frame.\n\n"
        "It is the nearest-neighbour distance that only 5% of cells fall "
        "below. Small means a dense field where cells are easy to mix up."),
    "ambiguity": (
        "Ambiguity (D / N)",
        "Motion scale divided by crowding.\n\n"
        f"Below about {AMBIGUITY_THRESHOLD:g}, a cell is much closer to its own "
        "previous position than to any neighbour, so linking is easy. "
        f"Above {AMBIGUITY_THRESHOLD:g}, a cell could plausibly be matched to a "
        "neighbour, so tracking needs the more careful presets."),
    "segmentation": (
        "Segmentation quality",
        "Look at a few frames in the viewer. Cells that vanish for a frame "
        "or that are cut into two pieces make tracks break, so the helper "
        "allows longer gaps to be bridged."),
}

_SEGMENTATION_CHOICES = [("No", "no"), ("Some", "some"), ("Often", "often")]


def _value_label(text="…"):
    label = QLabel(text)
    label.setStyleSheet(
        "background: #2a2a2e; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
    return label


class BtrackPresetHelper(QDialog):
    """Dialog that turns measurements and two answers into a suggestion."""

    def __init__(self, motion, crowding, interval_min, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggest btrack preset")
        self.setModal(True)
        self._motion, self._crowding, self._interval = motion, crowding, interval_min
        self.suggestion = None
        self._labels = {p[0]: p[1] for p in list_presets()}
        self._build_ui()
        self._refresh()

    # -- layout ------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        note = QLabel("Measured from your segmentation (no tracking needed). "
                      "These values are read-only.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(note)

        grid = QGridLayout()
        self._d_label, self._n_label = _value_label(), _value_label()
        self._a_label, self._a_badge = _value_label(), QLabel()
        self._i_label = _value_label()
        rows = [
            ("Motion scale (D)", self._d_label, "motion"),
            ("Crowding (N)", self._n_label, "crowding"),
            ("Ambiguity (D / N)", self._a_label, "ambiguity"),
        ]
        for r, (text, widget, key) in enumerate(rows):
            grid.addWidget(QLabel(text), r, 0)
            grid.addWidget(widget, r, 1)
            grid.addWidget(HelpButton(*_HELP[key]), r, 2)
        grid.addWidget(self._a_badge, 2, 3)
        grid.addWidget(QLabel("Frame interval"), 3, 0)
        grid.addWidget(self._i_label, 3, 1)
        root.addLayout(grid)

        root.addWidget(QLabel("<b>Your answers</b>"))
        answers = QGridLayout()
        self._seg_combo = QComboBox()
        for text, value in _SEGMENTATION_CHOICES:
            self._seg_combo.addItem(text, value)
        self._div_combo = QComboBox()
        self._div_combo.addItem("No", False)
        self._div_combo.addItem("Yes", True)
        answers.addWidget(QLabel("Segmentation misses or splits cells?"), 0, 0)
        answers.addWidget(self._seg_combo, 0, 1)
        answers.addWidget(HelpButton(*_HELP["segmentation"]), 0, 2)
        answers.addWidget(QLabel("Do the cells divide?"), 1, 0)
        answers.addWidget(self._div_combo, 1, 1)
        root.addLayout(answers)
        self._seg_combo.currentIndexChanged.connect(self._refresh)
        self._div_combo.currentIndexChanged.connect(self._refresh)

        head = QHBoxLayout()
        head.addWidget(QLabel("Recommended:"))
        self._preset_label = QLabel()
        self._preset_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        head.addWidget(self._preset_label, stretch=1)
        root.addLayout(head)
        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #aaa;")
        root.addWidget(self._reason_label)

        self._table = QGridLayout()
        table_wrap = QWidget()
        table_wrap.setLayout(self._table)
        root.addWidget(table_wrap)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(
            "background: #4a3b16; color: #f0d58a; border-radius: 4px; padding: 6px;")
        root.addWidget(self._warning_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        apply_btn = buttons.addButton("Apply to fields", QDialogButtonBox.AcceptRole)
        apply_btn.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # -- logic -------------------------------------------------------------
    def _refresh(self, *_):
        s = suggest_preset(
            self._motion, self._crowding, self._interval,
            segmentation=self._seg_combo.currentData(),
            divides=bool(self._div_combo.currentData()),
        )
        self.suggestion = s
        ambiguity = s["ambiguity"]
        self._d_label.setText(f"{self._motion:.1f} µm")
        self._n_label.setText(f"{self._crowding:.1f} µm")
        self._a_label.setText(f"{ambiguity:.2f}")
        self._a_badge.setText("ambiguous" if ambiguity >= AMBIGUITY_THRESHOLD else "clear")
        self._i_label.setText(
            "unknown" if self._interval is None else f"{self._interval:g} min")
        name = self._labels.get(s["preset"], s["preset"])
        self._preset_label.setText(name + (" + Allow divisions" if s["allow_divisions"] else ""))
        self._reason_label.setText(s["reason"])
        self._fill_table(s["adjustments"])
        self._warning_label.setText("\n".join(s["warnings"]))
        self._warning_label.setVisible(bool(s["warnings"]))

    def _fill_table(self, adjustments):
        while self._table.count():
            item = self._table.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for c, title in enumerate(("Field", "Preset", "Applied", "Why")):
            header = QLabel(title)
            header.setStyleSheet("color: #888;")
            self._table.addWidget(header, 0, c)
        for r, (key, (base, applied, why)) in enumerate(adjustments.items(), start=1):
            changed = why is not None
            applied_label = QLabel(f"{applied:g}" if key != "time_thresh" else f"{int(applied)}")
            if changed:
                applied_label.setStyleSheet("font-weight: bold; color: #7fb4e8;")
            self._table.addWidget(QLabel(_FIELD_LABELS[key]), r, 0)
            self._table.addWidget(QLabel(f"{base:g}"), r, 1)
            self._table.addWidget(applied_label, r, 2)
            self._table.addWidget(QLabel(why or "unchanged"), r, 3)


def open_preset_helper(panel):
    """Measure the panel's segmentations, show the dialog, apply on accept."""
    from behav3d.napari._btrack_panel import apply_suggestion

    metadata = panel.metadata_loader.metadata
    if metadata is None or len(metadata) == 0:
        QMessageBox.information(panel, "Suggest btrack preset", "Load metadata first.")
        return
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        from behav3d.preprocessing.tracking.btrack_tracking import estimate_motion_crowding
        motion, crowding = estimate_motion_crowding(metadata, panel.cell_type)
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.warning(
            panel, "Suggest btrack preset", f"Could not measure the segmentation: {exc}")
        return
    QApplication.restoreOverrideCursor()
    if motion is None or crowding is None:
        QMessageBox.information(
            panel, "Suggest btrack preset",
            "Not enough segmented objects were found to measure motion and crowding.")
        return
    minutes, valid = minutes_per_frame_from_metadata(metadata)
    dialog = BtrackPresetHelper(motion, crowding, minutes if valid else None, parent=panel)
    if dialog.exec_() == QDialog.Accepted:
        apply_suggestion(panel, dialog.suggestion)

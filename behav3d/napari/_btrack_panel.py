"""btrack page of the Tracking tab: widgets, preset loading and persistence.

The page is built onto a ``CellTypeTrackingPanel`` (attributes ``bt_*``) so
the assistant and the batch/queue code keep addressing widgets by name. All
distances are in µm and variances in µm²; the physical/pixel switch only
changes how they are displayed (see :mod:`behav3d.napari._units`).

Field values come, in priority order, from the previous run of this
experiment (``behav3d_parameters.yml``), then from the selected preset JSON,
then from the hard defaults. Editing a field never touches the preset file.
"""
import os
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QFileDialog, QMessageBox,
)

from behav3d.core.utils import minutes_per_frame_from_metadata
from behav3d.napari._units import UnitGroupManager
from behav3d.napari._widgets import HelpButton, make_help_row
from behav3d.preprocessing.tracking.btrack_config import (
    DEFAULT_PRESET,
    LONG_INTERVAL_MIN,
    MAX_ACCURACY,
    PANEL_DEFAULTS,
    PRESET_MAP,
    list_presets,
    read_panel_values,
    resolve_config,
    write_custom_config,
)

# (value key, widget attribute, unit kind or None)
FIELD_SPECS = [
    ("max_search_radius", "bt_max_search_radius", "distance"),
    ("assignment_tolerance", "bt_assignment_tolerance", "distance"),
    ("new_track_position_variance", "bt_new_track_variance", "area"),
    ("lost_track_probability", "bt_lost_prob", None),
    ("use_optimize", "bt_use_optimize", None),
    ("dist_thresh", "bt_dist_thresh", "distance"),
    ("time_thresh", "bt_time_thresh", None),
    ("link_tolerance", "bt_link_tolerance", "distance"),
    ("optimizer_time_limit_s", "bt_optimizer_time_limit", None),
    ("measurement_noise", "bt_measurement_noise", "area"),
    ("track_start_end_penalty", "bt_start_end_penalty", "distance"),
    ("allow_divisions", "bt_allow_divisions", None),
]
_INT_KEYS = {"time_thresh", "optimizer_time_limit_s"}
_BOOL_KEYS = {"use_optimize", "allow_divisions"}
_STEP2_WIDGETS = ("bt_dist_thresh", "bt_time_thresh", "bt_link_tolerance",
                  "bt_optimizer_time_limit", "bt_allow_divisions")

_INTERVAL_NOTE = (
    "Frame interval above 10 min: tracking accuracy can drop whatever "
    "parameters you choose, because cells move too far between frames to be "
    "matched reliably."
)


# ---------------------------------------------------------------------------
# Value access
# ---------------------------------------------------------------------------
def _get(panel, key, attr, kind):
    widget = getattr(panel, attr)
    if kind is not None:
        return float(panel._bt_unit_mgr.get_native(widget))
    if key in _BOOL_KEYS:
        return bool(widget.isChecked())
    return int(widget.value()) if key in _INT_KEYS else float(widget.value())


def btrack_values(panel):
    """Current panel values in native units (µm, µm², frames, s)."""
    return {key: _get(panel, key, attr, kind) for key, attr, kind in FIELD_SPECS}


def set_btrack_values(panel, values):
    """Write ``values`` into the widgets without marking them as edited."""
    was_loading = panel._bt_loading
    panel._bt_loading = True
    try:
        for key, attr, kind in FIELD_SPECS:
            if key not in values or values[key] is None:
                continue
            widget = getattr(panel, attr)
            if kind is not None:
                panel._bt_unit_mgr.set_native(widget, values[key])
            elif key in _BOOL_KEYS:
                widget.setChecked(bool(values[key]))
            else:
                widget.setValue(int(values[key]) if key in _INT_KEYS
                                else float(values[key]))
        _update_step2_enabled(panel)
    finally:
        panel._bt_loading = was_loading


def _differs(values, base):
    for key in PANEL_DEFAULTS:
        a, b = values.get(key), base.get(key)
        if a is None or b is None:
            continue
        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                return True
        elif abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(b))):
            return True
    return False


# ---------------------------------------------------------------------------
# Preset handling
# ---------------------------------------------------------------------------
def _split_saved_preset(saved):
    """``(preset_key, custom_path)`` from a saved/collected btrack dict."""
    preset = str(saved.get("config_preset", DEFAULT_PRESET) or DEFAULT_PRESET).strip()
    if preset.lower() in PRESET_MAP:
        return preset.lower(), str(saved.get("config_path", "") or "")
    return None, preset or str(saved.get("config_path", "") or "")


def btrack_config_preset(panel):
    """Preset key, or the custom JSON path when *Custom JSON…* is selected."""
    key = panel.bt_config_preset.currentData()
    return key if key else panel.bt_config_path.text().strip()


def _current_config_path(panel):
    return resolve_config(btrack_config_preset(panel))


def _set_modified(panel, modified):
    panel._bt_modified = bool(modified)
    panel.bt_modified_label.setVisible(panel._bt_modified)


def _mark_modified(panel):
    if not panel._bt_loading:
        _set_modified(panel, True)


def _update_step2_enabled(panel):
    enabled = panel.bt_use_optimize.isChecked()
    for attr in _STEP2_WIDGETS:
        getattr(panel, attr).setEnabled(enabled)


def _update_path_enabled(panel):
    custom = panel.bt_config_preset.currentData() is None
    panel.bt_config_path.setEnabled(custom)
    panel.bt_browse_btn.setEnabled(custom)


def load_preset_values(panel, announce=True):
    """Fill every field from the selected preset/custom JSON."""
    try:
        path = _current_config_path(panel)
        values = read_panel_values(path)
    except Exception as exc:  # unreadable custom file, malformed JSON, ...
        panel.log(f"Could not read btrack configuration: {exc}")
        return False
    set_btrack_values(panel, values)
    _set_modified(panel, False)
    if announce:
        panel.log(f"Loaded btrack values from {Path(path).stem}.")
    return True


def _on_preset_changed(panel):
    if panel._bt_loading:
        return
    _update_path_enabled(panel)
    custom = panel.bt_config_preset.currentData() is None
    if custom and not panel.bt_config_path.text().strip():
        return
    load_preset_values(panel)


def _browse_config(panel):
    path, _ = QFileDialog.getOpenFileName(
        panel, "Select btrack config JSON", "",
        "JSON files (*.json);;All files (*)")
    if path:
        panel.bt_config_path.setText(path)
        load_preset_values(panel)


def _export_json(panel):
    try:
        src = _current_config_path(panel)
    except Exception as exc:
        QMessageBox.warning(panel, "Export btrack configuration", str(exc))
        return
    path, _ = QFileDialog.getSaveFileName(
        panel, "Export btrack configuration", "btrack_custom.json",
        "JSON files (*.json)")
    if not path:
        return
    try:
        write_custom_config(src, btrack_values(panel), path)
    except ValueError as exc:
        QMessageBox.warning(panel, "Export btrack configuration", str(exc))
        return
    panel.log(f"Exported btrack configuration to {path}.")
    answer = QMessageBox.question(
        panel, "Export btrack configuration",
        "Exported. Select the new file as the Custom JSON for this cell type?")
    if answer == QMessageBox.Yes:
        panel._bt_loading = True
        try:
            panel.bt_config_preset.setCurrentIndex(panel.bt_config_preset.count() - 1)
            panel.bt_config_path.setText(path)
        finally:
            panel._bt_loading = False
        _update_path_enabled(panel)
        _set_modified(panel, False)


def _open_helper(panel):
    from behav3d.napari._btrack_preset_helper import open_preset_helper
    open_preset_helper(panel)


def apply_suggestion(panel, suggestion):
    """Select the suggested preset, then apply its adjustments on top.

    The preset shows as modified when any value was adjusted.
    """
    panel._bt_loading = True
    try:
        idx = panel.bt_config_preset.findData(suggestion["preset"])
        panel.bt_config_preset.setCurrentIndex(idx)
        panel.bt_config_path.setText("")
    finally:
        panel._bt_loading = False
    _update_path_enabled(panel)
    load_preset_values(panel, announce=False)
    changes = {key: applied
               for key, (_base, applied, why) in suggestion["adjustments"].items()
               if why is not None}
    if suggestion.get("allow_divisions"):
        changes["allow_divisions"] = True
    if changes:
        set_btrack_values(panel, changes)
        _set_modified(panel, True)
    panel.log(f"Applied suggested btrack preset: {suggestion['preset']}.")


def apply_btrack_params(panel, params):
    """Copy a collected btrack dict into this panel (apply-to-others)."""
    key, custom = _split_saved_preset(params)
    panel._bt_loading = True
    try:
        idx = panel.bt_config_preset.findData(key) if key else -1
        if idx < 0:
            idx = panel.bt_config_preset.count() - 1
        panel.bt_config_preset.setCurrentIndex(idx)
        panel.bt_config_path.setText("" if key else custom)
        panel.bt_n_workers.setValue(int(params.get("n_workers", 1)))
    finally:
        panel._bt_loading = False
    _update_path_enabled(panel)
    set_btrack_values(panel, params)
    _set_modified(panel, params.get("preset_modified", False))


# ---------------------------------------------------------------------------
# Persistence / run parameters
# ---------------------------------------------------------------------------
def collect_btrack_params(panel):
    """Dict persisted to ``behav3d_parameters.yml`` (native units)."""
    params = btrack_values(panel)
    params.update({
        "config_preset": btrack_config_preset(panel),
        "config_path": panel.bt_config_path.text().strip(),
        "n_workers": max(1, int(panel.bt_n_workers.value())),
        "preset_modified": bool(panel._bt_modified),
    })
    return params


def btrack_run_kwargs(params):
    """``run_btracking`` keyword arguments from a collected btrack dict."""
    return {
        "config_preset": params["config_preset"],
        "max_search_radius": float(params["max_search_radius"]),
        "use_optimize": bool(params["use_optimize"]),
        "dist_thresh": float(params["dist_thresh"]),
        "time_thresh": int(params["time_thresh"]),
        "assignment_tolerance": float(params["assignment_tolerance"]),
        "new_track_position_variance": float(params["new_track_position_variance"]),
        "lost_track_probability": float(params["lost_track_probability"]),
        "measurement_noise": float(params["measurement_noise"]),
        "link_tolerance": float(params["link_tolerance"]),
        "track_start_end_penalty": float(params["track_start_end_penalty"]),
        "allow_divisions": bool(params["allow_divisions"]),
        "optimizer_time_limit_s": int(params["optimizer_time_limit_s"]),
        "n_workers": int(params["n_workers"]),
    }


def _start_state(bt_cfg, log):
    """Preset key, custom path, values and modified flag for a new panel."""
    key, custom = _split_saved_preset(bt_cfg)
    try:
        path = resolve_config(key or custom)
    except FileNotFoundError as exc:
        log(f"{exc} Falling back to the default preset.")
        key, custom, path = DEFAULT_PRESET, "", PRESET_MAP[DEFAULT_PRESET]
    base = read_panel_values(path)
    values = dict(base)
    for name in PANEL_DEFAULTS:  # previous run of this experiment wins
        if bt_cfg.get(name) is not None:
            values[name] = bt_cfg[name]
    return key, custom, values, _differs(values, base)


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------
def _dspin(lo, hi, decimals, step, width=110):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setMaximumWidth(width)
    return w


def _form():
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setSpacing(3)
    form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    return form


def _build_widgets(panel):
    """Create the field widgets (without values) as ``panel`` attributes."""
    panel.bt_max_search_radius = _dspin(0.1, 99999, 1, 5)
    panel.bt_assignment_tolerance = _dspin(0.1, 99999, 2, 0.1, 90)
    panel.bt_new_track_variance = _dspin(0.01, 99999, 2, 1)
    panel.bt_lost_prob = _dspin(0.0001, 0.9, 4, 0.01, 90)
    panel.bt_use_optimize = QCheckBox("Enable global track optimization")
    panel.bt_dist_thresh = _dspin(0.1, 99999, 1, 1)
    panel.bt_time_thresh = QSpinBox()
    panel.bt_time_thresh.setRange(1, 999)
    panel.bt_time_thresh.setMaximumWidth(70)
    panel.bt_link_tolerance = _dspin(0.1, 99999, 2, 5)
    panel.bt_optimizer_time_limit = QSpinBox()
    panel.bt_optimizer_time_limit.setRange(5, 3600)
    panel.bt_optimizer_time_limit.setSuffix(" s")
    panel.bt_optimizer_time_limit.setMaximumWidth(90)
    panel.bt_measurement_noise = _dspin(0.01, 99999, 2, 1)
    panel.bt_start_end_penalty = _dspin(0.01, 99999, 2, 0.1)
    panel.bt_allow_divisions = QCheckBox("Allow divisions (P_branch)")
    n_cores = os.cpu_count() or 4
    panel._bt_max_workers = (n_cores, max(1, n_cores - 1))
    panel.bt_n_workers = QSpinBox()
    panel.bt_n_workers.setRange(1, panel._bt_max_workers[1])
    panel.bt_n_workers.setMaximumWidth(60)


def _register_units(panel, values):
    for key, attr, kind in FIELD_SPECS:
        if kind is None:
            continue
        limit = MAX_ACCURACY if key == "assignment_tolerance" else None
        panel._bt_unit_mgr.register(
            getattr(panel, attr), kind, values[key],
            native_unit="physical", max_native=limit)


_PRESET_HELP = (
    "A preset is a complete btrack configuration. Choosing one fills every "
    "field below; editing a field never changes the preset file.\n\n"
    "• T cells presets: tuned settings, in µm, for motile cells.\n"
    "• btrack default (untuned): btrack's own example settings, kept for "
    "old projects.\n"
    "• Custom JSON: your own btrack configuration file; the fields show its "
    "values.\n\n"
    "Use Reset to preset to discard edits, or Export JSON to save the "
    "current values as a new file."
)


def _preset_block(panel):
    """Preset combo, state label, action buttons and custom-path row."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)

    panel.bt_config_preset = QComboBox()
    for key, label, desc, _path in list_presets():
        panel.bt_config_preset.addItem(label, key)
        panel.bt_config_preset.setItemData(
            panel.bt_config_preset.count() - 1, desc, Qt.ToolTipRole)
    panel.bt_config_preset.addItem("Custom JSON…", None)
    panel.bt_modified_label = QLabel("(modified)")
    panel.bt_modified_label.setStyleSheet("color: #d9a441; font-size: 10px;")
    row = QHBoxLayout()
    row.addWidget(QLabel("Preset:"))
    row.addWidget(panel.bt_config_preset, stretch=1)
    row.addWidget(panel.bt_modified_label)
    row.addWidget(HelpButton("Preset", _PRESET_HELP))
    lay.addLayout(row)

    panel._bt_button_row = QHBoxLayout()
    panel.bt_suggest_btn = QPushButton("Suggest…")
    panel.bt_suggest_btn.setToolTip(
        "Measure how far cells move and how crowded they are, and pick a "
        "preset for you.")
    panel._bt_button_row.addWidget(panel.bt_suggest_btn)
    panel.bt_reset_btn = QPushButton("Reset to preset")
    panel.bt_export_btn = QPushButton("Export JSON…")
    panel._bt_button_row.addWidget(panel.bt_reset_btn)
    panel._bt_button_row.addWidget(panel.bt_export_btn)
    panel._bt_button_row.addStretch()
    lay.addLayout(panel._bt_button_row)

    path_row = QHBoxLayout()
    panel.bt_config_path = QLineEdit()
    panel.bt_config_path.setPlaceholderText("Path to custom btrack JSON…")
    panel.bt_browse_btn = QPushButton("Browse")
    panel.bt_browse_btn.setFixedWidth(60)
    path_row.addWidget(panel.bt_config_path)
    path_row.addWidget(panel.bt_browse_btn)
    lay.addLayout(path_row)
    return box


def _add_step1_rows(panel, form):
    h = make_help_row
    form.addRow("Max search radius:", h(
        panel.bt_max_search_radius, "Max search radius",
        "Longest single-frame jump that may be linked. It only clips links; "
        "it does not choose between candidates.\n\n"
        "Too small: broken tracks and lost cells. Any value comfortably "
        "above the fastest cell's step behaves the same (about 3× the "
        "typical step is plenty)."))
    form.addRow("Assignment tolerance:", h(
        panel.bt_assignment_tolerance, "Assignment tolerance",
        "Half-width of the box over which the predicted position of a track "
        "is compared with each detection.\n\n"
        f"Must be between 0.1 and {MAX_ACCURACY:g} µm: btrack crashes above "
        f"{MAX_ACCURACY:g}. Below about 5 µm almost nothing is linked with "
        "the tuned presets.\n\n"
        "It works relative to the predicted spread (new-track uncertainty "
        "plus measurement noise)."))
    form.addRow("New-track uncertainty:", h(
        panel.bt_new_track_variance, "New-track position uncertainty",
        "How uncertain the position of a newly started track is (variance, "
        "µm²).\n\n"
        "Too large: new tracks never find their second detection. Too "
        "small: over-confident tracks cannot follow a moving cell."))
    form.addRow("Lost-track probability:", h(
        panel.bt_lost_prob, "Lost-track probability",
        "Prior probability that a track has no detection in the next frame.\n\n"
        "Higher: tracks break more readily (the optimizer can rejoin them). "
        "Lower: tracks hold on and may grab a wrong neighbour."))
    form.addRow("Workers:", h(
        panel.bt_n_workers, "Number of workers",
        "CPU cores for parallel object measurement and zarr writing (the "
        "tracker itself runs on one core).\n\n"
        f"Your machine has {panel._bt_max_workers[0]} cores; use at most "
        f"{panel._bt_max_workers[1]} to keep the system responsive."))


def _add_step2_rows(panel, form):
    h = make_help_row
    form.addRow("Gap-bridging distance:", h(
        panel.bt_dist_thresh, "Gap-bridging distance",
        "Track pieces whose ends are farther apart than this are never "
        "considered for joining.\n\n"
        "Larger: longer gaps are bridged but the optimizer is slower and "
        "wrong joins become more likely. About 4–6× the typical one-frame "
        "step is a good start."))
    form.addRow("Gap-bridging frames:", h(
        panel.bt_time_thresh, "Gap-bridging frames",
        "Longest disappearance (in frames) that can be bridged.\n\n"
        "Raise it (up to about 10) when the segmentation misses cells."))
    form.addRow("Link tolerance:", h(
        panel.bt_link_tolerance, "Link tolerance",
        "How quickly the probability of joining two track pieces falls with "
        "their distance.\n\n"
        "Lower: only close pieces are joined (purer, more fragments). "
        "Higher: more gaps bridged, more wrong joins.\n\n"
        "The optimizer compares joining against ending one track and "
        "starting another (Track start/end penalty); only their ratio "
        "matters."))


def _add_advanced_rows(panel, form):
    h = make_help_row
    form.addRow("Optimizer time limit:", h(
        panel.bt_optimizer_time_limit, "Optimizer time limit",
        "Stops the optimizer's solver after this long; the best solution "
        "found so far is used. Protects against very long runs on dense "
        "data."))
    form.addRow("Measurement noise:", h(
        panel.bt_measurement_noise, "Measurement noise",
        "How much the filter trusts each detected centre (variance, µm²).\n\n"
        "Too small: it follows segmentation jitter. Too large: sluggish "
        "prediction. Raise it for noisy centres (split or merged objects)."))
    form.addRow("Track start/end penalty:", h(
        panel.bt_start_end_penalty, "Track start/end penalty",
        "How easily a track may begin or end away from the image border.\n\n"
        "Higher: fewer wrong joins, more fragments. Lower: more complete "
        "tracks, more risk of wrong joins."))
    form.addRow("", h(
        panel.bt_allow_divisions, "Allow divisions",
        "Adds the P_branch hypothesis for dividing cells. The parent track "
        "ends at the division and each daughter starts a new TrackID.\n\n"
        "Leave off for cells that do not divide: it creates false "
        "divisions and can make the optimizer very slow on dense data."))


def _connect_signals(panel):
    panel.bt_config_preset.currentIndexChanged.connect(
        lambda _i: _on_preset_changed(panel))
    panel.bt_browse_btn.clicked.connect(lambda: _browse_config(panel))
    panel.bt_config_path.editingFinished.connect(
        lambda: None if panel._bt_loading else load_preset_values(panel))
    panel.bt_suggest_btn.clicked.connect(lambda: _open_helper(panel))
    panel.bt_reset_btn.clicked.connect(lambda: load_preset_values(panel))
    panel.bt_export_btn.clicked.connect(lambda: _export_json(panel))
    panel.bt_use_optimize.toggled.connect(lambda _c: _update_step2_enabled(panel))
    for _key, attr, _kind in FIELD_SPECS:
        widget = getattr(panel, attr)
        signal = widget.toggled if isinstance(widget, QCheckBox) else widget.valueChanged
        signal.connect(lambda *_a: _mark_modified(panel))


def build_btrack_page(panel, bt_cfg):
    """Build the btrack page for ``panel`` and return its root widget."""
    from behav3d.napari._analysis import CollapsibleSection

    panel._bt_loading = True
    panel._bt_modified = False
    key, custom, values, modified = _start_state(bt_cfg, panel.log)

    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(6, 4, 6, 4)
    lay.setSpacing(4)

    banner = QLabel(
        "<b>btrack</b> — Bayesian multi-object tracker.<br>"
        "<b>Step 1</b> (Kalman filter) links detections frame to frame.<br>"
        "<b>Step 2</b> (optimizer) joins track pieces and is recommended.<br>"
        "See the <b>btrack (Bayesian)</b> page of the documentation "
        "(Tracking) for the full parameter reference.")
    banner.setWordWrap(True)
    banner.setStyleSheet("color: #888; font-size: 10px; padding: 2px 0 6px 0;")
    lay.addWidget(banner)

    _build_widgets(panel)
    panel._bt_unit_mgr = UnitGroupManager(
        panel.metadata_loader.metadata, default_physical=True)
    unit_row = QHBoxLayout()
    unit_row.addWidget(QLabel("Distance units:"))
    unit_row.addWidget(panel._bt_unit_mgr.header_row(label=""))
    unit_wrap = QWidget()
    unit_wrap.setLayout(unit_row)
    lay.addWidget(unit_wrap)
    lay.addWidget(_preset_block(panel))

    panel.bt_interval_note = QLabel(_INTERVAL_NOTE)
    panel.bt_interval_note.setWordWrap(True)
    panel.bt_interval_note.setStyleSheet(
        "color: #d9a441; font-size: 10px; padding: 2px 0;")
    minutes, valid = minutes_per_frame_from_metadata(panel.metadata_loader.metadata)
    panel.bt_interval_note.setVisible(bool(valid and minutes > LONG_INTERVAL_MIN))
    lay.addWidget(panel.bt_interval_note)

    step1_group = QGroupBox("Step 1 — Frame-to-frame linking")
    step1_form = _form()
    step1_group.setLayout(step1_form)
    _add_step1_rows(panel, step1_form)

    step2_group = QGroupBox("Step 2 — Global optimizer")
    step2_lay = QVBoxLayout(step2_group)
    step2_lay.setContentsMargins(6, 4, 6, 4)
    step2_lay.setSpacing(3)
    panel.bt_use_optimize.setToolTip(
        "Step 2 joins broken tracks, bridges gaps and removes spurious "
        "fragments. Recommended.")
    step2_lay.addWidget(panel.bt_use_optimize)
    step2_form = _form()
    _add_step2_rows(panel, step2_form)
    step2_lay.addLayout(step2_form)

    advanced = CollapsibleSection("Advanced", expanded=False)
    advanced_form = _form()
    _add_advanced_rows(panel, advanced_form)
    advanced.contentLayout().addLayout(advanced_form)

    lay.addWidget(step1_group)
    lay.addWidget(step2_group)
    lay.addWidget(advanced)
    lay.addStretch()

    idx = panel.bt_config_preset.findData(key) if key else -1
    panel.bt_config_preset.setCurrentIndex(
        idx if idx >= 0 else panel.bt_config_preset.count() - 1)
    panel.bt_config_path.setText("" if key else custom)
    panel.bt_n_workers.setValue(
        min(int(bt_cfg.get("n_workers", 1)), panel._bt_max_workers[1]))
    _register_units(panel, values)
    set_btrack_values(panel, values)
    _update_path_enabled(panel)
    _set_modified(panel, modified)
    _connect_signals(panel)

    panel._bt_loading = False
    return page

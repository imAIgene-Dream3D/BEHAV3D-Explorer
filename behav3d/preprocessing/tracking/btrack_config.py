"""btrack configuration helpers: presets, panel <-> JSON mapping, validation.

GUI-free and independent of the ``btrack`` package so it can be unit tested
and reused by the napari panel, the assistant and the tracking backend.

All lengths are in micrometres and all variances in square micrometres,
because BEHAV3D links on physical coordinates.
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"

# Preset key -> JSON file. The tuned presets come first so GUIs list them first.
PRESET_MAP = {
    "tcell_tuned": MODELS_DIR / "tcell_tuned.json",
    "tcell_full_optimum": MODELS_DIR / "tcell_full_optimum.json",
    "tcell_dense": MODELS_DIR / "tcell_dense.json",
    "tcell_long_interval": MODELS_DIR / "tcell_long_interval.json",
    "tcell_noisy_segmentation": MODELS_DIR / "tcell_noisy_segmentation.json",
    "cell": MODELS_DIR / "cell_config.json",
    "particle": MODELS_DIR / "particle_config.json",
}
DEFAULT_PRESET = "tcell_tuned"

_LEGACY_LABELS = {
    "cell": ("btrack default - cell (untuned)",
             "Bundled btrack example settings for cells; kept for old projects."),
    "particle": ("btrack default - particle (untuned)",
                 "Bundled btrack example settings for particles; kept for old projects."),
}

# btrack's C++ core aborts the whole process outside these ranges.
MAX_ACCURACY = 10.0
MAX_MAX_LOST = 10

# GLPK's ``tm_lim`` is a C int in milliseconds; its maximum (~24.8 days) is
# GLPK's own "no limit" default.
GLPK_NO_TIME_LIMIT_MS = 2**31 - 1


def solver_time_limit_ms(time_limit_s, no_time_limit=False):
    """GLPK ``tm_lim`` (ms) for the optimizer; no limit when requested."""
    if no_time_limit:
        return GLPK_NO_TIME_LIMIT_MS
    return min(int(1000 * time_limit_s), GLPK_NO_TIME_LIMIT_MS)

# Panel key -> key inside the ``BEHAV3D.panel`` block of a preset file.
_PANEL_KEYS = {
    "max_search_radius": "max_search_radius_um",
    "assignment_tolerance": "assignment_tolerance_um",
    "new_track_position_variance": "new_track_position_variance_um2",
    "lost_track_probability": "lost_track_probability",
    "measurement_noise": "measurement_noise_um2",
    "use_optimize": "use_optimizer",
    "dist_thresh": "gap_bridging_distance_um",
    "time_thresh": "gap_bridging_frames",
    "link_tolerance": "link_tolerance_um",
    "track_start_end_penalty": "track_start_end_penalty_um",
    "allow_divisions": "allow_divisions",
    "optimizer_time_limit_s": "optimizer_time_limit_s",
    "optimizer_no_time_limit": "optimizer_no_time_limit",
}

# Used when neither the BEHAV3D block nor the TrackerConfig provides a value.
PANEL_DEFAULTS = {
    "max_search_radius": 125.0,
    "assignment_tolerance": 8.0,
    "new_track_position_variance": 40.0,
    "lost_track_probability": 0.1,
    "measurement_noise": 5.0,
    "use_optimize": True,
    "dist_thresh": 40.0,
    "time_thresh": 5,
    "link_tolerance": 450.0,
    "track_start_end_penalty": 0.5,
    "allow_divisions": False,
    "optimizer_time_limit_s": 60,
    "optimizer_no_time_limit": False,
}

# Hypotheses that must always be present; P_branch is the only user choice.
_BASE_HYPOTHESES = ["P_FP", "P_init", "P_term", "P_link"]


def resolve_config(config_preset):
    """Return the Path of a btrack JSON for a preset key or a file path."""
    if isinstance(config_preset, Path):
        return config_preset
    key = str(config_preset).strip()
    if key.lower() in PRESET_MAP:
        return PRESET_MAP[key.lower()]
    path = Path(key)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"btrack config not found: '{config_preset}'. "
        f"Use a preset name ({list(PRESET_MAP)}) or an existing JSON path."
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tracker_config(cfg):
    """Inner TrackerConfig dict (the wrapper is optional in btrack files)."""
    return cfg.get("TrackerConfig", cfg)


def list_presets():
    """Return ``[(key, label, description, path), ...]`` in display order."""
    out = []
    for key, path in PRESET_MAP.items():
        if key in _LEGACY_LABELS:
            label, desc = _LEGACY_LABELS[key]
        else:
            block = load_config(path).get("BEHAV3D", {})
            label = block.get("label", key)
            desc = " ".join(
                s for s in (block.get("description"), block.get("when_to_use")) if s
            )
        out.append((key, label, desc, path))
    return out


# ---------------------------------------------------------------------------
# JSON -> panel values
# ---------------------------------------------------------------------------
def _matrix_diag0(block):
    matrix = block.get("matrix") or [1.0]
    return float(matrix[0]) or 1.0


def _values_from_tracker_config(cfg):
    tc = _tracker_config(cfg)
    mm = tc.get("MotionModel", {})
    hm = tc.get("HypothesisModel", {})
    values = {}
    if "accuracy" in mm:
        values["assignment_tolerance"] = float(mm["accuracy"])
    if "prob_not_assign" in mm:
        values["lost_track_probability"] = float(mm["prob_not_assign"])
    if "P" in mm:
        values["new_track_position_variance"] = (
            float(mm["P"].get("sigma", 1.0)) * _matrix_diag0(mm["P"]))
    if "R" in mm:
        values["measurement_noise"] = (
            float(mm["R"].get("sigma", 1.0)) * _matrix_diag0(mm["R"]))
    if "dist_thresh" in hm:
        values["dist_thresh"] = float(hm["dist_thresh"])
    if "time_thresh" in hm:
        values["time_thresh"] = int(hm["time_thresh"])
    if "lambda_link" in hm:
        values["link_tolerance"] = float(hm["lambda_link"])
    if "lambda_dist" in hm:
        values["track_start_end_penalty"] = float(hm["lambda_dist"])
    if "hypotheses" in hm:
        values["allow_divisions"] = "P_branch" in hm["hypotheses"]
    return values


def read_panel_values(config_path):
    """Panel field values for a preset/custom JSON.

    Priority per field: ``BEHAV3D.panel`` block, then ``TrackerConfig``, then
    :data:`PANEL_DEFAULTS`.
    """
    cfg = load_config(config_path)
    values = dict(PANEL_DEFAULTS)
    values.update(_values_from_tracker_config(cfg))
    panel = cfg.get("BEHAV3D", {}).get("panel", {})
    for key, panel_key in _PANEL_KEYS.items():
        if panel_key in panel:
            values[key] = panel[panel_key]
    return values


# ---------------------------------------------------------------------------
# panel values -> JSON
# ---------------------------------------------------------------------------
def build_active_config(config_path, overrides=None, hypotheses=None):
    """Load a config and apply the non-``None`` overrides; return a dict.

    Only the fields in ``overrides`` change; every other JSON field passes
    through untouched. The ``BEHAV3D`` block is dropped (btrack ignores it).
    ``hypotheses`` is the legacy explicit list and wins over
    ``allow_divisions``.
    """
    cfg = copy.deepcopy(load_config(config_path))
    cfg.pop("BEHAV3D", None)
    _apply_overrides(cfg, overrides or {}, hypotheses)
    validate_config(cfg)
    return cfg


def _apply_overrides(cfg, overrides, hypotheses=None):
    ov = {k: v for k, v in overrides.items() if v is not None}
    tc = _tracker_config(cfg)
    mm = tc.setdefault("MotionModel", {})
    hm = tc.setdefault("HypothesisModel", {})
    if "assignment_tolerance" in ov:
        mm["accuracy"] = float(ov["assignment_tolerance"])
    if "lost_track_probability" in ov:
        mm["prob_not_assign"] = float(ov["lost_track_probability"])
    for key, block in (("new_track_position_variance", "P"),
                       ("measurement_noise", "R")):
        if key in ov:
            entry = mm.setdefault(block, {"sigma": 1.0, "matrix": [1.0]})
            entry["sigma"] = float(ov[key]) / _matrix_diag0(entry)
    if "dist_thresh" in ov:
        hm["dist_thresh"] = float(ov["dist_thresh"])
    if "time_thresh" in ov:
        hm["time_thresh"] = int(ov["time_thresh"])
    if "link_tolerance" in ov:
        hm["lambda_link"] = float(ov["link_tolerance"])
    if "track_start_end_penalty" in ov:
        hm["lambda_dist"] = float(ov["track_start_end_penalty"])
    if hypotheses is not None:
        hm["hypotheses"] = list(hypotheses)
    elif "allow_divisions" in ov:
        hyps = [h for h in hm.get("hypotheses", _BASE_HYPOTHESES)
                if h != "P_branch"]
        for base in _BASE_HYPOTHESES:
            if base not in hyps:
                hyps.append(base)
        if ov["allow_divisions"]:
            hyps.append("P_branch")
        hm["hypotheses"] = hyps


def validate_config(cfg):
    """Raise ``ValueError`` for values that make btrack abort or misbehave."""
    tc = _tracker_config(cfg)
    mm = tc.get("MotionModel", {})
    acc = mm.get("accuracy")
    if acc is not None and not 0 < float(acc) <= MAX_ACCURACY:
        raise ValueError(
            f"btrack accuracy (assignment tolerance) must be > 0 and <= "
            f"{MAX_ACCURACY:g}; got {acc}. Larger values crash btrack.")
    max_lost = mm.get("max_lost")
    if max_lost is not None and not 0 <= int(max_lost) <= MAX_MAX_LOST:
        raise ValueError(
            f"btrack max_lost must be between 0 and {MAX_MAX_LOST}; "
            f"got {max_lost}. Larger values crash btrack.")
    prob = mm.get("prob_not_assign")
    if prob is not None and not 0 < float(prob) < 1:
        raise ValueError(
            f"btrack prob_not_assign (lost-track probability) must be "
            f"between 0 and 1 (exclusive); got {prob}.")
    for name in ("P", "R"):
        entry = mm.get(name)
        if entry is not None and float(entry.get("sigma", 1.0)) <= 0:
            raise ValueError(
                f"btrack MotionModel.{name}.sigma must be positive; "
                f"got {entry.get('sigma')}.")


def write_temp_config(cfg):
    """Write ``cfg`` to a temporary JSON; the caller deletes the file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def write_custom_config(config_path, values, out_path):
    """Export a preset plus the given panel values as a new JSON file.

    The ``BEHAV3D`` block is kept and updated so the panel can reload the
    runtime fields (search radius, optimizer switch, time limit).
    """
    cfg = copy.deepcopy(load_config(config_path))
    _apply_overrides(cfg, values)
    validate_config(cfg)
    block = cfg.setdefault("BEHAV3D", {})
    label = str(block.get("label", Path(config_path).stem))
    if not label.endswith("(custom)"):
        label += " (custom)"
    block["label"] = label
    panel = block.setdefault("panel", {})
    for key, panel_key in _PANEL_KEYS.items():
        if values.get(key) is not None:
            panel[panel_key] = values[key]
    block.pop("validation_composite", None)
    block.pop("validation_note", None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Preset suggestion (no tracking needed)
# ---------------------------------------------------------------------------
AMBIGUITY_THRESHOLD = 0.9
CROWDED_NN_UM = 9.0
LONG_STEP_UM = 10.0
LONG_INTERVAL_MIN = 10.0


def motion_crowding_from_frames(frames, n_pairs=20):
    """Measure motion scale D and crowding N from per-frame centroids.

    ``frames`` is a list of ``(n_i, 3)`` arrays in physical units (one per
    timepoint). D is the 95th percentile of the distance from each object to
    the nearest object of the next frame; N is the 5th percentile of the
    within-frame nearest-neighbour distance. Returns ``(D, N)`` or
    ``(None, None)`` when there is too little data.
    """
    from scipy.spatial import cKDTree

    usable = [i for i in range(len(frames) - 1)
              if len(frames[i]) >= 2 and len(frames[i + 1]) >= 1]
    if not usable:
        return None, None
    picks = np.unique(np.linspace(0, len(usable) - 1,
                                  min(n_pairs, len(usable))).astype(int))
    step, nn = [], []
    for idx in (usable[p] for p in picks):
        a, b = np.asarray(frames[idx]), np.asarray(frames[idx + 1])
        step.append(cKDTree(b).query(a)[0])
        nn.append(cKDTree(a).query(a, k=2)[0][:, 1])
    return (float(np.percentile(np.concatenate(step), 95)),
            float(np.percentile(np.concatenate(nn), 5)))


def suggest_preset(D, N, interval_min, segmentation="no", divides=False):
    """Pick a preset and parameter adjustments from measurements and answers.

    Returns a dict with ``preset``, ``reason``, ``allow_divisions``,
    ``ambiguity``, ``adjustments`` (field -> ``(preset_value, applied,
    reason or None)``) and ``warnings``.
    """
    ambiguity = D / N if N else float("inf")
    if divides:
        key, reason = "tcell_tuned", "Cells divide: tuned preset with divisions allowed."
    elif segmentation == "often":
        key, reason = "tcell_dense", "Segmentation often misses or splits cells."
    elif ambiguity >= AMBIGUITY_THRESHOLD and N < CROWDED_NN_UM:
        key, reason = "tcell_dense", (
            f"Crowded field: 5% of neighbours are closer than {N:.1f} um.")
    elif ambiguity >= AMBIGUITY_THRESHOLD or D > LONG_STEP_UM:
        key, reason = "tcell_long_interval", (
            f"Cells move {D:.1f} um between frames, close to their "
            "neighbour spacing.")
    else:
        key, reason = "tcell_tuned", "Cells move little compared with their spacing."

    base = read_panel_values(PRESET_MAP[key])
    applied_radius = max(base["max_search_radius"], 3 * D)
    applied_gap = max(base["dist_thresh"], 4 * D)
    applied_frames = 10 if segmentation != "no" else base["time_thresh"]
    adjustments = {
        "max_search_radius": (
            base["max_search_radius"], applied_radius,
            "at least 3 x motion scale" if applied_radius != base["max_search_radius"] else None),
        "dist_thresh": (
            base["dist_thresh"], applied_gap,
            "at least 4 x motion scale" if applied_gap != base["dist_thresh"] else None),
        "time_thresh": (
            base["time_thresh"], applied_frames,
            "segmentation is not perfect" if applied_frames != base["time_thresh"] else None),
    }
    warnings = []
    if interval_min is not None and interval_min > LONG_INTERVAL_MIN:
        warnings.append(
            "Frame interval above 10 min: tracking accuracy can drop whatever "
            "parameters you choose, because cells move too far between frames "
            "to be matched reliably.")
    return {
        "preset": key, "reason": reason, "allow_divisions": bool(divides),
        "ambiguity": ambiguity, "adjustments": adjustments, "warnings": warnings,
    }

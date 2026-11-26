import copy
import inspect
import json

import numpy as np
import pytest

from behav3d.preprocessing.tracking import btrack_config as bc


ALL_PRESETS = list(bc.PRESET_MAP)
TUNED_PRESETS = [k for k in ALL_PRESETS if k.startswith("tcell_")]


def _hidden_fields(cfg):
    """Fields the panel never edits; they must survive any round trip."""
    tc = copy.deepcopy(cfg["TrackerConfig"])
    mm, hm = tc["MotionModel"], tc["HypothesisModel"]
    for key in ("accuracy", "prob_not_assign"):
        mm.pop(key, None)
    for key in ("P", "R"):
        mm[key].pop("sigma", None)
    for key in ("dist_thresh", "time_thresh", "lambda_link", "lambda_dist", "hypotheses"):
        hm.pop(key, None)
    return tc


@pytest.mark.parametrize("key", ALL_PRESETS)
def test_every_preset_file_exists_and_is_valid(key):
    cfg = bc.load_config(bc.resolve_config(key))
    bc.validate_config(cfg)


def test_list_presets_puts_tuned_first_and_labels_legacy_untuned():
    presets = bc.list_presets()
    keys = [p[0] for p in presets]
    assert keys[0] == bc.DEFAULT_PRESET
    assert keys.index("cell") > keys.index("tcell_noisy_segmentation")
    labels = {p[0]: p[1] for p in presets}
    assert "untuned" in labels["cell"] and "untuned" in labels["particle"]


@pytest.mark.parametrize("key", TUNED_PRESETS)
def test_round_trip_preserves_values_and_hidden_fields(key):
    path = bc.resolve_config(key)
    original = bc.load_config(path)
    values = bc.read_panel_values(path)

    rebuilt = bc.build_active_config(path, values)

    assert "BEHAV3D" not in rebuilt
    assert _hidden_fields(rebuilt) == _hidden_fields(original)
    tc = rebuilt["TrackerConfig"]
    assert tc["MotionModel"]["accuracy"] == pytest.approx(values["assignment_tolerance"])
    assert tc["HypothesisModel"]["dist_thresh"] == pytest.approx(values["dist_thresh"])
    assert tc["HypothesisModel"]["lambda_link"] == pytest.approx(values["link_tolerance"])
    p = tc["MotionModel"]["P"]
    assert p["sigma"] * p["matrix"][0] == pytest.approx(values["new_track_position_variance"])
    r = tc["MotionModel"]["R"]
    assert r["sigma"] * r["matrix"][0] == pytest.approx(values["measurement_noise"])


def test_tuned_preset_values_match_panel_block():
    values = bc.read_panel_values(bc.resolve_config("tcell_tuned"))
    assert values["assignment_tolerance"] == 8.0
    assert values["max_search_radius"] == 125.0
    assert values["use_optimize"] is True
    assert values["allow_divisions"] is False
    assert values["optimizer_time_limit_s"] == 60


def test_dense_preset_gives_optimizer_more_time():
    values = bc.read_panel_values(bc.resolve_config("tcell_dense"))
    assert values["optimizer_time_limit_s"] == 600


def test_legacy_preset_reads_values_from_tracker_config():
    values = bc.read_panel_values(bc.resolve_config("cell"))
    assert values["assignment_tolerance"] == 10
    assert values["lost_track_probability"] == 0.01
    assert values["dist_thresh"] == 60
    assert values["new_track_position_variance"] == pytest.approx(20.0)  # 200 * 0.1
    assert values["max_search_radius"] == bc.PANEL_DEFAULTS["max_search_radius"]


def test_overrides_apply_without_optimizer_and_keep_float_distance():
    cfg = bc.build_active_config(bc.resolve_config("cell"), {"dist_thresh": 33.5})
    assert cfg["TrackerConfig"]["HypothesisModel"]["dist_thresh"] == 33.5


def test_none_overrides_leave_json_untouched():
    path = bc.resolve_config("tcell_full_optimum")
    original = bc.load_config(path)
    cfg = bc.build_active_config(path, {k: None for k in bc.PANEL_DEFAULTS})
    assert cfg["TrackerConfig"] == original["TrackerConfig"]


def test_allow_divisions_toggles_p_branch_and_keeps_base_hypotheses():
    path = bc.resolve_config("tcell_tuned")
    on = bc.build_active_config(path, {"allow_divisions": True})
    off = bc.build_active_config(path, {"allow_divisions": False})
    hyps_on = on["TrackerConfig"]["HypothesisModel"]["hypotheses"]
    hyps_off = off["TrackerConfig"]["HypothesisModel"]["hypotheses"]
    assert "P_branch" in hyps_on and "P_branch" not in hyps_off
    assert {"P_FP", "P_init", "P_term", "P_link"} <= set(hyps_on) & set(hyps_off)


def test_legacy_hypotheses_list_is_honoured():
    cfg = bc.build_active_config(
        bc.resolve_config("tcell_tuned"), {}, hypotheses=["P_FP", "P_link"])
    assert cfg["TrackerConfig"]["HypothesisModel"]["hypotheses"] == ["P_FP", "P_link"]


@pytest.mark.parametrize("overrides", [
    {"assignment_tolerance": 10.5},
    {"assignment_tolerance": 0.0},
    {"lost_track_probability": 1.0},
    {"lost_track_probability": 0.0},
    {"new_track_position_variance": -1.0},
    {"measurement_noise": 0.0},
])
def test_validation_rejects_values_that_crash_or_break_btrack(overrides):
    with pytest.raises(ValueError):
        bc.build_active_config(bc.resolve_config("tcell_tuned"), overrides)


def test_validation_rejects_max_lost_above_ten():
    cfg = bc.load_config(bc.resolve_config("cell"))
    cfg["TrackerConfig"]["MotionModel"]["max_lost"] = 11
    with pytest.raises(ValueError, match="max_lost"):
        bc.validate_config(cfg)


def test_accuracy_ten_is_accepted():
    bc.build_active_config(bc.resolve_config("tcell_tuned"), {"assignment_tolerance": 10.0})


def test_missing_preset_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        bc.resolve_config("does_not_exist")


def test_write_custom_config_round_trip(tmp_path):
    src = bc.resolve_config("tcell_full_optimum")
    values = bc.read_panel_values(src)
    values["assignment_tolerance"] = 6.5
    values["use_optimize"] = False
    values["max_search_radius"] = 77.0

    out = bc.write_custom_config(src, values, tmp_path / "custom.json")

    again = bc.read_panel_values(out)
    assert again["assignment_tolerance"] == pytest.approx(6.5)
    assert again["use_optimize"] is False
    assert again["max_search_radius"] == 77.0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["BEHAV3D"]["label"].endswith("(custom)")
    assert _hidden_fields(saved) == _hidden_fields(bc.load_config(src))
    assert "validation_composite" not in saved["BEHAV3D"]


def test_write_custom_config_rejects_invalid_values(tmp_path):
    src = bc.resolve_config("tcell_tuned")
    values = bc.read_panel_values(src)
    values["assignment_tolerance"] = 12.0
    with pytest.raises(ValueError):
        bc.write_custom_config(src, values, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_custom_json_without_behav3d_block_falls_back_to_defaults(tmp_path):
    cfg = bc.load_config(bc.resolve_config("cell"))
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    values = bc.read_panel_values(path)
    assert values["use_optimize"] is True
    assert values["optimizer_time_limit_s"] == 60


# --- measurements and preset suggestion ------------------------------------
def _grid(spacing, n=6, shift=0.0):
    g = np.arange(n) * spacing
    return np.array([(x + shift, y, 0.0) for x in g for y in g])


def test_motion_crowding_known_grid():
    frames = [_grid(10.0, shift=0.0), _grid(10.0, shift=3.0), _grid(10.0, shift=6.0)]
    d, n = bc.motion_crowding_from_frames(frames)
    assert d == pytest.approx(3.0)
    assert n == pytest.approx(10.0)


def test_motion_crowding_without_data_returns_none():
    assert bc.motion_crowding_from_frames([np.zeros((1, 3))]) == (None, None)
    assert bc.motion_crowding_from_frames([]) == (None, None)


def test_suggest_clear_field_uses_tuned_preset():
    s = bc.suggest_preset(D=6.3, N=11.4, interval_min=2)
    assert s["preset"] == "tcell_tuned"
    assert s["warnings"] == []
    assert s["adjustments"]["time_thresh"][1] == 5


def test_suggest_crowded_field_uses_dense_preset():
    s = bc.suggest_preset(D=9.1, N=7.1, interval_min=2)
    assert s["preset"] == "tcell_dense"
    assert s["adjustments"]["max_search_radius"][1] == pytest.approx(252.51)


def test_suggest_motion_driven_ambiguity_uses_long_interval():
    s = bc.suggest_preset(D=14.4, N=11.4, interval_min=8)
    assert s["preset"] == "tcell_long_interval"
    assert s["adjustments"]["dist_thresh"][1] == pytest.approx(61.71)
    assert s["adjustments"]["dist_thresh"][2] is None


def test_suggest_poor_segmentation_prefers_dense_and_ten_gap_frames():
    s = bc.suggest_preset(D=6.3, N=11.4, interval_min=2, segmentation="often")
    assert s["preset"] == "tcell_dense"
    assert s["adjustments"]["time_thresh"][1] == 10
    some = bc.suggest_preset(D=6.3, N=11.4, interval_min=2, segmentation="some")
    assert some["preset"] == "tcell_tuned"
    assert some["adjustments"]["time_thresh"][1] == 10


def test_suggest_divisions_enable_p_branch():
    s = bc.suggest_preset(D=6.3, N=11.4, interval_min=2, divides=True)
    assert s["preset"] == "tcell_tuned" and s["allow_divisions"] is True


def test_suggest_scales_search_radius_and_gap_with_motion():
    s = bc.suggest_preset(D=50.0, N=60.0, interval_min=2)
    assert s["adjustments"]["max_search_radius"][1] == pytest.approx(150.0)
    assert s["adjustments"]["dist_thresh"][1] == pytest.approx(200.0)


def test_suggest_warns_only_for_long_frame_interval():
    assert bc.suggest_preset(6.3, 11.4, interval_min=10)["warnings"] == []
    warn = bc.suggest_preset(6.3, 11.4, interval_min=12)["warnings"]
    assert len(warn) == 1 and "10 min" in warn[0]


# --- backend API compatibility ----------------------------------------------
def test_backend_keeps_legacy_kwargs_and_adds_new_ones():
    pytest.importorskip("btrack")
    from behav3d.preprocessing.tracking import btrack_tracking as bt

    for fn in (bt.btrack_image, bt.run_btracking):
        params = inspect.signature(fn).parameters
        for legacy in ("update_method", "step_size", "use_visual_features",
                       "hypotheses", "dist_thresh", "time_thresh", "use_optimize"):
            assert legacy in params
        for new in ("assignment_tolerance", "new_track_position_variance",
                    "lost_track_probability", "measurement_noise",
                    "link_tolerance", "track_start_end_penalty",
                    "allow_divisions", "optimizer_time_limit_s"):
            assert new in params
        assert params["config_preset"].default == "tcell_tuned"

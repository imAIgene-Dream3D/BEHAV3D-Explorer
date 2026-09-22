"""Tests for behav3d.analysis.killing_validation.

The acceptance property of the Active Killing rework: at lower effector
density the number of death events and the credit per attributed death do not
change, whereas the previous algorithm's verdicts per death event scale with
how many effectors are present.
"""
import json

import numpy as np
import pandas as pd
import pytest

from behav3d.analysis.killing_validation import (
    _prepare,
    density_subsampling_invariance,
    run_killing_validation,
    select_radius,
    subsampling_slopes,
)
from behav3d.features.advanced_timepoint_features import resolve_active_killing_params
from behav3d.io.formats.zarr import save_as_zarr

T, Z, Y, X = 16, 20, 40, 40


@pytest.fixture
def crowded(tmp_path):
    """One organoid, two death patches (t=6 right side, t=11 left side); a true killer
    beside each patch and four bystanders touching the organoid the whole movie."""
    img = tmp_path / "images" / "s1"
    img.mkdir(parents=True)
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    labels = np.zeros((T, Z, Y, X), np.uint16)
    labels[:, ((zz - 10) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) <= 81] = 5
    dead = np.zeros_like(labels)
    dead[6:, 8:12, 18:22, 23:27] = 1
    dead[11:, 8:12, 18:22, 13:17] = 1
    immune = np.zeros_like(labels)
    rows = []

    def eff(tid, box, frames):
        for t in frames:
            immune[(t,) + box] = tid
        for t in range(T):
            rows.append({"sample_name": "s1", "TrackID": tid, "position_t": t,
                         "touching_organoids": "5" if t in frames else ""})

    eff(1, (slice(8, 11), slice(18, 21), slice(30, 33)), range(2, 6))
    eff(2, (slice(8, 11), slice(18, 21), slice(7, 10)), range(7, 11))
    for tid, box in zip((3, 4, 5, 6), (
            (slice(8, 11), slice(30, 33), slice(18, 21)), (slice(8, 11), slice(7, 10), slice(18, 21)),
            (slice(1, 4), slice(18, 21), slice(18, 21)), (slice(17, 20), slice(18, 21), slice(18, 21)))):
        eff(tid, box, range(T))
    for name, arr in (("s1_organoid_tracked", labels), ("s1_tcell_tracked", immune), ("s1_mask_dead", dead)):
        save_as_zarr(arr, img / f"{name}.zarr")
    md = pd.DataFrame([{
        "sample_name": "s1", "pixel_distance_xy": 1.0, "pixel_distance_z": 1.0, "distance_unit": "um",
        "time_interval": 2.0, "time_unit": "m", "dead_mask_path": str(img / "s1_mask_dead.zarr"),
        "or_organoid_tracks_image_path": str(img / "s1_organoid_tracked.zarr"),
        "im_tcell_tracks_image_path": str(img / "s1_tcell_tracked.zarr"),
    }])
    raw = tmp_path / "analysis" / "tcell" / "track_features" / "BEHAV3D_tcell_combined_track_features.csv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(raw, index=False)
    return tmp_path, md


KW = dict(target_cell_diameter_um=6.0, causal_window_min=10.0, attribution_radius_um=15.0)


def test_subsampling_keeps_death_events_and_credit_per_death(crowded):
    tmp, md = crowded
    dp, ap = resolve_active_killing_params(**KW)
    prep = _prepare(md, tmp, "tcell", ["organoid"], dp, ap, log_fn=lambda *a: None)
    dens = density_subsampling_invariance(prep, ap, immune_type="tcell", fractions=(0.34, 0.67, 1.0),
                                          n_replicates=6, seed=0)
    assert dens["n_death_events"].nunique() == 1 and dens["n_death_events"].iloc[0] == 2
    credit = dens["kill_credit_per_attributed_death"].dropna()
    assert np.allclose(credit, 1.0)
    # The previous algorithm's structure: verdicts per death grow with effector density.
    legacy = dens.groupby("fraction")["legacy_style_verdicts_per_death_event"].mean()
    assert legacy.is_monotonic_increasing and legacy.iloc[-1] > legacy.iloc[0]
    slopes = subsampling_slopes(dens).set_index("metric")["slope_log_log"]
    assert abs(slopes["n_death_events"]) < 1e-9
    assert slopes["legacy_style_verdicts_per_death_event"] > 0.5
    # Full-density coverage matches the analytic expectation exactly.
    full = dens[dens["fraction"] == 1.0].iloc[0]
    assert full["n_attributed"] == pytest.approx(full["expected_n_attributed"])


def test_select_radius_takes_the_largest_radius_under_alpha():
    fdr = pd.DataFrame({"radius_um": [2, 4, 6, 8], "observed_events": [0, 30, 40, 45],
                        "fdr": [1.0, 0.02, 0.04, 0.2]})
    assert select_radius(fdr, 0.05) == 6
    assert select_radius(fdr, 0.01) is None
    assert select_radius(pd.DataFrame(), 0.05) is None


def test_validation_reports_fdr_floor_and_refuses_when_unreachable(crowded):
    tmp, md = crowded
    res = run_killing_validation(md, tmp, "tcell", ["organoid"], n_replicates=2, n_permutations=5,
                                 log_fn=lambda *a: None, **KW)
    # Two informative events: the (E0+1)/(O+1) estimator cannot go below 1/3.
    assert res["n_informative_events"] == 2
    assert res["fdr_floor"] == pytest.approx(1 / 3)
    assert res["calibrated_radius_um"] is None and res["attribution_supported"] is False
    vdir = tmp / "analysis" / "tcell" / "active_killing" / "organoid" / "validation"
    for name in ("density_subsampling.csv", "density_subsampling_slopes.csv", "fdr_radius_sweep.csv",
                 "density_subsampling.png", "fdr_radius_sweep.png", "validation_summary.json"):
        assert (vdir / name).exists(), name
    summary = json.loads((vdir / "validation_summary.json").read_text())
    assert summary["fdr_floor"] == pytest.approx(1 / 3)

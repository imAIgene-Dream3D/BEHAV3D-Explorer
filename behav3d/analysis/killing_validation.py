"""
Validation and calibration for the Active Killing attribution.

Two questions the analysis must answer about itself rather than assume:

1. **Is it density-independent?** Retain random subsets of whole effector
   tracks and re-attribute. The death-event count cannot change (death events
   are detected once, with the dead-mask cleaning fixed at the full effector
   set -- otherwise the test would be circular), and the per-effector rates
   should stay flat, while a legacy-style "every contacting effector gets the
   verdict" count scales with the number of effectors kept. There is an exact
   prediction for attributed coverage: ``E[#attributed] = sum_e 1-(1-f)^k_e``.

2. **Is attribution better than chance at this density?** The spatial
   permutation null rotates each death patch about its organoid's centroid
   (rejecting rotations that leave the organoid), which preserves patch size,
   depth, organoid shape and effector density exactly and destroys only the
   patch<->effector relation. Sweeping the attribution radius gives an
   empirical FDR, ``(E0(r)+1) / (O(r)+1)``; the calibrated radius is the largest
   radius with FDR <= alpha. If no radius achieves it, attribution is refused.

Everything here re-uses the cached death events and a single distance
measurement per (event, effector, contact frame); only the null geometry
re-reads effector frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _prepare(metadata, output_dir, immune_cell_type, target_cell_types, dparams, aparams, log_fn=print):
    """Load death events (cached), contacts and the full pair table with distances."""
    from behav3d.features.death_events import detect_death_events, resolve_tracks_image_path
    from behav3d.features.kill_attribution import (
        attribute_death_events, explode_contacts, identify_contact_events_per_target)

    output_dir = Path(output_dir)
    raw = output_dir / "analysis" / immune_cell_type / "track_features" / f"BEHAV3D_{immune_cell_type}_combined_track_features.csv"
    df_immune = pd.read_csv(raw)
    for col in df_immune.columns:
        if col.startswith("touching_"):
            df_immune[col] = df_immune[col].fillna("").astype(str)
    md = metadata[metadata["sample_name"].astype(str).isin(set(df_immune["sample_name"].astype(str)))]
    immune_paths = {}
    for _, row in md.iterrows():
        p = resolve_tracks_image_path(row, immune_cell_type)
        if p is not None and p.exists():
            immune_paths[str(row["sample_name"])] = str(p)
    events, ts, patches, sample_info = detect_death_events(md, output_dir, target_cell_types, dparams,
                                                            reuse_cache=True, log_fn=log_fn)
    contacts = explode_contacts(df_immune, target_cell_types)
    _, contacts = identify_contact_events_per_target(contacts)
    cand, events_annot, pairs = attribute_death_events(
        events, contacts, patches, ts, immune_type=immune_cell_type, immune_tracks_paths=immune_paths,
        sample_info=sample_info, params=aparams, return_pairs=True)
    return dict(df_immune=df_immune, md=md, immune_paths=immune_paths, events=events, events_annot=events_annot,
                ts=ts, patches=patches, sample_info=sample_info, contacts=contacts, pairs=pairs, cand=cand)


def _legacy_style_verdicts(contacts: pd.DataFrame, events: pd.DataFrame, sample_info, window_min: float) -> int:
    """Count (effector, contact timepoint) rows whose target has a death onset in [t, t+W].

    Mimics the previous algorithm's structure -- every contacting effector gets
    the verdict for every contact timepoint whose forward window contains a
    death -- so its inflation with effector density can be shown on the same
    axes as the new metrics.
    """
    if contacts.empty or events.empty:
        return 0
    n = 0
    for (s, o, tid), g in contacts.groupby(["sample_name", "organoid_type", "target_track_id"]):
        on = events[(events["sample_name"] == s) & (events["organoid_type"] == o)
                    & (events["target_track_id"] == tid)]["t_onset"].to_numpy()
        if on.size == 0:
            continue
        w = int(math.ceil(window_min / sample_info[s]["minutes_per_frame"]))
        t = g["position_t"].to_numpy()
        n += int(np.sum([(np.any((on >= ti) & (on <= ti + w))) for ti in t]))
    return n


def density_subsampling_invariance(
    prepared: dict,
    aparams,
    *,
    immune_type: str,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    n_replicates: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Re-attribute on random subsets of whole effector tracks.

    Returns one row per (fraction, replicate) with the event count, attributed
    count, the analytic expectation, per-effector means of kills_attributed,
    kills_exclusive and hit-delivery rate, and the legacy-style verdict count
    per effector.
    """
    from behav3d.features.kill_attribution import credit_from_pairs, summarize_per_effector

    rng = np.random.default_rng(seed)
    pairs, events = prepared["pairs"], prepared["events_annot"]
    df_immune, contacts, sinfo = prepared["df_immune"], prepared["contacts"], prepared["sample_info"]
    tracks = df_immune[["sample_name", "TrackID"]].drop_duplicates()
    n_full = len(tracks)
    # k_e: candidates per attributed event at full density, for the analytic curve.
    k = prepared["cand"].groupby(["sample_name", "organoid_type", "death_event_id"]).size() \
        if not prepared["cand"].empty else pd.Series(dtype=int)
    rows = []
    for f in fractions:
        for rep in range(1 if f >= 1.0 else int(n_replicates)):
            keep = tracks.sample(n=max(1, int(round(f * n_full))), random_state=int(rng.integers(1 << 31)))
            keys = set(zip(keep["sample_name"].astype(str), keep["TrackID"].astype(int)))
            sub_pairs = pairs[[(str(s), int(i)) in keys for s, i in zip(pairs["sample_name"], pairs["immune_track_id"])]] \
                if not pairs.empty else pairs
            base_events = events.drop(columns=[c for c in ("attribution_class", "n_candidates", "n_contacting_in_window")
                                               if c in events.columns])
            cand, ev = credit_from_pairs(sub_pairs, base_events, params=aparams, sample_info=sinfo,
                                         immune_type=immune_type)
            sub_imm = df_immune[[(str(s), int(i)) in keys for s, i in zip(df_immune["sample_name"], df_immune["TrackID"])]]
            sub_con = contacts[[(str(s), int(i)) in keys for s, i in zip(contacts["sample_name"], contacts["immune_track_id"])]]
            from behav3d.features.kill_attribution import identify_contact_events_per_target
            ce, _ = identify_contact_events_per_target(sub_con.drop(columns=["contact_event_id"], errors="ignore"))
            eff = summarize_per_effector(cand, sub_imm, ce, sinfo)
            engaged = eff[eff["n_contact_events"] > 0]
            counted = ev[ev["attribution_class"] != "excluded_border"]
            rows.append({
                "fraction": float(f), "replicate": rep, "n_effectors": int(len(eff)),
                "n_death_events": int(len(counted)),
                "n_attributed": int(counted["attribution_class"].isin(["exclusive", "shared"]).sum()),
                "expected_n_attributed": float(np.sum(1.0 - (1.0 - f) ** k.to_numpy())) if len(k) else 0.0,
                "total_kill_credit": float(cand["kill_credit"].sum()) if not cand.empty else 0.0,
                "mean_kills_attributed": float(engaged["kills_attributed"].mean()) if len(engaged) else np.nan,
                "mean_kills_exclusive": float(engaged["kills_exclusive"].mean()) if len(engaged) else np.nan,
                "mean_hit_delivery_rate": float(engaged["hit_delivery_rate"].mean()) if len(engaged) else np.nan,
                # The old algorithm's inflation lives in the total per death:
                # every contacting effector received the verdict, so verdicts per
                # death event grow with how many effectors are present. The new
                # credit per attributed death is exactly 1 by construction.
                "legacy_style_verdicts_per_death_event": (
                    _legacy_style_verdicts(sub_con, prepared["events"], sinfo, aparams.causal_window_min)
                    / max(int(len(counted)), 1)),
                "kill_credit_per_attributed_death": (
                    (float(cand["kill_credit"].sum()) / n_att) if (n_att := int(
                        counted["attribution_class"].isin(["exclusive", "shared"]).sum())) else np.nan),
            })
    return pd.DataFrame(rows)


def subsampling_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Slope of log(metric) vs log(fraction) for each metric (0 = density-independent)."""
    out = []
    for col in ("n_death_events", "kill_credit_per_attributed_death", "mean_kills_exclusive",
                "mean_hit_delivery_rate", "mean_kills_attributed", "legacy_style_verdicts_per_death_event"):
        d = df[["fraction", col]].dropna()
        d = d[d[col] > 0]
        if d["fraction"].nunique() < 2:
            out.append({"metric": col, "slope_log_log": np.nan, "n": len(d)})
            continue
        slope = float(np.polyfit(np.log(d["fraction"]), np.log(d[col]), 1)[0])
        out.append({"metric": col, "slope_log_log": slope, "n": len(d)})
    return pd.DataFrame(out)


def _rotation_matrices(n, rng):
    """n uniformly random 3-D rotation matrices (QR of Gaussian matrices)."""
    mats = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        mats.append(q)
    return mats


def spatial_permutation_null(
    prepared: dict,
    aparams,
    *,
    n_permutations: int = 50,
    min_inside_fraction: float = 0.8,
    max_tries: int = 20,
    seed: int = 0,
    radii_um: Sequence[float] = tuple(range(1, 41)),
) -> Dict[str, pd.DataFrame]:
    """Observed vs rotated-patch minimum candidate distances, and the FDR sweep.

    Only events with at least one effector in contact within the causal window
    are informative (the null keeps those contacts and moves only the patch).
    """
    from behav3d.features.death_events import resolve_tracks_image_path
    from behav3d.features.kill_attribution import measure_event_effector_distances
    from behav3d.io.images import open_image_timepoints

    rng = np.random.default_rng(seed)
    pairs, events = prepared["pairs"], prepared["events_annot"]
    if pairs.empty:
        return {"observed": pd.DataFrame(), "null": pd.DataFrame(), "fdr": pd.DataFrame()}
    key = ["sample_name", "organoid_type", "death_event_id"]
    # Observed and null distances must be measured out to the same range: the
    # attribution run only measures up to its own radius, which would cap the
    # observed side of the sweep and bias the FDR against larger radii.
    reach = float(max(radii_um)) + 1.0
    base = pairs.drop(columns=["distance_um"], errors="ignore")
    d_obs = measure_event_effector_distances(base, events, prepared["patches"], prepared["ts"],
                                             prepared["immune_paths"], prepared["sample_info"], reach)
    observed = base.assign(distance_um=d_obs.to_numpy()).groupby(key)["distance_um"].min()         .rename("d_min_um").reset_index()
    ev = events.set_index(key)
    label_handles = {}
    null_rows = []
    patches = prepared["patches"]
    for perm in range(int(n_permutations)):
        rotated = {}
        for (s, o, eid) in observed[key].itertuples(index=False):
            coords = patches.get((s, o), {}).get(int(eid))
            if coords is None or len(coords) == 0:
                continue
            e = ev.loc[(s, o, eid)]
            if (s, o) not in label_handles:
                row = prepared["md"][prepared["md"]["sample_name"].astype(str) == str(s)].iloc[0]
                label_handles[(s, o)] = open_image_timepoints(resolve_tracks_image_path(row, o))
            lab = np.asarray(label_handles[(s, o)][int(e["t_onset"])]) == int(e["target_track_id"])
            centre = np.argwhere(lab).mean(axis=0)
            spacing = np.asarray(prepared["sample_info"][s]["voxel_spacing"])
            placed = None
            for R in _rotation_matrices(max_tries, rng):
                c = np.round(((coords - centre) * spacing) @ R.T / spacing + centre).astype(np.int64)
                ok = np.all((c >= 0) & (c < np.asarray(lab.shape)), axis=1)
                if ok.mean() < 1.0:
                    continue
                if lab[c[:, 0], c[:, 1], c[:, 2]].mean() >= min_inside_fraction:
                    placed = c
                    break
            if placed is not None:
                rotated.setdefault((s, o), {})[int(eid)] = placed
        placed_keys = {(a, b, int(c)) for (a, b), d in rotated.items() for c in d}
        sub = base[[(s, o, int(i)) in placed_keys
                    for s, o, i in zip(base["sample_name"], base["organoid_type"], base["death_event_id"])]]
        if sub.empty:
            continue
        d = measure_event_effector_distances(sub, events, rotated, prepared["ts"],
                                             prepared["immune_paths"], prepared["sample_info"], reach)
        mins = sub.assign(distance_um=d.to_numpy()).groupby(key)["distance_um"].min().reset_index()
        mins["permutation"] = perm
        null_rows.append(mins)
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame(columns=key + ["distance_um", "permutation"])
    fdr = []
    n_perm_done = max(1, null["permutation"].nunique()) if not null.empty else 1
    for r in radii_um:
        O = int((observed["d_min_um"] <= r).sum())
        E0 = float((null["distance_um"] <= r).sum()) / n_perm_done if not null.empty else float("nan")
        fdr.append({"radius_um": float(r), "observed_events": O, "expected_by_chance": E0,
                    "fdr": (E0 + 1.0) / (O + 1.0) if np.isfinite(E0) else float("nan")})
    return {"observed": observed, "null": null, "fdr": pd.DataFrame(fdr)}


def select_radius(fdr: pd.DataFrame, alpha: float = 0.05) -> Optional[float]:
    """Largest radius whose empirical FDR is <= alpha, or None if there is none."""
    if fdr is None or fdr.empty:
        return None
    ok = fdr[(fdr["fdr"] <= alpha) & (fdr["observed_events"] > 0)]
    return float(ok["radius_um"].max()) if not ok.empty else None


def run_killing_validation(
    metadata: pd.DataFrame,
    output_dir,
    immune_cell_type: str,
    target_cell_types: Sequence[str],
    *,
    target_cell_diameter_um: float = 10.0,
    causal_window_min: float = 120.0,
    attribution_radius_um: float = 15.0,
    advanced: Optional[dict] = None,
    results_subfolder: Optional[str] = None,
    n_replicates: int = 20,
    n_permutations: int = 50,
    alpha: float = 0.05,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Density-subsampling invariance + FDR calibration, written to ``<results>/validation/``.

    Returns ``{"calibrated_radius_um", "fdr_at_radius", "slopes", "paths"}``.
    ``calibrated_radius_um`` is None when no radius separates attribution from
    chance at ``alpha`` -- the caller must then not treat attribution as valid.
    """
    from behav3d.features.advanced_timepoint_features import resolve_active_killing_params

    dparams, aparams = resolve_active_killing_params(target_cell_diameter_um, causal_window_min,
                                                     attribution_radius_um, advanced)
    targets = list(target_cell_types)
    sub = results_subfolder or ("combined" if len(targets) > 1 else targets[0])
    vdir = Path(output_dir) / "analysis" / immune_cell_type / "active_killing" / sub / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    log_fn("Validation: preparing death events and effector distances...")
    prep = _prepare(metadata, output_dir, immune_cell_type, targets, dparams, aparams, log_fn=log_fn)

    log_fn(f"Validation: density subsampling ({n_replicates} replicates per fraction)...")
    dens = density_subsampling_invariance(prep, aparams, immune_type=immune_cell_type,
                                          n_replicates=n_replicates, seed=seed)
    slopes = subsampling_slopes(dens)
    dens.to_csv(vdir / "density_subsampling.csv", index=False)
    slopes.to_csv(vdir / "density_subsampling_slopes.csv", index=False)

    log_fn(f"Validation: spatial permutation null ({n_permutations} rotations per event)...")
    null = spatial_permutation_null(prep, aparams, n_permutations=n_permutations, seed=seed)
    radius = select_radius(null["fdr"], alpha)
    null["fdr"].to_csv(vdir / "fdr_radius_sweep.csv", index=False)
    null["observed"].to_csv(vdir / "observed_min_distances.csv", index=False)
    null["null"].to_csv(vdir / "null_min_distances.csv", index=False)
    fdr_at = None
    if radius is not None:
        fdr_at = float(null["fdr"].loc[null["fdr"]["radius_um"] == radius, "fdr"].iloc[0])

    # Figures ---------------------------------------------------------------
    fig = Figure(figsize=(11, 4.2))
    ax1, ax2 = fig.subplots(1, 2)
    g = dens.groupby("fraction")
    for col, label, color in (("kill_credit_per_attributed_death", "kill credit / attributed death (new)", "#7E57C2"),
                              ("mean_hit_delivery_rate", "hit delivery rate / effector (new)", "#26A69A"),
                              ("mean_kills_exclusive", "exclusive kills / effector (new)", "#5C6BC0"),
                              ("legacy_style_verdicts_per_death_event",
                               "verdicts / death event (previous algorithm)", "#E53935")):
        m = g[col].mean()
        if m.notna().any() and (m.iloc[-1] if len(m) else 0):
            ax1.plot(m.index, m / m.iloc[-1], "o-", color=color, label=label)
    ax1.axhline(1.0, color="#bbbbbb", ls="--")
    ax1.set_xlabel("Fraction of effector tracks kept")
    ax1.set_ylabel("Metric relative to full data")
    ax1.set_title("Density invariance (flat = density-independent)", fontsize=10)
    ax1.legend(fontsize=7, frameon=False)
    ax2.plot(g["n_attributed"].mean().index, g["n_attributed"].mean(), "o-", color="#7E57C2", label="observed")
    ax2.plot(g["expected_n_attributed"].mean().index, g["expected_n_attributed"].mean(), "--", color="#555555",
             label="analytic expectation")
    ax2.plot(g["n_death_events"].mean().index, g["n_death_events"].mean(), ":", color="#999999",
             label="death events (must not change)")
    ax2.set_xlabel("Fraction of effector tracks kept")
    ax2.set_ylabel("Events")
    ax2.set_title("Attributed coverage vs prediction", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)
    FigureCanvasAgg(fig)
    fig.savefig(vdir / "density_subsampling.png", dpi=130, bbox_inches="tight")

    fig = Figure(figsize=(7, 4.2))
    ax = fig.subplots()
    f = null["fdr"]
    if not f.empty:
        ax.plot(f["radius_um"], f["fdr"], "o-", color="#7E57C2", ms=3)
        ax.axhline(alpha, color="#E53935", ls="--", label=f"alpha = {alpha:g}")
        if radius is not None:
            ax.axvline(radius, color="#26A69A", ls=":", label=f"calibrated radius {radius:g} µm")
        ax.set_xlabel("Attribution radius (µm)")
        ax.set_ylabel("Empirical FDR (rotated-patch null)")
        ax.legend(fontsize=8, frameon=False)
    ax.set_title("Attribution radius calibration", fontsize=10)
    FigureCanvasAgg(fig)
    fig.savefig(vdir / "fdr_radius_sweep.png", dpi=130, bbox_inches="tight")

    n_informative = int(len(null["observed"])) if not null["observed"].empty else 0
    best = (null["fdr"].dropna(subset=["fdr"]).sort_values(["fdr", "radius_um"]).head(1)
            if not null["fdr"].empty else pd.DataFrame())
    summary = {
        "n_informative_events": n_informative,
        # (E0 + 1) / (O + 1) can never go below 1 / (O + 1): with few death
        # events, a strict alpha is unreachable however clean the data are.
        "fdr_floor": (1.0 / (n_informative + 1)) if n_informative else None,
        "best_fdr": float(best["fdr"].iloc[0]) if not best.empty else None,
        "radius_at_best_fdr_um": float(best["radius_um"].iloc[0]) if not best.empty else None,
        "calibrated_radius_um": radius,
        "fdr_at_radius": fdr_at,
        "alpha": alpha,
        "n_permutations": n_permutations,
        "n_replicates": n_replicates,
        "requested_radius_um": float(attribution_radius_um),
        "attribution_supported": radius is not None,
        "slopes": slopes.to_dict(orient="records"),
    }
    (vdir / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if radius is None:
        floor = summary["fdr_floor"]
        msg = (f"Validation: NO attribution radius reaches an empirical FDR <= {alpha:g}. Best FDR "
               f"{summary['best_fdr'] if summary['best_fdr'] is not None else float('nan'):.3f} at "
               f"{summary['radius_at_best_fdr_um']} µm, from {n_informative} informative death events.")
        if floor is not None and floor > alpha:
            msg += (f" Note: with {n_informative} events the lowest reachable FDR is {floor:.3f}, so "
                    f"alpha={alpha:g} is unreachable here regardless of data quality; pool more samples "
                    f"or judge the curve rather than the cut-off.")
        elif summary["best_fdr"] is not None and summary["best_fdr"] < 0.25:
            msg += (f" Nothing was applied automatically. The lowest-FDR radius, "
                    f"{summary['radius_at_best_fdr_um']:g} µm, separates attribution from chance "
                    f"reasonably well; set it manually if an FDR of {summary['best_fdr']:.2f} is acceptable "
                    f"for your question, and inspect the curve in fdr_radius_sweep.png.")
        else:
            msg += " Attribution is not distinguishable from chance at this effector density."
        log_fn(msg)
    else:
        log_fn(f"Validation: calibrated attribution radius {radius:g} µm (empirical FDR {fdr_at:.3f}).")
    return {**summary, "paths": {"dir": str(vdir)}}

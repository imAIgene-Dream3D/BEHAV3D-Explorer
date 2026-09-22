"""
Figures for the Active Killing analysis.

Written by :func:`behav3d.features.advanced_timepoint_features.run_active_killing_analysis`
into ``<results_dir>/plots/`` so the napari panel, the notebook and the
processing queue all produce the same outputs. Every figure writes its
backing table next to it (``<name>.csv``), following the repo's plot-data
convention, so any figure can be re-plotted without re-running the analysis.

Uses the object-oriented ``matplotlib.figure.Figure`` API (never pyplot):
the napari run executes on a background thread.

Figures
-------
Attributed vs unattributed death (always):
    death_event_fate_over_time, attributed_fraction_by_condition,
    attribution_funnel, patch_depth_attributed_vs_unattributed
Killing concentration:
    killing_concentration
Serial killing and timing:
    serial_killing_and_timing
Target side:
    organoid_swimmer_top<N>  (only the targets of the top-N killers)
    engagement_dose_response (all targets)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ATTRIBUTED = ("exclusive", "shared")
_C_ATT = "#7E57C2"
_C_UN = "#90A4AE"
_C_TRUNC = "#CFD8DC"
# Contact -> apoptosis lag reported for melanoma targets (Weigelin et al. 2021).
_LIT_LAG_MIN = (108.0, 90.0)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(fig: Figure, tables: Dict[str, pd.DataFrame], plot_dir: Path, name: str, log_fn) -> Path:
    FigureCanvasAgg(fig)
    out = plot_dir / f"{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    for tname, table in tables.items():
        tpath = plot_dir / f"{tname}.csv"
        table.to_csv(tpath, index=False)
        log_fn(f"   Plot data saved: {tpath.name}")
    return out


def _empty(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#999999")
    ax.set_xticks([])
    ax.set_yticks([])


def _km(durations: np.ndarray, events: np.ndarray):
    """Kaplan-Meier survival of 'no kill yet'; returns (times, survival)."""
    order = np.argsort(durations)
    d, e = durations[order], events[order].astype(bool)
    times, surv, s, n = [0.0], [1.0], 1.0, len(d)
    for t in np.unique(d):
        at_risk = int(np.sum(d >= t))
        deaths = int(np.sum((d == t) & e))
        if at_risk > 0 and deaths > 0:
            s *= 1.0 - deaths / at_risk
            times.append(float(t))
            surv.append(s)
    return np.asarray(times), np.asarray(surv)


def write_active_killing_figures(
    results_dir,
    immune_type: str,
    *,
    top_n: int = 5,
    condition_map: Optional[Dict[str, str]] = None,
    log_fn=print,
) -> Dict[str, Path]:
    """Render every Active Killing figure from the CSVs in ``results_dir``."""
    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    im = immune_type
    ev = _read(results_dir / f"death_events_{im}.csv")
    cand = _read(results_dir / f"kill_candidates_{im}.csv")
    eff = _read(results_dir / f"per_effector_killing_{im}.csv")
    tgt = _read(results_dir / f"per_target_death_{im}.csv")
    ce = _read(results_dir / f"contact_events_{im}.csv")
    cond = {str(k): str(v) for k, v in (condition_map or {}).items()}
    out: Dict[str, Path] = {}
    if not ev.empty:
        ev = ev[ev["attribution_class"] != "excluded_border"].copy()
        ev["group"] = ev["sample_name"].astype(str).map(lambda s: cond.get(s, s))

    # 1. Fate over time -------------------------------------------------------
    fig = Figure(figsize=(10, 6.5))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    table = pd.DataFrame()
    if ev.empty:
        _empty(ax1, "No death events")
        _empty(ax2, "")
    else:
        t_min = ev["time_onset_min"].astype(float)
        edges = np.histogram_bin_edges(t_min, bins=min(30, max(5, int(np.sqrt(len(ev))) * 2)))
        ev["_bin"] = pd.cut(t_min, edges, include_lowest=True, labels=False)
        att = ev[ev["attribution_class"].isin(ATTRIBUTED)].groupby("_bin").size()
        un = ev[ev["attribution_class"] == "unattributed"].groupby("_bin").size()
        tr = ev[ev["attribution_class"] == "unattributed_truncated"].groupby("_bin").size()
        idx = np.arange(len(edges) - 1)
        a, u, r = (att.reindex(idx, fill_value=0).to_numpy(), un.reindex(idx, fill_value=0).to_numpy(),
                   tr.reindex(idx, fill_value=0).to_numpy())
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = float(np.diff(edges).mean()) * 0.9
        ax1.bar(centers, a, width, color=_C_ATT, label=f"Attributed to {im}")
        ax1.bar(centers, u, width, bottom=a, color=_C_UN, label="Unattributed (background)")
        ax1.bar(centers, r, width, bottom=a + u, color=_C_TRUNC,
                label="Unattributed, window before movie start")
        ax1.set_ylabel("Death events")
        ax1.legend(fontsize=8, frameon=False)
        tot = a + u
        frac = np.where(tot > 0, a / np.maximum(tot, 1), np.nan)
        ax2.plot(centers, frac, "o-", color=_C_ATT)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("Attributed fraction")
        ax2.set_xlabel("Death-event onset (min)")
        table = pd.DataFrame({"bin_start_min": edges[:-1], "bin_end_min": edges[1:], "n_attributed": a,
                              "n_unattributed": u, "n_unattributed_truncated": r, "attributed_fraction": frac})
    ax1.set_title("Death events over time: attributed vs unattributed")
    out["death_event_fate_over_time"] = _save(fig, {"death_event_fate_over_time": table}, plot_dir,
                                              "death_event_fate_over_time", log_fn)

    # 2. Attributed fraction by condition --------------------------------------
    fig = Figure(figsize=(max(6, 1.2 * max(1, ev["group"].nunique() if not ev.empty else 1) + 3), 4.5))
    ax = fig.subplots()
    table = pd.DataFrame()
    if ev.empty:
        _empty(ax, "No death events")
    else:
        counted = ev[ev["attribution_class"] != "unattributed_truncated"]
        g = counted.assign(_att=counted["attribution_class"].isin(ATTRIBUTED),
                           _vatt=np.where(counted["attribution_class"].isin(ATTRIBUTED),
                                          counted["onset_volume_um3"], 0.0))
        table = g.groupby("group").agg(n_death_events=("death_event_id", "size"), n_attributed=("_att", "sum"),
                                       volume_attributed_um3=("_vatt", "sum"),
                                       volume_um3=("onset_volume_um3", "sum")).reset_index()
        table["attributed_fraction_events"] = table["n_attributed"] / table["n_death_events"]
        table["attributed_fraction_volume"] = np.where(table["volume_um3"] > 0,
                                                       table["volume_attributed_um3"] / table["volume_um3"], np.nan)
        x = np.arange(len(table))
        ax.bar(x - 0.2, table["attributed_fraction_events"], 0.4, color=_C_ATT, label="by events")
        ax.bar(x + 0.2, table["attributed_fraction_volume"], 0.4, color="#B39DDB", label="by dead volume")
        for xi, (n, na) in enumerate(zip(table["n_death_events"], table["n_attributed"])):
            ax.text(xi, 1.02, f"{int(na)}/{int(n)}", ha="center", fontsize=8, color="#555555")
        ax.set_xticks(x)
        ax.set_xticklabels(table["group"], rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"Death attributed to {im}")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Immune-attributed fraction of death by condition\n"
                 "(a no-effector control arm is the empirical zero)", fontsize=10)
    out["attributed_fraction_by_condition"] = _save(fig, {"attributed_fraction_by_condition": table},
                                                    plot_dir, "attributed_fraction_by_condition", log_fn)

    # 3. Funnel ------------------------------------------------------------------
    fig = Figure(figsize=(7.5, 4))
    ax = fig.subplots()
    table = pd.DataFrame()
    if ev.empty:
        _empty(ax, "No death events")
    else:
        steps = [
            ("All death events", len(ev)),
            ("Effector in contact within causal window",
             int((ev.get("n_contacting_in_window", pd.Series(0, index=ev.index)) > 0).sum())),
            ("...and within attribution radius", int(ev["attribution_class"].isin(ATTRIBUTED).sum())),
        ]
        table = pd.DataFrame(steps, columns=["step", "n_events"])
        y = np.arange(len(steps))[::-1]
        ax.barh(y, table["n_events"], color=[_C_UN, "#B39DDB", _C_ATT])
        for yi, (lab, n) in zip(y, steps):
            ax.text(n, yi, f"  {n}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(table["step"], fontsize=9)
        ax.set_xlabel("Death events")
    ax.set_title("Attribution funnel -- where unattributed death drops out", fontsize=10)
    out["attribution_funnel"] = _save(fig, {"attribution_funnel": table}, plot_dir, "attribution_funnel", log_fn)

    # 4. Patch depth ------------------------------------------------------------------
    fig = Figure(figsize=(9, 4))
    ax1, ax2 = fig.subplots(1, 2)
    table = pd.DataFrame()
    if ev.empty or "patch_depth_um" not in ev:
        _empty(ax1, "No death events")
        _empty(ax2, "")
    else:
        d = ev[["death_event_id", "sample_name", "attribution_class", "patch_depth_um"]].dropna()
        d = d[d["attribution_class"].isin(ATTRIBUTED + ("unattributed",))]
        d["attributed"] = d["attribution_class"].isin(ATTRIBUTED)
        table = d
        data, labels, colors = [], [], []
        for flag, lab, col in ((True, "Attributed", _C_ATT), (False, "Unattributed", _C_UN)):
            v = np.sort(d.loc[d["attributed"] == flag, "patch_depth_um"].to_numpy())
            if v.size:
                ax1.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col, label=f"{lab} (n={v.size})")
                data.append(v)
                labels.append(lab)
                colors.append(col)
        if data:
            parts = ax2.violinplot(data, showmedians=True)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col)
                body.set_alpha(0.6)
            ax2.set_xticks(np.arange(1, len(labels) + 1))
            ax2.set_xticklabels(labels)
        ax1.set_xlabel("Patch depth below the organoid surface (µm)")
        ax1.set_ylabel("Cumulative fraction")
        ax1.legend(fontsize=8, frameon=False)
        ax2.set_ylabel("Patch depth (µm)")
    fig.suptitle("Immune killing should be superficial; core necrosis is deep", fontsize=10)
    out["patch_depth_attributed_vs_unattributed"] = _save(
        fig, {"patch_depth_attributed_vs_unattributed": table}, plot_dir,
        "patch_depth_attributed_vs_unattributed", log_fn)

    # 5. Killing concentration --------------------------------------------------------
    fig = Figure(figsize=(10, 4.2))
    ax1, ax2 = fig.subplots(1, 2)
    tables = {}
    engaged = eff[eff["n_contact_events"] > 0] if not eff.empty else eff
    if engaged.empty or engaged["kills_attributed"].sum() <= 0:
        _empty(ax1, "No attributed kills")
        _empty(ax2, "")
    else:
        v = np.sort(engaged["kills_attributed"].to_numpy(dtype=float))
        cum = np.concatenate([[0.0], np.cumsum(v) / v.sum()])
        pop = np.linspace(0, 1, v.size + 1)
        n = v.size
        gini = float((2.0 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum())) - (n + 1.0) / n)
        k = max(1, int(np.ceil(0.1 * n)))
        top10 = float(np.sort(v)[::-1][:k].sum() / v.sum())
        ax1.plot(pop, cum, color=_C_ATT)
        ax1.plot([0, 1], [0, 1], "--", color="#bbbbbb")
        ax1.set_xlabel("Cumulative fraction of engaged effectors (fewest kills first)")
        ax1.set_ylabel("Cumulative fraction of kill credit")
        ax1.set_title(f"Lorenz curve -- Gini {gini:.2f}, top 10% hold {top10:.0%}", fontsize=10)
        hours = engaged["contact_minutes"].astype(float) / 60.0
        ax2.scatter(hours, engaged["kills_attributed"], s=14, color=_C_ATT, alpha=0.6)
        ax2.set_xlabel("Contact time (h)")
        ax2.set_ylabel("Kill credit")
        ax2.set_title("Kills vs contact time (slope = per-contact-hour efficiency)", fontsize=10)
        tables["killing_concentration_lorenz"] = pd.DataFrame({"effector_fraction": pop, "credit_fraction": cum,
                                                                "gini": gini, "top10pct_share": top10})
        tables["killing_concentration_scatter"] = engaged[["sample_name", "immune_track_id", "contact_minutes",
                                                           "kills_attributed", "kill_per_contact_hour"]]
    out["killing_concentration"] = _save(fig, tables, plot_dir, "killing_concentration", log_fn)

    # 6. Serial killing & timing ------------------------------------------------------
    fig = Figure(figsize=(11, 8))
    (a1, a2), (a3, a4) = fig.subplots(2, 2)
    tables = {}
    if eff.empty:
        for a in (a1, a2, a3, a4):
            _empty(a, "No effector data")
    else:
        iki = eff["inter_kill_interval_min"].dropna()
        if iki.empty:
            _empty(a1, "No effector killed twice")
        else:
            a1.hist(iki, bins=20, color=_C_ATT)
            a1.axvline(50, ls="--", color="#555555")
            a1.text(50, a1.get_ylim()[1] * 0.9, " 50 min", fontsize=8)
        a1.set_xlabel("Median inter-kill interval (min)")
        a1.set_title("Serial killing interval", fontsize=10)
        ndt = eff.loc[eff["n_contact_events"] > 0, "n_distinct_targets_killed"]
        if ndt.empty:
            _empty(a2, "No engaged effectors")
        else:
            vc = ndt.value_counts().sort_index()
            a2.bar(vc.index.astype(int), vc.to_numpy(), color=_C_ATT)
        a2.set_xlabel("Distinct targets killed per engaged effector")
        a2.set_title("Serial killing breadth", fontsize=10)
        eng = eff[eff["n_contact_events"] > 0].copy()
        if eng.empty:
            _empty(a3, "No engaged effectors")
        else:
            eng["group"] = eng["sample_name"].astype(str).map(lambda s: cond.get(s, s))
            killed = eng["time_to_first_kill_min"].notna()
            mpf = np.where(eng["n_timepoints"] > 0, eng["track_minutes"] / eng["n_timepoints"], np.nan)
            followed = (eng["last_t"] - eng["first_contact_t"]) * mpf
            eng["km_duration_min"] = np.where(killed, eng["time_to_first_kill_min"], followed)
            eng["km_event"] = killed
            km_rows = []
            for grp, g in eng.groupby("group"):
                t, s = _km(g["km_duration_min"].to_numpy(dtype=float), g["km_event"].to_numpy())
                a3.step(t, s, where="post", label=f"{grp} (n={len(g)})")
                km_rows.append(pd.DataFrame({"group": grp, "time_min": t, "survival_no_kill": s}))
            a3.set_ylim(-0.02, 1.02)
            a3.legend(fontsize=7, frameon=False)
            tables["serial_killing_time_to_first_kill_km"] = pd.concat(km_rows, ignore_index=True)
        a3.set_xlabel("Time from first contact (min)")
        a3.set_ylabel("Fraction without a kill yet")
        a3.set_title("Time to first kill (Kaplan-Meier, censored at track end)", fontsize=10)
        if cand.empty:
            _empty(a4, "No attributed kills")
        else:
            a4.hist(cand["lag_min"], bins=20, color=_C_ATT)
            mu, sd = _LIT_LAG_MIN
            a4.axvspan(max(0, mu - sd), mu + sd, color="#FFE082", alpha=0.5, label="melanoma 1.8 ± 1.5 h")
            a4.legend(fontsize=7, frameon=False)
        a4.set_xlabel("Lag: last contact -> death onset (min)")
        a4.set_title("Contact-to-death lag", fontsize=10)
        tables["serial_killing_per_effector"] = eff[[c for c in (
            "sample_name", "immune_track_id", "kills_attributed", "kills_exclusive", "kills_nearest",
            "n_distinct_targets_killed", "inter_kill_interval_min", "time_to_first_kill_min",
            "is_serial_killer") if c in eff.columns]]
        if not cand.empty:
            tables["serial_killing_lag"] = cand[["death_event_id", "sample_name", "immune_track_id", "lag_min"]]
    out["serial_killing_and_timing"] = _save(fig, tables, plot_dir, "serial_killing_and_timing", log_fn)

    # 7. Swimmer plot for the top-N killers' targets ------------------------------------
    from behav3d.features.kill_attribution import rank_top_killers
    top = rank_top_killers(eff, top_n) if not eff.empty else pd.DataFrame()
    name = f"organoid_swimmer_top{int(top_n)}"
    tables = {}
    rows = pd.DataFrame()
    if not top.empty and not cand.empty:
        keys = set(zip(top["sample_name"].astype(str), top["immune_track_id"].astype(int)))
        mine = cand[[(str(s), int(i)) in keys for s, i in zip(cand["sample_name"], cand["immune_track_id"])]]
        rows = mine[["sample_name", "organoid_type", "target_track_id"]].drop_duplicates()
    fig = Figure(figsize=(10, max(3.0, 0.35 * len(rows) + 1.5)))
    ax = fig.subplots()
    if rows.empty:
        _empty(ax, "No attributed kills among the top killers")
    else:
        labels = []
        ev_rows, ce_rows = [], []
        for yi, (_, r) in enumerate(rows.iterrows()):
            s, o, tid = str(r["sample_name"]), r["organoid_type"], int(r["target_track_id"])
            labels.append(f"{s} · {o} #{tid}")
            c = ce[(ce["sample_name"].astype(str) == s) & (ce["organoid_type"] == o)
                   & (ce["target_track_id"].astype(int) == tid)] if not ce.empty else ce
            for _, cr in c.iterrows():
                ax.barh(yi, cr["contact_end_t"] - cr["contact_start_t"] + 1, left=cr["contact_start_t"],
                        color="#E3F2FD", edgecolor="none", height=0.7)
                ce_rows.append(dict(cr, row=yi))
            e = ev[(ev["sample_name"].astype(str) == s) & (ev["organoid_type"] == o)
                   & (ev["target_track_id"].astype(int) == tid)] if not ev.empty else ev
            for _, er in e.iterrows():
                att = er["attribution_class"] in ATTRIBUTED
                ax.scatter(er["t_onset"], yi, s=20 + 2.0 * np.sqrt(er["onset_volume_um3"]),
                           color=_C_ATT if att else _C_UN, edgecolor="white", zorder=3)
                ev_rows.append(dict(er, row=yi))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Timepoint")
        handles = [
            ax.scatter([], [], color=_C_ATT, label="attributed death event"),
            ax.scatter([], [], color=_C_UN, label="unattributed death event"),
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
        tables[f"{name}_events"] = pd.DataFrame(ev_rows)
        tables[f"{name}_contacts"] = pd.DataFrame(ce_rows)
    ax.set_title(f"Targets of the top-{int(top_n)} killers: contact periods (blue bands) and death events "
                 f"(size = volume)", fontsize=10)
    out[name] = _save(fig, tables, plot_dir, name, log_fn)

    # 8. Engagement dose-response --------------------------------------------------------
    fig = Figure(figsize=(10, 4.2))
    ax1, ax2 = fig.subplots(1, 2)
    table = pd.DataFrame()
    need = {"n_attributed", "n_distinct_effectors", "total_contact_frames"}
    if tgt.empty or not need.issubset(tgt.columns):
        _empty(ax1, "No target data")
        _empty(ax2, "")
    else:
        table = tgt[["sample_name", "organoid_type", "target_track_id", "n_distinct_effectors",
                     "total_contact_frames", "n_attributed", "n_death_events"]]
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(table))
        ax1.scatter(table["n_distinct_effectors"] + jitter, table["n_attributed"], s=12, color=_C_ATT, alpha=0.6)
        ax1.set_xlabel("Distinct effectors that contacted the target")
        ax1.set_ylabel("Attributed death events")
        ax2.scatter(table["total_contact_frames"], table["n_attributed"], s=12, color=_C_ATT, alpha=0.6)
        ax2.set_xlabel("Total effector contact (timepoints)")
    fig.suptitle("Engagement dose-response per target (should rise; flat means something is wrong upstream)",
                 fontsize=10)
    out["engagement_dose_response"] = _save(fig, {"engagement_dose_response": table}, plot_dir,
                                            "engagement_dose_response", log_fn)
    return out


# ---------------------------------------------------------------------------
# Event GIFs
# ---------------------------------------------------------------------------

def render_killing_event_gifs(
    metadata: pd.DataFrame,
    output_dir,
    immune_type: str,
    results_dir,
    *,
    top_n: int = 5,
    frames_after_onset: int = 10,
    crop_half_width_px: int = 60,
    log_fn=None,
):
    """Render one GIF per top-N killer, centred on its largest attributed death event.

    Killers are ranked with :func:`behav3d.features.kill_attribution.rank_top_killers`
    (the same ranking as the swimmer plot and the viewer points). Each movie is a
    cropped max-intensity projection from the killer's last contact before the
    event to ``frames_after_onset`` after its onset: raw signal in grey, the dead
    mask in red, **the attributed death patch in yellow**, and the killer in
    purple. Written to ``<results_dir>/gallery/<sample>/``; returns the paths.
    """
    from PIL import Image as PILImage, ImageDraw
    from behav3d.features.death_events import (
        death_events_dir, load_death_patches, resolve_dead_mask_path, resolve_tracks_image_path)
    from behav3d.features.kill_attribution import rank_top_killers
    from behav3d.io.images import open_image_timepoints

    def _log(msg):
        if log_fn is not None:
            log_fn(msg)

    output_dir, results_dir = Path(output_dir), Path(results_dir)
    cand = _read(results_dir / f"kill_candidates_{immune_type}.csv")
    ev = _read(results_dir / f"death_events_{immune_type}.csv")
    eff = _read(results_dir / f"per_effector_killing_{immune_type}.csv")
    if cand.empty or ev.empty or eff.empty:
        return []
    top = rank_top_killers(eff, top_n)
    ev_idx = ev.set_index(["sample_name", "organoid_type", "death_event_id"])
    written = []
    handles = {}
    patch_cache = {}
    for _, killer in top.iterrows():
        sample, tid = str(killer["sample_name"]), int(killer["immune_track_id"])
        try:
            mine = cand[(cand["sample_name"].astype(str) == sample) & (cand["immune_track_id"] == tid)]
            best = mine.sort_values("attributed_death_volume_um3", ascending=False).iloc[0]
            e = ev_idx.loc[(best["sample_name"], best["organoid_type"], best["death_event_id"])]
            md = metadata[metadata["sample_name"].astype(str) == sample]
            if md.empty:
                continue
            row = md.iloc[0]
            if sample not in handles:
                img_dir = output_dir / "images" / sample
                raw = img_dir / f"{sample}.zarr"
                dead, _ = resolve_dead_mask_path(row, output_dir)
                imm = resolve_tracks_image_path(row, immune_type)
                handles[sample] = {k: (open_image_timepoints(p) if p is not None and Path(p).exists() else None)
                                   for k, p in (("raw", raw), ("dead", dead), ("imm", imm))}
            h = handles[sample]
            if h["raw"] is None:
                _log(f"  ⚠️ {sample}: no raw zarr for the GIF backdrop - skipped")
                continue
            key = (sample, best["organoid_type"])
            if key not in patch_cache:
                pp = death_events_dir(output_dir, best["organoid_type"]) / f"{sample}_death_patches.npz"
                patch_cache[key] = load_death_patches(pp) if pp.exists() else {}
            patch = patch_cache[key].get(int(best["death_event_id"]))
            cy, cx = int(round(e["y_onset_vox"])), int(round(e["x_onset_vox"]))
            t_from = int(best["last_contact_t"])
            t_onset = int(best["t_onset"])
            n_t = int(h["raw"].shape[0])
            t_to = min(n_t - 1, t_onset + int(frames_after_onset))
            ny, nx = int(h["raw"].shape[-2]), int(h["raw"].shape[-1])
            y1, y2 = max(0, cy - crop_half_width_px), min(ny, cy + crop_half_width_px)
            x1, x2 = max(0, cx - crop_half_width_px), min(nx, cx + crop_half_width_px)
            patch_mip = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            if patch is not None and len(patch):
                py, px = patch[:, 1] - y1, patch[:, 2] - x1
                ok = (py >= 0) & (py < y2 - y1) & (px >= 0) & (px < x2 - x1)
                patch_mip[py[ok], px[ok]] = True
            frames = []
            for t in range(t_from, t_to + 1):
                r = np.asarray(h["raw"][t])
                r = r[..., y1:y2, x1:x2]
                gray = (r.max(axis=0).max(axis=0) if r.ndim == 4 else r.max(axis=0)).astype(np.float32)
                p1, p99 = np.percentile(gray, [1, 99])
                gray = np.clip((gray - p1) / (p99 - p1 + 1e-10), 0, 1)
                rgb = np.stack([gray] * 3, axis=-1)
                if h["dead"] is not None:
                    d = np.asarray(h["dead"][t])[..., y1:y2, x1:x2].max(axis=0) > 0
                    rgb[d] = rgb[d] * 0.7 + np.array([0.3, 0.0, 0.0])
                if t >= t_onset:
                    rgb[patch_mip] = rgb[patch_mip] * 0.4 + np.array([0.6, 0.55, 0.0])
                if h["imm"] is not None:
                    m = (np.asarray(h["imm"][t])[..., y1:y2, x1:x2] == tid).max(axis=0)
                    rgb[m] = rgb[m] * 0.5 + np.array([0.5, 0.0, 0.5])
                pil = PILImage.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(pil)
                label = f"T={t}" + ("  death onset" if t == t_onset else "")
                draw.text((6, 6), label, fill="black")
                draw.text((5, 5), label, fill="white")
                frames.append(pil)
            if not frames:
                continue
            gdir = results_dir / "gallery" / sample
            gdir.mkdir(parents=True, exist_ok=True)
            gif = gdir / f"killing_event_{sample}_T{tid}_event{int(best['death_event_id'])}.gif"
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=200, loop=0)
            written.append(gif)
            _log(f"  ✓ {gif.name}")
        except Exception as exc:
            _log(f"  ⚠️ Skipped killer {sample} #{tid}: {exc}")
    return written

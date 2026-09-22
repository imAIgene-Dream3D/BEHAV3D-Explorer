"""
Attribution of localised death events to effector cells.

Every death event (see :mod:`behav3d.features.death_events`) carries exactly
one unit of killing credit in total. Credit is shared among the effectors
that (1) were in contact with the dying target inside the causal window
before the event's onset, and (2) are within ``attribution_radius_um`` of the
death patch, surface to surface. Total credit therefore equals the number of
attributed death events -- it tracks how much dying happened, never how many
effectors happened to be present.

Credit (fractional, the default headline)::

    chi(i,t) = 1{contact(i,L,t)} / n_targets(i,t)                 # hit sharing
    h_i      = (dt / c_ref) * sum_t chi(i,t) * exp(-(t0-t)*dt/tau)  # decay-weighted hit-equivalents
    g_i      = exp(-d_i / lambda)                                  # spatial locality
    w_i      = min(h_i, hit_cap) * g_i
    a_i      = w_i / sum_j w_j                                     # sum_i a_i = 1

``nearest`` and ``exclusive`` variants are always derived from the same table
(``is_nearest_effector``; ``n_candidates == 1``), so no mode switch is needed.

Credit is placed on each effector's **last contact frame before the onset**:
that row is guaranteed to exist (contact is read from it), which keeps the
per-timepoint credit conserved, and it keeps ``is_active_killing`` true only
on timepoints where the effector is touching the target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


@dataclass
class AttributionParams:
    """Attribution parameters.

    User-facing: ``causal_window_min`` and ``attribution_radius_um``.
    The rest are literature-anchored constants (overridable via
    ``active_killing.advanced``): tau from the ~57 min sublethal-damage
    half-life (Weigelin et al. 2021), c_ref the ~15 min typical contact,
    hit_cap the ~3 hits that saturate a lethal dose.
    """

    causal_window_min: float = 120.0
    attribution_radius_um: float = 15.0
    min_lag_min: float = 0.0
    damage_tau_min: float = 56.7 / math.log(2.0)
    contact_ref_min: float = 15.0
    hit_cap: float = 3.0
    spatial_kernel_um: Optional[float] = None   # None -> attribution_radius_um / 3
    exclude_border_patches: bool = True

    def resolved(self) -> "AttributionParams":
        out = AttributionParams(**asdict(self))
        if out.causal_window_min <= 0:
            raise ValueError("causal_window_min must be > 0")
        if out.attribution_radius_um <= 0:
            raise ValueError("attribution_radius_um must be > 0")
        if out.spatial_kernel_um is None:
            out.spatial_kernel_um = out.attribution_radius_um / 3.0
        return out


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

def explode_contacts(df_immune: pd.DataFrame, target_cell_types: Sequence[str]) -> pd.DataFrame:
    """One row per (sample, effector, t, target) the effector touches.

    ``touching_{type}s`` is built from the same ``real_distances <=
    contact_threshold`` test as ``{type}_contact_on_distance``, so membership
    in that list *is* the distance contact gate. ``n_targets`` counts every
    target touched at that timepoint, across all target types (hit sharing).
    """
    parts = []
    for ttype in target_cell_types:
        col = f"touching_{ttype}s"
        if col not in df_immune.columns:
            continue
        sub = df_immune[["sample_name", "TrackID", "position_t", col]].copy()
        sub[col] = sub[col].fillna("").astype(str)
        sub = sub[sub[col].str.strip().ne("") & sub[col].str.lower().ne("nan")]
        if sub.empty:
            continue
        sub = sub.assign(target_track_id=sub[col].str.split(",")).explode("target_track_id")
        sub["target_track_id"] = pd.to_numeric(sub["target_track_id"].str.strip(), errors="coerce")
        sub = sub.dropna(subset=["target_track_id"])
        sub["target_track_id"] = sub["target_track_id"].astype(float).astype(np.int64)
        sub["organoid_type"] = ttype
        parts.append(sub.drop(columns=[col]))
    cols = ["sample_name", "immune_track_id", "position_t", "organoid_type", "target_track_id", "n_targets"]
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True).rename(columns={"TrackID": "immune_track_id"})
    df["position_t"] = df["position_t"].astype(np.int64)
    df = df.drop_duplicates(["sample_name", "immune_track_id", "position_t", "organoid_type", "target_track_id"])
    df["n_targets"] = df.groupby(["sample_name", "immune_track_id", "position_t"])["target_track_id"].transform("size")
    return df[cols].reset_index(drop=True)


def identify_contact_events_per_target(df_contacts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group contacts into continuous events per (effector, target) pair.

    Returns ``(df_contact_events, df_contacts_with_ids)``. A contact event is a
    run of consecutive timepoints in which one effector touches one specific
    target. Grouping per pair (rather than "any target", as before) is what
    makes each engagement a separate, countable unit.
    """
    ev_cols = ["contact_event_id", "sample_name", "immune_track_id", "organoid_type", "target_track_id",
               "contact_start_t", "contact_end_t", "contact_duration"]
    if df_contacts.empty:
        return pd.DataFrame(columns=ev_cols), df_contacts.assign(contact_event_id=pd.Series(dtype=np.int64))
    key = ["sample_name", "immune_track_id", "organoid_type", "target_track_id"]
    df = df_contacts.sort_values(key + ["position_t"]).reset_index(drop=True)
    gap = df.groupby(key, sort=False)["position_t"].diff().ne(1)
    df["contact_event_id"] = gap.cumsum().astype(np.int64)
    ev = (df.groupby("contact_event_id")
          .agg(sample_name=("sample_name", "first"), immune_track_id=("immune_track_id", "first"),
               organoid_type=("organoid_type", "first"), target_track_id=("target_track_id", "first"),
               contact_start_t=("position_t", "min"), contact_end_t=("position_t", "max"),
               contact_duration=("position_t", "size"))
          .reset_index())
    return ev[ev_cols], df


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def _patch_distance_map(coords: np.ndarray, spacing, pad_vox, shape):
    lo = np.maximum(coords.min(axis=0) - pad_vox, 0)
    hi = np.minimum(coords.max(axis=0) + pad_vox + 1, np.asarray(shape))
    if np.any(hi <= lo):
        return None, None
    mask = np.zeros(tuple(int(v) for v in hi - lo), dtype=bool)
    local = coords - lo
    keep = np.all((local >= 0) & (local < (hi - lo)), axis=1)
    if not keep.any():
        return None, None
    local = local[keep]
    mask[local[:, 0], local[:, 1], local[:, 2]] = True
    return distance_transform_edt(~mask, sampling=spacing), tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def measure_event_effector_distances(
    df_pairs: pd.DataFrame,
    df_events: pd.DataFrame,
    patches: Dict[Tuple[str, str], Dict[int, np.ndarray]],
    df_timeseries: pd.DataFrame,
    immune_tracks_paths: Dict[str, str],
    sample_info: Dict[str, Dict],
    radius_um: float,
) -> pd.Series:
    """Surface-to-surface distance (µm) between each death patch and effector.

    ``df_pairs`` has one row per (event, effector, contact frame t). The
    patch, defined at its onset frame, is warped into frame ``t`` by the
    target's own centroid displacement before measuring -- the effector was
    somewhere else, earlier, and the target may have moved in between.
    Effectors with no voxels within ``radius_um`` of the patch get ``inf``.
    """
    from behav3d.io.images import open_image_timepoints

    out = pd.Series(np.inf, index=df_pairs.index, dtype=float)
    if df_pairs.empty:
        return out
    ev = df_events.set_index(["sample_name", "organoid_type", "death_event_id"])
    cent = None
    if not df_timeseries.empty:
        cent = df_timeseries.set_index(["sample_name", "organoid_type", "target_track_id", "position_t"])[
            ["centroid_z_vox", "centroid_y_vox", "centroid_x_vox"]]
    for sample_name, g_sample in df_pairs.groupby("sample_name"):
        path = immune_tracks_paths.get(sample_name)
        if path is None:
            continue
        handle = open_image_timepoints(path)
        spacing = sample_info[sample_name]["voxel_spacing"]
        pad = np.array([int(math.ceil(radius_um / s)) + 1 for s in spacing])
        shape = handle.shape[1:]
        for t, g_t in g_sample.groupby("position_t"):
            frame = None
            for (otype, eid), g_e in g_t.groupby(["organoid_type", "death_event_id"]):
                coords = patches.get((sample_name, otype), {}).get(int(eid))
                if coords is None or len(coords) == 0:
                    continue
                rec = ev.loc[(sample_name, otype, eid)]
                shift = np.zeros(3, dtype=np.int64)
                if cent is not None:
                    k_now = (sample_name, otype, int(rec["target_track_id"]), int(t))
                    k_on = (sample_name, otype, int(rec["target_track_id"]), int(rec["t_onset"]))
                    if k_now in cent.index and k_on in cent.index:
                        shift = np.round(cent.loc[k_now].to_numpy() - cent.loc[k_on].to_numpy()).astype(np.int64)
                dist, sl = _patch_distance_map(coords.astype(np.int64) + shift, spacing, pad, shape)
                if dist is None:
                    continue
                if frame is None:
                    frame = np.asarray(handle[int(t)])
                crop = frame[sl]
                for idx, iid in zip(g_e.index, g_e["immune_track_id"]):
                    sel = crop == int(iid)
                    if sel.any():
                        out.at[idx] = float(dist[sel].min())
    return out


# ---------------------------------------------------------------------------
# Credit
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = [
    "death_event_id", "sample_name", "organoid_type", "target_track_id", "t_onset",
    "immune_type", "immune_track_id", "last_contact_t", "contact_event_id", "n_contact_frames",
    "adjusted_contact_min", "lag_min", "distance_to_death_um", "distance_mode",
    "hit_weight_raw", "kill_credit", "is_nearest_effector", "n_candidates", "attribution_class",
    "onset_volume_um3", "attributed_death_volume_um3",
]


def attribute_death_events(
    df_events: pd.DataFrame,
    df_contacts: pd.DataFrame,
    patches: Dict[Tuple[str, str], Dict[int, np.ndarray]],
    df_timeseries: pd.DataFrame,
    *,
    immune_type: str,
    immune_tracks_paths: Dict[str, str],
    sample_info: Dict[str, Dict],
    params: AttributionParams,
    timepoint_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    return_pairs: bool = False,
):
    """Attribute every death event to the effectors that plausibly caused it.

    ``df_contacts`` must carry ``contact_event_id`` (from
    :func:`identify_contact_events_per_target`). Returns
    ``(df_candidates, df_events_annotated)``; the candidate table has one row
    per (event, candidate effector) and ``groupby(event).kill_credit.sum()``
    is exactly 1 for every attributed event.
    """
    p = params.resolved()
    events = df_events.copy()
    if events.empty:
        events["attribution_class"] = pd.Series(dtype=str)
        events["n_candidates"] = pd.Series(dtype=np.int64)
        empty = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        return (empty, events, pd.DataFrame()) if return_pairs else (empty, events)

    range_start = 0 if timepoint_range is None or timepoint_range[0] is None else int(timepoint_range[0])
    rows = []
    for sample_name, g in events.groupby("sample_name"):
        dt = float(sample_info[sample_name]["minutes_per_frame"])
        w_max = int(math.ceil(p.causal_window_min / dt))
        w_min = int(math.floor(p.min_lag_min / dt))
        sc = df_contacts[df_contacts["sample_name"] == sample_name]
        for _, e in g.iterrows():
            t0 = int(e["t_onset"])
            lo, hi = t0 - w_max, t0 - w_min
            m = ((sc["organoid_type"] == e["organoid_type"]) & (sc["target_track_id"] == int(e["target_track_id"]))
                 & (sc["position_t"] >= max(lo, range_start)) & (sc["position_t"] <= hi))
            hit = sc.loc[m, ["immune_track_id", "position_t", "n_targets", "contact_event_id"]].copy()
            hit["death_event_id"] = int(e["death_event_id"])
            hit["organoid_type"] = e["organoid_type"]
            hit["sample_name"] = sample_name
            hit["window_truncated"] = lo < range_start
            rows.append(hit)
    pairs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    truncated = {}
    for sample_name, g in events.groupby("sample_name"):
        dt = float(sample_info[sample_name]["minutes_per_frame"])
        w_max = int(math.ceil(p.causal_window_min / dt))
        for idx, e in g.iterrows():
            truncated[idx] = (int(e["t_onset"]) - w_max) < range_start
    events["window_truncated"] = pd.Series(truncated)

    if p.exclude_border_patches and "is_border_patch" in events.columns and not pairs.empty:
        border = events.loc[events["is_border_patch"].astype(bool), ["sample_name", "organoid_type", "death_event_id"]]
        if not border.empty:
            pairs = pairs.merge(border.assign(_b=True), on=["sample_name", "organoid_type", "death_event_id"], how="left")
            pairs = pairs[pairs["_b"].isna()].drop(columns="_b")

    if not pairs.empty:
        pairs = pairs.reset_index(drop=True)
        pairs["distance_um"] = measure_event_effector_distances(
            pairs, events, patches, df_timeseries, immune_tracks_paths, sample_info, p.attribution_radius_um)
    cand, events = credit_from_pairs(pairs, events, params=p, sample_info=sample_info, immune_type=immune_type)
    if return_pairs:
        return cand, events, pairs
    return cand, events


def credit_from_pairs(
    pairs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    params: AttributionParams,
    sample_info: Dict[str, Dict],
    immune_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign credit from (event, effector, contact frame) pairs with measured distances.

    Split out of :func:`attribute_death_events` so validation can re-credit a
    subsample of effectors (or a permuted geometry) without re-reading images.
    ``events`` must already carry ``window_truncated``.
    """
    p = params.resolved()
    events = events.copy()
    cand = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if not pairs.empty:
        ev_lookup = events.set_index(["sample_name", "organoid_type", "death_event_id"])
        recs = []
        for (sample_name, otype, eid), g in pairs.groupby(["sample_name", "organoid_type", "death_event_id"]):
            e = ev_lookup.loc[(sample_name, otype, eid)]
            dt = float(sample_info[sample_name]["minutes_per_frame"])
            t0 = int(e["t_onset"])
            per = []
            for iid, gi in g.groupby("immune_track_id"):
                d = float(gi["distance_um"].min())
                if not np.isfinite(d) or d > p.attribution_radius_um:
                    continue
                chi = 1.0 / gi["n_targets"].astype(float).to_numpy()
                decay = np.exp(-(t0 - gi["position_t"].to_numpy()) * dt / p.damage_tau_min)
                h = (dt / p.contact_ref_min) * float(np.sum(chi * decay))
                w = min(h, p.hit_cap) * math.exp(-d / p.spatial_kernel_um)
                last = gi.loc[gi["position_t"].idxmax()]
                per.append({
                    "death_event_id": int(eid), "sample_name": sample_name, "organoid_type": otype,
                    "target_track_id": int(e["target_track_id"]), "t_onset": t0, "immune_type": immune_type,
                    "immune_track_id": int(iid), "last_contact_t": int(last["position_t"]),
                    "contact_event_id": int(last["contact_event_id"]), "n_contact_frames": int(len(gi)),
                    "adjusted_contact_min": float(np.sum(chi) * dt),
                    "lag_min": float((t0 - int(last["position_t"])) * dt),
                    "distance_to_death_um": d, "distance_mode": "surface", "hit_weight_raw": float(w),
                    "onset_volume_um3": float(e["onset_volume_um3"]),
                })
            if not per:
                continue
            total = sum(r["hit_weight_raw"] for r in per)
            nearest = min(per, key=lambda r: (r["distance_to_death_um"], -r["hit_weight_raw"], r["immune_track_id"]))
            for r in per:
                r["kill_credit"] = r["hit_weight_raw"] / total
                r["is_nearest_effector"] = r is nearest
                r["n_candidates"] = len(per)
                r["attribution_class"] = "exclusive" if len(per) == 1 else "shared"
                r["attributed_death_volume_um3"] = r["kill_credit"] * r["onset_volume_um3"]
            recs.extend(per)
        if recs:
            cand = pd.DataFrame(recs)[CANDIDATE_COLUMNS]

    n_cand = cand.groupby(["sample_name", "organoid_type", "death_event_id"]).size() if not cand.empty else pd.Series(dtype=int)
    keys = list(zip(events["sample_name"], events["organoid_type"], events["death_event_id"]))
    events["n_candidates"] = [int(n_cand.get(k, 0)) for k in keys]
    # Effectors that touched the target inside the causal window, whatever their
    # distance to the patch: the funnel step between "no contact at all" and
    # "contacted, but too far from this patch".
    n_touch = (pairs.groupby(["sample_name", "organoid_type", "death_event_id"])["immune_track_id"].nunique()
               if not pairs.empty else pd.Series(dtype=int))
    events["n_contacting_in_window"] = [int(n_touch.get(k, 0)) for k in keys]
    cls = []
    for (_, e), n in zip(events.iterrows(), events["n_candidates"]):
        if p.exclude_border_patches and bool(e.get("is_border_patch", False)):
            cls.append("excluded_border")
        elif n == 1:
            cls.append("exclusive")
        elif n > 1:
            cls.append("shared")
        elif bool(e["window_truncated"]):
            # Its candidate set reaches back before the analysed window, so the
            # absence of a cause is unknowable -- not evidence of background death.
            cls.append("unattributed_truncated")
        else:
            cls.append("unattributed")
    events["attribution_class"] = cls
    return cand, events


# ---------------------------------------------------------------------------
# Per-timepoint placement
# ---------------------------------------------------------------------------

def per_timepoint_credit(df_candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse the candidate table onto (sample, effector, timepoint) rows.

    Several events can land on the same row; credit and weight add, and the
    descriptive columns come from the event with the largest credit there.
    """
    cols = ["sample_name", "TrackID", "position_t", "kill_credit", "hit_weight_raw", "is_nearest_effector",
            "death_event_id", "targeted_cell_type", "targeted_track_id", "attribution_class",
            "distance_to_death_um", "lag_min", "n_cokillers", "attributed_death_volume_um3", "n_events_credited"]
    if df_candidates.empty:
        return pd.DataFrame(columns=cols)
    c = df_candidates.sort_values("kill_credit", ascending=False)
    key = ["sample_name", "immune_track_id", "last_contact_t"]
    sums = c.groupby(key).agg(kill_credit=("kill_credit", "sum"), hit_weight_raw=("hit_weight_raw", "sum"),
                              is_nearest_effector=("is_nearest_effector", "any"),
                              attributed_death_volume_um3=("attributed_death_volume_um3", "sum"),
                              n_events_credited=("death_event_id", "size"))
    top = c.drop_duplicates(key).set_index(key)[
        ["death_event_id", "organoid_type", "target_track_id", "attribution_class", "distance_to_death_um",
         "lag_min", "n_candidates"]]
    out = sums.join(top).reset_index().rename(columns={
        "immune_track_id": "TrackID", "last_contact_t": "position_t", "organoid_type": "targeted_cell_type",
        "target_track_id": "targeted_track_id"})
    out["n_cokillers"] = out.pop("n_candidates") - 1
    return out[cols]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def rank_top_killers(df_per_effector: pd.DataFrame, n: int) -> pd.DataFrame:
    """The top-``n`` effectors by attributed kills (tie-break: attributed death volume).

    The single ranking used by the GIF gallery, the viewer points and the
    swimmer plot, so all three always show the same cells.
    """
    if df_per_effector is None or df_per_effector.empty:
        return pd.DataFrame(columns=getattr(df_per_effector, "columns", []))
    ranked = df_per_effector[df_per_effector["kills_attributed"] > 0].sort_values(
        ["kills_attributed", "attributed_death_volume_um3", "sample_name", "immune_track_id"],
        ascending=[False, False, True, True])
    return ranked.head(int(n)).reset_index(drop=True)


def _gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() <= 0:
        return float("nan")
    n = v.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum())) - (n + 1.0) / n)


def _top_share(values: np.ndarray, frac: float = 0.1) -> float:
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    if v.size == 0 or v.sum() <= 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * v.size)))
    return float(v[:k].sum() / v.sum())


def summarize_per_effector(df_candidates, df_immune, df_contact_events, sample_info) -> pd.DataFrame:
    base = df_immune.groupby(["sample_name", "TrackID"]).agg(
        n_timepoints=("position_t", "size"), first_t=("position_t", "min"),
        last_t=("position_t", "max")).reset_index()
    base["track_minutes"] = [n * sample_info[s]["minutes_per_frame"] for s, n in zip(base["sample_name"], base["n_timepoints"])]
    ce = df_contact_events.groupby(["sample_name", "immune_track_id"]).agg(
        n_contact_events=("contact_event_id", "size"), contact_frames=("contact_duration", "sum"),
        first_contact_t=("contact_start_t", "min"), n_targets_contacted=("target_track_id", "nunique")
    ).reset_index().rename(columns={"immune_track_id": "TrackID"}) if not df_contact_events.empty else pd.DataFrame(
        columns=["sample_name", "TrackID", "n_contact_events", "contact_frames", "first_contact_t", "n_targets_contacted"])
    out = base.merge(ce, on=["sample_name", "TrackID"], how="left")
    for col in ("n_contact_events", "contact_frames", "n_targets_contacted"):
        out[col] = out[col].fillna(0).astype(np.int64)
    out["contact_minutes"] = [f * sample_info[s]["minutes_per_frame"] for s, f in zip(out["sample_name"], out["contact_frames"])]

    stats = []
    if not df_candidates.empty:
        for (s, iid), g in df_candidates.groupby(["sample_name", "immune_track_id"]):
            dt = sample_info[s]["minutes_per_frame"]
            per_target = g.groupby(["organoid_type", "target_track_id"])["kill_credit"].sum()
            principal = g[g["is_nearest_effector"]].sort_values("t_onset")
            ikis = np.diff(principal["t_onset"].to_numpy()) * dt
            stats.append({
                "sample_name": s, "TrackID": int(iid),
                "kills_attributed": float(g["kill_credit"].sum()),
                "kills_exclusive": int((g["n_candidates"] == 1).sum()),
                "kills_nearest": int(g["is_nearest_effector"].sum()),
                "hit_weight_total": float(g["hit_weight_raw"].sum()),
                "n_distinct_targets_killed": int((per_target > 0).sum()),
                "is_serial_killer": bool((per_target >= 0.5).sum() >= 2),
                "inter_kill_interval_min": float(np.median(ikis)) if ikis.size else float("nan"),
                "first_kill_onset_t": int(g["t_onset"].min()),
                "attributed_death_volume_um3": float(g["attributed_death_volume_um3"].sum()),
            })
    st = pd.DataFrame(stats) if stats else pd.DataFrame(columns=[
        "sample_name", "TrackID", "kills_attributed", "kills_exclusive", "kills_nearest", "hit_weight_total",
        "n_distinct_targets_killed", "is_serial_killer", "inter_kill_interval_min", "first_kill_onset_t",
        "attributed_death_volume_um3"])
    out = out.merge(st, on=["sample_name", "TrackID"], how="left")
    for col in ("kills_attributed", "hit_weight_total", "attributed_death_volume_um3"):
        out[col] = out[col].fillna(0.0).astype(float)
    for col in ("kills_exclusive", "kills_nearest", "n_distinct_targets_killed"):
        out[col] = out[col].fillna(0).astype(np.int64)
    out["is_serial_killer"] = out["is_serial_killer"].fillna(False).astype(bool)
    hours = out["track_minutes"] / 60.0
    chours = out["contact_minutes"] / 60.0
    out["killing_rate_per_hour"] = np.where(hours > 0, out["kills_attributed"] / hours, np.nan)
    out["hit_delivery_rate"] = np.where(hours > 0, out["hit_weight_total"] / hours, np.nan)
    out["kill_per_contact_hour"] = np.where(chours > 0, out["kills_attributed"] / chours, np.nan)
    out["kill_per_contact_event"] = np.where(out["n_contact_events"] > 0,
                                             out["kills_attributed"] / out["n_contact_events"], np.nan)
    out["serial_index"] = np.where(out["n_distinct_targets_killed"] > 0,
                                   out["kills_attributed"] / out["n_distinct_targets_killed"], np.nan)
    ttfk = []
    for s, fk, fc in zip(out["sample_name"], out["first_kill_onset_t"], out["first_contact_t"]):
        ttfk.append((fk - fc) * sample_info[s]["minutes_per_frame"] if pd.notna(fk) and pd.notna(fc) else np.nan)
    out["time_to_first_kill_min"] = ttfk
    return out.drop(columns=["first_kill_onset_t"]).rename(columns={"TrackID": "immune_track_id"})


def summarize_per_target(df_events_annot, df_candidates, df_timeseries, df_contact_events, sample_info) -> pd.DataFrame:
    key = ["sample_name", "organoid_type", "target_track_id"]
    if df_timeseries.empty:
        return pd.DataFrame(columns=key)
    ts = df_timeseries.sort_values("position_t")
    out = ts.groupby(key).agg(
        n_timepoints=("position_t", "size"), total_new_dead_volume_um3=("new_dead_volume_um3", "sum"),
        final_dead_volume_um3=("dead_volume_um3", "last"), final_volume_um3=("organoid_volume_um3", "last"),
    ).reset_index()
    out["final_dead_fraction"] = np.where(out["final_volume_um3"] > 0,
                                          out["final_dead_volume_um3"] / out["final_volume_um3"], np.nan)
    ev = df_events_annot[~df_events_annot.get("attribution_class", pd.Series(dtype=str)).eq("excluded_border")] \
        if not df_events_annot.empty else df_events_annot
    if not ev.empty:
        cls = ev.assign(
            _att=ev["attribution_class"].isin(["exclusive", "shared"]),
            _un=ev["attribution_class"].eq("unattributed"),
            _unt=ev["attribution_class"].eq("unattributed_truncated"),
            _vatt=np.where(ev["attribution_class"].isin(["exclusive", "shared"]), ev["onset_volume_um3"], 0.0),
        ).groupby(key).agg(n_death_events=("death_event_id", "size"), n_attributed=("_att", "sum"),
                           n_unattributed=("_un", "sum"), n_unattributed_truncated=("_unt", "sum"),
                           dead_volume_attributed_um3=("_vatt", "sum"),
                           dead_volume_events_um3=("onset_volume_um3", "sum"),
                           first_onset_t=("t_onset", "min")).reset_index()
        out = out.merge(cls, on=key, how="left")
    for col in ("n_death_events", "n_attributed", "n_unattributed", "n_unattributed_truncated"):
        out[col] = out[col].fillna(0).astype(np.int64) if col in out else 0
    for col in ("dead_volume_attributed_um3", "dead_volume_events_um3"):
        out[col] = out[col].fillna(0.0) if col in out else 0.0
    out["immune_attributed_fraction_events"] = np.where(out["n_death_events"] > 0,
                                                        out["n_attributed"] / out["n_death_events"], np.nan)
    out["immune_attributed_fraction_volume"] = np.where(out["dead_volume_events_um3"] > 0,
                                                        out["dead_volume_attributed_um3"] / out["dead_volume_events_um3"], np.nan)
    if not df_candidates.empty:
        amb = df_candidates.groupby(key + ["death_event_id"])["kill_credit"].max().groupby(level=[0, 1, 2]).apply(
            lambda s: float(np.mean(1.0 - s))).rename("death_attribution_ambiguity").reset_index()
        out = out.merge(amb, on=key, how="left")
    else:
        out["death_attribution_ambiguity"] = np.nan
    if not df_contact_events.empty:
        ce = df_contact_events.groupby(key).agg(
            n_distinct_effectors=("immune_track_id", "nunique"), n_effector_contacts=("contact_event_id", "size"),
            total_contact_frames=("contact_duration", "sum"), first_contact_t=("contact_start_t", "min")).reset_index()
        out = out.merge(ce, on=key, how="left")
        if "first_onset_t" in out:
            starts = df_contact_events[key + ["contact_start_t"]]
            before = out[key + ["first_onset_t"]].merge(starts, on=key, how="left")
            before = before[before["contact_start_t"] < before["first_onset_t"]].groupby(key).size().rename(
                "n_contacts_before_first_death").reset_index()
            out = out.merge(before, on=key, how="left")
    for col in ("n_distinct_effectors", "n_effector_contacts", "total_contact_frames", "n_contacts_before_first_death"):
        if col in out:
            out[col] = out[col].fillna(0).astype(np.int64)
    if "first_contact_t" in out and "first_onset_t" in out:
        dts = out["sample_name"].map(lambda s: sample_info[s]["minutes_per_frame"])
        out["lag_first_contact_to_first_onset_min"] = (out["first_onset_t"] - out["first_contact_t"]) * dts
        out.loc[out["lag_first_contact_to_first_onset_min"] < 0, "lag_first_contact_to_first_onset_min"] = np.nan
    return out


def summarize_per_sample(df_events_annot, df_candidates, df_per_effector, df_per_target, df_contact_events,
                         df_timeseries, sample_info, params: AttributionParams) -> pd.DataFrame:
    rows = []
    samples = sorted(set(df_per_target.get("sample_name", pd.Series(dtype=str))) |
                     set(df_per_effector.get("sample_name", pd.Series(dtype=str))))
    for s in samples:
        dt = sample_info[s]["minutes_per_frame"]
        ev = df_events_annot[df_events_annot["sample_name"] == s] if not df_events_annot.empty else df_events_annot
        ev = ev[ev["attribution_class"] != "excluded_border"] if not ev.empty else ev
        n_ev = int(len(ev))
        n_att = int(ev["attribution_class"].isin(["exclusive", "shared"]).sum()) if n_ev else 0
        n_un = int((ev["attribution_class"] == "unattributed").sum()) if n_ev else 0
        n_unt = int((ev["attribution_class"] == "unattributed_truncated").sum()) if n_ev else 0
        n_ce = int((df_contact_events["sample_name"] == s).sum()) if not df_contact_events.empty else 0
        org_hours = (float((df_timeseries["sample_name"] == s).sum()) * dt / 60.0) if not df_timeseries.empty else 0.0
        eff = df_per_effector[df_per_effector["sample_name"] == s]
        engaged = eff[eff["n_contact_events"] > 0]
        cand = df_candidates[df_candidates["sample_name"] == s] if not df_candidates.empty else df_candidates
        amb = (cand.groupby("death_event_id")["kill_credit"].max().rsub(1.0).mean() if not cand.empty else np.nan)
        vol_att = float(ev.loc[ev["attribution_class"].isin(["exclusive", "shared"]), "onset_volume_um3"].sum()) if n_ev else 0.0
        vol_all = float(ev["onset_volume_um3"].sum()) if n_ev else 0.0
        eff_hours = float(eff["track_minutes"].sum()) / 60.0
        rows.append({
            "sample_name": s,
            "conversion_rate": n_att / n_ce if n_ce else np.nan,
            "background_death_rate": n_un / org_hours if org_hours else np.nan,
            "attribution_ambiguity": float(amb) if pd.notna(amb) else np.nan,
            "immune_attributed_fraction_events": n_att / n_ev if n_ev else np.nan,
            "immune_attributed_fraction_volume": vol_att / vol_all if vol_all else np.nan,
            "attributed_events_per_effector_hour": n_att / eff_hours if eff_hours else np.nan,
            "killing_gini": _gini(engaged["kills_attributed"].to_numpy()),
            "top10pct_share": _top_share(engaged["kills_attributed"].to_numpy()),
            "median_lag_min": float(cand["lag_min"].median()) if not cand.empty else np.nan,
            "median_inter_kill_interval_min": float(eff["inter_kill_interval_min"].median()) if len(eff) else np.nan,
            "serial_killer_fraction": float(engaged["is_serial_killer"].mean()) if len(engaged) else np.nan,
            "n_death_events": n_ev, "n_attributed": n_att, "n_unattributed": n_un,
            "n_unattributed_truncated": n_unt, "n_contact_events": n_ce,
            "n_effectors": int(len(eff)), "n_engaged_effectors": int(len(engaged)),
            "total_kill_credit": float(cand["kill_credit"].sum()) if not cand.empty else 0.0,
            "attribution_radius_um": float(params.attribution_radius_um),
            "causal_window_min": float(params.causal_window_min),
        })
    return pd.DataFrame(rows)

"""
Advanced Feature Extraction for BEHAV3D

This module provides analyses that go beyond basic track feature extraction,
chiefly Active Killing.

-------------------------------------
------------ ACTIVE KILLING ---------
-------------------------------------

Active Killing attributes **localised death events** to the effector cells
that plausibly caused them.

1. Death events (``behav3d.features.death_events``): the nucleation of a
   connected region of *newly* dead voxels inside one target track, read from
   the annotated dead mask (never by re-thresholding the raw death channel).
   No effector data enters detection, so the number of events cannot grow
   when more effectors are present. Cached per target type under
   ``analysis/<target>/death_events/``.
2. Contact events: continuous runs in which one effector touches one specific
   target (``touching_<type>s``, i.e. the distance contact gate).
3. Attribution (``behav3d.features.kill_attribution``): each event's one unit
   of credit is shared among effectors that contacted the target within
   ``causal_window_min`` before onset and are within ``attribution_radius_um``
   of the death patch (surface to surface).

Because every event carries exactly one unit of credit in total, total credit
equals the number of attributed death events: it measures how much dying
happened, not how many effectors were touching. The previous algorithm gave
every contacting effector an identical full verdict and inflated with density.

Outputs (``analysis/<immune>/active_killing/<target|combined>/``)
- ``kill_candidates_<immune>.csv``: one row per (death event, candidate effector)
- ``death_events_<immune>.csv``: every death event with its attribution class
- ``per_effector_killing_<immune>.csv``, ``per_target_death_<immune>.csv``,
  ``per_sample_killing_<immune>.csv``
- ``contact_events_<immune>.csv``: one row per (effector, target) contact event
- ``BEHAV3D_<immune>_advanced_track_features.csv``: the effector per-timepoint
  table plus the kill columns (credit sits on each effector's last contact
  frame before the onset)
- ``active_killing_run_params.json``: schema version + resolved parameters

Active Killing deliberately scans the full movie from the RAW track features;
analysis windows are applied where the outputs are read.
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from behav3d.core.metadata import detect_organoid_types_from_metadata
from behav3d.core.utils import get_current_time, format_time
from behav3d.io.images import load_image


ACTIVE_KILLING_SCHEMA_VERSION = 2
RUN_PARAMS_FILENAME = "active_killing_run_params.json"

# Per-timepoint columns added to the effector feature table (all NA-free).
KILL_COLUMN_DEFAULTS = {
    "is_active_killing": False,
    "kill_credit": 0.0,
    "cum_kill_credit": 0.0,
    "hit_weight_raw": 0.0,
    "is_nearest_effector": False,
    "death_event_id": -1,
    "targeted_cell_type": "",
    "targeted_track_id": -1,
    "attribution_class": "",
    "distance_to_death_um": -1.0,
    "lag_min": -1.0,
    "n_cokillers": 0,
    "attributed_death_volume_um3": 0.0,
    "n_events_credited": 0,
    "contact_event_id": -1,
}


class StaleDataError(RuntimeError):
    """Raised when a derived CSV predates the raw input it was built from."""


class ActiveKillingInputError(ValueError):
    """Raised when the inputs cannot support attribution (rather than silently yielding zeros)."""


def _newest_mtime(paths: Iterable[Union[str, Path]]) -> Tuple[float, Optional[Path]]:
    """Newest mtime over existing paths. For a .zarr store, stat the store
    directory and its array metadata only -- never walk the chunks."""
    best, which = -1.0, None
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        m = p.stat().st_mtime
        for meta in ("zarr.json", ".zarray"):
            mp = p / meta
            if mp.exists():
                m = max(m, mp.stat().st_mtime)
        if m > best:
            best, which = m, p
    return best, which


def _check_not_stale(derived_path: Path, raw_path, derived_label: str, upstream_label: str) -> None:
    """
    Raise StaleDataError if ``raw_path`` (a path or several) was modified more
    recently than ``derived_path`` -- meaning ``upstream_label`` was rerun after
    ``derived_label`` last produced this file.
    """
    derived_path = Path(derived_path)
    if not derived_path.exists():
        return
    paths = [raw_path] if isinstance(raw_path, (str, Path)) else list(raw_path or [])
    newest, which = _newest_mtime(paths)
    if which is not None and newest > derived_path.stat().st_mtime:
        raise StaleDataError(
            f"{derived_path.name} is older than {which.name} — it looks like "
            f"{upstream_label} was rerun after {derived_label} last produced this file. "
            f"The underlying data no longer matches. Re-run {derived_label} to refresh "
            f"it from the current data before continuing."
        )


def _print_rerun_filtering_warning(stale_filtered_cell_types: set) -> None:
    """
    Print a warning that any existing filtered CSV for these cell types was
    built before this Active Killing run and should be refreshed.

    Filtering reads Active Killing's advanced-features CSV as its input when
    one exists (see find_advanced_features_csv / filter_tracks' df_input_path),
    so a filtered CSV produced before this run does not include the killing
    columns this run just (re)computed.
    """
    if not stale_filtered_cell_types:
        return
    cell_types = ", ".join(sorted(stale_filtered_cell_types))
    print(
        f"{get_current_time()} - WARNING: Active Killing just ran using the unfiltered track "
        f"features. The existing filtered CSV for {cell_types} was produced before this run and "
        f"does not include these Active Killing results. Re-run Filtering to refresh it."
    )


def _read_run_params(results_dir: Path) -> Optional[dict]:
    p = Path(results_dir) / RUN_PARAMS_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_advanced_features_csv(output_dir: Union[str, Path], cell_type: str) -> Optional[Path]:
    """
    Locate the active-killing advanced-features CSV for a cell type.

    run_active_killing_analysis() writes into a per-target subfolder (the
    target cell type name, or "combined" for multi-target runs), so this
    searches analysis/<cell_type>/active_killing/*/. Returns the most recently
    modified match, or None if none exist.

    Raises StaleDataError if the best match predates the cell type's raw
    combined_track_features.csv (Feature Extraction was rerun after Active
    Killing), or was produced by the previous Active Killing algorithm (a
    different column schema that must not be consumed as if it were current).
    """
    output_dir = Path(output_dir)
    active_killing_dir = output_dir / "analysis" / cell_type / "active_killing"
    if not active_killing_dir.exists():
        return None
    filename = f"BEHAV3D_{cell_type}_advanced_track_features.csv"
    candidates = list(active_killing_dir.glob(f"*/{filename}"))
    legacy = active_killing_dir / filename
    if legacy.exists():
        candidates.append(legacy)
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    raw_path = output_dir / "analysis" / cell_type / "track_features" / f"BEHAV3D_{cell_type}_combined_track_features.csv"
    _check_not_stale(best, raw_path, derived_label="Active Killing", upstream_label="Feature Extraction")
    run_params = _read_run_params(best.parent)
    if not run_params or int(run_params.get("schema_version", 0)) < ACTIVE_KILLING_SCHEMA_VERSION:
        raise StaleDataError(
            f"{best.name} was produced by the previous Active Killing algorithm, whose columns "
            f"(e.g. killing_efficiency) no longer exist and whose counts inflated with effector "
            f"density. Re-run Active Killing to regenerate it before continuing."
        )
    _check_not_stale(best, run_params.get("upstream_paths", []),
                     derived_label="Active Killing", upstream_label="Segmentation / Tracking")
    return best


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def resolve_active_killing_params(
    target_cell_diameter_um: float = 10.0,
    causal_window_min: float = 120.0,
    attribution_radius_um: float = 15.0,
    advanced: Optional[dict] = None,
):
    """Build (DeathEventParams, AttributionParams) from the three user-facing
    values plus any ``advanced`` overrides (keys matching either dataclass)."""
    from dataclasses import fields
    from behav3d.features.death_events import DeathEventParams
    from behav3d.features.kill_attribution import AttributionParams

    advanced = dict(advanced or {})
    dnames = {f.name for f in fields(DeathEventParams)}
    anames = {f.name for f in fields(AttributionParams)}
    unknown = sorted(set(advanced) - dnames - anames)
    if unknown:
        raise ValueError(f"Unknown active_killing.advanced keys: {unknown}")
    dp = DeathEventParams(target_cell_diameter_um=float(target_cell_diameter_um),
                          **{k: v for k, v in advanced.items() if k in dnames and k != "target_cell_diameter_um"})
    ap = AttributionParams(causal_window_min=float(causal_window_min),
                           attribution_radius_um=float(attribution_radius_um),
                           **{k: v for k, v in advanced.items()
                              if k in anames and k not in ("causal_window_min", "attribution_radius_um")})
    return dp, ap


def _validate_contact_inputs(df_immune: pd.DataFrame, target_cell_types: List[str], immune_cell_type: str) -> None:
    """Refuse to run on inputs that would silently yield zero attribution."""
    cols = [f"touching_{t}s" for t in target_cell_types]
    present = [c for c in cols if c in df_immune.columns]
    if not present:
        raise ActiveKillingInputError(
            f"The {immune_cell_type} track features have no contact columns for "
            f"{', '.join(target_cell_types)} (expected {', '.join(cols)}). Enable 'contact' in "
            f"Feature Extraction for {immune_cell_type} and re-run it -- Active Killing needs to "
            f"know which target each effector touches."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_active_killing_analysis(
    metadata: pd.DataFrame,
    output_dir: Union[str, Path],
    immune_cell_type: str = "tcell",
    target_cell_types: Optional[List[str]] = None,
    target_cell_diameter_um: float = 10.0,
    causal_window_min: float = 120.0,
    attribution_radius_um: float = 15.0,
    advanced: Optional[dict] = None,
    save_results: bool = True,
    output_subfolder: str = "",
    reuse_death_event_cache: bool = True,
    progress_cb=None,
    top_n_killers: int = 5,
    write_figures: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run Active Killing: detect death events, attribute them to effectors, and
    write the per-event, per-effector, per-target and per-sample tables.

    Returns ``(df_candidates, df_per_sample, stats)``. ``df_candidates`` has one
    row per (death event, candidate effector); ``kill_credit`` sums to exactly
    1 per attributed event.
    """
    from behav3d.features.death_events import detect_death_events, resolve_tracks_image_path
    from behav3d.features.kill_attribution import (
        attribute_death_events, explode_contacts, identify_contact_events_per_target,
        per_timepoint_credit, summarize_per_effector, summarize_per_sample, summarize_per_target,
    )

    print("--------------- Running Active Killing Analysis ---------------")
    start_time = time.time()
    output_dir = Path(output_dir)
    if target_cell_types is None:
        target_cell_types = detect_organoid_types_from_metadata(metadata)
        if not target_cell_types:
            raise ValueError("No organoid types detected in metadata. Please specify target_cell_types.")
    target_cell_types = list(target_cell_types)
    dparams, aparams = resolve_active_killing_params(
        target_cell_diameter_um, causal_window_min, attribution_radius_um, advanced)
    print(f"Effector: {immune_cell_type} | targets: {target_cell_types}")
    print(f"Target cell diameter: {dparams.target_cell_diameter_um} µm | causal window: "
          f"{aparams.causal_window_min} min | attribution radius: {aparams.attribution_radius_um} µm")

    # RAW (unfiltered) features only: Filtering in turn reads this module's
    # advanced CSV as its input, so reading the filtered CSV here would make
    # each step the other's upstream. The zarr masks and the death-event cache
    # add upstream edges but no cycle -- nothing downstream writes them.
    stale_filtered = set()
    immune_dir = output_dir / "analysis" / immune_cell_type / "track_features"
    immune_raw = immune_dir / f"BEHAV3D_{immune_cell_type}_combined_track_features.csv"
    if (immune_dir / f"BEHAV3D_{immune_cell_type}_combined_track_features_filtered.csv").exists():
        stale_filtered.add(immune_cell_type)
    if not immune_raw.exists():
        raise FileNotFoundError(f"Could not find immune cell tracks at {immune_raw}")
    print(f"{get_current_time()} - Loading effector tracks from {immune_raw}")
    df_immune = pd.read_csv(immune_raw)
    for col in df_immune.columns:
        if col.startswith("touching_"):
            df_immune[col] = df_immune[col].fillna("").astype(str)
    _validate_contact_inputs(df_immune, target_cell_types, immune_cell_type)

    # Samples that actually have this effector.
    md = metadata.copy()
    immune_paths = {}
    for _, row in md.iterrows():
        p = resolve_tracks_image_path(row, immune_cell_type)
        if p is not None and p.exists():
            immune_paths[str(row["sample_name"])] = str(p)
    md = md[md["sample_name"].astype(str).isin(set(df_immune["sample_name"].astype(str)))]
    df_immune = df_immune[df_immune["sample_name"].astype(str).isin(set(md["sample_name"].astype(str)))]
    if df_immune.empty:
        raise ActiveKillingInputError(
            f"None of the samples in the {immune_cell_type} track features appear in the metadata, "
            f"so there are no masks to read death events from."
        )

    print(f"{get_current_time()} - Death events (cached per target type)...")
    df_events, df_ts, patches, sample_info = detect_death_events(
        md, output_dir, target_cell_types, dparams, reuse_cache=reuse_death_event_cache,
        progress_cb=progress_cb,
    )

    contacts = explode_contacts(df_immune, target_cell_types)
    df_contact_events, contacts = identify_contact_events_per_target(contacts)
    print(f"{get_current_time()} - {len(df_events)} death events, {len(df_contact_events)} contact events")
    if contacts.empty:
        raise ActiveKillingInputError(
            f"No {immune_cell_type} row touches any of {', '.join(target_cell_types)} "
            f"(every touching_* entry is empty). Check the Contact Threshold used in Feature "
            f"Extraction -- with no contacts nothing can be attributed, and reporting "
            f"'0 attributed, 100% background death' would look like a result."
        )

    df_cand, df_events_annot = attribute_death_events(
        df_events, contacts, patches, df_ts, immune_type=immune_cell_type,
        immune_tracks_paths=immune_paths, sample_info=sample_info, params=aparams,
    )
    df_eff = summarize_per_effector(df_cand, df_immune, df_contact_events, sample_info)
    df_tgt = summarize_per_target(df_events_annot, df_cand, df_ts, df_contact_events, sample_info)
    df_smp = summarize_per_sample(df_events_annot, df_cand, df_eff, df_tgt, df_contact_events, df_ts,
                                  sample_info, aparams)

    n_att = int(df_events_annot["attribution_class"].isin(["exclusive", "shared"]).sum()) if not df_events_annot.empty else 0
    n_counted = int((df_events_annot["attribution_class"] != "excluded_border").sum()) if not df_events_annot.empty else 0
    stats = {
        "schema_version": ACTIVE_KILLING_SCHEMA_VERSION,
        "n_death_events": n_counted,
        "n_attributed": n_att,
        "n_unattributed": int((df_events_annot.get("attribution_class", pd.Series(dtype=str)) == "unattributed").sum()),
        "n_contact_events": int(len(df_contact_events)),
        "total_kill_credit": float(df_cand["kill_credit"].sum()) if not df_cand.empty else 0.0,
        # Kept as the key both panels print; now the headline conversion rate.
        "overall_killing_rate": (n_att / len(df_contact_events)) if len(df_contact_events) else 0.0,
        "conversion_rate": (n_att / len(df_contact_events)) if len(df_contact_events) else float("nan"),
        "target_cell_diameter_um": float(dparams.target_cell_diameter_um),
        "causal_window_min": float(aparams.causal_window_min),
        "attribution_radius_um": float(aparams.attribution_radius_um),
        "filtering_needs_rerun_for": sorted(stale_filtered),
    }
    print(f"{get_current_time()} - Active Killing: {n_att}/{n_counted} death events attributed, "
          f"conversion rate {stats['conversion_rate']:.3f}, total credit {stats['total_kill_credit']:.2f}")

    if save_results:
        results_dir = output_dir / "analysis" / immune_cell_type / "active_killing"
        if output_subfolder:
            results_dir = results_dir / output_subfolder
        results_dir.mkdir(parents=True, exist_ok=True)
        df_cand.to_csv(results_dir / f"kill_candidates_{immune_cell_type}.csv", index=False)
        df_events_annot.to_csv(results_dir / f"death_events_{immune_cell_type}.csv", index=False)
        df_eff.to_csv(results_dir / f"per_effector_killing_{immune_cell_type}.csv", index=False)
        df_tgt.to_csv(results_dir / f"per_target_death_{immune_cell_type}.csv", index=False)
        df_smp.to_csv(results_dir / f"per_sample_killing_{immune_cell_type}.csv", index=False)
        df_contact_events.to_csv(results_dir / f"contact_events_{immune_cell_type}.csv", index=False)
        df_adv = create_advanced_features_csv(df_immune, df_cand, contacts)
        df_adv.to_csv(results_dir / f"BEHAV3D_{immune_cell_type}_advanced_track_features.csv", index=False)
        upstream = sorted({*immune_paths.values(), *[
            str(p) for _, r in md.iterrows() for p in [resolve_tracks_image_path(r, t) for t in target_cell_types]
            if p is not None and p.exists()]})
        from dataclasses import asdict
        run_params = {
            "schema_version": ACTIVE_KILLING_SCHEMA_VERSION,
            "immune_cell_type": immune_cell_type,
            "target_cell_types": target_cell_types,
            "death_event_params": asdict(dparams),
            "attribution_params": asdict(aparams),
            "advanced_overrides": dict(advanced or {}),
            "upstream_paths": upstream,
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (results_dir / RUN_PARAMS_FILENAME).write_text(json.dumps(run_params, indent=2, default=str), encoding="utf-8")
        print(f"{get_current_time()} - Results saved to {results_dir}")
        if write_figures:
            lc_col = f"im_{immune_cell_type}_line_condition"
            cond = (dict(zip(md["sample_name"].astype(str), md[lc_col].astype(str)))
                    if lc_col in md.columns else None)
            try:
                from behav3d.analysis.killing_figures import write_active_killing_figures
                figs = write_active_killing_figures(results_dir, immune_cell_type, top_n=top_n_killers,
                                                    condition_map=cond)
                stats["figures"] = {k: str(v) for k, v in figs.items()}
                print(f"{get_current_time()} - {len(figs)} figures saved to {results_dir / 'plots'}")
            except Exception as exc:
                # The tables above are complete and valid; a plotting failure
                # must not discard them, but it must not pass silently either.
                import traceback
                traceback.print_exc()
                stats["figures_error"] = f"{type(exc).__name__}: {exc}"
                print(f"{get_current_time()} - ERROR: Active Killing figures failed ({exc}); "
                      f"the result tables in {results_dir} are unaffected.")
        _print_rerun_filtering_warning(stale_filtered)

    h, m, s = format_time(start_time, time.time())
    print(f"### DONE - elapsed time: {h}:{m:02}:{s:02}\n")
    return df_cand, df_smp, stats


def create_advanced_features_csv(
    df_immune_tracks: pd.DataFrame,
    df_candidates: pd.DataFrame,
    df_contacts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge the kill columns onto the effector per-timepoint feature table.

    Credit sits on each effector's last contact frame before a death event's
    onset (a row that always exists), so the per-timepoint ``kill_credit``
    sums to the same total as the candidate table. Every raw column is passed
    through unchanged, and no column is left NA (defaults in
    ``KILL_COLUMN_DEFAULTS``).
    """
    from behav3d.features.kill_attribution import per_timepoint_credit

    df = df_immune_tracks.copy()
    for col, default in KILL_COLUMN_DEFAULTS.items():
        df[col] = default
    key = ["sample_name", "TrackID", "position_t"]
    df["_order"] = np.arange(len(df))

    if df_contacts is not None and not df_contacts.empty:
        ce = (df_contacts.sort_values("contact_event_id")
              .drop_duplicates(["sample_name", "immune_track_id", "position_t"])
              .rename(columns={"immune_track_id": "TrackID"})[key + ["contact_event_id"]])
        df = df.drop(columns="contact_event_id").merge(ce, on=key, how="left")
        df["contact_event_id"] = df["contact_event_id"].fillna(-1).astype(np.int64)

    pt = per_timepoint_credit(df_candidates)
    if not pt.empty:
        cols = [c for c in pt.columns if c not in key]
        df = df.drop(columns=cols).merge(pt, on=key, how="left")
        for col in cols:
            df[col] = df[col].fillna(KILL_COLUMN_DEFAULTS.get(col, 0))
        # The credited contact event, where known, takes precedence.
        if df_contacts is not None and not df_candidates.empty:
            cred = df_candidates.sort_values("kill_credit", ascending=False).drop_duplicates(
                ["sample_name", "immune_track_id", "last_contact_t"]).rename(
                columns={"immune_track_id": "TrackID", "last_contact_t": "position_t"})[key + ["contact_event_id"]]
            df = df.merge(cred.rename(columns={"contact_event_id": "_ce"}), on=key, how="left")
            df["contact_event_id"] = np.where(df["_ce"].notna(), df["_ce"], df["contact_event_id"]).astype(np.int64)
            df = df.drop(columns="_ce")

    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df["is_active_killing"] = df["kill_credit"].astype(float) > 0
    df["is_nearest_effector"] = df["is_nearest_effector"].astype(bool)
    for col in ("death_event_id", "targeted_track_id", "n_cokillers", "n_events_credited", "contact_event_id"):
        df[col] = df[col].astype(np.int64)
    df["targeted_cell_type"] = df["targeted_cell_type"].astype(str)
    df["attribution_class"] = df["attribution_class"].astype(str)
    df["cum_kill_credit"] = (df.sort_values(key).groupby(["sample_name", "TrackID"])["kill_credit"]
                             .cumsum().reindex(df.index))
    return df


def calculate_invasiveness_single_timepoint(args):
    """
    Calculate organoid invasiveness features for immune cells at a single timepoint.
    
    Invasiveness measures what percentage of an immune cell's surface is in contact
    with organoid surfaces. A cell is considered invasive if >50% of its surface
    is in contact.
    
    Parameters
    ----------
    args : tuple
        Contains:
        - t: timepoint index
        - current_cell_segments_path: path to immune cell segments
        - organoid_segments_paths: dict of organoid segment paths
        - element_size_x, element_size_y, element_size_z: voxel spacing (um)
        - contact_threshold: distance threshold (um) for determining contact
        - calculate_from: cell type being analyzed (should be immune type)
    
    Returns
    -------
    pd.DataFrame
        Columns: TrackID, position_t, <type>_invasiveness, <type>_invasiveness_perc, any_org_invasiveness_perc
    """
    from scipy.ndimage import distance_transform_edt
    import math
    
    (
        t,
        current_cell_segments_path,
        organoid_segments_paths,
        element_size_x,
        element_size_y,
        element_size_z,
        contact_threshold,
        calculate_from
    ) = args

    # Load current cell type's segments for this timepoint
    current_segments = np.asarray(load_image(current_cell_segments_path)[t])
    
    # Load all organoid types' segments
    organoid_segments_dict = {}
    for org_type, org_path in organoid_segments_paths.items():
        organoid_segments_dict[org_type] = np.asarray(load_image(org_path)[t])
    
    df_invasiveness = []
    segment_ids = np.unique(current_segments)
    
    # If there are no foreground segments at this timepoint, return an empty DataFrame
    foreground_ids = [sid for sid in segment_ids if sid != 0]
    if not foreground_ids:
        return pd.DataFrame(columns=["TrackID", "position_t"])
    
    for segment_id in foreground_ids:
        stack_max_z, stack_max_y, stack_max_x = current_segments.shape
        seg_locs = np.argwhere(current_segments == segment_id)
        min_z, min_y, min_x = seg_locs.min(axis=0)
        max_z, max_y, max_x = seg_locs.max(axis=0)
        
        z_ext = 2 * math.ceil(contact_threshold / element_size_z)
        y_ext = 2 * math.ceil(contact_threshold / element_size_y)
        x_ext = 2 * math.ceil(contact_threshold / element_size_x)
        
        slicer = (
            slice(max(0, min_z - z_ext), min(stack_max_z, max_z + z_ext + 1)),
            slice(max(0, min_y - y_ext), min(stack_max_y, max_y + y_ext + 1)),
            slice(max(0, min_x - x_ext), min(stack_max_x, max_x + x_ext + 1))
        )
        
        seg_cutout = current_segments[slicer]
        
        # Calculate distance transform from current cell boundary
        real_distances = distance_transform_edt(
            seg_cutout != segment_id,
            sampling=[element_size_z, element_size_y, element_size_x]
        )
        
        # Define "surface" as pixels within 2 um of cell boundary
        surface_threshold = 2.0  # um
        surface_mask = real_distances <= surface_threshold
        total_surface_pixels = np.sum(surface_mask)
        
        if total_surface_pixels == 0:
            # Cell has no surface (single pixel?), skip
            continue
        
        invasiveness_data = {
            'TrackID': segment_id,
            'position_t': t,
        }
        
        any_invasiveness_list = []
        invasiveness_perc_list = []
        
        # Calculate invasiveness for each organoid type
        for org_type, org_segments in organoid_segments_dict.items():
            org_cutout = org_segments[slicer]
            
            # Count surface pixels in contact with this organoid type
            org_contact_mask = (real_distances <= contact_threshold) & (org_cutout != 0)
            contacted_surface_pixels = np.sum(org_contact_mask)
            
            # Calculate percentage
            invasiveness_perc = (contacted_surface_pixels / total_surface_pixels) * 100.0
            
            # Boolean: invasive if >= 50% of surface is in contact
            invasiveness_bool = invasiveness_perc >= 50.0
            
            invasiveness_data[f'{org_type}_invasiveness'] = invasiveness_bool
            invasiveness_data[f'{org_type}_invasiveness_perc'] = invasiveness_perc
            
            any_invasiveness_list.append(invasiveness_bool)
            invasiveness_perc_list.append(invasiveness_perc)
        
        # Add aggregate: True if invasive against ANY organoid type
        invasiveness_data['any_org_invasiveness'] = any(any_invasiveness_list)
        invasiveness_data['any_org_invasiveness_perc'] = max(invasiveness_perc_list) if invasiveness_perc_list else 0.0
        
        df_invasiveness.append(pd.DataFrame([invasiveness_data]))
    
    if df_invasiveness:
        return pd.concat(df_invasiveness, ignore_index=True)
    else:
        return pd.DataFrame(columns=["TrackID", "position_t"])


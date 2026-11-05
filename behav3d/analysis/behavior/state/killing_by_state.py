"""Killing efficiency per behavioral state: the data behind the Killing by State report.

Joins per-timepoint state labels with Active Killing's per-timepoint `killing_efficiency`,
flags which contact timepoints are solo (no other effector cell on the same targets during
the observation window) and which have a full window before the end of the video, and
aggregates per cell x state, image x state and state. Purely descriptive: nothing here fits a
model or runs a test. `killing_timepoints.csv` keeps every contact timepoint with its flags,
so it can be fed straight into a mixed model downstream.
"""
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from behav3d.analysis.behavior.track.contact_grouping import _normalize_id_column

SAMPLE_COL = "sample_name"
TRACK_COL = "TrackID"
TIME_COL = "position_t"
KEY_COLS = [SAMPLE_COL, TRACK_COL, TIME_COL]

_WINDOW_COLUMN_RE = re.compile(r"^death_signal_increase_(\d+)tp$")


def parse_extracted_window(columns) -> Optional[int]:
    """Observation window encoded in a `death_signal_increase_<N>tp` column name, or None."""
    for col in columns:
        match = _WINDOW_COLUMN_RE.match(str(col))
        if match:
            return int(match.group(1))
    return None


def _read_extracted_threshold(output_dir, cell_type):
    """(mode, threshold) of the latest Active Killing run, read from its per-timepoint table.

    A single threshold across all timepoints means an absolute run. Varying values mean a
    multiplier run, whose multiplier can't be read back from the data, so its threshold is
    None. (None, None) when no such table exists.
    """
    if not output_dir or not cell_type:
        return None, None
    root = Path(output_dir) / "analysis" / cell_type / "active_killing"
    filename = f"active_killing_per_timepoint_{cell_type}.csv"
    candidates = list(root.glob(f"*/{filename}")) + [root / filename]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        values = pd.read_csv(latest, usecols=["killing_threshold_used"])["killing_threshold_used"].dropna()
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return None, None
    if values.empty:
        return None, None
    if values.nunique() == 1:
        return "absolute", float(values.iloc[0])
    return "multiplier", None


def _format_threshold(mode, value):
    if value is None:
        return "multiplier" if mode == "multiplier" else "unknown"
    return f"{value:g}x" if mode == "multiplier" else f"{value:g}"


@dataclass(frozen=True)
class KillingSettings:
    """The window and threshold a report uses, next to what the extracted features used."""

    window: int
    threshold_mode: str  # "absolute" or "multiplier"
    threshold: float
    target_types: tuple = ()
    contact_column: str = "contact"
    death_signal_column: str = "mean_dead_dye"
    persistent_death_signal: bool = True
    min_contact_duration: int = 1
    extracted_window: Optional[int] = None
    extracted_threshold_mode: Optional[str] = None
    extracted_threshold: Optional[float] = None
    differences: tuple = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "differences", tuple(self._find_differences()))

    def _find_differences(self):
        diffs = []
        if self.extracted_window is not None and int(self.window) != int(self.extracted_window):
            diffs.append(
                f"Window {int(self.window)} tp differs from extracted features "
                f"({int(self.extracted_window)} tp)."
            )
        used = _format_threshold(self.threshold_mode, self.threshold)
        if self.extracted_threshold_mode is not None and self.extracted_threshold_mode != self.threshold_mode:
            extracted = _format_threshold(self.extracted_threshold_mode, self.extracted_threshold)
            diffs.append(
                f"Threshold {used} ({self.threshold_mode}) differs from extracted features "
                f"({extracted}, {self.extracted_threshold_mode})."
            )
        elif self.extracted_threshold is not None and not math.isclose(
            float(self.threshold), float(self.extracted_threshold), rel_tol=1e-9, abs_tol=1e-9
        ):
            extracted = _format_threshold(self.threshold_mode, self.extracted_threshold)
            diffs.append(f"Threshold {used} differs from extracted features ({extracted}).")
        return diffs

    @property
    def needs_recompute(self) -> bool:
        return bool(self.differences)


def resolve_killing_settings(
    active_killing_config,
    output_dir,
    cell_type,
    columns,
    *,
    window_override=None,
    threshold_override=None,
) -> KillingSettings:
    """Settings for a report: the Active Killing tab's values (`active_killing_config`, the
    `active_killing` section of behav3d_parameters.yml) unless overridden.

    The values the extracted features were really made with are read from the data (`columns`
    of the features table, and the Active Killing per-timepoint table), falling back to the tab
    only where the data can't tell, so a tab edited without re-running Active Killing also
    shows up as a difference.
    """
    cfg = dict(active_killing_config or {})
    extracted_window = parse_extracted_window(columns)
    extracted_mode, extracted_threshold = _read_extracted_threshold(output_dir, cell_type)

    window = cfg.get("observation_window")
    if window is None:
        window = extracted_window if extracted_window is not None else 5

    absolute = cfg.get("absolute_killing_threshold")
    if cfg.get("use_absolute_threshold") or (absolute is not None and "use_absolute_threshold" not in cfg):
        threshold_mode = "absolute"
        threshold = float(absolute) if absolute is not None else 0.0
    elif not cfg and extracted_mode == "absolute":
        threshold_mode = "absolute"
        threshold = extracted_threshold
    else:
        threshold_mode = "multiplier"
        threshold = float(cfg.get("killing_threshold_multiplier", 1.5))

    # A multiplier run can't be read back from the data: the tab's value is the best baseline.
    if extracted_mode == "multiplier" and threshold_mode == "multiplier":
        extracted_threshold = threshold

    return KillingSettings(
        window=int(window_override) if window_override is not None else int(window),
        threshold_mode=threshold_mode,
        threshold=float(threshold_override) if threshold_override is not None else threshold,
        target_types=tuple(str(t) for t in (cfg.get("target_types") or [])),
        contact_column=str(cfg.get("contact_column", "contact")),
        death_signal_column=str(cfg.get("death_signal_column", "mean_dead_dye")),
        persistent_death_signal=bool(cfg.get("persistent_death_signal", True)),
        min_contact_duration=int(cfg.get("min_contact_duration", 1)),
        extracted_window=extracted_window,
        extracted_threshold_mode=extracted_mode,
        extracted_threshold=extracted_threshold,
    )


def resolve_killing_features_csv(output_dir, cell_type, preferred_csv) -> Path:
    """The per-timepoint table to read killing efficiency from.

    `preferred_csv` (normally the filtered table) when it holds the killing columns, otherwise
    Active Killing's own table. Raises ValueError when neither exists, and StaleDataError when
    Active Killing's table predates Feature Extraction.
    """
    if preferred_csv is not None and Path(preferred_csv).exists():
        if "killing_efficiency" in pd.read_csv(preferred_csv, nrows=0).columns:
            return Path(preferred_csv)
    from behav3d.features.advanced_timepoint_features import find_advanced_features_csv

    advanced = find_advanced_features_csv(output_dir, cell_type)
    if advanced is None:
        raise ValueError(
            f"No killing efficiency found for '{cell_type}'. Run Active Killing first, then Filtering."
        )
    return Path(advanced)


def read_killing_table(path) -> pd.DataFrame:
    """The columns of a per-timepoint features table this report uses; these tables can be huge."""
    def _wanted(col):
        col = str(col)
        return (
            col in KEY_COLS
            or col in ("killing_efficiency", "is_active_killing")
            or col.startswith("touching_")
            or col.startswith("death_signal_increase_")
            or col.endswith("_contact")
            or col.endswith("_contact_on_distance")
        )
    return pd.read_csv(path, usecols=_wanted, low_memory=False)


def read_state_labels(h5ad_path, state_col) -> list:
    """The labels of one state column of a behavioral-states h5ad, without loading the file.
    Empty when the column isn't there."""
    import h5py
    from behav3d.core.h5_access import h5_read

    def _text(values):
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]

    with h5_read(), h5py.File(str(h5ad_path), "r") as f:
        obs = f.get("obs")
        if obs is None or state_col not in obs:
            return []
        node = obs[state_col]
        if isinstance(node, h5py.Group):
            return _text(node["categories"][:])
        return sorted(set(_text(node[:])))


def load_video_ends(output_dir, target_types):
    """Last timepoint per sample across the target tracks of `target_types`."""
    frames = []
    for target in target_types:
        path = _track_features_csv(output_dir, target)
        if path.exists():
            frames.append(pd.read_csv(path, usecols=[SAMPLE_COL, TIME_COL]))
    if not frames:
        raise FileNotFoundError(
            f"No track features found for the Active Killing targets {list(target_types)}."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined[SAMPLE_COL] = combined[SAMPLE_COL].astype(str)
    return combined.groupby(SAMPLE_COL)[TIME_COL].max().to_dict()


def _track_features_csv(output_dir, cell_type):
    return (
        Path(output_dir) / "analysis" / cell_type / "track_features"
        / f"BEHAV3D_{cell_type}_combined_track_features.csv"
    )


def recompute_killing_efficiency(output_dir, cell_type, settings: KillingSettings) -> pd.DataFrame:
    """Per-timepoint killing values at `settings`, using Active Killing's own calculation on
    the same raw tracks. Writes nothing: the Active Killing outputs and features table stay as
    they are. Returns KEY_COLS + killing_efficiency, death_signal_increase, is_active_killing.
    """
    from behav3d.features.advanced_timepoint_features import (
        analyze_active_killing_per_timepoint,
        identify_contact_events_global,
    )

    immune_path = _track_features_csv(output_dir, cell_type)
    if not immune_path.exists():
        raise FileNotFoundError(f"Track features not found: {immune_path}")
    wanted = set(KEY_COLS)
    for target in settings.target_types:
        wanted |= {f"{target}_{settings.contact_column}", f"touching_{target}s"}
    immune = pd.read_csv(immune_path, usecols=lambda c: c in wanted, low_memory=False)

    target_frames = []
    for target in settings.target_types:
        path = _track_features_csv(output_dir, target)
        if path.exists():
            target_frames.append(pd.read_csv(path, usecols=KEY_COLS + [settings.death_signal_column]))
    if not target_frames:
        raise FileNotFoundError(
            f"No track features found for the Active Killing targets {list(settings.target_types)}."
        )
    targets = pd.concat(target_frames, ignore_index=True)

    events = identify_contact_events_global(
        immune,
        list(settings.target_types),
        min_contact_duration=settings.min_contact_duration,
        contact_column=settings.contact_column,
    )
    absolute = settings.threshold_mode == "absolute"
    per_timepoint = analyze_active_killing_per_timepoint(
        immune,
        targets,
        events,
        observation_window=settings.window,
        death_signal_column=settings.death_signal_column,
        killing_threshold_multiplier=settings.threshold if not absolute else 1.5,
        absolute_killing_threshold=settings.threshold if absolute else None,
        persistent_death_signal=settings.persistent_death_signal,
    )
    columns = ["killing_efficiency", "death_signal_increase", "is_active_killing"]
    if per_timepoint.empty:
        return pd.DataFrame(columns=KEY_COLS + columns)
    out = per_timepoint.rename(columns={"immune_track_id": TRACK_COL})[KEY_COLS + columns]
    # A contact event can only overlap itself, but guard the later merge against duplicates.
    return out.drop_duplicates(subset=KEY_COLS)


def _normalize_keys(df):
    out = df.copy()
    out[SAMPLE_COL] = _normalize_id_column(out[SAMPLE_COL])
    out[TRACK_COL] = _normalize_id_column(out[TRACK_COL])
    out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce").round()
    return out.dropna(subset=[TIME_COL])


def _build_target_occupancy(df_table, target_types):
    """Long table of (sample, cell, time, target key) from the `touching_<target>s` columns."""
    frames = []
    for target in target_types:
        col = f"touching_{target}s"
        if col not in df_table.columns:
            continue
        sub = df_table.loc[df_table[col].notna(), KEY_COLS + [col]]
        if sub.empty:
            continue
        exploded = sub.assign(_id=sub[col].astype(str).str.split(",")).explode("_id", ignore_index=True)
        numeric = pd.to_numeric(exploded["_id"].str.strip(), errors="coerce")
        keep = numeric.notna()
        frames.append(
            exploded.loc[keep, KEY_COLS].assign(_target_type=target, _target_id=numeric[keep].round().astype(int))
        )
    if not frames:
        return pd.DataFrame(columns=KEY_COLS + ["_target_type", "_target_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def count_other_cells_on_targets(rows, df_table, target_types, window) -> np.ndarray:
    """For each row, the number of other cells touching any of its targets during [t, t+window].

    NaN where the cell has no recorded target at that timepoint. Occupancy comes from
    `df_table`, so cells absent from that table (e.g. removed by Filtering) don't count.
    `rows` and `df_table` must have normalized keys.
    """
    occupancy = _build_target_occupancy(df_table, target_types)
    result = np.full(len(rows), np.nan)
    if occupancy.empty or len(rows) == 0:
        return result

    timelines = {}
    for key, group in occupancy.groupby([SAMPLE_COL, "_target_type", "_target_id"], sort=False):
        group = group.sort_values(TIME_COL)
        timelines[key] = (group[TIME_COL].to_numpy(), group[TRACK_COL].to_numpy())
    targets_at = {
        key: list(zip(group["_target_type"], group["_target_id"]))
        for key, group in occupancy.groupby(KEY_COLS, sort=False)
    }

    samples = rows[SAMPLE_COL].to_numpy()
    tracks = rows[TRACK_COL].to_numpy()
    times = rows[TIME_COL].to_numpy()
    for i in range(len(rows)):
        touched = targets_at.get((samples[i], tracks[i], times[i]))
        if touched is None:
            continue
        others = set()
        for target_type, target_id in touched:
            target_times, target_tracks = timelines[(samples[i], target_type, target_id)]
            start = np.searchsorted(target_times, times[i], side="left")
            end = np.searchsorted(target_times, times[i] + window, side="right")
            others.update(target_tracks[start:end].tolist())
        others.discard(tracks[i])
        result[i] = len(others)
    return result


def _touching_targets(df, target_types):
    """Comma-joined targets per row; prefixed with the target type when there are several."""
    present = [t for t in target_types if f"touching_{t}s" in df.columns]
    if len(present) == 1:
        return df[f"touching_{present[0]}s"].astype(object).where(lambda s: s.notna(), "").astype(str)
    parts = []
    for target in present:
        values = df[f"touching_{target}s"].astype(object).where(lambda s: s.notna(), "").astype(str)
        parts.append(
            values.map(
                lambda text, t=target: ",".join(
                    f"{t}:{v.strip()}" for v in text.split(",") if v.strip()
                )
            )
        )
    if not parts:
        return pd.Series("", index=df.index)
    return pd.concat(parts, axis=1).apply(lambda r: ",".join(v for v in r if v), axis=1)


def _contact_mask(df, settings):
    columns = [f"{t}_{settings.contact_column}" for t in settings.target_types]
    present = [c for c in columns if c in df.columns]
    if not present:
        raise KeyError(
            f"None of the contact columns {columns} are in the features table. "
            "Check the targets on the Active Killing tab."
        )
    as_numeric = df[present].apply(pd.to_numeric, errors="coerce").fillna(0)
    return as_numeric.astype(bool).any(axis=1)


def build_killing_timepoints(
    adata_states,
    df_table,
    *,
    state_col,
    settings: KillingSettings,
    video_ends,
    states=None,
    solo_only=True,
    full_windows_only=True,
    group_cols=None,
    recomputed=None,
):
    """One row per contact timepoint of the included states, with filter flags.

    `states=None` includes every state with at least one contact timepoint. `recomputed`, when
    given, replaces the stored killing columns (see `recompute_killing_efficiency`); contact
    timepoints absent from it have no qualifying contact event, so they get 0 like Active
    Killing's own table.

    Returns (timepoints, state_order_observed) where `used` marks the rows the summaries use.
    """
    if not settings.target_types:
        raise ValueError(
            "No Active Killing targets are set. Choose them on the Active Killing tab and run it."
        )
    group_cols = [str(c) for c in (group_cols or [])]
    obs = adata_states.obs
    missing = [c for c in KEY_COLS + [state_col] + group_cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in the behavioral states table: {missing}")
    missing = [c for c in KEY_COLS if c not in df_table.columns]
    if missing:
        raise KeyError(f"Missing columns in the features table: {missing}")
    if recomputed is None and "killing_efficiency" not in df_table.columns:
        raise ValueError("The features table has no killing_efficiency. Run Active Killing, then Filtering.")

    state_rows = obs[KEY_COLS + [state_col] + group_cols].copy()
    state_rows[state_col] = state_rows[state_col].astype(str)
    for col in group_cols:
        state_rows[col] = state_rows[col].astype(object).where(state_rows[col].notna(), "NA").astype(str)
    state_rows = _normalize_keys(state_rows).drop_duplicates(subset=KEY_COLS)

    table = _normalize_keys(df_table).drop_duplicates(subset=KEY_COLS)
    table = table.assign(
        _contact=_contact_mask(table, settings),
        touching_targets=_touching_targets(table, settings.target_types),
    )
    window_col = f"death_signal_increase_{settings.extracted_window}tp"
    stored = ["killing_efficiency", "is_active_killing"]
    if window_col in table.columns:
        table = table.rename(columns={window_col: "death_signal_increase"})
        stored.append("death_signal_increase")
    keep = KEY_COLS + ["_contact", "touching_targets"] + [c for c in stored if c in table.columns]

    merged = state_rows.merge(table[keep], on=KEY_COLS, how="inner")
    if merged.empty and not state_rows.empty:
        raise ValueError(
            "No timepoint of the behavioral states matches the features table: "
            "the classification and features no longer belong together."
        )
    merged = merged[merged["_contact"]].drop(columns="_contact")
    merged = merged.rename(columns={state_col: "state"})

    observed_states = pd.unique(merged["state"]).tolist()
    if states:
        wanted = {str(s) for s in states}
        merged = merged[merged["state"].isin(wanted)]
    merged = merged.reset_index(drop=True)

    if recomputed is not None:
        merged = merged.drop(columns=[c for c in stored if c in merged.columns])
        fresh = _normalize_keys(recomputed)
        merged = merged.merge(fresh, on=KEY_COLS, how="left")
        merged["killing_efficiency"] = merged["killing_efficiency"].fillna(0.0)
        merged["death_signal_increase"] = merged["death_signal_increase"].fillna(0.0)
        merged["is_active_killing"] = merged["is_active_killing"].eq(True)

    if "death_signal_increase" not in merged.columns:
        merged["death_signal_increase"] = np.nan
    if "is_active_killing" not in merged.columns:
        merged["is_active_killing"] = pd.to_numeric(merged["killing_efficiency"], errors="coerce") > 1
    merged["is_active_killing"] = merged["is_active_killing"].astype(bool)

    merged["n_other_cells"] = count_other_cells_on_targets(
        merged, table, settings.target_types, settings.window
    )
    ends = {str(k): v for k, v in (video_ends or {}).items()}
    table_ends = table.groupby(SAMPLE_COL)[TIME_COL].max().to_dict()
    merged["video_end"] = merged[SAMPLE_COL].map(
        lambda s: max(ends.get(s, table_ends.get(s, np.nan)), table_ends.get(s, np.nan))
    )
    merged["solo"] = merged["n_other_cells"] == 0
    merged["full_window"] = merged[TIME_COL] <= merged["video_end"] - settings.window
    used = pd.Series(True, index=merged.index)
    if solo_only:
        used &= merged["solo"]
    if full_windows_only:
        used &= merged["full_window"]
    merged["used"] = used
    merged["cell_id"] = merged[SAMPLE_COL] + "_" + merged[TRACK_COL]
    merged["cell_state_id"] = merged["cell_id"] + "_" + merged["state"]

    columns = KEY_COLS + [
        "state", "killing_efficiency", "death_signal_increase", "is_active_killing",
        "touching_targets", "n_other_cells", "video_end", "solo", "full_window", "used",
        "cell_id", "cell_state_id",
    ] + group_cols
    merged = merged[columns]
    merged[TIME_COL] = merged[TIME_COL].astype(int)
    return merged, observed_states


def filter_counts(timepoints) -> dict:
    """How many contact timepoints each filter step leaves."""
    return {
        "contact_timepoints": int(len(timepoints)),
        "without_target": int(timepoints["n_other_cells"].isna().sum()),
        "solo": int(timepoints["solo"].sum()),
        "full_window": int(timepoints["full_window"].sum()),
        "used": int(timepoints["used"].sum()),
        "cells": int(timepoints.loc[timepoints["used"], "cell_id"].nunique()),
        "cell_state_units": int(timepoints.loc[timepoints["used"], "cell_state_id"].nunique()),
        "images": int(timepoints.loc[timepoints["used"], SAMPLE_COL].nunique()),
    }


def summarize_killing_by_state(timepoints, state_order, group_cols=None):
    """Descriptive tables from the `used` rows: (cell_state, image_state, state_summary).

    The two means answer different questions. `cell_mean` averages timepoints per cell first, so
    each cell counts once; `timepoint_mean` averages timepoints directly, so cells that stay
    longer in a state weigh more. Every state in `state_order` gets a summary row, with n = 0
    when no timepoint was used.
    """
    group_cols = [str(c) for c in (group_cols or [])]
    used = timepoints[timepoints["used"]]

    cell_state = (
        used.groupby(group_cols + [SAMPLE_COL, TRACK_COL, "cell_id", "state", "cell_state_id"], observed=True, sort=False)
        .agg(
            n_timepoints=("killing_efficiency", "size"),
            mean_killing_efficiency=("killing_efficiency", "mean"),
            frac_active_timepoints=("is_active_killing", "mean"),
        )
        .reset_index()
    )
    image_state = (
        cell_state.groupby(group_cols + [SAMPLE_COL, "state"], observed=True, sort=False)
        .agg(
            n_cells=("cell_id", "nunique"),
            n_timepoints=("n_timepoints", "sum"),
            cell_mean=("mean_killing_efficiency", "mean"),
        )
        .reset_index()
    )
    tp_means = (
        used.groupby(group_cols + [SAMPLE_COL, "state"], observed=True, sort=False)["killing_efficiency"]
        .mean().rename("timepoint_mean").reset_index()
    )
    image_state = image_state.merge(tp_means, on=group_cols + [SAMPLE_COL, "state"], how="left")

    combos = (
        timepoints[group_cols].drop_duplicates().sort_values(group_cols).to_dict("records")
        if group_cols else [{}]
    )
    rows = []
    for combo in combos:
        def _select(df):
            mask = pd.Series(True, index=df.index)
            for col, val in combo.items():
                mask &= df[col] == val
            return df[mask]

        group_used, group_cells, group_images = _select(used), _select(cell_state), _select(image_state)
        for state in state_order:
            tp = group_used.loc[group_used["state"] == state, "killing_efficiency"]
            active = group_used.loc[group_used["state"] == state, "is_active_killing"]
            cells = group_cells.loc[group_cells["state"] == state, "mean_killing_efficiency"]
            images = group_images.loc[group_images["state"] == state, "cell_mean"]
            rows.append({
                **combo,
                "state": state,
                "n_timepoints": int(len(tp)),
                "n_cells": int(len(cells)),
                "n_images": int(len(images)),
                "timepoint_mean": tp.mean(),
                "timepoint_sem": tp.sem(),
                "cell_mean": cells.mean(),
                "cell_median": cells.median(),
                "cell_sd": cells.std(),
                "cell_sem": cells.sem(),
                "image_mean": images.mean(),
                "image_sem": images.sem(),
                "frac_active_timepoints": active.mean() if len(active) else np.nan,
            })
    return cell_state, image_state, pd.DataFrame(rows)

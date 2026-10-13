"""
Per-item run planning for pipeline steps (Qt-free).

A *run plan* says, for every ``(sample, cell_type)`` item of a step, whether to
skip it, run whatever is missing, or overwrite it.  The napari dialogs build a
plan from the user's choice and hand it to the backends; backends that are still
called with a plain ``overwrite: bool`` keep working through
:func:`resolve_action`.

Status probes only stat files and read zarr metadata (never array data), so they
are safe to call on the Qt thread even for very large datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Plan key used for the joint "track all organoids together" run: one item per
# sample covers the combined output and every per-type output.
ALL_ORGANOIDS = "all_organoids"

# Plan key for engines that process every cell type of a sample together.
ALL_CELL_TYPES = "all cell types"


class Status(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class Action(str, Enum):
    SKIP = "skip"            # leave existing outputs untouched
    RUN = "run"              # do whatever is missing, reuse anything valid
    OVERWRITE = "overwrite"  # discard existing outputs and redo


@dataclass(frozen=True)
class PlanItem:
    sample: str
    cell_type: str
    status: Status
    detail: str = ""
    # Display label of the queue step the item belongs to ("" outside the queue).
    step: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.sample, self.cell_type)


class RunPlan:
    """Immutable mapping ``(sample, cell_type) -> Action``.

    An item the plan does not know about gets :attr:`Action.RUN`, so a plan can
    never silently drop work it was not told about.
    """

    def __init__(self, items: Iterable[PlanItem], actions: Mapping[Tuple[str, str], Action]):
        self.items: Tuple[PlanItem, ...] = tuple(items)
        status = {it.key: it.status for it in self.items}
        resolved: Dict[Tuple[str, str], Action] = {}
        for it in self.items:
            act = Action(actions.get(it.key, Action.RUN))
            # Invariant: only COMPLETE items may be skipped, so after any run
            # everything is complete and later steps never see holes.
            if act is Action.SKIP and status[it.key] is not Status.COMPLETE:
                act = Action.RUN
            resolved[it.key] = act
        self._actions = resolved

    # -- constructors ------------------------------------------------------
    @classmethod
    def overwrite_all(cls, items: Iterable[PlanItem]) -> "RunPlan":
        items = list(items)
        return cls(items, {it.key: Action.OVERWRITE for it in items})

    @classmethod
    def skip_existing(cls, items: Iterable[PlanItem]) -> "RunPlan":
        items = list(items)
        return cls(
            items,
            {
                it.key: (Action.SKIP if it.status is Status.COMPLETE else Action.RUN)
                for it in items
            },
        )

    @classmethod
    def custom(cls, items: Iterable[PlanItem], choices: Mapping[Tuple[str, str], Action]) -> "RunPlan":
        return cls(list(items), choices)

    # -- queries -----------------------------------------------------------
    def action(self, sample: str, cell_type: str) -> Action:
        return self._actions.get((str(sample), str(cell_type)), Action.RUN)

    def for_cell_type(self, cell_type: str) -> "RunPlan":
        keep = [it for it in self.items if it.cell_type == cell_type]
        return RunPlan(keep, {it.key: self._actions[it.key] for it in keep})

    def any_work(self) -> bool:
        return any(a is not Action.SKIP for a in self._actions.values()) or not self._actions

    def counts(self) -> Dict[Action, int]:
        out = {a: 0 for a in Action}
        for a in self._actions.values():
            out[a] += 1
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        c = self.counts()
        return (
            f"RunPlan(skip={c[Action.SKIP]}, run={c[Action.RUN]}, "
            f"overwrite={c[Action.OVERWRITE]})"
        )


def resolve_action(
    plan: Optional[RunPlan], overwrite: bool, sample: str, cell_type: str
) -> Action:
    """Action for one item: the plan wins; without a plan fall back to ``overwrite``."""
    if plan is not None:
        return plan.action(sample, cell_type)
    return Action.OVERWRITE if overwrite else Action.RUN


# ─────────────────────────────────────────────────────────────────────────────
# Status probes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProbeResult:
    status: Status
    detail: str = ""
    # Tracked zarr is valid and only the csv is absent: it can be rebuilt from
    # the zarr without re-running the tracker.
    csv_only: bool = False


def zarr_n_timepoints(path) -> Optional[int]:
    """Length of axis 0 of a zarr, read from metadata only. ``None`` if unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        from behav3d.io.formats.zarr import load_zarr

        shape = load_zarr(path).shape
        return int(shape[0]) if len(shape) else None
    except Exception:
        return None


def tracking_paths(out_dir, sample: str, cell_type: str) -> Tuple[Path, Path]:
    """``(tracked_zarr, tracks_csv)`` for one sample and cell type."""
    out_dir = Path(out_dir)
    zarr_path = out_dir / "images" / sample / f"{sample}_{cell_type}_tracked.zarr"
    csv_path = out_dir / "trackdata" / sample / cell_type / f"{sample}_{cell_type}_tracks.csv"
    return zarr_path, csv_path


def _zarr_state(zarr_path: Path, expected_t: Optional[int]) -> Tuple[bool, str]:
    """``(valid, problem)`` for a tracked zarr; ``valid`` False if absent or short."""
    if not zarr_path.exists():
        return False, "tracked zarr missing"
    n_t = zarr_n_timepoints(zarr_path)
    if n_t is None or n_t == 0:
        return False, "tracked zarr unreadable"
    if expected_t is not None and n_t != expected_t:
        return False, f"tracked zarr has {n_t}/{expected_t} timepoints"
    return True, ""


def tracking_status(
    out_dir, sample: str, cell_type: str, segments_path=None
) -> ProbeResult:
    """File-based completeness of one tracking output (no journal / marker).

    COMPLETE = csv and zarr both exist (zarr length matches the source segments
    when those are readable).  PARTIAL = anything in between.  MISSING = neither.
    """
    zarr_path, csv_path = tracking_paths(out_dir, sample, cell_type)
    return tracking_status_for_paths(zarr_path, csv_path, segments_path)


def tracking_status_for_paths(zarr_path, csv_path, segments_path=None) -> ProbeResult:
    zarr_path, csv_path = Path(zarr_path), Path(csv_path)
    has_csv = csv_path.exists()
    has_zarr = zarr_path.exists()
    if not has_csv and not has_zarr:
        return ProbeResult(Status.MISSING, "not tracked")

    expected_t = zarr_n_timepoints(segments_path) if segments_path else None
    zarr_ok, problem = _zarr_state(zarr_path, expected_t)
    if has_csv and zarr_ok:
        return ProbeResult(Status.COMPLETE, "zarr + csv present")
    if zarr_ok and not has_csv:
        return ProbeResult(Status.PARTIAL, "csv missing (rebuilt from zarr)", csv_only=True)
    if has_csv and not has_zarr:
        return ProbeResult(Status.PARTIAL, "zarr missing (re-tracked)")
    return ProbeResult(Status.PARTIAL, f"{problem} (re-tracked)")


def all_organoids_paths(out_dir, sample: str, organoid_types: Sequence[str]) -> dict:
    """Output paths of the joint all-organoids run (mirrors ``_build_output_paths``)."""
    out_dir = Path(out_dir)
    image_dir = out_dir / "images" / sample
    csv_dir = out_dir / "trackdata" / sample / ALL_ORGANOIDS
    return {
        "combined_img": image_dir / f"{sample}_{ALL_ORGANOIDS}_tracked.zarr",
        "combined_csv": csv_dir / f"{sample}_{ALL_ORGANOIDS}_tracks.csv",
        "split": {
            ct: {
                "img": image_dir / f"{sample}_{ct}_tracked.zarr",
                "csv": out_dir / "trackdata" / sample / ct / f"{sample}_{ct}_tracks.csv",
            }
            for ct in organoid_types
        },
    }


def all_organoids_status(
    out_dir,
    sample: str,
    organoid_types: Sequence[str],
    segments_paths: Optional[Mapping[str, object]] = None,
) -> ProbeResult:
    """Completeness of the joint all-organoids output group for one sample.

    TrackIDs are shared across organoid types, so any gap other than missing
    per-type csvs (which are split out of the combined csv) forces a full redo.
    """
    paths = all_organoids_paths(out_dir, sample, organoid_types)
    required = [paths["combined_img"], paths["combined_csv"]]
    for p in paths["split"].values():
        required.extend([p["img"], p["csv"]])
    if not any(p.exists() for p in required):
        return ProbeResult(Status.MISSING, "not tracked")

    expected_t = None
    for ct in organoid_types:
        seg = (segments_paths or {}).get(ct)
        if seg:
            expected_t = zarr_n_timepoints(seg)
            if expected_t is not None:
                break

    zarrs = [paths["combined_img"]] + [p["img"] for p in paths["split"].values()]
    zarrs_ok = all(_zarr_state(z, expected_t)[0] for z in zarrs)
    combined_csv_ok = paths["combined_csv"].exists()
    missing_split_csv = [ct for ct, p in paths["split"].items() if not p["csv"].exists()]

    if zarrs_ok and combined_csv_ok and not missing_split_csv:
        return ProbeResult(Status.COMPLETE, "all organoid outputs present")
    if zarrs_ok and combined_csv_ok:
        return ProbeResult(
            Status.PARTIAL,
            f"csv missing for {', '.join(missing_split_csv)} (rebuilt from combined csv)",
            csv_only=True,
        )
    return ProbeResult(Status.PARTIAL, "incomplete group (re-tracked together)")


# ─────────────────────────────────────────────────────────────────────────────
# Scans: metadata -> PlanItems
# ─────────────────────────────────────────────────────────────────────────────
def _sample_name(sample) -> str:
    return str(sample.get("sample_name", "unknown"))


def segments_path_for(sample, cell_type: str) -> Optional[Path]:
    """Segments zarr of ``cell_type`` for one metadata row (``None`` if not set)."""
    import pandas as pd

    cols = [f"{prefix}_{cell_type}_segments_image_path" for prefix in ("or", "im", "ot")]
    cols.append(f"{cell_type}_segments_image_path")
    for col in cols:
        if col in sample.index:
            value = sample[col]
            if pd.notna(value) and str(value).strip():
                return Path(str(value))
    return None


def scan_tracking_items(metadata, out_dir, cell_types: Iterable[str]) -> list:
    """One :class:`PlanItem` per (sample, cell type) for the per-cell-type tracking output."""
    items = []
    for ct in cell_types:
        for _, sample in metadata.iterrows():
            sn = _sample_name(sample)
            seg = segments_path_for(sample, ct)
            probe = tracking_status(out_dir, sn, ct, seg)
            detail = probe.detail
            if probe.status is Status.MISSING and seg is None:
                detail = "no segmentation available"
            items.append(PlanItem(sn, ct, probe.status, detail))
    return items


def scan_all_organoids_items(metadata, out_dir, organoid_types: Sequence[str]) -> list:
    """One :class:`PlanItem` per sample for the joint all-organoids output group."""
    label = ", ".join(organoid_types)
    items = []
    for _, sample in metadata.iterrows():
        sn = _sample_name(sample)
        segs = {ct: segments_path_for(sample, ct) for ct in organoid_types}
        available = [ct for ct in organoid_types if segs[ct] is not None]
        if not available:
            items.append(PlanItem(sn, ALL_ORGANOIDS, Status.MISSING, "no segmentation available"))
            continue
        probe = all_organoids_status(out_dir, sn, available, segs)
        items.append(PlanItem(sn, ALL_ORGANOIDS, probe.status, f"{label}: {probe.detail}"))
    return items


def merge_plan_for_multicolor(
    plan: RunPlan, channels: Sequence[str], merged_name: str
) -> RunPlan:
    """Plan for the merge step of a multicolor group.

    The merged output is built from the channel outputs, so it must be rebuilt
    for any sample where a channel is (re)tracked, even if the merged output is
    otherwise complete and marked Skip.
    """
    items = [it for it in plan.items if it.cell_type == merged_name]
    choices = {}
    for it in items:
        act = plan.action(it.sample, merged_name)
        if act is Action.SKIP and any(
            plan.action(it.sample, ch) is not Action.SKIP for ch in channels
        ):
            act = Action.RUN
        choices[it.key] = act
    return RunPlan(items, choices)


# Pseudo-sample for steps whose output is one file per cell type.
ALL_SAMPLES = "all samples"


def feature_paths(out_dir, sample: str, cell_type: str) -> Tuple[Path, Path]:
    """``(per_sample_features_csv, combined_features_csv)`` for feature extraction."""
    out_dir = Path(out_dir)
    per_sample = out_dir / "trackdata" / sample / cell_type / f"{sample}_{cell_type}_track_features.csv"
    combined = (
        out_dir / "analysis" / cell_type / "track_features"
        / f"BEHAV3D_{cell_type}_combined_track_features.csv"
    )
    return per_sample, combined


def scan_feature_items(metadata, out_dir, cell_types: Iterable[str]) -> list:
    """One item per (sample, cell type) for feature extraction.

    COMPLETE = the sample's feature csv exists and the combined csv exists.
    A combined csv without the per-sample file (older runs, or a sample that
    produced no tracks) is PARTIAL, so existing results are never replaced
    without asking.
    """
    items = []
    for ct in cell_types:
        for _, sample in metadata.iterrows():
            sn = _sample_name(sample)
            per_sample, combined = feature_paths(out_dir, sn, ct)
            if per_sample.exists() and combined.exists():
                items.append(PlanItem(sn, ct, Status.COMPLETE, "features present"))
            elif per_sample.exists():
                items.append(PlanItem(sn, ct, Status.PARTIAL, "combined csv missing (rebuilt)"))
            elif combined.exists():
                items.append(PlanItem(sn, ct, Status.PARTIAL, "no per-sample features (recomputed)"))
            else:
                items.append(PlanItem(sn, ct, Status.MISSING, "no features yet"))
    return items


def scan_cell_type_files(cell_types: Iterable[str], path_fn) -> list:
    """One item per cell type for steps that write a single file per cell type."""
    items = []
    for ct in cell_types:
        path = Path(path_fn(ct))
        if path.exists():
            items.append(PlanItem(ALL_SAMPLES, ct, Status.COMPLETE, path.name))
        else:
            items.append(PlanItem(ALL_SAMPLES, ct, Status.MISSING, "not computed yet"))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation scans
# ─────────────────────────────────────────────────────────────────────────────
def segmentation_output_path(out_dir, sample: str, cell_type: str, kind: str = "segments") -> Path:
    """Path of a segmentation output.  The ``dead`` type is ``<sn>_mask_dead.zarr``."""
    base = Path(out_dir) / "images" / sample
    if cell_type == "dead":
        return base / f"{sample}_mask_dead.zarr"
    return base / f"{sample}_{cell_type}_{kind}.zarr"


def _journal_status(path) -> ProbeResult:
    """Status of one journalled array.  An array with no journal predates the
    bookkeeping and is treated as complete, exactly as ``plan_output`` does."""
    from behav3d.preprocessing.segmentation.segment_journal import describe_state, journal_state

    state, n_done, t_total = journal_state(path)
    if state == "missing":
        return ProbeResult(Status.MISSING, "not segmented")
    if state == "partial":
        return ProbeResult(Status.PARTIAL, describe_state(state, n_done, t_total))
    return ProbeResult(Status.COMPLETE, describe_state(state, n_done, t_total))


def segmentation_status(out_dir, sample: str, cell_type: str, include_mask: bool = False) -> ProbeResult:
    seg = _journal_status(segmentation_output_path(out_dir, sample, cell_type))
    if not include_mask or cell_type == "dead":
        return seg
    mask = _journal_status(segmentation_output_path(out_dir, sample, cell_type, "mask"))
    if seg.status is Status.COMPLETE and mask.status is not Status.COMPLETE:
        return ProbeResult(Status.PARTIAL, f"segments complete, mask {mask.detail}")
    if seg.status is Status.MISSING and mask.status is not Status.MISSING:
        return ProbeResult(Status.PARTIAL, f"mask present, segments {seg.detail}")
    return seg


def scan_segmentation_items(
    metadata, out_dir, cell_types: Iterable[str], *, include_mask: bool = False,
    include_dead: bool = False,
) -> list:
    """One item per (sample, cell type) for APOC / ConvPaint / Cellpose style outputs."""
    names = [str(n) for n in metadata["sample_name"].dropna().unique()]
    cts = list(cell_types) + (["dead"] if include_dead else [])
    items = []
    for sn in names:
        for ct in cts:
            probe = segmentation_status(out_dir, sn, ct, include_mask=include_mask)
            items.append(PlanItem(sn, ct, probe.status, probe.detail))
    return items


def scan_pixelclassifier_items(metadata, out_dir, cell_types: Sequence[str]) -> list:
    """One item per sample for the CPU pixel-classifier engine (all cell types together)."""
    items = []
    for sn in [str(n) for n in metadata["sample_name"].dropna().unique()]:
        img_dir = Path(out_dir) / "images" / sn
        present = [segmentation_output_path(out_dir, sn, ct).exists() for ct in cell_types]
        interrupted = (img_dir / ".seg_processing").exists()
        if not any(present) and not interrupted:
            items.append(PlanItem(sn, ALL_CELL_TYPES, Status.MISSING, "not segmented"))
        elif interrupted:
            items.append(PlanItem(sn, ALL_CELL_TYPES, Status.PARTIAL, "interrupted run (resumed)"))
        elif all(present):
            items.append(PlanItem(sn, ALL_CELL_TYPES, Status.COMPLETE, "all cell types segmented"))
        else:
            missing = [ct for ct, ok in zip(cell_types, present) if not ok]
            items.append(PlanItem(sn, ALL_CELL_TYPES, Status.PARTIAL, f"missing: {', '.join(missing)}"))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Queue: keep downstream outputs consistent with upstream changes
# ─────────────────────────────────────────────────────────────────────────────
def cascade_plans(ordered_plans: Sequence[RunPlan], members: Optional[Mapping[str, Sequence[str]]] = None):
    """Force re-runs of items made stale by an earlier step in the same queue.

    ``ordered_plans`` holds one plan per queue step, in execution order.  When a
    step will (re)compute ``(sample, cell_type)``, any later item for the same
    sample and cell type that was marked Skip is no longer valid, so it becomes
    Overwrite.  ``members`` maps group keys to their member cell types (the joint
    all-organoids key to the organoid types, ``<base>_merged`` to its channels), so
    a change to a member also invalidates the group and vice versa.  Filtering
    items (keyed by :data:`ALL_SAMPLES`) go stale when any sample of that cell
    type changed.

    Returns ``(plans, notes)`` where ``notes`` describe each forced re-run.
    """
    members = {k: list(v) for k, v in (members or {}).items()}
    touched = set()          # (sample, cell_type) recomputed by an earlier step
    touched_samples = set()  # samples whose every cell type was recomputed
    out_plans, notes = [], []

    def stale(sample, ct):
        if (sample, ct) in touched or sample in touched_samples:
            return True
        if ct in members and any((sample, m) in touched for m in members[ct]):
            return True
        if sample == ALL_SAMPLES:
            return bool(touched_samples) or any(c == ct for (_s, c) in touched)
        return False

    for plan in ordered_plans:
        actions = {}
        for it in plan.items:
            act = plan.action(it.sample, it.cell_type)
            if act is Action.SKIP and stale(it.sample, it.cell_type):
                act = Action.OVERWRITE
                label = f"{it.step + ': ' if it.step else ''}{it.sample} / {_label(it.cell_type)}"
                notes.append(f"{label} will be re-run because an earlier step changes its input")
            actions[it.key] = act
        new_plan = RunPlan(plan.items, actions)
        out_plans.append(new_plan)
        for it in new_plan.items:
            if new_plan.action(it.sample, it.cell_type) is Action.SKIP:
                continue
            if it.cell_type == ALL_CELL_TYPES:
                touched_samples.add(it.sample)
            elif it.sample != ALL_SAMPLES:
                touched.add(it.key)
                for m in members.get(it.cell_type, ()):
                    touched.add((it.sample, m))
    return out_plans, notes


def _label(cell_type: str) -> str:
    return "all organoids" if cell_type == ALL_ORGANOIDS else cell_type

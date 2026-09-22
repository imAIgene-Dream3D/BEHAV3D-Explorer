"""
Localised death-event detection for BEHAV3D Active Killing.

A *death event* is the nucleation of a spatially connected region of
**newly** dead voxels inside one target (e.g. organoid) track. It is the
atomic unit of the Active Killing analysis: every event later carries
exactly one unit of killing credit in total, so the number of events -- and
therefore the total credit -- cannot grow when more effector cells are
present. No effector data enters the detection at all (effector segments are
only used to *remove* their own dye from the dead mask).

Death is read from the annotated binary dead mask (``dead_mask_path``), never
by re-thresholding the raw death channel. Per frame the detector reads the
dead mask, the target's tracked labels (voxel value == TrackID) and the
effector tracked labels.

Algorithm (per sample, per target type, sequential in time)
-----------------------------------------------------------
1. Clean the dead mask under every effector segment.
2. Persistence: a voxel counts as dead at ``t`` only if it is also dead
   (within ``novelty_tolerance_um``) at ``t + 1``. Single-frame classifier
   flicker therefore never nucleates an event, and the last analysed frame
   cannot nucleate one.
3. Motion compensation: each target's cumulative dead set ("EVER") and its
   open patches are carried along with the target by the integer voxel shift
   of its centroid between frames.
4. Novelty: ``NEW_t = dead_t \\ dilate(EVER_{t-1}, tolerance)``. A voxel that
   goes dead -> alive -> dead is never novel twice. Death already present the
   first time a target is seen seeds EVER and is never counted as new.
5. Patch formation: close NEW by ``patch_closing_um``, label with
   6-connectivity.
6. Patch tracking: a component within ``patch_link_radius_um`` of an open
   patch is growth of that patch -- never an event. An unlinked component
   nucleates an event if it reaches ``min_patch_volume_um3``; a smaller seed
   is held for ``nucleation_grace_frames`` and, if it grows past the floor,
   nucleates with its onset back-dated to when it first appeared. Merging
   patches keep every event they already represent.

   One death event = one nucleation. Growth, merges and re-brightening are
   never events.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.spatial import cKDTree

from behav3d.core.utils import convert_distance, convert_time, get_current_time


DEATH_EVENTS_SCHEMA_VERSION = 1


class StaleDeathEventsError(RuntimeError):
    """Raised when a cached death-event table predates the masks it was built from."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class DeathEventParams:
    """Detection parameters.

    Only ``target_cell_diameter_um`` is a user-facing choice. Everything else
    is derived from it, from the voxel spacing, or is a fixed constant; any
    of them can be overridden explicitly (``active_killing.advanced`` in the
    parameters YAML) and ``None`` means "derive".
    """

    target_cell_diameter_um: float = 10.0
    min_patch_volume_um3: Optional[float] = None       # 0.25 * (4/3) pi (d/2)^3
    patch_closing_um: Optional[float] = None           # d / 6
    patch_link_radius_um: Optional[float] = None       # d / 3
    novelty_tolerance_um: Optional[float] = None       # max(1.5, voxel diagonal)
    persistence_frames: int = 2
    nucleation_grace_frames: int = 2
    patch_idle_close_min: float = 30.0
    bad_frame_drop_fraction: float = 0.8

    def resolved(self, voxel_spacing: Sequence[float]) -> "DeathEventParams":
        """Return a copy with every derived value filled in."""
        d = float(self.target_cell_diameter_um)
        if not np.isfinite(d) or d <= 0:
            raise ValueError(f"target_cell_diameter_um must be > 0, got {self.target_cell_diameter_um!r}")
        diag = float(np.sqrt(np.sum(np.square(np.asarray(voxel_spacing, dtype=float)))))
        out = DeathEventParams(**asdict(self))
        if out.min_patch_volume_um3 is None:
            out.min_patch_volume_um3 = 0.25 * (4.0 / 3.0) * math.pi * (d / 2.0) ** 3
        if out.patch_closing_um is None:
            out.patch_closing_um = d / 6.0
        if out.patch_link_radius_um is None:
            out.patch_link_radius_um = d / 3.0
        if out.novelty_tolerance_um is None:
            out.novelty_tolerance_um = max(1.5, diag)
        return out

    def cache_key(self) -> Dict:
        return {k: (None if v is None else (float(v) if isinstance(v, float) else v))
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Geometry helpers (anisotropic, EDT-based, bounding-box restricted)
# ---------------------------------------------------------------------------

def _pad_vox(radius_um: float, spacing: Sequence[float]) -> np.ndarray:
    return np.array([int(math.ceil(radius_um / s)) + 1 for s in spacing], dtype=int)


def _bbox_slices(mask: np.ndarray, pad: Sequence[int]) -> Optional[Tuple[slice, ...]]:
    nz = np.nonzero(mask)
    if nz[0].size == 0:
        return None
    return tuple(
        slice(max(0, int(ax.min()) - int(p)), min(n, int(ax.max()) + int(p) + 1))
        for ax, p, n in zip(nz, pad, mask.shape)
    )


def _dilate_um(mask: np.ndarray, radius_um: float, spacing: Sequence[float]) -> np.ndarray:
    """Dilate a boolean mask by a physical radius (anisotropy-correct)."""
    mask = np.asarray(mask, dtype=bool)
    if radius_um <= 0 or not mask.any():
        return mask.copy()
    sl = _bbox_slices(mask, _pad_vox(radius_um, spacing))
    out = np.zeros_like(mask)
    sub = mask[sl]
    out[sl] = distance_transform_edt(~sub, sampling=spacing) <= radius_um
    return out


def _erode_um(mask: np.ndarray, radius_um: float, spacing: Sequence[float]) -> np.ndarray:
    """Erode a boolean mask by a physical radius; outside the array counts as background."""
    mask = np.asarray(mask, dtype=bool)
    if radius_um <= 0 or not mask.any():
        return mask.copy()
    sl = _bbox_slices(mask, [1] * mask.ndim)
    sub = np.pad(mask[sl], 1, constant_values=False)
    inner = distance_transform_edt(sub, sampling=spacing)[tuple(slice(1, -1) for _ in range(mask.ndim))]
    out = np.zeros_like(mask)
    out[sl] = inner > radius_um
    return out


def _close_um(mask: np.ndarray, radius_um: float, spacing: Sequence[float]) -> np.ndarray:
    if radius_um <= 0:
        return np.asarray(mask, dtype=bool).copy()
    # Pad first so the dilation is never clipped by the crop edge, which would
    # otherwise make the subsequent erosion eat real voxels there.
    pad = _pad_vox(radius_um, spacing)
    padded = np.pad(np.asarray(mask, dtype=bool), [(p, p) for p in pad], constant_values=False)
    closed = _erode_um(_dilate_um(padded, radius_um, spacing), radius_um, spacing)
    return closed[tuple(slice(p, p + n) for p, n in zip(pad, mask.shape))]


def _paste(coords: np.ndarray, origin: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Rasterise global voxel coords into a boolean crop whose corner is ``origin``."""
    out = np.zeros(tuple(shape), dtype=bool)
    if coords is None or len(coords) == 0:
        return out
    local = coords - origin
    keep = np.all((local >= 0) & (local < np.asarray(shape)), axis=1)
    local = local[keep]
    out[local[:, 0], local[:, 1], local[:, 2]] = True
    return out


_ENC_OFF = 1 << 15
_ENC_MUL = 1 << 17


def _unique_coords(coords: np.ndarray) -> np.ndarray:
    """Deduplicate (N, 3) integer voxel coords (linear-index encoding, much faster than axis=0 unique)."""
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    c = coords.astype(np.int64) + _ENC_OFF
    key = (c[:, 0] * _ENC_MUL + c[:, 1]) * _ENC_MUL + c[:, 2]
    _, idx = np.unique(key, return_index=True)
    return coords[np.sort(idx)].astype(np.int32)


# ---------------------------------------------------------------------------
# Per-target tracker
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class _Patch:
    coords: np.ndarray                 # (N, 3) int32, global voxel coords in the *current* frame
    first_seen_t: int
    last_growth_t: int
    event_id: Optional[int] = None     # None while still a sub-threshold seed
    extra_event_ids: List[int] = field(default_factory=list)


@dataclass(eq=False)
class _TargetState:
    centroid: np.ndarray
    ever: np.ndarray
    patches: List[_Patch]
    last_seen_t: int


class OrganoidDeathTracker:
    """Detect death-event nucleations inside every label of one target type.

    Call :meth:`update` once per frame, in increasing ``t``; it returns the
    events nucleated at that frame. Events are also accumulated in
    :attr:`events`, their onset patches in :attr:`onset_patches` and the
    per-(target, t) dead-volume series in :attr:`timeseries`.
    """

    def __init__(
        self,
        *,
        sample_name: str,
        target_type: str,
        voxel_spacing: Sequence[float],
        minutes_per_frame: float,
        params: DeathEventParams,
        image_shape: Sequence[int],
        event_id_start: int = 1,
    ):
        self.sample_name = str(sample_name)
        self.target_type = str(target_type)
        self.spacing = tuple(float(s) for s in voxel_spacing)
        self.voxel_volume = float(np.prod(self.spacing))
        self.dt = float(minutes_per_frame)
        self.p = params.resolved(self.spacing)
        self.image_shape = tuple(int(n) for n in image_shape)
        self._next_event_id = int(event_id_start)
        self._state: Dict[int, _TargetState] = {}
        self._centroid_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.events: List[Dict] = []
        self._event_index: Dict[int, Dict] = {}
        self.onset_patches: Dict[int, np.ndarray] = {}
        self.timeseries: List[Dict] = []
        self._crop_pad = _pad_vox(
            self.p.novelty_tolerance_um + self.p.patch_link_radius_um + self.p.patch_closing_um,
            self.spacing,
        )
        self._idle_frames = max(1, int(math.ceil(self.p.patch_idle_close_min / max(self.dt, 1e-9))))

    # -- helpers -----------------------------------------------------------------
    def _crop(self, sl: Tuple[slice, ...]) -> Tuple[Tuple[slice, ...], np.ndarray]:
        crop = tuple(
            slice(max(0, s.start - int(p)), min(n, s.stop + int(p)))
            for s, p, n in zip(sl, self._crop_pad, self.image_shape)
        )
        return crop, np.array([c.start for c in crop], dtype=np.int32)

    def _is_border(self, coords: np.ndarray) -> bool:
        # Lateral (x/y) image border only: a patch cut by the field of view has
        # unknown volume and an unknown neighbourhood. Touching the first/last z
        # slice is a normal imaging condition for organoids resting on glass.
        ny, nx = self.image_shape[1], self.image_shape[2]
        return bool(
            np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == ny - 1)
            or np.any(coords[:, 2] == 0) or np.any(coords[:, 2] == nx - 1)
        )

    def _nucleate(self, patch: _Patch, t_now: int, label: int, state: _TargetState,
                  org_mask_local: np.ndarray, crop_origin: np.ndarray,
                  dead_fraction: float, organoid_volume_um3: float) -> Dict:
        onset_t = int(patch.first_seen_t)
        # Warp the patch back into the onset frame by the target's own motion.
        c_now = self._centroid_history[label].get(t_now, state.centroid)
        c_onset = self._centroid_history[label].get(onset_t, c_now)
        shift = np.round(c_now - c_onset).astype(np.int32)
        onset_coords = _unique_coords(patch.coords - shift)
        eid = self._next_event_id
        self._next_event_id += 1
        patch.event_id = eid

        centroid_vox = onset_coords.mean(axis=0)
        # Patch depth: distance of the patch centroid from the target surface,
        # measured in the current crop (sub-voxel motion ignored).
        depth_um = float("nan")
        local_c = np.round(patch.coords.mean(axis=0)).astype(int) - crop_origin
        if np.all(local_c >= 0) and np.all(local_c < np.asarray(org_mask_local.shape)):
            inside = distance_transform_edt(
                np.pad(org_mask_local, 1, constant_values=False), sampling=self.spacing
            )[tuple(slice(1, -1) for _ in range(3))]
            depth_um = float(inside[tuple(local_c)])

        record = {
            "death_event_id": eid,
            "sample_name": self.sample_name,
            "organoid_type": self.target_type,
            "target_track_id": int(label),
            "t_onset": onset_t,
            "time_onset_min": onset_t * self.dt,
            "z_onset_vox": float(centroid_vox[0]),
            "y_onset_vox": float(centroid_vox[1]),
            "x_onset_vox": float(centroid_vox[2]),
            "z_onset_um": float(centroid_vox[0] * self.spacing[0]),
            "y_onset_um": float(centroid_vox[1] * self.spacing[1]),
            "x_onset_um": float(centroid_vox[2] * self.spacing[2]),
            "onset_volume_um3": float(len(onset_coords) * self.voxel_volume),
            "peak_volume_um3": float(len(patch.coords) * self.voxel_volume),
            "patch_depth_um": depth_um,
            "organoid_volume_um3": float(organoid_volume_um3),
            "organoid_dead_fraction_at_onset": float(dead_fraction),
            "is_border_patch": self._is_border(onset_coords),
            "back_dated_frames": int(t_now - onset_t),
            "n_components_merged": 1,
            "merged_into": -1,
        }
        self.events.append(record)
        self._event_index[eid] = record
        self.onset_patches[eid] = onset_coords
        return record

    # -- main entry --------------------------------------------------------------
    def update(
        self,
        t: int,
        labels_t: np.ndarray,
        dead_t: np.ndarray,
        dead_next: Optional[np.ndarray],
        *,
        allow_novelty: bool = True,
    ) -> List[Dict]:
        """Process frame ``t``.

        ``dead_t`` / ``dead_next`` are the cleaned boolean dead masks at ``t``
        and ``t + 1`` (``dead_next`` None at the last analysed frame, which then
        cannot nucleate). ``allow_novelty=False`` marks a bad frame: state is
        carried forward, nothing nucleates.
        """
        p = self.p
        new_events: List[Dict] = []
        slices = find_objects(labels_t)
        present = set()
        for idx, sl in enumerate(slices):
            if sl is None:
                continue
            label = idx + 1
            present.add(label)
            crop, origin = self._crop(sl)
            lab_crop = labels_t[crop]
            org = lab_crop == label
            org_coords = np.argwhere(org).astype(np.int32) + origin
            if org_coords.size == 0:
                continue
            centroid = org_coords.mean(axis=0)
            self._centroid_history.setdefault(label, {})[int(t)] = centroid

            state = self._state.get(label)
            first_sighting = state is None
            if first_sighting:
                state = _TargetState(centroid=centroid, ever=np.zeros((0, 3), np.int32),
                                     patches=[], last_seen_t=int(t))
                self._state[label] = state
            else:
                shift = np.round(centroid - state.centroid).astype(np.int32)
                if np.any(shift):
                    state.ever = state.ever + shift
                    for patch in state.patches:
                        patch.coords = patch.coords + shift
                state.centroid = centroid
                state.last_seen_t = int(t)

            raw_dead = np.asarray(dead_t[crop], dtype=bool) & org
            # Persistence: only voxels still dead (within tolerance) at t+1 may
            # count as death. The reported dead volume uses the raw mask.
            if dead_next is not None and raw_dead.any():
                confirm = _dilate_um(np.asarray(dead_next[crop], dtype=bool), p.novelty_tolerance_um, self.spacing)
                dead_crop = raw_dead & confirm
            else:
                dead_crop = np.zeros_like(raw_dead)

            n_org = int(org.sum())
            organoid_volume_um3 = n_org * self.voxel_volume
            dead_fraction = float(raw_dead.sum()) / max(n_org, 1)
            n_new = 0

            # On a bad frame nothing is learned: EVER and patches are carried
            # forward unchanged, so a classifier dropout is never read as
            # resurrection followed by mass re-death.
            if first_sighting and dead_crop.any():
                # Death already present the first time a target is seen (movie
                # start, or a track appearing mid-movie) is not *new* death: it
                # seeds the reference instead of nucleating events.
                state.ever = _unique_coords(np.argwhere(dead_crop).astype(np.int32) + origin)
            elif dead_crop.any() and allow_novelty:
                ref = _paste(state.ever, origin, org.shape)
                seen = _dilate_um(ref, p.novelty_tolerance_um, self.spacing) if ref.any() else ref
                new = dead_crop & ~seen
                n_new = int(new.sum())

                # Grow EVER with everything dead now, then re-anchor it to the
                # target so memory of regions the target left is forgotten.
                ever = _unique_coords(np.concatenate([state.ever, np.argwhere(dead_crop).astype(np.int32) + origin]))
                near = _dilate_um(org, p.novelty_tolerance_um, self.spacing)
                local = ever - origin
                inside = np.all((local >= 0) & (local < np.asarray(org.shape)), axis=1)
                keep = np.zeros(len(ever), dtype=bool)
                keep[inside] = near[local[inside, 0], local[inside, 1], local[inside, 2]]
                state.ever = ever[keep]

                if new.any():
                    closed = _close_um(new, p.patch_closing_um, self.spacing) & org
                    comp, n_comp = ndimage.label(closed, structure=ndimage.generate_binary_structure(3, 1))
                    for ci in range(1, n_comp + 1):
                        c_coords = np.argwhere(comp == ci).astype(np.int32) + origin
                        self._link_component(t, state, c_coords)
                # Region-grow open patches into newly dead voxels near them. This
                # is what captures growth thinner than the novelty tolerance (which
                # never shows up as a NEW component): a seed growing by thin rims
                # still reaches the floor, and two growing patches still meet.
                self._grow_patches(t, state, dead_crop & ~ref, origin)
                self._merge_touching_patches(state)
                for patch in state.patches:
                    if patch.event_id is None and len(patch.coords) * self.voxel_volume >= p.min_patch_volume_um3:
                        new_events.append(self._nucleate(
                            patch, t, label, state, org, origin, dead_fraction, organoid_volume_um3,
                        ))
                    elif patch.event_id is not None:
                        rec = self._event_index[patch.event_id]
                        rec["peak_volume_um3"] = max(rec["peak_volume_um3"], len(patch.coords) * self.voxel_volume)

            # Seeds that never reached the floor, and patches that stopped growing.
            for patch in list(state.patches):
                if patch.event_id is None and t - patch.first_seen_t >= p.nucleation_grace_frames:
                    state.patches.remove(patch)
                elif patch.event_id is not None and t - patch.last_growth_t > self._idle_frames:
                    state.patches.remove(patch)

            self.timeseries.append({
                "sample_name": self.sample_name,
                "organoid_type": self.target_type,
                "target_track_id": int(label),
                "position_t": int(t),
                "centroid_z_vox": float(centroid[0]),
                "centroid_y_vox": float(centroid[1]),
                "centroid_x_vox": float(centroid[2]),
                "organoid_volume_um3": organoid_volume_um3,
                "dead_volume_um3": float(raw_dead.sum()) * self.voxel_volume,
                "new_dead_volume_um3": n_new * self.voxel_volume,
                "frame_quality": "good" if allow_novelty else "bad",
            })
        return new_events

    def _merge_into(self, base: _Patch, other: _Patch) -> None:
        base.coords = _unique_coords(np.concatenate([base.coords, other.coords]))
        base.first_seen_t = min(base.first_seen_t, other.first_seen_t)
        base.last_growth_t = max(base.last_growth_t, other.last_growth_t)
        if other.event_id is not None:
            # A merge never deletes an event: both nucleations stay counted;
            # the later one records what it merged into.
            self._event_index[other.event_id]["merged_into"] = base.event_id
            base.extra_event_ids.append(other.event_id)
        base.extra_event_ids.extend(other.extra_event_ids)

    def _pick_base(self, patches: List[_Patch]) -> _Patch:
        nucleated = [pt for pt in patches if pt.event_id is not None]
        if nucleated:
            return min(nucleated, key=lambda pt: (self._event_index[pt.event_id]["t_onset"], pt.event_id))
        return min(patches, key=lambda pt: pt.first_seen_t)

    def _link_component(self, t: int, state: _TargetState, c_coords: np.ndarray) -> None:
        """Attach a NEW component to the open patches it touches, or open a new seed."""
        c_um = c_coords * np.asarray(self.spacing)
        linked = []
        for patch in state.patches:
            if len(patch.coords) == 0:
                continue
            d, _ = cKDTree(patch.coords * np.asarray(self.spacing)).query(c_um, k=1)
            if np.min(d) <= self.p.patch_link_radius_um:
                linked.append(patch)
        if not linked:
            state.patches.append(_Patch(coords=_unique_coords(c_coords), first_seen_t=int(t), last_growth_t=int(t)))
            return
        base = self._pick_base(linked)
        for other in linked:
            if other is not base:
                self._merge_into(base, other)
                state.patches.remove(other)
        if base.event_id is not None:
            self._event_index[base.event_id]["n_components_merged"] += 1
        base.coords = _unique_coords(np.concatenate([base.coords, c_coords]))
        base.last_growth_t = int(t)

    def _grow_patches(self, t: int, state: _TargetState, candidates: np.ndarray, origin: np.ndarray) -> None:
        if not candidates.any():
            return
        pad = _pad_vox(self.p.patch_link_radius_um, self.spacing)
        shape = np.asarray(candidates.shape)
        for patch in state.patches:
            local = patch.coords - origin
            inside = np.all((local >= 0) & (local < shape), axis=1)
            if not inside.any():
                continue
            local = local[inside]
            lo = np.maximum(local.min(axis=0) - pad, 0)
            hi = np.minimum(local.max(axis=0) + pad + 1, shape)
            sub = tuple(slice(int(x0), int(x1)) for x0, x1 in zip(lo, hi))
            pm = np.zeros(tuple(int(v) for v in (hi - lo)), dtype=bool)
            ll = local - lo
            pm[ll[:, 0], ll[:, 1], ll[:, 2]] = True
            grow = candidates[sub] & _dilate_um(pm, self.p.patch_link_radius_um, self.spacing) & ~pm
            if grow.any():
                patch.coords = _unique_coords(np.concatenate(
                    [patch.coords, np.argwhere(grow).astype(np.int32) + lo.astype(np.int32) + origin]))
                patch.last_growth_t = int(t)

    def _merge_touching_patches(self, state: _TargetState) -> None:
        merged = True
        while merged and len(state.patches) > 1:
            merged = False
            for i in range(len(state.patches)):
                a_patch = state.patches[i]
                tree = cKDTree(a_patch.coords * np.asarray(self.spacing))
                for j in range(i + 1, len(state.patches)):
                    b_patch = state.patches[j]
                    d, _ = tree.query(b_patch.coords * np.asarray(self.spacing), k=1)
                    if np.min(d) <= self.p.patch_link_radius_um:
                        base = self._pick_base([a_patch, b_patch])
                        other = b_patch if base is a_patch else a_patch
                        self._merge_into(base, other)
                        state.patches.remove(other)
                        merged = True
                        break
                if merged:
                    break

    @property
    def next_event_id(self) -> int:
        return self._next_event_id


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_dead_mask_path(sample_row: pd.Series, output_dir: Union[str, Path], log_fn=None):
    """Return ``(path, tried)`` for a sample's annotated dead mask, or ``(None, tried)``.

    Only real, annotated dead masks are accepted: the metadata
    ``dead_mask_path`` column, then the canonical ``{sample}_mask_dead.zarr``
    / ``{sample}_dead_mask.zarr`` names. There is deliberately **no** fallback
    that re-thresholds the raw death channel -- that would silently produce a
    mask different from the one every other death feature used.
    """
    _log = log_fn or (lambda m: None)
    tried: List[str] = []
    dm_val = sample_row.get("dead_mask_path")
    if dm_val is not None and pd.notna(dm_val) and str(dm_val).strip():
        p = Path(str(dm_val).strip().strip('"').strip("'"))
        tried.append(str(p))
        if p.exists():
            return p, tried
        _log(f"  [dead mask] metadata dead_mask_path does not exist on disk: {p}")
    sample_name = str(sample_row.get("sample_name", ""))
    img_dir = Path(output_dir) / "images" / sample_name
    for name in (f"{sample_name}_mask_dead.zarr", f"{sample_name}_dead_mask.zarr"):
        p = img_dir / name
        tried.append(str(p))
        if p.exists():
            return p, tried
    return None, tried


def resolve_tracks_image_path(sample_row: pd.Series, cell_type: str) -> Optional[Path]:
    """Return the tracked-label zarr for ``cell_type`` (any of the or_/im_/ot_ prefixes)."""
    for prefix in ("or", "im", "ot"):
        col = f"{prefix}_{cell_type}_tracks_image_path"
        if col in sample_row.index and pd.notna(sample_row[col]) and str(sample_row[col]).strip():
            return Path(str(sample_row[col]).strip())
    return None


def voxel_spacing_from_sample(sample_row: pd.Series) -> Tuple[float, float, float]:
    """``(dz, dy, dx)`` in µm for one metadata row."""
    unit = sample_row.get("distance_unit", "µm")
    if unit in ("_m", "µm"):
        unit = "μm"
    xy = float(convert_distance(sample_row["pixel_distance_xy"], unit))
    z = float(convert_distance(sample_row["pixel_distance_z"], unit))
    return (z, xy, xy)


def minutes_per_frame_from_sample(sample_row: pd.Series) -> float:
    return float(convert_time(sample_row["time_interval"], sample_row["time_unit"], convert_to="h")) * 60.0


# ---------------------------------------------------------------------------
# Per-sample detection
# ---------------------------------------------------------------------------

def _frame_window(n_frames: int, timepoint_range: Optional[Tuple[Optional[int], Optional[int]]]) -> Tuple[int, int]:
    start, end = 0, n_frames - 1
    if timepoint_range is not None:
        if timepoint_range[0] is not None:
            start = max(start, int(timepoint_range[0]))
        if timepoint_range[1] is not None:
            end = min(end, int(timepoint_range[1]))
    return start, end


def detect_death_events_sample(
    *,
    sample_name: str,
    target_type: str,
    target_tracks_path: Union[str, Path],
    dead_mask_path: Union[str, Path],
    immune_tracks_paths: Dict[str, Union[str, Path]],
    voxel_spacing: Sequence[float],
    minutes_per_frame: float,
    params: DeathEventParams,
    timepoint_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    progress_cb=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, np.ndarray]]:
    """Detect death events for one (sample, target type).

    Returns ``(df_events, df_timeseries, onset_patches)`` where
    ``onset_patches`` maps ``death_event_id`` to the ``(N, 3)`` voxel coords
    of the patch in its onset frame.
    """
    from behav3d.io.images import open_image_timepoints
    from behav3d.features.timepoint_features import _zero_dead_mask_under_segments

    labels_h = open_image_timepoints(target_tracks_path)
    dead_h = open_image_timepoints(dead_mask_path)
    immune_h = {k: open_image_timepoints(v) for k, v in immune_tracks_paths.items()}
    n_frames = int(min(labels_h.shape[0], dead_h.shape[0]))
    if labels_h.shape[1:] != dead_h.shape[1:]:
        raise ValueError(
            f"{sample_name}: dead mask shape {tuple(dead_h.shape[1:])} does not match "
            f"{target_type} labels shape {tuple(labels_h.shape[1:])}"
        )
    start, end = _frame_window(n_frames, timepoint_range)
    if end < start:
        empty = pd.DataFrame()
        return empty, empty, {}

    cache: Dict[int, np.ndarray] = {}

    def dead_at(t: int) -> np.ndarray:
        if t not in cache:
            d = np.asarray(dead_h[t]) > 0
            if immune_h:
                d = _zero_dead_mask_under_segments(d, {k: np.asarray(h[t]) for k, h in immune_h.items()})
            cache[t] = np.asarray(d, dtype=bool)
            for old in [k for k in cache if k < t - 3]:
                del cache[old]
        return cache[t]

    counts: Dict[int, int] = {}

    def count_at(t: int) -> int:
        if t not in counts:
            counts[t] = int(np.count_nonzero(dead_at(t)))
        return counts[t]

    tracker = OrganoidDeathTracker(
        sample_name=sample_name, target_type=target_type, voxel_spacing=voxel_spacing,
        minutes_per_frame=minutes_per_frame, params=params, image_shape=labels_h.shape[1:],
    )
    total = end - start + 1
    for i, t in enumerate(range(start, end + 1)):
        neighbours = [count_at(u) for u in range(t - 2, t + 3) if u != t and start <= u <= end]
        med = float(np.median(neighbours)) if neighbours else 0.0
        good = not (med > 0 and count_at(t) < (1.0 - params.bad_frame_drop_fraction) * med)
        labels_t = np.asarray(labels_h[t])
        dead_next = dead_at(t + 1) if t + 1 <= end else None
        tracker.update(t, labels_t, dead_at(t), dead_next, allow_novelty=good)
        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, f"{sample_name} · {target_type} · t={t}")
            except Exception:
                pass

    df_events = pd.DataFrame(tracker.events)
    df_ts = pd.DataFrame(tracker.timeseries)
    return df_events, df_ts, dict(tracker.onset_patches)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def death_events_dir(output_dir: Union[str, Path], target_type: str) -> Path:
    return Path(output_dir) / "analysis" / target_type / "death_events"


def _cache_paths(output_dir, target_type, sample_name) -> Dict[str, Path]:
    d = death_events_dir(output_dir, target_type)
    return {
        "events": d / f"{sample_name}_death_events.csv",
        "timeseries": d / f"{sample_name}_death_timeseries.csv",
        "patches": d / f"{sample_name}_death_patches.npz",
        "params": d / "death_events_params.json",
    }


def _cache_record(params: DeathEventParams, timepoint_range, immune_types, voxel_spacing, minutes_per_frame) -> Dict:
    rec = {
        "schema_version": DEATH_EVENTS_SCHEMA_VERSION,
        "params": params.cache_key(),
        "timepoint_range": list(timepoint_range) if timepoint_range is not None else None,
        "immune_types_used_for_cleaning": sorted(immune_types),
        "voxel_spacing_um": [float(s) for s in voxel_spacing],
        "minutes_per_frame": float(minutes_per_frame),
    }
    rec["key"] = hashlib.sha1(json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()
    return rec


def _path_mtime(p: Union[str, Path]) -> float:
    # For a .zarr store, stat the store directory / its array metadata, never
    # walk the chunks (they can number in the hundreds of thousands).
    p = Path(p)
    best = p.stat().st_mtime
    for meta in ("zarr.json", ".zarray"):
        m = p / meta
        if m.exists():
            best = max(best, m.stat().st_mtime)
    return best


def find_death_event_table(
    output_dir, target_type: str, sample_name: str, *, expected_key: Optional[str] = None,
    upstream_paths: Sequence[Union[str, Path]] = (),
) -> Optional[Dict[str, Path]]:
    """Return the cache paths for a sample if a fresh, matching cache exists.

    Returns None when there is no cache or the parameters/range/immune set
    differ (the caller should re-detect). Raises :class:`StaleDeathEventsError`
    when the cache predates one of ``upstream_paths`` (a mask was regenerated).
    """
    paths = _cache_paths(output_dir, target_type, sample_name)
    if not (paths["events"].exists() and paths["patches"].exists() and paths["params"].exists()):
        return None
    try:
        registry = json.loads(paths["params"].read_text(encoding="utf-8"))
    except Exception:
        return None
    rec = registry.get("samples", {}).get(sample_name)
    if not rec or (expected_key is not None and rec.get("key") != expected_key):
        return None
    cache_mtime = paths["events"].stat().st_mtime
    for up in upstream_paths:
        if up and Path(up).exists() and _path_mtime(up) > cache_mtime:
            raise StaleDeathEventsError(
                f"The cached death events for {sample_name} ({target_type}) are older than "
                f"{Path(up).name} -- Segmentation or Tracking was rerun after they were detected. "
                f"They will be re-detected."
            )
    return paths


def _save_cache(output_dir, target_type, sample_name, df_events, df_ts, patches, record):
    paths = _cache_paths(output_dir, target_type, sample_name)
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    df_events.to_csv(paths["events"], index=False)
    df_ts.to_csv(paths["timeseries"], index=False)
    np.savez_compressed(paths["patches"], **{f"e{k}": v.astype(np.int32) for k, v in patches.items()})
    registry = {"samples": {}}
    if paths["params"].exists():
        try:
            registry = json.loads(paths["params"].read_text(encoding="utf-8"))
        except Exception:
            registry = {"samples": {}}
    registry.setdefault("samples", {})[sample_name] = dict(record, computed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    paths["params"].write_text(json.dumps(registry, indent=2), encoding="utf-8")


def load_death_patches(path: Union[str, Path]) -> Dict[int, np.ndarray]:
    with np.load(path) as z:
        return {int(k[1:]): np.asarray(z[k]) for k in z.files}


def detect_death_events(
    metadata: pd.DataFrame,
    output_dir: Union[str, Path],
    target_cell_types: Sequence[str],
    params: DeathEventParams,
    *,
    timepoint_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    reuse_cache: bool = True,
    progress_cb=None,
    log_fn=print,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Dict[int, np.ndarray]], Dict]:
    """Detect (or load cached) death events for every sample and target type.

    Returns ``(df_events, df_timeseries, patches, sample_info)`` where
    ``patches[(sample, target_type)]`` maps event id -> onset voxel coords and
    ``sample_info[sample]`` carries the voxel spacing and frame interval.
    Event ids are unique within ``(sample_name, organoid_type)``.
    """
    from behav3d.core.metadata import resolve_immune_track_paths_for_sample

    all_events, all_ts = [], []
    patches: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
    sample_info: Dict[str, Dict] = {}
    rows = [r for _, r in metadata.iterrows()]
    total_jobs = max(1, len(rows) * len(target_cell_types))
    job = 0
    for row in rows:
        sample_name = str(row["sample_name"])
        spacing = voxel_spacing_from_sample(row)
        mpf = minutes_per_frame_from_sample(row)
        sample_info[sample_name] = {"voxel_spacing": spacing, "minutes_per_frame": mpf}
        dead_path, tried = resolve_dead_mask_path(row, output_dir)
        if dead_path is None:
            raise FileNotFoundError(
                f"No annotated dead mask found for sample {sample_name}. Active Killing reads death "
                f"from the dead mask (not the raw death channel); run dead-mask segmentation first. "
                f"Tried: {tried}"
            )
        immune_paths = {k: v for k, v in resolve_immune_track_paths_for_sample(row).items() if Path(v).exists()}
        for target_type in target_cell_types:
            job += 1
            target_path = resolve_tracks_image_path(row, target_type)
            if target_path is None or not target_path.exists():
                log_fn(f"{get_current_time()} -   {sample_name}: no tracked {target_type} image - skipped")
                continue
            cleaning = {k: v for k, v in immune_paths.items() if k != target_type}
            record = _cache_record(params, timepoint_range, cleaning.keys(), spacing, mpf)
            cached = None
            if reuse_cache:
                try:
                    cached = find_death_event_table(
                        output_dir, target_type, sample_name, expected_key=record["key"],
                        upstream_paths=[dead_path, target_path, *cleaning.values()],
                    )
                except StaleDeathEventsError as exc:
                    log_fn(f"{get_current_time()} -   {exc}")
                    cached = None
            if cached is not None:
                df_e = pd.read_csv(cached["events"]) if cached["events"].stat().st_size > 1 else pd.DataFrame()
                df_t = pd.read_csv(cached["timeseries"]) if cached["timeseries"].stat().st_size > 1 else pd.DataFrame()
                pt = load_death_patches(cached["patches"])
                log_fn(f"{get_current_time()} -   {sample_name} · {target_type}: reused {len(df_e)} cached death events")
            else:
                log_fn(f"{get_current_time()} -   {sample_name} · {target_type}: detecting death events...")

                def _cb(i, n, label, _job=job):
                    if progress_cb is not None:
                        progress_cb((_job - 1) + i / max(n, 1), total_jobs, label)

                df_e, df_t, pt = detect_death_events_sample(
                    sample_name=sample_name, target_type=target_type, target_tracks_path=target_path,
                    dead_mask_path=dead_path, immune_tracks_paths=cleaning, voxel_spacing=spacing,
                    minutes_per_frame=mpf, params=params, timepoint_range=timepoint_range, progress_cb=_cb,
                )
                _save_cache(output_dir, target_type, sample_name, df_e, df_t, pt, record)
                log_fn(f"{get_current_time()} -   {sample_name} · {target_type}: {len(df_e)} death events")
            all_events.append(df_e)
            all_ts.append(df_t)
            patches[(sample_name, target_type)] = pt
    df_events = pd.concat([d for d in all_events if not d.empty], ignore_index=True) if any(
        not d.empty for d in all_events) else pd.DataFrame()
    df_ts = pd.concat([d for d in all_ts if not d.empty], ignore_index=True) if any(
        not d.empty for d in all_ts) else pd.DataFrame()
    return df_events, df_ts, patches, sample_info


# ---------------------------------------------------------------------------
# Single-frame preview (calibration aid)
# ---------------------------------------------------------------------------

def preview_new_death_patches(
    *,
    target_tracks_path: Union[str, Path],
    dead_mask_path: Union[str, Path],
    immune_tracks_paths: Dict[str, Union[str, Path]],
    t: int,
    voxel_spacing: Sequence[float],
    params: DeathEventParams,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Candidate new-death patches at frame ``t`` for calibrating the size floor.

    An **approximation** of the full detector for interactive use: novelty is
    measured against frame ``t - 1`` only (the detector uses the whole history)
    and there is no patch tracking, so a patch still growing from an earlier
    frame can show up here. Persistence (confirmation at ``t + 1``) and the
    effector cleaning are the same as in the detector.

    Returns ``(labels, table)``: a (Z, Y, X) uint16 volume with one label per
    candidate patch, and a table with each patch's volume and whether it passes
    ``min_patch_volume_um3`` (``passes_floor``).
    """
    from behav3d.io.images import open_image_timepoints
    from behav3d.features.timepoint_features import _zero_dead_mask_under_segments

    p = params.resolved(voxel_spacing)
    labels_h = open_image_timepoints(target_tracks_path)
    dead_h = open_image_timepoints(dead_mask_path)
    imm = {k: open_image_timepoints(v) for k, v in immune_tracks_paths.items()}
    n = int(min(labels_h.shape[0], dead_h.shape[0]))
    t = int(max(0, min(t, n - 1)))

    def dead_at(u):
        d = np.asarray(dead_h[u]) > 0
        if imm:
            d = _zero_dead_mask_under_segments(d, {k: np.asarray(h[u]) for k, h in imm.items()})
        return np.asarray(d, dtype=bool)

    org = np.asarray(labels_h[t]) > 0
    cur = dead_at(t) & org
    if t + 1 < n:
        cur &= _dilate_um(dead_at(t + 1), p.novelty_tolerance_um, voxel_spacing)
    prev = dead_at(t - 1) if t > 0 else np.zeros_like(cur)
    new = cur & ~_dilate_um(prev, p.novelty_tolerance_um, voxel_spacing)
    new = _close_um(new, p.patch_closing_um, voxel_spacing) & org
    comp, n_comp = ndimage.label(new, structure=ndimage.generate_binary_structure(3, 1))
    vox = float(np.prod(voxel_spacing))
    rows = []
    if n_comp:
        sizes = ndimage.sum_labels(np.ones_like(comp, dtype=np.int64), comp, index=np.arange(1, n_comp + 1))
        cents = ndimage.center_of_mass(new, comp, index=np.arange(1, n_comp + 1))
        lab_t = np.asarray(labels_h[t])
        for i, (sz, c) in enumerate(zip(sizes, cents), start=1):
            ci = tuple(int(round(v)) for v in c)
            rows.append({"patch": i, "volume_um3": float(sz) * vox, "n_voxels": int(sz),
                         "target_track_id": int(lab_t[ci]), "z_vox": c[0], "y_vox": c[1], "x_vox": c[2],
                         "passes_floor": float(sz) * vox >= p.min_patch_volume_um3})
    return comp.astype(np.uint16), pd.DataFrame(rows)

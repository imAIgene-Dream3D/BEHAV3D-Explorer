"""Shared helpers for reading, writing and annotating ``behav3d_parameters.yml``.

Every tab/panel in the plugin and the notebook widgets historically
re-implemented ``yaml.safe_dump(params, f, sort_keys=False)`` against a
locally rebuilt path.  New code should go through the helpers here so the
file name, the dump options (insertion order preserved, no key sorting)
and the provenance block layout stay in one place.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

PARAMS_FILENAME = "behav3d_parameters.yml"

# Axis order every converted zarr is written in (see RAW_TARGET_AXIS_ORDER
# in behav3d/io/images.py); recorded alongside the shapes so a reader knows
# how to interpret 'new_shape'.
ZARR_TARGET_AXIS_ORDER = "TCZYX"


def params_path_for(output_dir) -> Path:
    """Return ``<output_dir>/behav3d_parameters.yml``."""
    return Path(output_dir) / PARAMS_FILENAME


def load_params(output_dir) -> dict:
    """Read the parameters file for ``output_dir`` (``{}`` when absent)."""
    path = params_path_for(output_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_params(params: dict, output_dir) -> Path:
    """Write ``params`` to ``<output_dir>/behav3d_parameters.yml``.

    ``sort_keys=False`` keeps the dict's insertion order, which is what
    pins ``zarr_conversion`` to the top of the file (it is the first key
    of the ``_DEFAULT_CONFIG`` dicts every loader merges onto).
    """
    path = params_path_for(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return path


def _as_int_list(shape):
    """Normalise a shape to a plain list of ints (numpy ints don't dump)."""
    if shape is None:
        return None
    try:
        return [int(v) for v in shape]
    except TypeError:
        return None


def make_zarr_sample_record(
    original_shape=None,
    original_dimension_order=None,
    new_shape=None,
    mode="converted",
) -> dict:
    """Build one per-sample entry of the ``zarr_conversion.samples`` block.

    ``original_shape`` is in ``original_dimension_order`` (whatever the raw
    file used); ``new_shape`` is always TCZYX, since that is what the
    conversion writes.
    """
    original_shape = _as_int_list(original_shape)
    new_shape = _as_int_list(new_shape)

    original_n_t = None
    if original_shape and original_dimension_order and "T" in str(original_dimension_order):
        order = str(original_dimension_order)
        if len(order) == len(original_shape):
            original_n_t = original_shape[order.index("T")]

    return {
        "original_shape": original_shape,
        "original_dimension_order": (
            str(original_dimension_order) if original_dimension_order else None
        ),
        "original_n_timepoints": original_n_t,
        "new_shape": new_shape,
        "new_dimension_order": ZARR_TARGET_AXIS_ORDER if new_shape else None,
        "new_n_timepoints": new_shape[0] if new_shape else None,
        "mode": mode,
    }


def record_zarr_conversion(
    output_dir,
    t_start=None,
    t_end=None,
    sample_records: dict | None = None,
    params: dict | None = None,
    write: bool = True,
) -> dict:
    """Merge a zarr-conversion provenance block into the parameters file.

    Records how the images were cut so the cut stays recoverable after the
    original raw files are gone: the applied ``[t_start, t_end]`` window
    plus, per sample, the original shape/axis order and the resulting
    TCZYX shape.

    Per-sample entries are *merged* rather than replaced, so a single
    conversion run that first clips already-converted samples and then
    converts the remaining ones ends up with one coherent record.

    ``params`` is the caller's in-memory parameters dict; when given it is
    updated in place (and returned) so the caller stays in sync with disk.
    Pass ``write=False`` to update the dict without touching the file.
    """
    if params is None:
        params = load_params(output_dir)

    block = params.get("zarr_conversion")
    if not isinstance(block, dict):
        block = {}

    existing_samples = block.get("samples")
    if not isinstance(existing_samples, dict):
        existing_samples = {}
    for sample, record in (sample_records or {}).items():
        previous = existing_samples.get(sample)
        # Clipping an already-converted zarr only knows the pre-clip zarr
        # shape. If an earlier run recorded the true raw-file dimensions
        # for this sample, keep those as the 'original' rather than
        # overwriting them with an intermediate shape.
        if (
            record.get("mode") == "clipped_existing"
            and isinstance(previous, dict)
            and previous.get("original_shape")
        ):
            record = dict(record)
            for key in (
                "original_shape",
                "original_dimension_order",
                "original_n_timepoints",
            ):
                record[key] = previous.get(key)
        existing_samples[sample] = record

    clipped = t_start is not None or t_end is not None
    n_kept = None
    if clipped and t_start is not None and t_end is not None:
        n_kept = int(t_end) - int(t_start) + 1

    # Rebuild in a fixed key order so the block reads top-down as
    # "what was cut" then "what each sample became".
    params["zarr_conversion"] = {
        "clipped": bool(clipped),
        "t_start": int(t_start) if t_start is not None else None,
        "t_end": int(t_end) if t_end is not None else None,
        "n_timepoints_kept": n_kept,
        "converted_at": datetime.now().isoformat(timespec="seconds"),
        "samples": existing_samples,
    }

    if write:
        save_params(params, output_dir)
    return params


# ═══════════════════════════════════════════════════════════════════════
# Global timepoint range
# ═══════════════════════════════════════════════════════════════════════
# The analysis timepoint window is deliberately GLOBAL rather than
# per-cell-type: every cross-cell-type analysis (interaction, invasiveness,
# multi-organoid death dynamics, contact grouping) joins cell types on
# 'position_t' and silently produces wrong denominators, fabricated track
# extensions or dropped rows when two cell types cover different spans.
# Storing it once makes that impossible to get wrong.


def load_timepoint_range(output_dir, params: dict | None = None):
    """Return the global ``(start, end)`` window, or ``None`` when disabled.

    Bounds are inclusive absolute ``position_t`` values (frames).
    """
    if params is None:
        params = load_params(output_dir)
    block = params.get("timepoint_range")
    if not isinstance(block, dict) or not block.get("enabled"):
        return None
    start = block.get("start")
    end = block.get("end")
    if start is None and end is None:
        return None
    return (
        int(start) if start is not None else None,
        int(end) if end is not None else None,
    )


def save_timepoint_range(
    output_dir,
    enabled: bool,
    start=None,
    end=None,
    params: dict | None = None,
    write: bool = True,
) -> dict:
    """Write the global timepoint window into the parameters file.

    ``params`` is the caller's in-memory dict; when given it is updated in
    place (and returned) so the caller stays in sync with disk.
    """
    if params is None:
        params = load_params(output_dir)

    params["timepoint_range"] = {
        "enabled": bool(enabled),
        "start": int(start) if start is not None else None,
        "end": int(end) if end is not None else None,
    }

    if write and output_dir:
        save_params(params, output_dir)
    return params

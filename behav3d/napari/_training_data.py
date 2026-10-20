"""Qt-free builders for the pixel-classifier "Generate Training Data" payloads.

The three pixel-classifier pages (CPU Pixel Classifier, APOC, ConvPaint) used to
read, pad, stack and cache the training frames, compute contrast limits and
restore saved label / prediction zarrs *on the GUI thread*, interleaved with
``viewer.add_*`` calls -- which froze napari for the whole load and let a second
click crash it.

Each ``build_*_training_payload`` here does only the pure part (file I/O, numpy
and, for the CPU classifier, feature extraction) and returns a plain dict. It is
meant to run inside :class:`behav3d.napari._background_runner.BackgroundOperation`.
Everything that touches napari or Qt (clearing the viewer, adding layers, building
the training widgets, flipping session flags) stays in the page's ``on_done``,
which consumes the ``layers`` list of the payload.

``run_training_data_job`` and ``add_layer_specs`` are the thin GUI-side glue shared
by the three pages (they import Qt-aware modules lazily; the builders never do).

A *layer spec* is a dict::

    {"kind": "image" | "labels", "name": str, "data": ndarray,
     "kwargs": {...add_image/add_labels kwargs...},
     "tag": "channel" | "user_labels" | "dead_pred" | "plain",
     "contrast_limits_range": (lo, hi) | None}

``log`` must be thread-safe (the pages' ``self.log`` is); ``progress_cb`` is the
``(current, total, label)`` callback injected by ``BackgroundOperation``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Same palette the pages used inline.
APOC_CHANNEL_COLORS = [
    "cyan", "yellow", "red", "green", "magenta", "blue",
    "gray", "turbo", "viridis", "plasma", "inferno", "twilight",
]
RF_CHANNEL_COLORS = ["cyan", "yellow", "red", "green", "magenta", "blue", "gray"]


def _noop(*_args, **_kwargs) -> None:
    return None


# ---------------------------------------------------------------------------
# GUI-side glue (the only Qt-aware part; imports are lazy)
# ---------------------------------------------------------------------------
def add_layer_specs(viewer, specs, *, configure_user_labels=None, style_dead_pred=None):
    """Add ``specs`` to ``viewer`` in order. GUI thread only. Returns the new layers."""
    added = []
    for spec in specs:
        if spec["kind"] == "image":
            layer = viewer.add_image(spec["data"], name=spec["name"], **spec["kwargs"])
        else:
            layer = viewer.add_labels(spec["data"], name=spec["name"], **spec["kwargs"])
        rng = spec.get("contrast_limits_range")
        if rng is not None:
            layer.contrast_limits_range = rng
        tag = spec.get("tag")
        if tag == "user_labels" and configure_user_labels is not None:
            configure_user_labels(layer)
        elif tag == "dead_pred" and style_dead_pred is not None:
            style_dead_pred(layer)
        added.append(layer)
    return added


# ---------------------------------------------------------------------------
# Saving user labels
# ---------------------------------------------------------------------------
# Saving is split in two so the GUI never waits for disk: the page *snapshots* the
# label layers on the GUI thread (a plain ``copy``: the user may keep painting while
# the write runs, and tab exit removes the layers right after), and a worker
# *writes* the snapshot. A snapshot is a list of items::
#
#     {"label": "Dead labels", "path": Path, "data": ndarray}      # a label zarr
#     {"label": "label map",   "fn": callable}                      # extra best-effort write
def label_item(label, path, layer_data) -> Dict[str, Any]:
    """Snapshot item for one label layer. GUI thread: copies the layer's array."""
    return {"label": label, "path": Path(path), "data": np.array(layer_data, copy=True)}


def write_label_snapshot(items, log: Callable[[str], None], progress_cb: Optional[Callable] = None) -> int:
    """Write snapshot ``items`` to disk. Qt-free, safe on a worker thread.

    Returns the number of label zarrs written. ``log`` must be thread-safe.
    """
    from behav3d.core.utils import rmtree_ignore_missing
    from behav3d.io.images import save_as_zarr

    progress_cb = progress_cb or _noop
    written = 0
    n = len(items)
    for i, item in enumerate(items):
        progress_cb(i, n, f"Saving {item['label']}…")
        if "fn" in item:
            try:
                item["fn"]()
            except Exception:
                pass  # best-effort extras (e.g. the ConvPaint label map sidecar)
            continue
        path = item["path"]
        if path.exists():
            rmtree_ignore_missing(path)
        save_as_zarr(item["data"], path)
        log(f"Saved {item['label']} → {path.name}")
        written += 1
    if written:
        log("✅ All user labels saved.")
    return written


def start_label_save(bg, items, log: Callable[[str], None], *, progress_row=None, buttons=(),
                     viewer=None, on_done: Optional[Callable[[], None]] = None) -> bool:
    """Write the snapshot ``items`` on ``bg`` (a ``BackgroundOperation``).

    Returns ``True`` when the background save started. ``False`` means nothing was
    started (another job is running, or the global busy guard refused): the caller
    decides whether that matters -- tab exit falls back to :func:`write_label_snapshot`
    so labels are never silently dropped.
    """
    if bg.is_running():
        log("⚠️ Another job is running on this page; labels were not saved. Try again when it finishes.")
        return False

    def save_user_label_zarrs(progress_cb=None):
        write_label_snapshot(items, log, progress_cb)

    def _done(_result):
        if on_done is not None:
            on_done()

    def _failed(err):
        log(f"❌ Could not save user labels: {err}")

    bg.run(
        fn=save_user_label_zarrs,
        desc="Saving user labels…",
        progress_row=progress_row,
        buttons=list(buttons),
        viewer=viewer,
        on_done=_done,
        on_failed=_failed,
    )
    return bg.is_running()


def training_data_key(metadata_loader):
    """Identity of the dataset a load was started for: output dir + sample names.

    Compared again when the worker finishes; a mismatch means the user loaded
    other metadata meanwhile and the payload must not be applied.
    """
    md = getattr(metadata_loader, "metadata", None)
    samples = tuple(str(s) for s in md["sample_name"]) if md is not None and "sample_name" in md else ()
    return (str(getattr(metadata_loader, "output_dir", "")), samples)


def run_training_data_job(page, *, desc, builder, snap, apply, extra_callbacks=None,
                          block=False, buttons=(), stale_check=None,
                          failure_prefix="❌ Error during training data generation"):
    """Run ``builder(snap, log, progress_cb)`` off the GUI thread, then ``apply(payload)`` on it.

    ``page`` is one of the pixel-classifier widgets (``.log``, ``._bg``,
    ``.viewer``). ``apply`` returns ``False`` when nothing was applied (empty
    payload), in which case the queue hook gets ``on_failed``.

    ``block=True`` runs both phases inline (no thread) for synchronous callers.
    ``stale_check()`` returning True drops the payload (dataset changed meanwhile).
    Exceptions in ``apply`` are logged here: ``BackgroundOperation`` only prints
    exceptions raised by ``on_done`` callbacks to stderr.
    """
    import traceback

    from behav3d.napari._background_runner import fire_extra_callback

    log = page.log

    def _fail(msg: str) -> None:
        log(f"{failure_prefix}: {msg}")
        fire_extra_callback(extra_callbacks, "on_failed", str(msg))

    def _finish(payload) -> None:
        if stale_check is not None and stale_check():
            log("⚠️ Training data discarded: the dataset changed while it was loading. "
                "Click Generate Training Data again.")
            fire_extra_callback(extra_callbacks, "on_failed", "dataset changed while loading")
            return
        try:
            applied = apply(payload)
        except Exception as exc:
            traceback.print_exc()
            _fail(str(exc))
            return
        if applied is False:
            fire_extra_callback(extra_callbacks, "on_failed", "no training data")
            return
        fire_extra_callback(extra_callbacks, "on_done", None)

    def _work(progress_cb=None):
        return builder(snap, log, progress_cb)

    # Descriptive name: BackgroundOperation's crash log records ``fn.__name__``.
    _work.__name__ = getattr(builder, "__name__", "training_data_job")

    if block:
        try:
            payload = _work(None)
        except Exception as exc:
            traceback.print_exc()
            _fail(str(exc))
            return
        _finish(payload)
        return

    page._bg.run(
        fn=_work,
        desc=desc,
        progress_row=getattr(page, "tab_progress_row", None),
        buttons=list(buttons),
        viewer=getattr(page, "viewer", None),
        on_done=_finish,
        on_failed=_fail,
    )


def _percentile_clim(channel_data: np.ndarray, floor: float = 1e-3):
    """``(0, 99.8th percentile of the non-zero voxels)``, or ``(0, floor)`` when empty."""
    nonzero = channel_data[channel_data > 0]
    if nonzero.size > 0:
        return (0, float(np.percentile(nonzero, 99.8)))
    return (0, floor)


def _layer(kind: str, name: str, data: np.ndarray, tag: str = "plain",
           contrast_limits_range=None, **kwargs) -> Dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "data": data,
        "kwargs": kwargs,
        "tag": tag,
        "contrast_limits_range": contrast_limits_range,
    }


def _pad_and_stack(image_list: List[np.ndarray]) -> np.ndarray:
    """Zero-pad every ``(C, Z, Y, X)`` frame to the largest shape and stack along T."""
    from behav3d.preprocessing import zeropad_image_to_match_shape

    max_shape = list(image_list[0].shape)
    for img in image_list[1:]:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[i])
    image_list = [zeropad_image_to_match_shape(img, max_shape) for img in image_list]
    return np.stack(image_list, axis=0)  # (T_total, C, Z, Y, X)


# ---------------------------------------------------------------------------
# APOC
# ---------------------------------------------------------------------------
def build_apoc_training_payload(snap: Dict[str, Any], log: Callable[[str], None],
                                progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """Pure half of APOC's *Generate Training Data*.

    ``snap`` keys: ``metadata``, ``output_dir``, ``examples_per_sample``,
    ``organoid_types``, ``immune_types``, ``other_types``, ``load_existing``.
    """
    from behav3d.io.images import load_zarr
    from behav3d.preprocessing.segmentation.apoc_train import (
        _load_training_images, _predicted_labels_path, _probability_map_path,
    )

    progress_cb = progress_cb or _noop
    log("Running _load_training_images...")
    image_list, pixel_class_outdir, has_death, all_cell_types = _load_training_images(
        metadata=snap["metadata"],
        output_dir=str(snap["output_dir"]),
        examples_per_sample=snap["examples_per_sample"],
        organoid_types=snap["organoid_types"],
        immune_types=snap["immune_types"],
        other_types=snap["other_types"],
        overwrite_images=not snap["load_existing"],
        progress_cb=progress_cb,
    )
    if not image_list:
        return {"empty": "⚠️ No training images found!"}

    progress_cb(0, 1, "Padding and stacking images…")
    log("Padding and stacking images...")
    stacked = _pad_and_stack(image_list)
    T_total, n_channels = stacked.shape[0], stacked.shape[1]
    label_shape = (T_total,) + stacked.shape[2:]

    layers: List[Dict[str, Any]] = []
    for ch in range(n_channels):
        progress_cb(ch, n_channels, f"Preparing channel {ch}…")
        channel_data = stacked[:, ch, :, :, :]
        layers.append(_layer(
            "image", f"Channel {ch}", channel_data, tag="channel",
            contrast_limits_range=(0, max(float(channel_data.max()), 1e-3)),
            contrast_limits=_percentile_clim(channel_data),
            colormap=APOC_CHANNEL_COLORS[ch % len(APOC_CHANNEL_COLORS)],
            blending="additive", opacity=0.8,
        ))

    def _restore_labels(path: Path, ct_label: str):
        if path.exists():
            existing = np.asarray(load_zarr(path))
            if existing.shape == label_shape:
                log(f"  ↩ Restored saved labels for '{ct_label}'")
                return existing
            log(f"  ⚠️ Saved labels shape {existing.shape} ≠ expected {label_shape} — starting fresh")
        return np.zeros(label_shape, dtype=np.int16)

    for i, cell_type in enumerate(all_cell_types):
        progress_cb(i, len(all_cell_types), f"Restoring labels for {cell_type}…")
        saved = Path(pixel_class_outdir, f"PixelClassifier_User{cell_type.capitalize()}Labels.zarr")
        layers.append(_layer(
            "labels", f"User Provided Labels ({cell_type.capitalize()})",
            _restore_labels(saved, cell_type), tag="user_labels", opacity=0.5,
        ))

    if has_death:
        dead_path = Path(pixel_class_outdir, "PixelClassifier_UserDeadLabels.zarr")
        layers.append(_layer(
            "labels", "User Provided Labels (Dead)",
            _restore_labels(dead_path, "dead"), tag="user_labels", opacity=0.5,
        ))

    def _try_load(path: Optional[Path]):
        """Previously generated predictions are optional: ignore unreadable ones."""
        if not (path and path.exists()):
            return None
        try:
            arr = np.asarray(load_zarr(path))
        except Exception:
            return None
        return arr if arr.shape == label_shape else None

    for cell_type in all_cell_types:
        progress_cb(0, 0, f"Restoring predictions for {cell_type}…")
        pred = _try_load(_predicted_labels_path(pixel_class_outdir, cell_type))
        if pred is not None:
            layers.append(_layer(
                "labels", f"{cell_type.capitalize()} Segments", pred,
                opacity=0.8, visible=False,
            ))
            log(f"  ↩ Restored predicted labels for '{cell_type}'")
        prob = _try_load(_probability_map_path(pixel_class_outdir, cell_type))
        if prob is not None:
            layers.append(_layer(
                "image", f"Probability Map ({cell_type.capitalize()})", prob,
                opacity=0.6, blending="additive", colormap="magma",
                contrast_limits=(0.0, 1.0), visible=False,
            ))
            log(f"  ↩ Restored probability map for '{cell_type}'")

    if has_death:
        pred = _try_load(_predicted_labels_path(pixel_class_outdir, "dead"))
        if pred is not None:
            layers.append(_layer(
                "labels", "Pixel Classification (Dead)", pred, tag="dead_pred",
                opacity=0.8, visible=False,
            ))
            log("  ↩ Restored predicted labels for 'dead'")
        prob = _try_load(_probability_map_path(pixel_class_outdir, "dead"))
        if prob is not None:
            layers.append(_layer(
                "image", "Probability Map (Dead)", prob,
                opacity=0.6, blending="additive", colormap="magma",
                contrast_limits=(0.0, 1.0), visible=False,
            ))
            log("  ↩ Restored probability map for 'dead'")

    return {
        "layers": layers,
        "has_death": bool(has_death),
        "all_cell_types": list(all_cell_types),
        "pixel_class_outdir": Path(pixel_class_outdir),
    }


# ---------------------------------------------------------------------------
# ConvPaint
# ---------------------------------------------------------------------------
def build_convpaint_training_payload(snap: Dict[str, Any], log: Callable[[str], None],
                                     progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """Pure half of ConvPaint's *Generate Training Data*.

    ``snap`` keys: ``metadata``, ``output_dir``, ``examples_per_sample``,
    ``organoid_types``, ``immune_types``, ``other_types`` (already merge-filtered),
    ``overwrite_images``.
    """
    from behav3d.io.images import load_zarr
    from behav3d.preprocessing.segmentation.convpaint_label_map import (
        build_label_map, label_map_matches, load_label_map,
        unified_label_map_path, unified_predicted_labels_path,
        unified_user_labels_path,
    )
    from behav3d.preprocessing.segmentation.convpaint_train import (
        CHANNEL_COLORS, DEAD_LABELS_LAYER_NAME, DEAD_PREDICTED_LAYER_NAME,
        UNIFIED_LABELS_LAYER_NAME, UNIFIED_PREDICTED_LAYER_NAME,
        _derive_convpaint_input_channels, _filter_merge_types, _load_training_images,
        _predicted_labels_path, _probability_layer_name, _probability_map_path,
        _segments_layer_name,
    )

    progress_cb = progress_cb or _noop
    md = snap["metadata"]
    all_images, pixel_class_outdir, has_death, all_cell_types = _load_training_images(
        md, str(snap["output_dir"]), snap["examples_per_sample"],
        snap["organoid_types"], snap["immune_types"], snap["other_types"],
        overwrite_images=snap["overwrite_images"], progress_cb=progress_cb,
    )
    all_cell_types = _filter_merge_types(all_cell_types)
    base = {"all_cell_types": list(all_cell_types)}

    if not all_images:
        return {**base, "empty": "⚠️ No training images loaded."}
    if not all_cell_types:
        return {**base, "empty": "⚠️ No cell types detected — nothing to train."}

    n_channels = all_images[0].shape[0]
    unified_input_channels, death_input_channels = _derive_convpaint_input_channels(md, n_channels)
    log(f"Loaded {len(all_images)} training images with {n_channels} channels.")
    log(f"  Unified input channels: {unified_input_channels}")
    if has_death:
        log(f"  Death input channels: {death_input_channels}")

    progress_cb(0, 1, "Stacking images…")
    stacked = np.stack(all_images, axis=0)  # (T_total, C, Z, Y, X)
    T_total = stacked.shape[0]
    label_shape = (T_total,) + stacked.shape[2:]

    layers: List[Dict[str, Any]] = []
    for ch in range(n_channels):
        progress_cb(ch, n_channels, f"Preparing channel {ch}…")
        ch_data = stacked[:, ch, :, :, :]
        layers.append(_layer(
            "image", f"Channel {ch}", ch_data, tag="channel",
            contrast_limits_range=(0, float(ch_data.max())),
            contrast_limits=_percentile_clim(ch_data),
            colormap=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)],
            blending="additive", opacity=0.8,
        ))

    # Restore/create the unified User Provided Labels.
    label_map = build_label_map(all_cell_types)
    unified_path = unified_user_labels_path(pixel_class_outdir)
    saved_map_path = unified_label_map_path(pixel_class_outdir)
    unified_data = np.zeros(label_shape, dtype=np.int16)
    if unified_path.exists():
        try:
            existing = np.asarray(load_zarr(unified_path))
            if existing.shape == label_shape:
                if saved_map_path.exists():
                    saved_map = load_label_map(saved_map_path)
                    if label_map_matches(saved_map, all_cell_types):
                        unified_data = existing
                        log(f"  ↩ Restored unified labels ({unified_path.name})")
                    else:
                        log(
                            "  ⚠️ Cached unified labels were trained with a "
                            "different cell-type set — starting fresh."
                        )
                else:
                    unified_data = existing
                    log("  ↩ Restored unified labels (no map sidecar).")
            else:
                log(
                    f"  ⚠️ Unified labels shape mismatch ({existing.shape} vs "
                    f"{label_shape}); starting fresh."
                )
        except Exception as exc:
            log(f"  ⚠️ Could not restore unified labels: {exc}")
    layers.append(_layer("labels", UNIFIED_LABELS_LAYER_NAME, unified_data, opacity=0.5))

    if has_death:
        dead_path = Path(pixel_class_outdir, "PixelClassifier_UserDeadLabels.zarr")
        dead_data = np.zeros(label_shape, dtype=np.int16)
        if dead_path.exists():
            try:
                existing = np.asarray(load_zarr(dead_path))
                if existing.shape == label_shape:
                    dead_data = existing
                    log(f"  ↩ Restored death labels ({dead_path.name})")
            except Exception as exc:
                log(f"  ⚠️ Could not restore death labels: {exc}")
        layers.append(_layer("labels", DEAD_LABELS_LAYER_NAME, dead_data, opacity=0.5))

    def _try_load(path: Optional[Path]):
        if not (path and path.exists()):
            return None
        try:
            arr = np.asarray(load_zarr(path))
        except Exception:
            return None
        return arr if arr.shape == label_shape else None

    # Cached previews (unified predicted + per-cell-type seg/prob).
    pred_path = unified_predicted_labels_path(pixel_class_outdir)
    if pred_path.exists():
        try:
            pred = np.asarray(load_zarr(pred_path))
            if pred.shape == label_shape:
                layers.append(_layer(
                    "labels", UNIFIED_PREDICTED_LAYER_NAME, pred,
                    opacity=0.6, visible=False,
                ))
        except Exception as exc:
            log(f"  ⚠️ Could not restore unified predicted labels: {exc}")

    for i, ct in enumerate(all_cell_types):
        progress_cb(i, len(all_cell_types), f"Restoring previews for {ct}…")
        pred = _try_load(_predicted_labels_path(pixel_class_outdir, ct))
        if pred is not None:
            layers.append(_layer(
                "labels", _segments_layer_name(ct), pred, opacity=0.8, visible=False,
            ))
        prob = _try_load(_probability_map_path(pixel_class_outdir, ct))
        if prob is not None:
            layers.append(_layer(
                "image", _probability_layer_name(ct), prob,
                opacity=0.6, blending="additive", colormap="magma",
                contrast_limits=(0.0, 1.0), visible=False,
            ))

    if has_death:
        pred = _try_load(_predicted_labels_path(pixel_class_outdir, "dead"))
        if pred is not None:
            layers.append(_layer(
                "labels", DEAD_PREDICTED_LAYER_NAME, pred, tag="dead_pred",
                opacity=0.8, visible=False,
            ))

    return {
        **base,
        "layers": layers,
        "all_images": all_images,
        "has_death": bool(has_death),
        "pixel_class_outdir": Path(pixel_class_outdir),
        "n_channels": n_channels,
        "unified_input_channels": unified_input_channels,
        "death_input_channels": death_input_channels,
        "label_map": label_map,
    }


# ---------------------------------------------------------------------------
# CPU Pixel Classifier (random forest)
# ---------------------------------------------------------------------------
def build_rf_training_payload(snap: Dict[str, Any], log: Callable[[str], None],
                              progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """Pure half of the CPU pixel classifier's *Generate Training Data*.

    ``snap`` keys: ``metadata``, ``output_dir``, ``examples_per_sample``,
    ``load_existing``, ``exists`` (cache present *and* still valid),
    ``saved_examples``, ``all_cell_types``, ``has_death``.

    The returned ``param_updates`` (extractor version + pixel sizes) are written
    into ``behav3d_parameters`` by the caller on the GUI thread.
    """
    from behav3d.io.images import load_image, load_zarr, save_as_zarr
    from behav3d.preprocessing import zeropad_image_to_match_shape
    from behav3d.preprocessing.segmentation.napari_pixelclassifier import make_features

    progress_cb = progress_cb or _noop
    metadata = snap["metadata"]
    output_dir = Path(snap["output_dir"])
    pixel_class_outdir = output_dir / "images" / "PixelClassification"
    pixel_class_outdir.mkdir(parents=True, exist_ok=True)
    features_outpath = pixel_class_outdir / "PixelClassifier_Features.zarr"
    image_outpath = pixel_class_outdir / "PixelClassifier_Images.zarr"
    examples_per_sample = snap["examples_per_sample"]
    saved_examples = snap["saved_examples"]
    exists = snap["exists"]
    all_cell_types = list(snap["all_cell_types"])
    has_death = bool(snap["has_death"])
    param_updates: Dict[str, Any] = {}

    if snap["load_existing"]:
        # --- FAST PATH: load from cached zarr files ---
        log(f"Loading cached images and features from disk ({saved_examples} examples/sample)...")
        progress_cb(0, 2, "Loading cached images…")
        stacked_images = np.asarray(load_image(image_outpath))
        progress_cb(1, 2, "Loading cached features…")
        stacked_features = np.asarray(load_image(features_outpath))
        log(f"  Cached images shape: {stacked_images.shape}")
        log(f"  Cached features shape: {stacked_features.shape}")
    else:
        # --- SLOW PATH: regenerate images and features ---
        if exists:
            log("Generating new training data (overwriting)...")
        elif saved_examples is not None and saved_examples != examples_per_sample:
            log(f"examples_per_sample changed ({saved_examples} → {examples_per_sample}), regenerating...")
        else:
            log("No cached data found, generating images and features...")

        all_images = []
        log(f"Checking {len(metadata)} samples...")
        n_rows = len(metadata)
        for row_i, (_, sample) in enumerate(metadata.iterrows()):
            sample_name = sample["sample_name"]
            raw_image_path = Path(sample["raw_image_path"])
            progress_cb(row_i, n_rows, f"Reading {sample_name}…")

            if not raw_image_path.exists():
                log(f"Error: Raw image path not found: {raw_image_path}")
                continue

            raw_image_zarr = output_dir / "images" / sample_name / f"{sample_name}.zarr"
            if not raw_image_zarr.exists():
                log(f"- Converting {sample_name} to .zarr...")
                try:
                    images = load_image(raw_image_path)
                    save_as_zarr(img=images, path=raw_image_zarr)
                except Exception as e:
                    log(f"Error loading/converting {sample_name}: {e}")
                    continue

            try:
                images = load_image(raw_image_zarr)
                if len(images.shape) == 5:
                    max_t = images.shape[0] - 1
                elif len(images.shape) == 4:
                    log(f"Warning: Image {sample_name} has shape {images.shape}. Assuming T,C,Z,Y,X logic might fail.")
                    max_t = images.shape[0] - 1
                else:
                    log(f"Warning: unexpected shape {images.shape}")
                    max_t = 0
                idc = np.unique(np.linspace(0, max_t, examples_per_sample, dtype=int))
                log(f"  {sample_name}: Taking timepoints {idc}")
                all_images += [images[t] for t in idc]
            except Exception as e:
                log(f"Error reading {raw_image_zarr}: {e}")

        if not all_images:
            return {"empty": "No images loaded."}

        max_shape = [0] * all_images[0].ndim
        for img in all_images:
            for i in range(img.ndim):
                max_shape[i] = max(max_shape[i], img.shape[i])
        max_shape = tuple(max_shape)
        log(f"Max shape for padding: {max_shape}")

        # Remove old cached files, labels, and predictions before regenerating.
        log("Cleaning up old training files, labels and predictions...")
        to_delete = [
            image_outpath,
            features_outpath,
            pixel_class_outdir / "PixelClassifier_UserDeadLabels.zarr",
            pixel_class_outdir / "PixelClassifier_Death_PredictedLabels.zarr",
        ]
        for ct in all_cell_types:
            to_delete.append(pixel_class_outdir / f"PixelClassifier_User{ct.capitalize()}Labels.zarr")
            to_delete.append(pixel_class_outdir / f"PixelClassifier_{ct.capitalize()}_PredictedLabels.zarr")
        for p in to_delete:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

        log("Calculating features (this may take a while)...")
        # Build the extractor from the first sample's pixel sizes so sigma values
        # are in physical units -- matching batch inference.
        pixel_xy = float(metadata["pixel_distance_xy"].iloc[0])
        pixel_z = float(metadata["pixel_distance_z"].iloc[0])
        extract_features = make_features(pixel_xy, pixel_z, verbose=True)
        log(
            f"Feature extractor: make_features "
            f"(pixel_xy={pixel_xy:.4f} µm, pixel_z={pixel_z:.4f} µm)"
        )

        processed_images, processed_features = [], []
        for i, img in enumerate(all_images):
            progress_cb(i, len(all_images), f"Computing features {i + 1}/{len(all_images)}…")
            log(f"Processing image {i + 1}/{len(all_images)}")
            try:
                feats = extract_features(img)
                processed_images.append(zeropad_image_to_match_shape(img, max_shape))
                processed_features.append(feats)
            except Exception as e:
                log(f"Error calculating features: {e}")

        if not processed_images:
            return {"empty": "No images could be processed."}

        stacked_images = np.stack(processed_images)

        feat_max_shape = [0] * processed_features[0].ndim
        for f in processed_features:
            for i in range(f.ndim):
                feat_max_shape[i] = max(feat_max_shape[i], f.shape[i])
        feat_max_shape = tuple(feat_max_shape)
        stacked_features = np.stack(
            [zeropad_image_to_match_shape(f, feat_max_shape) for f in processed_features]
        )

        progress_cb(0, 2, "Saving images…")
        save_as_zarr(stacked_images, image_outpath)
        progress_cb(1, 2, "Saving features…")
        save_as_zarr(stacked_features, features_outpath)

        # Cache-invalidation marker + the pixel sizes used (applied by the caller).
        param_updates = {
            "features_extractor_version": "make_features_v1",
            "pixel_size_xy": pixel_xy,
            "pixel_size_z": pixel_z,
        }
        log("Images and features saved to disk.")

    # --- Layer specs ---
    log("Adding layers to viewer...")
    dims = stacked_images.shape  # (T, C, Z, Y, X)
    log(f"  Images shape: {dims}")
    if len(dims) == 5:
        label_dims = (dims[0], dims[2], dims[3], dims[4])
    elif len(dims) == 4:
        label_dims = (dims[0], 1, dims[2], dims[3])  # (T, C, Y, X) assumed
        log("  Warning: 4D images detected, assuming (T, C, Y, X)")
    else:
        label_dims = dims
    log(f"  Labels shape: {label_dims}")

    layers: List[Dict[str, Any]] = []
    n_channels = stacked_images.shape[1]
    for ch in range(n_channels):
        progress_cb(ch, n_channels, f"Preparing channel {ch + 1}…")
        channel_data = stacked_images[:, ch, :, :, :]
        kwargs = dict(
            colormap=RF_CHANNEL_COLORS[ch % len(RF_CHANNEL_COLORS)],
            blending="additive", visible=True,
        )
        try:
            flat = channel_data[channel_data > 0]
            if flat.size > 0:
                kwargs["contrast_limits"] = (0, np.percentile(flat, 99.8))
        except Exception:
            pass
        layers.append(_layer("image", f"Channel {ch + 1}", channel_data, tag="channel", **kwargs))

    def _load_or_zeros(path: Path, what: str):
        if path.exists():
            log(f"  Loading existing {what} from disk...")
            arr = np.asarray(load_zarr(path))
            if arr.shape == label_dims:
                return arr
            log(f"  Warning: {what} shape mismatch {arr.shape} vs {label_dims}. Using zeros.")
        return np.zeros(label_dims, dtype=np.uint16)

    if has_death:
        dead = _load_or_zeros(
            pixel_class_outdir / "PixelClassifier_UserDeadLabels.zarr", "dead labels",
        )
        layers.append(_layer("labels", "User Provided Labels (Dead)", dead,
                             tag="user_labels", opacity=0.3))
    for cell_type in all_cell_types:
        ul = _load_or_zeros(
            pixel_class_outdir / f"PixelClassifier_User{cell_type.capitalize()}Labels.zarr",
            f"{cell_type} user labels",
        )
        layers.append(_layer("labels", f"User Provided Labels ({cell_type.capitalize()})", ul,
                             tag="user_labels", opacity=0.3))

    if has_death:
        pd_ = _load_or_zeros(
            pixel_class_outdir / "PixelClassifier_Death_PredictedLabels.zarr", "dead predictions",
        )
        layers.append(_layer("labels", "Pixel Classification (Dead)", pd_,
                             tag="dead_pred", opacity=0.3, visible=False))
    for cell_type in all_cell_types:
        pr = _load_or_zeros(
            pixel_class_outdir / f"PixelClassifier_{cell_type.capitalize()}_PredictedLabels.zarr",
            f"{cell_type} predictions",
        )
        layers.append(_layer("labels", f"Pixel Classification ({cell_type.capitalize()})", pr,
                             opacity=0.3, visible=False))

    for cell_type in all_cell_types:
        layers.append(_layer(
            "labels", f"{cell_type.capitalize()} Segments",
            np.zeros(label_dims, dtype=np.uint16), opacity=0.7, visible=False,
        ))

    return {
        "layers": layers,
        "stacked_images": stacked_images,
        "stacked_features": stacked_features,
        "param_updates": param_updates,
        "pixel_class_outdir": pixel_class_outdir,
    }

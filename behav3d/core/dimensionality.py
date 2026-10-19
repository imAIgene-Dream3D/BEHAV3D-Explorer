"""Experiment-level image dimensionality (2D vs 3D).

BEHAV3D was built for 3D time-lapses and 3D stays the default.  An
experiment can instead be declared 2D in Data Preparation; the choice is
stored once per experiment, both as the top-level ``dimensionality`` key in
``behav3d_parameters.yml`` and as a ``dimensionality`` column (identical in
every row) in the metadata CSV, so code that only sees a metadata row can
still read it.

Accepted raw layouts (the user's dimension order says which axis is which):

====  ========  ============  ==========================================
Mode  Raw ndim  Layout        Conversion
====  ========  ============  ==========================================
3D    4         TZYX          singleton C inserted
3D    5         TCZYX         as is
2D    3         TYX           singleton C inserted
2D    4         TCYX          as is
====  ========  ============  ==========================================

On disk nothing changes shape: a 2D image is still written as a 5D TCZYX
zarr with Z=1 (labels TZYX with Z=1, ``position_z == 0``).  The flag
switches algorithms and the UI, not array layouts.

A project that records no mode (for example a hand-written metadata CSV) is 3D.
"""
from __future__ import annotations

DIM_2D = "2D"
DIM_3D = "3D"
DEFAULT_DIMENSIONALITY = DIM_3D
DIMENSIONALITIES = (DIM_3D, DIM_2D)

# Name of both the YAML key and the metadata CSV column.
DIMENSIONALITY_KEY = "dimensionality"

# Pixel Z written for 2D experiments: keeps every z * spacing product finite
# while position_z stays 0.
PIXEL_DISTANCE_Z_2D = 1.0

# Dimension orders offered per mode, grouped by the raw image ndim.
_DIM_ORDERS = {
    DIM_3D: {
        5: ["TCZYX", "TZCYX", "ZCTYX", "ZTCYX", "CZTYX", "CTZYX"],
        4: ["TZYX", "ZTYX"],
    },
    DIM_2D: {
        4: ["TCYX", "CTYX"],
        3: ["TYX"],
    },
}

# Some readers (LIF, IMS, CZI) always report a 5D TCZYX shape, with Z=1 for a
# 2D acquisition. In 2D mode those files are offered the 5D orders and are
# accepted only when the Z axis really has size 1.
_ACCEPTED_NDIM = {DIM_3D: (4, 5), DIM_2D: (3, 4)}


def normalize_dimensionality(value) -> str | None:
    """Return ``"2D"``/``"3D"`` for any reasonable spelling, else ``None``."""
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    text = str(value).strip().upper()
    if text in ("2D", "2"):
        return DIM_2D
    if text in ("3D", "3"):
        return DIM_3D
    return None


def _from_params(params) -> str | None:
    if isinstance(params, dict):
        return normalize_dimensionality(params.get(DIMENSIONALITY_KEY))
    return None


def _from_metadata(metadata) -> str | None:
    """Read the CSV column from a DataFrame (first filled row) or a row."""
    if metadata is None:
        return None
    columns = getattr(metadata, "columns", None)
    if columns is not None:
        if DIMENSIONALITY_KEY not in columns:
            return None
        for value in metadata[DIMENSIONALITY_KEY].tolist():
            mode = normalize_dimensionality(value)
            if mode:
                return mode
        return None
    getter = getattr(metadata, "get", None)
    if callable(getter):
        return normalize_dimensionality(getter(DIMENSIONALITY_KEY))
    return None


def get_dimensionality(source=None, default: str = DEFAULT_DIMENSIONALITY) -> str:
    """Return the experiment dimensionality from whatever the caller holds.

    ``source`` may be:

    * a parameters dict (``behav3d_parameters``),
    * a metadata DataFrame or a single metadata row (Series / dict),
    * a metadata loader (e.g. the napari ``DataPreparationTab``), read via
      its ``dimensionality`` attribute, then its parameters, then its
      metadata.

    Anything without the flag falls back to ``default`` (3D).
    """
    if source is None:
        return default
    if isinstance(source, str):
        return normalize_dimensionality(source) or default
    if isinstance(source, dict):
        mode = _from_params(source) or _from_metadata(source)
        return mode or default
    if hasattr(source, "columns") or hasattr(source, "index"):
        return _from_metadata(source) or default

    mode = normalize_dimensionality(getattr(source, DIMENSIONALITY_KEY, None))
    if mode:
        return mode
    mode = _from_params(getattr(source, "behav3d_parameters", None))
    if mode:
        return mode
    try:
        mode = _from_metadata(getattr(source, "metadata", None))
    except Exception:
        mode = None
    return mode or default


def is_2d(source=None) -> bool:
    """True when ``source`` declares a 2D experiment (see get_dimensionality)."""
    return get_dimensionality(source) == DIM_2D


def logical_dim_order(mode: str) -> str:
    """Axis order of a converted image as the user thinks of it."""
    return "TCYX" if normalize_dimensionality(mode) == DIM_2D else "TCZYX"


def allowed_dim_orders(mode: str, ndim: int | None = None) -> list[str]:
    """Dimension orders the user can pick for ``mode``.

    With ``ndim`` the list is limited to orders of that length. A 2D file
    whose reader reports 5 dimensions (LIF/IMS/CZI with Z=1) gets the 5D
    orders; validation then checks that Z really is 1.
    """
    mode = normalize_dimensionality(mode) or DEFAULT_DIMENSIONALITY
    table = _DIM_ORDERS[mode]
    if ndim is None:
        return [o for n in sorted(table, reverse=True) for o in table[n]]
    if mode == DIM_2D and ndim == 5:
        return list(_DIM_ORDERS[DIM_3D][5])
    return list(table.get(int(ndim), []))


def validate_shape_for_mode(mode: str, shape) -> str | None:
    """Return an error when an image with ``shape`` can never fit ``mode``.

    2D accepts 3 or 4 dimensions (plus 5 when a reader reports a singleton
    Z, which only the dimension order can confirm); 3D accepts 4 or 5.
    """
    mode = normalize_dimensionality(mode) or DEFAULT_DIMENSIONALITY
    try:
        shape = tuple(int(s) for s in shape)
    except (TypeError, ValueError):
        return None
    ndim = len(shape)
    accepted = _ACCEPTED_NDIM[mode] + ((5,) if mode == DIM_2D else ())
    if ndim in accepted:
        return None
    if mode == DIM_2D:
        return (
            f"Image has {ndim} dimensions {shape}; 2D mode accepts "
            "3 (TYX, single channel) or 4 (TCYX)."
        )
    return (
        f"Image has {ndim} dimensions {shape}; 3D mode accepts "
        "4 (TZYX, single channel) or 5 (TCZYX)."
    )


def validate_order_for_mode(order, mode: str, shape=None) -> str | None:
    """Return a human-readable error, or ``None`` when ``order`` fits ``mode``.

    ``shape`` is the raw image shape; without it only the order itself is
    checked. In 2D an order with Z is only valid when that axis has size 1
    (LIF/IMS/CZI report 2D images that way), so without a shape it is not
    rejected yet.
    """
    mode = normalize_dimensionality(mode) or DEFAULT_DIMENSIONALITY
    order = str(order or "").strip().upper()
    if not order or order.startswith("--"):
        return "No dimension order selected."
    bad = [ax for ax in order if ax not in "TCZYX"]
    if "\\" in order or "/" in order or order.endswith((".CZI", ".TIF", ".TIFF", ".LIF", ".IMS", ".ZARR")):
        return (
            "The dimension order is a file path. Put the axes here "
            "(e.g. TYX or TCYX for 2D) and the file path in 'Image path'."
        )
    if bad:
        return f"Order '{order}' contains unsupported axes {bad}."
    if len(set(order)) != len(order):
        return f"Order '{order}' has duplicate axes."
    for ax in "TYX":
        if ax not in order:
            return f"Order '{order}' has no {ax} axis (T, Y and X are required)."

    ndim = None
    if shape is not None:
        try:
            shape = tuple(int(s) for s in shape)
            ndim = len(shape)
        except (TypeError, ValueError):
            shape = None

    if ndim is not None and len(order) != ndim:
        return (
            f"Order '{order}' has {len(order)} axes but the image has "
            f"{ndim} dimensions {shape}."
        )

    if mode == DIM_3D:
        if "Z" not in order:
            return (
                f"3D mode needs a Z axis, but '{order}' has none. "
                "If this is 2D data, switch the experiment to 2D."
            )
        if ndim is not None and ndim not in _ACCEPTED_NDIM[DIM_3D]:
            return (
                f"Image has {ndim} dimensions {shape}; 3D mode accepts "
                "4 (TZYX, single channel) or 5 (TCZYX)."
            )
        return None

    # 2D
    if "Z" in order:
        if shape is not None and ndim == len(order):
            z_size = shape[order.index("Z")]
            if z_size == 1:
                return None  # reader-reported singleton Z (LIF/IMS/CZI)
            return (
                f"2D mode selected but the image has Z={z_size} slices "
                f"({ndim} dimensions {shape}). Switch the experiment to 3D, "
                "or pick an order without Z."
            )
        return None
    if ndim is not None and ndim not in _ACCEPTED_NDIM[DIM_2D]:
        return (
            f"Image has {ndim} dimensions {shape}; 2D mode accepts "
            "3 (TYX, single channel) or 4 (TCYX)."
        )
    return None


def is_dimensionality_locked(params=None, metadata=None) -> bool:
    """True once any raw image has been converted to zarr.

    The provenance block ``zarr_conversion.samples`` is the primary signal;
    a ``raw_image_path`` already pointing to a ``.zarr`` also counts (for
    example images converted outside the app).
    """
    if isinstance(params, dict):
        block = params.get("zarr_conversion")
        if isinstance(block, dict) and block.get("samples"):
            return True
    columns = getattr(metadata, "columns", None)
    if columns is not None and "raw_image_path" in columns:
        for value in metadata["raw_image_path"].tolist():
            text = str(value).strip().strip('"').strip("'").lower()
            if text.endswith(".zarr") or text.endswith(".zarr.zip"):
                return True
    return False

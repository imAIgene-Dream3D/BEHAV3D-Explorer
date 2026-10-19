"""Which centroid coordinates the trackers link on."""
from __future__ import annotations


def linking_position_columns(df, columns=("position_z", "position_y", "position_x")) -> list:
    """Return the coordinate columns to link objects on.

    2D experiments are stored with Z=1, so every centroid has z = 0; z is
    then dropped and the trackers link in Y/X only. The z column itself
    stays in the output CSV (``pixel_position_z == 0``) for downstream
    readers. A 3D stack only loses z when every object sits at the same z,
    where it carries no information anyway.
    """
    z_col = columns[0]
    if z_col in df.columns and df[z_col].nunique(dropna=False) <= 1:
        return list(columns[1:])
    return list(columns)

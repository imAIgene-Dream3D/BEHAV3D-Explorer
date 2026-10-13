"""Re-link absolute paths stored in a BEHAV3D project after it was moved.

``metadata.csv`` (and a few ``.uns`` entries in h5ad files) store absolute
paths such as ``D:/runs/projA/images/S1/S1_tcell_tracked.zarr``. When the
output folder is copied to another drive or machine those paths go stale even
though the same file exists at ``<new output dir>/images/S1/...``.

Everything BEHAV3D writes lives below one of a few top-level folders of the
output directory (``images``, ``trackdata``, ``analysis``), so a stale path
can be mapped onto the new output directory by locating that anchor folder in
it. Stdlib only: this module is imported on the metadata-load path.
"""
from __future__ import annotations

import re
from pathlib import Path

ANCHOR_DIRS = ("images", "trackdata", "analysis")


def split_path_parts(path) -> list[str]:
    """Split a path written on any OS into its components.

    Handles both separators, so a path stored on Linux/macOS (``/mnt/nas/x``)
    or Windows (``D:\\x``, ``\\\\server\\share\\x``) splits the same way here.
    """
    text = str(path).strip().strip('"').strip("'")
    return [p for p in re.split(r"[\\/]+", text) if p]


def _same_prefix(parts, prefix) -> bool:
    if len(parts) < len(prefix):
        return False
    return all(a.casefold() == b.casefold() for a, b in zip(parts, prefix))


def is_under(path, root) -> bool:
    """True when *path* is *root* or lies below it (separator/case-insensitive)."""
    return _same_prefix(split_path_parts(path), split_path_parts(root))


def find_anchor_tail(path, new_root, anchors=ANCHOR_DIRS):
    """Return ``(old_root_parts, new_path)`` for a path whose tail exists under *new_root*.

    Scans from the end for the last anchor folder such that
    ``new_root/<anchor>/<rest>`` exists, so a sample that happens to be
    called "images" does not confuse it. Returns ``None`` when no tail of
    *path* exists under *new_root*.
    """
    parts = split_path_parts(path)
    anchor_set = {a.casefold() for a in anchors}
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].casefold() in anchor_set:
            candidate = Path(new_root, *parts[i:])
            if candidate.exists():
                return parts[:i], candidate
    return None


def rebase_onto(path, old_root_parts, new_root):
    """Swap the *old_root_parts* prefix of *path* for *new_root*; ``None`` if not under it."""
    parts = split_path_parts(path)
    if not old_root_parts or not _same_prefix(parts, old_root_parts) or len(parts) == len(old_root_parts):
        return None
    return Path(new_root, *parts[len(old_root_parts):])


def relocate_path(path, new_root, old_root=None):
    """Best-effort location of a stored project path inside *new_root*.

    Returns the path unchanged when it exists, otherwise the matching path
    below *new_root* (via *old_root* when known, else an anchor folder), or
    ``None`` when nothing matching exists.
    """
    if path is None or str(path).strip() == "":
        return None
    p = Path(str(path).strip().strip('"').strip("'"))
    if p.exists():
        return p
    if not new_root:
        return None
    if old_root:
        candidate = rebase_onto(p, split_path_parts(old_root), new_root)
        if candidate is not None and candidate.exists():
            return candidate
    found = find_anchor_tail(p, new_root)
    return found[1] if found else None


def join_parts_for_display(parts) -> str:
    """Render split parts back into a readable path string."""
    if not parts:
        return ""
    head = parts[0]
    if re.fullmatch(r"[A-Za-z]:", head):
        return head + "\\" + "\\".join(parts[1:])
    return "/" + "/".join(parts)

"""Pack a BEHAV3D output folder into a few large archives, verify them, unpack them.

Directory-backed zarr stores hold one small file per timepoint, each three
or four folders deep, so a project is tens of thousands of filesystem
entries. Copying those to a USB drive, an SMB share or a sync client is
dominated by per-entry overhead. A *bundle* replaces them with one archive
per sample plus one for the rest of the project, written straight to the
destination:

    <project>_behav3d_bundle/
        bundle.json         what is inside, source output dir, archive hashes
        SHA256SUMS.txt      same hashes, checkable without BEHAV3D
        project.zip         metadata, parameters, analysis/, PixelClassification, ...
        sample_<name>.zip   images/<name>/ and trackdata/<name>/

Archives are plain ZIP64 files whose member paths are relative to the
output directory, so extracting them all into one folder (with any unzip
tool) rebuilds the project; paths in metadata.csv are re-linked on load.
Zarr chunks are stored uncompressed (they are zstd-compressed already),
text files are deflated. Every archive carries a manifest with the SHA-256
of each member; unpacking checks every file against it before anything is
moved into place.

Command line::

    python -m behav3d.io.transfer pack   <output_dir> <destination> [--metadata CSV] [--samples A B]
    python -m behav3d.io.transfer verify <bundle_dir> [--deep]
    python -m behav3d.io.transfer unpack <bundle_dir> <new_output_dir> [--overwrite]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BUNDLE_SUFFIX = "_behav3d_bundle"
BUNDLE_JSON = "bundle.json"
SUMS_FILE = "SHA256SUMS.txt"
MANIFEST_NAME = ".behav3d_manifest.json"
PROJECT_ARCHIVE = "project.zip"
SAMPLE_PREFIX = "sample_"
IMPORT_TMP_SUFFIX = ".b3dtmp"
BUNDLE_FORMAT = 1

SAMPLE_TOP_DIRS = ("images", "trackdata")
TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".yml", ".yaml", ".txt", ".log", ".md"}
EXCLUDE_FILE_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
EXCLUDE_DIR_NAMES = {"__pycache__"}
INTERRUPTED_MARKER = ".seg_processing"

_BLOCK = 8 * 1024 * 1024


class TransferError(Exception):
    """A transfer step failed in a way the user has to act on."""


class TransferCancelled(TransferError):
    """The user cancelled the operation."""


class InterruptedRunError(TransferError):
    """A sample has a ``.seg_processing`` marker; pass ``force=True`` to pack anyway."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Progress:
    """Throttled ``progress(done, total, message)`` + cancel checks.

    *done* / *total* are units of I/O work, not file sizes: a pack writes,
    re-reads and hashes every byte, so *total* is a multiple of the data size.
    Show them as a fraction, never as a size. ``phase`` names the current pass.
    """

    def __init__(self, total, callback=None, cancel=None, interval=0.2):
        self.total = int(total)
        self.done = 0
        self._callback = callback
        self._cancel = cancel
        self._interval = interval
        self._last = 0.0
        self.message = ""
        self.phase = ""

    def check_cancel(self):
        if self._cancel is not None and self._cancel():
            raise TransferCancelled("Cancelled by user")

    def advance(self, n, message=None, force=False):
        self.done += int(n)
        if message is not None:
            self.message = message
        now = time.monotonic()
        if self._callback is not None and (force or now - self._last >= self._interval):
            self._last = now
            text = f"{self.phase}: {self.message}" if self.phase and self.message else (self.phase or self.message)
            self._callback(self.done, self.total, text)


def _behav3d_version():
    try:
        from importlib.metadata import version
        return version("behav3d")
    except Exception:
        return "unknown"


def _sha256_file(path, progress=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if progress is not None:
                progress.check_cancel()
            block = f.read(_BLOCK)
            if not block:
                break
            h.update(block)
            if progress is not None:
                progress.advance(len(block))
    return h.hexdigest()


def _write_text_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_excluded_file(name):
    low = name.lower()
    return low in EXCLUDE_FILE_NAMES or low.endswith(".partial") or "_clip_tmp" in low


def _is_excluded_dir(name):
    return name in EXCLUDE_DIR_NAMES or name.endswith(IMPORT_TMP_SUFFIX) or "_clip_tmp" in name.lower()


def _lp(path):
    """Path usable beyond Windows' 260-character limit (no-op elsewhere).

    Nested zarr chunk paths get long; the destination folder can be deeper
    than the one they were written in.
    """
    path = Path(path)
    if os.name != "nt":
        return path
    text = str(path.absolute())
    if len(text) < 240 or text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def _safe_member_path(rel):
    """Reject absolute paths and ``..`` in archive member names (zip-slip)."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise TransferError(f"Unsafe path in archive: {rel!r}")
    return p


def _same_or_inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _archive_name_for_sample(sample):
    return f"{SAMPLE_PREFIX}{sample}.zip"


def _fingerprint(entries):
    """Hash of (path, size, mtime) of an archive's sources: detects changes cheaply."""
    h = hashlib.sha256()
    for _abs, rel, size, mtime_ns in sorted(entries, key=lambda e: e[1]):
        h.update(f"{rel}\0{size}\0{mtime_ns}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def _read_metadata(metadata_csv, output_dir):
    """Return (sample names, folder owners, external paths) from a metadata CSV.

    *folder owners* maps ``(top, folder)`` - e.g. ``("images", "SOrg_x")`` - to
    the sample whose ``*_path`` columns point into it: the raw zarr folder is
    named after the raw file, not the sample. Folders claimed by two samples
    are left to the project archive.
    """
    if metadata_csv is None or not Path(metadata_csv).exists():
        return None, {}, []
    from behav3d.core.metadata import load_behav3d_metadata
    from behav3d.core.portable_paths import is_under, split_path_parts

    md = load_behav3d_metadata(metadata_csv)
    if "sample_name" not in md.columns:
        return [], {}, []
    samples, owners, shared, external = [], {}, set(), []
    path_cols = [c for c in md.columns if str(c).endswith("_path")]
    for _idx, row in md.iterrows():
        if row.isna()["sample_name"]:
            continue
        sample = str(row["sample_name"])
        samples.append(sample)
        for col in path_cols:
            value = row[col]
            if value is None or (isinstance(value, float) and value != value):
                continue
            text = str(value).strip().strip('"').strip("'")
            if not text or text.lower() == "nan":
                continue
            if not is_under(text, output_dir):
                external.append(text)
            parts = split_path_parts(text)
            for i in range(len(parts) - 2, -1, -1):
                if parts[i] in SAMPLE_TOP_DIRS:
                    key = (parts[i], parts[i + 1])
                    if owners.get(key, sample) != sample:
                        shared.add(key)
                    owners.setdefault(key, sample)
                    break
    for key in shared:
        owners.pop(key, None)
    return samples, owners, sorted(set(external))


def plan_bundle(output_dir, metadata_csv=None, samples=None):
    """Work out which file goes into which archive.

    Returns a dict with ``archives`` ({name: {"files": [(abs, rel, size, mtime_ns)],
    "dirs": [rel], "samples": [..]}}), ``total_bytes``, ``metadata_csv`` (relative
    name inside the bundle), ``external_paths``, ``interrupted`` (samples with a
    ``.seg_processing`` marker) and ``all_samples``.
    """
    output_dir = Path(output_dir).absolute()
    if not output_dir.is_dir():
        raise TransferError(f"Output folder not found: {output_dir}")

    if metadata_csv is None and (output_dir / "metadata.csv").exists():
        metadata_csv = output_dir / "metadata.csv"
    md_samples, owners, external = _read_metadata(metadata_csv, output_dir)

    if md_samples is None:
        # No metadata: every images/<x> folder except PixelClassification is a sample.
        md_samples = sorted(
            {d.name for top in SAMPLE_TOP_DIRS if (output_dir / top).is_dir()
             for d in (output_dir / top).iterdir()
             if d.is_dir() and d.name != "PixelClassification"}
        )
    all_samples = list(dict.fromkeys(md_samples))
    sample_set = set(all_samples)
    selected = set(all_samples if samples is None else samples)
    unknown = selected - sample_set
    if unknown:
        raise TransferError(f"Unknown sample(s): {', '.join(sorted(unknown))}")

    archives = {PROJECT_ARCHIVE: {"files": [], "dirs": [], "samples": []}}
    for s in all_samples:
        if s in selected:
            archives[_archive_name_for_sample(s)] = {"files": [], "dirs": [], "samples": [s]}

    interrupted = set()

    def _sample_of(rel_parts):
        if len(rel_parts) < 2 or rel_parts[0] not in SAMPLE_TOP_DIRS:
            return None
        sample = owners.get((rel_parts[0], rel_parts[1]))
        if sample is None and rel_parts[1] in sample_set:
            sample = rel_parts[1]
        return sample

    def _archive_for(rel_parts):
        sample = _sample_of(rel_parts)
        if sample is None:
            return PROJECT_ARCHIVE
        return _archive_name_for_sample(sample) if sample in selected else None

    for root, dirs, files in os.walk(output_dir):
        dirs[:] = sorted(d for d in dirs if not _is_excluded_dir(d))
        root_path = Path(root)
        rel_root = root_path.relative_to(output_dir)
        rel_root_parts = rel_root.parts
        target_for_dir = _archive_for(rel_root_parts + ("_",)) if rel_root_parts else PROJECT_ARCHIVE
        if INTERRUPTED_MARKER in files and _sample_of(rel_root_parts + ("_",)) in selected:
            interrupted.add(_sample_of(rel_root_parts + ("_",)))
        kept_files = sorted(f for f in files if not _is_excluded_file(f))
        if not kept_files and not dirs and rel_root_parts and target_for_dir is not None:
            archives[target_for_dir]["dirs"].append(rel_root.as_posix())
        for name in kept_files:
            rel_parts = rel_root_parts + (name,)
            target = _archive_for(rel_parts)
            if target is None:
                continue
            abs_path = root_path / name
            st = abs_path.stat()
            archives[target]["files"].append(
                (abs_path, PurePosixPath(*rel_parts).as_posix(), st.st_size, st.st_mtime_ns)
            )

    # The metadata CSV may live outside the output folder; carry it along.
    md_rel = None
    if metadata_csv is not None and Path(metadata_csv).exists():
        md_abs = Path(metadata_csv)
        if _same_or_inside(md_abs, output_dir):
            md_rel = md_abs.resolve().relative_to(output_dir.resolve()).as_posix()
        else:
            md_rel = md_abs.name
            existing = {rel for _a, rel, _s, _m in archives[PROJECT_ARCHIVE]["files"]}
            if md_rel in existing:
                md_rel = f"_external_metadata/{md_abs.name}"
            st = md_abs.stat()
            archives[PROJECT_ARCHIVE]["files"].append((md_abs, md_rel, st.st_size, st.st_mtime_ns))

    total = sum(size for a in archives.values() for _x, _r, size, _m in a["files"])
    return {
        "output_dir": output_dir,
        "archives": archives,
        "total_bytes": total,
        "metadata_csv": md_rel,
        "external_paths": external,
        "interrupted": sorted(interrupted),
        "all_samples": all_samples,
        "selected_samples": [s for s in all_samples if s in selected],
    }


def default_bundle_dir(output_dir, destination):
    """``<destination>/<project>_behav3d_bundle`` unless *destination* already is a bundle."""
    destination = Path(destination)
    if destination.name.endswith(BUNDLE_SUFFIX) or (destination / BUNDLE_JSON).exists():
        return destination
    return destination / f"{Path(output_dir).name}{BUNDLE_SUFFIX}"


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
def _write_archive(partial_path, archive_name, entry, progress, log):
    manifest = {"format": BUNDLE_FORMAT, "archive": archive_name, "files": [], "dirs": list(entry["dirs"])}
    # compresslevel 1: the deflated members are text; speed matters more than ratio.
    with zipfile.ZipFile(partial_path, "w", allowZip64=True, compresslevel=1, strict_timestamps=False) as zf:
        for rel_dir in entry["dirs"]:
            zf.writestr(zipfile.ZipInfo(rel_dir.rstrip("/") + "/"), b"")
        for abs_path, rel, _size, _mtime in entry["files"]:
            progress.check_cancel()
            zinfo = zipfile.ZipInfo.from_file(_lp(abs_path), rel, strict_timestamps=False)
            if Path(rel).suffix.lower() in TEXT_SUFFIXES:
                zinfo.compress_type = zipfile.ZIP_DEFLATED
            else:
                zinfo.compress_type = zipfile.ZIP_STORED
            h = hashlib.sha256()
            size = 0
            with open(_lp(abs_path), "rb") as src, zf.open(zinfo, "w") as dst:
                while True:
                    block = src.read(_BLOCK)
                    if not block:
                        break
                    h.update(block)
                    dst.write(block)
                    size += len(block)
                    progress.advance(len(block))
                    progress.check_cancel()
            manifest["files"].append({"path": rel, "size": size, "sha256": h.hexdigest()})
            progress.advance(0, message=f"{archive_name}: {rel}")
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=1))
    with open(partial_path, "rb+") as f:
        os.fsync(f.fileno())
    return manifest


def _check_archive_contents(archive_path, progress=None):
    """Re-read every member, compare SHA-256 with the manifest. Returns a list of problems."""
    problems = []
    with zipfile.ZipFile(archive_path) as zf:
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME))
        except KeyError:
            return [f"{Path(archive_path).name}: manifest missing"]
        names = set(zf.namelist())
        for item in manifest["files"]:
            rel = item["path"]
            if rel not in names:
                problems.append(f"{rel}: missing from archive")
                continue
            h = hashlib.sha256()
            try:
                with zf.open(rel) as src:
                    while True:
                        if progress is not None:
                            progress.check_cancel()
                        block = src.read(_BLOCK)
                        if not block:
                            break
                        h.update(block)
                        if progress is not None:
                            progress.advance(len(block))
            except (zipfile.BadZipFile, OSError) as e:
                problems.append(f"{rel}: unreadable ({e})")
                continue
            if h.hexdigest() != item["sha256"]:
                problems.append(f"{rel}: checksum mismatch")
    return problems


def _load_bundle_json(bundle_dir):
    path = Path(bundle_dir) / BUNDLE_JSON
    if not path.exists():
        raise TransferError(f"No {BUNDLE_JSON} in {bundle_dir} - is this a BEHAV3D bundle folder?")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    if int(info.get("format", 0)) > BUNDLE_FORMAT:
        raise TransferError("This bundle was made by a newer BEHAV3D version; please update BEHAV3D.")
    return info


def _save_bundle_json(bundle_dir, info):
    _write_text_atomic(Path(bundle_dir) / BUNDLE_JSON, json.dumps(info, indent=2))
    lines = [f"{a['sha256']} *{a['name']}\n" for a in info["archives"]]
    _write_text_atomic(Path(bundle_dir) / SUMS_FILE, "".join(lines))


def pack_project(
    output_dir,
    destination,
    metadata_csv=None,
    samples=None,
    *,
    force=False,
    verify=True,
    progress=None,
    cancel=None,
    log=print,
):
    """Pack *output_dir* into a bundle folder under *destination*.

    Archives whose sources are unchanged since the last pack into the same
    bundle are kept, so re-running after an interruption or after adding a
    sample only writes what changed. Returns the bundle folder.
    """
    output_dir = Path(output_dir)
    plan = plan_bundle(output_dir, metadata_csv=metadata_csv, samples=samples)
    bundle_dir = default_bundle_dir(output_dir, destination)

    if _same_or_inside(bundle_dir, output_dir):
        raise TransferError("The bundle cannot be written inside the output folder it packs.")
    if plan["interrupted"] and not force:
        raise InterruptedRunError(
            "Segmentation was interrupted for: " + ", ".join(plan["interrupted"])
            + ". Finish or re-run it first, or pack anyway with force."
        )
    for ext in plan["external_paths"][:20]:
        log(f"⚠️ Not included (outside the output folder): {ext}")
    if len(plan["external_paths"]) > 20:
        log(f"⚠️ ... and {len(plan['external_paths']) - 20} more paths outside the output folder")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (bundle_dir / BUNDLE_JSON).exists():
        try:
            previous = {a["name"]: a for a in _load_bundle_json(bundle_dir)["archives"]}
        except Exception:
            previous = {}

    todo = []
    kept = []
    for name, entry in plan["archives"].items():
        fp = _fingerprint(entry["files"] + [(None, d + "/", 0, 0) for d in entry["dirs"]])
        prev = previous.get(name)
        final = bundle_dir / name
        if prev and prev.get("fingerprint") == fp and final.exists() and final.stat().st_size == prev.get("size"):
            kept.append(prev)
        else:
            todo.append((name, entry, fp))

    need = sum(size for _n, e, _f in todo for _a, _r, size, _m in e["files"])
    free = shutil.disk_usage(bundle_dir).free
    if need > free:
        raise TransferError(f"Not enough space at destination: need {_fmt_bytes(need)}, free {_fmt_bytes(free)}.")

    read_back = need * (2 if verify else 1)
    prog = _Progress(need + read_back, progress, cancel)
    log(f"📦 Packing {len(todo)} archive(s), {_fmt_bytes(need)} → {bundle_dir}"
        + (f" ({len(kept)} unchanged, kept)" if kept else ""))

    info = {
        "format": BUNDLE_FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "behav3d_version": _behav3d_version(),
        "project_name": output_dir.name,
        "source_output_dir": str(output_dir.resolve()),
        "metadata_csv": plan["metadata_csv"],
        "samples": plan["selected_samples"],
        "external_paths": plan["external_paths"],
        "archives": list(kept),
    }

    for name, entry, fp in todo:
        final = bundle_dir / name
        partial = bundle_dir / (name + ".partial")
        try:
            prog.phase = "Writing"
            manifest = _write_archive(partial, name, entry, prog, log)
            if verify:
                prog.phase = "Verifying contents"
                problems = _check_archive_contents(partial, prog)
                if problems:
                    raise TransferError(f"{name} failed verification after writing: " + "; ".join(problems[:5]))
            prog.phase = "Checksum"
            prog.advance(0, message=name)
            digest = _sha256_file(partial, prog)
            os.replace(partial, final)
        except BaseException:
            try:
                partial.unlink()
            except OSError:
                pass
            raise
        record = {
            "name": name,
            "sha256": digest,
            "size": final.stat().st_size,
            "n_files": len(manifest["files"]),
            "data_bytes": sum(f["size"] for f in manifest["files"]),
            "samples": entry["samples"],
            "fingerprint": fp,
        }
        info["archives"] = [a for a in info["archives"] if a["name"] != name] + [record]
        _save_bundle_json(bundle_dir, info)
        log(f"  ✅ {name}: {record['n_files']} files, {_fmt_bytes(record['size'])}")

    order = list(plan["archives"])
    info["archives"].sort(key=lambda a: order.index(a["name"]))
    _save_bundle_json(bundle_dir, info)
    stray = sorted(p.name for p in bundle_dir.glob("*.zip") if p.name not in order)
    if stray:
        log(f"ℹ️ Not part of this bundle (left untouched): {', '.join(stray)}")
    prog.phase = ""
    prog.advance(0, message="done", force=True)
    log(f"✅ Bundle ready: {bundle_dir}")
    return bundle_dir


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def verify_bundle(bundle_dir, *, deep=False, progress=None, cancel=None, log=print):
    """Check every archive of a bundle against the hashes in bundle.json.

    ``deep=True`` also re-reads every member against its manifest hash.
    Returns ``{"ok": bool, "problems": [str], "archives": int}``.
    """
    bundle_dir = Path(bundle_dir)
    info = _load_bundle_json(bundle_dir)
    total = sum(a["size"] for a in info["archives"])
    if deep:
        total += sum(a.get("data_bytes", 0) for a in info["archives"])
    prog = _Progress(total, progress, cancel)
    problems = []
    for a in info["archives"]:
        path = bundle_dir / a["name"]
        prog.phase = "Checksum"
        prog.advance(0, message=a["name"])
        if not path.exists():
            problems.append(f"{a['name']}: missing")
            continue
        if path.stat().st_size != a["size"]:
            problems.append(f"{a['name']}: size {path.stat().st_size} != expected {a['size']} (incomplete copy?)")
            continue
        if _sha256_file(path, prog) != a["sha256"]:
            problems.append(f"{a['name']}: checksum mismatch (corrupted copy)")
            continue
        if deep:
            prog.phase = "Verifying contents"
            problems.extend(f"{a['name']}: {p}" for p in _check_archive_contents(path, prog))
        log(f"  ✅ {a['name']}")
    for p in problems:
        log(f"  ❌ {p}")
    ok = not problems
    log(f"{'✅ Bundle OK' if ok else '❌ Bundle has problems'}: {len(info['archives'])} archive(s) checked")
    return {"ok": ok, "problems": problems, "archives": len(info["archives"])}


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------
def _unit_of(rel):
    """Conflict unit of a member: its enclosing ``*.zarr`` store, else the file itself."""
    parts = PurePosixPath(rel).parts
    for i, part in enumerate(parts):
        if part.endswith(".zarr"):
            return PurePosixPath(*parts[: i + 1]).as_posix()
    return PurePosixPath(rel).as_posix()


def unpack_bundle(
    bundle_dir,
    output_dir,
    *,
    overwrite=False,
    progress=None,
    cancel=None,
    log=print,
):
    """Extract a bundle into *output_dir*, check every file, then re-link paths.

    Files are extracted to a sibling ``<output_dir>.b3dtmp`` folder (same
    drive, so the final move is a rename) and only moved into place once
    every SHA-256 matches. Existing files or zarr stores are
    never touched unless *overwrite* is set (a whole store is replaced, never
    merged chunk by chunk). Returns ``{"output_dir", "metadata_csv", "relinked"}``.
    """
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    info = _load_bundle_json(bundle_dir)

    # Read manifests first: conflict check before writing anything.
    manifests = []
    for a in info["archives"]:
        path = bundle_dir / a["name"]
        if not path.exists():
            raise TransferError(f"Archive missing from bundle: {a['name']}")
        with zipfile.ZipFile(path) as zf:
            try:
                manifest = json.loads(zf.read(MANIFEST_NAME))
            except KeyError:
                raise TransferError(f"{a['name']}: manifest missing - not a BEHAV3D archive?")
        for item in manifest["files"]:
            _safe_member_path(item["path"])
        for d in manifest.get("dirs", []):
            _safe_member_path(d)
        manifests.append((path, manifest))

    units = sorted({_unit_of(item["path"]) for _p, m in manifests for item in m["files"]})
    conflicts = [u for u in units if (output_dir / u).exists()]
    if conflicts and not overwrite:
        shown = ", ".join(conflicts[:5]) + (f" (+{len(conflicts) - 5} more)" if len(conflicts) > 5 else "")
        raise TransferError(f"{len(conflicts)} item(s) already exist in {output_dir}: {shown}. "
                            "Choose an empty folder or allow overwriting.")

    created_output_dir = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(item["size"] for _p, m in manifests for item in m["files"])
    free = shutil.disk_usage(output_dir).free
    if total > free:
        raise TransferError(f"Not enough space: need {_fmt_bytes(total)}, free {_fmt_bytes(free)}.")

    output_dir = output_dir.absolute()
    if output_dir.parent == output_dir:  # drive root
        tmp = output_dir / IMPORT_TMP_SUFFIX
    else:
        tmp = output_dir.parent / (output_dir.name + IMPORT_TMP_SUFFIX)
    if tmp.exists():
        shutil.rmtree(_lp(tmp))
    tmp.mkdir()
    prog = _Progress(total, progress, cancel)
    prog.phase = "Extracting"
    log(f"📥 Importing {len(manifests)} archive(s), {_fmt_bytes(total)} → {output_dir}")
    try:
        for path, manifest in manifests:
            with zipfile.ZipFile(path) as zf:
                for d in manifest.get("dirs", []):
                    _lp(tmp / d).mkdir(parents=True, exist_ok=True)
                for item in manifest["files"]:
                    prog.check_cancel()
                    target = _lp(tmp / item["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    try:
                        with zf.open(item["path"]) as src, open(target, "wb") as dst:
                            while True:
                                block = src.read(_BLOCK)
                                if not block:
                                    break
                                h.update(block)
                                dst.write(block)
                                prog.advance(len(block))
                                prog.check_cancel()
                    except (zipfile.BadZipFile, KeyError, OSError) as e:
                        raise TransferError(f"{path.name}: cannot extract {item['path']} ({e})")
                    if h.hexdigest() != item["sha256"]:
                        raise TransferError(f"{path.name}: {item['path']} is corrupted (checksum mismatch). "
                                            "Copy this archive again and retry.")
                    prog.advance(0, message=f"{path.name}: {item['path']}")
            log(f"  ✅ {path.name} checked")

        # Everything verified: move into place.
        if overwrite:
            for u in conflicts:
                target = _lp(output_dir / u)
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        for _p, manifest in manifests:
            for d in manifest.get("dirs", []):
                _lp(output_dir / d).mkdir(parents=True, exist_ok=True)
            for item in manifest["files"]:
                target = _lp(output_dir / item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(_lp(tmp / item["path"]), target)
    except BaseException:
        if created_output_dir:
            try:
                output_dir.rmdir()  # only succeeds if nothing was moved in
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(_lp(tmp), ignore_errors=True)

    result = {"output_dir": output_dir, "metadata_csv": None, "relinked": 0}
    md_rel = info.get("metadata_csv")
    if md_rel:
        md_path = output_dir / md_rel
        result["metadata_csv"] = md_path
        if md_path.exists():
            from behav3d.core.metadata import load_behav3d_metadata, relink_metadata_csv
            md = load_behav3d_metadata(md_path)
            _md, report = relink_metadata_csv(
                md, md_path, output_dir, old_output_dir=info.get("source_output_dir"), log=log, backup=False
            )
            result["relinked"] = len(report["changed"])
    _update_params_paths(output_dir, result["metadata_csv"])
    prog.phase = ""
    prog.advance(0, message="done", force=True)
    log(f"✅ Imported into {output_dir}")
    return result


def _update_params_paths(output_dir, metadata_csv):
    from behav3d.io.parameters import load_params, params_path_for, save_params

    if not params_path_for(output_dir).exists():
        return
    params = load_params(output_dir)
    paths = params.setdefault("paths", {}) or {}
    params["paths"] = paths
    paths["output_dir"] = str(output_dir)
    if metadata_csv is not None:
        paths["metadata_csv"] = str(metadata_csv)
    save_params(params, output_dir)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _cli_progress():
    from tqdm import tqdm

    bar = {"t": None}

    def update(done, total, _message):
        if bar["t"] is None:
            bar["t"] = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        bar["t"].total = total
        bar["t"].n = done
        bar["t"].refresh()

    def close():
        if bar["t"] is not None:
            bar["t"].close()

    return update, close


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m behav3d.io.transfer", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="pack an output folder into a bundle")
    p.add_argument("output_dir")
    p.add_argument("destination", help="folder to create the bundle in (e.g. a USB drive or NAS share)")
    p.add_argument("--metadata", help="metadata CSV (default: <output_dir>/metadata.csv)")
    p.add_argument("--samples", nargs="+", help="only these samples (project files are always included)")
    p.add_argument("--force", action="store_true", help="pack even if a segmentation run was interrupted")
    p.add_argument("--no-verify", action="store_true", help="skip re-reading archives after writing")
    v = sub.add_parser("verify", help="check a bundle's archives against their checksums")
    v.add_argument("bundle_dir")
    v.add_argument("--deep", action="store_true", help="also check every file inside the archives")
    u = sub.add_parser("unpack", help="extract a bundle into a new output folder")
    u.add_argument("bundle_dir")
    u.add_argument("output_dir")
    u.add_argument("--overwrite", action="store_true", help="replace files/zarr stores that already exist")
    args = parser.parse_args(argv)

    from tqdm import tqdm

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # emoji on a cp1252 console
        except (AttributeError, ValueError):
            pass
    update, close = _cli_progress()
    log = tqdm.write  # prints above the progress bar instead of through it
    try:
        if args.cmd == "pack":
            pack_project(args.output_dir, args.destination, metadata_csv=args.metadata, samples=args.samples,
                         force=args.force, verify=not args.no_verify, progress=update, log=log)
        elif args.cmd == "verify":
            if not verify_bundle(args.bundle_dir, deep=args.deep, progress=update, log=log)["ok"]:
                return 1
        elif args.cmd == "unpack":
            unpack_bundle(args.bundle_dir, args.output_dir, overwrite=args.overwrite, progress=update, log=log)
    except TransferError as e:
        close()
        print(f"Error: {e}", file=sys.stderr)
        return 2
    close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

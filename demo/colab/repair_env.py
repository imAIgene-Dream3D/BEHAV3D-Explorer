"""Repair a conda-pack environment that pip and conda disagree about.

``demo/build_env.sh`` solves the environment with conda and then pip-installs
PyTorch and Cellpose on top. When one of those pip steps has to *change* a
package conda already owns - ``cellpose==3.1.1.2`` requires ``numpy<2.1``, which
walks numpy, scipy, numba and llvmlite back a version - pip overwrites conda's
files but conda's records in ``conda-meta/`` still describe the version it
installed. The container keeps working, because on disk everything is pip's
coherent set.

``conda-pack`` then packs from conda's records. What comes out is a mix: conda's
files for those packages, pip's ``.dist-info`` beside them. It only fails once it
is unpacked somewhere else, as an import error with no obvious cause::

    ModuleNotFoundError: No module named 'numpy._utils._conversions'
    ModuleNotFoundError: No module named 'scipy._external'

This script finds every package where the two views disagree and puts back the
version pip recorded - that is the one the rest of the environment was resolved
against.

    python repair_env.py            # repair in place (run with the env's python)
    python repair_env.py --check    # report only; exit 1 if anything is mixed

``--check`` is the same comparison, which is why ``build_env.sh`` runs it as a
build-time guard: a tarball that fails it must never reach a visitor.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from importlib.metadata import distributions


def _normalise(name):
    """conda and pip spell the same package slightly differently."""
    return name.lower().replace("_", "-")


def find_mismatches():
    """Packages whose conda record and installed dist-info disagree.

    Returns a list of ``(name, conda_version, pip_version, dist)``. Both views
    come from the *running* interpreter, so this must be run with the python of
    the environment being inspected.
    """
    prefix = pathlib.Path(sys.prefix)
    installed = {}
    for dist in distributions():
        name = _normalise(dist.metadata["Name"] or "")
        if name:
            installed[name] = dist

    mismatches = []
    for meta in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            record = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        dist = installed.get(_normalise(record.get("name", "")))
        if dist is not None and dist.version != record.get("version"):
            mismatches.append((_normalise(record["name"]), record["version"], dist.version, dist))
    return mismatches


def _clear(dist):
    """Delete the package's top-level directories so a reinstall starts clean.

    ``--force-reinstall`` alone would overwrite whatever the wheel ships but
    leave the other version's extra modules behind, which is how the mix started.
    """
    site = pathlib.Path(dist.locate_file(""))
    for entry in {str(f).split("/")[0] for f in (dist.files or [])}:
        if entry.startswith("..") or entry.endswith(".dist-info"):
            continue
        target = site / entry
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


def repair(check_only=False):
    """Reinstall every mixed package at the version pip recorded.

    Returns True when the environment is (or has been made) consistent.
    """
    mismatches = find_mismatches()
    if not mismatches:
        print("[repair_env] conda records and installed packages agree")
        return True

    print(f"[repair_env] {len(mismatches)} package(s) mixed by conda-pack:")
    for name, conda_version, pip_version, _ in mismatches:
        print(f"[repair_env]   {name}: conda shipped {conda_version}, pip recorded {pip_version}")
    if check_only:
        return False

    for _, _, _, dist in mismatches:
        _clear(dist)

    specs = [f"{name}=={pip_version}" for name, _, pip_version, _ in mismatches]
    print("[repair_env] reinstalling: " + " ".join(specs))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--force-reinstall", "--no-deps", *specs],
        check=True,
    )
    print("[repair_env] repaired")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report mismatches and exit 1 if there are any; change nothing")
    args = parser.parse_args(argv)
    return 0 if repair(check_only=args.check) else 1


if __name__ == "__main__":
    sys.exit(main())

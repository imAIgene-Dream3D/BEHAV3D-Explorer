"""
Import-weight regression guard.

The napari plugin imports ``behav3d.napari._widget`` on the Qt main thread, and the
Single Cell tabs touch a few helpers on every metadata load. A single stray
``import umap`` (pynndescent -> ~9 s of numba JIT) or a top-level import of
``state.classification`` (scanpy, seaborn, sklearn, hmmlearn, ...) has twice undone
the metadata-loading speed-up (see commits ce9c4a8, 2e922fa, beb2aaf). These tests
fail loudly when that happens again.

Each check runs in a fresh interpreter so ``sys.modules`` is clean.

Runs under pytest *or* standalone: python test/test_import_weight.py
Requires the `behav3d` conda env.

``numba`` and ``skimage`` are deliberately NOT asserted: napari itself imports both
at startup (napari/utils/colormaps), so they are loaded regardless of this package.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Heavy packages that must stay out of plugin-load and of the metadata-load path.
FORBIDDEN_AT_PLUGIN_LOAD = [
    "umap",
    "pynndescent",
    "scanpy",
    "anndata",
    "seaborn",
    "sklearn",
    "hmmlearn",
    "plotly",
    "torch",
    "cellpose",
]

# Imports the napari Single Cell tabs perform on every metadata load.
METADATA_LOAD_STATEMENTS = (
    "from behav3d.analysis.behavior.track.utils import _peek_track_outfolder, "
    "get_track_classifier_filename\n"
    "from behav3d.core.state_columns import FULL_STATE_COL, INTRINSIC_STATE_COL, "
    "resolve_full_state_col, resolve_intrinsic_state_col\n"
    "from behav3d.core.column_detection import normalize_binary_value\n"
)


def _loaded_after(statements, forbidden):
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import behav3d\n"
        f"assert behav3d.__file__.startswith({str(REPO)!r}), behav3d.__file__\n"
        f"{statements}\n"
        f"print('RESULT=' + json.dumps([m for m in {forbidden!r} if m in sys.modules]))\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")][-1]
    return json.loads(line[len("RESULT="):])


def test_plugin_load_does_not_import_heavy_stack():
    loaded = _loaded_after("import behav3d.napari._widget", FORBIDDEN_AT_PLUGIN_LOAD)
    assert loaded == [], f"plugin load imported heavy packages: {loaded}"


def test_metadata_load_helpers_do_not_import_heavy_stack():
    loaded = _loaded_after(METADATA_LOAD_STATEMENTS, FORBIDDEN_AT_PLUGIN_LOAD)
    assert loaded == [], f"metadata-load helpers imported heavy packages: {loaded}"


def test_track_report_modules_do_not_import_umap_eagerly():
    # umap is only needed when a UMAP is actually fitted; importing the report /
    # clustering modules must not pay for it.
    loaded = _loaded_after(
        "import behav3d.analysis.behavior.track.bouts\n"
        "import behav3d.analysis.behavior.track.visualization.backprojection\n"
        "import behav3d.analysis.behavior.track.visualization.plots.reports\n"
        "import behav3d.analysis.behavior.track.dtw\n",
        ["umap", "pynndescent"],
    )
    assert loaded == [], f"track modules imported umap eagerly: {loaded}"


def test_state_column_reexports_are_the_same_objects():
    # The leaf module must stay the single source of truth; the old import path
    # is kept for backwards compatibility.
    from behav3d.analysis.behavior.state import classification as old
    from behav3d.core import state_columns as new

    for name in (
        "FULL_STATE_COL",
        "INTRINSIC_STATE_COL",
        "HMM_INTRINSIC_RAW_STATE_COL",
        "resolve_full_state_col",
        "resolve_intrinsic_state_col",
        "_format_hmm_raw_state_series_for_key",
    ):
        assert getattr(old, name) is getattr(new, name), name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)

"""
Tests for the "no O(data) work on the GUI thread" changes:

* ``classify_feature_columns`` equals the old two-pass detectors and is memoized
  per file version (one parse per CSV version, shared by the State tab and the
  Trajectory Feature selector);
* the vectorized longest-track computation matches the old per-row
  implementation, for both h5ad and CSV sources;
* ``_read_h5ad_light_meta`` equals a backed ``read_h5ad`` without reading obs;
* ``ResultsPanel.refresh_if_stale`` rescans only when something under
  ``analysis/`` changed;
* the informational ("notify") background tier shows in the banner list without
  counting as work the user must wait for;
* ``TrackClassificationSubTab._reload`` keeps an unchanged h5ad, and does not run
  any whole-file read on the GUI thread.

Call-count / equality tests only; nothing here measures time.
"""
import threading
import time
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

import behav3d.napari._analysis  # noqa: F401  (resolves a circular import first)
from behav3d.core import column_detection as cd

_APP = None


def _qt_app():
    global _APP
    from qtpy.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump_until(condition, timeout_s=10.0):
    app = _qt_app()
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return condition()


def _tracks_csv(path, n_rows=400):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "sample_name": np.where(np.arange(n_rows) % 2 == 0, "s_a", "s_b"),
        "TrackID": np.arange(n_rows) % 7,
        "position_t": np.arange(n_rows) // 7,
        "speed": rng.normal(size=n_rows),
        "touching": (np.arange(n_rows) % 3 == 0).astype(int),
        "flag": np.where(np.arange(n_rows) % 2 == 0, "true", "false"),
        "label": np.where(np.arange(n_rows) % 5 == 0, "um", "5"),
    })
    df.to_csv(path, index=False)
    return df


# ── classify_feature_columns ───────────────────────────────────────────────

def test_classify_matches_old_two_pass_detectors(tmp_path):
    csv = tmp_path / "t.csv"
    _tracks_csv(csv)
    cols = ["speed", "touching", "flag", "label"]

    old_bin = cd.detect_binary_columns_from_csv(csv, cols, chunksize=50)
    old_cands = [c for c in cols if c not in set(old_bin)]
    old_non_numeric = cd.detect_non_numeric_columns_from_csv(csv, old_cands, chunksize=50)

    new = cd.classify_feature_columns(csv, cols, chunksize=50)
    assert new["bin_cols"] == old_bin
    assert new["non_numeric"] == old_non_numeric
    assert new["feat_cols"] == [c for c in old_cands if c not in set(old_non_numeric)]


def test_classify_is_memoized_per_file_version(tmp_path, monkeypatch):
    csv = tmp_path / "t.csv"
    _tracks_csv(csv)
    cols = ["speed", "touching"]
    reads = []
    real = pd.read_csv

    def counting(*a, **k):
        reads.append(1)
        return real(*a, **k)

    monkeypatch.setattr(cd.pd, "read_csv", counting)
    first = cd.classify_feature_columns(csv, cols)
    n_after_first = len(reads)
    assert n_after_first >= 1
    second = cd.classify_feature_columns(csv, cols)
    assert second == first
    assert len(reads) == n_after_first  # served from the memo

    # Rewriting the file (new size/mtime) invalidates it.
    _tracks_csv(csv, n_rows=450)
    cd.classify_feature_columns(csv, cols)
    assert len(reads) > n_after_first


def test_classify_cancelled_scan_is_not_memoized(tmp_path):
    csv = tmp_path / "t.csv"
    _tracks_csv(csv)
    cols = ["speed", "touching"]
    cd.classify_feature_columns(csv, cols, chunksize=50, cancel_check=lambda: True)
    result = cd.classify_feature_columns(csv, cols, chunksize=50)
    assert result["bin_cols"] == ["touching"]


# ── longest track ──────────────────────────────────────────────────────────

def _old_max_track_length_h5ad(path):
    """The previous implementation (per-row Python lists + pandas groupby)."""
    import h5py
    from behav3d.napari._single_cell import TrackClassificationSubTab

    read = TrackClassificationSubTab._read_h5ad_obs_column
    with h5py.File(str(path), "r") as f:
        names = read(f, "sample_name")
        ids = read(f, "TrackID")
    counts = pd.DataFrame({"sample_name": names, "TrackID": ids}).groupby(
        ["sample_name", "TrackID"]
    ).size()
    return int(counts.max())


def _states_h5ad(path, categorical=True):
    df = _tracks_csv(path.with_suffix(".csv"))[["sample_name", "TrackID", "position_t"]].copy()
    df["TrackID"] = df["TrackID"].astype(int)
    if categorical:
        df["sample_name"] = df["sample_name"].astype("category")
    else:
        df["sample_name"] = df["sample_name"].astype(str)
    adata = ad.AnnData(
        X=np.zeros((len(df), 2)),
        obs=df.reset_index(drop=True),
        var=pd.DataFrame(index=pd.Index(["a", "b"], name="feature")),
    )
    adata.uns["preprocessing"] = {
        "continuous_feature_cols": ["a", "b"],
        "binary_cols_to_merge": ["position_t"],
    }
    adata.write(path)
    return path


def test_max_track_length_matches_old_implementation_h5ad(tmp_path):
    from behav3d.napari._single_cell import _compute_max_track_length

    for categorical in (True, False):
        path = _states_h5ad(tmp_path / f"s_{categorical}.h5ad", categorical=categorical)
        assert _compute_max_track_length("h5ad", str(path)) == _old_max_track_length_h5ad(path)


def test_max_track_length_csv_and_missing_keys(tmp_path):
    from behav3d.napari._single_cell import _compute_max_track_length

    csv = tmp_path / "p.csv"
    df = _tracks_csv(csv)
    expected = int(df.groupby(["sample_name", "TrackID"]).size().max())
    assert _compute_max_track_length("csv", str(csv)) == expected
    assert _compute_max_track_length("csv", str(tmp_path / "missing.csv")) is None

    no_keys = tmp_path / "nokeys.h5ad"
    ad.AnnData(X=np.zeros((3, 1))).write(no_keys)
    assert _compute_max_track_length("h5ad", str(no_keys)) is None


# ── light h5ad metadata ────────────────────────────────────────────────────

def test_light_meta_matches_backed_read(tmp_path):
    from behav3d.napari._single_cell import _read_h5ad_light_meta

    path = _states_h5ad(tmp_path / "states.h5ad")
    meta = _read_h5ad_light_meta(path)
    assert meta is not None
    pre, var_names, obs_columns = meta
    backed = ad.read_h5ad(str(path), backed="r")
    try:
        assert var_names == [str(v) for v in backed.var_names]
        assert obs_columns == set(backed.obs.columns)
        assert list(pre["continuous_feature_cols"]) == ["a", "b"]
        assert list(pre["binary_cols_to_merge"]) == ["position_t"]
    finally:
        backed.file.close()


# ── Results panel ──────────────────────────────────────────────────────────

def test_results_panel_refresh_if_stale(tmp_path):
    _qt_app()
    from behav3d.napari._results_panel import ResultsPanel

    out = tmp_path / "out"
    (out / "analysis" / "tcell" / "reports").mkdir(parents=True)
    (out / "analysis" / "tcell" / "reports" / "a.csv").write_text("x\n1\n")
    loader = SimpleNamespace(output_dir=str(out))
    panel = ResultsPanel(viewer=None, metadata_loader=loader)

    scans = []
    real_refresh = panel.refresh

    def counting_refresh():
        scans.append(1)
        real_refresh()

    panel.refresh = counting_refresh
    assert _pump_until(lambda: panel._last_scan_files is not None)

    panel.refresh_if_stale()
    panel.refresh_if_stale()
    assert scans == []  # nothing changed: no clear / rescan

    (out / "analysis" / "tcell" / "reports" / "b.pdf").write_text("x")
    panel.refresh_if_stale()
    assert scans == [1]
    assert _pump_until(lambda: not panel._scan_bg.is_running())


# ── informational busy tier ────────────────────────────────────────────────

def test_notify_operations_are_listed_but_never_block():
    _qt_app()
    from qtpy.QtWidgets import QWidget
    from behav3d.napari import _background_runner as br

    owner = QWidget()
    quiet = br.BackgroundOperation(owner, silent=True)
    shown = br.BackgroundOperation(owner, silent=True, notify=True)
    release = threading.Event()

    def _wait(cancel_check=None):
        release.wait(5)
        return 1

    quiet.run(fn=_wait, inject_progress=False, desc="quiet scan")
    shown.run(fn=_wait, inject_progress=False, desc="visible scan")
    try:
        assert "visible scan" in br.informational_work_descriptions()
        assert "quiet scan" not in br.informational_work_descriptions()
        # Neither counts as work the user has to wait for.
        assert not br.any_background_work(include_queue=False)
        assert "visible scan" not in br.background_work_descriptions()
    finally:
        release.set()
        assert _pump_until(lambda: not quiet.is_running() and not shown.is_running())


# ── Track sub-tab: no whole-file reads on the GUI thread ───────────────────

def _make_track_tab(tmp_path, cell_type="tcell"):
    _qt_app()
    from behav3d.napari._single_cell import TrackClassificationSubTab

    loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={}, metadata=None)
    return TrackClassificationSubTab(
        viewer=None, metadata_loader=loader, cell_type_getter=lambda: cell_type
    )


def test_track_reload_keeps_unchanged_h5ad_and_stays_off_gui_thread(tmp_path, monkeypatch):
    import behav3d.napari._single_cell as sc

    cell_type = "tcell"
    states_dir = tmp_path / "analysis" / cell_type / "behavioral_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    states = _states_h5ad(states_dir / f"BEHAV3D_{cell_type}_behavioral_states.h5ad")
    feats = tmp_path / "analysis" / cell_type / "track_features"
    feats.mkdir(parents=True, exist_ok=True)
    _tracks_csv(feats / f"BEHAV3D_{cell_type}_combined_track_features_filtered.csv")

    main = threading.main_thread()
    on_gui = []

    def guard(name, fn):
        def wrapper(*a, **k):
            if threading.current_thread() is main:
                on_gui.append(name)
            return fn(*a, **k)
        return wrapper

    monkeypatch.setattr(sc, "_compute_max_track_length",
                        guard("max_track_length", sc._compute_max_track_length))
    monkeypatch.setattr(cd, "classify_feature_columns",
                        guard("classify", cd.classify_feature_columns))

    tab = _make_track_tab(tmp_path, cell_type)
    tab._reload()
    # The longest track caps the spinbox -- delivered once the background scan ends.
    expected = _old_max_track_length_h5ad(states)
    assert _pump_until(lambda: tab.spin_traj_size.maximum() == expected)
    assert _pump_until(lambda: not tab._preload_bg.is_running())
    assert on_gui == []

    # Pretend the track h5ad was loaded: an unchanged file must be kept.
    track_h5 = tab._track_adata_path(cell_type)
    if track_h5 is not None:
        sentinel = object()
        tab._track_adata = sentinel
        tab._track_adata_key = sc._file_key(track_h5)
        tab._reload()
        assert tab._track_adata is sentinel

    # A second reload does not touch the heavy readers again.
    on_gui.clear()
    tab._reload()
    _pump_until(lambda: not tab._maxlen_bg.is_running())
    assert on_gui == []


# ── Analysis tab: refresh only when something changed ──────────────────────

def test_analysis_refresh_if_stale_only_runs_when_outputs_change(tmp_path, monkeypatch):
    _qt_app()
    from behav3d.napari._analysis import AnalysisTab

    md = pd.DataFrame({
        "sample_name": ["s1"],
        "im_tcell_line_condition": ["WT_ctrl"],
        "raw_image_path": [str(tmp_path / "s1.zarr")],
    })
    loader = SimpleNamespace(
        output_dir=str(tmp_path), behav3d_parameters={}, metadata=md,
        metadata_loaded=SimpleNamespace(connect=lambda *_a, **_k: None),
    )
    tab = AnalysisTab(viewer=None, metadata_loader=loader)

    cascades = []
    real = tab._on_metadata_updated
    monkeypatch.setattr(tab, "_on_metadata_updated", lambda *a: (cascades.append(1), real(*a))[1])

    # First visit after construction: nothing recorded yet, so it refreshes once...
    assert tab.refresh_if_stale() is True
    assert _pump_until(lambda: not tab.single_cell_tab.state_tab._colscan_bg.is_running())
    # ...and then not again while nothing on disk changed.
    assert tab.refresh_if_stale() is False
    assert tab.refresh_if_stale() is False
    assert cascades == [1]

    # A new filtered CSV (written by the Filtering tab) makes the next visit refresh.
    feats = tmp_path / "analysis" / "tcell" / "track_features"
    feats.mkdir(parents=True)
    (feats / "BEHAV3D_tcell_combined_track_features_filtered.csv").write_text("sample_name,TrackID\n")
    assert tab.refresh_if_stale() is True
    assert tab.refresh_if_stale() is False

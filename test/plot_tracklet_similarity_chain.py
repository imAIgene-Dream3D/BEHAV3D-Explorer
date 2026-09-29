from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import pairwise_distances

from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.state.utils import _set_classification_state_order
from behav3d.analysis.behavior.state.visualization.backprojection import (
    _resolve_behavioral_states_h5ad_path,
)
from behav3d.analysis.behavior.track.utils import _filter_tracks_for_dtaidistance
from behav3d.analysis.behavior.track.dtw import (
    extract_categorical_track_sequences,
    compute_dtaidistance_onehot_distance_matrix,
    compute_transition_profile_distance_matrix,
    _validate_distance_matrix,
)
from behav3d.analysis.behavior.track.visualization.plots.exemplar_track_per_cluster import (
    plot_tracks_bars_on_ax,
    _build_state_color_map,
)
from behav3d.analysis.behavior.utils import _to_numpy_2d
from behav3d.features.state_descriptive_features import (
    extract_descibing_track_state_features,
    clr_transform_columns,
    clr_transform_transition_rows,
    log1p_transform_columns,
    scale_feature_blocks,
)

# NOTE: this is a throwaway diagnostic script for looking at candidate ways of ordering
# tracklets - a greedy nearest-neighbor similarity chain anchored on a chosen reference
# tracklet, optionally steered by a user-editable semantic state order - before deciding
# whether it is worth building into the library proper. It intentionally saves a
# single-page PDF rather than following the rest of the codebase's multi-page PdfPages
# convention, since this is meant for fast visual iteration, not a report.
#
# Running this script (`python test/plot_tracklet_similarity_chain.py`) pops up a small
# Qt dialog for choosing every option below instead of editing these constants by hand;
# the constants only seed the dialog's initial values.


# %%
# Interactive configuration
# These seed the dialog's initial field values; edit them if you want a different
# starting point each time, but you no longer need to edit this file to change a run.
INTERACTIVE_BEHAV3D_FOLDER = None
INTERACTIVE_OUTPUT_DIR = "/Volumes/T7_Sam/BHVD_BEHAV3D/BEHAV3D_python/runs/NatureBriefComm/SafetyProfiling"
INTERACTIVE_CELL_TYPE = "TEG"
STATE_COL = FULL_STATE_COL

# Tracklet length ("e.g. 50 timepoints, user choice"). Long tracks are cut into
# independent, non-overlapping windows of this many frames; tracks shorter
# than this are dropped.
TRACKLET_LENGTH = 50

# Anchor tracklet placed at the bottom of the plot. Set REFERENCE_TRACK to pin
# an exact tracklet as (sample_name, TrackID) or
# (sample_name, TrackID, trajectory_window_id); otherwise the first tracklet
# whose entire sequence is REFERENCE_STATE_LABEL (e.g. every frame "dead") is
# used.
REFERENCE_TRACK = None
REFERENCE_STATE_LABEL = "dead"

# "dtw": DTW/transition-Jaccard distance computed directly over the raw per-timepoint
#   state sequence (see DTW_DISTANCE_METRIC below).
# "bouts": Euclidean/cosine distance over the same descriptive feature vector (state
#   fractions, bout stats, transitions, bigrams, trigrams) that the bouts-based track
#   clustering pipeline (run_state_based_analysis) clusters on, with the same
#   CLR/log1p/MFA-block-scaling preprocessing it applies.
TRACK_SIMILARITY_METHOD = "bouts"

# Only used when TRACK_SIMILARITY_METHOD == "dtw":
# "dtw_onehot" (timepoint-aligned DTW over one-hot states) or
# "transition_profile" (Jaccard distance over which state transitions occur).
DTW_DISTANCE_METRIC = "dtw_onehot"

# Only used when TRACK_SIMILARITY_METHOD == "bouts": distance metric over the descriptive
# feature vectors, e.g. "euclidean" or "cosine" (same convention as `leiden_metric` in
# run_state_based_analysis).
BOUTS_DISTANCE_METRIC = "euclidean"

# Only used when TRACK_SIMILARITY_METHOD == "bouts": which descriptive-feature blocks
# feed the final distance computation. The CLR/log1p/MFA-block-scaling preprocessing
# always still runs over the FULL block set (matching run_state_based_analysis exactly);
# this only controls which already-scaled columns are concatenated into X immediately
# before pairwise_distances. Motivating case: transition/bigram/trigram columns are
# noisy/ill-defined for near-constant-state tracklets (e.g. "dead" tracklets barely
# transition), which can scatter same-dominant-state tracklets apart under the full
# feature vector.
BOUTS_FEATURE_BLOCKS = ("fractions", "bout_stats", "transitions", "bigrams", "trigrams")

# Illustrative starting guess for the canonical semantic ordering of behavioral states,
# low -> high. The real category names in STATE_COL are dataset-specific and may not
# match these at all - the dialog's "state order" list is seeded from this constant plus
# whatever real categories are found in the loaded data (unlisted ones appended at the
# end with a warning), and you drag it into the right order for your dataset there.
STATE_ORDER = ["dead", "static", "scanner", "engager", "killer"]

# Which chain-ordering strategy to use (all derived from the same `distances` matrix and
# the same per-tracklet semantic score - see compute_semantic_state_scores):
#   "greedy"           - original behavior; STATE_ORDER/LAMBDA unused.
#   "score"            - order primarily by ascending semantic score; distances only
#                        break ties, via the greedy walk's own visiting order.
#   "bucket"           - bucket tracklets by their single dominant state, order buckets
#                        by the resolved state order, greedy-chain within each bucket.
#   "penalized_greedy" - same greedy walk as "greedy", but run on
#                        distances + LAMBDA * |score_i - score_j|.
ORDERING_METHOD = "greedy"

# Only used when ORDERING_METHOD == "penalized_greedy": weight of the semantic-score
# penalty added before the greedy walk. LAMBDA=0.0 reproduces "greedy" exactly. Score
# gaps range 1..len(STATE_ORDER) (see compute_semantic_state_scores), so re-tune LAMBDA
# relative to the typical magnitude of `distances` - bouts-Euclidean and DTW distances
# live on very different scales.
LAMBDA = 1.0

OUTPUT_PATH = "/Users/s.deblank-3/Downloads/tracklet_similarity_chain.pdf"


def resolve_behav3d_output_dir(
    behav3d_folder: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve the BEHAV3D output folder used by the state-h5ad path helper."""
    candidate = output_dir if output_dir not in {None, ""} else behav3d_folder
    if candidate in {None, ""}:
        raise ValueError(
            "Set INTERACTIVE_BEHAV3D_FOLDER or INTERACTIVE_OUTPUT_DIR to your "
            "BEHAV3D results folder."
        )
    resolved = Path(candidate).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"BEHAV3D folder does not exist: '{resolved}'")
    return resolved


def load_tracklet_data(output_dir: Path, cell_type: str, tracklet_length: int, state_col: str):
    """Resolve the behavioral-states h5ad, window tracks into fixed-length tracklets, and
    extract per-tracklet state sequences. Returns
    (adata_windowed, sequences, meta_df, categories_in_data).
    """
    h5ad_path = _resolve_behavioral_states_h5ad_path(output_dir=output_dir, cell_type=cell_type)
    adata = sc.read_h5ad(h5ad_path)

    adata_windowed = _filter_tracks_for_dtaidistance(
        adata,
        groupby_cols=("sample_name", "TrackID"),
        time_col="position_t",
        trajectory_size=tracklet_length,
        min_length=tracklet_length,
        trim_mode="last",
        split_long_tracks=True,
        window_col="trajectory_window_id",
    )

    sequences, meta_df = extract_categorical_track_sequences(
        adata_windowed,
        state_cols=(state_col,),
        groupby_cols=("sample_name", "TrackID", "trajectory_window_id"),
        time_col="position_t",
    )

    if pd.api.types.is_categorical_dtype(adata_windowed.obs[state_col]):
        categories_in_data = [str(v) for v in adata_windowed.obs[state_col].cat.categories]
    else:
        categories_in_data = list(dict.fromkeys(str(v) for v in adata_windowed.obs[state_col].dropna()))

    return adata_windowed, sequences, meta_df, categories_in_data


def resolve_reference_index(sequences, meta_df, *, reference_track=None, reference_state_label=None) -> int:
    """Return the row index (into `meta_df`/`sequences`) of the tracklet to anchor the chain at."""
    if reference_track is not None:
        reference_track = tuple(reference_track)
        key_cols = ["sample_name", "TrackID", "trajectory_window_id"][: len(reference_track)]
        mask = np.ones(len(meta_df), dtype=bool)
        for col, value in zip(key_cols, reference_track):
            mask &= meta_df[col].astype(str).to_numpy() == str(value)
        matches = np.flatnonzero(mask)
        if len(matches) == 0:
            raise ValueError(f"REFERENCE_TRACK={reference_track!r} did not match any tracklet.")
        return int(matches[0])

    if reference_state_label is not None:
        label = str(reference_state_label)
        for i, seq in enumerate(sequences):
            if set(seq) == {label}:
                return i
        raise ValueError(
            f"No tracklet found whose entire sequence is REFERENCE_STATE_LABEL={label!r}."
        )

    raise ValueError("Set either REFERENCE_TRACK or REFERENCE_STATE_LABEL to anchor the chain.")


def greedy_chain_order(distances: np.ndarray, start_idx: int) -> list[int]:
    """Greedy nearest-neighbor walk: start at `start_idx`, then repeatedly append
    whichever unvisited tracklet is closest to the last one added, until every
    tracklet has been placed once.

    Deliberately not a clustering algorithm - there is no notion of discrete
    groups, just "closest remaining neighbor next." Being greedy, it can
    occasionally have to take a longer jump once a local neighborhood of
    similar tracklets is exhausted, so a handful of dissimilar tracklets may
    still end up adjacent; that's an accepted limitation for this prototype.
    """
    n = distances.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [int(start_idx)]
    visited[start_idx] = True
    current = int(start_idx)
    for _ in range(n - 1):
        row = distances[current].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order


def resolve_semantic_state_order(categories_in_data, canonical_order) -> tuple[list[str], list[str]]:
    """Return (resolved_order, appended_unknown).

    `resolved_order` is `canonical_order` (str-cast, de-duplicated, order preserved)
    followed by every category in `categories_in_data` not already present, in
    first-seen order within `categories_in_data`. Canonical entries absent from the data
    are kept as-is - harmless, since no real tracklet can ever score against a rank that
    never occurs.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for state in canonical_order:
        state = str(state)
        if state not in seen:
            seen.add(state)
            resolved.append(state)

    appended_unknown: list[str] = []
    for state in categories_in_data:
        state = str(state)
        if state not in seen:
            seen.add(state)
            resolved.append(state)
            appended_unknown.append(state)

    return resolved, appended_unknown


def compute_semantic_state_scores(sequences, resolved_order) -> np.ndarray:
    """Per-tracklet scalar score: the frame-count-weighted mean rank of a tracklet's own
    states within `resolved_order` (1-based ranks, so an all-lowest-rank tracklet doesn't
    collide with a tracklet that has zero ranked frames at all). NaN for a tracklet whose
    entire sequence is states absent from `resolved_order`.
    """
    rank_of = {state: i + 1 for i, state in enumerate(resolved_order)}
    scores = np.full(len(sequences), np.nan, dtype=float)
    for i, seq in enumerate(sequences):
        counts = Counter(str(v) for v in seq)
        total_ranked = 0
        weighted_sum = 0.0
        for state, count in counts.items():
            rank = rank_of.get(state)
            if rank is None:
                continue
            weighted_sum += rank * count
            total_ranked += count
        if total_ranked > 0:
            scores[i] = weighted_sum / total_ranked
    return scores


def order_by_semantic_score(distances: np.ndarray, scores: np.ndarray, start_idx: int) -> list[int]:
    """Method "score": order tracklets primarily by ascending semantic score (NaN sorts
    last); ties are broken by each tracklet's position in one global greedy
    nearest-neighbor walk, so visually similar tracklets stay adjacent within a tie
    without disturbing the overall low-to-high progression.
    """
    primary = np.where(np.isnan(scores), np.inf, scores)
    chain_rank = np.empty(len(scores), dtype=int)
    for rank, idx in enumerate(greedy_chain_order(distances, start_idx)):
        chain_rank[idx] = rank
    order = np.lexsort((chain_rank, primary))
    return [int(i) for i in order]


def dominant_state_per_tracklet(sequences) -> list[str]:
    """Most frequent state per tracklet (ties broken by first-encountered, via
    Counter.most_common's own stable ordering)."""
    return [Counter(str(v) for v in seq).most_common(1)[0][0] for seq in sequences]


def order_by_dominant_state_buckets(
    distances: np.ndarray,
    sequences,
    resolved_order: list[str],
    start_idx: int,
) -> list[int]:
    """Method "bucket": bucket tracklets by their single dominant state, iterate buckets
    in canonical rank order (only ranks that actually occur - empty buckets are
    structurally impossible), and within each bucket run the existing
    `greedy_chain_order` on that bucket's own distance submatrix, entering each bucket
    from whichever tracklet is closest to the previous bucket's last tracklet. Global
    indices are mapped to/from each bucket's local index space explicitly so
    `greedy_chain_order` never sees an out-of-range or mismatched index.
    """
    dominant = dominant_state_per_tracklet(sequences)
    rank_of = {state: i for i, state in enumerate(resolved_order)}
    unranked_rank = len(resolved_order)
    bucket_rank = np.array([rank_of.get(d, unranked_rank) for d in dominant])

    global_order: list[int] = []
    anchor_global_idx = int(start_idx)
    first = True
    for rank in sorted(set(bucket_rank.tolist())):
        bucket_global_idx = np.flatnonzero(bucket_rank == rank)
        local_index_of_global = {int(g): i for i, g in enumerate(bucket_global_idx)}

        if first and int(start_idx) in local_index_of_global:
            local_start = local_index_of_global[int(start_idx)]
        else:
            if first:
                print(
                    "[tracklet-chain] NOTE: reference tracklet's dominant-state bucket is "
                    "not first in 'bucket' ordering; the anchor will not land at chain "
                    "position 0."
                )
            local_start = int(np.argmin(distances[anchor_global_idx, bucket_global_idx]))

        sub_distances = distances[np.ix_(bucket_global_idx, bucket_global_idx)]
        local_walk = greedy_chain_order(sub_distances, local_start)
        bucket_global_order = [int(bucket_global_idx[i]) for i in local_walk]

        global_order.extend(bucket_global_order)
        anchor_global_idx = bucket_global_order[-1]
        first = False

    return global_order


def penalize_distance_matrix_by_score(distances: np.ndarray, scores: np.ndarray, lam: float) -> np.ndarray:
    """Used by method "penalized_greedy": add `lam * |score_i - score_j|` to `distances`
    so the greedy walk is biased away from jumping back to an already-passed semantic
    region without strictly forbidding it. Pairs touching a NaN-score (unranked) tracklet
    get zero penalty, so those fall back to plain feature/DTW similarity instead of
    poisoning the walk with NaNs.
    """
    diff = np.abs(np.subtract.outer(scores, scores))
    diff = np.nan_to_num(diff, nan=0.0)
    penalized = distances + float(lam) * diff
    return _validate_distance_matrix(penalized, distances.shape[0])


def compute_bouts_feature_distance_matrix(
    adata_windowed,
    meta_df,
    *,
    state_col,
    groupby_cols=("sample_name", "TrackID", "trajectory_window_id"),
    time_col="position_t",
    metric="euclidean",
    selected_feature_blocks=("fractions", "bout_stats", "transitions", "bigrams", "trigrams"),
) -> np.ndarray:
    """Pairwise distance matrix over per-tracklet descriptive feature vectors (state
    fractions, bout stats, transitions, bigrams, trigrams), instead of DTW over the raw
    per-timepoint state sequence.

    Mirrors the CLR + log1p + MFA-block-scaling preprocessing `run_state_based_analysis`
    applies before clustering, so this is "the same feature space the bouts-based
    clustering pipeline works in", just used for pairwise distance instead of Leiden/
    Agglomerative clustering. Rows are reordered to match `meta_df` so the result can be
    dropped in wherever a DTW distance matrix was used.

    `selected_feature_blocks` controls only which already-preprocessed columns are
    concatenated into the final distance computation - the CLR/log1p/MFA-block-scaling
    preprocessing itself always still runs over the full block set, since
    `scale_feature_blocks(mode="mfa")` derives each block's own scale factor from that
    block's own columns and subsetting earlier would change that math for blocks kept.
    """
    groupby_cols = list(groupby_cols)
    adata_feats, _blocks = extract_descibing_track_state_features(
        adata_windowed,
        group_col=tuple(groupby_cols),
        time_col=time_col,
        state_col=state_col,
    )

    feature_blocks = dict(adata_feats.uns.get("feature_blocks", {}))
    fraction_cols = feature_blocks.get("fractions", [])
    bout_cols = feature_blocks.get("bout_stats", [])
    transition_cols = feature_blocks.get("transitions", [])
    bouts_nr_cols = [c for c in bout_cols if str(c).startswith("bouts_nr_")]
    length_cols = [
        c for c in bout_cols
        if str(c).startswith("bouts_mean_length_") or str(c).startswith("bouts_max_length_")
    ]
    states_for_clr = pd.Index(adata_windowed.obs[state_col].astype("category").cat.categories).tolist()

    adata_feats = clr_transform_columns(adata_feats, fraction_cols, pseudocount=1e-3)
    adata_feats = clr_transform_columns(adata_feats, bouts_nr_cols, pseudocount=1e-3)
    adata_feats = clr_transform_transition_rows(adata_feats, transition_cols, states_for_clr, pseudocount=1e-3)
    adata_feats = log1p_transform_columns(adata_feats, length_cols)
    adata_feats = scale_feature_blocks(
        adata_feats,
        blocks=[
            fraction_cols,
            bout_cols,
            transition_cols,
            feature_blocks.get("bigrams", []),
            feature_blocks.get("trigrams", []),
        ],
        mode="mfa",
    )

    valid_block_names = ("fractions", "bout_stats", "transitions", "bigrams", "trigrams")
    requested = [str(b) for b in list(selected_feature_blocks or [])]
    unknown = sorted(set(requested) - set(valid_block_names))
    if unknown:
        raise ValueError(f"Unknown selected_feature_blocks entries {unknown}; valid names: {list(valid_block_names)}")
    if len(requested) == 0:
        raise ValueError("selected_feature_blocks must select at least one feature block.")

    selected_cols = [c for name in valid_block_names if name in requested for c in feature_blocks.get(name, [])]
    if len(selected_cols) == 0:
        raise ValueError(
            f"selected_feature_blocks={requested!r} matched no columns; "
            f"available non-empty blocks: {sorted(k for k, v in feature_blocks.items() if v)}"
        )

    feats_keys = adata_feats.obs[groupby_cols].astype(str)
    row_by_key = {key: i for i, key in enumerate(feats_keys.itertuples(index=False, name=None))}
    meta_keys = meta_df[groupby_cols].astype(str)
    try:
        row_order = [row_by_key[key] for key in meta_keys.itertuples(index=False, name=None)]
    except KeyError as exc:
        raise ValueError(
            f"Could not align bouts feature rows to tracklet metadata; missing key {exc.args[0]!r}."
        ) from exc

    X_full = _to_numpy_2d(adata_feats.X).astype(float, copy=False)[row_order]
    col_idx = adata_feats.var_names.get_indexer(selected_cols)
    if np.any(col_idx < 0):
        missing = [c for c, j in zip(selected_cols, col_idx) if j < 0]
        raise ValueError(f"Selected feature-block columns missing from adata_feats.var_names: {missing[:20]}")
    X = X_full[:, col_idx]

    distances = pairwise_distances(X, metric=str(metric))
    return _validate_distance_matrix(distances, X.shape[0])


class TrackletChainConfigDialog(QDialog):
    """Small Qt dialog for configuring and running the tracklet-similarity-chain plot.

    Loading data (resolving the h5ad, filtering/windowing tracks, extracting per-tracklet
    state sequences) happens once via the "Load data" button; only after that succeeds
    do the state-order list and reference-state dropdown get populated with the real
    category names actually present in this dataset, since those can't be known upfront.
    """

    def __init__(self, defaults: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracklet similarity chain - configuration")
        self.defaults = defaults
        self.adata_windowed = None
        self.sequences = None
        self.meta_df = None
        self.categories_in_data: list[str] = []

        self._build_ui()
        self._wire_signals()
        self._on_similarity_method_changed()
        self._on_ordering_method_changed()
        self._set_loaded_controls_enabled(False)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        data_group = QGroupBox("Data")
        data_form = QFormLayout(data_group)
        self.edit_output_dir = QLineEdit(str(self.defaults["output_dir"] or ""))
        btn_browse_dir = QPushButton("Browse…")
        btn_browse_dir.clicked.connect(self._browse_output_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.edit_output_dir)
        dir_row.addWidget(btn_browse_dir)
        dir_row_widget = QWidget()
        dir_row_widget.setLayout(dir_row)
        data_form.addRow("Output dir:", dir_row_widget)

        self.combo_cell_type = QComboBox()
        self.combo_cell_type.setEditable(True)
        self.combo_cell_type.addItem(str(self.defaults["cell_type"]))
        data_form.addRow("Cell type:", self.combo_cell_type)

        self.spin_tracklet_length = QSpinBox()
        self.spin_tracklet_length.setRange(2, 100000)
        self.spin_tracklet_length.setValue(int(self.defaults["tracklet_length"]))
        data_form.addRow("Tracklet length:", self.spin_tracklet_length)

        self.btn_load = QPushButton("Load data")
        data_form.addRow(self.btn_load)
        layout.addWidget(data_group)

        sim_group = QGroupBox("Similarity")
        sim_form = QFormLayout(sim_group)
        self.combo_similarity_method = QComboBox()
        self.combo_similarity_method.addItems(["bouts", "dtw"])
        self.combo_similarity_method.setCurrentText(str(self.defaults["similarity_method"]))
        sim_form.addRow("Method:", self.combo_similarity_method)

        self.combo_dtw_metric = QComboBox()
        self.combo_dtw_metric.addItems(["dtw_onehot", "transition_profile"])
        self.combo_dtw_metric.setCurrentText(str(self.defaults["dtw_metric"]))
        sim_form.addRow("DTW metric:", self.combo_dtw_metric)

        self.combo_bouts_metric = QComboBox()
        self.combo_bouts_metric.addItems(["euclidean", "cosine"])
        self.combo_bouts_metric.setCurrentText(str(self.defaults["bouts_metric"]))
        sim_form.addRow("Bouts metric:", self.combo_bouts_metric)

        self.bouts_block_checks: dict[str, QCheckBox] = {}
        blocks_widget = QWidget()
        blocks_layout = QHBoxLayout(blocks_widget)
        blocks_layout.setContentsMargins(0, 0, 0, 0)
        for block in ("fractions", "bout_stats", "transitions", "bigrams", "trigrams"):
            cb = QCheckBox(block)
            cb.setChecked(block in self.defaults["bouts_feature_blocks"])
            self.bouts_block_checks[block] = cb
            blocks_layout.addWidget(cb)
        sim_form.addRow("Bouts feature blocks:", blocks_widget)
        layout.addWidget(sim_group)
        self.sim_group = sim_group

        order_group = QGroupBox("Canonical state order (drag to reorder, low → high)")
        order_layout = QVBoxLayout(order_group)
        self.list_state_order = QListWidget()
        self.list_state_order.setDragDropMode(QAbstractItemView.InternalMove)
        order_layout.addWidget(self.list_state_order)
        layout.addWidget(order_group)
        self.order_group = order_group

        ordering_group = QGroupBox("Chain ordering")
        ordering_form = QFormLayout(ordering_group)
        self.combo_ordering_method = QComboBox()
        self.combo_ordering_method.addItems(["greedy", "score", "bucket", "penalized_greedy"])
        self.combo_ordering_method.setCurrentText(str(self.defaults["ordering_method"]))
        ordering_form.addRow("Method:", self.combo_ordering_method)

        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setRange(0.0, 1000.0)
        self.spin_lambda.setDecimals(3)
        self.spin_lambda.setSingleStep(0.1)
        self.spin_lambda.setValue(float(self.defaults["lam"]))
        ordering_form.addRow("Lambda:", self.spin_lambda)

        self.combo_reference_state = QComboBox()
        ordering_form.addRow("Reference state:", self.combo_reference_state)

        self.edit_reference_track = QLineEdit(str(self.defaults.get("reference_track_text", "")))
        self.edit_reference_track.setPlaceholderText(
            "advanced: sample_name,TrackID[,window_id] (overrides reference state)"
        )
        ordering_form.addRow("Reference track override:", self.edit_reference_track)
        layout.addWidget(ordering_group)
        self.ordering_group = ordering_group

        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)
        self.edit_output_path = QLineEdit(str(self.defaults["output_path"]))
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._browse_output_path)
        out_row = QHBoxLayout()
        out_row.addWidget(self.edit_output_path)
        out_row.addWidget(btn_browse_out)
        out_row_widget = QWidget()
        out_row_widget.setLayout(out_row)
        out_form.addRow("PDF path:", out_row_widget)
        layout.addWidget(out_group)

        self.btn_run = QPushButton("Run")
        self.btn_run.setDefault(True)
        btn_close = QPushButton("Close")
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        btn_row.addWidget(self.btn_run)
        layout.addLayout(btn_row)
        self._btn_close = btn_close

    def _wire_signals(self):
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.combo_similarity_method.currentTextChanged.connect(self._on_similarity_method_changed)
        self.combo_ordering_method.currentTextChanged.connect(self._on_ordering_method_changed)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self._btn_close.clicked.connect(self.reject)
        self.edit_output_dir.textChanged.connect(self._mark_stale)
        self.combo_cell_type.editTextChanged.connect(self._mark_stale)
        self.spin_tracklet_length.valueChanged.connect(self._mark_stale)

    def _mark_stale(self, *_args):
        self._set_loaded_controls_enabled(False)

    def _set_loaded_controls_enabled(self, enabled: bool):
        self.sim_group.setEnabled(enabled)
        self.order_group.setEnabled(enabled)
        self.ordering_group.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)

    def _on_similarity_method_changed(self, *_args):
        is_bouts = self.combo_similarity_method.currentText() == "bouts"
        self.combo_bouts_metric.setEnabled(is_bouts)
        for cb in self.bouts_block_checks.values():
            cb.setEnabled(is_bouts)
        self.combo_dtw_metric.setEnabled(not is_bouts)

    def _on_ordering_method_changed(self, *_args):
        self.spin_lambda.setEnabled(self.combo_ordering_method.currentText() == "penalized_greedy")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select BEHAV3D output folder", self.edit_output_dir.text())
        if path:
            self.edit_output_dir.setText(path)

    def _browse_output_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save chain plot as", self.edit_output_path.text(), "PDF files (*.pdf)"
        )
        if path:
            self.edit_output_path.setText(path)

    def _on_load_clicked(self):
        try:
            output_dir = resolve_behav3d_output_dir(None, self.edit_output_dir.text())
            cell_type = self.combo_cell_type.currentText().strip()
            tracklet_length = int(self.spin_tracklet_length.value())
            adata_windowed, sequences, meta_df, categories_in_data = load_tracklet_data(
                output_dir, cell_type, tracklet_length, STATE_COL
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load data", str(exc))
            return

        self.adata_windowed = adata_windowed
        self.sequences = sequences
        self.meta_df = meta_df
        self.categories_in_data = categories_in_data

        # Preserve any manual drag-reordering across a reload; only fall back to the
        # module-constant default the first time.
        current_order = [self.list_state_order.item(i).text() for i in range(self.list_state_order.count())]
        seed_order = current_order if current_order else list(self.defaults["state_order"])
        resolved_order, unknown = resolve_semantic_state_order(categories_in_data, seed_order)

        self.list_state_order.clear()
        for state in resolved_order:
            self.list_state_order.addItem(QListWidgetItem(state))

        if unknown:
            QMessageBox.information(
                self,
                "Unlisted states found",
                "These states are present in the data but weren't in the canonical order; "
                "appended at the end - drag to reposition:\n" + ", ".join(unknown),
            )

        previous_ref = self.combo_reference_state.currentText()
        self.combo_reference_state.clear()
        self.combo_reference_state.addItems(resolved_order)
        default_ref = previous_ref or str(self.defaults.get("reference_state_label") or "")
        if default_ref in resolved_order:
            self.combo_reference_state.setCurrentText(default_ref)

        self._set_loaded_controls_enabled(True)
        print(f"[tracklet-chain] loaded {len(sequences)} tracklets of length {tracklet_length}")

    def _on_run_clicked(self):
        """Run one plot with the current config without closing the dialog, so the same
        loaded data can be re-run with tweaked settings as many times as wanted."""
        if self.combo_similarity_method.currentText() == "bouts" and not any(
            cb.isChecked() for cb in self.bouts_block_checks.values()
        ):
            QMessageBox.warning(self, "No feature blocks selected", "Select at least one bouts feature block.")
            return

        config = self.get_config()
        self.btn_run.setEnabled(False)
        try:
            output_path = run_tracklet_chain_analysis(
                config, self.adata_windowed, self.sequences, self.meta_df
            )
        except Exception as exc:
            QMessageBox.critical(self, "Run failed", str(exc))
            return
        finally:
            self.btn_run.setEnabled(True)
        print(f"[tracklet-chain] saved {output_path}")

    def get_config(self) -> dict:
        state_order = [self.list_state_order.item(i).text() for i in range(self.list_state_order.count())]
        bouts_feature_blocks = tuple(name for name, cb in self.bouts_block_checks.items() if cb.isChecked())

        reference_track_text = self.edit_reference_track.text().strip()
        reference_track = tuple(p.strip() for p in reference_track_text.split(",") if p.strip()) or None

        return dict(
            similarity_method=self.combo_similarity_method.currentText(),
            dtw_metric=self.combo_dtw_metric.currentText(),
            bouts_metric=self.combo_bouts_metric.currentText(),
            bouts_feature_blocks=bouts_feature_blocks,
            state_order=state_order,
            ordering_method=self.combo_ordering_method.currentText(),
            lam=float(self.spin_lambda.value()),
            reference_track=reference_track,
            reference_state_label=(None if reference_track else self.combo_reference_state.currentText()),
            output_path=self.edit_output_path.text(),
        )


def run_tracklet_chain_analysis(config: dict, adata_windowed, sequences, meta_df) -> str:
    """Compute one tracklet-similarity chain plot for `config` and save it to
    `config['output_path']`. Returns the path written.

    Pulled out of `main()` so the dialog's "Run" button can call this directly and stay
    open afterwards, instead of the dialog having to close (as a QDialog.accept() would)
    before the plot could be produced.
    """
    if config["similarity_method"] == "dtw":
        if config["dtw_metric"] == "dtw_onehot":
            distances, _categories = compute_dtaidistance_onehot_distance_matrix(sequences)
        elif config["dtw_metric"] == "transition_profile":
            distances, _vocab = compute_transition_profile_distance_matrix(sequences)
        else:
            raise ValueError(f"Unknown dtw metric: {config['dtw_metric']!r}")
        similarity_label = config["dtw_metric"]
    elif config["similarity_method"] == "bouts":
        distances = compute_bouts_feature_distance_matrix(
            adata_windowed,
            meta_df,
            state_col=STATE_COL,
            groupby_cols=("sample_name", "TrackID", "trajectory_window_id"),
            time_col="position_t",
            metric=config["bouts_metric"],
            selected_feature_blocks=config["bouts_feature_blocks"],
        )
        similarity_label = f"bouts_{config['bouts_metric']}"
    else:
        raise ValueError(f"Unknown similarity method: {config['similarity_method']!r}")

    resolved_order = config["state_order"]
    _set_classification_state_order(adata_windowed, STATE_COL, resolved_order)

    scores = compute_semantic_state_scores(sequences, resolved_order)
    n_nan_scores = int(np.isnan(scores).sum())
    if n_nan_scores > 0:
        print(f"[tracklet-chain] WARNING: {n_nan_scores} tracklet(s) have zero ranked frames and got score=NaN.")

    start_idx = resolve_reference_index(
        sequences,
        meta_df,
        reference_track=config["reference_track"],
        reference_state_label=config["reference_state_label"],
    )

    ordering_method = config["ordering_method"]
    if ordering_method == "greedy":
        chain_order = greedy_chain_order(distances, start_idx)
    elif ordering_method == "score":
        chain_order = order_by_semantic_score(distances, scores, start_idx)
        if chain_order[0] != start_idx:
            print(
                f"[tracklet-chain] NOTE: 'score' ordering placed the reference tracklet at "
                f"chain position {chain_order.index(start_idx)}, not 0."
            )
    elif ordering_method == "bucket":
        chain_order = order_by_dominant_state_buckets(distances, sequences, resolved_order, start_idx)
    elif ordering_method == "penalized_greedy":
        chain_order = greedy_chain_order(
            penalize_distance_matrix_by_score(distances, scores, config["lam"]), start_idx
        )
    else:
        raise ValueError(f"Unknown ordering method: {ordering_method!r}")

    # `plot_tracks_bars_on_ax` draws row i of chosen_df at y0 = i * (bar_h +
    # y_gap), so row 0 already lands at the bottom of the plot - no reversal
    # needed to put the reference tracklet at the bottom.
    chosen_df = meta_df.iloc[chain_order].reset_index(drop=True)

    state_values, color_map = _build_state_color_map(adata_windowed, STATE_COL)

    n_tracklets = len(chosen_df)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.18 * n_tracklets)))
    plot_tracks_bars_on_ax(
        adata_windowed,
        chosen_df,
        ax,
        sample_key="sample_name",
        track_key="TrackID",
        time_key="position_t",
        state_key=STATE_COL,
        x_mode="relative",
        window_key="trajectory_window_id",
        state_color_map=color_map,
        title=f"{n_tracklets} tracklets, {ordering_method} ordering ({similarity_label})",
    )
    legend_handles = [
        Patch(facecolor=color_map.get(str(v), "grey"), edgecolor="k", label=str(v))
        for v in state_values
    ]
    fig.legend(
        handles=legend_handles,
        title=STATE_COL,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    output_path = config["output_path"]
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    defaults = dict(
        output_dir=INTERACTIVE_OUTPUT_DIR or INTERACTIVE_BEHAV3D_FOLDER or "",
        cell_type=INTERACTIVE_CELL_TYPE,
        tracklet_length=TRACKLET_LENGTH,
        similarity_method=TRACK_SIMILARITY_METHOD,
        dtw_metric=DTW_DISTANCE_METRIC,
        bouts_metric=BOUTS_DISTANCE_METRIC,
        bouts_feature_blocks=BOUTS_FEATURE_BLOCKS,
        state_order=STATE_ORDER,
        ordering_method=ORDERING_METHOD,
        lam=LAMBDA,
        reference_state_label=REFERENCE_STATE_LABEL,
        reference_track_text=(",".join(str(p) for p in REFERENCE_TRACK) if REFERENCE_TRACK else ""),
        output_path=OUTPUT_PATH,
    )

    dialog = TrackletChainConfigDialog(defaults)
    dialog.exec_()


if __name__ == "__main__":
    main()

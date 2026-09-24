"""Pooled inter-cluster transition analysis for trajectory windows: a circular transition
diagram plus the row-normalized transition-probability matrix, both built from the same pooled
window-to-window trajectory-cluster counts that :mod:`window_transitions` already computes.

Where :mod:`window_transitions`'s Sankey keeps each track's own window sequence intact (answering
"when does this track switch clusters?"), this module collapses window index entirely and asks
the population-level question :mod:`behav3d.analysis.behavior.state.visualization.plots.
state_transitions` already answers for per-timepoint states: pooled across every track and every
window-to-window transition, how often does cluster A become cluster B?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from behav3d.analysis.behavior.utils import _natural_sort_key
from behav3d.analysis.behavior.state.utils import (
    _apply_state_order,
    _get_classification_state_colors,
    _get_classification_state_order,
    _normalize_label_color_map,
)
from behav3d.analysis.behavior.state.visualization.plots.state_transitions import (
    _plot_transition_heatmap_quad_page,
)
from behav3d.analysis.behavior.general.visualization.plots.circular_transition_diagram import (
    plot_circular_transition_diagram,
    plot_circular_transition_diagram_grid,
)
from behav3d.analysis.behavior.track.utils import _resolve_track_paths, _winfo
from behav3d.analysis.behavior.track.visualization.plots.window_transitions import (
    compute_window_transition_links,
)


def compute_window_cluster_transition_matrix(
    links_pooled_df,
    *,
    state_order=None,
    only_transitions=False,
):
    """Collapse pooled window-to-window transition counts into a cluster x cluster matrix.

    `links_pooled_df` is the "pooled" table from `compute_window_transition_links` (columns
    window_from, window_to, cluster_from, cluster_to, count). Which window-index pair a
    transition happened at is deliberately summed away here - the question this answers is
    "cluster A -> cluster B", not "window 3 -> window 4". Mirrors
    `state_transitions.compute_cluster_transition_matrix`'s crosstab/normalize logic exactly,
    just starting from already-aggregated counts instead of raw per-timepoint rows.

    Parameters
    ----------
    links_pooled_df : pandas.DataFrame
        Columns cluster_from, cluster_to, count (window_from/window_to are summed over).
    state_order : list of str, optional
        Saved display order for the clusters (e.g. from `_get_classification_state_order`).
        Clusters not present in this list are appended afterwards.
    only_transitions : bool, default False
        If True, zero the diagonal (self-transitions) in the *returned* matrices and
        renormalize each row over the remaining off-diagonal entries only.

    Returns
    -------
    transition_counts, transition_probs : pandas.DataFrame
        Square cluster x cluster matrices, rows = current cluster, columns = next cluster.
        `transition_probs` is row-normalized; if `only_transitions=True` it is
        P(next | current, next != current).
    """
    df = links_pooled_df.copy()
    df["cluster_from"] = df["cluster_from"].astype(str)
    df["cluster_to"] = df["cluster_to"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    clusters_raw = sorted(set(df["cluster_from"]) | set(df["cluster_to"]), key=_natural_sort_key)
    clusters = _apply_state_order(clusters_raw, state_order)

    aggregated = df.groupby(["cluster_from", "cluster_to"], observed=True)["count"].sum()
    transition_counts = aggregated.unstack(fill_value=0.0).reindex(
        index=clusters, columns=clusters, fill_value=0.0
    )

    if only_transitions:
        counts_arr = transition_counts.to_numpy(copy=True)
        np.fill_diagonal(counts_arr, 0.0)
        transition_counts = pd.DataFrame(
            counts_arr, index=transition_counts.index, columns=transition_counts.columns,
        )

    row_sums = transition_counts.sum(axis=1)
    transition_probs = transition_counts.div(row_sums.replace(0, np.nan), axis=0)

    return transition_counts, transition_probs


def save_window_cluster_transition_analysis(
    adata_tracks,
    output_dir,
    cell_type,
    *,
    cluster_key=None,
    window_col="trajectory_window_id",
    id_cols=("sample_name", "TrackID"),
    sample_col="sample_name",
    min_prob_to_draw=0.03,
    emphasis_gamma=2.0,
    curvature=0.28,
    label_style="on_node",
    include_transition_matrix=True,
    include_circular_diagram=True,
    circular_include_self_transitions=False,
    include_circular_diagram_per_cluster=True,
    state_colors=None,
    state_order=None,
    verbose=True,
):
    """Save the pooled inter-cluster transition analysis: a transition-matrix heatmap page (the
    same quad layout `state_transitions.save_state_transition_report` uses) plus a circular
    inter-cluster transition diagram page - both built by collapsing window index out of
    `compute_window_transition_links`'s pooled counts.

    Unlike `save_window_transition_report` (per-track window sequence, kept intact, one Sankey
    per sample), this pools every sample and every window transition into one population-level
    view, mirroring the granularity of `save_state_transition_report`.
    """
    meta = adata_tracks.uns.get("dtai_trajectory_clustering", {})
    meta = meta if isinstance(meta, dict) else {}
    if cluster_key is None:
        cluster_key = str(meta.get("cluster_key", "ClusterID"))
    if state_order is None:
        state_order = _get_classification_state_order(adata_tracks, cluster_key)
    if state_colors is None:
        state_colors = _get_classification_state_colors(adata_tracks, cluster_key)

    paths = _resolve_track_paths(output_dir, cell_type)
    out_dir = paths.outfolder / "transition_analysis"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    links = compute_window_transition_links(
        adata_tracks, cluster_key, window_col=window_col, id_cols=id_cols, sample_col=sample_col,
    )
    pooled = links["pooled"]

    counts, probs = compute_window_cluster_transition_matrix(
        pooled, state_order=state_order, only_transitions=False,
    )
    counts_no_self, probs_no_self = compute_window_cluster_transition_matrix(
        pooled, state_order=state_order, only_transitions=True,
    )

    counts_csv = data_dir / "transition_matrix_counts.csv"
    probs_csv = data_dir / "transition_matrix_probs.csv"
    counts_no_self_csv = data_dir / "transition_matrix_counts_no_self.csv"
    probs_no_self_csv = data_dir / "transition_matrix_probs_no_self.csv"
    probs_incoming_csv = data_dir / "transition_matrix_probs_incoming.csv"
    counts.to_csv(counts_csv)
    probs.to_csv(probs_csv)
    counts_no_self.to_csv(counts_no_self_csv)
    probs_no_self.to_csv(probs_no_self_csv)

    colors = _normalize_label_color_map(list(probs_no_self.index), colors=state_colors)

    pdf_path = out_dir / "transition_analysis.pdf"
    with PdfPages(pdf_path) as pdf:
        if include_transition_matrix:
            fig_quad = _plot_transition_heatmap_quad_page(
                probs, probs_no_self, counts, counts_no_self, state_col=str(cluster_key),
            )
            pdf.savefig(fig_quad, bbox_inches="tight")
            plt.close(fig_quad)

        if include_circular_diagram:
            # Self-transitions are never drawn as arrows; `circular_include_self_transitions`
            # instead picks which matrix feeds the diagram - see the matching comment in
            # `state_transitions.save_state_transition_report`.
            circular_counts = counts if circular_include_self_transitions else counts_no_self
            circular_probs = probs if circular_include_self_transitions else probs_no_self
            fig_circular = plot_circular_transition_diagram(
                circular_probs,
                state_colors=colors,
                state_order=state_order,
                title=f"{cell_type} — inter-cluster transition probability ({cluster_key})",
                min_prob_to_draw=min_prob_to_draw,
                emphasis_gamma=emphasis_gamma,
                curvature=curvature,
                label_style=label_style,
            )
            pdf.savefig(fig_circular, bbox_inches="tight")
            plt.close(fig_circular)

            # Absolute-scale counterpart of `fig_circular`, added alongside (not replacing) it
            # for comparison: probability maps directly onto arc thickness on a fixed 0-1 scale,
            # instead of being stretched relative to this diagram's own strongest edge (which is
            # often a large self-transition elsewhere in the matrix that would otherwise flatten
            # every other cluster's edges by comparison).
            fig_circular_absolute = plot_circular_transition_diagram(
                circular_probs,
                state_colors=colors,
                state_order=state_order,
                title=f"{cell_type} — inter-cluster transition probability, absolute scale ({cluster_key})",
                min_prob_to_draw=min_prob_to_draw,
                emphasis_gamma=emphasis_gamma,
                curvature=curvature,
                label_style=label_style,
                scale_mode="absolute",
            )
            pdf.savefig(fig_circular_absolute, bbox_inches="tight")
            plt.close(fig_circular_absolute)

            # Column-normalized counterpart of `circular_probs`: for each destination cluster,
            # what share of its arrivals came from each source. Must be derived from the counts
            # (not from the already row-normalized `circular_probs`), since normalizing an
            # already-normalized matrix's columns would not equal each source's true share.
            col_sums = circular_counts.sum(axis=0)
            circular_probs_incoming = circular_counts.div(col_sums.replace(0, np.nan), axis=1)
            circular_probs_incoming.to_csv(probs_incoming_csv)
            fig_circular_incoming = plot_circular_transition_diagram(
                circular_probs_incoming,
                state_colors=colors,
                state_order=state_order,
                title=f"{cell_type} — inter-cluster transition probability, incoming ({cluster_key})",
                min_prob_to_draw=min_prob_to_draw,
                emphasis_gamma=emphasis_gamma,
                curvature=curvature,
                label_style=label_style,
            )
            pdf.savefig(fig_circular_incoming, bbox_inches="tight")
            plt.close(fig_circular_incoming)

            fig_circular_incoming_absolute = plot_circular_transition_diagram(
                circular_probs_incoming,
                state_colors=colors,
                state_order=state_order,
                title=(
                    f"{cell_type} — inter-cluster transition probability, incoming, "
                    f"absolute scale ({cluster_key})"
                ),
                min_prob_to_draw=min_prob_to_draw,
                emphasis_gamma=emphasis_gamma,
                curvature=curvature,
                label_style=label_style,
                scale_mode="absolute",
            )
            pdf.savefig(fig_circular_incoming_absolute, bbox_inches="tight")
            plt.close(fig_circular_incoming_absolute)

            if include_circular_diagram_per_cluster:
                fig_circular_grid_out = plot_circular_transition_diagram_grid(
                    circular_probs,
                    state_colors=colors,
                    state_order=state_order,
                    title=f"{cell_type} — per-cluster outgoing transitions ({cluster_key})",
                    min_prob_to_draw=min_prob_to_draw,
                    emphasis_gamma=emphasis_gamma,
                    curvature=curvature,
                    label_style=label_style,
                    direction="outgoing",
                )
                pdf.savefig(fig_circular_grid_out, bbox_inches="tight")
                plt.close(fig_circular_grid_out)

                fig_circular_grid_in = plot_circular_transition_diagram_grid(
                    circular_probs_incoming,
                    state_colors=colors,
                    state_order=state_order,
                    title=f"{cell_type} — per-cluster incoming transitions ({cluster_key})",
                    min_prob_to_draw=min_prob_to_draw,
                    emphasis_gamma=emphasis_gamma,
                    curvature=curvature,
                    label_style=label_style,
                    direction="incoming",
                )
                pdf.savefig(fig_circular_grid_in, bbox_inches="tight")
                plt.close(fig_circular_grid_in)

    if verbose:
        _winfo("trajectory-dtai", f"saved transition matrix counts CSV: {counts_csv}")
        _winfo("trajectory-dtai", f"saved transition matrix probabilities CSV: {probs_csv}")
        _winfo("trajectory-dtai", f"saved transition analysis PDF: {pdf_path}")

    return {
        "output_dir": str(out_dir),
        "cluster_key": str(cluster_key),
        "transition_matrix_counts_csv": str(counts_csv),
        "transition_matrix_probs_csv": str(probs_csv),
        "transition_matrix_counts_no_self_csv": str(counts_no_self_csv),
        "transition_matrix_probs_no_self_csv": str(probs_no_self_csv),
        "transition_analysis_pdf": str(pdf_path),
    }

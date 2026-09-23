"""Generic circular ("chord") inter-cluster transition diagram.

Shared by :mod:`behav3d.analysis.behavior.state.visualization.plots.state_transitions`
(per-timepoint state transitions) and
:mod:`behav3d.analysis.behavior.track.visualization.plots.transition_analysis` (pooled
window-to-window trajectory-cluster transitions) so the drawing/visual-weighting logic exists in
exactly one place. Lives under `general/` (rather than `state/` or `track/`) because both of
those sibling modules need it and neither may import the other's plotting internals for this.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from behav3d.analysis.behavior.state.utils import _apply_state_order, _normalize_label_color_map


def _circular_node_positions(clusters, radius=1.0):
    """(x, y) for each cluster, evenly spaced on a circle starting at 12 o'clock, clockwise."""
    n = len(clusters)
    positions = {}
    for i, cluster in enumerate(clusters):
        theta = np.deg2rad(-90.0 + i * (360.0 / n))
        positions[cluster] = (radius * np.cos(theta), radius * np.sin(theta))
    return positions


def _draw_circular_transition_diagram_on_ax(
    ax,
    probs_df,
    *,
    colors,
    min_prob_to_draw=0.03,
    emphasis_gamma=2.0,
    min_linewidth=0.5,
    max_linewidth=7.0,
    min_alpha=0.12,
    max_alpha=0.95,
    node_radius=0.09,
    curvature=0.28,
    label_fontsize=11,
    label_style="on_node",
    source_cluster=None,
):
    """Draw a circular inter-cluster transition diagram on `ax`.

    One node per `probs_df` row/column, evenly spaced clockwise from 12 o'clock. One curved
    directed arrow per off-diagonal (from, to) pair with probability >= `min_prob_to_draw`,
    colored by the source cluster. Self-transitions are never drawn (this view is explicitly
    inter-cluster) even if `probs_df` has a nonzero diagonal.

    Visual weight (linewidth/alpha) is deliberately non-linear in probability so rare
    transitions fade out instead of cluttering the plot. `min_prob_to_draw` drops anything below
    the threshold entirely - not just faintly, gone. Among the edges that remain, probability is
    normalized to [0, 1] against the strongest drawn edge and raised to `emphasis_gamma` (>1
    bends the low end down hard) before mapping onto the linewidth/alpha ranges, so an edge just
    above the threshold renders as a thin, faint hairline while only the dominant transitions
    read as bold arcs.

    `label_style="on_node"` (default) draws each cluster's full label inside its dot. With
    `label_style="legend"`, dots are labeled with their 1-based clockwise index instead (short,
    never cramped) - the caller is expected to draw the index-to-label legend itself (this
    function only has an `ax`, not the figure a legend would need to sit outside of); see
    `plot_circular_transition_diagram`.

    `source_cluster`, when given, restricts the drawn edges to that one source: only its
    outgoing transitions are drawn (every node still renders, for context). Because `max_p`/
    `span` below are computed from the edges actually being drawn, this also rescales the
    visual weight to that cluster's own transitions rather than the whole matrix's - the
    strongest edge *out of this cluster* reads as the boldest arc, not the strongest edge
    overall. See `plot_circular_transition_diagram_grid`, which draws one such panel per
    cluster on a single page.
    """
    clusters = list(probs_df.index)
    positions = _circular_node_positions(clusters)

    edges = []
    for src in clusters:
        if source_cluster is not None and src != source_cluster:
            continue
        for dst in clusters:
            if src == dst:
                continue
            p = float(probs_df.loc[src, dst])
            if np.isfinite(p) and p >= min_prob_to_draw:
                edges.append((src, dst, p))

    if edges:
        max_p = max(p for _, _, p in edges)
        span = max(max_p - min_prob_to_draw, 1e-9)

        def visual_weight(p):
            t = (p - min_prob_to_draw) / span
            t = min(max(t, 0.0), 1.0)
            return t ** emphasis_gamma

        # Weakest first, so strong arcs always render on top of faint ones, never the reverse.
        for src, dst, p in sorted(edges, key=lambda e: e[2]):
            t = visual_weight(p)
            linewidth = min_linewidth + t * (max_linewidth - min_linewidth)
            alpha = min_alpha + t * (max_alpha - min_alpha)
            color = colors[src]

            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            dx, dy = x1 - x0, y1 - y0
            dist = float(np.hypot(dx, dy))
            ux, uy = dx / dist, dy / dist
            # Start/end just outside each node's circle so the stroke & arrowhead don't dip
            # under it; the target gets extra clearance to leave room for the arrowhead itself.
            start = (x0 + ux * node_radius, y0 + uy * node_radius)
            end = (x1 - ux * node_radius * 1.6, y1 - uy * node_radius * 1.6)

            arrow = FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>",
                mutation_scale=10 + 8 * t,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                shrinkA=0, shrinkB=0,
                zorder=2,
            )
            ax.add_patch(arrow)

    for i, cluster in enumerate(clusters):
        x, y = positions[cluster]
        ax.add_patch(Circle(
            (x, y), node_radius, facecolor=colors[cluster],
            edgecolor="white", linewidth=1.2, zorder=3,
        ))
        node_text = str(cluster) if label_style == "on_node" else str(i + 1)
        ax.text(
            x, y, node_text, ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold", color="white", zorder=4,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="black", alpha=0.55)],
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def _prepare_clusters_and_colors(probs_df, state_colors, state_order):
    """Shared setup for both public entry points: apply the display order, reindex `probs_df`
    to match, and resolve the cluster -> color map. Pulled out so
    `plot_circular_transition_diagram` and `plot_circular_transition_diagram_grid` can't drift
    apart on ordering/coloring even though each builds a different figure layout around it.
    """
    clusters = _apply_state_order([str(c) for c in probs_df.index], state_order)
    probs_df = probs_df.reindex(index=clusters, columns=clusters)
    colors = _normalize_label_color_map(clusters, colors=state_colors)
    return clusters, probs_df, colors


def _legend_handles(clusters, colors):
    return [
        Patch(facecolor=colors[cluster], edgecolor="none", label=f"{i + 1}: {cluster}")
        for i, cluster in enumerate(clusters)
    ]


def plot_circular_transition_diagram(
    probs_df,
    *,
    state_colors=None,
    state_order=None,
    title=None,
    min_prob_to_draw=0.03,
    emphasis_gamma=2.0,
    curvature=0.28,
    label_style="on_node",
    figsize=(7, 7),
):
    """Standalone figure: circular inter-cluster transition diagram.

    See `_draw_circular_transition_diagram_on_ax` for the drawing/visual-weight semantics.
    `probs_df` is typically a no-self transition-probability matrix (self-transitions zeroed
    and rows renormalized over the remaining off-diagonal entries).

    `label_style="legend"` moves the full cluster/state names off the (now numbered) dots and
    into a legend beside the plot instead - useful for long names or many clusters, where in-dot
    text gets cramped. No `rect`/`tight_layout` adjustment is needed for the legend to fit: every
    caller in this codebase saves figures with `bbox_inches="tight"`, which already expands the
    saved bounding box to include a legend placed outside the axes via `bbox_to_anchor`.
    """
    clusters, probs_df, colors = _prepare_clusters_and_colors(probs_df, state_colors, state_order)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_circular_transition_diagram_on_ax(
        ax, probs_df, colors=colors,
        min_prob_to_draw=min_prob_to_draw, emphasis_gamma=emphasis_gamma, curvature=curvature,
        label_style=label_style,
    )
    if title:
        ax.set_title(str(title), fontsize=12, fontweight="bold", pad=12)
    if label_style == "legend":
        fig.legend(
            handles=_legend_handles(clusters, colors), loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
        )
    fig.tight_layout()
    return fig


def plot_circular_transition_diagram_grid(
    probs_df,
    *,
    state_colors=None,
    state_order=None,
    title=None,
    min_prob_to_draw=0.03,
    emphasis_gamma=2.0,
    curvature=0.28,
    label_style="on_node",
    ncols=None,
    panel_size=3.0,
):
    """Grid figure: one panel per cluster, each showing only that cluster's outgoing
    transitions (see `source_cluster` on `_draw_circular_transition_diagram_on_ax`).

    Laying every cluster's outgoing-only view side by side - rather than the single overlaid
    `plot_circular_transition_diagram`, or one page per cluster - keeps the population-wide
    circular diagram readable while still letting each cluster's own transition pattern be
    inspected at a glance, all on one page. Each panel's visual weight (arrow thickness/alpha)
    is scaled to that cluster's own transitions, not the global maximum - a cluster that mostly
    stays put still shows its strongest available switch as a bold arc within its own panel.

    Panels are arranged in a roughly square grid (`ncols` columns, computed as `ceil(sqrt(n))`
    when not given) so the page stays close to square regardless of cluster count; unused
    trailing panels (when `n` doesn't fill the last row) are hidden rather than left blank-axed.

    `label_fontsize` for on-node labels is deliberately smaller here than
    `plot_circular_transition_diagram`'s default, since `panel_size` shrinks the node circles'
    on-screen size (a fixed *data*-coordinate radius) while text stays a fixed *point* size -
    without this, full cluster names would increasingly overflow their dots as panel count (and
    thus grid density) grows. `label_style="legend"` sidesteps the issue entirely and is the
    better choice for many/long-named clusters.
    """
    clusters, probs_df, colors = _prepare_clusters_and_colors(probs_df, state_colors, state_order)
    n = len(clusters)

    ncols = int(ncols) if ncols else max(1, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * panel_size, nrows * panel_size), squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, cluster in zip(axes_flat, clusters):
        _draw_circular_transition_diagram_on_ax(
            ax, probs_df, colors=colors, source_cluster=cluster,
            min_prob_to_draw=min_prob_to_draw, emphasis_gamma=emphasis_gamma, curvature=curvature,
            label_style=label_style, label_fontsize=8,
        )
        panel_title = str(cluster) if label_style == "on_node" else f"{clusters.index(cluster) + 1}: {cluster}"
        ax.set_title(panel_title, fontsize=9, fontweight="bold", pad=4)
    for ax in axes_flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(str(title), fontsize=12, fontweight="bold")
    if label_style == "legend":
        fig.legend(
            handles=_legend_handles(clusters, colors), loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
        )
    fig.tight_layout()
    return fig

"""Generic circular ("chord") inter-cluster transition diagram.

Shared by :mod:`behav3d.analysis.behavior.state.visualization.plots.state_transitions`
(per-timepoint state transitions) and
:mod:`behav3d.analysis.behavior.track.visualization.plots.transition_analysis` (pooled
window-to-window trajectory-cluster transitions) so the drawing/visual-weighting logic exists in
exactly one place. Lives under `general/` (rather than `state/` or `track/`) because both of
those sibling modules need it and neither may import the other's plotting internals for this.
"""
from functools import lru_cache
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from behav3d.analysis.behavior.state.utils import _apply_state_order, _normalize_label_color_map

_NODE_RADIUS = 0.09
# How far an arrow stops short of its target node, in node radii: leaves room for the arrowhead.
_ARROW_END_CLEARANCE = 1.6
_DEFAULT_CURVATURE = 0.28

# Node order search: every order is tried up to this many nodes (8! orders, since rotating the
# whole diagram changes nothing); above it, a swap search runs until the budget is used.
_MAX_EXHAUSTIVE_NODES = 9
_SEARCH_LOOKUP_BUDGET = 1_000_000_000
_SEARCH_SWAP_BATCH = 16
_SEARCH_RANDOM_RESTARTS = 100
_CROSSING_CURVE_SEGMENTS = 24
_SCORE_CHUNK_SIZE = 4096

# Clutter is crossings plus overlap: two arrows running on top of each other are much harder to
# read than two that cross cleanly, but counting only crossings rewards stacking them. A pair of
# arrows closer than this distance (about the width of the thickest arrow) over their full length
# counts as this many crossings. The weight was picked by comparing diagrams of real datasets.
_OVERLAP_DISTANCE = 0.05
_OVERLAP_WEIGHT = 5
_CLUTTER_UNITS_PER_CROSSING = 100


def _circular_node_positions(clusters, radius=1.0):
    """(x, y) for each cluster, evenly spaced on a circle starting at the bottom, counter-clockwise."""
    n = len(clusters)
    positions = {}
    for i, cluster in enumerate(clusters):
        theta = np.deg2rad(-90.0 + i * (360.0 / n))
        positions[cluster] = (radius * np.cos(theta), radius * np.sin(theta))
    return positions


def _arrow_endpoints(src_pos, dst_pos, node_radius=_NODE_RADIUS):
    """Where the arrow between two node centers starts and ends: just outside each node's circle
    so the stroke & arrowhead don't dip under it, with extra clearance at the target for the
    arrowhead itself."""
    x0, y0 = src_pos
    x1, y1 = dst_pos
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.hypot(dx, dy))
    ux, uy = dx / dist, dy / dist
    start = (x0 + ux * node_radius, y0 + uy * node_radius)
    end = (x1 - ux * node_radius * _ARROW_END_CLEARANCE, y1 - uy * node_radius * _ARROW_END_CLEARANCE)
    return start, end


def _arrow_curve(src_pos, dst_pos, curvature):
    """Points along the arrow between two node centers: the `arc3` curve `FancyArrowPatch` draws."""
    start, end = (np.asarray(p) for p in _arrow_endpoints(src_pos, dst_pos))
    dx, dy = end - start
    control = (start + end) / 2 + curvature * np.array([dy, -dx])
    t = np.linspace(0, 1, _CROSSING_CURVE_SEGMENTS + 1)[:, None]
    return (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t ** 2 * end


def _count_curve_crossings(curve_a, curve_b):
    """Number of times two curves, given as points along them, cross each other."""
    seg_a = np.diff(curve_a, axis=0)[:, None, :]
    seg_b = np.diff(curve_b, axis=0)[None, :, :]
    offset = curve_b[:-1][None, :, :] - curve_a[:-1][:, None, :]
    denominator = seg_a[..., 0] * seg_b[..., 1] - seg_a[..., 1] * seg_b[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        along_a = (offset[..., 0] * seg_b[..., 1] - offset[..., 1] * seg_b[..., 0]) / denominator
        along_b = (offset[..., 0] * seg_a[..., 1] - offset[..., 1] * seg_a[..., 0]) / denominator
    return int(np.sum((along_a >= 0) & (along_a < 1) & (along_b >= 0) & (along_b < 1)))


def _curve_overlap(curve_a, curve_b):
    """Share of two curves' length, from 0 to 1, that runs within `_OVERLAP_DISTANCE` of the other."""
    distances = np.linalg.norm(curve_a[:, None, :] - curve_b[None, :, :], axis=2)
    close_a = (distances.min(axis=1) < _OVERLAP_DISTANCE).mean()
    close_b = (distances.min(axis=0) < _OVERLAP_DISTANCE).mean()
    return (close_a + close_b) / 2


@lru_cache(maxsize=8)
def _arrow_pair_tables(n, curvature):
    """(crossings, overlap) between every two possible arrows, indexed by
    `from_position * n + to_position`.

    Rotating the whole diagram changes neither, so only arrows leaving position 0 are compared
    against all others; every other pair is looked up by rotating it onto one of those.
    """
    positions = _circular_node_positions(list(range(n)))
    curves = {
        (src, dst): _arrow_curve(positions[src], positions[dst], curvature)
        for src in range(n) for dst in range(n) if src != dst
    }
    crossings_from_zero = np.zeros((n, n, n), dtype=np.uint16)
    overlap_from_zero = np.zeros((n, n, n))
    for k in range(1, n):
        for (src, dst), curve in curves.items():
            if (src, dst) != (0, k):
                crossings_from_zero[k, src, dst] = _count_curve_crossings(curves[(0, k)], curve)
                overlap_from_zero[k, src, dst] = _curve_overlap(curves[(0, k)], curve)

    other_src, other_dst = np.divmod(np.arange(n * n), n)
    crossings = np.zeros((n * n, n * n), dtype=np.uint16)
    overlap = np.zeros((n * n, n * n))
    for src, dst in curves:
        rotated = ((dst - src) % n, (other_src - src) % n, (other_dst - src) % n)
        crossings[src * n + dst] = crossings_from_zero[rotated]
        overlap[src * n + dst] = overlap_from_zero[rotated]
    return np.maximum(crossings, crossings.T), np.maximum(overlap, overlap.T)


@lru_cache(maxsize=8)
def _clutter_table(n, curvature):
    """Crossings plus weighted overlap between every two possible arrows, as whole numbers."""
    crossings, overlap = _arrow_pair_tables(n, curvature)
    return (
        crossings.astype(np.int32) * _CLUTTER_UNITS_PER_CROSSING
        + np.round(overlap * _OVERLAP_WEIGHT * _CLUTTER_UNITS_PER_CROSSING).astype(np.int32)
    )


def _drawn_arrows(probs_df, labels, min_prob):
    """(from, to) indices into `labels` of the arrows drawn at this cutoff; self-loops can't cross."""
    values = probs_df.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    drawn = (values > 0) & (values >= min_prob)
    np.fill_diagonal(drawn, False)
    return np.argwhere(drawn)


def _with_string_labels(probs_df):
    """Copy of `probs_df` labeled by strings, as `_prepare_clusters_and_colors` labels its nodes."""
    return probs_df.rename(index=str, columns=str)


def _pair_totals_per_order(positions, arrows, table):
    """`table` summed over all arrow pairs, for each row of `positions`, which holds each node's
    position on the circle."""
    n = positions.shape[1]
    totals = []
    for start in range(0, len(positions), _SCORE_CHUNK_SIZE):
        chunk = positions[start:start + _SCORE_CHUNK_SIZE]
        ids = chunk[:, arrows[:, 0]] * n + chunk[:, arrows[:, 1]]
        totals.append(table[ids[:, :, None], ids[:, None, :]].sum(axis=(1, 2), dtype=np.int64) // 2)
    return np.concatenate(totals)


def count_crossings(probs_df, order, min_prob=0.0, curvature=_DEFAULT_CURVATURE):
    """Number of crossings between the arrows drawn at `min_prob` with the nodes in `order`."""
    order = [str(label) for label in order]
    n = len(order)
    if n == 0:
        return 0
    arrows = _drawn_arrows(_with_string_labels(probs_df), order, min_prob)
    crossings, _ = _arrow_pair_tables(n, curvature)
    return int(_pair_totals_per_order(np.arange(n)[None, :], arrows, crossings)[0])


def overlap_score(probs_df, order, min_prob=0.0, curvature=_DEFAULT_CURVATURE):
    """How much the arrows drawn at `min_prob` run on top of each other with the nodes in `order`:
    the share of their length that does so, added up over all pairs of arrows."""
    order = [str(label) for label in order]
    n = len(order)
    if n == 0:
        return 0.0
    arrows = _drawn_arrows(_with_string_labels(probs_df), order, min_prob)
    arrow_ids = arrows[:, 0] * n + arrows[:, 1]
    _, overlap = _arrow_pair_tables(n, curvature)
    return float(overlap[np.ix_(arrow_ids, arrow_ids)].sum() / 2)


def clutter_change_summary(prob_matrices, before, after, min_prob=0.0, curvature=_DEFAULT_CURVATURE):
    """One line comparing two node orders, added up over `prob_matrices`, for the report log."""
    crossings = [sum(count_crossings(m, o, min_prob, curvature) for m in prob_matrices) for o in (before, after)]
    overlaps = [sum(overlap_score(m, o, min_prob, curvature) for m in prob_matrices) for o in (before, after)]
    return (
        f"crossings {crossings[0]} -> {crossings[1]}, "
        f"overlap {overlaps[0]:.1f} -> {overlaps[1]:.1f} (all diagrams at the cutoff)"
    )


def _search_swaps(score, n, lookups_per_order):
    """Positions found by repeatedly swapping two nodes, starting from the given order and then
    from random ones, until the lookup budget is used. Seeded and budgeted by work rather than by
    time, so the same data gives the same order on every machine."""
    rng = np.random.default_rng(0)
    swaps = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    budget = _SEARCH_LOOKUP_BUDGET
    best = np.arange(n)
    best_score = score(best[None, :])[0]

    for restart in range(_SEARCH_RANDOM_RESTARTS + 1):
        current = best.copy() if restart == 0 else rng.permutation(n)
        current_score = score(current[None, :])[0]
        improved = True
        while improved and budget > 0:
            improved = False
            rng.shuffle(swaps)
            for batch_start in range(0, len(swaps), _SEARCH_SWAP_BATCH):
                batch = swaps[batch_start:batch_start + _SEARCH_SWAP_BATCH]
                neighbours = np.repeat(current[None, :], len(batch), axis=0)
                rows = np.arange(len(batch))
                neighbours[rows, batch[:, 0]] = current[batch[:, 1]]
                neighbours[rows, batch[:, 1]] = current[batch[:, 0]]
                neighbour_scores = score(neighbours)
                budget -= len(batch) * lookups_per_order
                if neighbour_scores.min() < current_score:
                    current = neighbours[np.argmin(neighbour_scores)]
                    current_score = neighbour_scores.min()
                    improved = True
                if budget <= 0:
                    break
        if current_score < best_score:
            best, best_score = current, current_score
        if budget <= 0:
            break
    return best


def least_cluttered_order(prob_matrices, min_prob=0.0, curvature=_DEFAULT_CURVATURE):
    """One node order with the least clutter at cutoff `min_prob` (see `_clutter_table`), added up
    over all `prob_matrices` (which share the same states), so that several diagrams can share a
    layout.

    Clutter is measured on the curved arrows as `_draw_circular_transition_diagram_on_ax` draws
    them. Orders that tie are compared on the clutter among all arrows (cutoff 0), and any
    remaining tie keeps the order closest to the first matrix's own. Up to
    `_MAX_EXHAUSTIVE_NODES` nodes this is the true minimum; above that every order would take too
    long, so the result is the best one a swap search finds, which is never worse than the
    starting order but may not be the very best.
    """
    prob_matrices = [_with_string_labels(m) for m in prob_matrices]
    labels = list(prob_matrices[0].index)
    n = len(labels)
    if n < 3:
        return labels

    table = _clutter_table(n, curvature)
    shown_arrows = [_drawn_arrows(m, labels, min_prob) for m in prob_matrices]
    all_arrows = [_drawn_arrows(m, labels, 0.0) for m in prob_matrices]
    tiebreak_scale = int(table.max()) * sum(len(a) for a in all_arrows) ** 2 + 1

    # Incoming and outgoing often draw the same arrows, and at cutoff 0 so do both levels of the
    # score, so identical arrow sets are only counted once and weighted instead.
    weighted_arrows = {}
    for arrows, weight in [(a, tiebreak_scale) for a in shown_arrows] + [(a, 1) for a in all_arrows]:
        key = arrows.tobytes()
        weighted_arrows[key] = (arrows, weighted_arrows.get(key, (arrows, 0))[1] + weight)

    def score(positions):
        return sum(
            _pair_totals_per_order(positions, arrows, table) * weight
            for arrows, weight in weighted_arrows.values()
        )

    if n <= _MAX_EXHAUSTIVE_NODES:
        positions = np.array([(0, *rest) for rest in permutations(range(1, n))])
        best = positions[np.argmin(score(positions))]
    else:
        lookups_per_order = sum(len(arrows) ** 2 for arrows, _ in weighted_arrows.values())
        best = _search_swaps(score, n, max(lookups_per_order, 1))

    return [labels[i] for i in np.argsort(best)]


def _draw_self_loop(ax, pos, *, color, node_radius, linewidth, alpha, mutation_scale, span_deg=70.0, loop_curvature=1.6):
    """Draw a small self-transition loop bulging outward from a node.

    Nodes sit on a circle centered at the diagram origin, so the outward direction at `pos` is
    just `pos` normalized. Two points are picked on the node's own boundary, straddling that
    outward direction by `span_deg`. `arc3`'s control point is `posA`'s midpoint offset by
    `rad * (posB - posA)` rotated 90 degrees - empirically, going from the "before" point (at
    `phi - half`) to the "after" point (at `phi + half`) with positive `rad` bulges the arc
    further outward; the reverse order bulges it back into the diagram interior where it would
    cross other edges.
    """
    x, y = pos
    phi = np.arctan2(y, x)
    half = np.deg2rad(span_deg) / 2.0

    start = (x + node_radius * np.cos(phi - half), y + node_radius * np.sin(phi - half))
    end = (x + node_radius * np.cos(phi + half), y + node_radius * np.sin(phi + half))

    loop = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={loop_curvature}",
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=0, shrinkB=0,
        zorder=2,
    )
    ax.add_patch(loop)


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
    node_radius=_NODE_RADIUS,
    curvature=_DEFAULT_CURVATURE,
    label_fontsize=11,
    label_style="on_node",
    source_cluster=None,
    target_cluster=None,
    scale_mode="relative",
):
    """Draw a circular inter-cluster transition diagram on `ax`.

    One node per `probs_df` row/column, evenly spaced counter-clockwise from the bottom. One curved
    directed arrow per off-diagonal (from, to) pair with probability >= `min_prob_to_draw`,
    colored by the source cluster. Diagonal (self-transition) entries are drawn too, as a small
    loop beside the node rather than an arc between two nodes - so a matrix with a zeroed
    diagonal (e.g. the renormalized no-self matrix) simply shows no loops, while a self-inclusive
    matrix shows each cluster's self-persistence alongside its switches.

    Visual weight (linewidth/alpha) comes from probability normalized to [0, 1] against the
    strongest edge and raised to `emphasis_gamma` (>1 fades weak transitions, <1 strengthens them)
    before mapping onto the linewidth/alpha ranges. `min_prob_to_draw` only removes the edges below
    it - not just faintly, gone - and does not affect the weight of the others, so the same edge
    looks identical at any cutoff. Self-loops share this same pool, so a cluster that mostly stays
    put shows a bold loop while its minor switches stay faint.

    `label_style="on_node"` (default) draws each cluster's full label inside its dot. With
    `label_style="legend"`, dots are labeled with their 1-based index instead (short,
    never cramped) - the caller is expected to draw the index-to-label legend itself (this
    function only has an `ax`, not the figure a legend would need to sit outside of); see
    `plot_circular_transition_diagram`.

    `source_cluster`/`target_cluster`, when given, restrict the drawn edges to that one source
    or destination respectively (every node still renders, for context); a cluster's own
    self-loop is drawn whenever it passes both filters, i.e. whenever that cluster is the
    selected source and/or destination. With `scale_mode="relative"` (default), the strongest edge
    is looked for among those filters' edges only, which scales the visual weight to that
    cluster's own transitions rather than the whole matrix's - the strongest edge into/out of this
    cluster reads as the boldest arc, not the strongest edge overall. See
    `plot_circular_transition_diagram_grid`, which draws one such panel per cluster on a single
    page.

    `scale_mode="absolute"` instead maps probability directly onto visual weight on a fixed
    0-1 scale (a 0.5 edge always reads as ~half of `max_linewidth`/`max_alpha`, regardless of
    what else is drawn), rather than stretching the strongest edge to fill the full range. This
    matters most for a diagram that overlays every cluster's edges together:
    with `"relative"` scaling there, one cluster's dominant edge (often a large self-transition)
    becomes the reference for 100%, which visually flattens every other cluster's edges even
    when, on their own terms, they aren't faint at all.
    """
    clusters = list(probs_df.index)
    positions = _circular_node_positions(clusters)

    edges = []
    for src in clusters:
        if source_cluster is not None and src != source_cluster:
            continue
        for dst in clusters:
            if target_cluster is not None and dst != target_cluster:
                continue
            p = float(probs_df.loc[src, dst])
            if np.isfinite(p) and p > 0:
                edges.append((src, dst, p))

    if edges:
        # Taken before the cutoff is applied, so the cutoff can't change any edge's weight.
        full_scale = 1.0 if scale_mode == "absolute" else max(p for _, _, p in edges)

        def visual_weight(p):
            return min(p / full_scale, 1.0) ** emphasis_gamma

        # Weakest first, so strong arcs always render on top of faint ones, never the reverse.
        drawn_edges = [edge for edge in edges if edge[2] >= min_prob_to_draw]
        for src, dst, p in sorted(drawn_edges, key=lambda e: e[2]):
            t = visual_weight(p)
            linewidth = min_linewidth + t * (max_linewidth - min_linewidth)
            alpha = min_alpha + t * (max_alpha - min_alpha)
            color = colors[src]
            mutation_scale = 10 + 8 * t

            if src == dst:
                _draw_self_loop(
                    ax, positions[src], color=color, node_radius=node_radius,
                    linewidth=linewidth, alpha=alpha, mutation_scale=mutation_scale,
                )
                continue

            start, end = _arrow_endpoints(positions[src], positions[dst], node_radius)

            arrow = FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
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
    scale_mode="relative",
):
    """Standalone figure: circular inter-cluster transition diagram.

    See `_draw_circular_transition_diagram_on_ax` for the drawing/visual-weight semantics,
    including what `scale_mode` controls. `probs_df` is typically a no-self
    transition-probability matrix (self-transitions zeroed and rows renormalized over the
    remaining off-diagonal entries).

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
        label_style=label_style, scale_mode=scale_mode,
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


def plot_circular_transition_diagram_pair(
    probs_df,
    *,
    state_colors=None,
    state_order=None,
    title=None,
    min_prob_to_draw=0.03,
    emphasis_gamma=2.0,
    curvature=0.28,
    label_style="on_node",
    panel_figsize=(7, 7),
    scale_mode="relative",
):
    """Two-panel figure: the full/unfiltered diagram (cutoff = 0) beside the same diagram at
    `min_prob_to_draw`, so the effect of the cutoff is visible by direct comparison rather than
    having to page back and forth between two single-panel figures.

    Both panels share one `colors`/`state_order` resolution and one legend (when
    `label_style="legend"`) - see `plot_circular_transition_diagram` for what `scale_mode` and
    the other drawing parameters control; they're applied identically to both panels here.
    """
    clusters, probs_df, colors = _prepare_clusters_and_colors(probs_df, state_colors, state_order)

    fig, axes = plt.subplots(1, 2, figsize=(panel_figsize[0] * 2, panel_figsize[1]))
    _draw_circular_transition_diagram_on_ax(
        axes[0], probs_df, colors=colors,
        min_prob_to_draw=0.0, emphasis_gamma=emphasis_gamma, curvature=curvature,
        label_style=label_style, scale_mode=scale_mode,
    )
    axes[0].set_title("All transitions (cutoff = 0)", fontsize=11, pad=10)

    _draw_circular_transition_diagram_on_ax(
        axes[1], probs_df, colors=colors,
        min_prob_to_draw=min_prob_to_draw, emphasis_gamma=emphasis_gamma, curvature=curvature,
        label_style=label_style, scale_mode=scale_mode,
    )
    axes[1].set_title(f"Cutoff = {min_prob_to_draw:g}", fontsize=11, pad=10)

    if title:
        fig.suptitle(str(title), fontsize=12, fontweight="bold")
    if label_style == "legend":
        fig.legend(
            handles=_legend_handles(clusters, colors), loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
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
    direction="outgoing",
):
    """Grid figure: one panel per cluster, each showing only that cluster's outgoing (or
    incoming) transitions (see `source_cluster`/`target_cluster` on
    `_draw_circular_transition_diagram_on_ax`) plus that cluster's own self-loop, if any.

    Laying every cluster's own view side by side - rather than the single overlaid
    `plot_circular_transition_diagram`, or one page per cluster - keeps the population-wide
    circular diagram readable while still letting each cluster's own transition pattern be
    inspected at a glance, all on one page. Each panel's visual weight (arrow thickness/alpha)
    is scaled to that cluster's own transitions, not the global maximum - a cluster that mostly
    stays put still shows its strongest available switch (or its self-loop) as a bold arc within
    its own panel.

    `direction="outgoing"` (default) restricts each panel to that cluster's own outgoing
    transitions (`source_cluster=cluster`); `direction="incoming"` restricts it to transitions
    into that cluster instead (`target_cluster=cluster`). Either way the cluster's own self-loop
    is included, since it passes both filters.

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
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")

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
            ax, probs_df, colors=colors,
            source_cluster=cluster if direction == "outgoing" else None,
            target_cluster=cluster if direction == "incoming" else None,
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

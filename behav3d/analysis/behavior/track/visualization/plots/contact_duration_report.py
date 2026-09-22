"""Standalone contact-duration comparison.

Pulled out of the bundled ``save_track_contact_group_analysis`` report (see ``reports.py``) so it
can be run on its own: for each track, how long it stayed in contact with each touched class of a
chosen target cell type (e.g. a T-cell/organoid track's contacts broken down by which macrophage
morphology class — round / elongated / plastic — it touched), compared:

- pairwise between every two classes, and
- each class vs. every other class pooled together ("rest"),

using either an unpaired Welch's t-test (the default, matching the rest of the app) or a paired
t-test where the pairs are formed by averaging within a user-chosen column (typically
``sample_name``) before pairing — mirroring a manual R workflow of
``group_by(sample_name, class) |> summarise(mean(...)) |> pivot_wider() |> t.test(paired=TRUE)``.
"""
import textwrap
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from behav3d.analysis.behavior.utils import _mixed_label_sort_key
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import (
    hash_stable_label_color_map,
    welch_ttest_stars,
    SIGNIFICANCE_LEGEND_TEXT,
    compute_class_by_stack_proportions,
    draw_stacked_proportion_barv,
    legend_layout,
    _chunk_list,
    _make_group_label,
)
from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    merge_track_contact_features_into_obs,
    compute_track_contact_target_class_features,
    _contact_group_col_name,
    _contact_class_mean_col_name,
    _contact_class_max_bout_col_name,
)

_MIN_GROUP_N_FOR_TEST = 2

# Cycled (not hashed, so neighboring samples stay visually distinct even when there are only a
# handful) per sorted sample name to give each one a stable marker shape -- used by the
# "connected by sample" boxplot panels so a sample's two dots stay identifiable without a line
# joining them (the line moved to a separate companion panel; see ``_draw_pair_connected_lines``/
# ``_draw_long_contact_connected_lines``).
_SAMPLE_MARKER_SHAPES = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def _sample_marker_map(sample_names):
    ordered = sorted({str(s) for s in sample_names}, key=_mixed_label_sort_key)
    return {name: _SAMPLE_MARKER_SHAPES[i % len(_SAMPLE_MARKER_SHAPES)] for i, name in enumerate(ordered)}


def _rest_label(other_classes):
    if len(other_classes) <= 3:
        return f"rest ({'+'.join(other_classes)})"
    return f"rest ({len(other_classes)} classes)"


def _build_comparisons(class_order):
    """``[(label_a, classes_a, label_b, classes_b), ...]`` — every pairwise combo between
    individual classes, plus (only once there are >= 3 classes, since with 2 it would just
    duplicate the pairwise entry) one "class vs. every other class pooled" entry per class."""
    comparisons = [(a, [a], b, [b]) for a, b in combinations(class_order, 2)]
    if len(class_order) >= 3:
        for cls in class_order:
            others = [c for c in class_order if c != cls]
            comparisons.append((cls, [cls], _rest_label(others), others))
    return comparisons


def _welch_stats(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n_a, n_b = int(len(values_a)), int(len(values_b))
    mean_a = float(np.mean(values_a)) if n_a else float("nan")
    mean_b = float(np.mean(values_b)) if n_b else float("nan")
    if n_a >= _MIN_GROUP_N_FOR_TEST and n_b >= _MIN_GROUP_N_FOR_TEST:
        t_stat, p_value = stats.ttest_ind(values_b, values_a, equal_var=False, nan_policy="omit")
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    return dict(n_a=n_a, n_b=n_b, mean_a=mean_a, mean_b=mean_b, t_stat=t_stat, p_value=p_value)


def _paired_stats(values_a, values_b):
    """``values_a``/``values_b`` are already the aligned per-pairing-unit mean durations (equal
    length, same pairing-unit order) — the caller does the inner join."""
    n = int(len(values_a))
    mean_a = float(np.mean(values_a)) if n else float("nan")
    mean_b = float(np.mean(values_b)) if n else float("nan")
    if n >= _MIN_GROUP_N_FOR_TEST:
        t_stat, p_value = stats.ttest_rel(values_b, values_a, nan_policy="omit")
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    return dict(n_a=n, n_b=n, mean_a=mean_a, mean_b=mean_b, t_stat=t_stat, p_value=p_value)


def _paired_sample_frame(duration_df, *, classes_a, classes_b, sample_col):
    """Per ``sample_col`` unit, the inner-joined mean ``duration_timepoints``/``duration_fraction``
    for ``classes_a`` vs. ``classes_b`` — the data behind the "connected by sample" plot (and its
    own paired significance test), always paired by ``sample_col`` regardless of whatever
    ``pairing_col``/``test_mode`` the main comparison uses. Joining both metrics in one frame
    (rather than two separate joins) guarantees the same sample set backs both panels.
    """
    per_unit = duration_df.groupby([sample_col, "target_class"])[["duration_timepoints", "duration_fraction"]].mean()
    target_level = per_unit.index.get_level_values("target_class")
    side_a = per_unit[target_level.isin(classes_a)].groupby(level=sample_col).mean()
    side_b = per_unit[target_level.isin(classes_b)].groupby(level=sample_col).mean()
    joined = side_a.join(side_b, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    joined.index = joined.index.astype(str)
    return joined


def _values_and_stats_for_column(duration_df, value_col, *, classes_a, classes_b, test_mode, pairing_col):
    """Shared welch/paired value-selection + stats logic for one metric column (``duration_timepoints``
    or ``duration_fraction``) — factored out of ``_comparison_row`` so both metrics reuse it."""
    if test_mode == "paired":
        per_unit = duration_df.groupby([pairing_col, "target_class"])[value_col].mean()
        target_level = per_unit.index.get_level_values("target_class")
        side_a = per_unit[target_level.isin(classes_a)].groupby(level=pairing_col).mean()
        side_b = per_unit[target_level.isin(classes_b)].groupby(level=pairing_col).mean()
        joined = pd.concat({"a": side_a, "b": side_b}, axis=1).dropna()
        values_a = joined["a"].to_numpy()
        values_b = joined["b"].to_numpy()
        stats_row = _paired_stats(values_a, values_b)
    else:
        values_a = duration_df.loc[duration_df["target_class"].isin(classes_a), value_col].to_numpy()
        values_b = duration_df.loc[duration_df["target_class"].isin(classes_b), value_col].to_numpy()
        stats_row = _welch_stats(values_a, values_b)
    return values_a, values_b, stats_row


def _comparison_row(
    duration_df, *, label_a, classes_a, label_b, classes_b, test_mode, pairing_col,
    sample_col, pairing_col_label=None,
):
    """Compute stats plus the actual values tested, for one comparison — both on bout-length
    (``duration_timepoints``) and on time-in-contact fraction (``duration_fraction``), as two
    independent statistical tests (fraction is not a unit conversion of bout-length, so it gets
    its own t-stat/p-value/stars).

    ``pairing_col`` is the actual dataframe column grouped on (may be an internal composite
    column when multiple pairing columns were combined); ``pairing_col_label`` is the
    human-readable name recorded in the output (defaults to ``pairing_col`` itself).

    Also always computes the per-``sample_col`` paired view (see ``_paired_sample_frame``) used by
    the "connected by sample" companion plot/CSV, independent of ``test_mode``/``pairing_col``.
    """
    values_a, values_b, stats_row = _values_and_stats_for_column(
        duration_df, "duration_timepoints", classes_a=classes_a, classes_b=classes_b,
        test_mode=test_mode, pairing_col=pairing_col,
    )
    diff = (
        stats_row["mean_b"] - stats_row["mean_a"]
        if np.isfinite(stats_row["mean_a"]) and np.isfinite(stats_row["mean_b"])
        else float("nan")
    )
    stats_row.update(
        group_a=label_a,
        group_b=label_b,
        diff=diff,
        stars=welch_ttest_stars(stats_row["p_value"]),
        test_mode=test_mode,
        pairing_col=(pairing_col_label or pairing_col) if test_mode == "paired" else None,
        values_a=values_a,
        values_b=values_b,
    )

    frac_values_a, frac_values_b, frac_stats = _values_and_stats_for_column(
        duration_df, "duration_fraction", classes_a=classes_a, classes_b=classes_b,
        test_mode=test_mode, pairing_col=pairing_col,
    )
    fraction_diff = (
        frac_stats["mean_b"] - frac_stats["mean_a"]
        if np.isfinite(frac_stats["mean_a"]) and np.isfinite(frac_stats["mean_b"])
        else float("nan")
    )
    stats_row.update(
        fraction_n_a=frac_stats["n_a"],
        fraction_n_b=frac_stats["n_b"],
        fraction_mean_a=frac_stats["mean_a"],
        fraction_mean_b=frac_stats["mean_b"],
        fraction_diff=fraction_diff,
        fraction_t_stat=frac_stats["t_stat"],
        fraction_p_value=frac_stats["p_value"],
        fraction_stars=welch_ttest_stars(frac_stats["p_value"]),
        fraction_values_a=frac_values_a,
        fraction_values_b=frac_values_b,
    )

    sample_frame = _paired_sample_frame(duration_df, classes_a=classes_a, classes_b=classes_b, sample_col=sample_col)
    sample_tp_a = sample_frame["duration_timepoints_a"].to_numpy()
    sample_tp_b = sample_frame["duration_timepoints_b"].to_numpy()
    sample_frac_a = sample_frame["duration_fraction_a"].to_numpy()
    sample_frac_b = sample_frame["duration_fraction_b"].to_numpy()
    sample_stats_tp = _paired_stats(sample_tp_a, sample_tp_b)
    sample_stats_frac = _paired_stats(sample_frac_a, sample_frac_b)
    stats_row.update(
        sample_names=sample_frame.index.tolist(),
        sample_values_a=sample_tp_a,
        sample_values_b=sample_tp_b,
        sample_n=int(len(sample_frame)),
        sample_p_value=sample_stats_tp["p_value"],
        sample_stars=welch_ttest_stars(sample_stats_tp["p_value"]),
        sample_fraction_values_a=sample_frac_a,
        sample_fraction_values_b=sample_frac_b,
        sample_fraction_p_value=sample_stats_frac["p_value"],
        sample_fraction_stars=welch_ttest_stars(sample_stats_frac["p_value"]),
    )
    return stats_row


def _add_significance_annotation(ax, finite_vals, stars):
    """Shared bracket/"n.s." annotation for a 2-box comparison panel at x-positions 1, 2 — factored
    out of ``_draw_pair_box`` so ``_draw_pair_connected_box`` draws the identical annotation."""
    if not finite_vals:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        return
    y_max, y_min = max(finite_vals), min(finite_vals)
    span = max(y_max - y_min, 1e-9)
    if stars and stars != "n.s.":
        y = y_max + span * 0.08
        tick = span * 0.03
        ax.plot([1, 1, 2, 2], [y, y + tick, y + tick, y], lw=0.8, color="black")
        ax.text(1.5, y + tick, stars, ha="center", va="bottom", fontsize=8)
        ax.set_ylim(top=y + tick + span * 0.15)
    elif stars == "n.s.":
        ax.text(0.5, 1.02, "n.s.", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color="#666")


def _draw_box_shell(ax, data, *, label_a, label_b, colors, widths, alpha):
    """The boxplot itself (positions 1, 2), shared by the jittered-dot and connected-dot panels."""
    wrapped_labels = [
        textwrap.fill(str(label_a), width=12, break_long_words=False),
        textwrap.fill(str(label_b), width=12, break_long_words=False),
    ]
    bp = ax.boxplot(data, tick_labels=wrapped_labels, patch_artist=True, widths=widths, showfliers=False, zorder=1)
    for patch, label in zip(bp["boxes"], (label_a, label_b)):
        patch.set_facecolor(colors.get(label, "#808080"))
        patch.set_alpha(alpha)
    return bp


def _draw_pair_box(ax, values_a, values_b, *, label_a, label_b, ylabel, stars, colors):
    data = [np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)]
    _draw_box_shell(ax, data, label_a=label_a, label_b=label_b, colors=colors, widths=0.8, alpha=0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color="black", alpha=0.4, zorder=3)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)

    finite_vals = [v for arr in data if len(arr) for v in arr[np.isfinite(arr)]]
    _add_significance_annotation(ax, finite_vals, stars)


def _draw_pair_connected_box(
    ax, sample_names, values_a, values_b, *, label_a, label_b, ylabel, stars, colors,
    sample_colors, sample_markers,
):
    """Same box as ``_draw_pair_box``, but each dot pair is one ``sample_col`` unit, drawn with a
    per-sample marker shape (``_sample_marker_map``) instead of a connecting line, so a sample's
    two dots stay identifiable without cluttering the box with lines — the actual line lives in
    the companion ``_draw_pair_connected_lines`` panel. Markers are solid black as long as
    ``sample_markers`` has a unique shape per sample; once there are more samples than shapes
    (``_SAMPLE_MARKER_SHAPES`` wraps around, so shape alone can no longer disambiguate), markers
    fall back to per-sample color instead.
    ``values_a``/``values_b`` are already the per-sample means (see ``_paired_sample_frame``), so
    the box reflects the same per-sample distribution the dots show (not the finer-grained
    per-track spread ``_draw_pair_box`` shows).
    """
    data = [np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)]
    _draw_box_shell(ax, data, label_a=label_a, label_b=label_b, colors=colors, widths=0.6, alpha=0.4)
    shapes_unique = len(sample_markers) <= len(_SAMPLE_MARKER_SHAPES)
    n = len(sample_names)
    rng = np.random.default_rng(0)
    jitter = (rng.random(n) - 0.5) * 0.25 if n else np.array([])
    for i, name in enumerate(sample_names):
        c = "black" if shapes_unique else sample_colors.get(str(name), "#808080")
        m = sample_markers.get(str(name), "o")
        xa, xb = 1 + jitter[i], 2 + jitter[i]
        ax.scatter(
            [xa, xb], [data[0][i], data[1][i]], color=c, marker=m, s=18, zorder=3,
            edgecolor="white", linewidth=0.3,
        )
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)

    finite_vals = [v for arr in data if len(arr) for v in arr[np.isfinite(arr)]]
    _add_significance_annotation(ax, finite_vals, stars)


def _draw_pair_connected_lines(ax, sample_names, values_a, values_b, *, label_a, label_b, colors):
    """Companion to ``_draw_pair_connected_box`` — no box, just each sample's line from
    ``label_a`` to ``label_b`` (the connecting lines the box panel deliberately omits), all
    starting/ending at the same x (no jitter — this panel has nothing else at that x to collide
    with). The line itself is a neutral grey (a sample's identity isn't the point here); what's
    colored is each *side* — every ``label_a`` dot in ``colors[label_a]``, every ``label_b`` dot in
    ``colors[label_b]`` (the same per-class colors the box panel's fill uses), so the panel reads
    as "this class vs. that class" rather than "these samples". A bold black line on top traces
    the mean from ``label_a`` to ``label_b``. Drawn with normal axes/ticks (not sharing the box
    panel's hidden-spine style) so it stands on its own.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    color_a = colors.get(label_a, "#4C72B0")
    color_b = colors.get(label_b, "#55A868")
    for va, vb in zip(values_a, values_b):
        ax.plot([1, 2], [va, vb], color="#999999", lw=1.0, alpha=0.7, zorder=2)
    if len(values_a):
        ax.scatter(np.full(len(values_a), 1), values_a, color=color_a, s=45, zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter(np.full(len(values_b), 2), values_b, color=color_b, s=45, zorder=3, edgecolor="white", linewidth=0.5)
    if len(values_a) and len(values_b):
        mean_a, mean_b = np.nanmean(values_a), np.nanmean(values_b)
        if np.isfinite(mean_a) and np.isfinite(mean_b):
            ax.plot([1, 2], [mean_a, mean_b], color="black", lw=2.5, alpha=0.9, zorder=5)
            ax.text(2.05, mean_b, "mean", fontsize=6, color="black", va="center", fontweight="bold")
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    # This panel is narrower than the box panel, so long class labels get rotated (rather than
    # wrapped, as the box panel's tick_labels are) to avoid the two colliding.
    ax.set_xticklabels([str(label_a), str(label_b)], fontsize=6.5, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=7)


def _plot_duration_comparison_page(
    page_rows, *, contact_col, target_cell_type_label, test_mode, pairing_col, minutes_per_frame,
    ncols, label_colors, section_label=None, pairing_col_label=None,
):
    n = len(page_rows)
    show_minutes = minutes_per_frame is not None
    n_panels = 3 if show_minutes else 2

    ncols_page = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols_page))
    panel_w, panel_h = 2.3, 3.0
    header_h, footer_h = 1.2, 0.4
    fig_w = max(6.0, panel_w * n_panels * ncols_page)
    fig_h = header_h + footer_h + panel_h * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        nrows=nrows, ncols=ncols_page, hspace=0.9, wspace=0.5,
        top=1 - header_h / fig_h, bottom=footer_h / fig_h,
    )

    for i, row in enumerate(page_rows):
        r, c = divmod(i, ncols_page)
        inner = outer[r, c].subgridspec(1, n_panels, wspace=0.5)
        ax_tp = fig.add_subplot(inner[0, 0])
        _draw_pair_box(
            ax_tp, row["values_a"], row["values_b"],
            label_a=row["group_a"], label_b=row["group_b"],
            ylabel="Duration (timepoints)", stars=row["stars"], colors=label_colors,
        )
        ax_tp.set_title(
            textwrap.fill(f"{row['group_a']} vs {row['group_b']}", width=20), fontsize=7,
        )
        next_panel = 1
        if show_minutes:
            ax_min = fig.add_subplot(inner[0, next_panel])
            _draw_pair_box(
                ax_min,
                np.asarray(row["values_a"], dtype=float) * minutes_per_frame,
                np.asarray(row["values_b"], dtype=float) * minutes_per_frame,
                label_a=row["group_a"], label_b=row["group_b"],
                ylabel="Duration (minutes)", stars=row["stars"], colors=label_colors,
            )
            next_panel += 1
        ax_frac = fig.add_subplot(inner[0, next_panel])
        _draw_pair_box(
            ax_frac, row["fraction_values_a"], row["fraction_values_b"],
            label_a=row["group_a"], label_b=row["group_b"],
            ylabel="Fraction of window in contact", stars=row["fraction_stars"], colors=label_colors,
        )

    subtitle = f"contact_col={contact_col}  target={target_cell_type_label}  test={test_mode}"
    if test_mode == "paired":
        subtitle += f"  pairing_col={pairing_col_label or pairing_col}"
    if not show_minutes:
        subtitle += "  (minutes unavailable — no time metadata)"
    title = f"Contact duration comparison — bout length & contact-time fraction by {target_cell_type_label} class"
    if section_label is not None:
        title += f"\ngroup: {section_label}"
    else:
        title += "\n(all data pooled)"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.02, SIGNIFICANCE_LEGEND_TEXT, ha="center", va="bottom", fontsize=7)
    return fig


def _plot_duration_comparison_connected_page(
    page_rows, *, contact_col, target_cell_type_label, sample_col, minutes_per_frame,
    ncols, label_colors, sample_colors, sample_markers, section_label=None,
):
    """Companion to ``_plot_duration_comparison_page`` — same panel layout, but each comparison
    gets a box+lines pair: a box built from per-``sample_col`` means, with dots colored by sample
    and shaped by ``sample_markers`` (no connecting line — see ``_draw_pair_connected_box``), and
    a narrower companion panel to its right showing just each sample's connecting line (see
    ``_draw_pair_connected_lines``), sharing the box's y-axis. Rendered as a separate page right
    after the original (unlinked jittered-dot) page, which is left unchanged.
    """
    n = len(page_rows)
    show_minutes = minutes_per_frame is not None
    n_panels = 3 if show_minutes else 2

    ncols_page = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols_page))
    box_w, lines_w, panel_h = 2.3, 1.3, 3.0
    panel_w = box_w + lines_w
    all_samples = sorted(
        {name for row in page_rows for name in row["sample_names"]}, key=_mixed_label_sort_key,
    )
    ncol, n_legend_rows, legend_margin_in = legend_layout(
        len(all_samples), max_ncol=8, row_height_in=0.2, base_margin_in=0.55,
    )
    header_h = 1.2
    footer_h = legend_margin_in + 0.3
    fig_w = max(6.0, panel_w * n_panels * ncols_page)
    fig_h = header_h + footer_h + panel_h * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    # A fixed inch margin rather than matplotlib's default *fraction*-of-figure-width margin —
    # on these wide (many-comparisons-per-row) figures a fractional margin wastes an increasing
    # amount of blank space as fig_w grows, pushing every panel toward the right.
    side_margin_in = 0.3
    outer = fig.add_gridspec(
        nrows=nrows, ncols=ncols_page, hspace=0.9, wspace=0.5,
        top=1 - header_h / fig_h, bottom=footer_h / fig_h,
        left=side_margin_in / fig_w, right=1 - side_margin_in / fig_w,
    )

    def _draw_metric_group(cell, values_a, values_b, ylabel, stars, row):
        pair = cell.subgridspec(1, 2, width_ratios=[box_w, lines_w], wspace=0.15)
        ax_box = fig.add_subplot(pair[0, 0])
        ax_lines = fig.add_subplot(pair[0, 1], sharey=ax_box)
        _draw_pair_connected_box(
            ax_box, row["sample_names"], values_a, values_b,
            label_a=row["group_a"], label_b=row["group_b"],
            ylabel=ylabel, stars=stars, colors=label_colors,
            sample_colors=sample_colors, sample_markers=sample_markers,
        )
        _draw_pair_connected_lines(
            ax_lines, row["sample_names"], values_a, values_b,
            label_a=row["group_a"], label_b=row["group_b"], colors=label_colors,
        )
        return ax_box

    for i, row in enumerate(page_rows):
        r, c = divmod(i, ncols_page)
        inner = outer[r, c].subgridspec(1, n_panels, wspace=0.6)
        ax_tp = _draw_metric_group(
            inner[0, 0], row["sample_values_a"], row["sample_values_b"],
            "Duration (timepoints)", row["sample_stars"], row,
        )
        ax_tp.set_title(
            textwrap.fill(f"{row['group_a']} vs {row['group_b']}", width=20), fontsize=7,
        )
        next_panel = 1
        if show_minutes:
            _draw_metric_group(
                inner[0, next_panel],
                np.asarray(row["sample_values_a"], dtype=float) * minutes_per_frame,
                np.asarray(row["sample_values_b"], dtype=float) * minutes_per_frame,
                "Duration (minutes)", row["sample_stars"], row,
            )
            next_panel += 1
        _draw_metric_group(
            inner[0, next_panel], row["sample_fraction_values_a"], row["sample_fraction_values_b"],
            "Fraction of window in contact", row["sample_fraction_stars"], row,
        )

    subtitle = f"contact_col={contact_col}  target={target_cell_type_label}  paired by {sample_col}"
    if not show_minutes:
        subtitle += "  (minutes unavailable — no time metadata)"
    title = f"Contact duration comparison — connected by {sample_col} (box = per-sample means)"
    if section_label is not None:
        title += f"\ngroup: {section_label}"
    else:
        title += "\n(all data pooled)"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")

    # Legend marker color mirrors the box panel's own rule (_draw_pair_connected_box): solid
    # black while every sample still has its own unique shape, per-sample color once shapes wrap
    # around. No line stroke -- the panel's connecting lines are class-colored/grey now, not
    # per-sample, so a line swatch here would no longer mean anything.
    shapes_unique = len(sample_markers) <= len(_SAMPLE_MARKER_SHAPES)
    handles = [
        Line2D(
            [0], [0], marker=sample_markers.get(s, "o"),
            color="black" if shapes_unique else sample_colors.get(s, "#808080"),
            label=s, linestyle="None", markersize=5,
        )
        for s in all_samples
    ]
    if handles:
        # 0.3in above the figure bottom, expressed as a *whole-figure* fraction (fig.legend's
        # bbox_transform is fig.transFigure, not the footer's own sub-rectangle) -- leaves the
        # significance-legend text (at fig-fraction 0.02) room below it.
        legend_y = 0.3 / fig_h if fig_h > 0 else 0.0
        fig.legend(
            handles=handles, title=f"sample ({sample_col})", loc="lower center",
            bbox_to_anchor=(0.5, legend_y), ncol=ncol, frameon=False, fontsize=6, title_fontsize=6.5,
            handlelength=1.3, labelspacing=0.3, columnspacing=1.0,
        )
    fig.text(0.5, 0.02, SIGNIFICANCE_LEGEND_TEXT, ha="center", va="bottom", fontsize=7)
    return fig


def _render_duration_comparison_section(
    pdf, sub_duration_df, *, class_order, contact_col, target_cell_type_label, test_mode,
    pairing_col, sample_col, sample_colors, sample_markers, minutes_per_frame, ncols, class_colors,
    comparisons_per_page, section_label=None, pairing_col_label=None,
):
    """Build + render the pairwise/one-vs-rest comparison page(s) for one data subset — either
    the pooled "general" set (``section_label=None``) or one ``group_cols`` page-split's slice.

    Classes are restricted to ``class_order`` (the globally touched/resolved class list) filtered
    down to whichever of those are actually present in ``sub_duration_df``, so a group missing one
    class simply skips comparisons involving it rather than erroring.

    Also renders one "connected by sample" companion page right after each original page (see
    ``_plot_duration_comparison_connected_page``) — same comparisons, but paired by ``sample_col``,
    with box dots colored/shaped by sample and the connecting lines drawn in their own panel next
    to each box instead of unlinked jittered dots.

    Returns ``(csv_rows, paired_csv_rows, n_pages)`` — both carry a ``"page_group"`` key
    (``"(all)"`` when ``section_label`` is None). ``paired_csv_rows`` has one row per
    (comparison, sample) — the per-sample values behind the connected page's dots/lines.
    """
    page_group_value = "(all)" if section_label is None else str(section_label)
    present_classes = set(sub_duration_df["target_class"].dropna().unique().tolist())
    local_class_order = [c for c in class_order if c in present_classes]

    if len(local_class_order) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        where = f" for group '{section_label}'" if section_label is not None else ""
        ax.text(
            0.5, 0.5,
            f"Fewer than 2 touched '{target_cell_type_label}' classes found{where} — nothing to compare.",
            ha="center", va="center", wrap=True,
        )
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        return [], [], 1

    comparisons = _build_comparisons(local_class_order)
    rows = [
        _comparison_row(
            sub_duration_df, label_a=label_a, classes_a=classes_a, label_b=label_b, classes_b=classes_b,
            test_mode=test_mode, pairing_col=pairing_col, sample_col=sample_col,
            pairing_col_label=pairing_col_label,
        )
        for label_a, classes_a, label_b, classes_b in comparisons
    ]

    all_labels = sorted({row["group_a"] for row in rows} | {row["group_b"] for row in rows})
    label_colors = hash_stable_label_color_map(all_labels, colors=class_colors)

    pages = _chunk_list(rows, comparisons_per_page)
    for page_rows in pages:
        fig = _plot_duration_comparison_page(
            page_rows, contact_col=contact_col, target_cell_type_label=target_cell_type_label,
            test_mode=test_mode, pairing_col=pairing_col, minutes_per_frame=minutes_per_frame,
            ncols=ncols, label_colors=label_colors, section_label=section_label,
            pairing_col_label=pairing_col_label,
        )
        pdf.savefig(fig)
        plt.close(fig)

        connected_fig = _plot_duration_comparison_connected_page(
            page_rows, contact_col=contact_col, target_cell_type_label=target_cell_type_label,
            sample_col=sample_col, minutes_per_frame=minutes_per_frame, ncols=ncols,
            label_colors=label_colors, sample_colors=sample_colors, sample_markers=sample_markers,
            section_label=section_label,
        )
        pdf.savefig(connected_fig)
        plt.close(connected_fig)

    csv_rows = []
    paired_csv_rows = []
    for row in rows:
        m_a = row["mean_a"] * minutes_per_frame if minutes_per_frame is not None and np.isfinite(row["mean_a"]) else float("nan")
        m_b = row["mean_b"] * minutes_per_frame if minutes_per_frame is not None and np.isfinite(row["mean_b"]) else float("nan")
        csv_rows.append({
            "page_group": page_group_value,
            "group_a": row["group_a"], "group_b": row["group_b"],
            "n_a": row["n_a"], "n_b": row["n_b"],
            "mean_a_timepoints": row["mean_a"], "mean_b_timepoints": row["mean_b"],
            "diff_timepoints": row["diff"], "mean_a_minutes": m_a, "mean_b_minutes": m_b,
            "mean_a_fraction": row["fraction_mean_a"], "mean_b_fraction": row["fraction_mean_b"],
            "diff_fraction": row["fraction_diff"],
            "t_stat": row["t_stat"], "p_value": row["p_value"], "stars": row["stars"],
            "t_stat_fraction": row["fraction_t_stat"], "p_value_fraction": row["fraction_p_value"],
            "stars_fraction": row["fraction_stars"],
            "test_mode": row["test_mode"], "pairing_col": row["pairing_col"],
        })
        for sample_name, tp_a, tp_b, frac_a, frac_b in zip(
            row["sample_names"], row["sample_values_a"], row["sample_values_b"],
            row["sample_fraction_values_a"], row["sample_fraction_values_b"],
        ):
            paired_csv_rows.append({
                "page_group": page_group_value,
                "group_a": row["group_a"], "group_b": row["group_b"],
                sample_col: sample_name,
                "mean_a_timepoints": tp_a, "mean_b_timepoints": tp_b,
                "diff_timepoints": tp_b - tp_a,
                "mean_a_minutes": tp_a * minutes_per_frame if minutes_per_frame is not None else float("nan"),
                "mean_b_minutes": tp_b * minutes_per_frame if minutes_per_frame is not None else float("nan"),
                "mean_a_fraction": frac_a, "mean_b_fraction": frac_b, "diff_fraction": frac_b - frac_a,
            })
    return csv_rows, paired_csv_rows, len(pages) * 2


_LONG_CONTACT_BUCKET_COL = "_long_contact_bucket"
_LONG_CONTACT_STACK_ORDER = ["short_contact", "long_contact"]
_LONG_CONTACT_STACK_COLORS = {"short_contact": "#B0B0B0", "long_contact": "#D1495B"}
_LONG_CONTACT_UNITS = ("percent", "seconds", "minutes", "hours")
# Minutes represented by one unit of each choice — lets the threshold (always expressed in its
# own unit, e.g. 5 for "5 minutes" or 30 for "30 seconds") be compared against
# ``duration_timepoints * minutes_per_frame`` (always in minutes) via a single division.
_LONG_CONTACT_MINUTES_PER_UNIT = {"seconds": 1.0 / 60.0, "minutes": 1.0, "hours": 60.0}
_LONG_CONTACT_UNIT_SUFFIX = {"percent": "%", "seconds": "s", "minutes": "min", "hours": "h"}


def _format_long_contact_threshold(long_contact_threshold, long_contact_unit):
    if long_contact_unit == "percent":
        return f"{long_contact_threshold:g}%"
    return f"{long_contact_threshold:g} {_LONG_CONTACT_UNIT_SUFFIX[long_contact_unit]}"


def _compute_per_sample_long_contact_pct(duration_df, *, sample_col, class_order):
    """Per (sample, class) actually touched, the % of that sample's tracks touching that class
    whose contact reached the long-contact threshold — the per-sample distribution boxplotted (one
    box per class) alongside the pooled stacked bar."""
    grouped = (
        duration_df.groupby([sample_col, "target_class"], observed=True)[_LONG_CONTACT_BUCKET_COL]
        .apply(lambda s: 100.0 * float((s == "long_contact").mean()))
        .rename("pct_long_contact")
        .reset_index()
    )
    return grouped


def _draw_long_contact_boxplot(ax, per_sample_df, class_order, colors):
    """One box per class — dots are per-sample % long contact (``_compute_per_sample_long_contact_pct``).

    The y-axis is scaled to the actual spread of the data (with headroom for the jittered dots
    and "n=" labels) rather than always spanning the full 0-100% range, since real long-contact
    percentages are often much smaller than 100%.
    """
    data = [
        per_sample_df.loc[per_sample_df["target_class"] == cls, "pct_long_contact"].to_numpy(dtype=float)
        for cls in class_order
    ]
    bp = ax.boxplot(data, tick_labels=class_order, patch_artist=True, widths=0.75, showfliers=False)
    for patch, cls in zip(bp["boxes"], class_order):
        patch.set_facecolor(colors.get(cls, "#808080"))
        patch.set_alpha(0.55)

    finite_vals = np.concatenate([v for v in data if len(v)]) if any(len(v) for v in data) else np.array([])
    y_max = float(finite_vals.max()) if finite_vals.size else 100.0
    y_min = min(0.0, float(finite_vals.min())) if finite_vals.size else 0.0
    span = max(y_max - y_min, 1.0)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, color="black", alpha=0.5, zorder=3)
        ax.text(i, y_max + span * 0.05, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.5, clip_on=False)
    ax.set_ylabel("% tracks with long contact", fontsize=8)
    ax.set_ylim(y_min - span * 0.08, y_max + span * 0.20)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _long_contact_pivot(per_sample_df, class_order, sample_col):
    pivot = per_sample_df.pivot(index=sample_col, columns="target_class", values="pct_long_contact")
    return pivot.reindex(columns=class_order)


def _long_contact_sample_jitters(pivot, width=0.25, seed=0):
    """One jitter offset per sample (keyed by sample name, not position) — used by
    ``_draw_long_contact_connected_box`` to keep same-class dots from stacking exactly on top of
    each other; the companion lines-only panel doesn't jitter (see
    ``_draw_long_contact_connected_lines``)."""
    rng = np.random.default_rng(seed)
    return {s: (rng.random() - 0.5) * width for s in pivot.index}


def _draw_long_contact_connected_box(ax, per_sample_df, class_order, colors, sample_colors, sample_markers, sample_col):
    """Companion to ``_draw_long_contact_boxplot`` — same per-class box (built from the same
    per-sample % values), but dots are drawn with a per-sample marker shape (``_sample_marker_map``)
    instead of unlinked black jittered dots — no connecting line (that lives in the companion
    ``_draw_long_contact_connected_lines`` panel). Markers are solid black as long as
    ``sample_markers`` has a unique shape per sample; once samples outnumber the available shapes,
    markers fall back to per-sample color so repeated shapes stay disambiguated."""
    pivot = _long_contact_pivot(per_sample_df, class_order, sample_col)

    data = [pivot[cls].dropna().to_numpy(dtype=float) for cls in class_order]
    bp = ax.boxplot(data, tick_labels=class_order, patch_artist=True, widths=0.6, showfliers=False, zorder=1)
    for patch, cls in zip(bp["boxes"], class_order):
        patch.set_facecolor(colors.get(cls, "#808080"))
        patch.set_alpha(0.4)

    finite_vals = np.concatenate([v for v in data if len(v)]) if any(len(v) for v in data) else np.array([])
    y_max = float(finite_vals.max()) if finite_vals.size else 100.0
    y_min = min(0.0, float(finite_vals.min())) if finite_vals.size else 0.0
    span = max(y_max - y_min, 1.0)

    shapes_unique = len(sample_markers) <= len(_SAMPLE_MARKER_SHAPES)
    x_positions = {cls: i + 1 for i, cls in enumerate(class_order)}
    jitters = _long_contact_sample_jitters(pivot)
    for sample_name, row_ser in pivot.iterrows():
        present = [(cls, row_ser[cls]) for cls in class_order if pd.notna(row_ser[cls])]
        if not present:
            continue
        c = "black" if shapes_unique else sample_colors.get(str(sample_name), "#808080")
        m = sample_markers.get(str(sample_name), "o")
        offset = jitters[sample_name]
        xs = [x_positions[cls] + offset for cls, _ in present]
        ys = [val for _, val in present]
        ax.scatter(xs, ys, color=c, marker=m, s=16, zorder=3, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("% tracks with long contact", fontsize=8)
    ax.set_ylim(y_min - span * 0.08, y_max + span * 0.20)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _draw_long_contact_connected_lines(ax, per_sample_df, class_order, colors, sample_col):
    """Companion to ``_draw_long_contact_connected_box`` — no box, just each sample's line across
    whichever classes it touched (the connecting lines the box panel deliberately omits), all at
    the same (unjittered) per-class x. The line itself is a neutral grey; each dot is colored by
    the class it sits at (``colors[cls]`` — the same per-class colors the box panel's fill uses),
    so the panel reads as a sequence of classes rather than a set of samples. A bold black line on
    top traces the per-class mean across ``class_order``. Drawn with normal axes/ticks (not
    sharing the box panel's hidden-spine style) so it stands on its own."""
    pivot = _long_contact_pivot(per_sample_df, class_order, sample_col)
    x_positions = {cls: i + 1 for i, cls in enumerate(class_order)}
    for sample_name, row_ser in pivot.iterrows():
        present = [(cls, row_ser[cls]) for cls in class_order if pd.notna(row_ser[cls])]
        if len(present) < 2:
            continue
        xs = [x_positions[cls] for cls, _ in present]
        ys = [val for _, val in present]
        ax.plot(xs, ys, color="#999999", lw=1.0, alpha=0.7, zorder=2)

    for cls in class_order:
        vals = pivot[cls].dropna().to_numpy(dtype=float)
        if len(vals):
            ax.scatter(
                np.full(len(vals), x_positions[cls]), vals, color=colors.get(cls, "#808080"),
                s=45, zorder=3, edgecolor="white", linewidth=0.5,
            )

    mean_present = [(cls, pivot[cls].dropna().mean()) for cls in class_order]
    mean_present = [(cls, m) for cls, m in mean_present if pd.notna(m)]
    if mean_present:
        mean_xs = [x_positions[cls] for cls, _ in mean_present]
        mean_ys = [m for _, m in mean_present]
        if len(mean_xs) >= 2:
            ax.plot(mean_xs, mean_ys, color="black", lw=2.5, alpha=0.9, zorder=5)
        ax.text(
            mean_xs[-1] + 0.15, mean_ys[-1], "mean", fontsize=6, color="black", va="center",
            fontweight="bold",
        )

    ax.set_xlim(0.5, len(class_order) + 0.5)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(class_order, fontsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _plot_long_contact_percentage_page(
    counts, per_sample_df, *, class_order, target_cell_type_label, long_contact_threshold,
    long_contact_unit, colors, section_label=None,
):
    """Two panels: the pooled long-vs-short stacked bar (all touching tracks), and a boxplot of
    each sample's % long contact per class — for tracks (already restricted to tracks that
    touched that class at all) whose contact with it reached ``long_contact_threshold`` (expressed
    in ``long_contact_unit``) ("long_contact") vs. didn't ("short_contact")."""
    props = counts.div(counts.sum(axis=1), axis=0).reindex(columns=_LONG_CONTACT_STACK_ORDER, fill_value=0.0)
    panel_w = max(0.6 * len(class_order), 4.0)
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(panel_w * 2, 4.2))
    draw_stacked_proportion_barv(
        ax_bar, props, class_order, _LONG_CONTACT_STACK_ORDER, _LONG_CONTACT_STACK_COLORS, ymax=1.0, xtick_fontsize=8,
    )
    ax_bar.set_ylabel("Fraction of tracks in contact", fontsize=8)
    for i, cls in enumerate(class_order):
        n_total = int(counts.loc[cls].sum()) if cls in counts.index else 0
        ax_bar.text(i, 1.02, f"n={n_total}", ha="center", va="bottom", fontsize=6.5, clip_on=False)

    _draw_long_contact_boxplot(ax_box, per_sample_df, class_order, colors)

    threshold_label = _format_long_contact_threshold(long_contact_threshold, long_contact_unit)
    legend_labels = {
        "long_contact": f"long contact (≥ {threshold_label})",
        "short_contact": f"short contact (< {threshold_label})",
    }
    handles = [
        Patch(facecolor=_LONG_CONTACT_STACK_COLORS[s], label=legend_labels[s])
        for s in reversed(_LONG_CONTACT_STACK_ORDER)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)

    title = f"Long vs. short contact — % of tracks in contact with each {target_cell_type_label} class"
    if section_label is not None:
        title += f"\ngroup: {section_label}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.86))
    return fig


def _plot_long_contact_percentage_connected_page(
    counts, per_sample_df, *, class_order, target_cell_type_label, long_contact_threshold,
    long_contact_unit, colors, sample_colors, sample_markers, sample_col, section_label=None,
):
    """Companion to ``_plot_long_contact_percentage_page`` — same pooled stacked bar and a box
    whose dots are colored/shaped by sample (``_draw_long_contact_connected_box``, no connecting
    line), plus a third panel to its right with just each sample's line across the classes it
    touched (``_draw_long_contact_connected_lines``), sharing the box's y-axis. Rendered as a
    separate page right after the original."""
    props = counts.div(counts.sum(axis=1), axis=0).reindex(columns=_LONG_CONTACT_STACK_ORDER, fill_value=0.0)
    bar_w = max(0.6 * len(class_order), 4.0)
    box_w = max(0.6 * len(class_order), 4.0)
    lines_w = max(0.4 * len(class_order), 2.5)
    sample_names = sorted(per_sample_df[sample_col].dropna().unique().tolist(), key=_mixed_label_sort_key)
    ncol, n_legend_rows, legend_margin = legend_layout(
        len(sample_names), max_ncol=8, row_height_in=0.2, base_margin_in=0.55,
    )
    fig, (ax_bar, ax_box, ax_lines) = plt.subplots(
        1, 3, figsize=(bar_w + box_w + lines_w, 4.2 + legend_margin),
        gridspec_kw={"width_ratios": [bar_w, box_w, lines_w]},
    )
    draw_stacked_proportion_barv(
        ax_bar, props, class_order, _LONG_CONTACT_STACK_ORDER, _LONG_CONTACT_STACK_COLORS, ymax=1.0, xtick_fontsize=8,
    )
    ax_bar.set_ylabel("Fraction of tracks in contact", fontsize=8)
    for i, cls in enumerate(class_order):
        n_total = int(counts.loc[cls].sum()) if cls in counts.index else 0
        ax_bar.text(i, 1.02, f"n={n_total}", ha="center", va="bottom", fontsize=6.5, clip_on=False)

    _draw_long_contact_connected_box(
        ax_box, per_sample_df, class_order, colors, sample_colors, sample_markers, sample_col,
    )
    ax_lines.sharey(ax_box)
    _draw_long_contact_connected_lines(ax_lines, per_sample_df, class_order, colors, sample_col)

    threshold_label = _format_long_contact_threshold(long_contact_threshold, long_contact_unit)
    legend_labels = {
        "long_contact": f"long contact (≥ {threshold_label})",
        "short_contact": f"short contact (< {threshold_label})",
    }
    bar_handles = [
        Patch(facecolor=_LONG_CONTACT_STACK_COLORS[s], label=legend_labels[s])
        for s in reversed(_LONG_CONTACT_STACK_ORDER)
    ]
    # See ``_plot_duration_comparison_connected_page`` for why this mirrors the box panel's own
    # black-until-shapes-run-out rule, and drops the line stroke.
    shapes_unique = len(sample_markers) <= len(_SAMPLE_MARKER_SHAPES)
    sample_handles = [
        Line2D(
            [0], [0], marker=sample_markers.get(s, "o"),
            color="black" if shapes_unique else sample_colors.get(s, "#808080"),
            label=s, linestyle="None", markersize=5,
        )
        for s in sample_names
    ]
    bottom_margin = 0.1 + legend_margin / (4.2 + legend_margin)
    bar_legend_y = bottom_margin - 0.04
    fig.legend(handles=bar_handles, loc="lower center", bbox_to_anchor=(0.5, bar_legend_y), ncol=2, frameon=False, fontsize=8)
    fig.legend(
        handles=sample_handles, title=f"sample ({sample_col})", loc="lower center",
        bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False, fontsize=6, title_fontsize=6.5,
        handlelength=1.3, labelspacing=0.3, columnspacing=1.0,
    )

    title = (
        f"Long vs. short contact — connected by {sample_col} "
        f"(% of tracks in contact with each {target_cell_type_label} class)"
    )
    if section_label is not None:
        title += f"\ngroup: {section_label}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.86))
    return fig


def _render_long_contact_percentage_section(
    pdf, sub_duration_df, *, class_order, target_cell_type_label, long_contact_threshold,
    long_contact_unit, sample_col, sample_colors, sample_markers, class_colors=None, section_label=None,
):
    """Build + render the long-vs-short contact percentage page for one data subset (mirrors
    ``_render_duration_comparison_section``'s pooled/group-split convention). Restricted, per
    class, to tracks that actually touched that class (``sub_duration_df`` already is) — this is
    a comparison of long vs. short contact *among tracks with any contact*, not vs. no contact at
    all.

    Also renders a "connected by sample" companion page right after the original (see
    ``_plot_long_contact_percentage_connected_page``). Returns ``(csv_rows, per_sample_csv_rows,
    n_pages)`` — ``per_sample_csv_rows`` is the per-(sample, class) % long-contact data behind the
    connected page's dots (one row per sample per touched class).
    """
    page_group_value = "(all)" if section_label is None else str(section_label)
    present_classes = set(sub_duration_df["target_class"].dropna().unique().tolist())
    local_class_order = [c for c in class_order if c in present_classes]

    if not local_class_order:
        fig, ax = plt.subplots(figsize=(8, 4))
        where = f" for group '{section_label}'" if section_label is not None else ""
        ax.text(
            0.5, 0.5,
            f"No touched '{target_cell_type_label}' classes found{where} — nothing to plot.",
            ha="center", va="center", wrap=True,
        )
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        return [], [], 1

    counts = (
        pd.crosstab(sub_duration_df["target_class"], sub_duration_df[_LONG_CONTACT_BUCKET_COL])
        .reindex(index=local_class_order, columns=_LONG_CONTACT_STACK_ORDER, fill_value=0)
    )
    per_sample_df = _compute_per_sample_long_contact_pct(
        sub_duration_df, sample_col=sample_col, class_order=local_class_order,
    )
    colors = hash_stable_label_color_map(local_class_order, colors=class_colors)

    fig = _plot_long_contact_percentage_page(
        counts, per_sample_df, class_order=local_class_order, target_cell_type_label=target_cell_type_label,
        long_contact_threshold=long_contact_threshold, long_contact_unit=long_contact_unit,
        colors=colors, section_label=section_label,
    )
    pdf.savefig(fig)
    plt.close(fig)

    connected_fig = _plot_long_contact_percentage_connected_page(
        counts, per_sample_df, class_order=local_class_order, target_cell_type_label=target_cell_type_label,
        long_contact_threshold=long_contact_threshold, long_contact_unit=long_contact_unit,
        colors=colors, sample_colors=sample_colors, sample_markers=sample_markers,
        sample_col=sample_col, section_label=section_label,
    )
    pdf.savefig(connected_fig)
    plt.close(connected_fig)

    csv_rows = []
    for cls in local_class_order:
        n_long = int(counts.loc[cls, "long_contact"])
        n_short = int(counts.loc[cls, "short_contact"])
        n_total = n_long + n_short
        csv_rows.append({
            "page_group": page_group_value,
            "target_class": cls,
            "n_total": n_total,
            "n_long_contact": n_long,
            "n_short_contact": n_short,
            "pct_long_contact": (n_long / n_total) if n_total else float("nan"),
            "pct_short_contact": (n_short / n_total) if n_total else float("nan"),
            "long_contact_threshold": long_contact_threshold,
            "long_contact_unit": long_contact_unit,
        })
    per_sample_csv_rows = [
        {"page_group": page_group_value, **row}
        for row in per_sample_df.to_dict(orient="records")
    ]
    return csv_rows, per_sample_csv_rows, 2


def save_track_contact_duration_comparison(
    adata_tracks,
    df_timepoints,
    out_dir,
    *,
    contact_col,
    min_bout_length,
    target_class_lookup,
    touching_col,
    time_varying,
    target_cell_type_label,
    class_order=None,
    class_colors=None,
    test_mode="welch",
    pairing_col=None,
    minutes_per_frame=None,
    long_contact_threshold=None,
    long_contact_unit="minutes",
    sample_col="sample_name",
    groupby_cols=("sample_name", "TrackID"),
    comparisons_per_page=12,
    group_cols=None,
    verbose=False,
):
    """For every class of ``target_cell_type_label`` actually touched, compare how long tracks
    stay in contact with that class — pairwise between classes, and each class vs. every other
    class pooled ("rest") — using an unpaired Welch's t-test or a paired t-test (pairs formed by
    averaging within ``pairing_col`` before pairing).

    ``pairing_col`` (only used when ``test_mode="paired"``) is a single column name (e.g.
    ``"sample_name"``) or a list of column names — when more than one is given, pairing units are
    formed on the combination of all of them (e.g. ``sample_name`` + ``exp_nr``), joined with
    ``" | "``; the CSV's ``pairing_col`` field records the human-readable ``" + "``-joined name(s).

    Requires the per-cell contact-attribution path (``target_class_lookup``/``touching_col`` —
    see ``contact_grouping.build_target_class_lookup_from_state_adata`` /
    ``build_target_class_lookup_from_track_adata``); there is nothing to compare "by class"
    without it.

    ``group_cols``, when given (column names in ``adata_tracks.obs``, e.g. ``["exp_nr"]``), splits
    the comparison into one page-group per unique combination of those columns' values — mirroring
    the "group per page" behavior of ``save_track_contact_group_analysis``. The pooled/general
    comparison (over all tracks, unaffected by ``group_cols``) is always rendered first, followed
    by one section (its own page(s)) per group; a group missing a touched class simply skips
    comparisons involving it. Columns not already in ``adata_tracks.obs`` raise ``KeyError`` (merge
    them in first, e.g. via ``core.metadata.merge_condition_columns_into_obs``).

    ``long_contact_threshold``, when given, adds one extra page per section: for tracks in contact
    with each touched class, the percentage whose contact with that class reached
    ``long_contact_threshold`` ("long_contact") vs. didn't ("short_contact") — restricted to tracks
    that touched that class at all (i.e. long vs. short contact, not vs. no contact) — as a pooled
    stacked bar plus a boxplot of each ``sample_col`` value's own % long contact per class, written
    to a separate ``contact_long_contact_percentage.csv``.

    ``long_contact_unit`` picks what ``long_contact_threshold`` is expressed in and which metric it
    is compared against:

    - ``"percent"`` — the track's time-in-contact fraction with that class (the proportion of its
      classified time window spent in contact with it, contiguity-agnostic — see
      ``compute_track_contact_target_class_features``'s ``{contact_col}_class_mean_fraction``),
      as a percentage (0-100]. Scale-invariant per track; no time metadata required.
    - ``"seconds"``/``"minutes"``/``"hours"`` — the longest sustained contact bout with that class
      (``duration_timepoints``), converted to real time via ``minutes_per_frame`` (required for
      these units).

    Writes one combined PDF (``contact_duration_comparison.pdf``, small boxplot groups — timepoints,
    minutes (when ``minutes_per_frame`` is given), and time-in-contact fraction side by side —
    paginated ``comparisons_per_page`` per page, plus the long-contact percentage page(s) when
    requested) plus a CSV with one row per comparison (a ``page_group`` column marks which section —
    ``"(all)"`` for the pooled one — each row belongs to), into the same
    ``{out_dir}/contact_analysis/{contact_col}/`` folder used by ``save_track_contact_group_analysis``.

    Right after each of those pages (and the long-contact percentage page, when requested), a
    "connected by sample" companion page is rendered: the same comparison, but each dot is a
    per-``sample_col`` mean, colored by sample and joined by a line across the two (or, for
    long-contact, all touched) classes, so an individual sample's shift is traceable — with its
    own paired significance test (independent of ``test_mode``, which always pairs by
    ``sample_col`` for this view). The per-sample values behind those dots are written to
    ``contact_duration_comparison_paired_samples.csv`` and, when ``long_contact_threshold`` is
    set, ``contact_long_contact_percentage_per_sample.csv``.

    Returns a dict of artifact paths (including ``paired_csv_path`` and, when applicable,
    ``long_contact_per_sample_csv_path``) plus ``n_comparisons``/``n_pages``/``class_order``/
    ``sample_col``/``group_cols``/``page_groups``.
    """
    test_mode = str(test_mode).strip().lower()
    if test_mode not in ("welch", "paired"):
        raise ValueError(f"test_mode must be 'welch' or 'paired', got {test_mode!r}.")

    pairing_cols = []
    if test_mode == "paired":
        pairing_cols = (
            [str(pairing_col)] if isinstance(pairing_col, str)
            else [str(c) for c in (pairing_col or []) if str(c).strip()]
        )
        if not pairing_cols:
            raise ValueError("pairing_col is required when test_mode='paired'.")
    pairing_col_label = " + ".join(pairing_cols) if pairing_cols else None

    if long_contact_threshold is not None:
        long_contact_unit = str(long_contact_unit).strip().lower()
        if long_contact_unit not in _LONG_CONTACT_UNITS:
            raise ValueError(
                f"long_contact_unit must be one of {_LONG_CONTACT_UNITS}, got {long_contact_unit!r}."
            )
        long_contact_threshold = float(long_contact_threshold)
        if long_contact_threshold <= 0:
            raise ValueError(f"long_contact_threshold must be > 0, got {long_contact_threshold!r}.")
        if long_contact_unit == "percent" and long_contact_threshold > 100.0:
            raise ValueError(
                f"long_contact_threshold must be in (0, 100] for long_contact_unit='percent', "
                f"got {long_contact_threshold!r}."
            )
        if long_contact_unit != "percent" and not minutes_per_frame:
            raise ValueError(
                f"long_contact_unit={long_contact_unit!r} requires minutes_per_frame (time "
                f"metadata) to convert contact-bout length to real time."
            )

    groupby_cols = [str(c) for c in list(groupby_cols)]
    group_col = _contact_group_col_name(contact_col)
    mean_col = _contact_class_mean_col_name(contact_col)
    max_bout_col = _contact_class_max_bout_col_name(contact_col)

    contact_features = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col=contact_col, min_bout_length=min_bout_length,
        groupby_cols=groupby_cols, verbose=verbose,
    )
    merge_track_contact_features_into_obs(
        adata_tracks, contact_features, contact_col=contact_col, min_bout_length=min_bout_length,
        groupby_cols=groupby_cols,
    )
    long_target_df, _group_df = compute_track_contact_target_class_features(
        df_timepoints, adata_tracks, target_class_lookup,
        contact_col=contact_col, touching_col=touching_col, time_varying=bool(time_varying),
        contact_group_col=group_col, groupby_cols=groupby_cols, verbose=verbose,
    )

    duration_df = long_target_df.reset_index().rename(
        columns={max_bout_col: "duration_timepoints", mean_col: "duration_fraction"}
    )
    duration_df["target_class"] = duration_df["target_class"].astype(str)

    if sample_col not in duration_df.columns:
        raise KeyError(
            f"sample_col={sample_col!r} not found — required (as the per-sample "
            f"connected-plot/boxplot unit) for both the duration comparison and the long-contact "
            f"percentage report."
        )

    group_cols = [str(c) for c in group_cols] if group_cols else []

    extra_cols_needed = [c for c in pairing_cols + group_cols if c not in duration_df.columns]
    if extra_cols_needed:
        missing_from_obs = [c for c in extra_cols_needed if c not in adata_tracks.obs.columns]
        if missing_from_obs:
            raise KeyError(
                f"Column(s) {missing_from_obs} not found in adata_tracks.obs — merge them in "
                f"before calling this function (e.g. via "
                f"core.metadata.merge_condition_columns_into_obs)."
            )
        extra_lookup = (
            adata_tracks.obs[groupby_cols + extra_cols_needed].drop_duplicates(subset=groupby_cols)
        )
        duration_df = duration_df.merge(extra_lookup, on=groupby_cols, how="left")

    pairing_col_actual = None
    if pairing_cols:
        if len(pairing_cols) == 1:
            pairing_col_actual = pairing_cols[0]
        else:
            duration_df["_pairing_composite"] = _make_group_label(duration_df, pairing_cols)
            pairing_col_actual = "_pairing_composite"

    if long_contact_threshold is not None:
        if long_contact_unit == "percent":
            long_contact_value = duration_df["duration_fraction"] * 100.0
        else:
            minutes_per_unit = _LONG_CONTACT_MINUTES_PER_UNIT[long_contact_unit]
            long_contact_value = duration_df["duration_timepoints"] * minutes_per_frame / minutes_per_unit
        duration_df[_LONG_CONTACT_BUCKET_COL] = np.where(
            long_contact_value >= long_contact_threshold, "long_contact", "short_contact",
        )

    touched_classes = sorted(
        duration_df["target_class"].dropna().unique().tolist(), key=_mixed_label_sort_key,
    )
    resolved_class_order = [str(c) for c in class_order] if class_order is not None else touched_classes
    resolved_class_order = [c for c in resolved_class_order if c in touched_classes]

    all_sample_names = sorted(
        duration_df[sample_col].dropna().astype(str).unique().tolist(), key=_mixed_label_sort_key,
    )
    sample_colors = hash_stable_label_color_map(all_sample_names)
    sample_markers = _sample_marker_map(all_sample_names)

    out_dir = Path(out_dir) / "contact_analysis" / str(contact_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_duration_comparison.pdf"
    csv_path = csv_dir / "contact_duration_comparison.csv"
    paired_csv_path = csv_dir / "contact_duration_comparison_paired_samples.csv"
    long_contact_csv_path = csv_dir / "contact_long_contact_percentage.csv"
    long_contact_per_sample_csv_path = csv_dir / "contact_long_contact_percentage_per_sample.csv"

    csv_columns = [
        "page_group", "group_a", "group_b", "n_a", "n_b", "mean_a_timepoints", "mean_b_timepoints",
        "diff_timepoints", "mean_a_minutes", "mean_b_minutes",
        "mean_a_fraction", "mean_b_fraction", "diff_fraction",
        "t_stat", "p_value", "stars", "t_stat_fraction", "p_value_fraction", "stars_fraction",
        "test_mode", "pairing_col",
    ]
    paired_csv_columns = [
        "page_group", "group_a", "group_b", sample_col,
        "mean_a_timepoints", "mean_b_timepoints", "diff_timepoints",
        "mean_a_minutes", "mean_b_minutes",
        "mean_a_fraction", "mean_b_fraction", "diff_fraction",
    ]
    long_csv_columns = [
        "page_group", "target_class", "n_total", "n_long_contact", "n_short_contact",
        "pct_long_contact", "pct_short_contact", "long_contact_threshold", "long_contact_unit",
    ]
    long_per_sample_csv_columns = ["page_group", sample_col, "target_class", "pct_long_contact"]

    comparisons_per_page = max(1, int(comparisons_per_page))
    ncols = min(4, max(1, int(np.ceil(np.sqrt(comparisons_per_page)))))

    group_label_series = _make_group_label(duration_df, group_cols) if group_cols else None
    page_groups = (
        sorted(group_label_series.dropna().unique().tolist(), key=_mixed_label_sort_key)
        if group_label_series is not None else []
    )

    csv_rows = []
    paired_csv_rows = []
    long_csv_rows = []
    long_per_sample_csv_rows = []
    total_pages = 0
    with PdfPages(pdf_path) as pdf:
        if len(resolved_class_order) < 2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5,
                f"Fewer than 2 touched '{target_cell_type_label}' classes found — nothing to compare.",
                ha="center", va="center", wrap=True,
            )
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            total_pages += 1
        else:
            general_rows, general_paired_rows, general_pages = _render_duration_comparison_section(
                pdf, duration_df, class_order=resolved_class_order, contact_col=contact_col,
                target_cell_type_label=target_cell_type_label, test_mode=test_mode,
                pairing_col=pairing_col_actual, sample_col=sample_col, sample_colors=sample_colors,
                sample_markers=sample_markers,
                pairing_col_label=pairing_col_label,
                minutes_per_frame=minutes_per_frame, ncols=ncols,
                class_colors=class_colors, comparisons_per_page=comparisons_per_page,
                section_label=None,
            )
            csv_rows.extend(general_rows)
            paired_csv_rows.extend(general_paired_rows)
            total_pages += general_pages

            for page_group in page_groups:
                sub_df = duration_df[group_label_series == page_group]
                if len(sub_df) == 0:
                    continue
                group_rows, group_paired_rows, group_pages = _render_duration_comparison_section(
                    pdf, sub_df, class_order=resolved_class_order, contact_col=contact_col,
                    target_cell_type_label=target_cell_type_label, test_mode=test_mode,
                    pairing_col=pairing_col_actual, sample_col=sample_col, sample_colors=sample_colors,
                    sample_markers=sample_markers,
                    pairing_col_label=pairing_col_label,
                    minutes_per_frame=minutes_per_frame, ncols=ncols,
                    class_colors=class_colors, comparisons_per_page=comparisons_per_page,
                    section_label=page_group,
                )
                csv_rows.extend(group_rows)
                paired_csv_rows.extend(group_paired_rows)
                total_pages += group_pages

        if long_contact_threshold is not None:
            long_rows, long_per_sample_rows, long_pages = _render_long_contact_percentage_section(
                pdf, duration_df, class_order=resolved_class_order,
                target_cell_type_label=target_cell_type_label,
                long_contact_threshold=long_contact_threshold, long_contact_unit=long_contact_unit,
                sample_col=sample_col, sample_colors=sample_colors, sample_markers=sample_markers,
                class_colors=class_colors,
                section_label=None,
            )
            long_csv_rows.extend(long_rows)
            long_per_sample_csv_rows.extend(long_per_sample_rows)
            total_pages += long_pages

            for page_group in page_groups:
                sub_df = duration_df[group_label_series == page_group]
                if len(sub_df) == 0:
                    continue
                long_rows, long_per_sample_rows, long_pages = _render_long_contact_percentage_section(
                    pdf, sub_df, class_order=resolved_class_order,
                    target_cell_type_label=target_cell_type_label,
                    long_contact_threshold=long_contact_threshold, long_contact_unit=long_contact_unit,
                    sample_col=sample_col, sample_colors=sample_colors, sample_markers=sample_markers,
                    class_colors=class_colors,
                    section_label=page_group,
                )
                long_csv_rows.extend(long_rows)
                long_per_sample_csv_rows.extend(long_per_sample_rows)
                total_pages += long_pages

    pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_path, index=False)
    pd.DataFrame(paired_csv_rows, columns=paired_csv_columns).to_csv(paired_csv_path, index=False)
    if long_contact_threshold is not None:
        pd.DataFrame(long_csv_rows, columns=long_csv_columns).to_csv(long_contact_csv_path, index=False)
        pd.DataFrame(long_per_sample_csv_rows, columns=long_per_sample_csv_columns).to_csv(
            long_contact_per_sample_csv_path, index=False,
        )

    if verbose:
        group_note = f", {len(page_groups)} group page(s) by {group_cols}" if group_cols else ""
        long_note = (
            f", long_contact_threshold={long_contact_threshold} {long_contact_unit}"
            if long_contact_threshold is not None else ""
        )
        print(
            f"Saved contact duration comparison ({len(csv_rows)} comparisons, {total_pages} page(s)"
            f"{group_note}, test_mode={test_mode}{long_note}): {pdf_path}"
        )

    return {
        "contact_col": str(contact_col),
        "min_bout_length": int(min_bout_length),
        "target_cell_type_label": str(target_cell_type_label),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "paired_csv_path": str(paired_csv_path),
        "csv_dir": str(csv_dir),
        "n_comparisons": len(csv_rows),
        "n_pages": total_pages,
        "test_mode": test_mode,
        "pairing_col": pairing_col_label,
        "sample_col": str(sample_col),
        "class_order": resolved_class_order,
        "minutes_per_frame": minutes_per_frame,
        "long_contact_threshold": long_contact_threshold,
        "long_contact_unit": long_contact_unit if long_contact_threshold is not None else None,
        "long_contact_csv_path": str(long_contact_csv_path) if long_contact_threshold is not None else None,
        "long_contact_per_sample_csv_path": (
            str(long_contact_per_sample_csv_path) if long_contact_threshold is not None else None
        ),
        "group_cols": group_cols,
        "page_groups": [str(g) for g in page_groups],
    }

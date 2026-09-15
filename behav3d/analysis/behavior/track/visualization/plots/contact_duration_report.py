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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from behav3d.analysis.behavior.utils import _mixed_label_sort_key
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import (
    hash_stable_label_color_map,
    welch_ttest_stars,
    SIGNIFICANCE_LEGEND_TEXT,
    compute_class_by_stack_proportions,
    draw_stacked_proportion_barv,
    _chunk_list,
    _make_group_label,
)
from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    merge_track_contact_features_into_obs,
    compute_track_contact_target_class_features,
    _contact_group_col_name,
    _contact_class_max_bout_col_name,
)

_MIN_GROUP_N_FOR_TEST = 2


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


def _comparison_row(duration_df, *, label_a, classes_a, label_b, classes_b, test_mode, pairing_col, pairing_col_label=None):
    """Compute stats plus the actual (timepoints) values tested, for one comparison.

    ``pairing_col`` is the actual dataframe column grouped on (may be an internal composite
    column when multiple pairing columns were combined); ``pairing_col_label`` is the
    human-readable name recorded in the output (defaults to ``pairing_col`` itself).
    """
    if test_mode == "paired":
        per_unit = duration_df.groupby([pairing_col, "target_class"])["duration_timepoints"].mean()
        target_level = per_unit.index.get_level_values("target_class")
        side_a = per_unit[target_level.isin(classes_a)].groupby(level=pairing_col).mean()
        side_b = per_unit[target_level.isin(classes_b)].groupby(level=pairing_col).mean()
        joined = pd.concat({"a": side_a, "b": side_b}, axis=1).dropna()
        values_a = joined["a"].to_numpy()
        values_b = joined["b"].to_numpy()
        stats_row = _paired_stats(values_a, values_b)
    else:
        values_a = duration_df.loc[duration_df["target_class"].isin(classes_a), "duration_timepoints"].to_numpy()
        values_b = duration_df.loc[duration_df["target_class"].isin(classes_b), "duration_timepoints"].to_numpy()
        stats_row = _welch_stats(values_a, values_b)

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
    return stats_row


def _draw_pair_box(ax, values_a, values_b, *, label_a, label_b, ylabel, stars, colors):
    data = [np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)]
    bp = ax.boxplot(data, tick_labels=[label_a, label_b], patch_artist=True, widths=0.6, showfliers=False)
    for patch, label in zip(bp["boxes"], (label_a, label_b)):
        patch.set_facecolor(colors.get(label, "#808080"))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.15
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color="black", alpha=0.4, zorder=3)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)

    finite_vals = [v for arr in data if len(arr) for v in arr[np.isfinite(arr)]]
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


def _plot_duration_comparison_page(
    page_rows, *, contact_col, target_cell_type_label, test_mode, pairing_col, minutes_per_frame,
    ncols, label_colors, section_label=None, pairing_col_label=None,
):
    n = len(page_rows)
    nrows = int(np.ceil(n / ncols))
    show_minutes = minutes_per_frame is not None
    fig = plt.figure(figsize=(11.69, 8.27))
    outer = fig.add_gridspec(nrows=max(1, nrows), ncols=max(1, ncols), hspace=1.0, wspace=0.6, top=0.86, bottom=0.10)

    for i, row in enumerate(page_rows):
        r, c = divmod(i, ncols)
        inner = outer[r, c].subgridspec(1, 2 if show_minutes else 1, wspace=0.7)
        ax_tp = fig.add_subplot(inner[0, 0])
        _draw_pair_box(
            ax_tp, row["values_a"], row["values_b"],
            label_a=row["group_a"], label_b=row["group_b"],
            ylabel="Duration (timepoints)", stars=row["stars"], colors=label_colors,
        )
        ax_tp.set_title(f"{row['group_a']} vs {row['group_b']}", fontsize=7)
        if show_minutes:
            ax_min = fig.add_subplot(inner[0, 1])
            _draw_pair_box(
                ax_min,
                np.asarray(row["values_a"], dtype=float) * minutes_per_frame,
                np.asarray(row["values_b"], dtype=float) * minutes_per_frame,
                label_a=row["group_a"], label_b=row["group_b"],
                ylabel="Duration (minutes)", stars=row["stars"], colors=label_colors,
            )

    subtitle = f"contact_col={contact_col}  target={target_cell_type_label}  test={test_mode}"
    if test_mode == "paired":
        subtitle += f"  pairing_col={pairing_col_label or pairing_col}"
    if not show_minutes:
        subtitle += "  (minutes unavailable — no time metadata)"
    title = f"Contact duration comparison — max sustained contact-bout length by {target_cell_type_label} class"
    if section_label is not None:
        title += f"\ngroup: {section_label}"
    else:
        title += "\n(all data pooled)"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.02, SIGNIFICANCE_LEGEND_TEXT, ha="center", va="bottom", fontsize=7)
    return fig


def _render_duration_comparison_section(
    pdf, sub_duration_df, *, class_order, contact_col, target_cell_type_label, test_mode,
    pairing_col, minutes_per_frame, ncols, class_colors, comparisons_per_page, section_label=None,
    pairing_col_label=None,
):
    """Build + render the pairwise/one-vs-rest comparison page(s) for one data subset — either
    the pooled "general" set (``section_label=None``) or one ``group_cols`` page-split's slice.

    Classes are restricted to ``class_order`` (the globally touched/resolved class list) filtered
    down to whichever of those are actually present in ``sub_duration_df``, so a group missing one
    class simply skips comparisons involving it rather than erroring.

    Returns ``(csv_rows, n_pages)`` — ``csv_rows`` already carries a ``"page_group"`` key
    (``"(all)"`` when ``section_label`` is None).
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
        return [], 1

    comparisons = _build_comparisons(local_class_order)
    rows = [
        _comparison_row(
            sub_duration_df, label_a=label_a, classes_a=classes_a, label_b=label_b, classes_b=classes_b,
            test_mode=test_mode, pairing_col=pairing_col, pairing_col_label=pairing_col_label,
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

    csv_rows = []
    for row in rows:
        m_a = row["mean_a"] * minutes_per_frame if minutes_per_frame is not None and np.isfinite(row["mean_a"]) else float("nan")
        m_b = row["mean_b"] * minutes_per_frame if minutes_per_frame is not None and np.isfinite(row["mean_b"]) else float("nan")
        csv_rows.append({
            "page_group": page_group_value,
            "group_a": row["group_a"], "group_b": row["group_b"],
            "n_a": row["n_a"], "n_b": row["n_b"],
            "mean_a_timepoints": row["mean_a"], "mean_b_timepoints": row["mean_b"],
            "diff_timepoints": row["diff"], "mean_a_minutes": m_a, "mean_b_minutes": m_b,
            "t_stat": row["t_stat"], "p_value": row["p_value"], "stars": row["stars"],
            "test_mode": row["test_mode"], "pairing_col": row["pairing_col"],
        })
    return csv_rows, len(pages)


_LONG_CONTACT_BUCKET_COL = "_long_contact_bucket"
_LONG_CONTACT_STACK_ORDER = ["short_contact", "long_contact"]
_LONG_CONTACT_STACK_COLORS = {"short_contact": "#B0B0B0", "long_contact": "#D1495B"}


def _compute_per_sample_long_contact_pct(duration_df, *, sample_col, class_order):
    """Per (sample, class) actually touched, the % of that sample's tracks touching that class
    whose contact reached ``long_contact_minutes`` — the per-sample distribution boxplotted (one
    box per class) alongside the pooled stacked bar."""
    grouped = (
        duration_df.groupby([sample_col, "target_class"], observed=True)[_LONG_CONTACT_BUCKET_COL]
        .apply(lambda s: 100.0 * float((s == "long_contact").mean()))
        .rename("pct_long_contact")
        .reset_index()
    )
    return grouped


def _draw_long_contact_boxplot(ax, per_sample_df, class_order, colors):
    """One box per class — dots are per-sample % long contact (``_compute_per_sample_long_contact_pct``)."""
    data = [
        per_sample_df.loc[per_sample_df["target_class"] == cls, "pct_long_contact"].to_numpy(dtype=float)
        for cls in class_order
    ]
    bp = ax.boxplot(data, tick_labels=class_order, patch_artist=True, widths=0.6, showfliers=False)
    for patch, cls in zip(bp["boxes"], class_order):
        patch.set_facecolor(colors.get(cls, "#808080"))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.15
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, color="black", alpha=0.5, zorder=3)
        ax.text(i, 102, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.5, clip_on=False)
    ax.set_ylabel("% tracks with long contact", fontsize=8)
    ax.set_ylim(-5, 112)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _plot_long_contact_percentage_page(
    counts, per_sample_df, *, class_order, target_cell_type_label, long_contact_minutes, colors, section_label=None,
):
    """Two panels: the pooled long-vs-short stacked bar (all touching tracks), and a boxplot of
    each sample's % long contact per class — for tracks (already restricted to tracks that
    touched that class at all) whose longest contact bout with it reached ``long_contact_minutes``
    ("long_contact") vs. didn't ("short_contact")."""
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

    legend_labels = {
        "long_contact": f"long contact (≥ {long_contact_minutes:g} min)",
        "short_contact": f"short contact (< {long_contact_minutes:g} min)",
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


def _render_long_contact_percentage_section(
    pdf, sub_duration_df, *, class_order, target_cell_type_label, long_contact_minutes, sample_col,
    class_colors=None, section_label=None,
):
    """Build + render the long-vs-short contact percentage page for one data subset (mirrors
    ``_render_duration_comparison_section``'s pooled/group-split convention). Restricted, per
    class, to tracks that actually touched that class (``sub_duration_df`` already is) — this is
    a comparison of long vs. short contact *among tracks with any contact*, not vs. no contact at
    all. Returns ``(csv_rows, n_pages)``.
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
        return [], 1

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
        long_contact_minutes=long_contact_minutes, colors=colors, section_label=section_label,
    )
    pdf.savefig(fig)
    plt.close(fig)

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
            "long_contact_minutes": long_contact_minutes,
        })
    return csv_rows, 1


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
    long_contact_minutes=None,
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

    ``long_contact_minutes``, when given (requires ``minutes_per_frame``), adds one extra page per
    section: for tracks in contact with each touched class, the percentage whose longest contact
    bout with that class reached ``long_contact_minutes`` ("long_contact") vs. didn't
    ("short_contact") — restricted to tracks that touched that class at all (i.e. long vs. short
    contact, not vs. no contact) — as a pooled stacked bar plus a boxplot of each ``sample_col``
    value's own % long contact per class, written to a separate
    ``contact_long_contact_percentage.csv``.

    Writes one combined PDF (``contact_duration_comparison.pdf``, small boxplot pairs — timepoints
    and minutes side by side when ``minutes_per_frame`` is given — paginated
    ``comparisons_per_page`` per page, plus the long-contact percentage page(s) when requested)
    plus a CSV with one row per comparison (a ``page_group`` column marks which section —
    ``"(all)"`` for the pooled one — each row belongs to), into the same
    ``{out_dir}/contact_analysis/{contact_col}/`` folder used by ``save_track_contact_group_analysis``.

    Returns a dict of artifact paths plus ``n_comparisons``/``n_pages``/``class_order``/
    ``group_cols``/``page_groups``.
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

    if long_contact_minutes is not None:
        long_contact_minutes = float(long_contact_minutes)
        if long_contact_minutes <= 0:
            raise ValueError(f"long_contact_minutes must be > 0, got {long_contact_minutes!r}.")
        if not minutes_per_frame:
            raise ValueError(
                "long_contact_minutes requires minutes_per_frame (time metadata) to convert "
                "minutes to timepoints."
            )

    groupby_cols = [str(c) for c in list(groupby_cols)]
    group_col = _contact_group_col_name(contact_col)
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

    duration_df = long_target_df.reset_index().rename(columns={max_bout_col: "duration_timepoints"})
    duration_df["target_class"] = duration_df["target_class"].astype(str)

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

    if long_contact_minutes is not None:
        if sample_col not in duration_df.columns:
            raise KeyError(
                f"sample_col={sample_col!r} not found — required (as the per-sample boxplot unit) "
                f"when long_contact_minutes is set."
            )
        long_contact_timepoints = long_contact_minutes / float(minutes_per_frame)
        duration_df[_LONG_CONTACT_BUCKET_COL] = np.where(
            duration_df["duration_timepoints"] >= long_contact_timepoints, "long_contact", "short_contact",
        )

    touched_classes = sorted(
        duration_df["target_class"].dropna().unique().tolist(), key=_mixed_label_sort_key,
    )
    resolved_class_order = [str(c) for c in class_order] if class_order is not None else touched_classes
    resolved_class_order = [c for c in resolved_class_order if c in touched_classes]

    out_dir = Path(out_dir) / "contact_analysis" / str(contact_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_duration_comparison.pdf"
    csv_path = csv_dir / "contact_duration_comparison.csv"
    long_contact_csv_path = csv_dir / "contact_long_contact_percentage.csv"

    csv_columns = [
        "page_group", "group_a", "group_b", "n_a", "n_b", "mean_a_timepoints", "mean_b_timepoints",
        "diff_timepoints", "mean_a_minutes", "mean_b_minutes", "t_stat", "p_value", "stars",
        "test_mode", "pairing_col",
    ]
    long_csv_columns = [
        "page_group", "target_class", "n_total", "n_long_contact", "n_short_contact",
        "pct_long_contact", "pct_short_contact", "long_contact_minutes",
    ]

    comparisons_per_page = max(1, int(comparisons_per_page))
    ncols = min(4, max(1, int(np.ceil(np.sqrt(comparisons_per_page)))))

    group_label_series = _make_group_label(duration_df, group_cols) if group_cols else None
    page_groups = (
        sorted(group_label_series.dropna().unique().tolist(), key=_mixed_label_sort_key)
        if group_label_series is not None else []
    )

    csv_rows = []
    long_csv_rows = []
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
            general_rows, general_pages = _render_duration_comparison_section(
                pdf, duration_df, class_order=resolved_class_order, contact_col=contact_col,
                target_cell_type_label=target_cell_type_label, test_mode=test_mode,
                pairing_col=pairing_col_actual, pairing_col_label=pairing_col_label,
                minutes_per_frame=minutes_per_frame, ncols=ncols,
                class_colors=class_colors, comparisons_per_page=comparisons_per_page,
                section_label=None,
            )
            csv_rows.extend(general_rows)
            total_pages += general_pages

            for page_group in page_groups:
                sub_df = duration_df[group_label_series == page_group]
                if len(sub_df) == 0:
                    continue
                group_rows, group_pages = _render_duration_comparison_section(
                    pdf, sub_df, class_order=resolved_class_order, contact_col=contact_col,
                    target_cell_type_label=target_cell_type_label, test_mode=test_mode,
                    pairing_col=pairing_col_actual, pairing_col_label=pairing_col_label,
                    minutes_per_frame=minutes_per_frame, ncols=ncols,
                    class_colors=class_colors, comparisons_per_page=comparisons_per_page,
                    section_label=page_group,
                )
                csv_rows.extend(group_rows)
                total_pages += group_pages

        if long_contact_minutes is not None:
            long_rows, long_pages = _render_long_contact_percentage_section(
                pdf, duration_df, class_order=resolved_class_order,
                target_cell_type_label=target_cell_type_label,
                long_contact_minutes=long_contact_minutes, sample_col=sample_col,
                class_colors=class_colors, section_label=None,
            )
            long_csv_rows.extend(long_rows)
            total_pages += long_pages

            for page_group in page_groups:
                sub_df = duration_df[group_label_series == page_group]
                if len(sub_df) == 0:
                    continue
                long_rows, long_pages = _render_long_contact_percentage_section(
                    pdf, sub_df, class_order=resolved_class_order,
                    target_cell_type_label=target_cell_type_label,
                    long_contact_minutes=long_contact_minutes, sample_col=sample_col,
                    class_colors=class_colors, section_label=page_group,
                )
                long_csv_rows.extend(long_rows)
                total_pages += long_pages

    pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_path, index=False)
    if long_contact_minutes is not None:
        pd.DataFrame(long_csv_rows, columns=long_csv_columns).to_csv(long_contact_csv_path, index=False)

    if verbose:
        group_note = f", {len(page_groups)} group page(s) by {group_cols}" if group_cols else ""
        long_note = f", long_contact_minutes={long_contact_minutes}" if long_contact_minutes is not None else ""
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
        "csv_dir": str(csv_dir),
        "n_comparisons": len(csv_rows),
        "n_pages": total_pages,
        "test_mode": test_mode,
        "pairing_col": pairing_col_label,
        "class_order": resolved_class_order,
        "minutes_per_frame": minutes_per_frame,
        "long_contact_minutes": long_contact_minutes,
        "long_contact_csv_path": str(long_contact_csv_path) if long_contact_minutes is not None else None,
        "group_cols": group_cols,
        "page_groups": [str(g) for g in page_groups],
    }

"""Killing by State report: descriptive killing efficiency per behavioral state.

Writes the CSVs for a downstream analysis of killing efficiency per state, plus one PDF with box
plots per cell and per timepoint side by side. No model is fitted and no test is run.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from behav3d.analysis.behavior.state.killing_by_state import (
    build_killing_timepoints,
    filter_counts,
    load_video_ends,
    recompute_killing_efficiency,
    summarize_killing_by_state,
)
from behav3d.analysis.behavior.state.utils import (
    _apply_state_order,
    _get_classification_state_colors,
    _get_classification_state_order,
    _normalize_label_color_map,
)

INK = "#333333"
MUTED = "#6b6b6b"
Y_LABEL = "Killing efficiency"

CELL_TITLE = "Killing efficiency per cell (cell means)"
CELL_DESCRIPTION = (
    "Each cell gives one value: the mean of its timepoints in that state. n = cells."
)
TIMEPOINT_TITLE = "Killing efficiency per timepoint"
TIMEPOINT_DESCRIPTION = (
    "Each timepoint gives one value, so cells that stay longer in a state weigh more. n = timepoints."
)
BOX_KEY = "Box: median and quartiles. Whiskers: 1.5 x IQR. Dots: values beyond them. Diamond: mean."


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, labelsize=7.5, labelcolor=MUTED)
    ax.tick_params(axis="y", width=0.8, labelsize=8.5)
    ax.margins(x=0.06)
    ax.set_axisbelow(True)


def _state_tick_labels(state_order, counts, unit="n"):
    labels = []
    for state in state_order:
        wrapped = "\n".join(textwrap.wrap(str(state), width=11, break_long_words=False)) or str(state)
        labels.append(f"{wrapped}\n{unit} = {int(counts.get(state, 0))}")
    return labels


def _settings_subtitle(settings, solo_only, full_windows_only, group_label=None):
    threshold = f"{settings.threshold:g}x" if settings.threshold_mode == "multiplier" else f"{settings.threshold:g}"
    parts = [f"window {settings.window} tp, threshold {threshold}"]
    parts.append("solo contacts" if solo_only else "all contacts")
    parts.append("full windows only" if full_windows_only else "all windows")
    lines = [" · ".join(parts), BOX_KEY]
    if group_label:
        lines.insert(0, group_label)
    if settings.needs_recompute:
        lines.append("Recomputed: " + " ".join(settings.differences))
    return "\n".join(textwrap.fill(line, width=110) for line in lines)


def _add_subtitle(fig, subtitle):
    """Settings line(s) at the top left, with only as much headroom as they need."""
    fig.text(0.01, 0.995, subtitle, ha="left", va="top", fontsize=6.5, color=MUTED)
    line_height = 0.125 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - line_height * (subtitle.count("\n") + 1) - 0.005))


def _draw_boxes(ax, values_per_state, colors, state_order):
    positions = np.arange(len(state_order))
    box = ax.boxplot(
        values_per_state, positions=positions, widths=0.62, patch_artist=True, showmeans=True,
        medianprops={"color": INK, "linewidth": 1.4},
        boxprops={"edgecolor": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.9},
        capprops={"color": INK, "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": "white",
                    "markeredgecolor": INK, "markeredgewidth": 0.4},
        meanprops={"marker": "D", "markersize": 4.5, "markerfacecolor": "white",
                   "markeredgecolor": "black", "markeredgewidth": 0.9},
    )
    for patch, state in zip(box["boxes"], state_order):
        patch.set_facecolor(colors[state])
    ax.set_xticks(positions)


def _fig_two_boxplots(cell_state, used_timepoints, state_order, colors, subtitle):
    width = max(7.6, 1.5 + 2 * 0.72 * len(state_order))
    # About 21 characters per inch at this size; keeps each description within its own panel.
    description_chars = int((width - 1.2) / 2 * 21)
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.2))
    panels = [
        (axes[0], cell_state, "mean_killing_efficiency", CELL_TITLE,
         textwrap.fill(CELL_DESCRIPTION, width=description_chars)),
        (axes[1], used_timepoints, "killing_efficiency", TIMEPOINT_TITLE,
         textwrap.fill(TIMEPOINT_DESCRIPTION, width=description_chars)),
    ]
    # The title sits above the longest description, in both panels so the titles stay level.
    description_lines = max(panel[-1].count("\n") + 1 for panel in panels)
    title_pad = 8 + 8.5 * description_lines
    for ax, table, column, title, description in panels:
        values = [table.loc[table["state"] == s, column].to_numpy() for s in state_order]
        _draw_boxes(ax, values, colors, state_order)
        counts = {s: len(v) for s, v in zip(state_order, values)}
        ax.set_xticklabels(_state_tick_labels(state_order, counts))
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color="black", pad=title_pad)
        ax.text(
            0.0, 1.02, description,
            transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom", ha="left",
        ).set_in_layout(False)
        ax.set_ylabel(Y_LABEL, fontsize=10)
        _style_axis(ax)
    _add_subtitle(fig, subtitle)
    return fig


def _no_data_page(message):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def save_state_killing_report(
    adata_states,
    df_table,
    out_dir,
    *,
    output_dir,
    cell_type,
    state_col,
    settings,
    states=None,
    solo_only=True,
    full_windows_only=True,
    extra_group_cols=None,
    video_ends=None,
    state_colors=None,
    verbose=False,
):
    """Write the Killing by State CSVs and PDF under
    `{out_dir}/killing_by_state/`, replacing the files of an earlier run.

    `df_table` is the per-timepoint features table holding the killing columns. `settings` comes
    from `killing_by_state.resolve_killing_settings`; when it differs from the extracted
    features, killing efficiency is recomputed for this report only. With `extra_group_cols`,
    the plot repeats per group in the same PDF and the CSVs gain those columns.

    Returns a dict of artifact paths.
    """
    group_cols = [str(c) for c in (extra_group_cols or [])]
    report_dir = Path(out_dir) / "killing_by_state"
    csv_dir = report_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "killing_by_state.pdf"
    paths = {
        "report_dir": str(report_dir),
        "pdf_path": str(pdf_path),
        "timepoints_csv": str(csv_dir / "killing_timepoints.csv"),
        "cell_state_csv": str(csv_dir / "killing_cell_state.csv"),
        "image_state_csv": str(csv_dir / "killing_image_state.csv"),
        "state_summary_csv": str(csv_dir / "killing_state_summary.csv"),
    }

    if video_ends is None:
        video_ends = load_video_ends(output_dir, settings.target_types)
    recomputed = recompute_killing_efficiency(output_dir, cell_type, settings) if settings.needs_recompute else None
    timepoints, observed_states = build_killing_timepoints(
        adata_states, df_table, state_col=state_col, settings=settings, video_ends=video_ends,
        states=states, solo_only=solo_only, full_windows_only=full_windows_only,
        group_cols=group_cols, recomputed=recomputed,
    )

    # The stored order runs bottom to top; shown left to right it reads like the stacked legends.
    wanted = [str(s) for s in states] if states else list(observed_states)
    state_order = list(reversed(_apply_state_order(
        sorted(wanted), _get_classification_state_order(adata_states, state_col)
    )))
    colors = dict(state_colors) if state_colors else _get_classification_state_colors(adata_states, state_col)
    colors = _normalize_label_color_map(state_order, colors=colors, cmap_name="tab20")

    cell_state, image_state, summary = summarize_killing_by_state(timepoints, state_order, group_cols=group_cols)
    timepoints.to_csv(paths["timepoints_csv"], index=False)
    cell_state.to_csv(paths["cell_state_csv"], index=False)
    image_state.to_csv(paths["image_state_csv"], index=False)
    summary.to_csv(paths["state_summary_csv"], index=False)
    counts = filter_counts(timepoints)

    combos = (
        timepoints[group_cols].drop_duplicates().sort_values(group_cols).to_dict("records")
        if group_cols else [{}]
    )

    def _select(df, combo):
        mask = pd.Series(True, index=df.index)
        for col, val in combo.items():
            mask &= df[col] == val
        return df[mask]

    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif"}):
        with PdfPages(pdf_path) as pdf:
            if counts["used"] == 0:
                fig = _no_data_page(
                    "No contact timepoints left after the filters. Try including more states, "
                    "or turning off solo contacts or full windows under Advanced Configuration."
                )
                pdf.savefig(fig)
                plt.close(fig)
            else:
                for combo in combos:
                    group_label = ", ".join(f"{c}={v}" for c, v in combo.items()) or None
                    subtitle = _settings_subtitle(settings, solo_only, full_windows_only, group_label)
                    group_cells = _select(cell_state, combo)
                    group_timepoints = _select(timepoints[timepoints["used"]], combo)
                    if group_timepoints.empty:
                        fig = _no_data_page(f"No contact timepoints left for {group_label}.")
                        pdf.savefig(fig)
                        plt.close(fig)
                        continue
                    fig = _fig_two_boxplots(group_cells, group_timepoints, state_order, colors, subtitle)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    if verbose:
        print(f"Saved Killing by State report ({counts['used']} timepoints used): {pdf_path}")
    paths["state_order"] = state_order
    paths["counts"] = counts
    paths["differences"] = list(settings.differences)
    return paths

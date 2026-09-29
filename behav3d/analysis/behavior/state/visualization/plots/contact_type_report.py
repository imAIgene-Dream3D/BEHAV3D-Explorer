"""Per-behavioral-state contact-type comparison across several organoid-contact columns.

State-native counterpart of
`behav3d.analysis.behavior.track.visualization.plots.contact_type_report` — lets a user pick
multiple per-organoid-type contact columns (e.g. `healthy_organoid_contact` vs.
`tumor_organoid_contact`) and, restricted to timepoints where the T cell **is** in organoid
contact, compare how the mix of organoid-contacting behavioral states differs between contact
types: a Welch's-t-test cluster-size-difference grid (same style as the state-composition
`condition_comparison` report) plus a stacked bar of state-composition proportions per
contact column.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from behav3d.analysis.behavior.utils import _contact_analysis_dir, _mixed_label_sort_key
from behav3d.analysis.behavior.state.utils import (
    _apply_state_order,
    _get_classification_state_colors,
    _get_classification_state_order,
    _normalize_label_color_map,
)
from behav3d.analysis.behavior.state.contact_grouping import compute_state_contact_type_cluster_proportions
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import (
    compute_condition_diff_stats_pairwise,
    plot_condition_diff_grid,
    draw_stacked_proportion_barv,
)


def _no_data_page(message):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def save_state_contact_type_comparison(
    adata_states,
    df_timepoints,
    out_dir,
    *,
    contact_cols,
    state_col,
    state_order=None,
    state_colors=None,
    groupby_cols=("sample_name", "TrackID"),
    verbose=False,
):
    """Per-`state_col` comparison of behavioral-state composition among timepoints where each
    of `contact_cols` (2+ organoid-contact columns picked by the user) is True — i.e. "for T
    cells with organoid contact, does the mix of on-organoid behaviors differ between organoid
    types".

    Reuses `contact_grouping.compute_state_contact_type_cluster_proportions` for the
    per-(track, contact column) state proportions, then the shared condition-comparison engine
    (`compute_condition_diff_stats_pairwise` / `plot_condition_diff_grid`) for a Welch's
    two-sided unpaired t-test cluster-size-difference grid — one row per pairwise contact-column
    comparison, generalizing to any number of selected contact columns — plus a stacked bar of
    mean state-composition proportions per contact column.

    Writes `contact_analysis/contact_type_comparison/<contact_cols joined by "_vs_">/contact_type_comparison.pdf`
    (one combined PDF: diff-bar page(s) then a stacked-composition page) and two sibling CSVs
    under `csv/`: `contact_type_diff_bars.csv` and `contact_type_stacked_composition.csv`. The
    subfolder is named after the sorted, `_vs_`-joined `contact_cols` (no special characters, so
    it stays filesystem-safe) so different column combinations don't overwrite each other.

    Returns a dict of artifact paths plus `state_order`/`contact_cols`.
    """
    contact_cols = [str(c) for c in list(contact_cols)]
    if len(contact_cols) < 2:
        raise ValueError(f"contact_cols must contain at least 2 columns to compare, got {contact_cols!r}.")

    per_unit_df, unit_metadata, observed_state_order = compute_state_contact_type_cluster_proportions(
        df_timepoints, adata_states, contact_cols=contact_cols, state_col=state_col,
        groupby_cols=groupby_cols, verbose=verbose,
    )

    combo_subfolder = "_vs_".join(sorted(contact_cols))
    out_dir = _contact_analysis_dir(out_dir, "contact_type_comparison", combo_subfolder)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_type_comparison.pdf"
    diff_csv_path = csv_dir / "contact_type_diff_bars.csv"
    stacked_csv_path = csv_dir / "contact_type_stacked_composition.csv"

    resolved_state_order = (
        [str(s) for s in state_order] if state_order is not None
        else sorted(observed_state_order, key=_mixed_label_sort_key)
    )
    resolved_state_order = _apply_state_order(resolved_state_order, _get_classification_state_order(adata_states, state_col))
    resolved_colors = dict(state_colors) if state_colors else _get_classification_state_colors(adata_states, state_col)
    resolved_colors = _normalize_label_color_map(resolved_state_order, colors=resolved_colors, cmap_name="tab20")

    if not resolved_state_order:
        with PdfPages(pdf_path) as pdf:
            fig = _no_data_page(f"No timepoints found where any of {contact_cols} is True — nothing to plot.")
            pdf.savefig(fig)
            plt.close(fig)
        pd.DataFrame(columns=["class", "mean_a", "mean_b", "diff", "group", "level_a", "level_b"]).to_csv(diff_csv_path, index=False)
        pd.DataFrame(columns=["contact_col", "state", "proportion"]).to_csv(stacked_csv_path, index=False)
        return {
            "contact_cols": contact_cols, "state_col": str(state_col),
            "pdf_path": str(pdf_path), "diff_csv_path": str(diff_csv_path),
            "stacked_csv_path": str(stacked_csv_path), "state_order": [],
        }

    # A caller-supplied state_order override may name a state absent from the contact-positive
    # subset (e.g. a state no track visited while touching this particular contact_cols
    # combination) — fill it as 0 rather than letting downstream lookups raise KeyError.
    per_unit_df = per_unit_df.reindex(columns=resolved_state_order, fill_value=0.0)

    diff_stats = compute_condition_diff_stats_pairwise(
        per_unit_df, unit_metadata, class_order=resolved_state_order, condition_col="contact_col",
    )

    mean_props_df = (
        per_unit_df.join(unit_metadata, how="inner")
        .groupby("contact_col", observed=True)[resolved_state_order]
        .mean()
        .reindex(contact_cols)
        .fillna(0.0)
    )

    with PdfPages(pdf_path) as pdf:
        plot_condition_diff_grid(
            diff_stats,
            class_order=resolved_state_order,
            colors=resolved_colors,
            title=f"Contact type comparison by {state_col} — cluster-size difference",
            out_pdf=pdf_path,
            out_csv=diff_csv_path,
            pdf_pages=pdf,
        )

        fig, ax = plt.subplots(figsize=(max(1.2 + 0.9 * len(contact_cols), 4.0), 5.0))
        draw_stacked_proportion_barv(ax, mean_props_df, contact_cols, resolved_state_order, resolved_colors)
        ax.set_ylabel("Mean proportion of contacting timepoints", fontsize=8)
        legend_state_order = list(reversed(resolved_state_order))
        handles = [plt.Rectangle((0, 0), 1, 1, color=resolved_colors[s]) for s in legend_state_order]
        fig.legend(handles, legend_state_order, loc="lower center", ncol=min(len(resolved_state_order), 6), frameon=False, fontsize=8)
        fig.suptitle(f"Contact type comparison by {state_col} — state composition", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.94))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    stacked_rows = [
        {"contact_col": c, "state": s, "proportion": float(mean_props_df.loc[c, s])}
        for c in contact_cols for s in resolved_state_order
    ]
    pd.DataFrame(stacked_rows).to_csv(stacked_csv_path, index=False)

    if verbose:
        print(
            f"Saved state contact type comparison ({len(contact_cols)} column(s), "
            f"{len(resolved_state_order)} state(s)): {pdf_path}"
        )

    return {
        "contact_cols": contact_cols,
        "state_col": str(state_col),
        "pdf_path": str(pdf_path),
        "diff_csv_path": str(diff_csv_path),
        "stacked_csv_path": str(stacked_csv_path),
        "state_order": resolved_state_order,
    }

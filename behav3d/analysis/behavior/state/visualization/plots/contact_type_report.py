"""Per-behavioral-state contact-fraction comparison across several organoid-contact columns.

State-native counterpart of
`behav3d.analysis.behavior.track.visualization.plots.contact_type_report` — lets a user pick
multiple per-organoid-type contact columns (e.g. `healthy_organoid_contact` vs.
`tumor_organoid_contact`) and compare, per behavioral state, the mean contact fraction for
each selected type side by side. Purely descriptive (mean +/- SEM, no significance test), the
same convention used by `proportion_bars.plot_condition_time_series_grid`.
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
from behav3d.analysis.behavior.state.contact_grouping import compute_state_contact_type_values
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import (
    compute_class_by_group_mean_sem,
    draw_grouped_value_barh,
    hash_stable_label_color_map,
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
    """Per-`state_col` comparison of mean contact fraction across `contact_cols` (2+
    organoid-contact columns picked by the user).

    Reuses `contact_grouping.compute_state_contact_type_values` for the per-(track, state,
    contact column) values, then aggregates to mean/SEM per (state, contact column) — no
    significance test, no p-values.

    Writes `contact_analysis/contact_type_comparison/<contact_cols joined by "+">/contact_type_comparison.pdf`
    (one grouped bar chart page) and a sibling `csv/contact_type_comparison.csv` (the long-form
    per-track values, for the user's own statistics if wanted). The subfolder is named after the
    sorted, `+`-joined `contact_cols` so different column combinations don't overwrite each other.

    Returns a dict of artifact paths plus `state_order`/`contact_cols`.
    """
    contact_cols = [str(c) for c in list(contact_cols)]
    if len(contact_cols) < 2:
        raise ValueError(f"contact_cols must contain at least 2 columns to compare, got {contact_cols!r}.")

    long_df = compute_state_contact_type_values(
        df_timepoints, adata_states, contact_cols=contact_cols, state_col=state_col,
        groupby_cols=groupby_cols, verbose=verbose,
    )
    long_df["mean_fraction_pct"] = pd.to_numeric(long_df["mean_fraction"], errors="coerce") * 100.0

    combo_subfolder = "+".join(sorted(contact_cols))
    out_dir = _contact_analysis_dir(out_dir, "contact_type_comparison", combo_subfolder)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_type_comparison.pdf"
    csv_path = csv_dir / "contact_type_comparison.csv"

    resolved_state_order = (
        [str(s) for s in state_order] if state_order is not None
        else sorted(long_df[state_col].dropna().astype(str).unique().tolist(), key=_mixed_label_sort_key)
    )
    resolved_state_order = _apply_state_order(resolved_state_order, _get_classification_state_order(adata_states, state_col))
    resolved_colors = dict(state_colors) if state_colors else _get_classification_state_colors(adata_states, state_col)
    resolved_colors = _normalize_label_color_map(resolved_state_order, colors=resolved_colors, cmap_name="tab20")

    if not resolved_state_order:
        with PdfPages(pdf_path) as pdf:
            fig = _no_data_page(f"No '{state_col}' states found — nothing to plot.")
            pdf.savefig(fig)
            plt.close(fig)
        pd.DataFrame(columns=[state_col, "contact_col", "mean_fraction"]).to_csv(csv_path, index=False)
        return {
            "contact_cols": contact_cols, "state_col": str(state_col),
            "pdf_path": str(pdf_path), "csv_path": str(csv_path), "state_order": [],
        }

    plot_df = long_df.copy()
    plot_df[state_col] = plot_df[state_col].astype(str)
    plot_df = plot_df[plot_df[state_col].isin(resolved_state_order)]

    mean_df, sem_df, n_df = compute_class_by_group_mean_sem(
        plot_df, class_col=state_col, group_col="contact_col", value_col="mean_fraction_pct",
        class_order=resolved_state_order, group_order=contact_cols,
    )
    contact_col_colors = hash_stable_label_color_map(contact_cols)

    fig, ax = plt.subplots(figsize=(7.0, max(1.0 + 0.4 * len(resolved_state_order), 3.0)))
    draw_grouped_value_barh(
        ax, resolved_state_order, mean_df, sem_df, contact_cols, contact_col_colors,
        xlabel="Mean fraction of time in contact (%)",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=contact_col_colors[c]) for c in contact_cols
    ]
    fig.legend(handles, contact_cols, loc="lower center", ncol=min(len(contact_cols), 4), frameon=False, fontsize=8)
    fig.suptitle(f"Contact type comparison by {state_col}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    long_df.to_csv(csv_path, index=False)

    if verbose:
        print(
            f"Saved state contact type comparison ({len(contact_cols)} column(s), "
            f"{len(resolved_state_order)} state(s)): {pdf_path}"
        )

    return {
        "contact_cols": contact_cols,
        "state_col": str(state_col),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "state_order": resolved_state_order,
    }

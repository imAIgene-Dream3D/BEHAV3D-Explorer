"""Per-cluster contact-fraction comparison across several organoid-contact columns.

Lets a user pick multiple per-organoid-type contact columns (e.g. `healthy_organoid_contact`
vs. `tumor_organoid_contact`, both aggregated into `any_organoid_contact` by
`behav3d.features.timepoint_features`) and compare, per behavioral cluster, the mean contact
fraction for each selected type side by side — purely descriptive (mean +/- SEM, no
significance test), matching the "descriptive only" convention already used by
`proportion_bars.plot_condition_time_series_grid`.
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
from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    merge_track_contact_features_into_obs,
    _contact_mean_col_name,
)
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


def save_track_contact_type_comparison(
    adata_tracks,
    df_timepoints,
    out_dir,
    *,
    contact_cols,
    min_bout_length,
    class_col="ClusterID",
    class_order=None,
    class_colors=None,
    groupby_cols=("sample_name", "TrackID"),
    verbose=False,
):
    """Per-`class_col` (default the track's own behavioral cluster) comparison of mean contact
    fraction across `contact_cols` (2+ organoid-contact columns picked by the user).

    Reuses `contact_grouping.compute_track_contact_features` /
    `merge_track_contact_features_into_obs` once per contact column (unchanged; same as
    `contact_cluster_heatmap.save_track_contact_cluster_heatmap`), then melts the resulting
    per-track `{contact_col}_mean_fraction` columns to long form and aggregates to mean/SEM per
    (cluster, contact column) — no significance test, no p-values.

    Writes `contact_analysis/contact_type_comparison/<contact_cols joined by "+">/contact_type_comparison.pdf`
    (one grouped bar chart page) and a sibling `csv/contact_type_comparison.csv` (the long-form
    per-track values, for the user's own statistics if wanted). The subfolder is named after the
    sorted, `+`-joined `contact_cols` so different column combinations don't overwrite each other.

    Returns a dict of artifact paths plus `class_order`/`contact_cols`.
    """
    contact_cols = [str(c) for c in list(contact_cols)]
    if len(contact_cols) < 2:
        raise ValueError(f"contact_cols must contain at least 2 columns to compare, got {contact_cols!r}.")
    groupby_cols = [str(c) for c in list(groupby_cols)]

    mean_cols = []
    for contact_col in contact_cols:
        contact_features = compute_track_contact_features(
            df_timepoints, adata_tracks, contact_col=contact_col, min_bout_length=min_bout_length,
            groupby_cols=groupby_cols, verbose=verbose,
        )
        merge_track_contact_features_into_obs(
            adata_tracks, contact_features, contact_col=contact_col, min_bout_length=min_bout_length,
            groupby_cols=groupby_cols,
        )
        mean_cols.append(_contact_mean_col_name(contact_col))

    obs = adata_tracks.obs
    if class_col not in obs.columns:
        raise KeyError(f"class_col={class_col!r} not found in adata_tracks.obs.")

    combo_subfolder = "+".join(sorted(contact_cols))
    out_dir = _contact_analysis_dir(out_dir, "contact_type_comparison", combo_subfolder)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_type_comparison.pdf"
    csv_path = csv_dir / "contact_type_comparison.csv"

    resolved_class_order = (
        [str(c) for c in class_order] if class_order is not None
        else sorted(obs[class_col].dropna().astype(str).unique().tolist(), key=_mixed_label_sort_key)
    )
    resolved_class_order = _apply_state_order(resolved_class_order, _get_classification_state_order(adata_tracks, class_col))
    resolved_colors = dict(class_colors) if class_colors else _get_classification_state_colors(adata_tracks, class_col)
    resolved_colors = _normalize_label_color_map(resolved_class_order, colors=resolved_colors, cmap_name="tab20")

    if not resolved_class_order:
        with PdfPages(pdf_path) as pdf:
            fig = _no_data_page(f"No '{class_col}' clusters found — nothing to plot.")
            pdf.savefig(fig)
            plt.close(fig)
        pd.DataFrame(columns=[class_col, "contact_col", "mean_fraction"]).to_csv(csv_path, index=False)
        return {
            "contact_cols": contact_cols, "min_bout_length": int(min_bout_length),
            "class_col": str(class_col), "pdf_path": str(pdf_path), "csv_path": str(csv_path),
            "class_order": [],
        }

    sub = obs[[class_col] + mean_cols].copy()
    sub[class_col] = sub[class_col].astype(str)
    sub = sub[sub[class_col].isin(resolved_class_order)]
    long_df = sub.melt(id_vars=[class_col], value_vars=mean_cols, var_name="_mean_col", value_name="mean_fraction")
    mean_col_to_contact_col = dict(zip(mean_cols, contact_cols))
    long_df["contact_col"] = long_df["_mean_col"].map(mean_col_to_contact_col)
    long_df = long_df.drop(columns=["_mean_col"])
    long_df["mean_fraction_pct"] = pd.to_numeric(long_df["mean_fraction"], errors="coerce") * 100.0

    mean_df, sem_df, n_df = compute_class_by_group_mean_sem(
        long_df, class_col=class_col, group_col="contact_col", value_col="mean_fraction_pct",
        class_order=resolved_class_order, group_order=contact_cols,
    )
    contact_col_colors = hash_stable_label_color_map(contact_cols)

    fig, ax = plt.subplots(figsize=(7.0, max(1.0 + 0.4 * len(resolved_class_order), 3.0)))
    draw_grouped_value_barh(
        ax, resolved_class_order, mean_df, sem_df, contact_cols, contact_col_colors,
        xlabel="Mean fraction of time in contact (%)",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=contact_col_colors[c]) for c in contact_cols
    ]
    fig.legend(handles, contact_cols, loc="lower center", ncol=min(len(contact_cols), 4), frameon=False, fontsize=8)
    fig.suptitle(f"Contact type comparison by {class_col}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    long_df[[class_col, "contact_col", "mean_fraction", "mean_fraction_pct"]].to_csv(csv_path, index=False)

    if verbose:
        print(f"Saved contact type comparison ({len(contact_cols)} column(s), {len(resolved_class_order)} cluster(s)): {pdf_path}")

    return {
        "contact_cols": contact_cols,
        "min_bout_length": int(min_bout_length),
        "class_col": str(class_col),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "class_order": resolved_class_order,
    }

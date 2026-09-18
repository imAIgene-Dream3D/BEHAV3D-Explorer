"""Per-behavioral-cluster contact-amount overview.

Complements `contact_duration_report.py` (duration split by the *touched* cell's class) and
`reports.py`'s `save_track_contact_group_analysis` (which already box-plots contact fraction/
duration by the track's own cluster) with a heatmap-first view of the same two per-track contact
metrics — crossed with the track's own behavioral cluster rather than the touched cell's class —
plus a heatmap of what fraction of tracks in each cluster made contact at all.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from behav3d.analysis.behavior.utils import _mixed_label_sort_key
from behav3d.analysis.behavior.state.utils import (
    _apply_state_order,
    _get_classification_state_colors,
    _get_classification_state_order,
    _normalize_label_color_map,
)
from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    merge_track_contact_features_into_obs,
    _contact_group_col_name,
    _contact_mean_col_name,
    _contact_max_bout_col_name,
)

A4_LANDSCAPE = (11.69, 8.27)


def _minmax_scale_row(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, 1e-9)
    return (values - lo) / span


def _draw_single_cluster_heatmap(ax, cluster_order, raw_values, *, row_label, cmap="viridis", vmin=None, vmax=None):
    df_raw = pd.DataFrame(
        np.asarray(raw_values, dtype=float).reshape(1, -1), index=[row_label], columns=cluster_order,
    )
    sns.heatmap(df_raw, ax=ax, cmap=cmap, cbar=True, annot=True, fmt=".1f", vmin=vmin, vmax=vmax, yticklabels=True)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8, rotation=0)


def _draw_raw_and_scaled_heatmaps(ax_raw, ax_scaled, cluster_order, raw_values, *, row_label, cmap="viridis"):
    """Left panel: heatmap colored by the raw values. Right panel: the same values colored by a
    0-1 min-max scaling across clusters (for visual contrast when the raw spread is small) — both
    panels are annotated with the raw numbers so the actual magnitudes stay legible."""
    raw_arr = np.asarray(raw_values, dtype=float)
    df_raw = pd.DataFrame(raw_arr.reshape(1, -1), index=[row_label], columns=cluster_order)
    df_scaled = pd.DataFrame(_minmax_scale_row(raw_arr).reshape(1, -1), index=[row_label], columns=cluster_order)

    sns.heatmap(df_raw, ax=ax_raw, cmap=cmap, cbar=True, annot=True, fmt=".2f", yticklabels=True)
    ax_raw.set_title("Raw values", fontsize=9)
    ax_raw.tick_params(axis="x", labelsize=7, rotation=0)
    ax_raw.tick_params(axis="y", labelsize=8, rotation=0)

    sns.heatmap(
        df_scaled, ax=ax_scaled, cmap=cmap, cbar=True, annot=df_raw.to_numpy(), fmt=".2f",
        vmin=0.0, vmax=1.0, yticklabels=False,
    )
    ax_scaled.set_title("Min-max scaled (0-1)", fontsize=9)
    ax_scaled.tick_params(axis="x", labelsize=7, rotation=0)


def _draw_track_violin(ax, df_long, *, cluster_order, value_col, class_col, ylabel):
    """Violin + jittered per-track dots + mean marker of `value_col` split by `class_col` — each
    point is one track, including the many zero/no-contact tracks."""
    sub = df_long[[class_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])
    if len(sub) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    sns.violinplot(data=sub, x=class_col, y=value_col, order=cluster_order, inner=None, cut=0, ax=ax)
    sns.stripplot(
        data=sub, x=class_col, y=value_col, order=cluster_order, ax=ax,
        color="black", dodge=False, jitter=0.2, alpha=0.4, size=4,
    )
    means = sub.groupby(class_col, observed=True)[value_col].mean().reindex(cluster_order)
    ax.scatter(
        np.arange(len(cluster_order)), means.to_numpy(),
        s=60, color="white", edgecolor="black", linewidths=1.0, zorder=3,
    )
    ax.set_xlabel(str(class_col), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _plot_metric_page(cluster_order, cluster_means, per_track_df, *, class_col, value_col, row_label, ylabel, page_title):
    fig = plt.figure(figsize=A4_LANDSCAPE)
    gs = fig.add_gridspec(
        nrows=2, ncols=2, height_ratios=[1, 3], hspace=0.6, wspace=0.25,
        top=0.88, bottom=0.10, left=0.10, right=0.95,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_scaled = fig.add_subplot(gs[0, 1])
    _draw_raw_and_scaled_heatmaps(ax_raw, ax_scaled, cluster_order, cluster_means, row_label=row_label)
    ax_violin = fig.add_subplot(gs[1, :])
    _draw_track_violin(ax_violin, per_track_df, cluster_order=cluster_order, value_col=value_col, class_col=class_col, ylabel=ylabel)
    fig.suptitle(page_title, fontsize=11, fontweight="bold")
    return fig


def _plot_percentage_page(cluster_order, pct_values, *, class_col, page_title):
    fig, ax = plt.subplots(figsize=(max(1.2 * len(cluster_order), 6.0), 2.8))
    _draw_single_cluster_heatmap(
        ax, cluster_order, pct_values, row_label="% tracks with contact", vmin=0.0, vmax=100.0,
    )
    ax.set_xlabel(str(class_col), fontsize=9)
    fig.suptitle(page_title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig


def _no_clusters_page(class_col):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5, 0.5, f"No '{class_col}' clusters found — nothing to plot.",
        ha="center", va="center", wrap=True,
    )
    ax.axis("off")
    return fig


def save_track_contact_cluster_heatmap(
    adata_tracks,
    df_timepoints,
    out_dir,
    *,
    contact_col,
    min_bout_length,
    class_col="ClusterID",
    class_order=None,
    class_colors=None,
    minutes_per_frame=None,
    sample_col="sample_name",
    groupby_cols=("sample_name", "TrackID"),
    verbose=False,
):
    """Per-`class_col` (default the track's own behavioral cluster) overview of how much contact
    tracks in that cluster have with `contact_col`'s target population.

    Writes one combined PDF (`contact_cluster_heatmap.pdf`) with:

    1. A page for "max contact-bout length" (longest contiguous contact bout per track, in
       minutes when `minutes_per_frame` is given, else timepoints) — raw-value and min-max
       scaled heatmaps of the per-cluster mean, with a violin + jittered per-track dots +
       mean marker of the underlying per-track distribution beneath them.
    2. The same layout for "mean fraction of time in contact" (expressed as a percentage of each
       track's classified window).
    3. A single heatmap of the percentage of tracks in each cluster that had any qualifying
       contact bout at all (>= `min_bout_length`).

    Also writes `csv/contact_cluster_summary.csv`, one row per cluster.

    Reuses `contact_grouping.compute_track_contact_features` (merged onto `adata_tracks.obs` via
    `merge_track_contact_features_into_obs`, same as `reports.save_track_contact_group_analysis`
    and `contact_duration_report.save_track_contact_duration_comparison`) for the per-track
    metrics, and the same cluster order/color resolution convention used throughout the other
    contact/cluster reports.

    Returns a dict of artifact paths plus `class_order`.
    """
    groupby_cols = [str(c) for c in list(groupby_cols)]
    mean_col = _contact_mean_col_name(contact_col)
    max_bout_col = _contact_max_bout_col_name(contact_col)
    group_col = _contact_group_col_name(contact_col)

    contact_features = compute_track_contact_features(
        df_timepoints, adata_tracks, contact_col=contact_col, min_bout_length=min_bout_length,
        groupby_cols=groupby_cols, verbose=verbose,
    )
    merge_track_contact_features_into_obs(
        adata_tracks, contact_features, contact_col=contact_col, min_bout_length=min_bout_length,
        groupby_cols=groupby_cols,
    )

    obs = adata_tracks.obs
    if class_col not in obs.columns:
        raise KeyError(f"class_col={class_col!r} not found in adata_tracks.obs.")

    resolved_class_order = (
        [str(c) for c in class_order] if class_order is not None
        else sorted(obs[class_col].dropna().astype(str).unique().tolist(), key=_mixed_label_sort_key)
    )
    resolved_class_order = _apply_state_order(resolved_class_order, _get_classification_state_order(adata_tracks, class_col))
    resolved_colors = dict(class_colors) if class_colors else _get_classification_state_colors(adata_tracks, class_col)
    resolved_colors = _normalize_label_color_map(resolved_class_order, colors=resolved_colors, cmap_name="tab20")

    out_dir = Path(out_dir) / "contact_analysis" / str(contact_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_cluster_heatmap.pdf"
    csv_path = csv_dir / "contact_cluster_summary.csv"

    if not resolved_class_order:
        with PdfPages(pdf_path) as pdf:
            fig = _no_clusters_page(class_col)
            pdf.savefig(fig)
            plt.close(fig)
        pd.DataFrame(columns=[class_col, "n_tracks", "n_contact", "pct_contact"]).to_csv(csv_path, index=False)
        return {
            "contact_col": str(contact_col), "min_bout_length": int(min_bout_length),
            "class_col": str(class_col), "pdf_path": str(pdf_path), "csv_path": str(csv_path),
            "class_order": [], "minutes_per_frame": minutes_per_frame,
        }

    sub = obs[[class_col, mean_col, max_bout_col, group_col]].copy()
    sub[class_col] = sub[class_col].astype(str)
    sub = sub[sub[class_col].isin(resolved_class_order)]

    show_minutes = minutes_per_frame is not None
    duration_col = "_duration_plot_value"
    sub[duration_col] = sub[max_bout_col] * (float(minutes_per_frame) if show_minutes else 1.0)
    duration_unit = "minutes" if show_minutes else "timepoints"

    fraction_col = "_fraction_plot_value"
    sub[fraction_col] = sub[mean_col] * 100.0

    grouped = sub.groupby(class_col, observed=True)
    cluster_means_duration = grouped[duration_col].mean().reindex(resolved_class_order).fillna(0.0)
    cluster_means_fraction = grouped[fraction_col].mean().reindex(resolved_class_order).fillna(0.0)
    cluster_pct_contact = (
        grouped[group_col].apply(lambda s: 100.0 * float((s == "contact").mean()))
        .reindex(resolved_class_order).fillna(0.0)
    )
    n_tracks = grouped.size().reindex(resolved_class_order).fillna(0).astype(int)
    n_contact = (
        grouped[group_col].apply(lambda s: int((s == "contact").sum()))
        .reindex(resolved_class_order).fillna(0).astype(int)
    )
    mean_max_bout_timepoints = grouped[max_bout_col].mean().reindex(resolved_class_order)

    with PdfPages(pdf_path) as pdf:
        fig = _plot_metric_page(
            resolved_class_order, cluster_means_duration.to_numpy(), sub,
            class_col=class_col, value_col=duration_col,
            row_label=f"Contact duration ({duration_unit})",
            ylabel=f"Max contact-bout length ({duration_unit})",
            page_title=f"Contact duration by {class_col} — contact_col={contact_col}",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_metric_page(
            resolved_class_order, cluster_means_fraction.to_numpy(), sub,
            class_col=class_col, value_col=fraction_col,
            row_label="Mean time in contact (%)",
            ylabel="Mean fraction of time in contact (%)",
            page_title=f"Time in contact by {class_col} — contact_col={contact_col}",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_percentage_page(
            resolved_class_order, cluster_pct_contact.to_numpy(), class_col=class_col,
            page_title=f"% of tracks with contact by {class_col} — contact_col={contact_col}",
        )
        pdf.savefig(fig)
        plt.close(fig)

    csv_columns = [class_col, "n_tracks", "n_contact", "pct_contact", "mean_max_bout_timepoints"]
    if show_minutes:
        csv_columns.append("mean_max_bout_minutes")
    csv_columns.append("mean_fraction_pct")

    csv_rows = []
    for cls in resolved_class_order:
        row = {
            class_col: cls,
            "n_tracks": int(n_tracks.get(cls, 0)),
            "n_contact": int(n_contact.get(cls, 0)),
            "pct_contact": float(cluster_pct_contact.get(cls, 0.0)),
            "mean_max_bout_timepoints": float(mean_max_bout_timepoints.get(cls, float("nan"))),
            "mean_fraction_pct": float(cluster_means_fraction.get(cls, 0.0)),
        }
        if show_minutes:
            row["mean_max_bout_minutes"] = float(cluster_means_duration.get(cls, 0.0))
        csv_rows.append(row)
    pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_path, index=False)

    if verbose:
        print(f"Saved contact cluster heatmap ({len(resolved_class_order)} cluster(s)): {pdf_path}")

    return {
        "contact_col": str(contact_col),
        "min_bout_length": int(min_bout_length),
        "class_col": str(class_col),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "class_order": resolved_class_order,
        "minutes_per_frame": minutes_per_frame,
    }

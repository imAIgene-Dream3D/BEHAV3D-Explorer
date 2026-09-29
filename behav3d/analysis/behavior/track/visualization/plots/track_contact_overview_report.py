from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from behav3d.features.state_descriptive_features import rle_encode
from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.track.contact_grouping import (
    compute_track_contact_features,
    _contact_group_col_name,
)
from behav3d.analysis.behavior.utils import _contact_analysis_dir
from behav3d.analysis.behavior.track.visualization.plots.exemplar_track_per_cluster import (
    _build_state_color_map,
    _extract_track_window_obs,
    _compute_state_bar_segments,
    _plot_statebar_segments_on_ax,
)
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import _chunk_list

_TRACK_OVERVIEW_WINDOW_COL = "trajectory_window_id"
_CONTACT_BAR_COLOR = "#2ca02c"
_NO_CONTACT_BAR_COLOR = "#bbbbbb"


def _compute_contact_bar_segments(times, is_contact, min_bout_length):
    """RLE-encode a track's per-timepoint contact boolean series into colored bar segments,
    time-aligned with `_compute_state_bar_segments` (same width convention). A run is green only
    if it's truthy AND long enough to meet `min_bout_length` -- the same threshold
    `contact_grouping.compute_track_contact_features` uses for "contact" grouping -- so a
    too-short contact blip renders grey, same as no contact at all.
    """
    times = np.asarray(times, dtype=float)
    is_contact = np.asarray(is_contact, dtype=bool)
    if len(times) == 0:
        raise ValueError("Cannot render contact bar: no timepoints found.")

    dx = np.diff(times)
    default_w = np.median(dx[dx > 0]) if np.any(dx > 0) else 1.0
    widths = np.r_[dx, default_w]

    min_bout_length = int(min_bout_length)
    segments = []
    pos = 0
    for value, length in rle_encode(is_contact.tolist()):
        start = float(times[pos])
        end = float(times[pos + length - 1] + widths[pos + length - 1])
        color = _CONTACT_BAR_COLOR if (bool(value) and length >= min_bout_length) else _NO_CONTACT_BAR_COLOR
        segments.append((start, max(0.0, end - start), color))
        pos += length
    return segments


def _plot_track_contact_overview_page(page_rows, *, sample_name, contact_col, min_bout_length, group_label=None):
    """One PDF page: each track gets its state-over-time bar (top) with the matching contact bar
    (bottom, grey/green) directly underneath. All rows share one page-wide time axis (rather than
    each track being independently stretched to the same width), so a track's bar width in the
    page reflects its actual duration relative to the other tracks on that page.
    """
    n = len(page_rows)
    page_xlim = (
        min(r["xlim"][0] for r in page_rows),
        max(r["xlim"][1] for r in page_rows),
    )
    fig = plt.figure(figsize=(11.0, max(2.4, 1.7 * n)))
    outer = fig.add_gridspec(nrows=n, ncols=1, hspace=0.75, top=0.90, bottom=0.10)

    for i, row_data in enumerate(page_rows):
        inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax_state = fig.add_subplot(inner[0])
        ax_contact = fig.add_subplot(inner[1], sharex=ax_state)

        _plot_statebar_segments_on_ax(
            ax_state,
            segments=row_data["state_segments"],
            xlim=page_xlim,
            title=f"{row_data['sample_name']} | Track {row_data['track_id']}",
        )
        ax_state.set_xlabel("")
        ax_state.tick_params(labelbottom=False)

        for start, width, color in row_data["contact_segments"]:
            ax_contact.broken_barh([(start, width)], (0.0, 1.0), facecolors=color)
        ax_contact.set_xlim(*page_xlim)
        ax_contact.set_ylim(0.0, 1.0)
        ax_contact.set_yticks([])
        ax_contact.set_xlabel("position_t", fontsize=8)

    legend_handles = [
        Patch(facecolor=_CONTACT_BAR_COLOR, edgecolor="k", label=f"contact (bout >= {min_bout_length})"),
        Patch(facecolor=_NO_CONTACT_BAR_COLOR, edgecolor="k", label="no contact"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    title = f"Track contact overview | sample: {sample_name} | contact_col={contact_col}"
    if group_label:
        title += f" | {group_label}"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig


def save_track_contact_overview_report(
    adata_tracks,
    df_timepoints,
    adata_states,
    out_dir,
    *,
    contact_col,
    min_bout_length,
    state_col=FULL_STATE_COL,
    sample_col="sample_name",
    track_col="TrackID",
    extra_group_cols=None,
    rows_per_page=6,
    state_order=None,
    state_colors=None,
    plot_dpi=200,
    verbose=False,
):
    """For every track whose contact meets `min_bout_length` (the same threshold used by
    `contact_grouping.compute_track_contact_features` for "contact" vs. "no_contact" grouping),
    plot its full (untrimmed, classified-window) behavioral-state trajectory as a colored bar,
    with a grey/green bar directly beneath it marking every `contact_col` bout of at least
    `min_bout_length` timepoints. Pages are grouped by sample -- a sample's tracks are never
    split across a page shared with the next sample's, even if that leaves the page under-full.

    If `extra_group_cols` is given (condition columns already present in `adata_tracks.obs`,
    constant per sample), tracks are also never split across a page shared with a different
    combination of those columns' values -- pages are grouped by (group combination, sample)
    instead of just sample, with each page's title annotated with its group combination.

    Writes into `{out_dir}/contact_analysis/contact_track_overview/{contact_col}/` as
    `track_contact_overview.pdf`.

    Returns a dict with `pdf_path`, `n_tracks`, `n_samples`.
    """
    sample_col = str(sample_col)
    track_col = str(track_col)
    extra_group_cols = [str(c) for c in extra_group_cols] if extra_group_cols else []
    groupby_cols = [sample_col, track_col]
    time_col = "position_t"

    if state_order is None or state_colors is None:
        resolved_order, resolved_colors = _build_state_color_map(adata_states, state_col)
        state_order = state_order if state_order is not None else resolved_order
        state_colors = state_colors if state_colors is not None else resolved_colors

    out_dir = _contact_analysis_dir(out_dir, "contact_track_overview", contact_col)
    pdf_path = out_dir / "track_contact_overview.pdf"

    contact_features = compute_track_contact_features(
        df_timepoints, adata_tracks,
        contact_col=contact_col, min_bout_length=min_bout_length,
        groupby_cols=groupby_cols, verbose=verbose,
    )
    group_col = _contact_group_col_name(contact_col)
    contact_tracks = contact_features[contact_features[group_col] == "contact"].reset_index()

    has_window_col = (
        _TRACK_OVERVIEW_WINDOW_COL in adata_tracks.obs.columns
        and _TRACK_OVERVIEW_WINDOW_COL in contact_tracks.columns
    )
    key_cols = groupby_cols + ([_TRACK_OVERVIEW_WINDOW_COL] if has_window_col else [])

    missing_group_cols = [c for c in extra_group_cols if c not in adata_tracks.obs.columns]
    if missing_group_cols:
        raise KeyError(f"Missing group-by columns in adata_tracks.obs: {missing_group_cols}")
    window_cols = key_cols + ["position_t_min", "position_t_max"] + [c for c in extra_group_cols if c not in key_cols]

    windows = (
        adata_tracks.obs[window_cols]
        .drop_duplicates(subset=key_cols)
        .copy()
    )
    for col in key_cols:
        contact_tracks[col] = contact_tracks[col].astype(str)
        windows[col] = windows[col].astype(str)
    tracks_df = contact_tracks.merge(windows, on=key_cols, how="left")

    if len(tracks_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"No tracks met the contact threshold (min_bout_length={min_bout_length}) for '{contact_col}'.",
            ha="center", va="center", fontsize=10,
        )
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=int(plot_dpi), bbox_inches="tight")
        plt.close(fig)
        return {"pdf_path": str(pdf_path), "n_tracks": 0, "n_samples": 0}

    prepared_rows = []
    sample_order = []
    for _, row in tracks_df.iterrows():
        sample_name = row[sample_col]
        track_id = row[track_col]
        tmin = row["position_t_min"]
        tmax = row["position_t_max"]
        if sample_name not in sample_order:
            sample_order.append(sample_name)

        state_track_df = _extract_track_window_obs(
            adata_states,
            sample_name=sample_name,
            track_id=track_id,
            tmin=tmin,
            tmax=tmax,
            sample_key=sample_col,
            track_key=track_col,
            time_key=time_col,
            extra_cols=[state_col],
        )
        state_segments, xlim = _compute_state_bar_segments(
            state_track_df, state_key=state_col, time_key=time_col, state_color_map=state_colors,
        )

        contact_track_df = df_timepoints[
            (df_timepoints[sample_col].astype(str) == str(sample_name))
            & (df_timepoints[track_col].astype(str) == str(track_id))
        ].copy()
        contact_track_df[time_col] = pd.to_numeric(contact_track_df[time_col], errors="coerce")
        contact_track_df = contact_track_df[
            (contact_track_df[time_col] >= float(tmin)) & (contact_track_df[time_col] <= float(tmax))
        ].sort_values(time_col)
        contact_segments = _compute_contact_bar_segments(
            contact_track_df[time_col].to_numpy(dtype=float),
            pd.to_numeric(contact_track_df[contact_col], errors="coerce").fillna(0).astype(bool).to_numpy(),
            min_bout_length,
        )

        group_key = tuple(row[c] for c in extra_group_cols) if extra_group_cols else ()

        prepared_rows.append(
            {
                "sample_name": sample_name,
                "track_id": track_id,
                "state_segments": state_segments,
                "xlim": xlim,
                "contact_segments": contact_segments,
                "group_key": group_key,
            }
        )

    if extra_group_cols:
        seen_combos = []
        for r in prepared_rows:
            if r["group_key"] not in seen_combos:
                seen_combos.append(r["group_key"])
        group_combos = sorted(seen_combos)
    else:
        group_combos = [()]

    rows_per_page = max(1, int(rows_per_page))
    with PdfPages(pdf_path) as pdf:
        for combo in group_combos:
            combo_rows = (
                [r for r in prepared_rows if r["group_key"] == combo] if extra_group_cols else prepared_rows
            )
            group_label = (
                ", ".join(f"{c}={v}" for c, v in zip(extra_group_cols, combo)) if extra_group_cols else None
            )
            combo_sample_order = []
            for r in combo_rows:
                if r["sample_name"] not in combo_sample_order:
                    combo_sample_order.append(r["sample_name"])
            for sample_name in combo_sample_order:
                sample_rows = [r for r in combo_rows if r["sample_name"] == sample_name]
                for page_rows in _chunk_list(sample_rows, rows_per_page):
                    fig = _plot_track_contact_overview_page(
                        page_rows, sample_name=sample_name, contact_col=contact_col,
                        min_bout_length=min_bout_length, group_label=group_label,
                    )
                    pdf.savefig(fig, dpi=int(plot_dpi), bbox_inches="tight")
                    plt.close(fig)

    if verbose:
        print(f"Saved track contact overview report: {pdf_path}")

    return {
        "contact_col": str(contact_col),
        "min_bout_length": int(min_bout_length),
        "pdf_path": str(pdf_path),
        "n_tracks": int(len(prepared_rows)),
        "n_samples": int(len(sample_order)),
    }

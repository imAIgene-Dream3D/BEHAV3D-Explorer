from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from behav3d.analysis.behavior.state.classification import FULL_STATE_COL
from behav3d.analysis.behavior.utils import _contact_analysis_dir
from behav3d.analysis.behavior.track.visualization.plots.exemplar_track_per_cluster import (
    _build_state_color_map,
)
from behav3d.analysis.behavior.general.visualization.plots.proportion_bars import (
    compute_class_by_stack_proportions,
    draw_stacked_proportion_barv,
    draw_diff_barh,
    SIGNIFICANCE_LEGEND_TEXT,
    _welch_diff_rows,
)
from behav3d.analysis.behavior.state.visualization.plots.state_composition import (
    compute_relative_state_time_matrix,
)
from behav3d.analysis.behavior.state.contact_state_shift import (
    compute_state_shift_features,
    summarize_state_shift_track_fractions,
    BEFORE_LABEL,
    AFTER_LABEL,
)

_CONTACT_GROUPS = ("contact", "no_contact")
_CONTACT_GROUP_TITLES = {"contact": "Contact tracks", "no_contact": "No-contact tracks (null)"}


def _plot_state_composition_over_relative_time(ax, panel_df, *, state_col, state_order, colors, fixed_window_length):
    """Continuous stacked composition vs. time-relative-to-contact (`relative_t`), x-axis capped
    to `[-fixed_window_length, fixed_window_length]` — the "state composition over time" view,
    windowed around the contact bout instead of spanning the whole track."""
    if len(panel_df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    mat = compute_relative_state_time_matrix(
        panel_df, time_col="relative_t", state_col=state_col, state_order=state_order,
    )
    x = mat.index.to_numpy(dtype=float)
    bottom = pd.Series(0.0, index=mat.index).to_numpy()
    for state in reversed(list(state_order)):
        v = mat[state].to_numpy(dtype=float)
        ax.bar(x, v, bottom=bottom, width=1.0, align="edge", linewidth=0, color=colors.get(state), label=str(state))
        bottom = bottom + v
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-float(fixed_window_length), float(fixed_window_length))
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0)
    ax.set_xlabel("Time relative to contact (timepoints)", fontsize=8)
    ax.set_ylabel("Proportion", fontsize=8)


def _plot_contact_state_shift_page(
    state_timepoints_df,
    track_fraction_df,
    *,
    state_col,
    state_order,
    colors,
    params_text,
    fixed_window_length,
    group_label=None,
):
    """Build the combined report figure: rows = {composition over time (capped), diff-bars,
    stacked before/after composition}, columns = {contact tracks, no-contact tracks (null)}.

    ``track_fraction_df`` is a flat (non-indexed) DataFrame with columns ``["contact_group",
    "period", *state_order]`` — one row per (track, period).

    Returns ``(fig, long_rows)`` where ``long_rows`` is a list of flat dicts (one per state x
    group x panel-type) for the combined CSV export.
    """
    fig = plt.figure(figsize=(11.0, 12.5))
    outer = GridSpec(nrows=4, ncols=2, height_ratios=[0.35, 1.0, 1.0, 1.0], hspace=0.6, wspace=0.35)

    ax_header = fig.add_subplot(outer[0, :])
    ax_header.axis("off")
    ax_header.text(0.0, 0.5, params_text, ha="left", va="center", fontsize=8, family="monospace")
    ax_header.text(1.0, 0.5, SIGNIFICANCE_LEGEND_TEXT, ha="right", va="center", fontsize=8)

    long_rows = []

    for c, group in enumerate(_CONTACT_GROUPS):
        ax_time = fig.add_subplot(outer[1, c])
        group_state_tp = state_timepoints_df[state_timepoints_df["contact_group"] == group]
        _plot_state_composition_over_relative_time(
            ax_time, group_state_tp, state_col=state_col, state_order=state_order, colors=colors,
            fixed_window_length=fixed_window_length,
        )
        ax_time.set_title(
            f"{_CONTACT_GROUP_TITLES[group]}\nState composition over time (window={fixed_window_length})",
            fontsize=10,
        )

        ax = fig.add_subplot(outer[2, c])
        group_fractions = track_fraction_df[track_fraction_df["contact_group"] == group]
        before_df = group_fractions[group_fractions["period"] == BEFORE_LABEL][state_order]
        after_df = group_fractions[group_fractions["period"] == AFTER_LABEL][state_order]

        if len(before_df) > 0 and len(after_df) > 0:
            diff_rows = _welch_diff_rows(state_order, before_df, after_df)
            diff_df = pd.DataFrame(diff_rows)
            draw_diff_barh(ax, state_order, diff_df, colors)
            for row in diff_rows:
                long_rows.append({
                    "panel": "diff_bar", "contact_group": group, "state": row["class"],
                    "mean_before": row["mean_a"], "mean_after": row["mean_b"],
                    "diff": row["diff"], "t_stat": row["t_stat"], "p_value": row["p_value"],
                    "stars": row["stars"], "n_before": row["n_a"], "n_after": row["n_b"],
                })
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        ax.set_title(f"{_CONTACT_GROUP_TITLES[group]}\nBefore → After state change", fontsize=10)

        ax2 = fig.add_subplot(outer[3, c])
        state_tp = state_timepoints_df[state_timepoints_df["contact_group"] == group]
        if len(state_tp) > 0:
            props_df = compute_class_by_stack_proportions(
                state_tp, class_col="period", stack_col=state_col,
                class_order=[BEFORE_LABEL, AFTER_LABEL], stack_order=state_order,
            )
            draw_stacked_proportion_barv(ax2, props_df, [BEFORE_LABEL, AFTER_LABEL], state_order, colors)
            for period in (BEFORE_LABEL, AFTER_LABEL):
                if period in props_df.index:
                    for state in state_order:
                        long_rows.append({
                            "panel": "stacked_composition", "contact_group": group,
                            "period": period, "state": state,
                            "proportion": float(props_df.loc[period, state]),
                        })
        else:
            ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)
            ax2.axis("off")
        ax2.set_title(f"{_CONTACT_GROUP_TITLES[group]}\nState composition", fontsize=10)

    legend_state_order = list(reversed(state_order))
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in legend_state_order]
    fig.legend(handles, legend_state_order, loc="lower center", ncol=min(len(state_order), 6), frameon=False, fontsize=8)
    title = "Contact-triggered behavioral-state shift (before vs. after)"
    if group_label:
        title += f" — {group_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, long_rows


def save_state_contact_shift_report(
    df_timepoints,
    adata_states,
    out_dir,
    *,
    contact_col,
    min_bout_length,
    state_col=FULL_STATE_COL,
    window_mode="fixed",
    fixed_window_length=10,
    min_window_timepoints=3,
    sample_col="sample_name",
    track_col="TrackID",
    extra_group_cols=None,
    state_order=None,
    state_colors=None,
    null_seed=0,
    verbose=False,
):
    """Compare each track's behavioral-state composition before vs. after its first sufficiently
    long contact bout (contact tracks), against a timing-matched null before/after split for
    no-contact tracks. Needs only behavioral-state classification (no track-DTW classification) —
    each track's window bounds are its own full timepoint span in ``df_timepoints``.

    Writes a single combined PDF (2x2 grid: diff-bars / stacked composition x contact /
    no-contact) plus CSVs, into ``{out_dir}/contact_analysis/contact_state_shift/{contact_col}/``.

    If ``extra_group_cols`` is given (condition columns already present in ``adata_states.obs``,
    constant per sample), one such 2x2-grid page is written per unique combination of their
    values instead of a single page for the whole dataset — each page restricted to the tracks in
    that combination, labeled in its title and appended to the same PDF. The CSVs likewise gain
    one column per ``extra_group_cols`` entry. With no ``extra_group_cols``, output is unchanged.

    Returns a dict of artifact paths plus ``n_contact_tracks``/``n_no_contact_tracks``/
    ``n_excluded_tracks``.
    """
    groupby_cols = [str(sample_col), str(track_col)]
    extra_group_cols = [str(c) for c in extra_group_cols] if extra_group_cols else []

    if state_order is None or state_colors is None:
        resolved_order, resolved_colors = _build_state_color_map(adata_states, state_col)
        state_order = state_order if state_order is not None else resolved_order
        state_colors = state_colors if state_colors is not None else resolved_colors

    features = compute_state_shift_features(
        df_timepoints,
        adata_states,
        contact_col=contact_col,
        min_bout_length=min_bout_length,
        state_col=state_col,
        window_mode=window_mode,
        fixed_window_length=fixed_window_length,
        min_window_timepoints=min_window_timepoints,
        groupby_cols=groupby_cols,
        null_seed=null_seed,
        verbose=verbose,
    )
    track_windows = features["track_windows"]
    state_timepoints = features["state_timepoints"]

    fraction_df = summarize_state_shift_track_fractions(
        state_timepoints, groupby_cols=groupby_cols, state_col=state_col, state_order=state_order,
    ).reset_index()
    # Attach contact_group (constant per track) as a plain column so the plotting step can filter
    # by it directly, without juggling a mixed-depth MultiIndex.
    contact_group_by_track = state_timepoints.drop_duplicates(subset=groupby_cols)[groupby_cols + ["contact_group"]]
    fraction_df = fraction_df.merge(contact_group_by_track, on=groupby_cols, how="left")
    track_windows_flat = track_windows.reset_index()

    if extra_group_cols:
        missing_group_cols = [c for c in extra_group_cols if c not in adata_states.obs.columns]
        if missing_group_cols:
            raise KeyError(f"Missing group-by columns in adata_states.obs: {missing_group_cols}")
        group_lookup = adata_states.obs[[sample_col] + extra_group_cols].copy()
        group_lookup[sample_col] = group_lookup[sample_col].astype(str)
        group_lookup = group_lookup.drop_duplicates(subset=[sample_col])
        state_timepoints = state_timepoints.merge(group_lookup, on=sample_col, how="left")
        fraction_df = fraction_df.merge(group_lookup, on=sample_col, how="left")
        track_windows_flat = track_windows_flat.merge(group_lookup, on=sample_col, how="left")

    out_dir = _contact_analysis_dir(out_dir, "contact_state_shift", contact_col)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "contact_state_shift.pdf"

    base_params_text = (
        f"contact_col={contact_col}  min_bout_length={min_bout_length}  "
        f"state_col={state_col}  window_mode={window_mode}"
        + (f"  fixed_window_length={fixed_window_length}" if window_mode == "fixed" else "")
        + f"  min_window_timepoints={min_window_timepoints}  null_seed={null_seed}"
    )

    if extra_group_cols:
        group_combos = (
            fraction_df[extra_group_cols]
            .drop_duplicates()
            .sort_values(extra_group_cols)
            .to_dict("records")
        )
    else:
        group_combos = [{}]

    all_long_rows = []
    with PdfPages(pdf_path) as pdf:
        for group_vals in group_combos:
            if group_vals:
                st_mask = pd.Series(True, index=state_timepoints.index)
                fr_mask = pd.Series(True, index=fraction_df.index)
                tw_mask = pd.Series(True, index=track_windows_flat.index)
                for col, val in group_vals.items():
                    st_mask &= state_timepoints[col] == val
                    fr_mask &= fraction_df[col] == val
                    tw_mask &= track_windows_flat[col] == val
                group_state_tp = state_timepoints[st_mask]
                group_fraction_df = fraction_df[fr_mask]
                group_windows = track_windows_flat[tw_mask]
                group_label = ", ".join(f"{c}={v}" for c, v in group_vals.items())
            else:
                group_state_tp = state_timepoints
                group_fraction_df = fraction_df
                group_windows = track_windows_flat
                group_label = None

            group_tracks = group_fraction_df.drop_duplicates(subset=groupby_cols)
            n_contact = int((group_tracks["contact_group"] == "contact").sum())
            n_no_contact = int((group_tracks["contact_group"] == "no_contact").sum())
            n_excluded = int(group_windows["excluded"].sum())
            group_params_text = (
                base_params_text
                + f"\nn_contact_tracks={n_contact}  n_no_contact_tracks={n_no_contact}  "
                f"n_excluded_tracks={n_excluded}"
            )

            fig, group_long_rows = _plot_contact_state_shift_page(
                group_state_tp, group_fraction_df,
                state_col=state_col, state_order=state_order, colors=state_colors,
                params_text=group_params_text, fixed_window_length=fixed_window_length,
                group_label=group_label,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for row in group_long_rows:
                row.update(group_vals)
            all_long_rows.extend(group_long_rows)

    windows_csv = csv_dir / "state_shift_track_windows.csv"
    track_windows_flat.to_csv(windows_csv, index=False)

    long_df = pd.DataFrame(all_long_rows)
    diff_csv = csv_dir / "state_shift_diff_bars.csv"
    stacked_csv = csv_dir / "state_shift_stacked_composition.csv"
    if len(long_df) > 0:
        long_df[long_df["panel"] == "diff_bar"].drop(columns=["panel"]).to_csv(diff_csv, index=False)
        long_df[long_df["panel"] == "stacked_composition"].drop(columns=["panel"]).to_csv(stacked_csv, index=False)
    else:
        pd.DataFrame().to_csv(diff_csv, index=False)
        pd.DataFrame().to_csv(stacked_csv, index=False)

    if verbose:
        print(f"Saved contact state-shift report: {pdf_path}")

    return {
        "contact_col": str(contact_col),
        "min_bout_length": int(min_bout_length),
        "window_mode": str(window_mode),
        "pdf_path": str(pdf_path),
        "csv_dir": str(csv_dir),
        "track_windows_csv": str(windows_csv),
        "diff_bars_csv": str(diff_csv),
        "stacked_composition_csv": str(stacked_csv),
        "n_contact_tracks": features["n_contact_tracks"],
        "n_no_contact_tracks": features["n_no_contact_tracks"],
        "n_excluded_tracks": features["n_excluded_tracks"],
    }

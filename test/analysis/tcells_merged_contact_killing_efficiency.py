"""
Contact-based killing efficiency for tcells_merged.

Unlike the threshold-gated Active Killing analysis (behav3d/features/advanced_timepoint_features.py,
analyze_active_killing_per_timepoint), which only runs per lineage (tcell1/tcell2/tcell3) and only
reports a positive result once a death-signal increase clears a threshold, this script credits EVERY
timepoint where a tcells_merged T cell -- regardless of its current behavioral state -- is in contact
with an organoid. For each such timepoint it looks `observation_window` timepoints into the future
and reports how much the organoid's death signal rose, using a persistent (running-max) version of
the signal so a transient segmentation dropout can never look like the organoid "un-dying" (the
signal used for the increase calculation can only go up or stay flat, never down).

Two killing-efficiency numbers are reported for every contact timepoint, no threshold involved:
- killing_efficiency_raw: the absolute increase in the death signal (nr_dead_mask_pixels by default)
  over the observation window. Bigger organoids naturally show bigger raw numbers.
- killing_efficiency_normalized: that same increase divided by the organoid's own persistent signal
  at the start of the window -- a unitless fold-change (0.5 = the signal grew 50% relative to where
  it was at contact time), comparable across organoids of different sizes. A small epsilon
  (`baseline_epsilon`) is substituted when the starting signal is exactly 0 so the ratio is never a
  divide-by-zero.

When a T cell touches more than one organoid at the same timepoint, only the organoid with the
largest raw increase is credited for that timepoint (mirrors the "best target" convention used by
the existing Active Killing analysis), carrying both its raw and normalized values.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from behav3d.features.advanced_timepoint_features import identify_contact_events_global


# --------------------------------------------------------------------------- #
# Configuration -- edit these to change what the script computes
# --------------------------------------------------------------------------- #
output_dir = Path("/Users/s.deblank-3/Documents/LowDensity_MultiColor")  # run root (parent of analysis/)
immune_cell_type = "tcells_merged"
target_cell_type = "organoid"

observation_window = 10          # how many timepoints into the future to look for a death-signal increase
death_signal_column = "nr_dead_mask_pixels"
contact_column = "contact"       # "contact" (pixel/mask adjacency) or "contact_on_distance" (um threshold)
min_contact_duration = 1         # minimum contiguous contact length (timepoints) to count as an event
baseline_epsilon = 0.1           # substituted for death_at_start when it's exactly 0, avoids divide-by-zero

# Purely a floating-point noise floor, not a biological detection threshold: the organoid
# track-features table has interpolated/smoothed frames, so death_signal_column isn't always
# an exact integer, which can leave a "should be zero" increase as e.g. 1e-13 instead of 0.
# Only used by the distribution plots below to decide "any measurable increase" (responder
# rate, nonzero-only magnitude); killing_efficiency_raw/normalized in the saved CSVs are
# never altered by it.
responder_noise_floor = 1e-6

# Which behavioral-state label to break the killing-efficiency summary/plot down by.
# "full_behavioral_state" is the intrinsic (movement-based) state crossed with binary
# context flags like organoid_contact/dead (e.g. "engaging", "idle"); switch to
# "hmm_intrinsic_behavioral_state" for the movement-only clusters instead (e.g. "fast scanner").
behavioral_state_column = "full_behavioral_state"
# --------------------------------------------------------------------------- #


def _build_persistent_signal_lookup(
    df_target_tracks: pd.DataFrame,
    death_signal_column: str,
) -> dict:
    """
    One running-max (persistent) death-signal timeline per (sample_name, TrackID) target,
    built once and reused for every contact timepoint that touches that target.
    """
    lookup = {}
    for (sample_name, track_id), group in df_target_tracks.groupby(["sample_name", "TrackID"]):
        group = group.sort_values("position_t")
        position_t = group["position_t"].to_numpy()
        signal = group[death_signal_column].to_numpy(dtype=float)
        persistent_signal = np.fmax.accumulate(signal)
        lookup[(sample_name, int(track_id))] = (position_t, persistent_signal)
    return lookup


def compute_contact_killing_efficiency_per_timepoint(
    df_immune_tracks: pd.DataFrame,
    df_target_tracks: pd.DataFrame,
    df_contact_events: pd.DataFrame,
    observation_window: int = 10,
    death_signal_column: str = "nr_dead_mask_pixels",
    baseline_epsilon: float = 0.1,
) -> pd.DataFrame:
    """
    For every contact event's every timepoint, look `observation_window` timepoints into the
    future and compute the raw and normalized increase in each touched organoid's persistent
    death signal. There is no threshold and no "active"/"inactive" classification here -- every
    contact timepoint gets a row.

    Returns
    -------
    pd.DataFrame with one row per (contact_event, timepoint): contact_event_id, sample_name,
    immune_track_id, position_t, targeted_track_id, killing_efficiency_raw,
    killing_efficiency_normalized, observation_complete.
    """
    if df_contact_events.empty:
        return pd.DataFrame()

    max_timepoints = df_target_tracks.groupby("sample_name")["position_t"].max().to_dict()
    persistent_lookup = _build_persistent_signal_lookup(df_target_tracks, death_signal_column)

    results = []
    for _, event in df_contact_events.iterrows():
        sample_name = event["sample_name"]
        immune_track_id = event["immune_track_id"]
        target_ids = [t.strip() for t in str(event["target_track_ids"]).split(",") if t.strip()]
        contact_timepoints = event["contact_timepoints"]
        max_t = max_timepoints.get(sample_name, contact_timepoints[-1])

        target_arrays = {}
        for target_id in target_ids:
            try:
                target_id_int = int(float(target_id))
            except (ValueError, TypeError):
                continue
            arrays = persistent_lookup.get((sample_name, target_id_int))
            if arrays is not None:
                target_arrays[target_id_int] = arrays

        for t in contact_timepoints:
            observation_end_t = t + observation_window
            observation_complete = observation_end_t <= max_t

            target_evals = []
            for target_id_int, (pt_arr, signal_arr) in target_arrays.items():
                # Signal at or just before t.
                start_idx = np.searchsorted(pt_arr, t, side="right") - 1
                if start_idx < 0:
                    continue
                death_at_start = signal_arr[start_idx]

                # Signal at the first frame >= observation_end_t, falling back to the
                # target's last available frame if the window runs past track end.
                end_idx = np.searchsorted(pt_arr, observation_end_t, side="left")
                if end_idx >= len(pt_arr):
                    end_idx = len(pt_arr) - 1
                death_at_end = signal_arr[end_idx]

                raw_increase = death_at_end - death_at_start
                effective_baseline = death_at_start if death_at_start != 0 else baseline_epsilon
                normalized_increase = raw_increase / effective_baseline

                target_evals.append({
                    "target_id": target_id_int,
                    "raw": raw_increase,
                    "normalized": normalized_increase,
                })

            if not target_evals:
                continue

            best = max(target_evals, key=lambda d: d["raw"])
            results.append({
                "contact_event_id": event["contact_event_id"],
                "sample_name": sample_name,
                "immune_track_id": immune_track_id,
                "position_t": t,
                "targeted_track_id": best["target_id"],
                "killing_efficiency_raw": best["raw"],
                "killing_efficiency_normalized": best["normalized"],
                "observation_complete": observation_complete,
            })

    return pd.DataFrame(results)


def summarize_killing_efficiency_by_behavioral_state(
    df_per_timepoint: pd.DataFrame,
    state_column: str = "full_behavioral_state",
) -> pd.DataFrame:
    """
    One row per behavioral state: how often T cells in that state were credited with a
    contact-based killing-efficiency value, and the distribution of both metrics. Rows
    with a missing state (no match in the behavioral-states table) are dropped.
    """
    df = df_per_timepoint.dropna(subset=[state_column])
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(state_column).agg(
        n_contact_timepoints=("position_t", "count"),
        n_unique_tracks=("immune_track_id", "nunique"),
        mean_killing_efficiency_raw=("killing_efficiency_raw", "mean"),
        median_killing_efficiency_raw=("killing_efficiency_raw", "median"),
        sem_killing_efficiency_raw=("killing_efficiency_raw", "sem"),
        sum_killing_efficiency_raw=("killing_efficiency_raw", "sum"),
        mean_killing_efficiency_normalized=("killing_efficiency_normalized", "mean"),
        median_killing_efficiency_normalized=("killing_efficiency_normalized", "median"),
        sem_killing_efficiency_normalized=("killing_efficiency_normalized", "sem"),
        sum_killing_efficiency_normalized=("killing_efficiency_normalized", "sum"),
    ).reset_index()
    summary = summary.sort_values("mean_killing_efficiency_raw", ascending=False).reset_index(drop=True)
    return summary


def _style_axis(ax: plt.Axes) -> None:
    """Minimal, consistent styling shared with the rest of the codebase's dashboard panels."""
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#888888")
    ax.tick_params(axis="both", which="both", length=4, width=0.6, colors="#555555")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)


def _state_palette(states: list) -> dict:
    """
    One fixed, colorblind-safe color per state, assigned in a stable order (by mean raw
    killing efficiency, descending -- the same order every plot below uses) and reused
    identically across all three plots so the same state is always the same color.
    """
    colors = sns.color_palette("colorblind", n_colors=max(len(states), 1))
    return dict(zip(states, colors))


def plot_killing_efficiency_by_behavioral_state(
    df_summary_by_state: pd.DataFrame,
    out_path: Path,
    state_column: str = "full_behavioral_state",
    palette: dict = None,
) -> None:
    """
    Horizontal bar chart, one panel for the raw metric and one for the normalized metric,
    states ordered by mean raw killing efficiency (descending, matches the summary table).
    Error bars are SEM; each bar is annotated with its timepoint count.
    """
    if df_summary_by_state.empty:
        return

    states = df_summary_by_state[state_column].tolist()
    palette = palette or _state_palette(states)
    bar_colors = [palette[s] for s in states]
    y = np.arange(len(states))
    bar_height = 0.65

    fig, (ax_raw, ax_norm) = plt.subplots(
        1, 2, figsize=(11, max(3, 0.5 * len(states) + 1.5)), sharey=True,
    )

    for ax, metric, title, xlabel in (
        (ax_raw, "raw", "Raw killing efficiency", "Mean death-signal increase (pixels)"),
        (ax_norm, "normalized", "Normalized killing efficiency", "Mean fold-change vs. baseline"),
    ):
        means = df_summary_by_state[f"mean_killing_efficiency_{metric}"].to_numpy()
        sems = df_summary_by_state[f"sem_killing_efficiency_{metric}"].fillna(0).to_numpy()
        ns = df_summary_by_state["n_contact_timepoints"].to_numpy()

        _style_axis(ax)
        ax.barh(
            y, means, height=bar_height, color=bar_colors, edgecolor="white",
            linewidth=0.8, alpha=0.88, xerr=sems, error_kw=dict(ecolor="#555555", elinewidth=1, capsize=3),
        )
        x_pad = max(means.max(), 1e-9) * 0.03
        for yi, m, s, n in zip(y, means, sems, ns):
            ax.text(
                m + s + x_pad, yi, f"n={n}", ha="left", va="center",
                fontsize=8.5, color="#444444",
            )
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)

    ax_raw.set_yticks(y)
    ax_raw.set_yticklabels(states, fontsize=9.5)
    ax_raw.invert_yaxis()  # highest mean (first row) at the top
    fig.suptitle(
        f"Mean contact-based killing efficiency by behavioral state ({state_column})",
        fontsize=13, fontweight="semibold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_killing_efficiency_distribution_by_state(
    df_per_timepoint: pd.DataFrame,
    out_path: Path,
    state_column: str = "full_behavioral_state",
    state_order: list = None,
    palette: dict = None,
    responder_noise_floor: float = 1e-6,
) -> None:
    """
    For each metric (raw, normalized): a "responder rate" panel (% of contact timepoints with
    any measurable increase, i.e. value > responder_noise_floor) next to a violin+strip of the
    magnitude among only the responding timepoints, on a log x-axis. responder_noise_floor is
    a floating-point noise floor (see the module-level constant of the same name), not a
    biological threshold -- it exists so an interpolated frame's near-zero float artifact
    (e.g. 1e-13) isn't counted as "a response" and doesn't get log-transformed into an
    extreme outlier that swamps the rest of the violin.

    A combined violin/box over ALL timepoints would be dominated by the large mass of exact
    zeros (every state's raw and normalized median is 0 -- most contact timepoints show no
    measurable increase within the observation window), hiding the actual shape of the
    distribution when something does happen. Splitting "how often" from "how much, when it
    happens" avoids that.
    """
    df = df_per_timepoint.dropna(subset=[state_column])
    if df.empty:
        return

    states = state_order or sorted(df[state_column].unique())
    palette = palette or _state_palette(states)
    colors = [palette[s] for s in states]
    y = np.arange(len(states))

    fig, axes = plt.subplots(
        2, 2, figsize=(12, (max(3, 0.55 * len(states) + 1.5)) * 2),
    )

    for row, (metric, metric_label) in enumerate((
        ("raw", "Raw (death-signal increase, pixels)"),
        ("normalized", "Normalized (fold-change vs. baseline)"),
    )):
        col = f"killing_efficiency_{metric}"
        ax_rate, ax_mag = axes[row]

        responder_pct, ns = [], []
        for s in states:
            vals = df.loc[df[state_column] == s, col]
            ns.append(len(vals))
            responder_pct.append(100.0 * (vals > responder_noise_floor).mean() if len(vals) else 0.0)

        _style_axis(ax_rate)
        ax_rate.barh(y, responder_pct, height=0.65, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
        for yi, pct, n in zip(y, responder_pct, ns):
            ax_rate.text(pct + 2, yi, f"{pct:.0f}% (n={n})", ha="left", va="center", fontsize=8.5, color="#444444")
        ax_rate.set_xlim(0, 112)
        ax_rate.set_yticks(y)
        ax_rate.set_yticklabels(states, fontsize=9.5)
        ax_rate.invert_yaxis()
        ax_rate.set_xlabel("Contact timepoints with any increase (%)", fontsize=10, labelpad=8)
        ax_rate.set_title(f"{metric_label}\nresponder rate", fontsize=11, fontweight="semibold", pad=8)

        df_nonzero = df[df[col] > responder_noise_floor]
        _style_axis(ax_mag)
        if not df_nonzero.empty:
            # Fit the violin's KDE in log10 space, not linear space -- these nonzero values
            # span many orders of magnitude (especially killing_efficiency_normalized, which
            # can reach huge values when baseline_epsilon kicks in), so a linear-space KDE
            # rescaled onto a log axis afterward comes out as a degenerate sliver/shark-fin
            # shape rather than a real violin. Plotting log10(value) directly and relabeling
            # the ticks back to the original units fits the KDE in the space the data is
            # actually shaped for.
            df_nonzero = df_nonzero.assign(_log10_value=np.log10(df_nonzero[col]))
            sns.violinplot(
                data=df_nonzero, y=state_column, x="_log10_value", order=states,
                orient="h", ax=ax_mag, hue=state_column, hue_order=states, palette=palette,
                legend=False, cut=0, inner=None, linewidth=0.8, saturation=0.9,
            )
            sns.stripplot(
                data=df_nonzero, y=state_column, x="_log10_value", order=states,
                orient="h", ax=ax_mag, color="#333333", alpha=0.15, size=2, jitter=0.3,
            )
            ax_mag.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{10 ** v:.3g}"))
        else:
            ax_mag.text(
                0.5, 0.5, "No nonzero timepoints", transform=ax_mag.transAxes,
                ha="center", va="center", fontsize=11, color="#999999",
            )
        ax_mag.set_ylabel("")
        ax_mag.set_yticklabels([])
        ax_mag.tick_params(left=False)
        ax_mag.set_xlabel(f"{metric_label} (log scale, nonzero only)", fontsize=10, labelpad=8)
        ax_mag.set_title(f"{metric_label}\nmagnitude when it happens", fontsize=11, fontweight="semibold", pad=8)

    fig.suptitle(
        f"Killing efficiency distribution by behavioral state ({state_column})",
        fontsize=13, fontweight="semibold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_killing_efficiency_ecdf_by_state(
    df_per_timepoint: pd.DataFrame,
    out_path: Path,
    state_column: str = "full_behavioral_state",
    state_order: list = None,
    palette: dict = None,
) -> None:
    """
    One ECDF line per behavioral state, one panel per metric. Where a line leaves x=0 shows
    the fraction of that state's contact timepoints with no measurable increase; its shape
    past that shows the distribution of the increase when there is one -- both the frequency
    and magnitude questions in a single view, without an arbitrary zero/nonzero split. A
    symlog x-axis keeps the exact-zero mass visible while compressing the long right tail.
    """
    df = df_per_timepoint.dropna(subset=[state_column])
    if df.empty:
        return

    states = state_order or sorted(df[state_column].unique())
    palette = palette or _state_palette(states)

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, metric, xlabel, linthresh in (
        (ax_raw, "raw", "Raw killing efficiency (death-signal increase, pixels)", 1.0),
        (ax_norm, "normalized", "Normalized killing efficiency (fold-change vs. baseline)", 0.05),
    ):
        col = f"killing_efficiency_{metric}"
        _style_axis(ax)
        sns.ecdfplot(
            data=df, x=col, hue=state_column, hue_order=states, palette=palette,
            ax=ax, legend=False, linewidth=1.8,
        )
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
        ax.set_ylabel("Cumulative proportion of contact timepoints", fontsize=10, labelpad=8)
        ax.set_title(xlabel.split(" (")[0], fontsize=12, fontweight="semibold", pad=10)

    fig.suptitle("Killing efficiency ECDF by behavioral state", fontsize=13, fontweight="semibold", y=1.06)
    handles = [plt.Line2D([0], [0], color=palette[s], lw=2.5, label=s) for s in states]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(states), 4), fontsize=9.5, frameon=True, framealpha=0.9,
        edgecolor="#cccccc", title=state_column,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    analysis_dir = output_dir / "analysis"
    immune_csv = (
        analysis_dir / immune_cell_type / "track_features"
        / f"BEHAV3D_{immune_cell_type}_combined_track_features_filtered.csv"
    )
    target_csv = (
        analysis_dir / target_cell_type / "track_features"
        / f"BEHAV3D_{target_cell_type}_combined_track_features.csv"
    )
    behavioral_states_csv = (
        analysis_dir / immune_cell_type / "behavioral_states"
        / f"BEHAV3D_{immune_cell_type}_behavioral_states.csv"
    )

    print(f"Loading immune cell tracks ({immune_cell_type}) from {immune_csv}")
    df_immune = pd.read_csv(immune_csv, low_memory=False)
    print(f"Loading target ({target_cell_type}) tracks from {target_csv}")
    df_target = pd.read_csv(target_csv, low_memory=False)

    print(
        f"Identifying contact events ({immune_cell_type} vs. {target_cell_type}, "
        f"contact_column={contact_column!r}, min_contact_duration={min_contact_duration})..."
    )
    df_contact_events = identify_contact_events_global(
        df_immune_tracks=df_immune,
        target_cell_types=[target_cell_type],
        min_contact_duration=min_contact_duration,
        contact_column=contact_column,
    )
    print(f"  Found {len(df_contact_events)} contact events.")

    print(
        f"Computing contact-based killing efficiency (observation_window={observation_window}, "
        f"death_signal_column={death_signal_column!r})..."
    )
    df_per_timepoint = compute_contact_killing_efficiency_per_timepoint(
        df_immune_tracks=df_immune,
        df_target_tracks=df_target,
        df_contact_events=df_contact_events,
        observation_window=observation_window,
        death_signal_column=death_signal_column,
        baseline_epsilon=baseline_epsilon,
    )

    # Context columns -- what behavioral state the T cell was in, and which original lineage
    # it came from. Informational only: never used to filter which rows get credited.
    #
    # tcells_merged's TrackID numbering is reassigned from scratch every time Cell Type
    # Grouping is (re)run (behav3d/analysis/grouping.py), so if behavioral_states.csv was
    # produced by State Classification *before* the most recent regroup, the same TrackID
    # number in the two files can refer to two completely different tracks. A plain merge on
    # (sample_name, TrackID, position_t) would silently attach the wrong state to a credited
    # row in that case. Every credited row is guaranteed to have real organoid contact (that's
    # how it got credited), so behavioral_states.csv's own organoid_contact column is used
    # below as a cross-check: a credited row whose matched behavioral-states row disagrees (or
    # has no match at all) gets its state columns nulled out rather than silently kept.
    if behavioral_states_csv.exists() and not df_per_timepoint.empty:
        state_result_cols = ["full_behavioral_state", "hmm_intrinsic_behavioral_state", "dead"]
        state_cols = ["sample_name", "TrackID", "position_t", "organoid_contact", *state_result_cols]
        df_states = pd.read_csv(behavioral_states_csv, usecols=lambda c: c in state_cols)
        df_per_timepoint = df_per_timepoint.merge(
            df_states,
            left_on=["sample_name", "immune_track_id", "position_t"],
            right_on=["sample_name", "TrackID", "position_t"],
            how="left",
        ).drop(columns=["TrackID"])

        if "organoid_contact" in df_per_timepoint.columns:
            n_total = len(df_per_timepoint)
            unreliable = (
                df_per_timepoint["organoid_contact"].isna()
                | (df_per_timepoint["organoid_contact"].astype("boolean") != True)  # noqa: E712
            )
            n_unreliable = int(unreliable.sum())
            present_result_cols = [c for c in state_result_cols if c in df_per_timepoint.columns]
            df_per_timepoint.loc[unreliable, present_result_cols] = np.nan
            df_per_timepoint = df_per_timepoint.drop(columns=["organoid_contact"])

            if n_total and n_unreliable / n_total > 0.05:
                n_tracks_tf = df_immune[["sample_name", "TrackID"]].drop_duplicates().shape[0]
                n_tracks_bs = df_states[["sample_name", "TrackID"]].drop_duplicates().shape[0]
                print(
                    f"\nWARNING: {n_unreliable}/{n_total} ({n_unreliable / n_total:.0%}) credited "
                    "timepoints could not be reliably matched to a behavioral state -- "
                    "behavioral_states.csv's own organoid_contact disagreed with (or had no row "
                    f"for) a timepoint this script already confirmed was in contact. "
                    f"{immune_cell_type}'s track-features table has {n_tracks_tf} unique tracks "
                    f"vs {n_tracks_bs} in its behavioral_states.csv -- this usually means Cell "
                    "Type Grouping was rerun (reassigning TrackIDs) after State Classification "
                    "last produced behavioral_states.csv, so the same TrackID number no longer "
                    "refers to the same track in both files. The behavioral-state columns for "
                    "those rows were dropped (not used in the by-state summary/plot below). "
                    f"Re-run State Classification for {immune_cell_type} to refresh "
                    "behavioral_states.csv against the current grouping, then re-run this script "
                    "for a complete breakdown.\n"
                )

    if not df_per_timepoint.empty:
        df_origin = df_immune[["sample_name", "TrackID", "origin_cell_type", "origin_TrackID"]].drop_duplicates(
            subset=["sample_name", "TrackID"]
        )
        df_per_timepoint = df_per_timepoint.merge(
            df_origin,
            left_on=["sample_name", "immune_track_id"],
            right_on=["sample_name", "TrackID"],
            how="left",
        ).drop(columns=["TrackID"])

    out_dir = analysis_dir / immune_cell_type / "contact_killing_efficiency" / target_cell_type
    out_dir.mkdir(parents=True, exist_ok=True)

    df_contact_events.drop(columns=["contact_timepoints"]).to_csv(
        out_dir / f"contact_events_{immune_cell_type}.csv", index=False
    )
    df_per_timepoint.to_csv(
        out_dir / f"contact_killing_efficiency_per_timepoint_{immune_cell_type}.csv", index=False
    )

    if df_per_timepoint.empty:
        print("No contact timepoints found -- skipping summary files.")
        return

    df_by_track = df_per_timepoint.groupby(["sample_name", "immune_track_id"]).agg(
        n_contact_timepoints=("position_t", "count"),
        n_contact_events=("contact_event_id", "nunique"),
        sum_killing_efficiency_raw=("killing_efficiency_raw", "sum"),
        mean_killing_efficiency_raw=("killing_efficiency_raw", "mean"),
        max_killing_efficiency_raw=("killing_efficiency_raw", "max"),
        sum_killing_efficiency_normalized=("killing_efficiency_normalized", "sum"),
        mean_killing_efficiency_normalized=("killing_efficiency_normalized", "mean"),
        max_killing_efficiency_normalized=("killing_efficiency_normalized", "max"),
    ).reset_index()
    event_durations = (
        df_contact_events.groupby(["sample_name", "immune_track_id"])["contact_duration"]
        .sum()
        .reset_index(name="total_contact_duration")
    )
    df_by_track = df_by_track.merge(event_durations, on=["sample_name", "immune_track_id"], how="left")

    df_by_organoid = df_per_timepoint.groupby(["sample_name", "targeted_track_id"]).agg(
        n_credited_timepoints=("position_t", "count"),
        n_contacting_tcells=("immune_track_id", "nunique"),
        sum_killing_efficiency_raw=("killing_efficiency_raw", "sum"),
        mean_killing_efficiency_raw=("killing_efficiency_raw", "mean"),
        max_killing_efficiency_raw=("killing_efficiency_raw", "max"),
        sum_killing_efficiency_normalized=("killing_efficiency_normalized", "sum"),
        mean_killing_efficiency_normalized=("killing_efficiency_normalized", "mean"),
        max_killing_efficiency_normalized=("killing_efficiency_normalized", "max"),
    ).reset_index()

    df_by_track.to_csv(
        out_dir / f"contact_killing_efficiency_summary_by_track_{immune_cell_type}.csv", index=False
    )
    df_by_organoid.to_csv(
        out_dir / f"contact_killing_efficiency_summary_by_organoid_{immune_cell_type}.csv", index=False
    )

    if behavioral_state_column in df_per_timepoint.columns:
        df_by_state = summarize_killing_efficiency_by_behavioral_state(
            df_per_timepoint, state_column=behavioral_state_column
        )
        if not df_by_state.empty:
            df_by_state.to_csv(
                out_dir / f"contact_killing_efficiency_summary_by_behavioral_state_{immune_cell_type}.csv",
                index=False,
            )
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Same state order and colors reused across all three plots below, so e.g.
            # "idle engager" is always the same color whichever figure you're looking at.
            state_order = df_by_state[behavioral_state_column].tolist()
            palette = _state_palette(state_order)

            plot_killing_efficiency_by_behavioral_state(
                df_by_state, plots_dir / f"killing_efficiency_by_{behavioral_state_column}.png",
                state_column=behavioral_state_column, palette=palette,
            )
            plot_killing_efficiency_distribution_by_state(
                df_per_timepoint,
                plots_dir / f"killing_efficiency_distribution_by_{behavioral_state_column}.png",
                state_column=behavioral_state_column, state_order=state_order, palette=palette,
                responder_noise_floor=responder_noise_floor,
            )
            plot_killing_efficiency_ecdf_by_state(
                df_per_timepoint,
                plots_dir / f"killing_efficiency_ecdf_by_{behavioral_state_column}.png",
                state_column=behavioral_state_column, state_order=state_order, palette=palette,
            )
            print(f"  Behavioral-state summary + 3 plots saved ({len(df_by_state)} states) -> {plots_dir}")
        else:
            print(f"  No rows had a {behavioral_state_column!r} value -- skipped behavioral-state summary/plot.")
    else:
        print(f"  Column {behavioral_state_column!r} not found -- skipped behavioral-state summary/plot.")

    print()
    print(
        "killing_efficiency_raw = absolute increase in the persistent death signal "
        f"({death_signal_column}) over {observation_window} timepoints.\n"
        "killing_efficiency_normalized = that same increase divided by the organoid's own "
        "signal at contact time -- a unitless fold-change, comparable across organoids of "
        "different sizes."
    )
    print()
    print(f"Saved outputs to {out_dir}")
    print(f"  Contact timepoints credited: {len(df_per_timepoint)}")
    print(f"  Unique T cell tracks: {df_per_timepoint['immune_track_id'].nunique()}")
    print(f"  Unique organoids credited: {df_per_timepoint['targeted_track_id'].nunique()}")
    print(f"  Mean killing_efficiency_raw: {df_per_timepoint['killing_efficiency_raw'].mean():.3f}")
    print(f"  Median killing_efficiency_raw: {df_per_timepoint['killing_efficiency_raw'].median():.3f}")
    print(f"  Mean killing_efficiency_normalized: {df_per_timepoint['killing_efficiency_normalized'].mean():.3f}")
    print(f"  Median killing_efficiency_normalized: {df_per_timepoint['killing_efficiency_normalized'].median():.3f}")


if __name__ == "__main__":
    main()

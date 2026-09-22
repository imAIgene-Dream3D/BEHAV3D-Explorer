"""Two-row figure for a two-class long-contact comparison, e.g. ameboid vs.
mesenchymal: per-sample paired dot+line plot on top, group-mean bar chart
directly underneath, sharing one x-axis so the bars line up under their
matching column of dots.

NOTE: this is a throwaway plotting script, in the same spirit as
``plot_long_contact_publication.py`` in this directory -- it turns one row of
the pipeline's own ``contact_long_contact_percentage_per_sample.csv`` (written
by ``save_track_contact_duration_comparison`` in
``behav3d/analysis/behavior/track/visualization/plots/contact_duration_report.py``
when ``long_contact_threshold`` is set) into a manuscript figure. Structured as
``# %%``-delimited cells so it runs line by line / cell by cell in VS Code's or
Jupyter's interactive window; it's also a plain script:
``python test/plot_long_contact_publication_figure.py`` runs every cell top to
bottom.
"""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# %%
# Configuration -- edit these, then run this cell and everything below it
CLASS_A, CLASS_A_LABEL = "ameboid", "Ameboid"
CLASS_B, CLASS_B_LABEL = "mesenchymal", "Mesenchymal"
CSV_PATH = "contact_long_contact_percentage_per_sample.csv"
OUT_PREFIX = "contact_long_contact_lineplot_bar_figure"

CLASS_A_COLOR = "#0072B2"    # Okabe-Ito blue
CLASS_B_COLOR = "#D55E00"    # Okabe-Ito vermillion
LINE_COLOR = "#8C8C8C"
RNG_SEED = 0

# %%
# Helper functions -- definitions only, safe to run once and forget


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,   # editable text in Illustrator, not outlined paths
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _clean_axes(ax, keep=("left", "bottom")):
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(spine in keep)
    ax.tick_params(direction="out")


def _load(csv_path):
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="sample_name", columns="target_class", values="pct_long_contact")
    pivot = pivot.sort_index()
    return pivot[CLASS_A].to_numpy(), pivot[CLASS_B].to_numpy(), pivot.index.tolist()


def _lineplot(ax, class_a_vals, class_b_vals):
    """Per-sample paired dots + connecting line. Both classes sit at fixed,
    explicit x positions (1 and 2, in CLASS_A -> CLASS_B order) so the bar
    panel below can reuse the same two positions and line up under them.
    """
    n = len(class_a_vals)
    rng = np.random.default_rng(RNG_SEED)
    jitter = (rng.random(n) - 0.5) * 0.16
    x_a, x_b = 1 + jitter, 2 + jitter

    for i in range(n):
        ax.plot([x_a[i], x_b[i]], [class_a_vals[i], class_b_vals[i]], color=LINE_COLOR, lw=0.7, alpha=0.6, zorder=2)
    ax.scatter(x_a, class_a_vals, s=16, color=CLASS_A_COLOR, edgecolor="white", linewidth=0.3, zorder=3)
    ax.scatter(x_b, class_b_vals, s=16, color=CLASS_B_COLOR, edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel("Tracks with long contact (%)")

    y_max = max(class_a_vals.max(), class_b_vals.max())
    ax.set_ylim(0, y_max * 1.15)

    _clean_axes(ax, keep=("left",))
    ax.tick_params(bottom=False, labelbottom=False)
    ax.text(-0.24, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _barplot(ax, class_a_vals, class_b_vals):
    """Group-mean bars at the same x = 1, 2 positions as the dots above, so
    both panels share one x-axis (CLASS_A -> CLASS_B, left to right).
    """
    means = [class_a_vals.mean(), class_b_vals.mean()]
    sems = [stats.sem(class_a_vals), stats.sem(class_b_vals)]
    colors = [CLASS_A_COLOR, CLASS_B_COLOR]

    ax.bar([1, 2], means, width=0.6, color=colors, zorder=2)
    ax.errorbar([1, 2], means, yerr=sems, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
    for x0, m, sem in zip((1, 2), means, sems):
        ax.annotate(f"{m:.1f}%", (x0, m + sem), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)

    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([CLASS_A_LABEL, CLASS_B_LABEL])
    ax.set_ylabel("Mean (%)")
    ax.set_ylim(0, max(means) * 1.35)

    _clean_axes(ax)
    ax.text(-0.24, 1.10, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


# %%
# Load the data
_style()
class_a_vals, class_b_vals, samples = _load(CSV_PATH)
print(f"n = {len(class_a_vals)} samples: {samples}")
print(f"{CLASS_A_LABEL} mean = {class_a_vals.mean():.2f}%, {CLASS_B_LABEL} mean = {class_b_vals.mean():.2f}%")

# %%
# Figure -- panel a (per-sample lines) stacked on panel b (group-mean bars),
# sharing one x-axis so bars sit directly under their matching column of dots
fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(2.6, 4.6), sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12},
)
fig.subplots_adjust(left=0.24, right=0.96, bottom=0.09, top=0.95)
_lineplot(ax_a, class_a_vals, class_b_vals)
_barplot(ax_b, class_a_vals, class_b_vals)
fig  # in an interactive window, re-run this line to preview progress

# %%
# Save
fig.savefig(f"{OUT_PREFIX}.pdf")
fig.savefig(f"{OUT_PREFIX}.png", dpi=300)
print(f"Saved: {OUT_PREFIX}.pdf, {OUT_PREFIX}.png")

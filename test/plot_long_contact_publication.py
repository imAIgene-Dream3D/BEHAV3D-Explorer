"""Publication-style (Nature Methods-ish) figure for a two-class long-contact
comparison, e.g. ameboid vs. mesenchymal.

NOTE: this is a throwaway plotting script for turning one row of the pipeline's own
``contact_long_contact_percentage_per_sample.csv`` (written by
``save_track_contact_duration_comparison`` in
``behav3d/analysis/behavior/track/visualization/plots/contact_duration_report.py`` when
``long_contact_threshold`` is set) into a single manuscript figure, rather than a page
of the full multi-comparison PDF report. It intentionally does not go through
``PdfPages``/the report's boxplot+jitter convention (see ``_draw_long_contact_*`` in
that module) -- this is meant for fast, deliberate styling of one specific comparison,
not the general-purpose report.

Two panels, in the spirit of "estimation graphics" (Ho, Tumkaya, Aryal, Choi &
Claridge-Chang, "Moving beyond P values: data analysis with estimation graphics",
Nature Methods 2019):

  a) Paired raw data. One thin line per sample connecting its class-A -> class-B
     value (jittered slightly for visibility), group mean +/- SEM as an offset
     crossbar, and a paired Wilcoxon signed-rank p-value bracket.
  b) Effect size. The paired mean difference (B - A) as a point estimate with a
     bootstrap 95% CI, drawn against the bootstrap sampling distribution, with each
     sample's own raw difference shown alongside it.

Structured as ``# %%``-delimited cells (same convention as ``plot_tracklet_similarity_
chain.py``) so it runs line by line / cell by cell in VS Code's or Jupyter's
interactive window -- edit the "Configuration" cell's constants, run the "Load data"
and "Statistics" cells and inspect the printed numbers, then run the two panel cells
one at a time to watch the figure build up before saving it. It's also still a plain
script: ``python test/plot_long_contact_publication.py`` runs every cell top to bottom.
"""
# %%
# Imports
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# scipy's exact Wilcoxon path hits a benign divide-by-zero internally when every
# paired difference shares the same sign (the extreme W=0 case) -- the returned
# p-value is still correct, so silence the warning rather than the computation.
warnings.filterwarnings("ignore", message="divide by zero encountered in vecdot")
warnings.filterwarnings("ignore", message="overflow encountered in vecdot")
warnings.filterwarnings("ignore", message="invalid value encountered in vecdot")

# %%
# Configuration -- edit these, then run this cell and everything below it
CLASS_A, CLASS_A_LABEL = "ameboid", "Ameboid"
CLASS_B, CLASS_B_LABEL = "mesenchymal", "Mesenchymal"
CSV_PATH = "contact_long_contact_percentage_per_sample.csv"
OUT_PREFIX = "contact_long_contact_publication_figure"

CLASS_A_COLOR = "#0072B2"    # Okabe-Ito blue
CLASS_B_COLOR = "#D55E00"    # Okabe-Ito vermillion
LINE_COLOR = "#8C8C8C"
N_BOOT = 5000
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


def _bootstrap_mean_diff(diffs, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    return diffs[idx].mean(axis=1)


def _panel_a(ax, class_a_vals, class_b_vals, p_value):
    n = len(class_a_vals)
    rng = np.random.default_rng(RNG_SEED)
    jitter = (rng.random(n) - 0.5) * 0.16
    x_a, x_b = 1 + jitter, 2 + jitter

    for i in range(n):
        ax.plot([x_a[i], x_b[i]], [class_a_vals[i], class_b_vals[i]], color=LINE_COLOR, lw=0.7, alpha=0.6, zorder=2)
    ax.scatter(x_a, class_a_vals, s=16, color=CLASS_A_COLOR, edgecolor="white", linewidth=0.3, zorder=3)
    ax.scatter(x_b, class_b_vals, s=16, color=CLASS_B_COLOR, edgecolor="white", linewidth=0.3, zorder=3)

    for x0, vals in ((1, class_a_vals), (2, class_b_vals)):
        mean_v, sem_v = np.mean(vals), stats.sem(vals)
        xs = x0 + 0.34
        ax.plot([xs - 0.08, xs + 0.08], [mean_v, mean_v], color="black", lw=1.1, zorder=4)
        ax.errorbar([xs], [mean_v], yerr=[sem_v], color="black", lw=0.8, capsize=2, capthick=0.8, zorder=4)

    ax.set_xlim(0.5, 2.85)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([CLASS_A_LABEL, CLASS_B_LABEL])
    ax.set_ylabel("Tracks with long contact (%)")

    y_max = max(class_a_vals.max(), class_b_vals.max())
    y_min = min(0, class_a_vals.min(), class_b_vals.min())
    span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - span * 0.05, y_max + span * 0.30)

    bracket_y = y_max + span * 0.14
    tick = span * 0.03
    ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + tick, bracket_y + tick, bracket_y], color="black", lw=0.7)
    p_text = f"P = {p_value:.4f}" if p_value >= 0.001 else "P < 0.001"
    ax.text(1.5, bracket_y + tick * 1.4, f"{p_text}\nWilcoxon, n = {n}", ha="center", va="bottom", fontsize=6.5)

    _clean_axes(ax)
    ax.text(-0.24, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _panel_b(ax, diffs):
    n = len(diffs)
    boot = _bootstrap_mean_diff(diffs)
    obs_mean = diffs.mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    ax.axhline(0, color="#B0B0B0", lw=0.6, linestyle=(0, (3, 2)), zorder=1)

    # bw_method widened beyond scipy's default: with only a handful of raw values the
    # bootstrap distribution is discrete/lumpy, and a wider kernel reads as a
    # distribution rather than an artifact of the resampling support.
    kde = stats.gaussian_kde(boot, bw_method=0.4)
    y_grid = np.linspace(boot.min(), boot.max(), 200)
    density = kde(y_grid)
    density = density / density.max() * 0.30
    x_violin = 1.15
    ax.fill_betweenx(y_grid, x_violin - density, x_violin + density, color="#1BAF7A", alpha=0.35, lw=0, zorder=1)

    rng = np.random.default_rng(RNG_SEED)
    jitter = (rng.random(n) - 0.5) * 0.12
    ax.scatter(np.full(n, 0.75) + jitter, diffs, s=14, color=LINE_COLOR, edgecolor="white", linewidth=0.3, zorder=3)

    ax.plot([x_violin, x_violin], [ci_lo, ci_hi], color="black", lw=1.1, zorder=4)
    ax.scatter([x_violin], [obs_mean], s=26, color="black", zorder=5)

    ax.set_xlim(0.4, 1.55)
    ax.set_xticks([0.75, 1.15])
    ax.set_xticklabels(["paired\ndiffs", "mean diff.\n(95% CI)"])
    ax.set_ylabel(f"Δ long contact, {CLASS_B_LABEL[0]} − {CLASS_A_LABEL[0]} (pp)")

    span = max(diffs.max() - min(0, diffs.min()), 1.0)
    ax.set_ylim(min(0, diffs.min()) - span * 0.12, diffs.max() + span * 0.18)

    label_y = ci_lo - span * 0.06
    ax.text(
        x_violin, label_y, f"{obs_mean:+.1f} pp\n[{ci_lo:+.1f}, {ci_hi:+.1f}]",
        ha="center", va="top", fontsize=6.5,
    )

    _clean_axes(ax)
    ax.text(-0.32, 1.06, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    return ci_lo, ci_hi, obs_mean


# %%
# Load the data
_style()
class_a_vals, class_b_vals, samples = _load(CSV_PATH)
diffs = class_b_vals - class_a_vals
print(f"n = {len(diffs)} samples: {samples}")
print(f"{CLASS_A_LABEL} mean = {class_a_vals.mean():.2f}%, {CLASS_B_LABEL} mean = {class_b_vals.mean():.2f}%")

# %%
# Statistics -- inspect these before trusting the annotated figure
wilcoxon = stats.wilcoxon(class_b_vals, class_a_vals, alternative="two-sided")
ttest = stats.ttest_rel(class_b_vals, class_a_vals)
print(f"Wilcoxon signed-rank (two-sided): W = {wilcoxon.statistic:.3f}, P = {wilcoxon.pvalue:.4f}")
print(f"Paired t-test (reference): t({len(diffs) - 1}) = {ttest.statistic:.3f}, P = {ttest.pvalue:.4f}")

# %%
# Panel a -- paired raw data
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.6, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0]})
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.80, wspace=0.65)
_panel_a(ax_a, class_a_vals, class_b_vals, wilcoxon.pvalue)
fig  # in an interactive window, re-run this line to preview progress

# %%
# Panel b -- effect size (paired mean difference + bootstrap 95% CI)
ci_lo, ci_hi, obs_mean = _panel_b(ax_b, diffs)
print(f"Mean difference (B - A) = {obs_mean:+.2f} pp, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}] (percentile bootstrap, {N_BOOT} resamples)")
fig

# %%
# Save
fig.savefig(f"{OUT_PREFIX}.pdf")
fig.savefig(f"{OUT_PREFIX}.png", dpi=300)
print(f"Saved: {OUT_PREFIX}.pdf, {OUT_PREFIX}.png")

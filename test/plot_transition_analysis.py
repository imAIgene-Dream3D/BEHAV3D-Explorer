from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scanpy as sc

from behav3d.analysis.behavior.track.utils import (
    _resolve_track_paths,
    get_dtaidistance_track_trajectories_filename,
)
from behav3d.analysis.behavior.track.visualization.plots.transition_analysis import (
    save_window_cluster_transition_analysis,
)

# NOTE: this is a throwaway diagnostic script for visually iterating on the circular
# inter-cluster transition diagram + transition matrix before it gets wired into the napari
# panel. Re-run it (`python test/plot_transition_analysis.py`) after tweaking the constants
# below to regenerate the PDF.


# %%
# Interactive configuration
INTERACTIVE_OUTPUT_DIR = "/Users/s.deblank-3/Documents/LowDensity_MultiColor"
INTERACTIVE_CELL_TYPE = "tcells_merged"

# Transitions below this row-normalized probability are dropped entirely, not just faded -
# see `_draw_circular_transition_diagram_on_ax`'s docstring for why this needs to be a hard cut
# rather than a floor.
MIN_PROB_TO_DRAW = 0.01

# >1 bends the low end of the remaining probability range down hard (a squared curve at 2.0),
# so a transition just above MIN_PROB_TO_DRAW still renders as a thin, faint hairline and only
# the dominant transitions read as bold arcs. Raise it further if rare transitions still look
# too prominent; lower it toward 1.0 for a more linear (less aggressively de-emphasized) look.
EMPHASIS_GAMMA = 2

# Signed curvature of every arc (matplotlib's `connectionstyle="arc3,rad=<curvature>"`) - same
# sign for every edge so arcs consistently bow the same way ("flower" look). Try a smaller
# magnitude for straighter arcs, or flip the sign to bow the other way.
CURVATURE = 0.4

# "on_node" (default): full cluster name written inside each dot. "legend": dots show a short
# numbered index instead, with the index-to-name mapping moved to a legend beside the plot -
# useful once cluster names are long or there are many clusters and in-dot text gets cramped.
LABEL_STYLE = "on_node"


def load_track_adata(output_dir: str, cell_type: str):
    """Resolve and load the trajectory-clustering h5ad the same way the napari panel does."""
    paths = _resolve_track_paths(output_dir, cell_type)
    h5ad_path = paths.outfolder / get_dtaidistance_track_trajectories_filename(cell_type)
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"No trajectory-clustering h5ad at '{h5ad_path}'. Run track (trajectory) clustering "
            "first, with 'Divide long tracks' enabled so 'trajectory_window_id' exists."
        )
    return sc.read_h5ad(h5ad_path)


def main():
    adata_tracks = load_track_adata(INTERACTIVE_OUTPUT_DIR, INTERACTIVE_CELL_TYPE)

    result = save_window_cluster_transition_analysis(
        adata_tracks,
        output_dir=INTERACTIVE_OUTPUT_DIR,
        cell_type=INTERACTIVE_CELL_TYPE,
        min_prob_to_draw=MIN_PROB_TO_DRAW,
        emphasis_gamma=EMPHASIS_GAMMA,
        curvature=CURVATURE,
        label_style=LABEL_STYLE,
        verbose=True,
    )
    print(f"[transition-analysis] saved {result['transition_analysis_pdf']}")


if __name__ == "__main__":
    main()

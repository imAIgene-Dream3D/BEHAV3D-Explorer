# Plugin Essentials

Some parts of the BEHAV3D EXPLORER plugin do not belong to a single pipeline step: you use them at every stage. This section documents those cross-tab tools and behaviours once; the pages of the individual steps link here.

| Page | What it covers |
|---|---|
| [Visualization](visualization.md) | The 👁 Visualization tab: load any sample (raw image, segments, tracked segments, tracks) into napari layers, toggle them, and start manual editing. Used after every step to inspect outputs. |
| [Running Steps](running_steps.md) | What every step has in common: background execution and tab locking, the **Overwrite / Skip / Customize…** dialog for existing results, the shared **Workers** setting and the µm/px and frames/hours unit switches. |
| [Processing Queue](processing_queue.md) | The 🛒 panel at the bottom of the dock widget. Batches steps from all tabs and runs them one after the other across all samples. |
| [Results panel](results_panel.md) | The 📄 Results panel at the bottom of the Feature Extraction, Filtering and Analysis tabs: browse and open every result file. |
| [Output Directory & File Layout](output_layout.md) | The folder tree every tab writes to, and what the project settings file stores. |
| [Moving Projects](moving_projects.md) | Pack a project into a few verified archives to move it to another drive or computer, and how paths are re-linked after a move. |

```{toctree}
:hidden:
:maxdepth: 1

visualization
running_steps
processing_queue
results_panel
output_layout
moving_projects
```

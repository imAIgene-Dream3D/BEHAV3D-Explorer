# 🎯 Active Killing

Where Death Dynamics asks how a population's signal rises overall, **Active Killing attributes that rise to individual effector cells** — it flags the timepoints at which a target the effector is touching shows a signal increase large enough to count as a killing event.

```{important}
**Active Killing is configured and run in the [Feature Extraction](../feature_extraction.md) tab**, in the collapsible **▶ 🧬  Extended Analysis — Active Killing (Immune Cells)** section below the batch button. The section is closed by default and only appears when the metadata has at least one **immune** cell type. Run Feature Extraction first for the immune type and the targets, with **Contact** and the death features (dead channel) on.
```

```{note}
**The effector must be declared as an `im_` (immune) cell type.** The effector dropdown lists immune types only, so an `ot_` (other) population cannot be selected as a killer — although it *can* be selected as a target, and it *can* act as an effector in Interaction Analysis.

Nothing in the calculation depends on the population being immune. The prefix is a **processing category, not a biological claim**: if your killer is a neutrophil, a non-immune cytotoxic line, a parasite, or anything else that engages and damages another object, declare it with the `im_` prefix and Active Killing works normally.
```

## How it works in plain terms

First the detector finds **contact events**: continuous stretches where the effector is touching at least one of the selected targets. An event has to last at least the **minimum contact duration** to be considered at all.

Then, **at every timepoint of that contact** — not just once at the start — it asks the same question of every target the effector is touching:

> *From this timepoint to N timepoints later, does the signal on this specific target rise enough to count as killing?*

Each touched target is judged only against **its own** signal — there is no sample-wide or cross-target averaging. At each timepoint, whichever touched target clears its own threshold by the largest margin is reported as that timepoint's target.

Because every contact timepoint is scored independently with its own forward-looking window, **one contact can be flagged as killing at some timepoints and not others** — the verdict is per timepoint, not a single label stamped across the whole event.

### The killing rule (how it is computed)

Fix one touched target and one contact timepoint $t$. Let $W$ be the observation window and $D$ the chosen death signal column. The detector reads that target's own signal at $t$ and again $W$ frames later:

$$
D_{\text{start}} = D(t)
\qquad
D_{\text{end}} = D(t + W)
\qquad
\Delta D = D_{\text{end}} - D_{\text{start}}
$$

By default, $D$ is read as its **running maximum** up to that timepoint on the target's own track. Segmentation is sometimes unstable — a dead-mask segment can be included in one frame and dropped in the next — so the raw signal can dip and recover instead of only rising as the cell dies (e.g. raw values `0, 10, 20, 200, 150, 100, 200, 200, 250` are read as `0, 10, 20, 200, 200, 200, 200, 200, 250`). This is the **Persistent death signal (running max)** option; turning it off reads the raw signal.

If the target's track ends before $t + W$, $D_{\text{end}}$ is read at its last available timepoint (no extrapolation). The rise $\Delta D$ is compared to a threshold $\theta$ set by the mode:

$$
\theta =
\begin{cases}
\theta_{\text{abs}} & \text{(absolute mode)} \\[4pt]
D_{\text{start}} \times (m - 1) & \text{(multiplier mode)}
\end{cases}
$$

Multiplier mode therefore requires $D_{\text{end}} > D_{\text{start}} \times m$ — the signal must grow past $m\times$ its value **at that timepoint**. A start value of exactly $0$ is replaced by $0.1$ so the threshold isn't trivially zero. The target is flagged as killed at $t$ when

$$
\Delta D > \theta
\qquad\text{with}\qquad
\text{killing\_efficiency} = \frac{\Delta D}{\theta}
$$

This is repeated for **every touched target at every timepoint of the contact**. At each timepoint the target with the **highest** `killing_efficiency` is reported as that timepoint's target (`targeted_track_id`, set only when it is actually flagged as killed), so the reported target can change from one timepoint to the next.

## Parameters

Values (except **Top-N killers to display:**) are saved to `behav3d_parameters.yml` when you run or queue the analysis, and restored next session. Defaults below are for a fresh project.

| Control | Default | Range | What it does |
|---|---|---|---|
| **Immune cell type:** | first immune type | dropdown | Which effector tracks to analyse. Immune (`im_`) types only — see the note above. |
| **Target cell types:** | all selected | organoid and other types | Targets to evaluate. Each selected target is analysed on its own; with several selected, a **combined** analysis over all of them is run as well. |
| **Observation window (tp):** | `5` | 1 – 100 timepoints | How many frames after **each contact timepoint** to look for the signal rise on each touched target. |
| **Death signal column:** | `percentage_dead_mask` | `percentage_dead_mask`, `mean_dead_dye`, `nr_dead_mask_pixels` | Which target column is read as the signal. If your `dead_channel` carries another reporter, this is that reporter's column. |
| **Killing threshold multiplier:** | `1.5` | 0.1 – 20 | Multiplier mode: the target's signal must exceed `signal at this timepoint × multiplier` by the end of the window. Disabled when **Use absolute threshold** is on. |
| **Use absolute threshold** | off | checkbox | Replaces the multiplier by the fixed **Absolute threshold**. With a death signal other than `nr_dead_mask_pixels`, a hint recommends switching to it: a flat pixel count is easier to reason about. |
| **Absolute threshold:** | `0` | follows the death signal column | Fixed minimum rise, in the death signal's units. `percentage_dead_mask` is a **fraction** (0 – 1), `nr_dead_mask_pixels` a pixel count, `mean_dead_dye` an intensity. Re-check it after changing the column. |
| **Min contact duration (tp):** | `1` | 1 – 50 timepoints | Minimum consecutive timepoints an effector must stay in contact with **any** selected target for the event to qualify (`1` = every contact counts). The touched target may change within an event. |
| **Contact column:** | `contact` | `contact`, `contact_on_distance` | What decides when a contact event starts and ends. `contact` = the segments touch (ignores the contact threshold); `contact_on_distance` = within the µm **contact threshold** set in Feature Extraction. Which touched target is evaluated is not affected. |
| **⚙ Advanced Configuration → Persistent death signal (running max)** | on | checkbox | Reads each target's death signal as a running maximum (see above). Off = raw signal. |
| **Viewer Preview — Top Active Killers → Top-N killers to display:** | `5` | 1 – 50 | Number of killers used by **👁 Load Top Killers in Viewer** and **🎞 Export Killing GIFs**. |

A status line under the selectors says whether the analysis can run (*"✓ Ready — using: …"*) or what is missing: no target selected, no feature table for the immune type, or no contact column of the chosen kind for the selected targets.

## Choosing the window and threshold

None of these values are universal — they depend on your **biology** and your **imaging cadence**.

- **Observation window** — how long, after contact, you expect the signal to register. **Set it as a duration, then convert**: decide how long the event plausibly takes in minutes, divide by your `time_interval`, and enter that many timepoints. A 5-timepoint window is 10 minutes at a 2-minute interval but 2.5 minutes at 30 seconds. Try a few lengths around your estimate.
- **Minimum contact duration** — is a single-timepoint touch enough for killing to plausibly begin, or must contact persist to count as a real attempt? Judge against your interval.
- **Absolute threshold vs. multiplier** — the **absolute** threshold (a fixed increase, e.g. +20–30 dead pixels) is the general recommendation, pairs naturally with `nr_dead_mask_pixels`, and is best when targets start near zero. The **multiplier** (fold-change over each target's own signal) suits a single target line, or wells where cell-to-cell baseline variation makes one fixed value unreliable. With **two target lines that have different baseline death rates the multiplier becomes unreliable** — the same fold-change is a very different absolute increase per line — so prefer the absolute threshold there.

```{tip}
**Calibrating an absolute pixel threshold to "one dead cell".** The right value depends on cell size and pixel resolution, so recompute it for your own setup. A cell of diameter $d$ imaged at resolution $r$ (µm/pixel) fills roughly

$$
\text{area} \approx \frac{\pi}{r^{2}} \left(\frac{d}{2}\right)^{2} \ \text{pixels}
$$

e.g. a ~10 µm cell at 1 µm/pixel is about $\pi \times 5^2 \approx 80$ pixels if the whole cell fills with dye. "One dead cell" could therefore mean anywhere from ~20 pixels (partial staining, smaller cells, coarser resolution) to ~80+ pixels (full-cell fill, larger cells, finer resolution). Use this as a starting point, then fine-tune by eye against a timepoint you trust.
```

## Running it

- **▶ Run Active Killing Analysis** — runs the detector for every sample, in the background. The Log reports the number of active-killing timepoints and their share of all contact timepoints; a pop-up **Active Killing Analysis Complete** then offers **Open Folder**.
- **+🛒** — adds the analysis with the current settings to the [Processing Queue](../../plugin_essentials/processing_queue.md).

```{warning}
Active Killing reads the **unfiltered** feature tables, and [Filtering](../filtering.md) reads Active Killing's table so the killing columns are carried into the filtered table. Hence:

- **Re-run Filtering after every Active Killing run.** The completion pop-up reminds you which cell types need it.
- **Re-run Active Killing after every new Feature Extraction or 🔄  Re-run death** for the immune type — otherwise Filtering stops with an error for that cell type.
```

## Checking the results in the viewer

Both buttons use the most recent Active Killing results of the selected immune type and need at least one run.

- **👁 Load Top Killers in Viewer** — ranks effectors by their number of active-killing timepoints (across all samples) and adds one coloured Points layer per top-N killer, named `[Active Killing] <immune> #<track> (<sample>, <n> events)`, with a point at every timepoint it was flagged as killing. Scrub through time to verify visually; earlier `[Active Killing]` layers are replaced.
- **🎞 Export Killing GIFs** — for the top-N killers **of each sample**, writes an animated GIF of the killer's strongest event: a cropped maximum-projection movie (raw signal in grey, dead mask in red, the killer in purple, timepoint label), running for twice the observation window from the event.

## 🎯 Active Killing outputs

Written under `analysis/<immune_type>/active_killing/<target>/` for each selected target, and `analysis/<immune_type>/active_killing/combined/` for the combined analysis when several targets are selected:

| File | Contents |
|---|---|
| `BEHAV3D_<immune_type>_advanced_track_features.csv` | The full effector feature table with extra columns: `is_active_killing`, `killing_efficiency`, `targeted_track_id`, `contact_event_id`, and `death_signal_increase_<N>tp` where N is your observation window. This is the table Filtering picks up. |
| `active_killing_per_timepoint_<immune_type>.csv` | One row per contact-event timepoint: the signal rise, the threshold used, efficiency, verdict and reported target. |
| `active_killing_summary_<immune_type>.csv` | Per sample: contact timepoints, active-killing timepoints, mean / median signal rise, mean killing efficiency, number of contact events, and the active-killing rate. |
| `contact_events_<immune_type>.csv` | One row per contact event: start / end timepoint, duration, touched target track IDs. |
| `plots/<sample>/killing_kinetics_summary_<sample>.png` | Four panels per sample: killing-efficiency distribution, killing events per timepoint (raw and smoothed), cumulative killing events, and number of killing events per effector. |
| `plots/combined_killing_efficiency_distribution.png` | Histogram of killing efficiency across all active-killing timepoints. |
| `gallery/<sample>/killing_event_<sample>_T<track>_start<t>.gif` | Written by **🎞 Export Killing GIFs** (in the `<target>` folder, or `combined` when several targets are selected). |

The per-timepoint and contact-event tables also feed the active-killing dashboard of the [Interaction Overview](interaction_analysis.md).

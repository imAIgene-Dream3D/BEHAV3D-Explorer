# Deriving settings for a new dataset

BEHAV3D settings should be derived from the current acquisition and previewed. Cell
names and values from earlier experiments are not presets, and the plugin's defaults
are starting points, not recommendations.

## Measurements to collect

Before choosing values, collect:

1. XY pixel size and Z spacing.
2. Time between consecutive frames.
3. Approximate object diameter, or number of cell widths across a multicellular structure.
4. Fastest plausible displacement between two consecutive frames.
5. Whether consecutive masks overlap, touch, merge, divide, enter, or leave the field.
6. Whether each signal is constitutive, switch-on, fluctuating, or intermittent.

Whenever a control uses frames, also work out the corresponding physical duration.

## Segmentation scale

For a single cell, 10 micrometres is a transparent initial diameter assumption only
when you have no measurement. For a multicellular structure, first estimate how many
approximately 10 micrometre cell widths span it. A 10-30-cell range can be used as a
sanity check, not as an automatic answer.

When the metadata has pixel sizes, **EDT threshold** and **Min size** (and the
Cellpose-SAM size filter) are shown in µm and µm³, so physical values can be typed in
directly. To enter pixels or voxels instead, first set **Units:** to **px**
([Units switches](plugin_essentials/running_steps.md#units-switches)).

For **EDT threshold**, preview 20%, 25%, and 30% of the diameter:

```text
EDT (µm) = 0.20 to 0.30 * diameter_micrometres
EDT (px) = EDT (µm) / XY_pixel_size
```

For a 3D **Min size** preview, estimate the full spherical volume and begin at 50% to
tolerate dim or incomplete masks:

```text
Min size (µm³)    = 0.5 * (4/3 * pi * (diameter_micrometres / 2)^3)
Min size (voxels) = Min size (µm³) / (XY_pixel_size^2 * Z_spacing)
```

Elongated, protrusive, and confluent objects do not fit a diameter model. Use a
permissive threshold, annotate borders/background, and inspect the segmentation. For
Cellpose-SAM, **👁 Preview segmentation** logs the smallest, median, and largest object
volume, which is a better basis for the size filter than a diameter estimate.

## Tracking method

Use frame-to-frame behavior, not the cell type. A fresh project pre-selects
Fragmentation Propagation for organoids, btrack for immune and LapTrack for other
types; change it when your data say otherwise ([Tracking](processing/tracking/index.md)).

| Observed behavior | Starting method |
|---|---|
| Detection loses overlap with its previous position | btrack |
| Detection overlaps and may fragment; no new objects appear during the movie | Fragmentation Propagation |
| Detection overlaps/touches and IDs must remain within connected regions, or new objects appear | Bounded Propagation |
| Physical object is static but its segmented reporter flickers | Reporter Propagation |

For btrack (distances in µm by default):

```text
Max search radius = fastest_measured_one_frame_displacement * (1 + margin)
```

Use a modest 10-25% margin. If you enable the global optimizer, set **Distance
threshold** and **Time threshold** from the largest spatial gap and missing-frame
duration that should reconnect. Tick **P_branch** for division; keep **P_init** and
**P_term** (on by default) for field-of-view entry and exit. Lower **Step size
(frames)** only after an out-of-memory error.

For a moving object with a flickering reporter, track a constitutive channel and use
the reporter as an intensity feature, or use permissive segmentation plus btrack.

## Feature extraction and analysis

Choose features from the research question:

- Movement for motility and directional behavior.
- Contact for strict touching (contact threshold `0` µm) or calibrated proximity.
- Morphology for shape/plasticity questions.
- Channel intensity for switch-on or fluctuating reporters.
- Invasiveness for surface engagement between selected objects.

Set the **Dead mask % threshold:** by eye with **👁  Preview Dead Threshold in Viewer**
at a timepoint you trust ([Feature Extraction](analysis/feature_extraction.md#preview-dead-threshold-in-viewer)).

Death Dynamics measures the population fraction with a switch-on signal over time.
Active Killing measures a contact-associated signal rise in a selected target; set its
observation window as a duration and convert it to timepoints
([Active Killing](analysis/population_analysis/active_killing.md#choosing-the-window-and-threshold)).
These names do not prove the signal represents death; keep your own biological
definition.

Behavioral State can classify any single-cell population. Decide whether states should
represent movement, contact, morphology, channel intensity, or a combination. Derive
**Window size (timepoints):** and **Smooth window (timepoints):** from event duration and
acquisition cadence (the real time is shown below each) rather than keeping the bare
default of `5` frames.

## Filtering

Inspect the track-length distribution (**📊 See Track Length Distributions**) before
choosing **Min length**. A minimum removes shorter tracks. **Trim tracks to maximum
length** is on by default (`30`, splitting longer tracks into 30-timepoint pieces):
untick it unless you need tracks of equal length; when you do, equal minimum and
maximum values are valid. Filtering must still run with all filters disabled because
it creates the table every analysis reads.

For how many cells survive a given minimum length, use the QueenB **Cell counts** quick
action, which counts from the current feature table, rather than estimating
([Filtering](analysis/filtering.md#preview-track-lengths)).

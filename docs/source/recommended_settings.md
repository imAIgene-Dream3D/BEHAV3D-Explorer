# Deriving settings for a new dataset

BEHAV3D EXPLORER settings should be derived from the current acquisition and previewed. Cell
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

$$
\begin{aligned}
\text{EDT}\,(\text{µm}) &= (0.20 \text{ to } 0.30) \times d \\[4pt]
\text{EDT}\,(\text{px}) &= \frac{\text{EDT}\,(\text{µm})}{\text{XY pixel size}}
\end{aligned}
$$

where $d$ is the cell diameter in µm, and the pixel size and Z spacing (in µm) are those of your metadata.
Values up to 1 px split nothing. APOC and ConvPaint measure this distance in 3-D and
count one Z step as one pixel, so there the EDT threshold (in px) must also stay below
about half the number of Z slices a cell spans, $d / (2 \times \text{Z spacing})$;
otherwise no seed is left and nothing is split
([Instance post-processing parameters](processing/segmentation/index.md#instance-post-processing-parameters)).

For a 3D **Min size** preview, estimate the full spherical volume and begin at 50% to
tolerate dim or incomplete masks:

$$
\begin{aligned}
\text{Min size}\,(\text{µm}^3) &= 0.5 \times \frac{4}{3}\,\pi \left(\frac{d}{2}\right)^{3} \\[4pt]
\text{Min size}\,(\text{voxels}) &= \frac{\text{Min size}\,(\text{µm}^3)}{(\text{XY pixel size})^{2} \times \text{Z spacing}}
\end{aligned}
$$

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
| Detection loses overlap with its previous position | btrack (LapTrack and TrackPy are alternatives) |
| Detection overlaps and may fragment; no new objects appear during the movie (objects appearing later get no ID) | Fragmentation Propagation |
| Detection overlaps/touches and IDs must remain within connected regions, or new objects appear (they get new IDs) | Bounded Propagation |
| Physical object is static but its segmented reporter flickers | Reporter Propagation |

For btrack (distances in µm by default):

$$
\text{Max search radius} = \text{fastest one-frame displacement} \times (1 + \text{margin})
$$

Use a modest 10-25% margin. If you enable the global optimizer, set **Distance
threshold** and **Time threshold** from the largest spatial gap and missing-frame
duration that should reconnect. Tick **P_branch** for division (see
[how divisions appear in the output](processing/tracking/btrack.md#tuning-the-parameters));
keep **P_init** and **P_term** (on by default) for field-of-view entry and exit.

For a moving object with a flickering reporter, track a constitutive channel and use
the reporter as an intensity feature, or use permissive segmentation plus btrack.

## Feature extraction and analysis

Contact and intensity (and, for immune and other types, movement) are always extracted
([feature families](analysis/feature_extraction.md#the-six-feature-families)). Choose
what to add, and what to analyse, from the research question:

- Movement for motility and directional behavior (optional only for organoids).
- Contact: `*_contact` flags touching masks; `*_contact_on_distance` uses the contact
  threshold, which at the default `0` µm counts only overlapping masks. Raise it (in µm)
  for a calibrated proximity ([Contact](analysis/feature_extraction.md#contact)).
- Morphology for shape/plasticity questions.
- Channel intensity for switch-on or fluctuating reporters.
- Invasiveness (immune types only) for how much of an effector cell lies inside or right
  around an organoid.

Set the **Dead mask % threshold:** by eye with **👁  Preview Dead Threshold in Viewer**
at a timepoint you trust ([Feature Extraction](analysis/feature_extraction.md#preview-dead-threshold-in-viewer)).

Death Dynamics measures the population fraction with a switch-on signal over time.
Active Killing measures a contact-associated signal rise in a selected target; set its
observation window as a duration and convert it to timepoints
([Active Killing](analysis/population_analysis/active_killing.md#choosing-the-window-and-threshold)).
These names do not prove the signal represents death; keep your own biological
definition.

For Behavioral State and State Trajectory, see
[Behavioral state classification](#behavioral-state-classification) and
[Track classification](#track-classification-state-trajectory) below.

## Filtering

Inspect the track-length distribution (**📊 See Track Length Distributions**) before
choosing **Min length**. **Trim tracks to maximum length** is on by default (`30`,
splitting longer tracks into 30-timepoint pieces): untick it unless you need tracks of
equal length; when you do, equal minimum and maximum values are valid. Run Filtering
even with all filters disabled: Population Dynamics reads only the filtered table, and
the other analyses use it whenever it exists.

For how many cells survive a given minimum length, use the QueenB **Cell counts** quick
action, which counts them from the unfiltered Feature Extraction table, rather than
estimating ([Filtering](analysis/filtering.md#preview-track-lengths)).

## Behavioral state classification

Behavioral State can classify any immune or other population (organoid types are
analysed in Population Dynamics). States are only as meaningful as the features they are
fitted on, so start from features your data can actually measure
([Behavioral State](analysis/single_cell/state_classification.md)):

- **Movement** (`speed`, and the window feature `net_displacement`; both defaults): the
  basis for motile cells. Needs a frame interval short enough for reliable tracking;
  otherwise speed mostly reflects noise. Add `straightness` for directional questions.
- **Morphology** (`elongation`, `sphericity`, `extent`, `solidity`; defaults): only when
  the XY and Z resolution capture the cell's shape. A cell spanning only a few pixels or
  Z slices gives noisy shape values; then untick these and cluster on movement alone.
- **Intensity**: add a reporter channel when it defines the state (e.g. activation or
  calcium). For single-frame events, keep the window and smoothing short.
- **Binary groups**: `dead` (if death is measured) and `organoid_contact` (to follow
  engagement). They are not fitted but split each state afterwards, so every extra flag
  multiplies the number of states to curate.

Start with about `4` states and refine from there
([Choose a number of states](analysis/single_cell/state_classification.md#choose-a-number-of-states)).

## Track classification (State Trajectory)

Choose the basis from what should make two tracks alike
([Choosing a clustering basis](analysis/single_cell/track_classification.md#choosing-a-clustering-basis)):

- **State-based** (once behavioural states are curated): groups tracks by the order and
  proportion of states they pass through. It inherits the feature choices above, so the
  clusters are easy to interpret.
- **Feature-based**: when no behavioural states exist, or for a quick first look over
  full tracks. The defaults `speed`, `net_displacement_tracklet`, `percentage_dead_mask`
  and `any_organoid_contact` suit a killing assay; drop death or contact features when
  the dataset has no death channel or no organoids, and the same resolution caveat for
  morphology applies.

**Trajectory size (timepoints)** must fit your filtered tracks: shorter tracks are
discarded, and [Filtering](#filtering) trims tracks to `30` timepoints by default.
Tick **Divide long tracks** to split longer tracks into consecutive windows of
*Trajectory size* instead of trimming them, so their later timepoints are clustered too
rather than thrown away.

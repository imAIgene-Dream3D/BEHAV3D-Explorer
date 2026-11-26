# btrack: Bayesian Cell Tracking in BEHAV3D

btrack predicts each object's motion with a **Kalman filter** (Step 1) and then uses a **global hypothesis optimizer** (Step 2) to join track pieces, bridge gaps and end tracks. BEHAV3D runs btrack on physical coordinates, so **every distance is in µm and every variance in µm²**, whatever the pixel size.

## Presets

Preset files live in this folder. Each tuned preset is a btrack JSON plus a `"BEHAV3D"` block (label, description, panel values) that btrack ignores and the GUI reads.

| Preset | Use when |
|---|---|
| `tcell_tuned` | Default. Motile cells imaged every ~2–6 min with a reasonable segmentation. |
| `tcell_full_optimum` | Same situations; also uses settings that are not shown in the panel (diagonal process noise, z noise). |
| `tcell_dense` | Crowded fields, or segmentation that often misses or splits cells. |
| `tcell_long_interval` | Frame interval around 8–10 min, or fast cells. |
| `tcell_noisy_segmentation` | Tuned for noisy centroids; kept for reference. |
| `cell`, `particle` | btrack's untuned example settings, kept so old projects still load. |

The panel's **Suggest…** button measures how far cells move and how crowded they are and picks a preset for you.

## Parameters

Select a preset (or a custom JSON) and the panel fields are filled from the file. Edits never change the preset file; **Export JSON…** writes the current values to a new file.

| Field | JSON target | Unit | Notes |
|---|---|---|---|
| Max search radius | runtime option | µm | Longest single-frame jump that may be linked. |
| Assignment tolerance | `MotionModel.accuracy` | µm | Must be above 0 and at most 10; larger values crash btrack. Usable window is roughly 5–10 µm. |
| New-track position uncertainty | `MotionModel.P` (position part × `sigma`) | µm² | How uncertain a newly started track is. |
| Lost-track probability | `MotionModel.prob_not_assign` | – | Prior that a track has no detection in the next frame. |
| Enable optimizer | runtime option | – | Step 2; recommended. |
| Gap-bridging distance | `HypothesisModel.dist_thresh` | µm | Track pieces farther apart are never joined. |
| Gap-bridging frames | `HypothesisModel.time_thresh` | frames | Longest disappearance that can be bridged. |
| Link tolerance | `HypothesisModel.lambda_link` | µm | How quickly the chance of joining two pieces falls with distance. |
| Optimizer time limit (advanced) | runtime option | s | Stops the solver; the best solution so far is used. |
| No limit (advanced, `optimizer_no_time_limit`) | runtime option | — | The solver runs until it finds the best solution; the time limit is ignored. |
| Measurement noise (advanced) | `MotionModel.R` | µm² | How much the filter trusts each centroid. |
| Track start/end penalty (advanced) | `HypothesisModel.lambda_dist` | µm | Larger values let tracks start and end mid-field. |
| Allow divisions (advanced) | `P_branch` in `hypotheses` | – | Only for dividing cells. |

Every other JSON field passes through unchanged. Frame intervals above about 10 min can reduce tracking accuracy whatever parameters you choose, because cells move too far between frames to be matched reliably.

Visual features, update method and step size are no longer in the panel. They are still accepted by the Python API for old scripts (visual features raise a deprecation warning).

## Custom JSON files

Any btrack legacy JSON works. Select *Custom JSON…*, browse to the file, and the panel reads the values from its `TrackerConfig`. Without a `"BEHAV3D"` block, the runtime fields (search radius, optimizer, time limit) start at the defaults of `tcell_tuned`.

```json
"MotionModel": {
  "accuracy": 8.0,
  "prob_not_assign": 0.1,
  "max_lost": 5
},
"HypothesisModel": {
  "hypotheses": ["P_FP", "P_init", "P_term", "P_link"],
  "dist_thresh": 40.0,
  "time_thresh": 5
}
```

`accuracy` above 10 and `max_lost` above 10 are rejected before tracking starts.

## Output

Each run produces two files per sample:

- `<sample>_tracks.csv`: BEHAV3D standard track table
  Columns: `TrackID`, `SegmentID`, `position_t`, `position_x/y/z`, `pixel_position_x/y/z`
- `<sample>_tracked.zarr`: Segmentation image with labels replaced by `TrackID`

## References

- btrack documentation: https://btrack.readthedocs.io
- Ulicna et al. (2021) *Automated deep lineage tree analysis using a Bayesian single cell tracking approach.* Frontiers in Computer Science.

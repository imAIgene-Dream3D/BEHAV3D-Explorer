# 🎯 Active Killing

Where Death Dynamics asks how a population's death signal rises overall, **Active Killing attributes individual death events to the effector cells that caused them** — and gives each death exactly one unit of killing credit, however many effectors were touching.

```{important}
**Active Killing is configured and run in the [Feature Extraction](../feature_extraction) tab**, not in the Analysis tab. Look for the collapsible **▶ Extended Analysis — Active Killing (Immune Cells)** section at the bottom of that tab. It appears only when the metadata contains at least one **immune** cell type.

It needs: the effector's combined feature CSV **with contact features**, the annotated **dead mask** (`dead_mask_path`), and the **tracked label images** of the effector and the target types.
```

```{note}
**The effector must be declared as an `im_` (immune) cell type.** The effector dropdown is built from immune types only, so an `ot_` (other) population cannot be selected as a killer — although it *can* be selected as a target. The prefix is a **processing category, not a biological claim**: any population that engages and damages another object can be declared with `im_`.
```

```{warning}
**Results from before the death-event rework are not comparable with current results.** The previous algorithm credited every effector touching a dying organoid with an identical, full "active killing" verdict at every contact timepoint, so its counts grew with effector density. Outputs of the previous algorithm are detected and refused (with a request to re-run) rather than silently mixed with new ones. See [What changed](#what-changed-from-the-previous-algorithm).
```

## How it works in plain terms

1. **Find death events.** A *death event* is the appearance of a **new**, connected patch in the dead mask inside one target (e.g. one organoid) — roughly one cell dying. It is detected from the dead mask alone; effector cells play no part except that their own dye is removed first. A patch that keeps growing is still **one** event; a patch that dims and comes back is still one event; death already present when a target first appears is not new death.
2. **Find the candidates.** An effector is a candidate for a death event if it **touched that target** (Feature Extraction's Contact Threshold) **within the causal window before the death**, and its surface lies **within the attribution radius** of the death patch.
3. **Share one unit of credit.** The event's single unit of credit is shared among its candidates. A contact that is longer, more recent and closer to the patch earns a larger share. A death with no candidate is **unattributed** — counted separately as background death.

Because every death event carries exactly one unit of credit in total, **total kill credit = number of attributed death events**. Adding bystander effectors can never increase it: more effectors near a death only split the same unit more ways.

### Why contact alone earns nothing

| Gate | What fails it |
|---|---|
| **A death event must exist** — a new connected dead patch at least as large as the death threshold | An organoid that never visibly dies gives **zero** credit to every effector that ever touched it, however long. So do mask flicker, sub-cell specks and further growth of an existing patch. |
| **Timing** — the death starts within the causal window after a contact | Contact that begins after the death; contact that ended too long before it. |
| **Proximity** — the effector is within the attribution radius of *that* patch | Touching the opposite side of the same organoid. |
| **Sharing** — credit is a share of one unit | Several qualifying effectors split one credit instead of each getting one. |

### The credit formula

For a death event with onset at frame $t_0$ on target $L$, and a candidate effector $i$ with contact frames $t$ in the causal window:

$$
h_i = \frac{\Delta t}{c_{\text{ref}}} \sum_{t} \frac{1}{n_{\text{targets}}(i,t)}\; e^{-(t_0 - t)\,\Delta t / \tau}
\qquad
w_i = \min(h_i,\, h_{\text{cap}})\; e^{-d_i/\lambda}
\qquad
\text{credit}_i = \frac{w_i}{\sum_j w_j}
$$

- $h_i$ counts **decay-weighted hit-equivalents**: each contact frame counts less the longer before the death it happened ($\tau = 81.8$ min, from the ~57 min half-life of sublethal damage reported for cytotoxic T cells), divided among the targets the effector was touching at that moment (a cell touching three targets cannot deliver a full hit to each). $c_{\text{ref}} = 15$ min (a typical single contact) and $h_{\text{cap}} = 3$ (hits that saturate a lethal dose).
- $d_i$ is the **surface-to-surface distance** (µm, anisotropy-correct) between the effector and the death patch, with the patch carried along by the target's own movement between the contact and the death. $\lambda$ = attribution radius / 3.
- Credits of one death sum to exactly 1.

The credit is placed on the effector's **last contact frame before the death**, which is always a real row of its track and a timepoint at which it was touching the target.

## Parameters

Active Killing has **three** analysis settings. Everything else is derived from them, from the voxel size, or is a published constant.

| Control | Default | Meaning |
|---|---|---|
| **Immune cell type** | first immune type | Which effector tracks to analyse. |
| **Target cell types** | all | Targets whose death events are attributed. Several targets produce one run per target plus a pooled `combined` run. |
| **Target cell diameter** | 10 µm | Diameter of **one target cell** (not the organoid). Sets the **death threshold**: a new dead patch must reach a quarter of one cell's volume, $0.25 \cdot \tfrac{4}{3}\pi (d/2)^3$ (≈131 µm³ for 10 µm), because dye may fill a dying cell only partly. Also sets how close two dead fragments must be to count as the same dying cell. |
| **= min death patch** | derived | The same threshold in µm³. Filled in from the diameter; type a value to set the threshold directly. The live hint shows the voxel equivalent and warns below ~20 voxels (weaker than mask noise) or above ~5000 (only catastrophic death is seen). |
| **Causal window** | 120 min | How long after a contact a death can still be credited to it. ~120 min for organoid / carcinoma targets (contact-to-apoptosis lags of 1.8 ± 1.5 h have been reported for melanoma), ~30 min for haematologic targets (5–25 min). Given in **minutes**; the hint shows the frame equivalent for your imaging interval. |
| **Attribution radius** | 15 µm | Maximum surface-to-surface distance between an effector and a death patch for the effector to be a candidate. Can be calibrated from the data (see [Validation](#validation-and-calibration)). |
| **Top-N killers** | 5 | Shared by the viewer points, the killing GIFs and the swimmer plot, so all three show the same effectors. |

```{tip}
**Choosing the target cell diameter.** Press **🔍 Preview death patches (current frame)** on a frame you trust. It overlays the candidate new-death patches that pass the threshold and logs every candidate's volume. The accepted patches should be cells you agree are dying: lower the diameter if real deaths are missed, raise it if specks are counted. (The preview compares with the previous frame only; the analysis uses the whole history, so a patch still growing may show up here.)
```

The **Contact Threshold** of Feature Extraction still decides which targets an effector is touching. Candidates must satisfy it **and** the attribution radius, so the stricter of the two limits who can be credited — the panel shows a hint when they differ by more than 2×. A Contact Threshold of 0 means the effector and target masks must overlap, i.e. strict touching.

### What changed from the previous algorithm

| Removed setting | What replaced it |
|---|---|
| Absolute killing threshold | The **death threshold** (min death patch, from the target cell diameter). The same kind of gate — an absolute amount of new death — measured on one connected new patch instead of the whole organoid's dead-pixel count. |
| Killing threshold multiplier | Nothing is needed: counting only voxels that were not dead before removes each organoid's baseline death from the question, so there is no longer anything to take a fold change of. |
| Death signal column | Death is read from the annotated dead mask; there is no signal column to choose. |
| Observation window (timepoints) | The **causal window**, in minutes. |
| Minimum contact duration | A brief contact now earns a proportionally small share of credit instead of being cut off. |
| Contact column | Always the distance-based contact (the one that matches the µm attribution radius). |

### Advanced overrides

Every derived value and constant can be overridden in `behav3d_parameters.yml` under `active_killing.advanced`. They are **not** in the GUI on purpose; any override is recorded in `active_killing_run_params.json`, and changing one breaks comparability with datasets analysed with the defaults.

| Key | Default | Source |
|---|---|---|
| `min_patch_volume_um3` | $0.25\cdot\tfrac{4}{3}\pi(d/2)^3$ | derived from the diameter (the GUI field writes this when set directly) |
| `patch_closing_um` | $d/6$ | bridges fragments of one dying cell before labelling |
| `patch_link_radius_um` | $d/3$ | new dead voxels this close to an open patch are its growth, not a new event |
| `novelty_tolerance_um` | max(1.5 µm, voxel diagonal) | absorbs mask jitter / sub-voxel motion |
| `persistence_frames` | 2 | a voxel must stay dead into the next frame (flicker never counts) |
| `nucleation_grace_frames` | 2 | a sub-threshold seed may grow past the threshold within this many frames (onset back-dated) |
| `patch_idle_close_min` | 30 | a patch that stopped growing stops absorbing neighbours after this long |
| `bad_frame_drop_fraction` | 0.8 | a frame whose dead volume drops by this much is treated as a mask failure |
| `damage_tau_min` | 81.8 | sublethal-damage half-life 56.7 min / ln 2 |
| `contact_ref_min` | 15 | typical single contact |
| `hit_cap` | 3 | hits saturating a lethal dose |
| `spatial_kernel_um` | radius / 3 | distance weighting of credit |
| `min_lag_min` | 0 | minimum time between contact and death |
| `exclude_border_patches` | true | patches cut by the lateral (x/y) image border have unknown volume |

## Outputs and plots

Written under `<output_dir>/analysis/<immune_type>/active_killing/<target or combined>/`:

| File | Contents |
|---|---|
| `kill_candidates_<immune>.csv` | One row per (death event, candidate effector): distance, lag, contact time, `kill_credit` (sums to 1 per attributed death), `is_nearest_effector`, `n_candidates`. |
| `death_events_<immune>.csv` | Every death event: onset, position, volume, depth below the organoid surface, and its **attribution class** (`exclusive`, `shared`, `unattributed`, `unattributed_truncated` when its causal window starts before the movie, `excluded_border`). |
| `per_effector_killing_<immune>.csv` | Per effector: `kills_attributed` (fractional), `kills_nearest` and `kills_exclusive` (integers), `killing_rate_per_hour`, `hit_delivery_rate`, `kill_per_contact_hour`, `kill_per_contact_event`, distinct targets killed, inter-kill interval, time to first kill, `is_serial_killer`. |
| `per_target_death_<immune>.csv` | Per target: death events, attributed / unattributed, attributed dead volume, contacts before the first death, contact-to-first-death lag. |
| `per_sample_killing_<immune>.csv` | Per sample, led by **`conversion_rate`** (attributed death events per contact event), plus `background_death_rate`, `attribution_ambiguity`, attributed fraction, killing Gini / top-10 % share, counts. |
| `contact_events_<immune>.csv` | One row per continuous (effector, target) contact event. |
| `BEHAV3D_<immune>_advanced_track_features.csv` | The effector feature table plus per-timepoint kill columns: `kill_credit`, `cum_kill_credit`, `is_active_killing` (true on credited timepoints), `is_nearest_effector`, `death_event_id`, `targeted_track_id` (`-1` when none), `distance_to_death_um`, `lag_min`, `n_cokillers`, `attributed_death_volume_um3`. Filtering uses this table as its input. |
| `active_killing_run_params.json` | Schema version and every resolved parameter. |
| `plots/` | The figures below, each with its backing CSV. |
| `gallery/<sample>/killing_event_*.gif` | **🎞 Export Killing GIFs**: each top killer's largest attributed death event — raw signal, dead mask in red, the attributed patch in yellow, the killer in purple. |

Death events themselves are cached per target under `<output_dir>/analysis/<target>/death_events/` and reused as long as the masks and detection settings are unchanged, so changing the causal window or radius is fast.

**Headline numbers.** `conversion_rate` — attributed deaths per contact event — is the number to compare between conditions. Individual contacts are usually not lethal, so values near 1 usually mean the thresholds are too loose. `background_death_rate` (unattributed deaths per organoid-hour) is a baseline you can subtract, and `attribution_ambiguity` (average of 1 − the largest share) says how much of the attribution was split between several effectors.

| Figure | How to read it |
|---|---|
| `death_event_fate_over_time` | Death events over time, attributed vs unattributed, and the attributed fraction. Immune-mediated deaths should rise after effector addition while background death stays flat. |
| `attributed_fraction_by_condition` | Immune-attributed fraction of death per condition, by events and by dead volume. A no-effector control arm is the empirical zero. |
| `attribution_funnel` | All death events → had an effector in contact within the causal window → and within the radius. Many events lost at the last step means the radius is tight; almost none having any contacting effector means the unattributed death is genuine background. |
| `patch_depth_attributed_vs_unattributed` | Depth of the death patch below the organoid surface. Immune killing should be superficial; core necrosis is deep. If attributed patches are not shallower, the analysis may be picking up necrosis. |
| `killing_concentration` | Lorenz curve and Gini of kill credit across engaged effectors (how concentrated killing is), and credit vs contact time (efficiency per contact hour). |
| `serial_killing_and_timing` | Inter-kill intervals, distinct targets killed, time to first kill (Kaplan–Meier, censored at track end) and the contact-to-death lag with the published melanoma band. |
| `organoid_swimmer_top<N>` | For the targets of the top-N killers: contact periods (blue bands) and death events (purple = attributed, grey = unattributed; size = volume). |
| `engagement_dose_response` | Attributed deaths vs the number of effectors and total contact per target. It should rise. |

The Interaction Overview dashboard (Population Dynamics) also reads these outputs; see [Interaction Analysis](interaction_analysis).

```{note}
**Two notions of "dead" coexist, on purpose.** The organoid *fate* used by Death Dynamics and the interaction dashboard (Live / Dying) is the whole-organoid call from `dead_mask_percentage_threshold`. Active Killing counts **localised** death events. An organoid can have attributed kills and still be "Live" because it never crossed the whole-organoid threshold; `organoid_dead_fraction_at_onset` on every death event shows the relationship.
```

## Validation and calibration

**📏 Calibrate radius & validate** (optional, slower; run after one Active Killing run) writes `validation/`:

- **Density invariance.** Effector tracks are randomly subsampled (25 / 50 / 75 / 100 %) and re-attributed, with the death events fixed. The number of death events and the credit per attributed death must not change, and the hit-delivery rate per effector should stay flat, while the previous algorithm's verdicts per death event grow with effector density. The attributed count is compared with its exact expectation $\sum_e 1 - (1-f)^{k_e}$. (`kills_exclusive` per effector is **not** density-invariant: with several effectors near a death, removing some makes the rest exclusive; treat it as a conservative count.)
- **Radius calibration.** Each death patch is rotated inside its organoid (keeping its size, depth and the effector positions) to measure how often an effector would be within a given radius by chance. For each radius, the empirical FDR is (chance + 1) / (observed + 1), and the largest radius with FDR ≤ 5 % is applied to the radius field. If no radius reaches it, nothing is applied and the log says so, reporting the radius with the lowest FDR so you can set it manually if that FDR is acceptable for your question. The default 15 µm is generous for small targets densely surrounded by effectors — the calibration curve shows how much of the attribution at a given radius could arise by chance. With few death events the estimator cannot go below 1 / (events + 1), so a strict threshold can be unreachable regardless of data quality — pool samples, or judge the curve.

## Limitations

1. **The dead mask is ground truth.** If the classifier bleeds at organoid edges, "new death" concentrates at the surface — exactly where effectors are — and looks attributable. A bias that is identical in every condition passes every check above. A no-effector control and visual review (GIFs, preview) are the defence.
2. **Where dye first appears is not necessarily where the synapse was.** Dye can appear first at the nucleus, and dying cells round up and move, so the attribution radius covers both real locality and this wander. The calibration is its principled anchor.
3. **Fractional credit is a model.** The decay and saturation constants encode the observation that most solid-tumour kills need several hits, often from different effectors. If a kill is really one stochastic hit, sharing smears a single-culprit signal across bystanders — which is why `kills_nearest` and `hit_delivery_rate` are always reported alongside.
4. **Unattributed is not proof of immune-independent death.** An unsegmented effector, a broken track or a contact just outside the window all end up there; the immune-attributed fraction is a lower bound.
5. **An organoid is not one cell.** A patch is a better unit than the whole organoid, but without nuclear segmentation the patch-to-cell mapping is unverified; volume-weighted numbers are more robust than counts.
6. **Rotation and deformation of the target are not compensated**, only its translation.
7. **Anisotropic z** makes the death threshold coarser along z; the hint warns when z/xy > 4.
8. **Death events are not independent** (those in one organoid share effectors); use per-sample and per-organoid random effects in downstream statistics.
9. **Few settings is a choice, not a guarantee.** The defaults carry the risk that tunable knobs would otherwise carry; overrides are recorded, and the validation report shows whether conclusions depend on them.

## Buttons

- **🔍 Preview death patches (current frame)** — see [the tip above](#parameters).
- **👁 Load Top Killers in Viewer** — the top-N killers as Points layers at their credited timepoints (size = credit).
- **🎞 Export Killing GIFs** — see Outputs.
- **📏 Calibrate radius & validate** — see Validation.
- **▶ Run Active Killing Analysis** / **+🛒** — run now, or queue with the current settings.

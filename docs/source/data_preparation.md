# 📋 Data Preparation

Tab 1 of the BEHAV3D EXPLORER dock widget. This is where every experiment starts. It owns four things that the rest of the plugin reads:

1. The **output directory**: everything BEHAV3D EXPLORER writes goes under it.
2. The **metadata.csv**: the single source of truth for samples, image paths, pixel sizes, time step and cell types.
3. The **dimension order** of each raw image (e.g. `TCZYX`).
4. The **Zarr-converted raw images** that the downstream tabs read for fast access.

The tab is a vertical list of six numbered sections, with a **Log** box at the bottom.

## 1 · Output Directory

```{important}
Set the output directory **before** saving or loading metadata. A new metadata.csv is saved into it, the project settings file is created in it, and every other tab reads and writes under it.
```

Type a path or click **Browse…** and pick a folder. The folder structure BEHAV3D EXPLORER creates inside it is described in [Output Directory & File Layout](plugin_essentials/output_layout.md). Short paths without special characters work best.

The **Transfer** button next to it opens a menu to pack the project into a few verified archives (to move it to another drive or computer), to check a copied bundle, or to import one. See [Moving Projects](plugin_essentials/moving_projects.md).

## 2 · Metadata Builder

The builder creates a metadata.csv from a form, so you never have to edit the CSV by hand. It is greyed out by default: tick the checkbox in the **2 · Metadata Builder** header to use it (unticking it again clears the form). If metadata is already loaded, BEHAV3D EXPLORER first asks whether to **Edit Current** (the form is filled with the loaded metadata) or **Generate New** (empty form).

```{note}
**Supported raw image formats**: `.czi`, `.lif` / `.liff`, `.tif` / `.tiff`, `.ims`, and already-converted `.zarr` folders.
```

### Workflow

1. **Number of samples:** (default `1`): how many samples (movies) the experiment has.
2. **Organoid types:**, **Immune types:**, **Other types:** (default `0` each): how many populations of each category your images contain. Tick **Include dead channel** if the acquisition has a separate live/dead (viability) dye channel; this adds a **Dead ch #** and a **Dead mask** field to every sample form.

   ```{important}
   **A population is defined by your images. Line and Condition are defined by your experiment.**

   A **population** is something you can tell apart in the imaging data, so it needs its own segmentation and its own track IDs. Count one for each such entity; that is all these numbers control.

   **Line** and **Condition** are free-text labels you fill in per sample. They are not visible in the image; only you know them. Use Line for the biological identity or source and Condition for the treatment or experimental state. Line is required, Condition is optional.

   | Situation | How to declare it |
   |---|---|
   | One immune population deliberately labelled in several colours, same biology throughout | **One** immune population, tick **Multicolor** and set the number of colours |
   | Two entities you can distinguish in the image, in different or overlapping channels | **Two** populations |
   | Visually identical, differing only by treatment, subclone or donor | **One** population; record the difference in **Line** or **Condition** |
   ```

3. Click **Configure Cell Types**. One name field appears per population, pre-filled with `organoid1`, `organoid2`…, `tcell` (first immune type), `immune2`…, and `other1`… Edit the names as you like. The category you counted the population under (organoid, immune or other) is saved as the prefix `or_`, `im_` or `ot_` and decides which options the population gets later — see [Population categories](#population-categories). Next to each immune name you can tick **Multicolor** and set how many colours (`2` populations by default, 2–10); see [Multicolor cell types](#multicolor-cell-types).
4. Click **Create Sample Forms**. One **Sample N** form appears per sample (fields below).
5. Fill in Sample 1, then click **Fill All from Sample 1** to copy every field except **Name** and **Image path** to the other samples. Adjust what differs.
6. Click **Save Metadata CSV**. The form is validated (all required fields filled, every image path exists) and saved as `metadata.csv` in the output directory; you are asked before an existing file is overwritten. The saved path is copied into **3 · Metadata Loader**.
7. Click **Load Metadata** in section 3 to read the image dimensions and load the project settings.

### Sample form fields

Fields marked `*` in the form are required.

| Field | Default | What to enter |
|---|---|---|
| **Name\*** | — | Unique sample name, used in every output file name. A purely numeric name (e.g. `12`) is renamed to `S_12`, because numeric names break parts of the pipeline. |
| **Exp #\*** | `1` | Experiment number. |
| **Well\*** | — | Plate well, e.g. `A1`. |
| **Image path\*** | — | Raw image. The **…** button asks whether you want to pick a file or a Zarr folder. |
| **Dim order** | empty | Optional axis order of the raw image (e.g. `TCZYX`). Usually left empty; it is detected automatically and can be corrected in section 5. |
| **Pixel XY (μm)\*** | `0.5` | Pixel size in X and Y. |
| **Pixel Z (μm)\*** | `2.0` | Z-step size. |
| **Dist unit\*** | `μm` | Unit of the two pixel sizes. |
| **Time interval\*** | `1.0` | Time between two frames. |
| **Time unit\*** | `s` | `s`, `m` or `h`. |
| **Dead ch #** | `0` | Only with **Include dead channel**. 0-based index of the dead-dye channel in the raw image (0 = first channel). |
| **Dead mask** | empty | Only with **Include dead channel**. Optional path to an existing dead mask; normally left empty and produced later in the Segmentation tab. |
| **Line\*** / **Condition** | — | One block per population (headed `── name ──`). |
| **Segments** / **Tracks img** / **Tracks csv** | empty | Optional paths to existing results for this population. Leave empty on a first run; the Segmentation and Tracking tabs fill them in. |

The pixel sizes and time interval are also what lets parameter panels in other tabs show values in µm or hours instead of pixels or frames (see [Running Steps](plugin_essentials/running_steps.md)).

### Editing an existing metadata.csv

Once metadata is loaded, the save button turns orange and reads **Edit Metadata CSV**. Click it (or tick the builder header) and choose **Edit Current**: the form is filled from the loaded metadata, the button becomes **Save Changes**, and saving overwrites the loaded CSV. Use this to add samples or populations later. The **➕ Add a new sample or cell type…** button in the Segmentation and Tracking import pages opens the builder directly in this mode.

## 3 · Metadata Loader

Type the path of a metadata.csv or click **Browse…**, then click **Load Metadata**. Use this whenever you come back to an existing project; you do not need the builder again.

Loading runs in the background and:

- checks that all [required columns](#required-columns-every-row-every-sample) are present and filled for every sample, and that every raw image exists;
- re-links paths automatically if the project folder was moved or copied (see [Moving Projects](plugin_essentials/moving_projects.md#paths-are-re-linked-automatically));
- creates or reads `behav3d_parameters.yml` in the output directory, which restores the settings you used last time in every tab;
- reads the shape and axis order of each raw image for section 5.

On success the line under the button reads e.g. `✅ Loaded 12 samples, 25 columns.  |  Organoid: organoid1  |  Immune: tcell`, and every other tab receives the metadata. On failure it shows `❌` followed by what is missing.

```{tip}
You do not need to reload after a Segmentation, Tracking or dead-mask run: those steps write the new file paths into metadata.csv and refresh all tabs themselves.
```

## 4 · Metadata Overview

A read-only table showing the first 10 rows of the loaded metadata, so you can check the paths and values at a glance.

## 5 · Dimension Order (Optional)

After loading, the table lists every sample with its **Shape** and **Dim Order**. The order is taken from the `dimension_order` column if filled, otherwise detected from the file (`TCZYX` if detection fails). If a sample reads in the wrong order, pick the right one from its drop-down: `TCZYX`, `TZCYX`, `ZCTYX`, `ZTCYX`, `CZTYX` or `CTZYX`.

- **Set all:** + **Apply to All** sets the same order for every sample.
- **Save to Metadata CSV** writes the orders into metadata.csv. Conversion (section 6) also saves them automatically before it starts.

Open one sample in the [Visualization tab](plugin_essentials/visualization.md) after conversion to confirm that channels, Z and time look right.

## 6 · Convert to Zarr

Converts every raw image to a Zarr image in a fixed `TCZYX` order, which every downstream tab reads.

| Control | Default | What it does |
|---|---|---|
| **Clip timepoints (select range)** | off | Keep only the frames from **Start:** to **End:** (0-based, both included) for every image. Use it to trim dead frames at the start or end, or to give all samples the same length. **End** must be greater than **Start**. |
| **Start:** / **End:** | `0` / `0` | Only editable when clipping is ticked. |
| **Workers:** | half of your CPU cores | Parallel processes for the conversion (up to cores − 1). This value is shared with the other tabs' **Workers** settings; see [Running Steps](plugin_essentials/running_steps.md#workers). |
| **Convert All Images to Zarr** | — | Starts the conversion in the background. |

A red warning appears above the controls when samples have different numbers of timepoints; clipping them to a common range is recommended, because downstream steps compare samples frame by frame.

What happens when you convert:

- Each image is written to `<output_dir>/images/<sample_name>/<sample_name>.zarr`. The original file is not modified.
- Samples whose Zarr already exists are **skipped**. To reconvert one, delete its `.zarr` folder first. If clipping is ticked and some samples are already converted, you are asked to **Clip Existing Zarr** (cut those files in place to the same range), **Leave As-Is**, or **Cancel**.
- metadata.csv is updated (`raw_image_path` now points to the Zarr, `dimension_order` becomes `TCZYX`) and all tabs are refreshed.
- The original shape, axis order and any timepoint cut of every sample are recorded in `behav3d_parameters.yml`, so they stay known even if the raw files are deleted.
- Finally BEHAV3D EXPLORER offers to delete the now redundant original files (**Free up disk space?**). Deleting needs a second confirmation (**Delete permanently**).

```{warning}
Deleted originals do **not** go to the Recycle Bin. If in doubt, decline: you can check a converted sample in the Visualization tab first and delete the originals by hand later.
```

---

## Metadata schema reference

BEHAV3D EXPLORER uses a **dynamic, prefixed** schema: it discovers the populations from the column names when the metadata is loaded. The prefix marks the **category** of a population:

| Prefix | Category in the GUI |
|---|---|
| `or_` | Organoid (multicellular structure) |
| `im_` | Immune (single motile cells) |
| `ot_` | Other |

You can have several populations per category, e.g. `or_organoid1`, `or_organoid2`, `im_tcell`. In output folders and file names the prefix is dropped (`organoid1`, `tcell`).

### Population categories

The category is a **processing choice, not a biological claim**. Pick the one whose options your population needs: for example, declare a non-immune killer line as immune (`im_`) to use Active Killing on it.

| Category | Typical use | What it changes in the plugin |
|---|---|---|
| Organoid (`or_`) | Multicellular structures (organoids, spheroids, tissue fragments) | Tracking: by default all organoid types are tracked together in one **Fragmentation Propagation** run (**🟣 All Organoids**). Feature Extraction: **Propagate dead signal** is on. Target in Death Dynamics, Interaction Analysis and Active Killing; the reference that Invasiveness is measured against. Organoid types are not listed one by one in Single Cell; only cell-type groups are. |
| Immune (`im_`) | Single motile effector cells (T cells, NK cells, macrophages…) | Tracking starts on **btrack**. **Movement** is always extracted. Dead-dye signal inside immune cells is ignored when the death of organoid and other populations is measured. The only category that can compute **Invasiveness**, be the effector in **Active Killing**, and be split into **Multicolor** populations. Effector in Interaction Analysis; analysed in Single Cell. |
| Other (`ot_`) | Any other population | Tracking starts on **LapTrack**. **Movement** is always extracted. Can be a target (Death Dynamics, Interaction Analysis, Active Killing) or an effector (Interaction Analysis); analysed in Single Cell. |

The starting methods and settings (including the Pixel Classifier's post-processing defaults, which differ per category) are only defaults: every population can use every segmentation and tracking method.

### Required columns (every row, every sample)

These 9 columns, plus one `{prefix}_{ct}_line_condition` column per population ([Per-population columns](#per-population-columns)), must be present and non-empty for every sample; loading or saving is refused otherwise. Each population's three path columns must also be present, but they may stay empty. The builder writes all of them for you.

| Column | Description | Example |
|---|---|---|
| `sample_name` | Unique sample identifier, used for output paths. | `Sample_A1` |
| `exp_nr` | Experiment number. | `1` |
| `well` | Plate well identifier. | `A1` |
| `raw_image_path` | Full path to the raw image (or, after conversion, to its Zarr). Forward slashes recommended, also on Windows. | `D:/data/exp1/sample_A1.czi` |
| `pixel_distance_xy` | Pixel size in the XY plane. | `0.5` |
| `pixel_distance_z` | Z-step size, same unit as `pixel_distance_xy`. | `2.0` |
| `distance_unit` | Unit of the two above. | `μm` |
| `time_interval` | Time between frames. | `1.0` |
| `time_unit` | `s`, `m`, or `h`. | `m` |

### Optional columns

| Column | When you need it | Notes |
|---|---|---|
| `dimension_order` | Optional | Axis order of the raw image (e.g. `TCZYX`). Normally set in section 5, not typed by hand. |
| `dead_channel` | Only if you have a dead-dye channel | 0-based channel index of the dead dye. Written when **Include dead channel** is ticked. |
| `dead_mask_path` | Only if `dead_channel` is set | Path to the dead mask. Empty on a first run; filled by the dead-mask step in the Segmentation tab. |

### Per-population columns

For every population (`{prefix}` = `or` / `im` / `ot`, `{ct}` = population name):

| Column | Required? | Description |
|---|---|---|
| `{prefix}_{ct}_line_condition` | **Yes**, filled | Your label for the sample (e.g. `lineA_untreated`), used to split plots and comparisons. You enter **Line** and **Condition** separately in the builder; they are joined with `_` when the CSV is saved. |
| `{prefix}_{ct}_segments_image_path` | Column yes, value optional | Segmentation result for this sample. |
| `{prefix}_{ct}_tracks_image_path` | Column yes, value optional | Tracked-labels image for this sample. |
| `{prefix}_{ct}_tracks_csv_path` | Column yes, value optional | Tracks table for this sample. |

```{note}
The three path columns (and `dead_mask_path`) are **blank on a first run**. When a Segmentation, Tracking or dead-mask run finishes, BEHAV3D EXPLORER writes the produced paths into metadata.csv for you.

To use segmentations or tracks made outside BEHAV3D EXPLORER, use the **Import** option of the [Segmentation](processing/segmentation/import.md) or [Tracking](processing/tracking/import.md) tab. It accepts TIFF or `.zarr` label images, checks them, converts or copies them into the Zarr layout every downstream step reads where needed (a valid segmentation Zarr is used where it is), and saves the paths. Typing a path into the CSV by hand is not enough.
```

### Which channel is used for segmentation?

The metadata does **not** record which channel holds which population. That choice is made later, in the **Segmentation** tab, and depends on the method. The **only** channel number stored in metadata.csv is `dead_channel`.

### Multicolor cell types

Multicolor is offered for **immune** (`im_`) populations only. Use it when the **same population was deliberately labelled with more than one fluorescent colour** in the same image. Biologically it is one population; only the acquisition label differs. Splitting a crowded population across colours makes each colour sparser, so segmentation separates cells more reliably and tracking has fewer ambiguous links. The colours are merged back after tracking, so you keep one population.

**How to declare it in the builder:**

1. After **Configure Cell Types**, tick **Multicolor** next to the immune name and set the number of colours (2–10). Count every label you can tell apart in the image, **combinations included**: two dyes used alone and together give three labels (dye A, dye B, A + B), so set `3`.
2. Click **Create Sample Forms**. The builder creates one entry per colour, named `{base}_1_multicolor`, `{base}_2_multicolor`, …, each with its own columns (`im_{base}_1_multicolor_line_condition`, `…_segments_image_path`, …).
3. **Line** and **Condition** are kept identical across the colours automatically: typing in one colour fills the others. The path fields stay separate.

**What happens downstream:**

- Segmentation and tracking treat each colour as its own population (one segmentation and one set of tracks per colour).
- A combination label (e.g. A + B) needs a method that can learn which channels must be bright together: the trainable pixel classifiers (APOC, ConvPaint, Pixel Classifier) see all channels and can be painted to tell double- from single-labelled cells. Cellpose and Cellpose-SAM segment every cell visible in the chosen channel(s), so they cannot separate A + B cells from A-only cells.
- Where objects of different colours overlap, an automatic clean-up after APOC and Pixel Classifier batches gives the shared voxels to the largest object; the others keep only their non-overlapping part, or are removed if that part is very thin (ConvPaint's classes never overlap). There is no clean-up after Cellpose, Cellpose-SAM or Import: colours that still overlap make the merge after tracking stop with a *Voxel overlap detected* error.
- After tracking, the colours are combined into one population named `{base}_merged`, and its tracked-image and tracks-table paths are added to metadata.csv twice: `im_{base}_merged_tracks_image_path` / `im_{base}_merged_tracks_csv_path`, and the same two names without the `im_` prefix.
- Feature Extraction, Filtering and Analysis work on `{base}_merged`, not on the individual colours.

```{tip}
If colours identify different populations, identities or treatments, declare them as separate populations instead, so they keep independent track IDs and features.
```

### Examples

**A minimal example (no multicolor)**: one organoid population and one immune population, each on a single channel.

| sample_name | exp_nr | well | raw_image_path | pixel_distance_xy | pixel_distance_z | distance_unit | time_interval | time_unit | or_organoid1_line_condition | im_tcell_line_condition |
|---|---|---|---|---|---|---|---|---|---|---|
| Sample_A1 | 1 | A1 | D:/data/exp1/A1.czi | 0.5 | 2.0 | μm | 1.0 | m | lineA_untreated | donor1_activated |
| Sample_A2 | 1 | A2 | D:/data/exp1/A2.czi | 0.5 | 2.0 | μm | 1.0 | m | lineA_treated | donor1_activated |

(Each population's three path columns are also in the file, empty because this example starts from raw images; they are left out here for space.)

**A multicolor example**: the same organoid population plus one T-cell population split across two colours, with a dead-dye channel. Only the columns that differ from the example above are shown:

| sample_name | … | dead_channel | or_organoid1_line_condition | im_tcell_1_multicolor_line_condition | im_tcell_2_multicolor_line_condition |
|---|---|---|---|---|---|
| Sample_A1 | … | 3 | lineA_untreated | donor1_activated | donor1_activated |
| Sample_A2 | … | 3 | lineA_treated | donor1_activated | donor1_activated |

After tracking, the `tcell_merged` path columns are added and the rest of the pipeline uses `tcell_merged`.

## Combining multiple experiments

There is no one-click "combine experiments" button, but you can merge several runs by hand and treat them as one batch:

1. **Segment and track each experiment separately first**, with whatever settings each one needs.
2. **Copy each experiment's `images/<sample>` and `trackdata/<sample>` folders into one shared output directory**, so all samples sit side by side.
3. **Merge the experiments' metadata.csv files into one** (one row per sample, sample names unique), save it in the shared directory and **Load Metadata** with that directory as output directory. Paths pointing to the old folders are re-linked to the copied files automatically.

You can then run **Feature Extraction**, **Filtering** and **Analysis** jointly across all samples.

## Tips & troubleshooting

- **Spaces and special characters in paths**: avoid them where possible.
- **"The raw_image_path … does not exist"** when saving or loading: a typo in an image path, or the drive is not connected. Open metadata.csv and check.
- **"Not all required columns are present"** or **"The following required fields have missing values"**: the message lists exactly which columns or samples to fix.
- **Channel index off by one**: channel indices start at 0.
- **Wrong channels or a time axis that looks like Z** in the Visualization tab: the dimension order was wrong at conversion. Delete the sample's `.zarr` folder, point its **Image path** back to the original file (**Edit Metadata CSV**), load the metadata again, set the right order in section 5 and convert again. Checking the order **before** converting (and before deleting the originals) avoids this.

## See also

- [Output Directory & File Layout](plugin_essentials/output_layout.md): what gets written where.
- [Visualization tab](plugin_essentials/visualization.md): open one sample after conversion to check channels.
- [Moving Projects](plugin_essentials/moving_projects.md): move a project to another drive or computer.
- [Segmentation overview](processing/segmentation/index.md): the next step in the pipeline.

# Moving Projects Between Drives and Computers

A BEHAV3D output folder stores every image as a `.zarr` folder with one small file per timepoint, each three to four sub-folders deep. A project with a few samples easily reaches tens of thousands of files and folders. Copying that with Explorer to a USB drive, a network share (NAS) or a sync service such as OneDrive is slow, mostly because of the per-file overhead rather than the amount of data. Such copies are also easy to leave incomplete without noticing.

The **Transfer** button next to the output directory (**Data Preparation → 1 · Output Directory**) solves this without changing how BEHAV3D stores data. Its menu has three entries: **Pack project for transfer…**, **Verify a copied bundle…** and **Import packed project…**.

## Pack → copy → import

1. **Pack project for transfer…** opens a dialog where you choose a **Destination folder** (for example the USB drive or NAS share itself; the free space there is shown) and tick the samples to include (all by default; project-level files are always included). Keep **Re-read and check archives after writing (recommended)** ticked, then click **Pack**. BEHAV3D writes a folder `<project>_behav3d_bundle` at the destination:

   | File | Content |
   |---|---|
   | `sample_<name>.zip` | one per sample: `images/<sample>/` and `trackdata/<sample>/` |
   | `project.zip` | `metadata.csv`, `behav3d_parameters.yml`, `analysis/`, `images/PixelClassification/` and any other files |
   | `bundle.json` | what is inside, where it came from, the checksum of every archive |
   | `SHA256SUMS.txt` | the same checksums in the standard format |

   The images are already compressed, so the archives are written without further compression: packing runs at disk speed. The progress window shows a percentage (not a size), because it covers writing, re-reading and checksumming, and reaches 100% only after all three. **Cancel** stops packing; finished archives are kept. Packing again into the same destination only rewrites the archives whose samples changed, so an interrupted or cancelled pack can simply be restarted.

   If a segmentation run was interrupted for one of the samples, BEHAV3D warns you first (**Interrupted segmentation**) and asks whether to pack anyway.

2. **Copy the bundle folder** with any tool. It is only a handful of large files, so this is fast on any drive, share or sync service.

3. On the receiving computer, **Verify a copied bundle…** (optional) checks that every archive arrived intact. Select the `…_behav3d_bundle` folder, then choose **Quick check** (compares each archive with its checksum) or **Full check** (also re-reads every file inside the archives). Damaged archives are listed so you can copy them again.

4. **Import packed project…**: select the bundle folder, then the folder to import into. If that folder is not empty, the project is imported into a new sub-folder named after it. Every file is checked against its checksum before anything is moved into place, and there must be enough free space; a damaged archive is reported and nothing is half-imported. Existing files are never overwritten. Afterwards BEHAV3D offers to load the imported project (**Load it now?**), which fills in the output directory and loads its metadata.

```{note}
Without BEHAV3D, the archives can still be extracted with Windows "Extract all" or 7-Zip: extract all of them into one folder and you get the original project folder back. The checksums can be checked with `certutil -hashfile <file> SHA256` (Windows) or `sha256sum -c SHA256SUMS.txt` (Linux/macOS).
```

## Paths are re-linked automatically

`metadata.csv` stores full paths to images, segments and tracks. When a project is loaded from a different location than where those paths were written, **Load Metadata** re-links every path whose file exists in the current output folder and saves the corrected CSV (keeping the original once as `metadata.csv.bak`). The log shows `🔗 Re-linked N path(s) from … to …`. This also works for folders that were copied by hand, and is done automatically at the end of an import.

Paths that point outside the output folder (for example the source experiments of a combined dataset, or raw images that were never converted) are left as they are, and the log warns about those it cannot find. **Pack** lists them in the log, because they are not included in the bundle.

## Command line

The same operations can be run without napari, for example on a NAS or a cluster. In a terminal with the BEHAV3D environment active:

```bash
python -m behav3d.io.transfer pack   D:\BEHAV3D_out\ProjectA E:\
python -m behav3d.io.transfer verify E:\ProjectA_behav3d_bundle --deep
python -m behav3d.io.transfer unpack E:\ProjectA_behav3d_bundle C:\Data\ProjectA
```

`pack` also accepts `--samples A B` (only these samples) and `--no-verify`; `verify --deep` is the **Full check**.

## Copying the folder directly

If you prefer copying the output folder itself, use a tool that copies many files in parallel, and compare the result afterwards:

```bash
robocopy "D:\BEHAV3D_out\ProjectA" "E:\ProjectA" /E /MT:32 /R:1 /W:1 /NFL /NDL
rclone copy "D:\BEHAV3D_out\ProjectA" "E:\ProjectA" --transfers 32 --checkers 32
rclone check "D:\BEHAV3D_out\ProjectA" "E:\ProjectA"
```

`robocopy` does not compare file contents; `rclone check` does. Do not open a project inside a folder that a sync client (OneDrive, Dropbox, …) is still uploading.

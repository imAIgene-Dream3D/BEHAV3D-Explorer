# Moving Projects Between Drives and Computers

A BEHAV3D output folder stores every image as a `.zarr` folder with one small file per timepoint, each three to four sub-folders deep. A project with a few samples easily reaches tens of thousands of files and folders. Copying that with Explorer to a USB drive, a network share (NAS) or a sync service such as OneDrive is slow, mostly because of the per-file overhead rather than the amount of data. Such copies are also easy to leave incomplete without noticing.

The **Transfer** button next to the output directory (Data Preparation → **1 · Output Directory**) solves this without changing how BEHAV3D stores data.

## Pack → copy → import

1. **Pack project for transfer…** — choose a destination (for example the USB drive or NAS share itself) and which samples to include. BEHAV3D writes a folder `<project>_behav3d_bundle` there:

   | File | Content |
   |---|---|
   | `sample_<name>.zip` | one per sample: `images/<sample folders>/` and `trackdata/<sample>/` |
   | `project.zip` | `metadata.csv`, `behav3d_parameters.yml`, `analysis/`, `images/PixelClassification/`, other files |
   | `bundle.json` | what is inside, where it came from, the checksum of every archive |
   | `SHA256SUMS.txt` | the same checksums in the standard format |

   The images are already compressed, so the archives are written without further compression: packing runs at disk speed. Each archive is re-read and checked after writing. The progress bar covers writing, re-reading and checksumming, so it shows a percentage (not a size) and reaches 100% only after all three. Packing again into the same destination only rewrites the archives whose samples changed, so an interrupted pack can simply be restarted.

2. **Copy the bundle folder** with any tool. It is only a handful of large files, so this is fast on any drive, share or sync service.

3. On the receiving computer, **Verify a copied bundle…** (optional) checks that every archive arrived intact. *Quick check* compares each archive with its checksum; *Full check* also re-reads every file inside.

4. **Import packed project…** — choose the bundle and a folder to import into. Every file is checked against its checksum before anything is moved into place; a damaged archive is reported and nothing is half-imported. Existing projects are never overwritten. Afterwards BEHAV3D offers to load the imported project.

```{note}
Without BEHAV3D, the archives can still be extracted with Windows "Extract all" or 7-Zip: extract all of them into one folder and you get the original project folder back. The checksums can be checked with `certutil -hashfile <file> SHA256` (Windows) or `sha256sum -c SHA256SUMS.txt` (Linux/macOS).
```

## Paths are re-linked automatically

`metadata.csv` stores full paths to images, segments and tracks. When a project is loaded from a different location than where those paths were written, **Load Metadata** re-links every path whose file exists in the current output folder and saves the corrected CSV (keeping the original once as `metadata.csv.bak`). The log shows `🔗 Re-linked N path(s) from … to …`. This also works for folders that were copied by hand.

Paths that point outside the output folder (for example the source experiments of a combined dataset, or raw images that were never converted) are left as they are. **Pack** lists them in the log, because they are not included in the bundle.

## Command line

The same operations are available outside napari, for example on a NAS or a cluster:

```bash
python -m behav3d.io.transfer pack   D:\BEHAV3D_out\ProjectA E:\
python -m behav3d.io.transfer verify E:\ProjectA_behav3d_bundle --deep
python -m behav3d.io.transfer unpack E:\ProjectA_behav3d_bundle C:\Data\ProjectA
```

## Copying the folder directly

If you prefer copying the output folder itself, use a tool that copies many files in parallel, and compare the result afterwards:

```bash
robocopy "D:\BEHAV3D_out\ProjectA" "E:\ProjectA" /E /MT:32 /R:1 /W:1 /NFL /NDL
rclone copy "D:\BEHAV3D_out\ProjectA" "E:\ProjectA" --transfers 32 --checkers 32
rclone check "D:\BEHAV3D_out\ProjectA" "E:\ProjectA"
```

`robocopy` does not compare file contents; `rclone check` does. Do not open a project inside a folder that a sync client (OneDrive, Dropbox, …) is still uploading.

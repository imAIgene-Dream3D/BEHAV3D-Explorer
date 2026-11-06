# Cellpose Segmentation

This guide explains how to use **Cellpose** for 3D segmentation with pretrained models. You run it in the **Segmentation** tab of the napari plugin.

## Workflow in the napari plugin

1. **Get a model**: the pretrained T cell and Organoid models are in this folder and on [Zenodo](https://zenodo.org/records/18872978). Models are single files with no extension.
2. **Prepare your data** in the **Data Preparation** tab: load or build `metadata.csv` and click **Convert All Images to Zarr** (samples that are not converted are skipped).
3. Open the **Segmentation** tab and pick **Cellpose (Deep Learning)** in **Method:**.
4. **Section 1 · Channel Configuration**: choose **Same for all** or **Per sample**, assign a population (or `dead`) to each channel, then click **💾 Save Channel Configuration**. Check this mapping carefully: the model only sees the channel mapped to the population you segment.
5. **Section 2 · Model & Cell Type**: **Browse…** to the model file and click **📂 Load Model**. Choose the **Cell type to segment** that matches the model. Keep **Process all timepoints** ticked, or set **Start:** / **End:**.
6. Click **▶ Run Cellpose — *cell type***. It runs in the background on every sample, with no preview, so check the first result in the **Visualization** tab before processing everything.
7. To segment another population (e.g. organoids and T cells), repeat steps 5–6 with the matching model.
8. **Dead-stain channel?** Click **☠ Run Dead Mask (Otsu)** to create the dead mask used later by Feature Extraction.
9. Optionally, queue the runs with the **+🛒** buttons to launch them together with the rest of the pipeline.

> **Note:** A run on a timepoint range replaces the whole segmentation of that population: timepoints outside the range are left empty. Test a model on a short range, then run all timepoints.

Full details, every control, tuning tips and output paths are in the wiki: [Cellpose segmentation page](https://imaigene-dream3d.github.io/BEHAV3D-Explorer/processing/segmentation/cellpose.html).

---

*Note: To train a new Cellpose model, refer to the `train_behav3d_cellpose.ipynb` notebook.*

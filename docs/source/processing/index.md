# Processing

Processing turns the prepared raw data into a set of **labelled, tracked objects** that the analysis stage can compute on.

```{mermaid}
flowchart LR
    Seg["🦠 Segmentation<br/>6 methods"] --> Trk["📍 Tracking<br/>btrack · Fragmentation Propagation · Bounded Propagation · Reporter Propagation · TrackPy · LapTrack · Import tracking"]
```

## Sections

| Page | Tab in napari | What it covers |
|---|---|---|
| [Segmentation overview](segmentation/index.md) | 🦠 Segmentation | When to pick APOC, ConvPaint, Pixel Classifier, Cellpose, Cellpose-SAM or Import segmentation |
| [Tracking overview](tracking/index.md) | 📍 Tracking | btrack (Bayesian), Fragmentation Propagation, Bounded Propagation, Reporter Propagation, TrackPy, LapTrack and Import tracking — chosen per cell type |

```{toctree}
:hidden:
:maxdepth: 2

segmentation/index
tracking/index
```

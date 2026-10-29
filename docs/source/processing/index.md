# Processing

Processing turns the prepared raw data into a set of **labelled, tracked objects** that the analysis stage can compute on.

```{mermaid}
flowchart LR
    Seg["🦠 Segmentation<br/>APOC · ConvPaint · Pixel Classifier<br/>Cellpose · Cellpose-SAM · Import segmentation"] --> Trk["📍 Tracking<br/>btrack · LapTrack · TrackPy<br/>Fragmentation · Bounded · Reporter Propagation<br/>Import tracking"]
```

## Sections

| Page | Tab in napari | What it covers |
|---|---|---|
| [Segmentation overview](segmentation/index.md) | 🦠 Segmentation | When to pick APOC, ConvPaint, Pixel Classifier, Cellpose, Cellpose-SAM or Import segmentation |
| [Tracking overview](tracking/index.md) | 📍 Tracking | btrack (Bayesian), Fragmentation Propagation, Bounded Propagation, Reporter Propagation, TrackPy, LapTrack and Import tracking — chosen per cell type — and [manual editing](tracking/manual_editing.md) of tracked segments |

```{toctree}
:hidden:
:maxdepth: 2

segmentation/index
tracking/index
```

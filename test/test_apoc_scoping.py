"""
Scoping tests for APOC single-cell-type runs.

A "Run <cell type>" must load and process only that cell type's classifier, must
not drag the death mask along, and a death-only run must load only the death
classifier. Previously every classifier (and the death one) was loaded before
``only_cell_types`` was applied.

The classifier loader and ``apoc.ObjectSegmenter`` are replaced by recorders, and
the metadata points at a raw image that does not exist, so the run stops after
the classifier-loading stage without touching a GPU.

Runs under pytest. Requires the `behav3d` conda env (apoc / pyclesperanto).
"""
import pandas as pd
import pytest

apoc_segment = pytest.importorskip("behav3d.preprocessing.segmentation.apoc_segment")


def _metadata(tmp_path, with_dead=True):
    return pd.DataFrame({
        "sample_name": ["s1"],
        "raw_image_path": [str(tmp_path / "missing.zarr")],
        "or_organoid_line_condition": ["a"],
        "im_tcell_line_condition": ["b"],
        "im_nk_line_condition": ["c"],
        "dead_channel": ["Channel 3" if with_dead else None],
    })


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    loaded = []
    death_loaded = []

    def fake_load_classifier(pixelclass_dir, cell_type, provided_path=None):
        loaded.append(cell_type)
        return object()

    class FakeObjectSegmenter:
        def __init__(self, opencl_filename=None):
            death_loaded.append(opencl_filename)

    monkeypatch.setattr(apoc_segment, "_load_classifier", fake_load_classifier)
    monkeypatch.setattr(apoc_segment.apoc, "ObjectSegmenter", FakeObjectSegmenter)

    pixelclass = tmp_path / "images" / "PixelClassification"
    pixelclass.mkdir(parents=True)
    (pixelclass / "PixelClassifier_Death.cl").write_text("/* fake */")
    return loaded, death_loaded


def _run(tmp_path, **kwargs):
    return apoc_segment.run_apoc_segmentation(
        output_dir=str(tmp_path),
        metadata=_metadata(tmp_path),
        metadata_csv_path="",
        **kwargs,
    )


def test_helpers():
    assert apoc_segment._death_selected(None) is True
    assert apoc_segment._death_selected(["tcell"]) is False
    assert apoc_segment._death_selected(["tcell", "dead"]) is True
    assert apoc_segment._death_selected(["death"]) is True
    assert apoc_segment._select_cell_types(["a", "b", "c"], None) == ["a", "b", "c"]
    assert apoc_segment._select_cell_types(["a", "b", "c"], ["c", "a"]) == ["a", "c"]
    assert apoc_segment._select_cell_types(["a", "b"], ["dead"]) == []


def test_single_cell_type_loads_only_that_classifier(recorder, tmp_path):
    loaded, death_loaded = recorder
    _run(tmp_path, only_cell_types=["tcell"])
    assert loaded == ["tcell"]
    assert death_loaded == []  # death mask is not part of a tcell run


def test_death_only_run_loads_only_the_death_classifier(recorder, tmp_path):
    loaded, death_loaded = recorder
    _run(tmp_path, only_cell_types=["dead"])
    assert loaded == []
    assert len(death_loaded) == 1 and death_loaded[0].endswith("PixelClassifier_Death.cl")


def test_unrestricted_run_still_loads_everything(recorder, tmp_path):
    loaded, death_loaded = recorder
    _run(tmp_path)
    assert sorted(loaded) == ["nk", "organoid", "tcell"]
    assert len(death_loaded) == 1


def test_selected_type_without_classifier_is_an_error(monkeypatch, recorder, tmp_path):
    monkeypatch.setattr(apoc_segment, "_load_classifier", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, only_cell_types=["tcell"])


def test_unrestricted_run_tolerates_untrained_types(monkeypatch, recorder, tmp_path):
    monkeypatch.setattr(apoc_segment, "_load_classifier", lambda *a, **k: None)
    _run(tmp_path)  # must not raise: types that were never trained are skipped

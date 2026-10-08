"""Re-linking absolute metadata.csv paths after a project folder is moved.

Runs under pytest or standalone: python test/test_metadata_rebase.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behav3d.core.metadata import rebase_metadata_paths, relink_metadata_csv  # noqa: E402
from behav3d.core.portable_paths import relocate_path, split_path_parts  # noqa: E402


def _make_project(root: Path):
    (root / "images" / "SRaw" / "SRaw.zarr").mkdir(parents=True)
    (root / "images" / "SRaw" / "SRaw.zarr" / "zarr.json").write_text("{}")
    (root / "images" / "S1" / "S1_tcell_tracked.zarr").mkdir(parents=True)
    (root / "images" / "S1" / "S1_tcell_tracked.zarr" / "zarr.json").write_text("{}")
    (root / "trackdata" / "S1" / "tcell").mkdir(parents=True)
    (root / "trackdata" / "S1" / "tcell" / "S1_tcell_tracks.csv").write_text("a\n1\n")


def _metadata(root, external):
    return pd.DataFrame({
        "sample_name": ["S1"],
        "raw_image_path": [str(root / "images" / "SRaw" / "SRaw.zarr")],
        "im_tcell_tracks_image_path": [str(root / "images" / "S1" / "S1_tcell_tracked.zarr")],
        "im_tcell_tracks_csv_path": [str(root / "trackdata" / "S1" / "tcell" / "S1_tcell_tracks.csv")],
        "dead_mask_path": [str(external)],
        "pixel_distance_xy": [1.0],
    })


def test_moved_project_is_relinked_and_externals_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old, new = tmp / "old" / "proj", tmp / "elsewhere" / "proj_copy"
        _make_project(old)
        external = tmp / "other_experiment" / "images" / "X" / "dead.zarr"
        external.mkdir(parents=True)
        md = _metadata(old, external)
        shutil.copytree(old, new)

        # Old copy still exists: paths must still move to the folder being used.
        out, report = rebase_metadata_paths(md, new)
        assert out.loc[0, "raw_image_path"] == str(new / "images" / "SRaw" / "SRaw.zarr")
        assert out.loc[0, "im_tcell_tracks_csv_path"] == str(new / "trackdata" / "S1" / "tcell" / "S1_tcell_tracks.csv")
        assert out.loc[0, "dead_mask_path"] == str(external)  # other experiment: untouched
        assert len(report["changed"]) == 3
        assert md.loc[0, "raw_image_path"].startswith(str(old))  # input not mutated

        # Old copy gone: same result.
        shutil.rmtree(old)
        out2, _ = rebase_metadata_paths(md, new)
        assert out2.loc[0, "im_tcell_tracks_image_path"] == str(new / "images" / "S1" / "S1_tcell_tracked.zarr")

        # Already correct: nothing to do.
        out3, report3 = rebase_metadata_paths(out2, new)
        assert report3["changed"] == []
        assert out3.equals(out2)


def test_posix_paths_from_another_os_and_case_differences():
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "proj"
        _make_project(new)
        md = pd.DataFrame({
            "sample_name": ["S1"],
            "raw_image_path": ["/mnt/NAS/Projects/Proj/Images/SRaw/SRaw.zarr"],
            "im_tcell_tracks_image_path": ["/mnt/NAS/Projects/Proj/images/S1/S1_tcell_tracked.zarr"],
        })
        out, report = rebase_metadata_paths(md, new)
        assert out.loc[0, "im_tcell_tracks_image_path"] == str(new / "images" / "S1" / "S1_tcell_tracked.zarr")
        assert Path(out.loc[0, "raw_image_path"]).exists()
        assert report["unresolved"] == []


def test_explicit_old_root_maps_missing_files_too():
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "proj"
        _make_project(new)
        md = pd.DataFrame({
            "sample_name": ["S1"],
            "raw_image_path": [r"D:\runs\proj\images\SRaw\SRaw.zarr"],
            "dead_mask_path": [r"D:\runs\proj\images\S1\not_made_yet.zarr"],
        })
        out, report = rebase_metadata_paths(md, new, old_output_dir=r"D:\runs\proj")
        assert out.loc[0, "dead_mask_path"] == str(new / "images" / "S1" / "not_made_yet.zarr")
        assert len(report["changed"]) == 2


def test_relink_metadata_csv_saves_once_with_backup():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old, new = tmp / "old", tmp / "new"
        _make_project(old)
        shutil.copytree(old, new)
        shutil.rmtree(old)
        csv = new / "metadata.csv"
        _metadata(old, tmp / "nowhere.zarr").to_csv(csv, index=False)
        original = csv.read_text()
        logs = []
        md, report = relink_metadata_csv(pd.read_csv(csv), csv, new, log=logs.append)
        assert len(report["changed"]) == 3
        assert (new / "metadata.csv.bak").read_text() == original
        assert str(new / "images" / "SRaw" / "SRaw.zarr") in csv.read_text()
        assert any("Re-linked 3" in m for m in logs)
        assert any("not found" in m for m in logs)  # the unresolvable dead mask is reported


def test_relocate_path_and_split():
    assert split_path_parts(r"C:\a\b/c") == ["C:", "a", "b", "c"]
    assert split_path_parts("//srv/share/x") == ["srv", "share", "x"]
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "proj"
        (new / "analysis" / "tcell").mkdir(parents=True)
        f = new / "analysis" / "tcell" / "states.h5ad"
        f.write_text("x")
        assert relocate_path(r"E:\gone\proj\analysis\tcell\states.h5ad", new) == f
        assert relocate_path(r"E:\gone\proj\analysis\tcell\missing.h5ad", new) is None
        assert relocate_path(f, new) == f


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASSED", name)

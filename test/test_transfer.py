"""Pack -> verify -> unpack of project transfer bundles (behav3d.io.transfer).

Runs under pytest or standalone: python test/test_transfer.py
"""
import filecmp
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behav3d.io import transfer as T  # noqa: E402

QUIET = dict(log=lambda m: None)


def _make_project(root: Path):
    """Two samples; S1's raw zarr folder is named after the raw file, not the sample."""
    for sample, raw_folder in (("S1", "SRaw1"), ("S2", "S2")):
        raw = root / "images" / raw_folder / f"{raw_folder}.zarr"
        arr = zarr.open(str(raw), mode="w", shape=(3, 2, 4, 4), chunks=(1, 2, 4, 4), dtype="uint16")
        arr[:] = np.arange(3 * 2 * 4 * 4, dtype="uint16").reshape(3, 2, 4, 4) + (7 if sample == "S2" else 0)
        trk = root / "images" / sample / f"{sample}_tcell_tracked.zarr"
        tarr = zarr.open(str(trk), mode="w", shape=(3, 4, 4), chunks=(1, 4, 4), dtype="uint16")
        tarr[1] = 5
        csv_dir = root / "trackdata" / sample / "tcell"
        csv_dir.mkdir(parents=True)
        (csv_dir / f"{sample}_tcell_tracks.csv").write_text("TrackID,t\n1,0\n1,1\n")
    (root / "analysis" / "tcell" / "track_features").mkdir(parents=True)
    (root / "analysis" / "tcell" / "track_features" / "feat.csv").write_text("x\n" + "1\n" * 1000)
    (root / "analysis" / "tcell" / "empty_dir").mkdir()
    (root / "images" / "PixelClassification").mkdir(parents=True)
    (root / "images" / "PixelClassification" / "model.cl").write_bytes(os.urandom(2048))
    (root / "behav3d_parameters.yml").write_text(f"paths:\n  output_dir: {root}\n")
    (root / "Thumbs.db").write_bytes(b"junk")
    md = pd.DataFrame({
        "sample_name": ["S1", "S2"],
        "raw_image_path": [str(root / "images" / "SRaw1" / "SRaw1.zarr"), str(root / "images" / "S2" / "S2.zarr")],
        "im_tcell_tracks_image_path": [str(root / "images" / s / f"{s}_tcell_tracked.zarr") for s in ("S1", "S2")],
        "im_tcell_tracks_csv_path": [str(root / "trackdata" / s / "tcell" / f"{s}_tcell_tracks.csv") for s in ("S1", "S2")],
    })
    md.to_csv(root / "metadata.csv", index=False)


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _setup(tmp):
    tmp = Path(tmp)
    src = tmp / "projA"
    _make_project(src)
    return tmp, src


def test_round_trip_is_byte_identical_and_relinked():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        bundle = T.pack_project(src, tmp / "usb", **QUIET)
        assert bundle == tmp / "usb" / "projA_behav3d_bundle"
        names = sorted(p.name for p in bundle.glob("*.zip"))
        assert names == ["project.zip", "sample_S1.zip", "sample_S2.zip"]

        with zipfile.ZipFile(bundle / "sample_S1.zip") as zf:
            members = zf.namelist()
            assert any(m.startswith("images/SRaw1/SRaw1.zarr/") for m in members)  # owned via metadata
            assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist() if "/c/" in i.filename)
        with zipfile.ZipFile(bundle / "project.zip") as zf:
            assert "analysis/tcell/empty_dir/" in zf.namelist()
            assert "Thumbs.db" not in zf.namelist()
            assert zf.getinfo("analysis/tcell/track_features/feat.csv").compress_type == zipfile.ZIP_DEFLATED

        sums = (bundle / "SHA256SUMS.txt").read_text().split("\n")
        info = json.loads((bundle / "bundle.json").read_text())
        assert {f"{a['sha256']} *{a['name']}" for a in info["archives"]} == {s for s in sums if s}

        assert T.verify_bundle(bundle, deep=True, **QUIET)["ok"]

        dest = tmp / "other_drive" / "projA_moved"
        result = T.unpack_bundle(bundle, dest, **QUIET)
        assert result["relinked"] == 6
        expected = [f for f in _files(src) if f != "Thumbs.db"]
        assert _files(dest) == expected
        assert (dest / "analysis" / "tcell" / "empty_dir").is_dir()
        for rel in expected:
            if rel not in ("metadata.csv", "behav3d_parameters.yml"):
                assert filecmp.cmp(src / rel, dest / rel, shallow=False), rel
        md = pd.read_csv(dest / "metadata.csv")
        assert md.loc[0, "raw_image_path"] == str(dest / "images" / "SRaw1" / "SRaw1.zarr")
        np.testing.assert_array_equal(
            zarr.open(md.loc[1, "raw_image_path"], mode="r")[:], zarr.open(str(src / "images/S2/S2.zarr"), mode="r")[:]
        )
        assert str(dest) in (dest / "behav3d_parameters.yml").read_text()
        assert not (dest / "metadata.csv.bak").exists()
        assert not list(dest.parent.glob("*.b3dtmp"))


def test_repack_reuses_unchanged_archives():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        bundle = T.pack_project(src, tmp / "usb", **QUIET)
        stamps = {p.name: p.stat().st_mtime_ns for p in bundle.glob("*.zip")}
        T.pack_project(src, tmp / "usb", **QUIET)
        assert stamps == {p.name: p.stat().st_mtime_ns for p in bundle.glob("*.zip")}

        (src / "trackdata" / "S2" / "tcell" / "S2_tcell_tracks.csv").write_text("TrackID,t\n2,0\n")
        T.pack_project(src, tmp / "usb", **QUIET)
        after = {p.name: p.stat().st_mtime_ns for p in bundle.glob("*.zip")}
        assert after["sample_S1.zip"] == stamps["sample_S1.zip"]
        assert after["sample_S2.zip"] != stamps["sample_S2.zip"]
        assert T.verify_bundle(bundle, **QUIET)["ok"]


def test_corruption_is_detected_and_nothing_is_left_behind():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        bundle = T.pack_project(src, tmp / "usb", **QUIET)
        z = bundle / "sample_S1.zip"
        with zipfile.ZipFile(z) as zf:
            info = zf.getinfo("images/SRaw1/SRaw1.zarr/c/1/0/0/0")
        data = bytearray(z.read_bytes())
        data[info.header_offset + 30 + len(info.filename) + 2] ^= 0xFF  # inside the member's data
        z.write_bytes(bytes(data))

        report = T.verify_bundle(bundle, **QUIET)
        assert not report["ok"] and "sample_S1.zip" in report["problems"][0]

        dest = tmp / "dest"
        try:
            T.unpack_bundle(bundle, dest, **QUIET)
        except T.TransferError as e:
            assert "sample_S1.zip" in str(e)
        else:
            raise AssertionError("corrupted bundle was imported")
        assert not dest.exists()
        assert not list(tmp.glob("*.b3dtmp"))

        # Truncated copy (e.g. drive pulled early).
        z.write_bytes(z.read_bytes()[:-100])
        assert "incomplete" in T.verify_bundle(bundle, **QUIET)["problems"][0]


def test_import_refuses_conflicts_unless_overwrite_replaces_whole_store():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        bundle = T.pack_project(src, tmp / "usb", **QUIET)
        dest = tmp / "dest"
        T.unpack_bundle(bundle, dest, **QUIET)
        stale = dest / "images" / "S2" / "S2.zarr" / "c" / "99"
        stale.mkdir(parents=True)
        (stale / "0").write_bytes(b"stale")
        try:
            T.unpack_bundle(bundle, dest, **QUIET)
        except T.TransferError as e:
            assert "already exist" in str(e)
        else:
            raise AssertionError("existing project was overwritten silently")
        T.unpack_bundle(bundle, dest, overwrite=True, **QUIET)
        assert not stale.exists()  # store replaced, not merged


def test_sample_subset_interrupted_marker_cancel_and_guards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        bundle = T.pack_project(src, tmp / "a", samples=["S2"], **QUIET)
        assert sorted(p.name for p in bundle.glob("*.zip")) == ["project.zip", "sample_S2.zip"]

        (src / "images" / "SRaw1" / ".seg_processing").write_text("")
        try:
            T.pack_project(src, tmp / "b", **QUIET)
        except T.InterruptedRunError as e:
            assert "S1" in str(e)
        else:
            raise AssertionError("interrupted run was packed without force")
        assert T.pack_project(src, tmp / "b", force=True, **QUIET).exists()

        try:
            T.pack_project(src, src / "inside", **QUIET)
        except T.TransferError:
            pass
        else:
            raise AssertionError("bundle inside the output folder was allowed")

        calls = []

        def cancel():
            calls.append(1)
            return len(calls) > 3

        try:
            T.pack_project(src, tmp / "c", force=True, cancel=cancel, **QUIET)
        except T.TransferCancelled:
            pass
        else:
            raise AssertionError("cancel was ignored")
        leftovers = list((tmp / "c").rglob("*.partial")) + list((tmp / "c").rglob("*.zip"))
        assert leftovers == []

        for bad in ("../evil.txt", "/abs.txt", "C:/x.txt"):
            try:
                T._safe_member_path(bad)
            except T.TransferError:
                continue
            raise AssertionError(bad)


def test_progress_total_is_work_units_with_phase_labels():
    # The progress total counts I/O passes (write + re-read + hash), so it is a
    # multiple of the data size: the GUI must show a percentage, not bytes.
    real = T._Progress
    T._Progress = lambda total, cb=None, cancel=None: real(total, cb, cancel, interval=0)  # no throttling
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp, src = _setup(tmp)
            data = T.plan_bundle(src)["total_bytes"]
            events = []
            bundle = T.pack_project(src, tmp / "usb", progress=lambda d, t, m: events.append((d, t, m)), **QUIET)
            assert events and {t for _d, t, _m in events} == {3 * data}
            done = [d for d, _t, _m in events]
            assert done == sorted(done)
            phases = {m.split(":")[0] for _d, _t, m in events}
            assert {"Writing", "Verifying contents", "Checksum"} <= phases

            events.clear()
            T.verify_bundle(bundle, deep=True, progress=lambda d, t, m: events.append((d, t, m)), **QUIET)
            assert events and events[0][1] > data  # checksum pass + content pass
            phases = {m.split(":")[0] for _d, _t, m in events}
            assert {"Checksum", "Verifying contents"} <= phases
    finally:
        T._Progress = real


def test_cli_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp, src = _setup(tmp)
        assert T.main(["pack", str(src), str(tmp / "usb")]) == 0
        bundle = tmp / "usb" / "projA_behav3d_bundle"
        assert T.main(["verify", str(bundle), "--deep"]) == 0
        assert T.main(["unpack", str(bundle), str(tmp / "dest")]) == 0
        assert T.main(["unpack", str(bundle), str(tmp / "dest")]) == 2  # conflict


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASSED", name)

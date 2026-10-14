"""`.h5ad` writes must not depend on ``anndata.settings.allow_write_nullable_strings``.

anndata >= 0.11 refuses pandas nullable strings (StringArray/ArrowStringArray)
unless that flag is on; files written while it was on read back as StringArray,
so a read -> rename -> write cycle failed ("Rename Failed" in the cluster
rename dialog). ``prepare_adata_for_write`` converts them before writing.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from behav3d.analysis.behavior.general import relabel_cluster_ids
from behav3d.core.h5_access import prepare_adata_for_write, write_adata


@pytest.fixture(autouse=True)
def _default_nullable_flag():
    with ad.settings.override(allow_write_nullable_strings=False):
        yield


def _string_adata():
    n = 5
    obs = pd.DataFrame(
        {
            "state": pd.array(["a", "b", pd.NA, "a", "c"], dtype="string"),
            "unique": pd.array([f"t{i}" for i in range(n)], dtype="string"),
            "ordered_cat": pd.Categorical(
                ["lo", "hi", "lo", None, "mid"],
                categories=pd.Index(["lo", "mid", "hi"], dtype="string"),
                ordered=True,
            ),
            "value": np.arange(n, dtype=float),
        },
        index=pd.Index([f"cell{i}" for i in range(n)], dtype="string"),
    )
    adata = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32), obs=obs)
    adata.uns["summary"] = {
        "table": pd.DataFrame({"label": pd.array(["x", "y"], dtype="string"), "n": [1, 2]})
    }
    return adata


def test_raw_write_fails_without_sanitizing(tmp_path):
    adata = _string_adata()
    with pytest.raises(Exception, match="allow_write_nullable_strings"):
        adata.write_h5ad(tmp_path / "raw.h5ad")


def test_write_adata_round_trip(tmp_path):
    path = tmp_path / "out.h5ad"
    write_adata(_string_adata(), path)
    back = ad.read_h5ad(path)

    state = back.obs["state"]
    assert state.isna().tolist() == [False, False, True, False, False]
    assert state.astype(object).where(state.notna(), None).tolist() == ["a", "b", None, "a", "c"]
    assert back.obs["unique"].astype(str).tolist() == [f"t{i}" for i in range(5)]

    cat = back.obs["ordered_cat"]
    assert list(cat.cat.categories) == ["lo", "mid", "hi"]
    assert cat.cat.ordered
    assert cat.isna().tolist() == [False, False, False, True, False]
    assert back.obs["value"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(back.obs_names) == [f"cell{i}" for i in range(5)]
    assert list(back.uns["summary"]["table"]["label"].astype(str)) == ["x", "y"]


def test_prepare_handles_views(tmp_path):
    view = _string_adata()[:3]
    assert view.is_view
    prepare_adata_for_write(view).write_h5ad(tmp_path / "view.h5ad")


def test_rename_on_string_column_is_writable(tmp_path):
    # The real failing state: a state column that came back from disk as StringArray.
    adata = _string_adata()
    relabel_cluster_ids(
        adata=adata,
        mapping={"a": "Elongated_scanner", "b": "Protrusive_engager"},
        cluster_key="state",
        overwrite_original=True,
        keep_unmapped=True,
        categories=["Protrusive_engager", "Elongated_scanner", "c"],
    )
    path = tmp_path / "renamed.h5ad"
    write_adata(adata, path)
    back = ad.read_h5ad(path)
    assert list(back.obs["state"].cat.categories) == ["Protrusive_engager", "Elongated_scanner", "c"]
    assert back.obs["state_original"].isna().sum() == 1

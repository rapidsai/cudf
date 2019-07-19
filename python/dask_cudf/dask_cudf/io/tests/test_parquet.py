import os

import pandas as pd
import pytest

import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
from dask.utils import natural_sort_key

import dask_cudf

nrows = 40
npartitions = 15
df = pd.DataFrame(
    {
        "x": [i * 7 % 5 for i in range(nrows)],  # Not sorted
        "y": [i * 2.5 for i in range(nrows)],
    }
)  # Sorted
ddf = dd.from_pandas(df, npartitions=npartitions)


def test_roundtrip_from_dask(tmpdir):
    tmpdir = str(tmpdir)
    ddf.to_parquet(tmpdir, engine="pyarrow")
    files = sorted(
        [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if not f.endswith("_metadata")
        ],
        key=natural_sort_key,
    )

    # Read list of parquet files
    ddf2 = dask_cudf.read_parquet(files)
    assert_eq(ddf, ddf2, check_divisions=False)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(files, columns=["x"])
    assert_eq(ddf[["x"]], ddf2, check_divisions=False)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(files, columns="y")
    assert_eq(ddf[["y"]], ddf2, check_divisions=False)

    # Now include metadata
    # Read list of parquet files
    ddf2 = dask_cudf.read_parquet(tmpdir)
    assert_eq(ddf, ddf2, check_divisions=False)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(tmpdir, columns=["x"])
    assert_eq(ddf[["x"]], ddf2, check_divisions=False)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(tmpdir, columns="y")
    assert_eq(ddf[["y"]], ddf2, check_divisions=False)

    # # Read parquet-dataset directory
    # # dask_cudf.read_parquet will ignore *_metadata files
    # ddf2 = dask_cudf.read_parquet(os.path.join(tmpdir, "*"))
    # assert_eq(ddf, ddf2, check_divisions=False)


def test_roundtrip_from_pandas(tmpdir):
    fn = str(tmpdir.join("test.parquet"))
    dfp = df.copy()
    dfp.index.name = "index"
    dfp.to_parquet(fn, engine="pyarrow")
    ddf2 = dask_cudf.read_parquet(fn, index="index")
    # Losing the index name her for some reason:
    assert_eq(dfp, ddf2, check_index=False)


@pytest.mark.parametrize("index", [False, True])
def test_empty(tmpdir, index):
    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": [11, 12, 12], "b": [4, 5, 6]})[:0]
    if index:
        dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)

    ddf2.to_parquet(fn, write_index=index, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    assert_eq(ddf2, read_df)


def test_simple(tmpdir):
    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": ["a", "b", "b"], "b": [4, 5, 6]})
    dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn, index=["a"])
    assert_eq(ddf2, read_df)

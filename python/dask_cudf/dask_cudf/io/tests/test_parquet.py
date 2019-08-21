import os

import pandas as pd
import pytest

import dask
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
    },
    index=pd.Index(range(nrows), name="index"),
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
    ddf2 = dask_cudf.read_parquet(files, gather_statistics=True)
    assert_eq(ddf, ddf2)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(files, columns=["x"], gather_statistics=True)
    assert_eq(ddf[["x"]], ddf2)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(files, columns="y", gather_statistics=True)
    assert_eq(ddf[["y"]], ddf2)

    # Now include metadata; gather_statistics is True by default
    # Read list of parquet files
    ddf2 = dask_cudf.read_parquet(tmpdir)
    assert_eq(ddf, ddf2)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(tmpdir, columns=["x"])
    assert_eq(ddf[["x"]], ddf2)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(tmpdir, columns="y")
    assert_eq(ddf[["y"]], ddf2)


def test_roundtrip_from_pandas(tmpdir):
    fn = str(tmpdir.join("test.parquet"))

    # First without specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=False)
    dfp = dfp.reset_index(drop=True)
    ddf2 = dask_cudf.read_parquet(fn)
    assert_eq(dfp, ddf2, check_index=True)

    # Now, specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=True)
    ddf2 = dask_cudf.read_parquet(fn, index=["index"])
    assert_eq(dfp, ddf2, check_index=True)


def test_strings(tmpdir):

    fn = str(tmpdir)
    dfp = pd.DataFrame(
        {"a": ["aa", "bbb", "cccc"], "b": ["hello", "dog", "man"]}
    )
    dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn, index=["a"])
    assert_eq(ddf2, read_df.compute().to_pandas())

    read_df_cats = dask_cudf.read_parquet(
        fn, index=["a"], strings_to_categorical=True
    )
    assert_eq(read_df_cats.dtypes, read_df_cats.compute().dtypes)
    assert_eq(read_df_cats.dtypes[0], "int32")


def test_dask_timeseries_from_pandas(tmpdir):

    fn = str(tmpdir.join("test.parquet"))
    ddf2 = dask.datasets.timeseries(freq="D")
    pdf = ddf2.compute()
    pdf.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    assert_eq(ddf2, read_df.compute().to_pandas())


def test_dask_timeseries_from_dask(tmpdir):

    fn = str(tmpdir)
    ddf2 = dask.datasets.timeseries(freq="D")
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    # Note: Loosing the index name here
    assert_eq(ddf2, read_df.compute().to_pandas(), check_index=False)


@pytest.mark.parametrize("index", [False, True])
def test_empty(tmpdir, index):

    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": [11.0, 12.0, 12.0], "b": [4, 5, 6]})[:0]
    if index:
        dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)

    ddf2.to_parquet(fn, write_index=index, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    assert_eq(ddf2, read_df.compute().to_pandas())


def test_filters(tmpdir):

    tmp_path = str(tmpdir)
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    ddf = dd.from_pandas(df, npartitions=5)
    assert ddf.npartitions == 5

    ddf.to_parquet(tmp_path, engine="pyarrow")

    a = dask_cudf.read_parquet(tmp_path, filters=[("x", ">", 4)])
    assert a.npartitions == 3
    assert (a.x > 3).all().compute()

    b = dask_cudf.read_parquet(tmp_path, filters=[("y", "==", "c")])
    assert b.npartitions == 1
    b = b.compute().to_pandas()
    assert (b.y == "c").all()

    c = dask_cudf.read_parquet(
        tmp_path, filters=[("y", "==", "c"), ("x", ">", 6)]
    )
    assert c.npartitions <= 1
    assert not len(c)

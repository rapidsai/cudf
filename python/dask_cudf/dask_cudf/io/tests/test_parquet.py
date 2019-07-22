import os

import pandas as pd
import pytest

import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
from dask.utils import natural_sort_key

import dask_cudf


try:
    from dask.dataframe.io.parquet.arrow import ArrowEngine
except ImportError:
    ArrowEngine = None


def check_arrow_engine():
    if ArrowEngine is None:
        pytest.skip("ArrowEngine needed for full dask-parquet support.")


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

    if ArrowEngine is None:

        # Read list of parquet files
        ddf2 = dask_cudf.read_parquet(files)
        assert_eq(ddf, ddf2, check_divisions=False)

        # Specify columns=['x']
        ddf2 = dask_cudf.read_parquet(files, columns=["x"])
        assert_eq(ddf[["x"]], ddf2, check_divisions=False)

        # Specify columns='y'
        ddf2 = dask_cudf.read_parquet(files, columns="y")
        assert_eq(ddf[["y"]], ddf2, check_divisions=False)

        # Read parquet-dataset directory
        # dask_cudf.read_parquet will ignore *_metadata files
        ddf2 = dask_cudf.read_parquet(os.path.join(tmpdir, "*"))
        assert_eq(ddf, ddf2, check_divisions=False)

    else:

        # Read list of parquet files
        ddf2 = dask_cudf.read_parquet(files, gather_statistics=True)
        assert_eq(ddf, ddf2)

        # Specify columns=['x']
        ddf2 = dask_cudf.read_parquet(
            files, columns=["x"], gather_statistics=True
        )
        assert_eq(ddf[["x"]], ddf2)

        # Specify columns='y'
        ddf2 = dask_cudf.read_parquet(
            files, columns="y", gather_statistics=True
        )
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
    dfp = df.copy()
    dfp.index.name = "index"
    dfp.to_parquet(fn, engine="pyarrow")
    ddf2 = dask_cudf.read_parquet(fn, index="index")
    # Losing the index name here for some reason
    assert_eq(dfp, ddf2, check_index=False)


def test_strings(tmpdir):
    check_arrow_engine()

    fn = str(tmpdir)
    dfp = pd.DataFrame(
        {"a": ["aa", "bbb", "cccc"], "b": ["hello", "dog", "man"]}
    )
    dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn, index=["a"])
    assert_eq(ddf2, read_df.compute().to_pandas())


@pytest.mark.parametrize("index", [False, True])
def test_empty(tmpdir, index):
    check_arrow_engine()

    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": [11.0, 12.0, 12.0], "b": [4, 5, 6]})[:0]
    if index:
        dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)

    ddf2.to_parquet(fn, write_index=index, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    assert_eq(ddf2, read_df.compute().to_pandas())


def test_filters(tmpdir):
    check_arrow_engine()

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

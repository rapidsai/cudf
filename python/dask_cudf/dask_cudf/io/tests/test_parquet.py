# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import math
import os

import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.utils import natural_sort_key, parse_bytes

import cudf

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


@pytest.mark.parametrize("stats", [True, False])
def test_roundtrip_from_dask(tmpdir, stats):
    tmpdir = str(tmpdir)
    ddf.to_parquet(tmpdir, engine="pyarrow")
    files = sorted(
        [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            # TODO: Allow "_metadata" in list after dask#6047
            if not f.endswith("_metadata")
        ],
        key=natural_sort_key,
    )

    # Read list of parquet files
    ddf2 = dask_cudf.read_parquet(files, gather_statistics=stats)
    dd.assert_eq(ddf, ddf2, check_divisions=stats)

    # Specify columns=['x']
    ddf2 = dask_cudf.read_parquet(
        files, columns=["x"], gather_statistics=stats
    )
    dd.assert_eq(ddf[["x"]], ddf2, check_divisions=stats)

    # Specify columns='y'
    ddf2 = dask_cudf.read_parquet(files, columns="y", gather_statistics=stats)
    dd.assert_eq(ddf[["y"]], ddf2, check_divisions=stats)

    # Now include metadata
    ddf2 = dask_cudf.read_parquet(tmpdir, gather_statistics=stats)
    dd.assert_eq(ddf, ddf2, check_divisions=stats)

    # Specify columns=['x'] (with metadata)
    ddf2 = dask_cudf.read_parquet(
        tmpdir, columns=["x"], gather_statistics=stats
    )
    dd.assert_eq(ddf[["x"]], ddf2, check_divisions=stats)

    # Specify columns='y' (with metadata)
    ddf2 = dask_cudf.read_parquet(tmpdir, columns="y", gather_statistics=stats)
    dd.assert_eq(ddf[["y"]], ddf2, check_divisions=stats)


@pytest.mark.parametrize("write_meta", [True, False])
def test_roundtrip_from_dask_cudf(tmpdir, write_meta):
    tmpdir = str(tmpdir)
    gddf = dask_cudf.from_dask_dataframe(ddf)
    gddf.to_parquet(tmpdir, write_metadata_file=write_meta)

    gddf2 = dask_cudf.read_parquet(tmpdir)
    dd.assert_eq(gddf, gddf2, check_divisions=write_meta)


def test_roundtrip_none_rangeindex(tmpdir):
    fn = str(tmpdir.join("test.parquet"))
    gdf = cudf.DataFrame(
        {"id": [0, 1, 2, 3], "val": [None, None, 0, 1]},
        index=pd.RangeIndex(start=5, stop=9),
    )
    dask_cudf.from_cudf(gdf, npartitions=2).to_parquet(fn)
    ddf2 = dask_cudf.read_parquet(fn)
    dd.assert_eq(gdf, ddf2, check_index=True)


def test_roundtrip_from_pandas(tmpdir):
    fn = str(tmpdir.join("test.parquet"))

    # First without specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=False)
    dfp = dfp.reset_index(drop=True)
    ddf2 = dask_cudf.read_parquet(fn)
    dd.assert_eq(dfp, ddf2, check_index=True)

    # Now, specifying an index
    dfp = df.copy()
    dfp.to_parquet(fn, engine="pyarrow", index=True)
    ddf2 = dask_cudf.read_parquet(fn, index=["index"])
    dd.assert_eq(dfp, ddf2, check_index=True)


def test_strings(tmpdir):

    fn = str(tmpdir)
    dfp = pd.DataFrame(
        {"a": ["aa", "bbb", "cccc"], "b": ["hello", "dog", "man"]}
    )
    dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)
    ddf2.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn, index=["a"])
    dd.assert_eq(ddf2, read_df.compute().to_pandas())

    read_df_cats = dask_cudf.read_parquet(
        fn, index=["a"], strings_to_categorical=True
    )
    dd.assert_eq(read_df_cats.dtypes, read_df_cats.compute().dtypes)
    dd.assert_eq(read_df_cats.dtypes[0], "int32")


def test_dask_timeseries_from_pandas(tmpdir):

    fn = str(tmpdir.join("test.parquet"))
    ddf2 = dask.datasets.timeseries(freq="D")
    pdf = ddf2.compute()
    pdf.to_parquet(fn, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    dd.assert_eq(ddf2, read_df.compute())


@pytest.mark.parametrize("index", [False, None])
@pytest.mark.parametrize("stats", [False, True])
def test_dask_timeseries_from_dask(tmpdir, index, stats):

    fn = str(tmpdir)
    ddf2 = dask.datasets.timeseries(freq="D")
    ddf2.to_parquet(fn, engine="pyarrow", write_index=index)
    read_df = dask_cudf.read_parquet(fn, index=index, gather_statistics=stats)
    dd.assert_eq(
        ddf2, read_df, check_divisions=(stats and index), check_index=index
    )


@pytest.mark.parametrize("index", [False, None])
@pytest.mark.parametrize("stats", [False, True])
def test_dask_timeseries_from_daskcudf(tmpdir, index, stats):

    fn = str(tmpdir)
    ddf2 = dask_cudf.from_cudf(
        cudf.datasets.timeseries(freq="D"), npartitions=4
    )
    ddf2.name = ddf2.name.astype("object")
    ddf2.to_parquet(fn, write_index=index)
    read_df = dask_cudf.read_parquet(fn, index=index, gather_statistics=stats)
    dd.assert_eq(
        ddf2, read_df, check_divisions=(stats and index), check_index=index
    )


@pytest.mark.parametrize("index", [False, True])
def test_empty(tmpdir, index):

    fn = str(tmpdir)
    dfp = pd.DataFrame({"a": [11.0, 12.0, 12.0], "b": [4, 5, 6]})[:0]
    if index:
        dfp.set_index("a", inplace=True, drop=True)
    ddf2 = dd.from_pandas(dfp, npartitions=2)

    ddf2.to_parquet(fn, write_index=index, engine="pyarrow")
    read_df = dask_cudf.read_parquet(fn)
    dd.assert_eq(ddf2, read_df.compute())


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


def test_filters_at_row_group_level(tmpdir):

    tmp_path = str(tmpdir)
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    ddf = dd.from_pandas(df, npartitions=5)
    assert ddf.npartitions == 5

    ddf.to_parquet(tmp_path, engine="pyarrow", row_group_size=10 / 5)

    a = dask_cudf.read_parquet(tmp_path, filters=[("x", "==", 1)])
    assert a.npartitions == 1
    assert (a.shape[0] == 2).compute()

    ddf.to_parquet(tmp_path, engine="pyarrow", row_group_size=1)

    b = dask_cudf.read_parquet(tmp_path, filters=[("x", "==", 1)])
    assert b.npartitions == 1
    assert (b.shape[0] == 1).compute()


@pytest.mark.parametrize("metadata", [True, False])
@pytest.mark.parametrize("daskcudf", [True, False])
@pytest.mark.parametrize(
    "parts", [["year", "month", "day"], ["year", "month"], ["year"]]
)
def test_roundtrip_from_dask_partitioned(tmpdir, parts, daskcudf, metadata):
    tmpdir = str(tmpdir)

    df = pd.DataFrame()
    df["year"] = [2018, 2019, 2019, 2019, 2020, 2021]
    df["month"] = [1, 2, 3, 3, 3, 2]
    df["day"] = [1, 1, 1, 2, 2, 1]
    df["data"] = [0, 0, 0, 0, 0, 0]
    df.index.name = "index"
    if daskcudf:
        ddf2 = dask_cudf.from_cudf(cudf.from_pandas(df), npartitions=2)
        ddf2.to_parquet(
            tmpdir, write_metadata_file=metadata, partition_on=parts
        )
    else:
        ddf2 = dd.from_pandas(df, npartitions=2)
        ddf2.to_parquet(
            tmpdir,
            engine="pyarrow",
            write_metadata_file=metadata,
            partition_on=parts,
        )
    df_read = dd.read_parquet(tmpdir, engine="pyarrow")
    gdf_read = dask_cudf.read_parquet(tmpdir)

    # TODO: Avoid column selection after `CudfEngine`
    # can be aligned with dask/dask#6534
    columns = list(df_read.columns)
    assert set(df_read.columns) == set(gdf_read.columns)
    dd.assert_eq(
        df_read.compute(scheduler=dask.get)[columns],
        gdf_read.compute(scheduler=dask.get)[columns],
    )

    assert gdf_read.index.name == "index"

    # Check that we don't have uuid4 file names
    for _, _, files in os.walk(tmpdir):
        for fn in files:
            if not fn.startswith("_"):
                assert "part" in fn


@pytest.mark.parametrize("metadata", [True, False])
@pytest.mark.parametrize("chunksize", [None, 1024, 4096, "1MiB"])
def test_chunksize(tmpdir, chunksize, metadata):
    nparts = 2
    df_size = 100
    row_group_size = 5
    row_group_byte_size = 451  # Empirically measured

    df = pd.DataFrame(
        {
            "a": np.random.choice(["apple", "banana", "carrot"], size=df_size),
            "b": np.random.random(size=df_size),
            "c": np.random.randint(1, 5, size=df_size),
            "index": np.arange(0, df_size),
        }
    ).set_index("index")

    ddf1 = dd.from_pandas(df, npartitions=nparts)
    ddf1.to_parquet(
        str(tmpdir),
        engine="pyarrow",
        row_group_size=row_group_size,
        write_metadata_file=metadata,
    )

    if metadata:
        path = str(tmpdir)
    else:
        dirname = str(tmpdir)
        files = os.listdir(dirname)
        assert "_metadata" not in files
        path = os.path.join(dirname, "*.parquet")

    ddf2 = dask_cudf.read_parquet(
        path,
        chunksize=chunksize,
        split_row_groups=True,
        gather_statistics=True,
    )

    dd.assert_eq(ddf1, ddf2, check_divisions=False)

    num_row_groups = df_size // row_group_size
    if not chunksize:
        assert ddf2.npartitions == num_row_groups
    else:
        # Check that we are really aggregating
        df_byte_size = row_group_byte_size * num_row_groups
        expected = df_byte_size // parse_bytes(chunksize)
        remainder = (df_byte_size % parse_bytes(chunksize)) > 0
        expected += int(remainder) * nparts
        assert ddf2.npartitions == max(nparts, expected)


@pytest.mark.parametrize("row_groups", [1, 3, 10, 12])
@pytest.mark.parametrize("index", [False, True])
def test_row_groups_per_part(tmpdir, row_groups, index):
    nparts = 2
    df_size = 100
    row_group_size = 5
    file_row_groups = 10  # Known apriori
    npartitions_expected = math.ceil(file_row_groups / row_groups) * 2

    df = pd.DataFrame(
        {
            "a": np.random.choice(["apple", "banana", "carrot"], size=df_size),
            "b": np.random.random(size=df_size),
            "c": np.random.randint(1, 5, size=df_size),
            "index": np.arange(0, df_size),
        }
    )
    if index:
        df = df.set_index("index")

    ddf1 = dd.from_pandas(df, npartitions=nparts)
    ddf1.to_parquet(
        str(tmpdir),
        engine="pyarrow",
        row_group_size=row_group_size,
        write_metadata_file=True,
    )

    ddf2 = dask_cudf.read_parquet(str(tmpdir), row_groups_per_part=row_groups,)

    dd.assert_eq(ddf1, ddf2, check_divisions=False)

    assert ddf2.npartitions == npartitions_expected

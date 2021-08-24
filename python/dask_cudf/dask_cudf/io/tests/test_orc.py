import glob
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from dask import dataframe as dd

import cudf

import dask_cudf

# import pyarrow.orc as orc

cur_dir = os.path.dirname(__file__)
sample_orc = os.path.join(cur_dir, "sample.orc")


@pytest.mark.parametrize("legacy", [True, False])
def test_read_orc_defaults(legacy):
    df1 = cudf.read_orc(sample_orc)
    df2 = dask_cudf.read_orc(sample_orc, legacy=legacy)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("legacy", [True, False])
def test_filepath_read_orc_defaults(legacy):
    path = "file://%s" % sample_orc
    df1 = cudf.read_orc(path)
    df2 = dask_cudf.read_orc(path, legacy=legacy)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("legacy", [True, False])
def test_filelist_read_orc_defaults(legacy):
    path = [sample_orc]
    df1 = cudf.read_orc(path[0])
    df2 = dask_cudf.read_orc(path, legacy=legacy)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize("columns", [["time", "date"], ["time"]])
@pytest.mark.parametrize("legacy", [True, False])
def test_read_orc_cols(engine, columns, legacy):
    df1 = cudf.read_orc(sample_orc, engine=engine, columns=columns)

    df2 = dask_cudf.read_orc(
        sample_orc, engine=engine, columns=columns, legacy=legacy
    )

    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize(
    "predicate,expected_len",
    [
        # (None, 70_000),
        (
            [("date", "==", datetime(1900, 12, 25, tzinfo=timezone.utc))],
            15_000,
        ),
        (
            [("date", "<=", datetime(1928, 12, 25, tzinfo=timezone.utc))],
            30_000,
        ),
        (
            [
                [("date", ">", datetime(1950, 12, 25, tzinfo=timezone.utc))],
                [("date", "<=", datetime(1928, 12, 25, tzinfo=timezone.utc))],
            ],
            55_000,
        ),
    ],
)
def test_read_orc_filtered(tmpdir, legacy, engine, predicate, expected_len):
    df = dask_cudf.read_orc(
        sample_orc, engine=engine, filters=predicate, legacy=legacy
    )
    df.compute(scheduler="synchronous")
    dd.assert_eq(len(df), expected_len)


@pytest.mark.parametrize("legacy", [True, False])
def test_read_orc_first_file_empty(tmpdir, legacy):

    # Write a 3-file dataset where the first file is empty
    # See: https://github.com/rapidsai/cudf/issues/8011
    path = str(tmpdir)
    os.makedirs(path, exist_ok=True)
    df1 = cudf.DataFrame({"id": [1, 2], "float": [1.0, 2.0]})
    df1.iloc[:0].to_orc(os.path.join(path, "data.0"))
    df1.iloc[:1].to_orc(os.path.join(path, "data.1"))
    df1.iloc[1:].to_orc(os.path.join(path, "data.2"))

    # Read back the files with dask_cudf,
    # and check the result.
    df2 = dask_cudf.read_orc(os.path.join(path, "*"), legacy=legacy)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("compute", [True, False])
@pytest.mark.parametrize("compression", [None, "snappy"])
@pytest.mark.parametrize(
    "dtypes",
    [
        {"index": int, "c": int, "a": str},
        {"index": int, "c": int, "a": str, "b": float},
        {"index": int, "c": str, "a": object},
    ],
)
def test_to_orc(tmpdir, dtypes, compression, compute):

    # Create cudf and dask_cudf dataframes
    df = cudf.datasets.randomdata(nrows=10, dtypes=dtypes, seed=1)
    df = df.set_index("index").sort_index()
    ddf = dask_cudf.from_cudf(df, npartitions=3)

    # Write cudf dataframe as single file
    # (preserve index by setting to column)
    fname = tmpdir.join("test.orc")
    df.reset_index().to_orc(fname, compression=compression)

    # Write dask_cudf dataframe as multiple files
    # (preserve index by `write_index=True`)
    to = ddf.to_orc(
        str(tmpdir), write_index=True, compression=compression, compute=compute
    )

    if not compute:
        to.compute()

    # Read back cudf dataframe
    df_read = cudf.read_orc(fname).set_index("index")

    # Read back dask_cudf dataframe
    paths = glob.glob(str(tmpdir) + "/part.*.orc")
    ddf_read = dask_cudf.read_orc(paths).set_index("index")

    # Make sure the dask_cudf dataframe matches
    # the cudf dataframes (df and df_read)
    dd.assert_eq(df, ddf_read)
    dd.assert_eq(df_read, ddf_read)


@pytest.mark.parametrize("split_stripes", [True, False, 5])
@pytest.mark.parametrize("aggregate_files", [False, True])
@pytest.mark.parametrize("filters", [None, [("a", ">=", 4)]])
@pytest.mark.parametrize("sample_data", [True, False])
def test_orc_roundtrip_aggregate_files(
    tmpdir, split_stripes, aggregate_files, filters, sample_data,
):

    # Test general round-trip behavior of non-partitioned orc data.
    # We check that filtering, stripe aggregation and file aggregation
    # behave as expected.

    tmp = str(tmpdir)
    data = cudf.DataFrame(
        {
            "a": np.arange(100, dtype=np.float64) // 20,
            "b": np.random.choice(["cat", "dog", "mouse"], size=100),
        }
    )
    df = dask_cudf.from_cudf(data, npartitions=5)
    df.to_orc(tmp, write_index=False)
    df2 = dask_cudf.read_orc(
        tmp,
        split_stripes=split_stripes,
        aggregate_files=aggregate_files,
        filters=filters,
    )

    if filters:
        # Check that partitions were filtered.
        # We assume that making 5 partitions
        # along 5 unique values will produce a
        # predictiable output-partitioning.
        assert df2.npartitions == 1

        # Check that the data is correct
        expect = data[data.a >= 4.0]
        dd.assert_eq(expect, df2, check_index=False)

    else:
        # In the absense of filetering
        if aggregate_files:
            if split_stripes:
                assert df2.npartitions == int(
                    df.npartitions / int(split_stripes)
                )
            else:
                assert df2.npartitions == df.npartitions

            # Catch-all data check
            dd.assert_eq(data, df2, check_index=False)


@pytest.mark.parametrize("columns", [None, ["b", "a1"]])
@pytest.mark.parametrize("aggregate_files", [False, "a1"])
@pytest.mark.parametrize("split_stripes", [False, 2, 100])
@pytest.mark.parametrize("index", [None, "a1"])
@pytest.mark.parametrize("filters", [None, [("a1", "<", "C")]])
def test_partition_on(
    tmpdir, columns, aggregate_files, split_stripes, index, filters
):

    # Test general round-trip behavior of partitioned orc data.
    # We check that filtering, stripe aggregation, file aggregation,
    # and index setting behave as expected.

    df = cudf.DataFrame(
        {
            "a1": ["A"] * 33 + ["B"] * 33 + ["C"] * 34,
            "a2": np.random.choice([1, 2, 3], size=100),
            "b": np.random.random(size=100),
            "c": np.random.randint(1, 5, size=100),
            "d": np.arange(0, 100),
        }
    )
    if index:
        df = df.set_index(index)
    d = dask_cudf.from_cudf(df, npartitions=2)
    d.to_orc(tmpdir, partition_on=["a1", "a2"], write_index=bool(index))

    # Read back
    ddf = dask_cudf.read_orc(
        tmpdir,
        index=index,
        columns=columns,
        filters=filters,
        aggregate_files=aggregate_files,
        split_stripes=split_stripes,
    )

    # Check that the final index is correct,
    # but reset it to simplify data checks
    assert ddf.index.name == df.index.name
    if df.index.name is not None:
        df = df.reset_index(drop=False)
        ddf = ddf.reset_index(drop=False)

    # Check file aggregation
    if aggregate_files and split_stripes == 100:
        if filters:
            assert ddf.npartitions == 2
        else:
            assert ddf.npartitions == 3
    out = ddf.compute()
    if filters:
        df = df[df.a1 < "C"]
    pdf = df.to_pandas()
    pout = out.to_pandas()
    for val in pdf.a1.unique():
        assert set(pdf.b[pdf.a1 == val]) == set(pout.b[pout.a1 == val])

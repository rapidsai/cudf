import os

import pytest

import dask.dataframe as dd

import cudf

import dask_cudf

# import pyarrow.orc as orc

cur_dir = os.path.dirname(__file__)
sample_orc = os.path.join(cur_dir, "sample.orc")


def test_read_orc_defaults():
    df1 = cudf.read_orc(sample_orc)
    df2 = dask_cudf.read_orc(sample_orc)
    df2.head().to_pandas()
    dd.assert_eq(df1, df2, check_index=False)


def test_filepath_read_orc_defaults():
    path = "file://%s" % sample_orc
    df1 = cudf.read_orc(path)
    df2 = dask_cudf.read_orc(path)
    dd.assert_eq(df1, df2, check_index=False)


def test_filelist_read_orc_defaults():
    path = [sample_orc]
    df1 = cudf.read_orc(path[0])
    df2 = dask_cudf.read_orc(path)
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize("columns", [["time", "date"], ["time"]])
def test_read_orc_cols(engine, columns):
    df1 = cudf.read_orc(sample_orc, engine=engine, columns=columns)

    df2 = dask_cudf.read_orc(sample_orc, engine=engine, columns=columns)

    dd.assert_eq(df1, df2, check_index=False)

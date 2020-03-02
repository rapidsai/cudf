import numpy as np
import pytest

import dask
import dask.dataframe as dd

import cudf


@pytest.mark.parametrize("by", ["a", "b", ["a", "b"]])
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 5, 10])
def test_sort_values(nelem, nparts, by):
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    ddf = dd.from_pandas(df, npartitions=nparts)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by)
    expect = df.sort_values(by=by)
    dd.assert_eq(got, expect, check_index=False)


@pytest.mark.parametrize("by", ["a", "b", ["a", "b"]])
def test_sort_values_single_partition(by):
    df = cudf.DataFrame()
    nelem = 1000
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    ddf = dd.from_pandas(df, npartitions=1)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by)
    expect = df.sort_values(by=by)
    dd.assert_eq(got, expect)

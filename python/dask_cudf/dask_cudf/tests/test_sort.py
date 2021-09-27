import numpy as np
import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf
from dask_cudf.sorting import quantile_divisions


@pytest.mark.parametrize("by", ["a", "b", "c", "d", ["a", "b"], ["c", "d"]])
@pytest.mark.parametrize("nelem", [10, 500])
@pytest.mark.parametrize("nparts", [1, 10])
def test_sort_values(nelem, nparts, by):
    np.random.seed(0)
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    df["c"] = df["b"].astype("str")
    df["d"] = df["a"].astype("category")
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


def test_sort_repartition():
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"a": [0, 0, 1, 2, 3, 4, 2]}), npartitions=2
    )

    new_ddf = ddf.shuffle(on="a", ignore_index=True, npartitions=3)

    dd.assert_eq(len(new_ddf), len(ddf))


@pytest.mark.parametrize("by", ["a", "b", ["a", "b"]])
def test_sort_values_with_nulls(by):
    df = cudf.DataFrame(
        {
            "a": list(range(50)) + [None] * 50 + list(range(50, 100)),
            "b": [None] * 100 + list(range(100, 150)),
        }
    )
    ddf = dd.from_pandas(df, npartitions=10)

    # assert that quantile divisions of dataframe contains nulls
    divisions = quantile_divisions(ddf, by, ddf.npartitions)
    if isinstance(divisions, list):
        assert None in divisions
    else:
        assert all([divisions[col].has_nulls for col in by])

    got = ddf.sort_values(by=by)
    expect = df.sort_values(by=by)

    dd.assert_eq(got, expect)
    

# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf
from dask_cudf.tests.utils import xfail_dask_expr


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        "c",
        "d",
        ["a", "b"],
        ["c", "d"],
    ],
)
@pytest.mark.parametrize("nelem", [10, 500])
@pytest.mark.parametrize("nparts", [1, 10])
def test_sort_values(nelem, nparts, by, ascending):
    _ = np.random.default_rng(seed=0)
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    df["c"] = df["b"].astype("str")
    df["d"] = df["a"].astype("category")
    ddf = dd.from_pandas(df, npartitions=nparts)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by, ascending=ascending)
    expect = df.sort_values(by=by, ascending=ascending)
    dd.assert_eq(got, expect, check_index=False)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("by", ["a", "b", ["a", "b"]])
def test_sort_values_single_partition(by, ascending):
    df = cudf.DataFrame()
    nelem = 1000
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    ddf = dd.from_pandas(df, npartitions=1)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by, ascending=ascending)
    expect = df.sort_values(by=by, ascending=ascending)
    dd.assert_eq(got, expect)


def test_sort_repartition():
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"a": [0, 0, 1, 2, 3, 4, 2]}), npartitions=2
    )

    new_ddf = ddf.shuffle(on="a", ignore_index=True, npartitions=3)

    dd.assert_eq(len(new_ddf), len(ddf))


@xfail_dask_expr("missing null support", lt_version="2024.5.1")
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("by", ["a", "b", ["a", "b"]])
@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [None] * 100 + list(range(100, 150)),
            "b": list(range(50)) + [None] * 50 + list(range(50, 100)),
        },
        {"a": list(range(15)) + [None] * 5, "b": list(reversed(range(20)))},
    ],
)
def test_sort_values_with_nulls(data, by, ascending, na_position):
    _ = np.random.default_rng(seed=0)
    cp.random.seed(0)
    df = cudf.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=5)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(
            by=by, ascending=ascending, na_position=na_position
        )
        expect = df.sort_values(
            by=by, ascending=ascending, na_position=na_position
        )

    # cudf ordering for nulls is non-deterministic
    dd.assert_eq(got[by], expect[by], check_index=False)


@pytest.mark.parametrize("by", [["a", "b"], ["b", "a"]])
@pytest.mark.parametrize("nparts", [1, 10])
def test_sort_values_custom_function(by, nparts):
    df = cudf.DataFrame({"a": [1, 2, 3] * 20, "b": [4, 5, 6, 7] * 15})
    ddf = dd.from_pandas(df, npartitions=nparts)

    def f(partition, by_columns, ascending, na_position, **kwargs):
        return partition.sort_values(
            by_columns, ascending=ascending, na_position=na_position
        )

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(
            by=by[0], sort_function=f, sort_function_kwargs={"by_columns": by}
        )
    expect = df.sort_values(by=by)
    dd.assert_eq(got, expect, check_index=False)


@pytest.mark.parametrize("by", ["a", "b", ["a", "b"], ["b", "a"]])
def test_sort_values_empty_string(by):
    df = cudf.DataFrame({"a": [3, 2, 1, 4], "b": [""] * 4})
    ddf = dd.from_pandas(df, npartitions=2)
    got = ddf.sort_values(by)
    if "a" in by:
        expect = df.sort_values(by)
        assert dd.assert_eq(got, expect, check_index=False)


def test_disk_shuffle():
    df = cudf.DataFrame({"a": [1, 2, 3] * 20, "b": [4, 5, 6, 7] * 15})
    ddf = dd.from_pandas(df, npartitions=4)
    got = dd.DataFrame.shuffle(ddf, "a", shuffle_method="disk")
    dd.assert_eq(got, df)

# Copyright (c) 2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture
def df():
    df = cudf.DataFrame()
    rng = np.random.default_rng(seed=0)

    arr = rng.integers(2, size=10, dtype=np.int64)
    df["foo"] = arr
    df["bar"] = cudf.Series([pd.Timestamp(x) for x in arr])

    return df


@pytest.fixture(params=["foo", "bar"])
def series_test_vals(request, df):
    actual = cudf.unique(df[request.param])
    expected = pd.unique(df[request.param].to_pandas())
    return actual, expected


def test_unique_series_obj(series_test_vals):
    actual, expected = series_test_vals

    assert isinstance(expected, np.ndarray)
    assert isinstance(actual, cudf.Series)
    assert_eq(actual, pd.Series(expected, name=actual.name))


@pytest.mark.parametrize(
    "index",
    [
        (cudf.Index, pd.Index),
        (cudf.MultiIndex, pd.MultiIndex),
        (cudf.DatetimeIndex, pd.DatetimeIndex),
        (cudf.CategoricalIndex, pd.CategoricalIndex),
    ],
)
@pytest.mark.parametrize("col", ["foo", "bar"])
def test_unique_index_obj(index, col, df):
    if index[0] == cudf.MultiIndex:
        df.index = cudf.MultiIndex.from_arrays([df[col], df[col]])
    else:
        df.index = index[0](df[col])
    actual = cudf.unique(df.index)
    expected = pd.unique(df.index.to_pandas())

    isinstance(expected, np.ndarray)
    assert isinstance(actual, index[0])

    if index[0] == cudf.MultiIndex:
        expect = index[1].from_arrays(
            [
                [x[0] for x in expected],
                [x[1] for x in expected],
            ],
            names=actual.names,
        )
        assert_eq(actual, expect)
    else:
        assert_eq(actual, index[1](expected, name=actual.name))


def test_unique_cupy_ndarray(df):
    arr = np.asarray(df["foo"].to_pandas())
    garr = cp.asarray(df["foo"])

    expected = pd.unique(arr)
    actual = cudf.unique(garr)

    isinstance(expected, np.ndarray)
    isinstance(actual, cp.ndarray)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "abc", "a", "def", None],
        [10, 20, 100, -10, 0, 1, None, 10, 100],
    ],
)
def test_category_dtype_unique(data):
    gs = cudf.Series(data, dtype="category")
    ps = gs.to_pandas()

    actual = cudf.unique(gs)
    expected = pd.unique(ps)

    assert isinstance(expected, pd.Categorical)
    assert isinstance(actual, cudf.Series)
    assert_eq(actual, pd.Series(expected))


def test_unique_fails_value_error(df):
    with pytest.raises(
        ValueError,
        match="Must pass cudf.Series, cudf.Index, or cupy.ndarray object",
    ):
        cudf.unique(df)


def test_unique_fails_not_implemented_error(df):
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(
            NotImplementedError, match="cudf.Categorical is not implemented"
        ):
            cudf.unique(cudf.Series(["foo", "foo"], dtype="category"))

# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import datetime

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Series
from cudf.core.index import Index, RangeIndex
from cudf.testing._utils import DATETIME_TYPES, NUMERIC_TYPES, TIMEDELTA_TYPES


@pytest.mark.parametrize(
    "values, item, expected",
    [
        [[1, 2, 3], 2, True],
        [[1, 2, 3], 4, False],
        [[1, 2, 3], "a", False],
        [["a", "b", "c"], "a", True],
        [["a", "b", "c"], "ab", False],
        [["a", "b", "c"], 6, False],
        [pd.Categorical(["a", "b", "c"]), "a", True],
        [pd.Categorical(["a", "b", "c"]), "ab", False],
        [pd.Categorical(["a", "b", "c"]), 6, False],
        [pd.date_range("20010101", periods=5, freq="D"), 20000101, False],
        [
            pd.date_range("20010101", periods=5, freq="D"),
            datetime.datetime(2000, 1, 1),
            False,
        ],
        [
            pd.date_range("20010101", periods=5, freq="D"),
            datetime.datetime(2001, 1, 1),
            True,
        ],
    ],
)
@pytest.mark.parametrize(
    "box", [Index, lambda x: Series(index=x)], ids=["index", "series"]
)
def test_contains(values, item, expected, box):
    assert (item in box(values)) is expected


def test_rangeindex_contains():
    ridx = RangeIndex(start=0, stop=10, name="Index")
    assert 9 in ridx
    assert 10 not in ridx


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_lists_contains(dtype):
    dtype = cudf.dtype(dtype)
    inner_data = np.array([1, 2, 3], dtype=dtype)

    data = Series([inner_data])

    contained_scalar = inner_data.dtype.type(2)
    not_contained_scalar = inner_data.dtype.type(42)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


@pytest.mark.parametrize("dtype", DATETIME_TYPES + TIMEDELTA_TYPES)
def test_lists_contains_datetime(dtype):
    dtype = cudf.dtype(dtype)
    inner_data = np.array([1, 2, 3])

    unit, _ = np.datetime_data(dtype)

    data = Series([inner_data])

    contained_scalar = inner_data.dtype.type(2)
    not_contained_scalar = inner_data.dtype.type(42)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


def test_lists_contains_bool():
    data = Series([[True, True, True]])

    contained_scalar = True
    not_contained_scalar = False

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]

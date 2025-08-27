# Copyright (c) 2025, NVIDIA CORPORATION.
import datetime

import numpy as np
import pandas as pd
import pytest

import cudf


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
    "box",
    [cudf.Index, lambda x: cudf.Series(index=x)],
    ids=["index", "series"],
)
def test_contains(values, item, expected, box):
    assert (item in box(values)) is expected


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, None],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        [1, 2, 3, 4, np.nan],
        [10, 9, 8, np.nan, 7],
        [10, 9, 8, 8, 7, np.nan],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["c", "d", "e", "f", None],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_index_is_unique_monotonic(testlist):
    index = cudf.Index(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing

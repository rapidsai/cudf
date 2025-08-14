# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf


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


def test_name():
    idx = cudf.Index(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_names():
    idx = cudf.Index([1, 2, 3], name="idx")
    assert idx.names == ("idx",)

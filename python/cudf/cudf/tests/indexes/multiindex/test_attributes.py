# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf import MultiIndex


def test_multiindex_is_unique_monotonic():
    pidx = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
        ],
    )
    pidx.names = ["alpha", "location", "weather", "sign"]
    gidx = cudf.from_pandas(pidx)

    assert pidx.is_unique == gidx.is_unique
    assert pidx.is_monotonic_increasing == gidx.is_monotonic_increasing
    assert pidx.is_monotonic_decreasing == gidx.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testarr",
    [
        (
            [
                ["bar", "bar", "foo", "foo", "qux", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "two"],
            ],
            ["first", "second"],
        ),
        (
            [
                ["bar", "bar", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two"],
            ],
            ["first", "second"],
        ),
    ],
)
def test_multiindex_tuples_is_unique_monotonic(testarr):
    tuples = list(zip(*testarr[0]))

    index = MultiIndex.from_tuples(tuples, names=testarr[1])
    index_pd = pd.MultiIndex.from_tuples(tuples, names=testarr[1])

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing

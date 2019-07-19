# Copyright (c) 2019, NVIDIA CORPORATION.

"""
Tests related to is_unique and is_monotonic attributes
"""
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.dataframe import MultiIndex, Series
from cudf.dataframe.index import (
    CategoricalIndex,
    DatetimeIndex,
    GenericIndex,
    RangeIndex,
    StringIndex,
)


@pytest.mark.parametrize("testrange", [(10, 20, 1), (0, -10, -1), (5, 5, 1)])
def test_range_index(testrange):

    index = RangeIndex(start=testrange[0], stop=testrange[1])
    index_pd = pd.RangeIndex(
        start=testrange[0], stop=testrange[1], step=testrange[2]
    )

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_generic_index(testlist):

    index = GenericIndex(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_string_index(testlist):

    index = StringIndex(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist", [["c", "d", "e", "f"], ["z", "y", "x", "r"]]
)
def test_categorical_index(testlist):

    # Assuming unordered catagorical data cannot be "monotonic"
    raw_cat = pd.Categorical(testlist, ordered=True)
    index = CategoricalIndex(raw_cat)
    index_pd = pd.CategoricalIndex(raw_cat)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-04-11 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-03-08 16:00:00",
            "2001-02-03 08:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
    ],
)
def test_datetime_index(testlist):

    index = DatetimeIndex(testlist)
    index_pd = pd.DatetimeIndex(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_series(testlist):
    series = Series(testlist)
    series_pd = pd.Series(testlist)

    assert series.is_unique == series_pd.is_unique
    assert series.is_monotonic == series_pd.is_monotonic
    assert series.is_monotonic_increasing == series_pd.is_monotonic_increasing
    assert series.is_monotonic_decreasing == series_pd.is_monotonic_decreasing


def test_multiindex():
    pdf = pd.DataFrame(np.random.rand(7, 5))
    pdf.index = pd.MultiIndex(
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
    pdf.index.names = ["alpha", "location", "weather", "sign"]
    gdf = cudf.from_pandas(pdf)

    assert pdf.index.is_unique == gdf.index.is_unique
    assert pdf.index.is_monotonic == gdf.index.is_monotonic
    assert (
        pdf.index.is_monotonic_increasing == gdf.index.is_monotonic_increasing
    )
    assert (
        pdf.index.is_monotonic_decreasing == gdf.index.is_monotonic_decreasing
    )


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
def test_multiindex_tuples(testarr):
    tuples = list(zip(*testarr[0]))

    index = MultiIndex.from_tuples(tuples, names=testarr[1])
    index_pd = pd.MultiIndex.from_tuples(tuples, names=testarr[1])

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic == index_pd.is_monotonic
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [10, 9, 8, 8, 7],
        [2.0, 5.0, 4.0, 3.0, 7.0],
        ["b", "d", "e", "a", "c"],
        ["frog", "cat", "bat", "dog"],
    ],
)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["ix", "loc", "getitem", None])
def test_get_slice_bound(testlist, side, kind):
    index = GenericIndex(testlist)
    index_pd = pd.Index(testlist)
    for label in testlist:
        assert index.get_slice_bound(
            label, side, kind
        ) == index_pd.get_slice_bound(label, side, kind)


@pytest.mark.parametrize("label", [1, 3, 5, 7, 9, 11])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["ix", "loc", "getitem", None])
def test_get_slice_bound_missing(label, side, kind):
    mylist = [2, 4, 6, 8, 10]
    index = GenericIndex(mylist)
    index_pd = pd.Index(mylist)
    assert index.get_slice_bound(
        label, side, kind
    ) == index_pd.get_slice_bound(label, side, kind)


@pytest.mark.xfail
@pytest.mark.parametrize("label", ["a", "c", "e", "g"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing_str(label, side):
    # Slicing for monotonic string indices not yet supported
    # when missing values are specified (allowed in pandas)
    mylist = ["b", "d", "f"]
    index = GenericIndex(mylist)
    index_pd = pd.Index(mylist)
    assert index.get_slice_bound(
        label, side, "getitem"
    ) == index_pd.get_slice_bound(label, side, "getitem")

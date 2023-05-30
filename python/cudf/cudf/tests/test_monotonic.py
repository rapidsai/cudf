# Copyright (c) 2019-2023, NVIDIA CORPORATION.

"""
Tests related to is_unique and is_monotonic attributes
"""
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import MultiIndex, Series
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    GenericIndex,
    RangeIndex,
)
from cudf.testing._utils import assert_eq, expect_warning_if


@pytest.mark.parametrize("testrange", [(10, 20, 1), (0, -10, -1), (5, 5, 1)])
def test_range_index(testrange):

    index = RangeIndex(
        start=testrange[0], stop=testrange[1], step=testrange[2]
    )
    index_pd = pd.RangeIndex(
        start=testrange[0], stop=testrange[1], step=testrange[2]
    )

    assert index.is_unique == index_pd.is_unique
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
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
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
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

    index = cudf.Index(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist", [["c", "d", "e", "f"], ["z", "y", "x", "r"]]
)
def test_categorical_index(testlist):

    # Assuming unordered categorical data cannot be "monotonic"
    raw_cat = pd.Categorical(testlist, ordered=True)
    index = CategoricalIndex(raw_cat)
    index_pd = pd.CategoricalIndex(raw_cat)

    assert index.is_unique == index_pd.is_unique
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
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
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
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
    with pytest.warns(FutureWarning):
        expect = series_pd.index.is_monotonic
    with pytest.warns(FutureWarning):
        got = series.index.is_monotonic
    assert got == expect
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
    with pytest.warns(FutureWarning):
        expect = pdf.index.is_monotonic
    with pytest.warns(FutureWarning):
        got = gdf.index.is_monotonic
    assert got == expect
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
    with pytest.warns(FutureWarning):
        expect = index_pd.is_monotonic
    with pytest.warns(FutureWarning):
        got = index.is_monotonic
    assert got == expect
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
@pytest.mark.parametrize("kind", ["loc", "getitem", None])
def test_get_slice_bound(testlist, side, kind):
    index = GenericIndex(testlist)
    index_pd = pd.Index(testlist)
    for label in testlist:
        with pytest.warns(FutureWarning):
            expect = index_pd.get_slice_bound(label, side, kind)
        with expect_warning_if(kind is not None, FutureWarning):
            got = index.get_slice_bound(label, side, kind)
        assert got == expect


@pytest.mark.parametrize("bounds", [(0, 10), (0, 1), (3, 4), (0, 0), (3, 3)])
@pytest.mark.parametrize(
    "indices",
    [[-1, 0, 5, 10, 11], [-1, 0, 1, 2], [2, 3, 4, 5], [-1, 0, 1], [2, 3, 4]],
)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["getitem", "loc"])
def test_rangeindex_get_slice_bound_basic(bounds, indices, side, kind):
    start, stop = bounds
    pd_index = pd.RangeIndex(start, stop)
    cudf_index = RangeIndex(start, stop)
    for idx in indices:
        with pytest.warns(FutureWarning):
            expect = pd_index.get_slice_bound(idx, side, kind)
        with expect_warning_if(kind is not None, FutureWarning):
            got = cudf_index.get_slice_bound(idx, side, kind)
        assert expect == got


@pytest.mark.parametrize(
    "bounds",
    [(3, 20, 5), (20, 3, -5), (20, 3, 5), (3, 20, -5), (0, 0, 2), (3, 3, 2)],
)
@pytest.mark.parametrize(
    "label",
    [3, 8, 13, 18, 20, 15, 10, 5, -1, 0, 19, 21, 6, 11, 17],
)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["getitem", "loc"])
def test_rangeindex_get_slice_bound_step(bounds, label, side, kind):
    start, stop, step = bounds
    pd_index = pd.RangeIndex(start, stop, step)
    cudf_index = RangeIndex(start, stop, step)

    with pytest.warns(FutureWarning):
        expect = pd_index.get_slice_bound(label, side, kind)
    with expect_warning_if(kind is not None, FutureWarning):
        got = cudf_index.get_slice_bound(label, side, kind)
    assert expect == got


@pytest.mark.parametrize("label", [1, 3, 5, 7, 9, 11])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["loc", "getitem", None])
def test_get_slice_bound_missing(label, side, kind):
    mylist = [2, 4, 6, 8, 10]
    index = GenericIndex(mylist)
    index_pd = pd.Index(mylist)

    with pytest.warns(FutureWarning):
        expect = index_pd.get_slice_bound(label, side, kind)
    with expect_warning_if(kind is not None, FutureWarning):
        got = index.get_slice_bound(label, side, kind)
    assert got == expect


@pytest.mark.xfail
@pytest.mark.parametrize("label", ["a", "c", "e", "g"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing_str(label, side):
    # Slicing for monotonic string indices not yet supported
    # when missing values are specified (allowed in pandas)
    mylist = ["b", "d", "f"]
    index = GenericIndex(mylist)
    index_pd = pd.Index(mylist)
    with pytest.warns(FutureWarning):
        got = index.get_slice_bound(label, side, "getitem")
    with pytest.warns(FutureWarning):
        expect = index_pd.get_slice_bound(label, side, "getitem")
    assert got == expect


testdata = [
    (
        Series(["2018-01-01", "2019-01-31", None], dtype="datetime64[ms]"),
        False,
    ),
    (Series([1, 2, 3, None]), False),
    (Series([None, 1, 2, 3]), False),
    (Series(["a", "b", "c", None]), False),
    (Series([None, "a", "b", "c"]), False),
]


@pytest.mark.parametrize("data, expected", testdata)
def test_is_monotonic_always_falls_for_null(data, expected):
    assert_eq(expected, data.is_monotonic_increasing)
    assert_eq(expected, data.is_monotonic_decreasing)

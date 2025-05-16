# Copyright (c) 2019-2024, NVIDIA CORPORATION.

"""
Tests related to is_unique, is_monotonic_increasing &
is_monotonic_decreasing attributes
"""

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Index, MultiIndex, Series
from cudf.core.index import CategoricalIndex, DatetimeIndex, RangeIndex
from cudf.testing import assert_eq


@pytest.mark.parametrize("testrange", [(10, 20, 1), (0, -10, -1), (5, 5, 1)])
def test_range_index(testrange):
    index = RangeIndex(
        start=testrange[0], stop=testrange[1], step=testrange[2]
    )
    index_pd = pd.RangeIndex(
        start=testrange[0], stop=testrange[1], step=testrange[2]
    )

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, None],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["c", "d", "e", "f", None],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_generic_index(testlist):
    index = Index(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4, np.nan],
        [10, 9, 8, np.nan, 7],
        [10, 9, 8, 8, 7, np.nan],
    ],
)
def test_float_index(testlist):
    index_pd = pd.Index(testlist)
    index = cudf.from_pandas(index_pd, nan_as_null=False)

    assert index.is_unique == index_pd.is_unique
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
    assert series.is_monotonic_increasing == series_pd.is_monotonic_increasing
    assert series.is_monotonic_decreasing == series_pd.is_monotonic_decreasing


def test_multiindex():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
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
def test_get_slice_bound(testlist, side):
    index = Index(testlist)
    index_pd = pd.Index(testlist)
    for label in testlist:
        expect = index_pd.get_slice_bound(label, side)
        got = index.get_slice_bound(label, side)
        assert got == expect


@pytest.mark.parametrize("bounds", [(0, 10), (0, 1), (3, 4), (0, 0), (3, 3)])
@pytest.mark.parametrize(
    "indices",
    [[-1, 0, 5, 10, 11], [-1, 0, 1, 2], [2, 3, 4, 5], [-1, 0, 1], [2, 3, 4]],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_rangeindex_get_slice_bound_basic(bounds, indices, side):
    start, stop = bounds
    pd_index = pd.RangeIndex(start, stop)
    cudf_index = RangeIndex(start, stop)
    for idx in indices:
        expect = pd_index.get_slice_bound(idx, side)
        got = cudf_index.get_slice_bound(idx, side)
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
def test_rangeindex_get_slice_bound_step(bounds, label, side):
    start, stop, step = bounds
    pd_index = pd.RangeIndex(start, stop, step)
    cudf_index = RangeIndex(start, stop, step)

    expect = pd_index.get_slice_bound(label, side)
    got = cudf_index.get_slice_bound(label, side)
    assert expect == got


@pytest.mark.parametrize("label", [1, 3, 5, 7, 9, 11])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing(label, side):
    mylist = [2, 4, 6, 8, 10]
    index = Index(mylist)
    index_pd = pd.Index(mylist)

    expect = index_pd.get_slice_bound(label, side)
    got = index.get_slice_bound(label, side)
    assert got == expect


@pytest.mark.parametrize("label", ["a", "c", "e", "g"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing_str(label, side):
    mylist = ["b", "d", "f"]
    index = Index(mylist)
    index_pd = pd.Index(mylist)
    got = index.get_slice_bound(label, side)
    expect = index_pd.get_slice_bound(label, side)
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


@pytest.mark.parametrize("box", [Series, Index])
@pytest.mark.parametrize(
    "value,na_like",
    [
        [1, None],
        [np.datetime64("2020-01-01", "ns"), np.datetime64("nat", "ns")],
        ["s", None],
        [1.0, np.nan],
    ],
    ids=repr,
)
def test_is_unique(box, value, na_like):
    obj = box([value], nan_as_null=False)
    assert obj.is_unique

    obj = box([value, value], nan_as_null=False)
    assert not obj.is_unique

    obj = box([None, value], nan_as_null=False)
    assert obj.is_unique

    obj = box([None, None, value], nan_as_null=False)
    assert not obj.is_unique

    if na_like is not None:
        obj = box([na_like, value], nan_as_null=False)
        assert obj.is_unique

        obj = box([na_like, na_like], nan_as_null=False)
        assert not obj.is_unique

        try:
            if not np.isnat(na_like):
                # pyarrow coerces nat to null
                obj = box([None, na_like, value], nan_as_null=False)
                assert obj.is_unique
        except TypeError:
            pass

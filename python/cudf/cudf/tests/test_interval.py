# Copyright (c) 2020-2024, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_create_interval_series(data1, data2, data3, data4, closed):
    expect = pd.Series(pd.Interval(data1, data2, closed), dtype="interval")
    got = cudf.Series(pd.Interval(data1, data2, closed), dtype="interval")
    assert_eq(expect, got)

    expect_two = pd.Series(
        [pd.Interval(data1, data2, closed), pd.Interval(data3, data4, closed)],
        dtype="interval",
    )
    got_two = cudf.Series(
        [pd.Interval(data1, data2, closed), pd.Interval(data3, data4, closed)],
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.Series(
        [
            pd.Interval(data1, data2, closed),
            pd.Interval(data3, data4, closed),
            pd.Interval(data1, data2, closed),
        ],
        dtype="interval",
    )
    got_three = cudf.Series(
        [
            pd.Interval(data1, data2, closed),
            pd.Interval(data3, data4, closed),
            pd.Interval(data1, data2, closed),
        ],
        dtype="interval",
    )
    assert_eq(expect_three, got_three)


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_create_interval_df(data1, data2, data3, data4, closed):
    # df for both pandas and cudf only works when interval is in a list
    expect = pd.DataFrame(
        [pd.Interval(data1, data2, closed)], dtype="interval"
    )
    got = cudf.DataFrame([pd.Interval(data1, data2, closed)], dtype="interval")
    assert_eq(expect, got)

    expect_two = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
            "b": [
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
            ],
        },
        dtype="interval",
    )
    got_two = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
            "b": [
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
            ],
            "b": [
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
            "c": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
        },
        dtype="interval",
    )

    got_three = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
            ],
            "b": [
                pd.Interval(data3, data4, closed),
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
            "c": [
                pd.Interval(data1, data2, closed),
                pd.Interval(data1, data2, closed),
                pd.Interval(data3, data4, closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_three, got_three)


def test_create_interval_index_from_list():
    interval_list = [
        np.nan,
        pd.Interval(2.0, 3.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
    ]

    expected = pd.Index(interval_list)
    actual = cudf.Index(interval_list)

    assert_eq(expected, actual)


def test_interval_index_unique():
    interval_list = [
        np.nan,
        pd.Interval(2.0, 3.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
        np.nan,
        pd.Interval(3.0, 4.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
    ]
    pi = pd.Index(interval_list)
    gi = cudf.from_pandas(pi)

    expected = pi.unique()
    actual = gi.unique()

    assert_eq(expected, actual)


@pytest.mark.parametrize("box", [pd.Series, pd.IntervalIndex])
@pytest.mark.parametrize("tz", ["US/Eastern", None])
def test_interval_with_datetime(tz, box):
    dti = pd.date_range(
        start=pd.Timestamp("20180101", tz=tz),
        end=pd.Timestamp("20181231", tz=tz),
        freq="ME",
    )
    pobj = box(pd.IntervalIndex.from_breaks(dti))
    if tz is None:
        gobj = cudf.from_pandas(pobj)
        assert_eq(pobj, gobj)
    else:
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pobj)


def test_from_pandas_intervaldtype():
    dtype = pd.IntervalDtype("int64", closed="left")
    result = cudf.from_pandas(dtype)
    expected = cudf.IntervalDtype("int64", closed="left")
    assert_eq(result, expected)

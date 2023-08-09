# Copyright (c) 2020-2023, NVIDIA CORPORATION.

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


@pytest.mark.parametrize(
    "start, stop, freq, periods",
    [
        (0, 10, 2, None),
        (0, None, 2, 5),
        (0, 10, None, 5),
        (None, 10, 2, 5),
        (0, 10, None, 11),
        (0, 10, None, 10),
        (0.0, None, 0.2, 5),
        (0.0, 1.0, None, 5),
        # (0.0, 1.0, 0.2, None), # Pandas returns only 4 intervals here
        (None, 1.0, 0.2, 5),
        # (0.0, 1.0, 0.1, None), # Pandas returns the wrong result here
        (0.0, 1.0, None, 10),
    ],
)
def test_interval_range(start, stop, freq, periods):
    expected = pd.interval_range(
        start=start, end=stop, freq=freq, periods=periods
    )
    got = cudf.interval_range(
        start=start, end=stop, freq=freq, periods=periods
    )
    assert_eq(expected, got)

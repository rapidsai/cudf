# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data1, data2", [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
def test_create_interval_series(data1, data2, data3, data4):
    expect = pd.Series(pd.Interval(data1, data2), dtype="interval")
    got = cudf.Series(pd.Interval(data1, data2), dtype="interval")
    assert_eq(expect, got)

    expect_two = pd.Series(
        [pd.Interval(data1, data2), pd.Interval(data3, data4)],
        dtype="interval",
    )
    got_two = cudf.Series(
        [pd.Interval(data1, data2), pd.Interval(data3, data4)],
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.Series(
        [
            pd.Interval(data1, data2),
            pd.Interval(data3, data4),
            pd.Interval(data1, data2),
        ],
        dtype="interval",
    )
    got_three = cudf.Series(
        [
            pd.Interval(data1, data2),
            pd.Interval(data3, data4),
            pd.Interval(data1, data2),
        ],
        dtype="interval",
    )
    assert_eq(expect_three, got_three)

# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[None]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [[2, 3, 4]]},
        {"a": [[1, 2, 3, None], [None]], "b": [[2, 3, 4], [5]], "c": None},
        {"a": [[1]], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[1, 2, None, 4]]},
        {"a": [["cat", None, "dog"]]},
        {
            "a": [[1, 2, 3, None], [4, None, 5]],
            "b": [None, ["fish", "bird"]],
            "c": [[], []],
        },
        {"a": [[1, 2, 3, None], [4, None, 5], None, [6, 7]]},
    ],
)
def test_serialize_list_columns(data):
    df = cudf.DataFrame(data)
    recreated = df.__class__.deserialize(*df.serialize())
    assert_eq(recreated, df)


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
def test_create_interval_df(data1, data2, data3, data4, interval_closed):
    # df for both pandas and cudf only works when interval is in a list
    expect = pd.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    got = cudf.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    assert_eq(expect, got)

    expect_two = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    got_two = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )

    got_three = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_three, got_three)

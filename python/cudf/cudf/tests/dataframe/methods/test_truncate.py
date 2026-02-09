# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_dataframe_truncate_axis_0():
    df = cudf.DataFrame(
        {
            "A": ["a", "b", "c", "d", "e"],
            "B": ["f", "g", "h", "i", "j"],
            "C": ["k", "l", "m", "n", "o"],
        },
        index=[1, 2, 3, 4, 5],
    )
    pdf = df.to_pandas()

    expected = pdf.truncate(before=2, after=4, axis="index")
    actual = df.truncate(before=2, after=4, axis="index")
    assert_eq(actual, expected)

    expected = pdf.truncate(before=1, after=4, axis=0)
    actual = df.truncate(before=1, after=4, axis=0)
    assert_eq(expected, actual)


def test_dataframe_truncate_axis_1():
    df = cudf.DataFrame(
        {
            "A": ["a", "b", "c", "d", "e"],
            "B": ["f", "g", "h", "i", "j"],
            "C": ["k", "l", "m", "n", "o"],
        },
        index=[1, 2, 3, 4, 5],
    )
    pdf = df.to_pandas()

    expected = pdf.truncate(before="A", after="B", axis="columns")
    actual = df.truncate(before="A", after="B", axis="columns")
    assert_eq(actual, expected)

    expected = pdf.truncate(before="A", after="B", axis=1)
    actual = df.truncate(before="A", after="B", axis=1)
    assert_eq(actual, expected)


def test_dataframe_truncate_datetimeindex():
    dates = cudf.date_range(
        "2021-01-01 23:45:00", "2021-01-01 23:46:00", freq="s"
    )
    df = cudf.DataFrame(data={"A": 1, "B": 2}, index=dates)
    pdf = df.to_pandas()
    expected = pdf.truncate(
        before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
    )
    actual = df.truncate(
        before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
    )

    assert_eq(actual, expected)

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import copy
import textwrap

import numpy as np
import pandas as pd

import cudf


def test_dataframe_to_string_with_skipped_rows():
    # Test skipped rows
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    with pd.option_context("display.max_rows", 5):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a   b
        0   1  11
        1   2  12
        .. ..  ..
        4   5  15
        5   6  16

        [6 rows x 2 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_skipped_rows_and_columns():
    # Test skipped rows and skipped columns
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [11, 12, 13, 14, 15, 16],
            "c": [11, 12, 13, 14, 15, 16],
            "d": [11, 12, 13, 14, 15, 16],
        }
    )

    with pd.option_context("display.max_rows", 5, "display.max_columns", 3):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a  ...   d
        0   1  ...  11
        1   2  ...  12
        .. ..  ...  ..
        4   5  ...  15
        5   6  ...  16

        [6 rows x 4 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_masked_data():
    # Test masked data
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    data = np.arange(6)
    masked = cudf.Series(data)
    masked.iloc[[1, 4]] = None
    assert masked.null_count == 2
    df["c"] = masked

    # Check data
    values = masked.copy()
    validids = [0, 2, 3, 5]
    densearray = masked.dropna().to_numpy()
    np.testing.assert_equal(data[validids], densearray)
    # Valid position is correct
    for i in validids:
        assert data[i] == values[i]
    # Null position is correct
    for i in range(len(values)):
        if i not in validids:
            assert values[i] is cudf.NA

    with pd.option_context("display.max_rows", 10):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a   b     c
        0  1  11     0
        1  2  12  <NA>
        2  3  13     2
        3  4  14     3
        4  5  15  <NA>
        5  6  16     5"""
    )
    assert got == expect


def test_dataframe_to_string_wide():
    # Test basic
    df = cudf.DataFrame({f"a{i}": [0, 1, 2] for i in range(100)})
    with pd.option_context("display.max_columns", 16):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a0  a1  a2  a3  a4  a5  a6  a7  ...  a92  a93  a94  a95  a96  a97  a98  a99
        0   0   0   0   0   0   0   0   0  ...    0    0    0    0    0    0    0    0
        1   1   1   1   1   1   1   1   1  ...    1    1    1    1    1    1    1    1
        2   2   2   2   2   2   2   2   2  ...    2    2    2    2    2    2    2    2

        [3 rows x 100 columns]"""
    )
    assert got == expect


def test_dataframe_empty_to_string():
    # Test for printing empty dataframe
    df = cudf.DataFrame()
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: []\nIndex: []"
    assert got == expect


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = cudf.DataFrame()
    df["a"] = []
    df["b"] = []
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: [a, b]\nIndex: []"
    assert got == expect


def test_dataframe_copy():
    # Test for copying the dataframe using python copy pkg
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = copy.copy(df)
    df2["b"] = [4, 5, 6]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect


def test_dataframe_copy_shallow():
    # Test for copy dataframe using class method
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = df.copy()
    df2["b"] = [4, 2, 3]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect

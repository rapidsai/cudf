# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import textwrap

import numpy as np
import pandas as pd

import cudf


def test_dataframe_to_string_with_skipped_rows():
    # Test skipped rows: to_string truncates only when max_rows is passed
    # explicitly (matching pandas), not via the display.max_rows option.
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    got = df.to_string(max_rows=5)
    expect = df.to_pandas().to_string(max_rows=5)
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

    got = df.to_string(max_rows=5, max_cols=3)
    expect = df.to_pandas().to_string(max_rows=5, max_cols=3)
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

    got = df.to_string(max_rows=10)
    expect = df.to_pandas().to_string(max_rows=10)
    assert got == expect


def test_dataframe_to_string_wide():
    # Test basic
    df = cudf.DataFrame({f"a{i}": [0, 1, 2] for i in range(100)})
    got = df.to_string(max_cols=16)

    expect = df.to_pandas().to_string(max_cols=16)
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


def test_to_string_ignores_display_options():
    # Unlike repr, to_string does not truncate based on the
    # display.max_rows / display.max_columns options (matches pandas).
    pdf = pd.DataFrame({"a": list(range(100)), "b": list(range(100))})
    gdf = cudf.from_pandas(pdf)
    with pd.option_context("display.max_rows", 10, "display.max_columns", 1):
        assert gdf.to_string() == pdf.to_string()
    assert "..." not in gdf.to_string()


def test_series_to_string_no_dtype_footer():
    # Series.to_string omits the dtype footer by default (unlike repr).
    ps = pd.Series([0, 1, None], dtype="Int64")
    gs = cudf.from_pandas(ps)
    assert gs.to_string() == ps.to_string()
    assert "dtype:" not in gs.to_string()

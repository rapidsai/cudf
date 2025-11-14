# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": [1, 2, None], "b": [None, None, 5]}),
        pd.DataFrame(
            {"a": [1, 2, None], "b": [None, None, 5]}, index=["a", "p", "z"]
        ),
        pd.DataFrame({"a": [1, 2, 3]}),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        10,
        pd.Series([10, 20, 30]),
        pd.Series([3, 4, 5]),
        pd.Series([10, 20, 30], index=["z", "a", "p"]),
        {"a": 5, "b": pd.Series([3, 4, 5])},
        {"a": 5001},
        {"b": pd.Series([11, 22, 33], index=["a", "p", "z"])},
        {"a": 5, "b": pd.Series([3, 4, 5], index=["a", "p", "z"])},
        {"c": 100},
        np.nan,
    ],
)
def test_fillna_dataframe(pdf, value, inplace):
    if inplace:
        pdf = pdf.copy(deep=True)
    gdf = cudf.from_pandas(pdf)

    fill_value_pd = value
    if isinstance(fill_value_pd, (pd.Series, pd.DataFrame)):
        fill_value_cudf = cudf.from_pandas(fill_value_pd)
    elif isinstance(fill_value_pd, dict):
        fill_value_cudf = {}
        for key in fill_value_pd:
            temp_val = fill_value_pd[key]
            if isinstance(temp_val, pd.Series):
                temp_val = cudf.from_pandas(temp_val)
            fill_value_cudf[key] = temp_val
    else:
        fill_value_cudf = value

    expect = pdf.fillna(fill_value_pd, inplace=inplace)
    got = gdf.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gdf
        expect = pdf

    assert_eq(expect, got)


def test_fillna_columns_multiindex():
    columns = pd.MultiIndex.from_tuples([("a", "b"), ("d", "e")])
    pdf = pd.DataFrame(
        {"0": [1, 2, None, 3, None], "1": [None, None, None, None, 4]}
    )
    pdf.columns = columns
    gdf = cudf.from_pandas(pdf)

    expected = pdf.fillna(10)
    actual = gdf.fillna(10)

    assert_eq(expected, actual)


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]", "int8"])
def test_dataframe_fillna_preserves_column_dtype(dtype):
    result = cudf.DataFrame([1, None], columns=cudf.Index([1], dtype=dtype))
    result = result.fillna(2).columns
    expected = pd.Index([1], dtype=dtype)
    assert_eq(result, expected)


def test_dataframe_fillna_preserves_column_rangeindex():
    result = cudf.DataFrame([1, None], columns=range(1))
    result = result.fillna(2).columns
    expected = pd.RangeIndex(1)
    assert_eq(result, expected)

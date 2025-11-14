# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import decimal
import string

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_multiindex_transpose():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        index=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.transpose(), gdf.transpose())


@pytest.mark.parametrize("num_cols", [1, 3])
@pytest.mark.parametrize("num_rows", [1, 4])
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_dataframe_transpose(
    nulls, num_cols, num_rows, all_supported_types_as_str
):
    # In case of `bool` dtype: pandas <= 1.2.5 type-casts
    # a boolean series to `float64` series if a `np.nan` is assigned to it:
    # >>> s = pd.Series([True, False, True])
    # >>> s
    # 0     True
    # 1    False
    # 2     True
    # dtype: bool
    # >>> s[[2]] = np.nan
    # >>> s
    # 0    1.0
    # 1    0.0
    # 2    NaN
    # dtype: float64
    # In pandas >= 1.3.2 this behavior is fixed:
    # >>> s = pd.Series([True, False, True])
    # >>> s
    # 0
    # True
    # 1
    # False
    # 2
    # True
    # dtype: bool
    # >>> s[[2]] = np.nan
    # >>> s
    # 0
    # True
    # 1
    # False
    # 2
    # NaN
    # dtype: object
    # In cudf we change `object` dtype to `str` type - for which there
    # is no transpose implemented yet. Hence we need to test transpose
    # against pandas nullable types as they are the ones that closely
    # resemble `cudf` dtypes behavior.
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    pdf = pd.DataFrame()
    rng = np.random.default_rng(seed=0)
    null_rep = (
        np.nan
        if all_supported_types_as_str in ["float32", "float64"]
        else None
    )
    np_dtype = all_supported_types_as_str
    dtype = np.dtype(all_supported_types_as_str)
    dtype = cudf.utils.dtypes.np_dtypes_to_pandas_dtypes.get(dtype, dtype)
    for i in range(num_cols):
        colname = string.ascii_lowercase[i]
        data = pd.Series(
            rng.integers(0, 26, num_rows).astype(np_dtype),
            dtype=dtype,
        )
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            if len(idx):
                data[idx] = null_rep
        elif nulls == "all":
            data[:] = null_rep
        pdf[colname] = data

    gdf = cudf.DataFrame(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()
    nullable = dtype.kind not in "Mm"

    assert_eq(expect, got_function.to_pandas(nullable=nullable))
    assert_eq(expect, got_property.to_pandas(nullable=nullable))


@pytest.mark.parametrize(
    "data",
    [
        {"col": [{"a": 1.1}, {"a": 2.1}, {"a": 10.0}, {"a": 11.2323}, None]},
        {"a": [[{"b": 567}], None] * 10},
        {"a": [decimal.Decimal(10), decimal.Decimal(20), None]},
    ],
)
def test_dataframe_transpose_complex_types(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = pdf.T
    actual = gdf.T

    assert_eq(expected, actual)


def test_dataframe_transpose_category():
    pdf = pd.DataFrame(
        {
            "a": pd.Series(["a", "b", "c"], dtype="category"),
            "b": pd.Series(["a", "b", "c"], dtype="category"),
        }
    )

    gdf = cudf.DataFrame(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function.to_pandas())
    assert_eq(expect, got_property.to_pandas())


def test_transpose_multiindex_columns_from_pandas():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(gdf, pdf)
    assert_eq(gdf.T, pdf.T)

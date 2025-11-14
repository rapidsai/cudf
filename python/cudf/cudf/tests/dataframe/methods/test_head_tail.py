# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("nelem", [0, 10])
def test_head_tail(nelem, numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(0, 1000, nelem).astype(numeric_types_as_str),
            "b": rng.integers(0, 1000, nelem).astype(numeric_types_as_str),
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(gdf.head(), pdf.head())
    assert_eq(gdf.head(3), pdf.head(3))
    assert_eq(gdf.head(-2), pdf.head(-2))
    assert_eq(gdf.head(0), pdf.head(0))

    assert_eq(gdf["a"].head(), pdf["a"].head())
    assert_eq(gdf["a"].head(3), pdf["a"].head(3))
    assert_eq(gdf["a"].head(-2), pdf["a"].head(-2))

    assert_eq(gdf.tail(), pdf.tail())
    assert_eq(gdf.tail(3), pdf.tail(3))
    assert_eq(gdf.tail(-2), pdf.tail(-2))
    assert_eq(gdf.tail(0), pdf.tail(0))

    assert_eq(gdf["a"].tail(), pdf["a"].tail())
    assert_eq(gdf["a"].tail(3), pdf["a"].tail(3))
    assert_eq(gdf["a"].tail(-2), pdf["a"].tail(-2))


def test_tail_for_string():
    gdf = cudf.DataFrame({"id": ["a", "b"], "v": [1, 2]})
    assert_eq(gdf.tail(3), gdf.to_pandas().tail(3))


def test_dataframe_0_row_dtype(all_supported_types_as_str):
    data = cudf.Series([1, 2, 3, 4, 5], dtype=all_supported_types_as_str)

    expect = cudf.DataFrame({"x": data, "y": data})
    got = expect.head(0)

    for col_name in got.columns:
        assert expect[col_name].dtype == got[col_name].dtype

    expect = cudf.Series(data)
    got = expect.head(0)

    assert expect.dtype == got.dtype


def test_one_row_head():
    gdf = cudf.DataFrame({"name": ["carl"], "score": [100]}, index=[123])
    pdf = gdf.to_pandas()

    head_gdf = gdf.head()
    head_pdf = pdf.head()

    assert_eq(head_pdf, head_gdf)


@pytest.mark.parametrize("index", [None, [123], ["a", "b"]])
def test_no_cols_head(index):
    pdf = pd.DataFrame(index=index)
    gdf = cudf.from_pandas(pdf)

    head_gdf = gdf.head()
    head_pdf = pdf.head()

    assert_eq(head_pdf, head_gdf)


@pytest.mark.parametrize("n", [-10, -2, 0, 1])
def test_empty_df_head_tail_index(n):
    df = cudf.DataFrame()
    pdf = pd.DataFrame()
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)

    df = cudf.DataFrame({"a": [11, 2, 33, 44, 55]})
    pdf = pd.DataFrame({"a": [11, 2, 33, 44, 55]})
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)

    df = cudf.DataFrame(index=[1, 2, 3])
    pdf = pd.DataFrame(index=[1, 2, 3])
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)

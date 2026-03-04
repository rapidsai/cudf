# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_dataframe_pipe():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def add_int_col(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    def add_str_col(df, column):
        df[column] = df._constructor_sliced(["a", "b", "xyz", "ai"])
        return df

    expected = (
        pdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )
    actual = (
        gdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )

    assert_eq(expected, actual)

    expected = (
        pdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )
    actual = (
        gdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )

    assert_eq(expected, actual)


def test_dataframe_pipe_error():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def custom_func(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    assert_exceptions_equal(
        lfunc=pdf.pipe,
        rfunc=gdf.pipe,
        lfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
        rfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
    )

# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_groupby_results_equal
from cudf.testing.dataset_generator import rand_dataframe


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row(shift_perc, direction):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 4)), columns=["x", "y", "val", "val2"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).diff(periods=n_shift)
    got = gdf.groupby(["x", "y"]).diff(periods=n_shift)

    assert_groupby_results_equal(expected, got)


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row_mixed_numerics(shift_perc, direction):
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).diff(periods=n_shift)
    got = gdf.groupby(["0"]).diff(periods=n_shift)

    assert_groupby_results_equal(expected, got)


def test_groupby_diff_row_zero_shift():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    gdf = cudf.from_pandas(t.to_pandas())

    expected = gdf
    got = gdf.groupby(["0"]).shift(periods=0)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    # fill the dataframe with the first non-null item in the column
    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    # cudf can't fillna with a pandas.Timedelta type
    fill_values["4"] = fill_values["4"].to_numpy()
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


# TODO: cudf.fillna does not support decimal column to column fill yet
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value_df():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    # fill the dataframe with the first non-null item in the column
    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    # cudf can't fillna with a pandas.Timedelta type
    fill_values["4"] = fill_values["4"].to_numpy()
    fill_values = pd.DataFrame(fill_values, index=pdf.index)
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)

    fill_values = cudf.from_pandas(fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


def test_groupby_select_then_diff():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].diff(1)
    actual = gdf.groupby("a")["c"].diff(1)

    assert_groupby_results_equal(expected, actual)

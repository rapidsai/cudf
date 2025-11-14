# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_groupby_results_equal
from cudf.testing._utils import assert_exceptions_equal
from cudf.testing.dataset_generator import rand_dataframe


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


@pytest.mark.parametrize(
    "by",
    [pd.Series([1, 1, 2, 2, 3, 4]), lambda x: x % 2 == 0, pd.Grouper(level=0)],
)
@pytest.mark.parametrize(
    "data", [[1, None, 2, None, 3, None], [1, 2, 3, 4, 5, 6]]
)
@pytest.mark.parametrize("args", [{"value": 42}, {"method": "ffill"}])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_various_by_fillna(by, data, args):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    with pytest.warns(FutureWarning):
        expect = ps.groupby(by).fillna(**args)
    if isinstance(by, pd.Grouper):
        by = cudf.Grouper(level=by.level)
    with pytest.warns(FutureWarning):
        got = gs.groupby(by).fillna(**args)

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_groupby_fillna_method(method):
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
            {
                "dtype": "list",
                "null_frequency": 0.4,
                "cardinality": 10,
                "lists_max_length": 10,
                "nesting_max_depth": 3,
                "value_type": "int64",
            },
            {"dtype": "category", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(method=method)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(method=method)

    assert_groupby_results_equal(
        expect[value_cols], got[value_cols], sort=False
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Does not warn on older versions of pandas",
)
def test_cat_groupby_fillna():
    ps = pd.Series(["a", "b", "c"], dtype="category")
    gs = cudf.from_pandas(ps)

    with pytest.warns(FutureWarning):
        pg = ps.groupby(ps)
    gg = gs.groupby(gs)

    assert_exceptions_equal(
        lfunc=pg.fillna,
        rfunc=gg.fillna,
        lfunc_args_and_kwargs=(("d",), {}),
        rfunc_args_and_kwargs=(("d",), {}),
    )

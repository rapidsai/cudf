# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
    expect_warning_if,
)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning introduced in pandas-2.2.0",
)
@pytest.mark.parametrize(
    "data, dtype",
    [
        (
            {
                "a": [0, 1, None, 2, 3],
                "b": [3, 2, 2, 3, None],
                "c": ["abc", "def", ".", None, None],
            },
            None,
        ),
        (
            {
                "a": ["one", "two", None, "three"],
                "b": ["one", None, "two", "three"],
            },
            "category",
        ),
        (
            {
                "col one": [None, 10, 11, None, 1000, 500, 600],
                "col two": ["abc", "def", "ghi", None, "pp", None, "a"],
                "a": [0.324, 0.234, 324.342, 23.32, 9.9, None, None],
            },
            None,
        ),
    ],
)
@pytest.mark.parametrize(
    "to_replace,value",
    [
        (0, 4),
        ([0, 1], [4, 5]),
        ([0, 1], 4),
        ({"a": 0, "b": 0}, {"a": 4, "b": 5}),
        ({"a": 0}, {"a": 4}),
        ("abc", "---"),
        ([".", "gh"], "hi"),
        ([".", "def"], ["_", None]),
        ({"c": 0}, {"a": 4, "b": 5}),
        ({"a": 2}, {"c": "a"}),
        ("two", "three"),
        ([1, 2], pd.Series([10, 11])),
        (pd.Series([10, 11], index=[3, 2]), None),
        (
            pd.Series(["a+", "+c", "p", "---"], index=["abc", "gh", "l", "z"]),
            None,
        ),
        (
            pd.Series([10, 11], index=[3, 2]),
            {"a": [-10, -30], "l": [-111, -222]},
        ),
        (pd.Series([10, 11], index=[3, 2]), 555),
        (
            pd.Series([10, 11], index=["a", "b"]),
            pd.Series([555, 1111], index=["a", "b"]),
        ),
        ({"a": "2", "b": "3", "zzz": "hi"}, None),
        ({"a": 2, "b": 3, "zzz": "hi"}, 324353),
        (
            {"a": 2, "b": 3, "zzz": "hi"},
            pd.Series([5, 6, 10], index=["a", "b", "col one"]),
        ),
    ],
)
def test_dataframe_replace(data, dtype, to_replace, value):
    gdf = cudf.DataFrame(data, dtype=dtype)
    pdf = gdf.to_pandas()

    pd_value = value
    if isinstance(value, pd.Series):
        gd_value = cudf.from_pandas(value)
    else:
        gd_value = value

    pd_to_replace = to_replace
    if isinstance(to_replace, pd.Series):
        gd_to_replace = cudf.from_pandas(to_replace)
    else:
        gd_to_replace = to_replace

    can_warn = (
        isinstance(gdf["a"].dtype, cudf.CategoricalDtype)
        and isinstance(to_replace, str)
        and to_replace == "two"
        and isinstance(value, str)
        and value == "three"
    )
    with expect_warning_if(can_warn):
        if pd_value is None:
            expected = pdf.replace(to_replace=pd_to_replace)
        else:
            expected = pdf.replace(to_replace=pd_to_replace, value=pd_value)
    with expect_warning_if(can_warn):
        actual = gdf.replace(to_replace=gd_to_replace, value=gd_value)

    expected_sorted = expected.sort_values(by=list(expected.columns), axis=0)
    actual_sorted = actual.sort_values(by=list(actual.columns), axis=0)

    assert_eq(expected_sorted, actual_sorted)


def test_dataframe_replace_with_nulls():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = cudf.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, None).fillna(4)
    assert_eq(gdf2, pdf2)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, None]).fillna(5)
    assert_eq(gdf6, pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], None).fillna(4)
    assert_eq(gdf7, pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": None, "b": 5}).fillna(4)
    assert_eq(gdf8, pdf8)

    gdf1 = cudf.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, None]})
    gdf9 = gdf1.replace([0, 1], [4, 5]).fillna(3)
    assert_eq(gdf9, pdf6)


def test_replace_df_error():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.replace,
        rfunc=gdf.replace,
        lfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
        rfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
    )


def test_replace_multiple_rows(datadir):
    path = datadir / "parquet" / "replace_multiple_rows.parquet"
    pdf = pd.read_parquet(path)
    gdf = cudf.read_parquet(path)

    pdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    gdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    assert_eq(pdf, gdf, check_dtype=False)

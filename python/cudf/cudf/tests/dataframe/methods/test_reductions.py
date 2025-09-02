# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize("null_flag", [False, True])
def test_kurtosis_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.kurtosis(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurtosis(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)

    got = data.kurt(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurt(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)


@pytest.mark.parametrize("null_flag", [False, True])
def test_skew_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()
    np.testing.assert_array_almost_equal(got, expected)


def test_single_q():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_index():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [7, 4, 4, 9, 13]}, index=[0, 4, 3, 2, 7])
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_multiindex():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame(
        {
            "index_1": [3, 1, 9, 7, 5],
            "index_2": [2, 4, 3, 5, 1],
            "a": [8, 4, 2, 3, 8],
        }
    )
    pdf.set_index(["index_1", "index_2"], inplace=True)

    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [10, 11, 12]},
        {"a": [1, 0, 3], "b": [10, 11, 12]},
        {"a": [1, 2, 3], "b": [10, 11, None]},
        {
            "a": [],
        },
        {},
    ],
)
@pytest.mark.parametrize("op", ["all", "any"])
def test_any_all_axis_none(data, op):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = getattr(pdf, op)(axis=None)
    actual = getattr(gdf, op)(axis=None)

    assert expected == actual


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning not given on older versions of pandas",
)
def test_reductions_axis_none_warning(request, reduction_methods):
    if reduction_methods == "quantile":
        pytest.skip(f"pandas {reduction_methods} doesn't support axis=None")
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [10, 2, 3]})
    pdf = df.to_pandas()
    with expect_warning_if(
        reduction_methods in {"sum", "product", "std", "var"},
        FutureWarning,
    ):
        actual = getattr(df, reduction_methods)(axis=None)
    with expect_warning_if(
        reduction_methods in {"sum", "product", "std", "var"},
        FutureWarning,
    ):
        expected = getattr(pdf, reduction_methods)(axis=None)
    assert_eq(expected, actual, check_dtype=False)


def test_dataframe_reduction_no_args(reduction_methods):
    df = cudf.DataFrame({"a": range(10), "b": range(10)})
    pdf = df.to_pandas()
    result = getattr(df, reduction_methods)()
    expected = getattr(pdf, reduction_methods)()
    assert_eq(result, expected)


def test_reduction_column_multiindex():
    idx = cudf.MultiIndex.from_tuples(
        [("a", 1), ("a", 2)], names=["foo", "bar"]
    )
    df = cudf.DataFrame(np.array([[1, 3], [2, 4]]), columns=idx)
    result = df.mean()
    expected = df.to_pandas().mean()
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "columns", [pd.RangeIndex(2), pd.Index([0, 1], dtype="int8")]
)
def test_dataframe_axis_0_preserve_column_type_in_index(columns):
    pd_df = pd.DataFrame([[1, 2]], columns=columns)
    cudf_df = cudf.DataFrame.from_pandas(pd_df)
    result = cudf_df.sum(axis=0)
    expected = pd_df.sum(axis=0)
    assert_eq(result, expected, check_index_type=True)

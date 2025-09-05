# Copyright (c) 2025, NVIDIA CORPORATION.
import operator
from contextlib import contextmanager

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize("arg", [[True, False, True], [True, True, True]])
@pytest.mark.parametrize("value", [0, -1])
def test_dataframe_setitem_bool_mask_scalar(arg, value):
    df = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.from_pandas(df)

    df[arg] = value
    gdf[arg] = value
    assert_eq(df, gdf)


def test_dataframe_setitem_scalar_bool():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df[[True, False, True]] = pd.DataFrame({"a": [-1, -2]})

    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    gdf[[True, False, True]] = cudf.DataFrame({"a": [-1, -2]})
    assert_eq(df, gdf)


@pytest.mark.parametrize(
    "df",
    [pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": ["x", "y", "z"]})],
)
@pytest.mark.parametrize("arg", [["a"], "a", "b"])
@pytest.mark.parametrize(
    "value", [-10, pd.DataFrame({"a": [-1, -2, -3]}), "abc"]
)
def test_dataframe_setitem_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "value",
    [
        pd.DataFrame({"0": [-1, -2, -3], "1": [-0, -10, -1]}),
        10,
        "rapids",
        0.32234,
        np.datetime64(1324232423423342, "ns"),
        np.timedelta64(34234324234324234, "ns"),
    ],
)
def test_dataframe_setitem_new_columns(value):
    df = pd.DataFrame({"a": [1, 2, 3]})
    arg = ["b", "c"]
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=True)


def test_series_setitem_index():
    df = pd.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )

    df["b"] = pd.Series(data=[12, 11, 10], index=[3, 2, 1])
    gdf = cudf.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )
    gdf["b"] = cudf.Series(data=[12, 11, 10], index=[3, 2, 1])
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.xfail(reason="Copy-on-Write should make a copy")
@pytest.mark.parametrize(
    "index",
    [
        pd.MultiIndex.from_frame(
            pd.DataFrame({"b": [3, 2, 1], "c": ["a", "b", "c"]})
        ),
        ["a", "b", "c"],
    ],
)
def test_setitem_dataframe_series_inplace(index):
    gdf = cudf.DataFrame({"a": [1, 2, 3]}, index=index)
    expected = gdf.copy()
    with cudf.option_context("copy_on_write", True):
        gdf["a"].replace(1, 500, inplace=True)

    assert_eq(expected, gdf)


def test_listcol_setitem_retain_dtype():
    df = cudf.DataFrame(
        {"a": cudf.Series([["a", "b"], []]), "b": [1, 2], "c": [123, 321]}
    )
    df1 = df.head(0)
    # Performing a setitem on `b` triggers a `column.column_empty` call
    # which tries to create an empty ListColumn.
    df1["b"] = df1["c"]
    # Performing a copy to trigger a copy dtype which is obtained by accessing
    # `ListColumn.children` that would have been corrupted in previous call
    # prior to this fix: https://github.com/rapidsai/cudf/pull/10151/
    df2 = df1.copy()
    assert df2["a"].dtype == df["a"].dtype


def test_setitem_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result["a"] = [2]
    expected["a"] = [2]
    assert_eq(result, expected)


def test_dataframe_assign_scalar_to_empty_series():
    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame({"a": []})
    expected.a = 0
    actual.a = 0
    assert_eq(expected, actual)


def test_dataframe_assign_cp_np_array():
    m, n = 5, 3
    cp_ndarray = cp.random.randn(m, n)
    pdf = pd.DataFrame({f"f_{i}": range(m) for i in range(n)})
    gdf = cudf.DataFrame({f"f_{i}": range(m) for i in range(n)})
    pdf[[f"f_{i}" for i in range(n)]] = cp.asnumpy(cp_ndarray)
    gdf[[f"f_{i}" for i in range(n)]] = cp_ndarray

    assert_eq(pdf, gdf)


def test_dataframe_setitem_cupy_array():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.standard_normal(size=(10, 2)))
    gdf = cudf.from_pandas(pdf)

    gpu_array = cp.array([True, False] * 5)
    pdf[gpu_array.get()] = 1.5
    gdf[gpu_array] = 1.5

    assert_eq(pdf, gdf)


def test_setitem_datetime():
    df = cudf.DataFrame({"date": pd.date_range("20010101", "20010105").values})
    assert df.date.dtype.kind == "M"


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)


def test_dataframe_cow_slice_setitem():
    with cudf.option_context("copy_on_write", True):
        df = cudf.DataFrame(
            {"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}
        )
        slice_df = df[1:4]

        assert_eq(
            slice_df,
            cudf.DataFrame(
                {"a": [11, 12, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
            ),
        )

        slice_df["a"][2] = 1111

        assert_eq(
            slice_df,
            cudf.DataFrame(
                {"a": [11, 1111, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
            ),
        )
        assert_eq(
            df,
            cudf.DataFrame(
                {"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}
            ),
        )


def test_multiindex_row_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(0, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "index", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "index", "c": gdfIndex}),
    )


def test_multiindex_column_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 0)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "columns", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "columns", "c": gdfIndex}),
    )


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


@pytest.mark.parametrize(
    "query",
    [
        ("a", "store", "clouds", "fire"),
        ("a", "store", "storm", "smoke"),
        ("a", "store"),
        ("b", "house"),
        ("a", "store", "storm"),
        ("a",),
        ("c", "forest", "clear"),
    ],
)
def test_multiindex_columns(query):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdf = pdf.T
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
    assert_eq(pdfIndex, gdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(query):
        expected = pdf[query]
    got = gdf[query]
    assert_eq(expected, got)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/43351",
)
def test_multicolumn_set_item():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
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
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    pdf["d"] = [1, 2, 3, 4, 5]
    gdf["d"] = [1, 2, 3, 4, 5]
    assert_eq(pdf, gdf)

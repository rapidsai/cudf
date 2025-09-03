# Copyright (c) 2025, NVIDIA CORPORATION.

import decimal
import functools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("digits", [0, 1, 4])
def test_dataframe_round_builtin(digits):
    pdf = pd.DataFrame(
        {
            "a": [1.2234242333234, 323432.3243423, np.nan],
            "b": ["a", "b", "c"],
            "c": pd.Series([34224, 324324, 324342], dtype="datetime64[ns]"),
            "d": pd.Series([224.242, None, 2424.234324], dtype="category"),
            "e": [
                decimal.Decimal("342.3243234234242"),
                decimal.Decimal("89.32432497687622"),
                None,
            ],
        }
    )
    gdf = cudf.from_pandas(pdf, nan_as_null=False)

    expected = round(pdf, digits)
    actual = round(gdf, digits)

    assert_eq(expected, actual)


def test_bool_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[cudf.DataFrame()]],
        rfunc_args_and_kwargs=[[pd.DataFrame()]],
    )


@pytest.mark.parametrize("name", [None, "foo", 1, 1.0])
def test_dataframe_column_name(name):
    df = cudf.DataFrame({"a": [1, 2, 3]})
    pdf = df.to_pandas()

    df.columns.name = name
    pdf.columns.name = name

    assert_eq(df, pdf)
    assert_eq(df.columns.name, pdf.columns.name)


def test_dataframe_columns_set_none_raises():
    df = cudf.DataFrame({"a": [0]})
    with pytest.raises(TypeError):
        df.columns = None


@pytest.mark.parametrize(
    "columns",
    [cudf.RangeIndex(1, name="foo"), pd.RangeIndex(1, name="foo"), range(1)],
)
def test_dataframe_columns_set_rangeindex(columns):
    df = cudf.DataFrame([1], columns=["a"])
    df.columns = columns
    result = df.columns
    expected = pd.RangeIndex(1, name=getattr(columns, "name", None))
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_dataframe_columns_set_multiindex(klass):
    columns = klass.from_arrays([[10]], names=["foo"])
    df = cudf.DataFrame([1], columns=["a"])
    df.columns = columns
    result = df.columns
    expected = pd.MultiIndex.from_arrays([[10]], names=["foo"])
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "klass",
    [
        functools.partial(cudf.Index, name="foo"),
        functools.partial(cudf.Series, name="foo"),
        functools.partial(pd.Index, name="foo"),
        functools.partial(pd.Series, name="foo"),
        np.array,
    ],
)
def test_dataframe_columns_set_preserve_type(klass):
    df = cudf.DataFrame([1], columns=["a"])
    columns = klass([10], dtype="int8")
    df.columns = columns
    result = df.columns
    expected = pd.Index(
        [10], dtype="int8", name=getattr(columns, "name", None)
    )
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("column", [range(1, 2), np.array([1], dtype=np.int8)])
@pytest.mark.parametrize(
    "operation",
    [
        lambda df: df.where(df < 2, 2),
        lambda df: df.nans_to_nulls(),
        lambda df: df.isna(),
        lambda df: df.notna(),
        lambda df: abs(df),
        lambda df: -df,
        lambda df: ~df,
        lambda df: df.cumsum(),
        lambda df: df.replace(1, 2),
        lambda df: df.replace(10, 20),
        lambda df: df.clip(0, 10),
        lambda df: df.rolling(1).mean(),
        lambda df: df.interpolate(),
        lambda df: df.shift(),
        lambda df: df.sort_values(1),
        lambda df: df.round(),
        lambda df: df.rank(),
    ],
)
def test_op_preserves_column_metadata(column, operation):
    df = cudf.DataFrame([1], columns=cudf.Index(column))
    result = operation(df).columns
    expected = pd.Index(column)
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2]},
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [1, 3, 5, 7]},
        {"a": [np.nan, 2, 3, 4], "b": [3, 4, np.nan, 6], "c": [1, 3, 5, 7]},
        {1: [1, 2, 3], 2: [3, 4, 5]},
        {"a": [1, None, None], "b": [3, np.nan, np.nan]},
        {1: ["a", "b", "c"], 2: ["q", "w", "u"]},
        {1: ["a", np.nan, "c"], 2: ["q", None, "u"]},
        {},
        {1: [], 2: [], 3: []},
        [1, 2, 3],
    ],
)
def test_axes(data):
    csr = cudf.DataFrame(data)
    psr = pd.DataFrame(data)

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a, exact=False)


def test_iter():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)
    assert list(pdf) == list(gdf)


def test_column_assignment():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    new_cols = ["q", "r", "s"]
    gdf.columns = new_cols
    assert list(gdf.columns) == new_cols


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert pdf.ndim == gdf.ndim


@pytest.mark.parametrize(
    "index",
    [
        ["a", "b", "c", "d", "e"],
        np.array(["a", "b", "c", "d", "e"]),
        pd.Index(["a", "b", "c", "d", "e"], name="name"),
    ],
)
def test_string_index(index):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 5)))
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdf.index = index
    gdf.index = index
    assert_eq(pdf, gdf)

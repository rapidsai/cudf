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


@pytest.mark.parametrize("names", [["abc", "def"], [1, 2], ["abc", 10]])
def test_dataframe_multiindex_column_names(names):
    arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
    tuples = list(zip(*arrays, strict=True))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

    pdf = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
    df = cudf.from_pandas(pdf)

    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)
    pdf.columns.names = names
    df.columns.names = names
    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)


@pytest.mark.parametrize("name", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("contains", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("other_names", [[], ["b", "c"], [1, 2]])
def test_dataframe_contains(name, contains, other_names):
    column_names = [name, *other_names]
    gdf = cudf.DataFrame({c: [0] for c in column_names})
    pdf = pd.DataFrame({c: [0] for c in column_names})

    assert_eq(gdf, pdf)

    if contains is cudf.NA or name is cudf.NA:
        expectation = contains is cudf.NA and name is cudf.NA
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation
    elif gdf.columns.dtype.kind == "f":
        # In some cases, the columns are converted to an Index[float] based on
        # the other column names. That casts name values from None to np.nan.
        expectation = contains is np.nan and (name is None or name is np.nan)
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation
    else:
        expectation = contains == name or (
            contains is np.nan and name is np.nan
        )
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation

    assert (contains in pdf) == (contains in gdf)


@pytest.mark.parametrize(
    "data",
    [
        {"col": [{"a": 1.1}, {"a": 2.1}, {"a": 10.0}, {"a": 11.2323}, None]},
        {"a": [[{"b": 567}], None] * 10},
        {"a": [decimal.Decimal(10), decimal.Decimal(20), None]},
    ],
)
def test_dataframe_values_complex_types(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(NotImplementedError):
        gdf.values


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_size(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.size, gdf.size)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_empty(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.empty, gdf.empty)


def test_cudf_arrow_array_error():
    df = cudf.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        df.__arrow_array__()

    sr = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()

    sr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()


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


def test_set_index_as_property():
    cdf = cudf.DataFrame()
    col1 = np.arange(10)
    col2 = np.arange(0, 20, 2)
    cdf["a"] = col1
    cdf["b"] = col2

    # Check set_index(Series)
    cdf.index = cdf["b"]

    assert_eq(cdf.index.to_numpy(), col2)

    with pytest.raises(ValueError):
        cdf.index = [list(range(10))]

    idx = pd.Index(np.arange(0, 1000, 100))
    cdf.index = idx
    assert_eq(cdf.index.to_pandas(), idx)

    df = cdf.to_pandas()
    assert_eq(df.index, idx)

    head = cdf.head().to_pandas()
    assert_eq(head.index, idx[:5])


@pytest.mark.parametrize(
    "index",
    [
        lambda: cudf.Index([1]),
        lambda: cudf.RangeIndex(1),
        lambda: cudf.MultiIndex(levels=[[0]], codes=[[0]]),
    ],
)
def test_index_assignment_no_shallow_copy(index):
    index = index()
    df = cudf.DataFrame(range(1))
    df.index = index
    assert df.index is index

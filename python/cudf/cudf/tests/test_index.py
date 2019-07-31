# Copyright (c) 2018, NVIDIA CORPORATION.

"""
Test related to Index
"""
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.dataframe import DataFrame
from cudf.dataframe.index import (
    CategoricalIndex,
    DatetimeIndex,
    GenericIndex,
    RangeIndex,
    as_index,
)
from cudf.tests.utils import assert_eq


def test_df_set_index_from_series():
    df = DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(Series)
    df2 = df.set_index(df["b"])
    assert list(df2.columns) == ["a", "b"]
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_set_index_from_name():
    df = DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(column_name)
    df2 = df.set_index("b")
    print(df2)
    # 1 less column because 'b' is used as index
    assert list(df2.columns) == ["a"]
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_slice_empty_index():
    df = DataFrame()
    assert isinstance(df.index, RangeIndex)
    assert isinstance(df.index[:1], RangeIndex)
    with pytest.raises(IndexError):
        df.index[1]


def test_index_find_label_range():
    # Monotonic Index
    idx = GenericIndex(np.asarray([4, 5, 6, 10]))
    assert idx.find_label_range(4, 6) == (0, 3)
    assert idx.find_label_range(5, 10) == (1, 4)
    assert idx.find_label_range(0, 6) == (0, 3)
    assert idx.find_label_range(4, 11) == (0, 4)

    # Non-monotonic Index
    idx_nm = GenericIndex(np.asarray([5, 4, 6, 10]))
    assert idx_nm.find_label_range(4, 6) == (1, 3)
    assert idx_nm.find_label_range(5, 10) == (0, 4)
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx_nm.find_label_range(0, 6)
    raises.match("value not found")
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx_nm.find_label_range(4, 11)
    raises.match("value not found")


def test_index_comparision():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    gi = GenericIndex(np.arange(start, stop))
    assert rg.equals(gi)
    assert gi.equals(rg)
    assert not rg[:-1].equals(gi)
    assert rg[:-1].equals(gi[:-1])


@pytest.mark.parametrize(
    "func", [lambda x: x.min(), lambda x: x.max(), lambda x: x.sum()]
)
def test_reductions(func):
    x = np.asarray([4, 5, 6, 10])
    idx = GenericIndex(np.asarray([4, 5, 6, 10]))

    assert func(x) == func(idx)


def test_name():
    idx = GenericIndex(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_immutable():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    with pytest.raises(TypeError):
        rg[1] = 5
    gi = GenericIndex(np.arange(start, stop))
    with pytest.raises(TypeError):
        gi[1] = 5


def test_categorical_index():
    pdf = pd.DataFrame()
    pdf["a"] = [1, 2, 3]
    pdf["index"] = pd.Categorical(["a", "b", "c"])
    initial_df = DataFrame.from_pandas(pdf)
    pdf = pdf.set_index("index")
    gdf1 = DataFrame.from_pandas(pdf)
    gdf2 = DataFrame()
    gdf2["a"] = [1, 2, 3]
    gdf2["index"] = pd.Categorical(["a", "b", "c"])
    assert_eq(initial_df.index, gdf2.index)
    gdf2 = gdf2.set_index("index")

    assert isinstance(gdf1.index, CategoricalIndex)
    assert_eq(pdf, gdf1)
    assert_eq(pdf.index, gdf1.index)
    assert_eq(pdf.index.codes, gdf1.index.codes.to_array())

    assert isinstance(gdf2.index, CategoricalIndex)
    assert_eq(pdf, gdf2)
    assert_eq(pdf.index, gdf2.index)
    assert_eq(pdf.index.codes, gdf2.index.codes.to_array())


def test_pandas_as_index():
    # Define Pandas Indexes
    pdf_int_index = pd.Int64Index([1, 2, 3, 4, 5])
    pdf_float_index = pd.Float64Index([1.0, 2.0, 3.0, 4.0, 5.0])
    pdf_datetime_index = pd.DatetimeIndex(
        [1000000, 2000000, 3000000, 4000000, 5000000]
    )
    pdf_category_index = pd.CategoricalIndex(["a", "b", "c", "b", "a"])

    # Define cudf Indexes
    gdf_int_index = as_index(pdf_int_index)
    gdf_float_index = as_index(pdf_float_index)
    gdf_datetime_index = as_index(pdf_datetime_index)
    gdf_category_index = as_index(pdf_category_index)

    # Check instance types
    assert isinstance(gdf_int_index, GenericIndex)
    assert isinstance(gdf_float_index, GenericIndex)
    assert isinstance(gdf_datetime_index, DatetimeIndex)
    assert isinstance(gdf_category_index, CategoricalIndex)

    # Check equality
    assert_eq(pdf_int_index, gdf_int_index)
    assert_eq(pdf_float_index, gdf_float_index)
    assert_eq(pdf_datetime_index, gdf_datetime_index)
    assert_eq(pdf_category_index, gdf_category_index)

    assert_eq(pdf_category_index.codes, gdf_category_index.codes.to_array())


def test_index_rename():
    pds = pd.Index([1, 2, 3], name="asdf")
    gds = as_index(pds)

    expect = pds.rename("new_name")
    got = gds.rename("new_name")

    assert_eq(expect, got)
    """
    From here on testing recursive creation
    and if name is being handles in recursive creation.
    """
    pds = pd.Index(expect)
    gds = as_index(got)

    assert_eq(pds, gds)

    pds = pd.Index(pds, name="abc")
    gds = as_index(gds, name="abc")
    assert_eq(pds, gds)


def test_set_index_as_property():
    cdf = DataFrame()
    col1 = np.arange(10)
    col2 = np.arange(0, 20, 2)
    cdf["a"] = col1
    cdf["b"] = col2

    # Check set_index(Series)
    cdf.index = cdf["b"]

    np.testing.assert_array_equal(cdf.index.values, col2)

    with pytest.raises(ValueError):
        cdf.index = [list(range(10))]

    idx = np.arange(0, 1000, 100)
    cdf.index = idx
    np.testing.assert_array_equal(cdf.index.values, idx)

    df = cdf.to_pandas()
    np.testing.assert_array_equal(df.index.values, idx)

    head = cdf.head().to_pandas()
    np.testing.assert_array_equal(head.index.values, idx[:5])


@pytest.mark.parametrize(
    "idx",
    [
        cudf.dataframe.index.RangeIndex(1, 5),
        cudf.dataframe.index.DatetimeIndex(["2001", "2003", "2003"]),
        cudf.dataframe.index.StringIndex(["a", "b", "c"]),
        cudf.dataframe.index.GenericIndex([1, 2, 3]),
        cudf.dataframe.index.CategoricalIndex(["a", "b", "c"]),
    ],
)
@pytest.mark.parametrize("deep", [True, False])
def test_index_copy(idx, deep):
    idx_copy = idx.copy(deep=deep)
    assert_eq(idx, idx_copy)
    assert type(idx) == type(idx_copy)

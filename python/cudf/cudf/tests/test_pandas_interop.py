# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame
from cudf.tests.utils import assert_eq


def test_to_pandas():
    df = DataFrame()
    df["a"] = np.arange(5, dtype=np.int32)
    df["b"] = np.arange(10, 15, dtype=np.float64)
    df["c"] = np.array([True, False, None, True, True])

    pdf = df.to_pandas(nullable_pd_dtype=False)

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df["a"].dtype == pdf["a"].dtype
    assert df["b"].dtype == pdf["b"].dtype

    # Notice, the dtype differ when Pandas and cudf boolean series
    # contains None/NaN
    assert df["c"].dtype == np.bool
    assert pdf["c"].dtype == np.object

    assert len(df["a"]) == len(pdf["a"])
    assert len(df["b"]) == len(pdf["b"])
    assert len(df["c"]) == len(pdf["c"])


def test_from_pandas():
    pdf = pd.DataFrame()
    pdf["a"] = np.arange(10, dtype=np.int32)
    pdf["b"] = np.arange(10, 20, dtype=np.float64)

    df = DataFrame.from_pandas(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df["a"].dtype == pdf["a"].dtype
    assert df["b"].dtype == pdf["b"].dtype

    assert len(df["a"]) == len(pdf["a"])
    assert len(df["b"]) == len(pdf["b"])


def test_from_pandas_ex1():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    df = DataFrame.from_pandas(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)
    assert np.all(df["a"].to_array() == pdf["a"])
    matches = df["b"].to_array(fillna="pandas") == pdf["b"]
    # the 3d element is False due to (nan == nan) == False
    assert np.all(matches == [True, True, False, True])
    assert np.isnan(df["b"].to_array(fillna="pandas")[2])
    assert np.isnan(pdf["b"][2])


def test_from_pandas_with_index():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    pdf = pdf.set_index(np.asarray([4, 3, 2, 1]))
    df = DataFrame.from_pandas(pdf)

    # Check columns
    assert_eq(df.a, pdf.a)
    assert_eq(df.b, pdf.b)
    # Check index
    assert_eq(df.index.values, pdf.index.values)
    # Check again using pandas testing tool on frames
    assert_eq(df, pdf)


def test_from_pandas_rangeindex():
    idx1 = pd.RangeIndex(start=0, stop=4, step=1, name="myindex")
    idx2 = cudf.from_pandas(idx1)

    # Check index
    assert_eq(idx1.values, idx2.values)
    assert idx1.name == idx2.name


def test_from_pandas_rangeindex_step():
    idx1 = pd.RangeIndex(start=0, stop=8, step=2, name="myindex")
    with pytest.raises(ValueError):
        cudf.from_pandas(idx1)

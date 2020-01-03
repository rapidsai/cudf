# Copyright (c) 2018, NVIDIA CORPORATION.

from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pytest
from numba import cuda

from cudf.core.dataframe import DataFrame
from cudf.tests.utils import assert_eq


"""
DataFrame copy expectations
* A shallow copy constructs a new compound object and then (to the extent
  possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts
  copies into it of the objects found in the original.

  A cuDF DataFrame is a compound object containing a few members:
  _index, _size, _cols, where _cols is an OrderedDict
"""


@pytest.mark.parametrize(
    "copy_parameters",
    [
        {"fn": lambda x: x.copy(), "expected_equality": False},
        {"fn": lambda x: x.copy(deep=True), "expected_equality": False},
        {"fn": lambda x: copy(x), "expected_equality": False},
        {"fn": lambda x: deepcopy(x), "expected_equality": False},
    ],
)
def test_dataframe_deep_copy(copy_parameters):
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    copy_pdf = copy_parameters["fn"](pdf)
    copy_gdf = copy_parameters["fn"](gdf)
    copy_pdf["b"] = [0, 0, 0]
    copy_gdf["b"] = [0, 0, 0]
    pdf_is_equal = np.array_equal(pdf["b"].values, copy_pdf["b"].values)
    gdf_is_equal = np.array_equal(
        gdf["b"].to_array(), copy_gdf["b"].to_array()
    )
    assert pdf_is_equal == copy_parameters["expected_equality"]
    assert gdf_is_equal == copy_parameters["expected_equality"]


@pytest.mark.parametrize(
    "copy_parameters",
    [
        {"fn": lambda x: x.copy(), "expected_equality": False},
        {"fn": lambda x: x.copy(deep=True), "expected_equality": False},
        {"fn": lambda x: copy(x), "expected_equality": False},
        {"fn": lambda x: deepcopy(x), "expected_equality": False},
    ],
)
def test_dataframe_deep_copy_and_insert(copy_parameters):
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    copy_pdf = copy_parameters["fn"](pdf)
    copy_gdf = copy_parameters["fn"](gdf)
    copy_pdf["b"] = [0, 0, 0]
    copy_gdf["b"] = [0, 0, 0]
    pdf_is_equal = np.array_equal(pdf["b"].values, copy_pdf["b"].values)
    gdf_is_equal = np.array_equal(
        gdf["b"].to_array(), copy_gdf["b"].to_array()
    )
    assert pdf_is_equal == copy_parameters["expected_equality"]
    assert gdf_is_equal == copy_parameters["expected_equality"]


"""
DataFrame copy bounds checking - sizes 0 through 10 perform as
expected_equality
"""


@pytest.mark.parametrize(
    "copy_fn",
    [
        lambda x: x.copy(),
        lambda x: x.copy(deep=True),
        lambda x: copy(x),
        lambda x: deepcopy(x),
        lambda x: x.copy(deep=False),
    ],
)
@pytest.mark.parametrize("ncols", [0, 1, 10])
@pytest.mark.parametrize(
    "data_type",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "category",
        "datetime64[ms]",
    ],
)
def test_cudf_dataframe_copy(copy_fn, ncols, data_type):
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i + ord("a"))] = pd.Series(
            np.random.randint(0, 1000, 20)
        ).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    copy_df = copy_fn(df)
    assert_eq(df, copy_df)


@pytest.mark.parametrize(
    "copy_fn",
    [
        lambda x: x.copy(),
        lambda x: x.copy(deep=True),
        lambda x: copy(x),
        lambda x: deepcopy(x),
        lambda x: x.copy(deep=False),
    ],
)
@pytest.mark.parametrize("ncols", [0, 1, 10])
@pytest.mark.parametrize(
    "data_type",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "category",
        "datetime64[ms]",
    ],
)
def test_cudf_dataframe_copy_then_insert(copy_fn, ncols, data_type):
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i + ord("a"))] = pd.Series(
            np.random.randint(0, 1000, 20)
        ).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    copy_df = copy_fn(df)
    copy_pdf = copy_fn(pdf)
    copy_df["aa"] = pd.Series(np.random.randint(0, 1000, 20)).astype(data_type)
    copy_pdf["aa"] = pd.Series(np.random.randint(0, 1000, 20)).astype(
        data_type
    )
    assert not copy_pdf.to_string().split() == pdf.to_string().split()
    assert not copy_df.to_string().split() == df.to_string().split()


@cuda.jit
def group_mean(data, segments, output):
    i = cuda.grid(1)
    if i < segments.size:
        s = segments[i]
        e = segments[i + 1] if (i + 1) < segments.size else data.size
        # mean calculation
        carry = 0.0
        n = e - s
        for j in range(s, e):
            carry += data[j]
        output[i] = carry / n


@cuda.jit
def add_one(data):
    i = cuda.grid(1)
    if i == 1:
        data[i] = data[i] + 1.0


def test_kernel_deep_copy():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    cdf = gdf.copy(deep=True)
    sr = gdf["b"]

    # column.to_gpu_array calls to_dense_buffer which returns a copy
    # need to access buffer directly and then call gpu_array
    add_one[1, len(sr)](sr._column.data_array_view)
    assert not gdf.to_string().split() == cdf.to_string().split()


def test_kernel_shallow_copy():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    cdf = gdf.copy(deep=False)
    sr = gdf["a"]
    add_one[1, len(sr)](sr.to_gpu_array())
    assert_eq(gdf, cdf)


@pytest.mark.xfail(reason="cudf column-wise shallow copy is immutable")
def test_dataframe_copy_shallow():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    copy_pdf = pdf.copy(deep=False)
    copy_gdf = gdf.copy(deep=False)
    copy_pdf["b"] = [0, 0, 0]
    copy_gdf["b"] = [0, 0, 0]
    assert_eq(pdf["b"], copy_pdf["b"])
    assert_eq(gdf["b"], copy_gdf["b"])

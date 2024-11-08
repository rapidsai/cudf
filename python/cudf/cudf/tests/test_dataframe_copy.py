# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from copy import copy, deepcopy

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from cudf.core.dataframe import DataFrame
from cudf.testing import assert_eq, assert_neq
from cudf.testing._utils import ALL_TYPES

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
        gdf["b"].to_numpy(), copy_gdf["b"].to_numpy()
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
        gdf["b"].to_numpy(), copy_gdf["b"].to_numpy()
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
@pytest.mark.parametrize("data_type", ALL_TYPES)
def test_cudf_dataframe_copy(copy_fn, ncols, data_type):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            chr(i + ord("a")): pd.Series(rng.integers(0, 1000, 20)).astype(
                data_type
            )
            for i in range(ncols)
        }
    )
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
@pytest.mark.parametrize("data_type", ALL_TYPES)
def test_cudf_dataframe_copy_then_insert(copy_fn, ncols, data_type):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            chr(i + ord("a")): pd.Series(rng.integers(0, 1000, 20)).astype(
                data_type
            )
            for i in range(ncols)
        }
    )
    df = DataFrame.from_pandas(pdf)
    copy_df = copy_fn(df)
    copy_pdf = copy_fn(pdf)
    copy_df["aa"] = pd.Series(rng.integers(0, 1000, 20)).astype(data_type)
    copy_pdf["aa"] = pd.Series(rng.integers(0, 1000, 20)).astype(data_type)
    assert not copy_pdf.to_string().split() == pdf.to_string().split()
    assert not copy_df.to_string().split() == df.to_string().split()


def test_deep_copy_write_in_place():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    cdf = gdf.copy(deep=True)
    sr = gdf["b"]

    # Write a value in-place on the deep copy.
    # This should only affect the copy and not the original.
    cp.asarray(sr._column)[1] = 42

    assert_neq(gdf, cdf)


def test_shallow_copy_write_in_place():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = DataFrame.from_pandas(pdf)
    cdf = gdf.copy(deep=False)
    sr = gdf["a"]

    # Write a value in-place on the shallow copy.
    # This should change the copy and original.
    cp.asarray(sr._column)[1] = 42

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

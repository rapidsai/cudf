# Copyright (c) 2020, NVIDIA CORPORATION.

from itertools import product
from math import floor

import numpy as np
import cudf
import pandas as pd
import pytest

from cudf import Series
from cudf.tests.utils import assert_eq, assert_exceptions_equal


def test_series_map_basic():
    gd1 = cudf.Series(["cat", np.nan, "rabbit", "dog"])
    pdf1 = gd1.to_pandas()

    expected_dict = pdf1.map({"cat": "kitten", "dog": "puppy"})
    actual_dict = gd1.map({"cat": "kitten", "dog": "puppy"})

    assert_eq(expected_dict, actual_dict)


@pytest.mark.parametrize("name", ["a", None, 2])
def test_series_map_series_input(name):
    gd1 = cudf.Series(["cat", "dog", np.nan, "rabbit"], name=name)
    pdf1 = gd1.to_pandas()

    expected_series = pdf1.map(pd.Series({"cat": "kitten", "dog": "puppy"}))
    actual_series = gd1.map(cudf.Series({"cat": "kitten", "dog": "puppy"}))

    assert_eq(expected_series, actual_series)


def test_series_map_callable_numeric_basic():
    gd2 = cudf.Series([1, 2, 3, 4, np.nan])
    pdf2 = gd2.to_pandas()

    expected_function = pdf2.map(lambda x: x ** 2)
    actual_function = gd2.map(lambda x: x ** 2)

    assert_eq(expected_function, actual_function)


@pytest.mark.parametrize("nelem", list(product([2, 10, 100, 1000])))
def test_series_map_callable_numeric_random(nelem):
    # Generate data
    np.random.seed(0)
    data = np.random.random(nelem) * 100

    sr = Series(data)
    pdsr = pd.Series(data)

    # Call applymap
    got = sr.map(lambda x: (floor(x) + 1 if x - floor(x) >= 0.5 else floor(x)))
    expect = pdsr.map(
        lambda x: (floor(x) + 1 if x - floor(x) >= 0.5 else floor(x))
    )

    # Check
    assert_eq(expect, got, check_dtype=False)


def test_series_map_callable_numeric_random_dtype_change():
    # Test for changing the out_dtype using applymap

    data = list(range(10))

    sr = Series(data)
    pdsr = pd.Series(data)

    got = sr.map(lambda x: float(x))
    expect = pdsr.map(lambda x: float(x))

    # Check
    assert_eq(expect, got)


def test_series_map_non_unique_index():
    # test for checking correct error is produced

    gd1 = cudf.Series([1, 2, 3, 4, np.nan])
    pd1 = pd.Series([1, 2, 3, 4, np.nan])

    gd_map_series = cudf.Series(["a", "b", "c"], index=[1, 1, 2])
    pd_map_series = pd.Series(["a", "b", "c"], index=[1, 1, 2])

    assert_exceptions_equal(
        lfunc=pd1.map,
        rfunc=gd1.map,
        check_exception_type=False,
        lfunc_args_and_kwargs=([pd_map_series],),
        rfunc_args_and_kwargs=([gd_map_series],),
        expected_error_message="Reindexing only valid with uniquely"
        " valued Index objects",
    )

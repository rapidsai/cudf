# Copyright (c) 2018, NVIDIA CORPORATION.

from itertools import product
from math import floor

import numpy as np
import cudf
import pandas as pd
import pytest

from cudf import Series
from cudf.tests import utils
from cudf.tests.utils import assert_eq


def test_series_map_basic():
    gd1 = cudf.Series(["cat", "dog", np.nan, "rabbit"])
    pdf1 = gd1.to_pandas()

    expected_dict = pdf1.map({"cat": "kitten", "dog": "puppy"})
    actual_dict = gd1.map({"cat": "kitten", "dog": "puppy"})

    assert_eq(expected_dict, actual_dict)


def test_series_map_series_input():
    gd1 = cudf.Series(["cat", "dog", np.nan, "rabbit"])
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


@pytest.mark.parametrize(
    "nelem", list(product([2, 10, 100, 1000]))
)
def test_series_map_callable_numeric_random(nelem):
    # Generate data
    np.random.seed(0)
    data = np.random.random(nelem) * 100

    sr = Series(data)

    # Call applymap
    out = sr.map(
        lambda x: (floor(x) + 1 if x - floor(x) >= 0.5 else floor(x))
    )

    # Check
    expect = np.round(data)
    got = out.to_array()
    np.testing.assert_array_almost_equal(expect, got)


def test_series_map_callable_numeric_random_dtype_change():
    # Test for changing the out_dtype using applymap

    data = list(range(10))

    sr = Series(data)

    out = sr.map(lambda x: float(x))

    # Check
    expect = np.array(data, dtype=float)
    got = out.to_array()
    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize("na_action", [None, 'ignore'])
def test_series_map_callable_string(na_action):
    gd3 = cudf.Series(["cat", "dog", np.nan, "rabbit"])
    pdf3 = gd3.to_pandas()

    expected_function = pdf3.map('I am a {}'.format, na_action=na_action)
    actual_function = gd3.map('I am a {}'.format, na_action=na_action)

    assert_eq(expected_function, actual_function)

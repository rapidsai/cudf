from __future__ import division

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import Series
from cudf.tests import utils


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_series_abs(dtype):
    arr = (np.random.random(1000) * 100).astype(dtype)
    sr = Series(arr)
    np.testing.assert_equal(sr.abs().to_array(), np.abs(arr))
    np.testing.assert_equal(abs(sr).to_array(), abs(arr))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_series_invert(dtype):
    arr = (np.random.random(1000) * 100).astype(dtype)
    sr = Series(arr)
    np.testing.assert_equal((~sr).to_array(), np.invert(arr))
    np.testing.assert_equal((~sr).to_array(), ~arr)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.bool_]
)
def test_series_not(dtype):
    import pandas as pd

    arr = pd.Series(np.random.choice([True, False], 1000)).astype(dtype)
    if dtype is not np.bool_:
        arr = arr * (np.random.random(1000) * 100).astype(dtype)
    sr = Series(arr)

    result = cudf.logical_not(sr).to_array()
    expect = np.logical_not(arr)
    np.testing.assert_equal(result, expect)
    np.testing.assert_equal((~sr).to_array(), ~arr)


def test_series_neg():
    arr = np.random.random(100) * 100
    sr = Series(arr)
    np.testing.assert_equal((-sr).to_array(), -arr)


def test_series_ceil():
    arr = np.random.random(100) * 100
    sr = Series(arr)
    np.testing.assert_equal(sr.ceil().to_array(), np.ceil(arr))


def test_series_floor():
    arr = np.random.random(100) * 100
    sr = Series(arr)
    np.testing.assert_equal(sr.floor().to_array(), np.floor(arr))


@pytest.mark.parametrize("nelem", [1, 7, 8, 9, 32, 64, 128])
def test_validity_ceil(nelem):
    # Data
    data = np.random.random(nelem) * 100
    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    sr = Series.from_masked_array(data, mask)

    # Result
    res = sr.ceil()

    na_value = -100000
    got = res.fillna(na_value).to_array()
    res_mask = np.asarray(bitmask, dtype=np.bool_)[: data.size]

    expect = np.ceil(data)
    expect[~res_mask] = na_value

    # Check
    print("expect")
    print(expect)
    print("got")
    print(got)

    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize("mth", ["min", "max", "sum", "product"])
def test_series_pandas_methods(mth):
    np.random.seed(0)
    arr = (1 + np.random.random(5) * 100).astype(np.int64)
    sr = Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(getattr(sr, mth)(), getattr(psr, mth)())


@pytest.mark.parametrize("mth", ["min", "max", "sum", "product", "quantile"])
def test_series_pandas_methods_empty(mth):
    arr = np.array([])
    sr = Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(getattr(sr, mth)(), getattr(psr, mth)())

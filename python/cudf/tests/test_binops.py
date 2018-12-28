# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import operator
import random
from itertools import product

import pytest
import numpy as np

from cudf.dataframe import Series

from . import utils


_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
]


@pytest.mark.parametrize('binop', _binops)
def test_series_binop(binop):
    arr = np.random.random(100)
    sr = Series(arr)
    np.testing.assert_equal(binop(sr, sr).to_array(), binop(arr, arr))


@pytest.mark.parametrize('nelem,binop', list(product([1, 2, 100], _binops)))
def test_series_binop_scalar(nelem, binop):
    arr = np.random.random(nelem)
    rhs = np.asscalar(random.choice(arr))
    sr = Series(arr)
    np.testing.assert_equal(binop(sr, rhs).to_array(), binop(arr, rhs))


_cmpops = [
    operator.lt,
    operator.gt,
    operator.le,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.mark.parametrize('cmpop', _cmpops)
def test_series_compare(cmpop):
    arr1 = np.random.random(100)
    arr2 = np.random.random(100)
    sr1 = Series(arr1)
    sr2 = Series(arr2)
    np.testing.assert_equal(cmpop(sr1, sr1).to_array(),  cmpop(arr1, arr1))
    np.testing.assert_equal(cmpop(sr2, sr2).to_array(),  cmpop(arr2, arr2))
    np.testing.assert_equal(cmpop(sr1, sr2).to_array(),  cmpop(arr1, arr2))


@pytest.mark.parametrize('nelem,cmpop', list(product([1, 2, 100], _cmpops)))
def test_series_compare_scalar(nelem, cmpop):
    arr1 = np.random.random(nelem)
    sr1 = Series(arr1)
    rhs = np.asscalar(random.choice(arr1))
    np.testing.assert_equal(cmpop(sr1, rhs).to_array(),  cmpop(arr1, rhs))
    np.testing.assert_equal(cmpop(rhs, sr1).to_array(),  cmpop(rhs, arr1))


@pytest.mark.parametrize('nelem', [1, 7, 8, 9, 32, 64, 128])
def test_validity_add(nelem):
    np.random.seed(0)
    # LHS
    lhs_data = np.random.random(nelem)
    lhs_mask = utils.random_bitmask(nelem)
    lhs_bitmask = utils.expand_bits_to_bytes(lhs_mask)[:nelem]
    lhs_null_count = utils.count_zero(lhs_bitmask)
    assert lhs_null_count >= 0
    lhs = Series.from_masked_array(lhs_data, lhs_mask)
    assert lhs.null_count == lhs_null_count
    # RHS
    rhs_data = np.random.random(nelem)
    rhs_mask = utils.random_bitmask(nelem)
    rhs_bitmask = utils.expand_bits_to_bytes(rhs_mask)[:nelem]
    rhs_null_count = utils.count_zero(rhs_bitmask)
    assert rhs_null_count >= 0
    rhs = Series.from_masked_array(rhs_data, rhs_mask)
    assert rhs.null_count == rhs_null_count
    # Result
    res = lhs + rhs
    res_mask = np.asarray(utils.expand_bits_to_bytes(lhs_mask & rhs_mask),
                          dtype=np.bool)[:nelem]
    # Fill NA values
    na_value = -10000
    got = res.fillna(na_value).to_array()
    expect = lhs_data + rhs_data
    expect[~res_mask] = na_value
    # Check
    print('expect')
    print(expect)
    print('got')
    print(got)

    np.testing.assert_array_equal(expect, got)


_dtypes = [
    np.int32, np.int64,
    np.float32, np.float64,
]


@pytest.mark.parametrize('binop,lhs_dtype,rhs_dtype',
                         list(product([operator.add, operator.mul],
                                      _dtypes, _dtypes)))
def test_series_binop_mixed_dtype(binop, lhs_dtype, rhs_dtype):
    nelem = 10
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)
    np.testing.assert_almost_equal(binop(Series(lhs), Series(rhs)).to_array(),
                                   binop(lhs, rhs))


@pytest.mark.parametrize('cmpop,lhs_dtype,rhs_dtype',
                         list(product(_cmpops, _dtypes, _dtypes)))
def test_series_cmpop_mixed_dtype(cmpop, lhs_dtype, rhs_dtype):
    print(cmpop, lhs_dtype, rhs_dtype)
    nelem = 5
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)
    print(lhs)
    print(rhs)
    exp = cmpop(lhs, rhs)
    got = cmpop(Series(lhs), Series(rhs)).to_array()
    print('exp', exp)
    print('got', got)

    np.testing.assert_array_equal(cmpop(Series(lhs), Series(rhs)).to_array(),
                                  cmpop(lhs, rhs))


_reflected_ops = [
    lambda x: 1 + x,
    lambda x: 2 * x,
    lambda x: 2 - x,
    lambda x: 2 // x,
    lambda x: 2 / x,
    lambda x: 3 + x,
    lambda x: 3 * x,
    lambda x: 3 - x,
    lambda x: 3 // x,
    lambda x: 3 / x,
    lambda x: -1 + x,
    lambda x: -2 * x,
    lambda x: -2 - x,
    lambda x: -2 // x,
    lambda x: -2 / x,
    lambda x: -3 + x,
    lambda x: -3 * x,
    lambda x: -3 - x,
    lambda x: -3 // x,
    lambda x: -3 / x,
    lambda x: 0 + x,
    lambda x: 0 * x,
    lambda x: 0 - x,
    lambda x: 0 // x,
    lambda x: 0 / x,
]


@pytest.mark.parametrize('func, dtype', list(product(_reflected_ops, _dtypes)))
def test_reflected_ops_scalar(func, dtype):
    import pandas as pd

    # create random series
    np.random.seed(12)
    random_series = pd.Series(np.random.sample(100) + 10, dtype=dtype)

    # gpu series
    gs = Series(random_series)
    gs_result = func(gs)

    # pandas
    ps_result = func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result)

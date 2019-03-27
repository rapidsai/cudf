# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import operator
import random
from itertools import product

import pytest
import numpy as np

from cudf.dataframe import Series
from cudf.dataframe.index import as_index

from cudf.tests import utils


_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.pow,
]


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('binop', _binops)
def test_series_binop(binop, obj_class):
    arr = np.random.random(100)
    sr = Series(arr)

    if obj_class == 'Index':
        sr = as_index(sr)

    result = binop(sr, sr)

    if obj_class == 'Index':
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr, arr))


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('nelem,binop', list(product([1, 2, 100], _binops)))
def test_series_binop_scalar(nelem, binop, obj_class):
    arr = np.random.random(nelem)
    rhs = np.asscalar(random.choice(arr))
    sr = Series(arr)
    if obj_class == 'Index':
        sr = as_index(sr)

    result = binop(sr, rhs)

    if obj_class == 'Index':
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr, rhs))


_bitwise_binops = [
    operator.and_,
    operator.or_,
    operator.xor,
]


_int_types = [
    'int8',
    'int16',
    'int32',
    'int64',
]


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('binop', _bitwise_binops)
@pytest.mark.parametrize('lhs_dtype,rhs_dtype',
                         list(product(_int_types, _int_types)))
def test_series_bitwise_binop(binop, obj_class, lhs_dtype, rhs_dtype):
    arr1 = (np.random.random(100) * 100).astype(lhs_dtype)
    sr1 = Series(arr1)

    arr2 = (np.random.random(100) * 100).astype(rhs_dtype)
    sr2 = Series(arr2)

    if obj_class == 'Index':
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = binop(sr1, sr2)

    if obj_class == 'Index':
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr1, arr2))


_cmpops = [
    operator.lt,
    operator.gt,
    operator.le,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('cmpop', _cmpops)
@pytest.mark.parametrize('dtype', ['int8', 'int32', 'int64',
                                   'float32', 'float64', 'datetime64[ms]'])
def test_series_compare(cmpop, obj_class, dtype):
    arr1 = np.random.randint(0, 100, 100).astype(dtype)
    arr2 = np.random.randint(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    sr2 = Series(arr2)

    if obj_class == 'Index':
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result1 = cmpop(sr1, sr1)
    result2 = cmpop(sr2, sr2)
    result3 = cmpop(sr1, sr2)

    if obj_class == 'Index':
        result1 = Series(result1)
        result2 = Series(result2)
        result3 = Series(result3)

    np.testing.assert_equal(result1.to_array(),  cmpop(arr1, arr1))
    np.testing.assert_equal(result2.to_array(),  cmpop(arr2, arr2))
    np.testing.assert_equal(result3.to_array(),  cmpop(arr1, arr2))


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('nelem', [1, 2, 100])
@pytest.mark.parametrize('cmpop', _cmpops)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64',
                                   'float32', 'float64', 'datetime64[ms]'])
def test_series_compare_scalar(nelem, cmpop, obj_class, dtype):
    arr1 = np.random.randint(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    rhs = np.asscalar(random.choice(arr1))

    if obj_class == 'Index':
        sr1 = as_index(sr1)

    result1 = cmpop(sr1, rhs)
    result2 = cmpop(rhs, sr1)

    if obj_class == 'Index':
        result1 = Series(result1)
        result2 = Series(result2)

    np.testing.assert_equal(result1.to_array(),  cmpop(arr1, rhs))
    np.testing.assert_equal(result2.to_array(),  cmpop(rhs, arr1))


_nulls = [
    'none',
    'some',
]


@pytest.mark.parametrize('nelem', [1, 7, 8, 9, 32, 64, 128])
@pytest.mark.parametrize('lhs_nulls,rhs_nulls', list(product(_nulls, _nulls)))
def test_validity_add(nelem, lhs_nulls, rhs_nulls):
    np.random.seed(0)
    # LHS
    lhs_data = np.random.random(nelem)
    if lhs_nulls == 'some':
        lhs_mask = utils.random_bitmask(nelem)
        lhs_bitmask = utils.expand_bits_to_bytes(lhs_mask)[:nelem]
        lhs_null_count = utils.count_zero(lhs_bitmask)
        assert lhs_null_count >= 0
        lhs = Series.from_masked_array(lhs_data, lhs_mask)
        assert lhs.null_count == lhs_null_count
    else:
        lhs = Series(lhs_data)
    # RHS
    rhs_data = np.random.random(nelem)
    if rhs_nulls == 'some':
        rhs_mask = utils.random_bitmask(nelem)
        rhs_bitmask = utils.expand_bits_to_bytes(rhs_mask)[:nelem]
        rhs_null_count = utils.count_zero(rhs_bitmask)
        assert rhs_null_count >= 0
        rhs = Series.from_masked_array(rhs_data, rhs_mask)
        assert rhs.null_count == rhs_null_count
    else:
        rhs = Series(rhs_data)
    # Result
    res = lhs + rhs
    if lhs_nulls == 'some' and rhs_nulls == 'some':
        res_mask = np.asarray(utils.expand_bits_to_bytes(lhs_mask & rhs_mask),
                              dtype=np.bool)[:nelem]
    if lhs_nulls == 'some' and rhs_nulls == 'none':
        res_mask = np.asarray(utils.expand_bits_to_bytes(lhs_mask),
                              dtype=np.bool)[:nelem]
    if lhs_nulls == 'none' and rhs_nulls == 'some':
        res_mask = np.asarray(utils.expand_bits_to_bytes(rhs_mask),
                              dtype=np.bool)[:nelem]
    # Fill NA values
    na_value = -10000
    got = res.fillna(na_value).to_array()
    expect = lhs_data + rhs_data
    if lhs_nulls == 'some' or rhs_nulls == 'some':
        expect[~res_mask] = na_value

    np.testing.assert_array_equal(expect, got)


_dtypes = [
    np.int32, np.int64,
    np.float32, np.float64,
]


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('binop,lhs_dtype,rhs_dtype',
                         list(product([operator.add, operator.mul],
                                      _dtypes, _dtypes)))
def test_series_binop_mixed_dtype(binop, lhs_dtype, rhs_dtype, obj_class):
    nelem = 10
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == 'Index':
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = binop(Series(sr1), Series(sr2))

    if obj_class == 'Index':
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(),
                                   binop(lhs, rhs))


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('cmpop,lhs_dtype,rhs_dtype',
                         list(product(_cmpops, _dtypes, _dtypes)))
def test_series_cmpop_mixed_dtype(cmpop, lhs_dtype, rhs_dtype, obj_class):
    nelem = 5
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == 'Index':
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = cmpop(Series(sr1), Series(sr2))

    if obj_class == 'Index':
        result = Series(result)

    np.testing.assert_array_equal(result.to_array(),
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


@pytest.mark.parametrize('obj_class', ['Series', 'Index'])
@pytest.mark.parametrize('func, dtype', list(product(_reflected_ops, _dtypes)))
def test_reflected_ops_scalar(func, dtype, obj_class):
    import pandas as pd

    # create random series
    np.random.seed(12)
    random_series = pd.Series(np.random.sample(100) + 10, dtype=dtype)

    # gpu series
    gs = Series(random_series)

    # class typing
    if obj_class == 'Index':
        gs = as_index(gs)

    gs_result = func(gs)

    # class typing
    if obj_class == 'Index':
        gs = Series(gs)

    # pandas
    ps_result = func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result)

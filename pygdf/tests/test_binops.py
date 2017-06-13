from __future__ import division

import pytest
import operator

import numpy as np

from pygdf.dataframe import Series

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
    sr = Series.from_any(arr)
    np.testing.assert_equal(binop(sr, sr).to_array(), binop(arr, arr))


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
    sr1 = Series.from_any(arr1)
    sr2 = Series.from_any(arr2)
    np.testing.assert_equal(cmpop(sr1, sr1).to_array(),  cmpop(arr1, arr1))
    np.testing.assert_equal(cmpop(sr2, sr2).to_array(),  cmpop(arr2, arr2))
    np.testing.assert_equal(cmpop(sr1, sr2).to_array(),  cmpop(arr1, arr2))


def test_validity_add():
    nelem = 8
    # LHS
    lhs_data = np.random.random(nelem)
    lhs_mask = utils.random_bitmask(nelem)
    lhs_bitmask = utils.expand_bits_to_bytes(lhs_mask)
    lhs_null_count = utils.count_zero(lhs_bitmask)
    print(lhs_null_count)
    lhs = Series.from_masked_array(lhs_data, lhs_mask, lhs_null_count)
    # RHS
    rhs_data = np.random.random(nelem)
    rhs_mask = utils.random_bitmask(nelem)
    rhs_bitmask = utils.expand_bits_to_bytes(rhs_mask)
    rhs_null_count = utils.count_zero(rhs_bitmask)
    rhs = Series.from_masked_array(rhs_data, rhs_mask, rhs_null_count)
    # Result
    res = lhs + rhs
    res_mask = np.asarray(utils.expand_bits_to_bytes(lhs_mask & rhs_mask),
                          dtype=np.bool)
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

from __future__ import division

import pytest
import operator

import numpy as np

from pygdf.dataframe import Series


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


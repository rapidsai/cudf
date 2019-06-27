# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import pytest
import numba
import numpy as np

from cudf.bindings import binops
from cudf.dataframe import Series

from numba import cuda
from numba import types
from packaging.version import Version


@pytest.mark.skipif(
    Version(numba.__version__) < Version('0.44.0a'),
    reason="Numba 0.44.0a or newer required"
)
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float32', 'float64'])
def test_generic_ptx(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    rhs_arr = np.random.random(size).astype(dtype)
    rhs_col = Series(rhs_arr)._column

    out_arr = np.random.random(size).astype(dtype)
    out_col = Series(out_arr)._column

    @cuda.jit(device=True)
    def add(a, b):
        return a**3 + b

    if dtype == 'float32':
        type_signature = (types.float32, types.float32)
    elif dtype == 'float64':
        type_signature = (types.float64, types.float64)
    elif dtype == 'int32':
        type_signature = (types.int32, types.int32)
    elif dtype == 'int64':
        type_signature = (types.int64, types.int64)

    add.compile(type_signature)
    ptx = add.inspect_ptx(type_signature)

    ptx_code = ptx.decode('utf-8')

    binops.apply_op_udf(lhs_col, rhs_col, out_col, ptx_code)

    result = lhs_arr**3 + rhs_arr

    np.testing.assert_almost_equal(result, out_col)

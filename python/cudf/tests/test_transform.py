# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import pytest
import numba
import numpy as np

from cudf.bindings import unaryops
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

    out_arr = np.random.random(size).astype(dtype)
    out_col = Series(out_arr)._column

    @cuda.jit(device=True)
    def generic_function(a):
        return a**3

    if dtype == 'float32':
        type_signature = (types.float32,)  # note the trailing comma
    elif dtype == 'float64':
        type_signature = (types.float64,)
    elif dtype == 'int32':
        type_signature = (types.int32,)
    elif dtype == 'int64':
        type_signature = (types.int64,)

    generic_function.compile(type_signature)
    ptx = generic_function.inspect_ptx(type_signature)

    ptx_code = ptx.decode('utf-8')

    unaryops.apply_op_udf(lhs_col, out_col, ptx_code)

    result = lhs_arr**3

    np.testing.assert_almost_equal(result, out_col)

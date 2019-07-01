# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import pytest
import numba
import numpy as np

from cudf.bindings import binops
from cudf.dataframe import Series

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

    @numba.cuda.jit(device=True)
    def generic_function(a, b):
        return a**3 + b

    nb_type = numba.numpy_support.from_dtype(np.dtype(dtype))
    type_signature = (nb_type, nb_type)

    result = generic_function.compile(type_signature)
    ptx = generic_function.inspect_ptx(type_signature)
    ptx_code = ptx.decode('utf-8')

    output_type = numba.numpy_support.as_dtype(result.signature.return_type)

    out_arr = np.random.random(size).astype(output_type)
    out_col = Series(out_arr)._column

    binops.apply_op_udf(lhs_col, rhs_col, out_col, ptx_code, output_type.type)

    result = lhs_arr**3 + rhs_arr

    np.testing.assert_almost_equal(result, out_col)

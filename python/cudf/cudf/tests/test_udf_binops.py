# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import numba
import numpy as np
import pytest

import cudf._lib as libcudf
import cudf.utils.dtypes as dtypeutils
from cudf.core import Series

try:
    # Numba >= 0.49
    from numba.np import numpy_support
except ImportError:
    # Numba <= 0.49
    from numba import numpy_support


@pytest.mark.parametrize(
    "dtype", sorted(list(dtypeutils.NUMERIC_TYPES - {"int8"}))
)
def test_generic_ptx(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    rhs_arr = np.random.random(size).astype(dtype)
    rhs_col = Series(rhs_arr)._column

    @numba.cuda.jit(device=True)
    def generic_function(a, b):
        return a ** 3 + b

    nb_type = numpy_support.from_dtype(np.dtype(dtype))
    type_signature = (nb_type, nb_type)

    result = generic_function.compile(type_signature)
    ptx = generic_function.inspect_ptx(type_signature)
    ptx_code = ptx.decode("utf-8")

    output_type = numpy_support.as_dtype(result.signature.return_type)

    out_col = libcudf.binaryop.binaryop_udf(
        lhs_col, rhs_col, ptx_code, output_type.type
    )

    result = lhs_arr ** 3 + rhs_arr

    np.testing.assert_almost_equal(result, out_col)

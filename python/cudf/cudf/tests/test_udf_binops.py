# Copyright (c) 2018, NVIDIA CORPORATION.
from __future__ import division

import numpy as np
import pytest
from numba.cuda import compile_ptx
from numba.np import numpy_support

import cudf
from cudf import Series, _lib as libcudf
from cudf.utils import dtypes as dtypeutils


@pytest.mark.parametrize(
    "dtype", sorted(list(dtypeutils.NUMERIC_TYPES - {"int8"}))
)
def test_generic_ptx(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    rhs_arr = np.random.random(size).astype(dtype)
    rhs_col = Series(rhs_arr)._column

    def generic_function(a, b):
        return a ** 3 + b

    nb_type = numpy_support.from_dtype(cudf.dtype(dtype))
    type_signature = (nb_type, nb_type)

    ptx_code, output_type = compile_ptx(
        generic_function, type_signature, device=True
    )

    dtype = numpy_support.as_dtype(output_type).type

    out_col = libcudf.binaryop.binaryop_udf(lhs_col, rhs_col, ptx_code, dtype)

    result = lhs_arr ** 3 + rhs_arr

    np.testing.assert_almost_equal(result, out_col.to_array())

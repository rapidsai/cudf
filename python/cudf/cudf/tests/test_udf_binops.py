# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import numba
import numpy as np
import pytest
from packaging.version import Version

import cudf._libxx as libcudfxx
from cudf.core import Series

supported_types = ["int16", "int32", "int64", "float32", "float64"]


@pytest.mark.skipif(
    Version(numba.__version__) < Version("0.44.0a"),
    reason="Numba 0.44.0a or newer required",
)
@pytest.mark.parametrize("dtype", supported_types)
def test_generic_ptx(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    rhs_arr = np.random.random(size).astype(dtype)
    rhs_col = Series(rhs_arr)._column

    @numba.cuda.jit(device=True)
    def generic_function(a, b):
        return a ** 3 + b

    nb_type = numba.numpy_support.from_dtype(np.dtype(dtype))
    type_signature = (nb_type, nb_type)

    result = generic_function.compile(type_signature)
    ptx = generic_function.inspect_ptx(type_signature)
    ptx_code = ptx.decode("utf-8")

    output_type = numba.numpy_support.as_dtype(result.signature.return_type)

    out_col = libcudfxx.binaryop.binaryop_udf(
        lhs_col, rhs_col, ptx_code, output_type.type
    )

    result = lhs_arr ** 3 + rhs_arr

    np.testing.assert_almost_equal(result, out_col)

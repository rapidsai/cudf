# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import pytest
import numba
import numpy as np

from cudf.dataframe import Series

from packaging.version import Version


@pytest.mark.skipif(
    Version(numba.__version__) < Version("0.44.0a"),
    reason="Numba 0.44.0a or newer required",
)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_applymap(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    @numba.cuda.jit(device=True)
    def generic_function(a):
        return a ** 3

    type_signature = (numba.numpy_support.from_dtype(np.dtype(dtype)),)

    result = generic_function.compile(type_signature)
    ptx = generic_function.inspect_ptx(type_signature)
    ptx_code = ptx.decode("utf-8")

    output_type = numba.numpy_support.as_dtype(result.signature.return_type)

    out_col = lhs_col.applymap_ptx(ptx_code, output_type.type)

    result = lhs_arr ** 3

    np.testing.assert_almost_equal(result, out_col)

# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import numba
import numpy as np
import pytest
from packaging.version import Version

from cudf.bindings import rolling
from cudf.dataframe import Series
from cudf.tests import utils
supported_types = ["int16", "int32", "int64", "float32", "float64"]


@pytest.mark.skipif(
    Version(numba.__version__) < Version("0.44.0a"),
    reason="Numba 0.44.0a or newer required",
)
@pytest.mark.parametrize("dtype", supported_types)
def test_rolling_udf(dtype):

    arr_size = 12

    input_arr = np.random.randint(low=0,high=arr_size,size=arr_size).astype(dtype)
    input_col = Series(input_arr)._column

    @numba.cuda.jit(device=True)
    def generic_agg(A, start, length):
        accumulation = 0
        for i in range(length):
            accumulation = accumulation + (A[start+i])**2 + A[start+i]
        return accumulation
    
    nb_type = numba.numpy_support.from_dtype(np.dtype(dtype))
    index_type = numba.numpy_support.types.int32
    type_signature = (nb_type[:], index_type, index_type)
    
    compiled = generic_agg.compile(type_signature)
    ptx_code = generic_agg.inspect_ptx(type_signature).decode('utf-8')
    
    output_type = numba.numpy_support.as_dtype(compiled.signature.return_type)

    op = (ptx_code,output_type.type)
    output_col = rolling.apply_rolling(input_col, 3, 2, True, op)

    expect_arr = np.full(arr_size, -1)
    for i in range(arr_size):
        accumulation = 0
        for j in range(max(0,i-1),min(arr_size,i+2)):
            accumulation = accumulation + input_arr[j]**2 + input_arr[j]
        expect_arr[i] = accumulation
    expect_col = Series(expect_arr)._column

    utils.assert_eq(expect_col, output_col, check_dtype=False)

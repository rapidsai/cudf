# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import numba
import numpy as np
import pytest
from packaging.version import Version

from cudf.dataframe import Series

supported_types = ["int8", "int16", "int32", "int64", "float32", "float64"]


@pytest.mark.skipif(
    Version(numba.__version__) < Version("0.44.0a"),
    reason="Numba 0.44.0a or newer required",
)
@pytest.mark.parametrize("dtype", supported_types)
def test_applymap(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    def generic_function(a):
        return a ** 3

    out_col = lhs_col.applymap(generic_function)

    result = lhs_arr ** 3

    np.testing.assert_almost_equal(result, out_col)

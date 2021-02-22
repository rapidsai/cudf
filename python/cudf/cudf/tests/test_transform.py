# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from __future__ import division

import numpy as np
import pytest

from cudf.core import Series
from cudf.tests.utils import NUMERIC_TYPES

supported_types = NUMERIC_TYPES


@pytest.mark.parametrize("dtype", supported_types)
def test_applymap(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column

    def generic_function(a):
        return a ** 3

    out_col = lhs_col.applymap(generic_function)

    result = lhs_arr ** 3

    np.testing.assert_almost_equal(result, out_col.to_array())


@pytest.mark.parametrize("dtype", supported_types)
def test_applymap_python_lambda(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_ser = Series(lhs_arr)

    # Note that the lambda has to be written this way.
    # In other words, the following code does NOT compile with numba:
    # test_list = [1, 2, 3, 4]
    # out_ser = lhs_ser.applymap(lambda x: x in test_list)
    out_ser = lhs_ser.applymap(lambda x: x in [1, 2, 3, 4])

    result = np.isin(lhs_arr, [1, 2, 3, 4])

    np.testing.assert_almost_equal(result, out_ser.to_array())

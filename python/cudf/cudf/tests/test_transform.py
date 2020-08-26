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

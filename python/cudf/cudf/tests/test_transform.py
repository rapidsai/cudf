# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from __future__ import division

import numpy as np
import pytest

from cudf.core import Series
from cudf.testing._utils import NUMERIC_TYPES

supported_types = NUMERIC_TYPES


def _generic_function(a):
    return a ** 3


@pytest.mark.parametrize("dtype", supported_types)
@pytest.mark.parametrize(
    "udf,testfunc",
    [
        (_generic_function, lambda ser: ser ** 3),
        (lambda x: x in [1, 2, 3, 4], lambda ser: np.isin(ser, [1, 2, 3, 4])),
    ],
)
def test_applymap_python_lambda(dtype, udf, testfunc):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_ser = Series(lhs_arr)

    out_ser = lhs_ser.applymap(udf)
    result = testfunc(lhs_arr)
    np.testing.assert_almost_equal(result, out_ser.to_array())

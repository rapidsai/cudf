# Copyright (c) 2018-2024, NVIDIA CORPORATION.


import numpy as np
import pytest

from cudf import Series
from cudf.testing._utils import NUMERIC_TYPES

supported_types = NUMERIC_TYPES


def _generic_function(a):
    return a**3


@pytest.mark.parametrize("dtype", supported_types)
@pytest.mark.parametrize(
    "udf,testfunc",
    [
        (_generic_function, lambda ser: ser**3),
        (lambda x: x in [1, 2, 3, 4], lambda ser: np.isin(ser, [1, 2, 3, 4])),
    ],
)
def test_apply_python_lambda(dtype, udf, testfunc):
    size = 500
    rng = np.random.default_rng(seed=0)
    lhs_arr = rng.random(size).astype(dtype)
    lhs_ser = Series(lhs_arr)

    out_ser = lhs_ser.apply(udf)
    result = testfunc(lhs_arr)
    np.testing.assert_almost_equal(result, out_ser.to_numpy())

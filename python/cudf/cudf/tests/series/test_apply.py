# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from cudf import Series


def _generic_function(a):
    return a**3


@pytest.mark.parametrize(
    "udf,testfunc",
    [
        (_generic_function, lambda ser: ser**3),
        (lambda x: x in [1, 2, 3, 4], lambda ser: np.isin(ser, [1, 2, 3, 4])),
    ],
)
def test_apply_python_lambda(numeric_types_as_str, udf, testfunc):
    size = 50
    rng = np.random.default_rng(seed=0)
    lhs_arr = rng.random(size).astype(numeric_types_as_str)
    lhs_ser = Series(lhs_arr)

    out_ser = lhs_ser.apply(udf)
    result = testfunc(lhs_arr)
    np.testing.assert_almost_equal(result, out_ser.to_numpy())

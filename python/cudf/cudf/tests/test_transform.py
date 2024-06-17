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
    # TODO: Remove again when NumPy is fixed:
    np_ver = np.lib.NumpyVersion(np.__version__)
    if dtype == "uint64" and (np_ver == "2.0.0rc1" or np_ver == "2.0.0rc2"):
        pytest.skip("test fails due to a NumPy bug on 2.0 rc versions")
    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_ser = Series(lhs_arr)

    out_ser = lhs_ser.apply(udf)
    result = testfunc(lhs_arr)
    np.testing.assert_almost_equal(result, out_ser.to_numpy())

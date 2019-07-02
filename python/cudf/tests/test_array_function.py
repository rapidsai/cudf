# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import cudf
import pytest
from cudf.utils.utils import IS_NEP18_ACTIVE

missing_arrfunc_cond = not IS_NEP18_ACTIVE
missing_arrfunc_reason = "NEP-18 support is not available in NumPy"

# Test implementation based on dask array test
# https://github.com/dask/dask/blob/master/dask/array/tests/test_array_function.py


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "python_ls",
    [
        np.random.random(100),
    ]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x),
        lambda x: np.sum(x),
        lambda x: np.var(x, ddof=1),
        lambda x: np.unique(x)
    ],
)
def test_array_func_cudf(python_ls, func):
    cudf_ser = cudf.Series(python_ls)

    expect = func(python_ls)
    got = func(cudf_ser)

    if np.isscalar(expect):
        np.testing.assert_approx_equal(expect, got)
    else:
        np.testing.assert_array_equal(expect, got.to_pandas().values)


# TODO: Make it future proof
# by adding check if these functions were implemented
@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "python_ls",
    [
        np.random.random(100)
    ]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x)
    ],
)
def test_array_func_missing(python_ls, func):
    cudf_ser = cudf.Series(python_ls)
    with pytest.raises(NotImplementedError):
        func(cudf_ser)
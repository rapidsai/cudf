# # Copyright (c) 2018, NVIDIA CORPORATION.
import numpy as np
import cudf
import pandas as pd
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
        [1, None, 3, 4, 5, None],
        [1, 2, 3],
     ]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x),
        lambda x: np.sum(x),
        lambda x: np.var(x, ddof=1),

    ],
)
def test_array_func_cudf(python_ls, func):
    pandas_ser = pd.Series(python_ls)
    cudf_ser = cudf.Series(python_ls)
    expect = func(pandas_ser)
    got = func(cudf_ser)
    np.testing.assert_approx_equal(expect, got)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "python_ls",
    [
        [1, None, 3, 4, 5, None],
        [1, 2, 3, 4, 5],
     ]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.concatenate([x, x, x]),
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x)
    ],
)
def test_array_func_cupy(python_ls, func):

    flag_none_ls = False

    if None in python_ls:
        flag_none_ls = True

    np_ar = pd.Series(python_ls)
    cudf_ser = cudf.Series(python_ls)

    if flag_none_ls:
        with pytest.raises(ValueError):
            func(cudf_ser)

    else:
        got = func(cudf_ser)
        expect = func(np_ar)

        if np.isscalar(expect):
            assert expect == got
        elif isinstance(expect, pd.DataFrame):
            pd.testing.assert_frame_equal(expect, got.to_pandas())
        else:
            np.testing.assert_array_equal(expect, got.to_pandas().values)

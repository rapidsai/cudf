# Copyright (c) 2018, NVIDIA CORPORATION.
import numpy as np
import cudf
import pytest
import pandas as pd
from cudf.utils.utils import IS_NEP18_ACTIVE

missing_arrfunc_cond = not IS_NEP18_ACTIVE
missing_arrfunc_reason = "NEP-18 support is not available in NumPy"

# Test implementation based on dask array test
# https://github.com/dask/dask/blob/master/dask/array/tests/test_array_function.py


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "np_ar",
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
def test_array_func_cudf_series(np_ar, func):
    cudf_ser = cudf.Series(np_ar)
    expect = func(np_ar)
    got = func(cudf_ser)

    if np.isscalar(expect):
        np.testing.assert_approx_equal(expect, got)
    else:
        np.testing.assert_array_equal(expect, got.to_pandas().values)


# TODO: Make it future proof
# by adding check if these functions were implemented
@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "np_ar",
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
def test_array_func_missing_cudf_series(np_ar, func):
    cudf_ser = cudf.Series(np_ar)
    with pytest.raises(TypeError):
        func(cudf_ser)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "pd_df",
    [
        pd.DataFrame(np.random.uniform(size=(100, 10))),
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
def test_array_func_cudf_dataframe(pd_df, func):
    cudf_df = cudf.from_pandas(pd_df)
    expect = func(pd_df)
    got = func(cudf_df)

    pd.testing.assert_series_equal(expect, got.to_pandas())


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "pd_df",
    [
        pd.DataFrame(np.random.uniform(size=(100, 10))),
    ]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_dataframe(pd_df, func):
    cudf_df = cudf.from_pandas(pd_df)
    with pytest.raises(TypeError):
        func(cudf_df)

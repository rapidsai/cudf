# Copyright (c) 2018-2022, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


# To determine if NEP18 is available in the current version of NumPy we simply
# attempt to concatenate an object with `__array_function__` defined and see if
# NumPy invokes the protocol or not. Taken from dask array
# https://github.com/dask/dask/blob/master/dask/array/utils.py#L352-L363
# TODO: Unclear if this is still necessary. NEP 18 was introduced as the
# default in 1.17 (https://github.com/numpy/numpy/releases/tag/v1.17.0) almost
# 3 years ago, and it was originally introduced one version before in 1.16
# (although not enabled by default then). Can we safely assume that testers
# will have a sufficiently new version of numpy to run these tests?
class _Test:
    def __array_function__(self, *args, **kwargs):
        return True


try:
    np.concatenate([_Test()])
except ValueError:
    missing_arrfunc_cond = True
else:
    missing_arrfunc_cond = False

del _Test

missing_arrfunc_reason = "NEP-18 support is not available in NumPy"


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize("np_ar", [np.random.random(100)])
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x),
        lambda x: np.sum(x),
        lambda x: np.var(x, ddof=1),
        lambda x: np.unique(x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x),
    ],
)
def test_array_func_cudf_series(np_ar, func):
    cudf_ser = cudf.Series(np_ar)
    expect = func(np_ar)
    got = func(cudf_ser)
    if np.isscalar(expect):
        assert_eq(expect, got)
    else:
        assert_eq(expect, got.to_numpy())


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "pd_df", [pd.DataFrame(np.random.uniform(size=(100, 10)))]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x, axis=0),
        lambda x: np.sum(x, axis=0),
        lambda x: np.var(x, ddof=1),
        lambda x: np.dot(x, x.transpose()),
    ],
)
def test_array_func_cudf_dataframe(pd_df, func):
    cudf_df = cudf.from_pandas(pd_df)
    expect = func(pd_df)
    got = func(cudf_df)
    assert_eq(expect, got)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "pd_df", [pd.DataFrame(np.random.uniform(size=(100, 10)))]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_dataframe(pd_df, func):
    cudf_df = cudf.from_pandas(pd_df)
    with pytest.raises(TypeError):
        func(cudf_df)


# we only implement sum among all numpy non-ufuncs
@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize("np_ar", [np.random.random(100)])
@pytest.mark.parametrize("func", [lambda x: np.sum(x), lambda x: np.dot(x, x)])
def test_array_func_cudf_index(np_ar, func):
    cudf_index = cudf.core.index.as_index(cudf.Series(np_ar))
    expect = func(np_ar)
    got = func(cudf_index)
    assert_eq(expect, got)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize("np_ar", [np.random.random(100)])
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_index(np_ar, func):
    cudf_index = cudf.core.index.as_index(cudf.Series(np_ar))
    with pytest.raises(TypeError):
        func(cudf_index)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_multi_index(func):
    levels = [["a", "b"], ["c", "d"]]
    codes = [[0, 1], [1, 0]]

    cudf_multi_index = cudf.MultiIndex(levels, codes)
    with pytest.raises(TypeError):
        func(cudf_multi_index)


@pytest.mark.skipif(missing_arrfunc_cond, reason=missing_arrfunc_reason)
def test_list_input_array_func():
    ar = np.array([1, 2, 3])

    s = cudf.Series(ar)
    with pytest.raises(TypeError):
        np.concatenate([s, s, s])

    s = cudf.Series(ar, index=[1, 2, 3])
    with pytest.raises(TypeError):
        np.concatenate([s, s, s])

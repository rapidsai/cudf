import cudf
import numpy as np
import cupy as cp
import pandas as pd
import pytest
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "np_ar_tup", [(np.random.random(100), np.random.random(100))]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: np.greater(x, y),
        lambda x, y: np.less(x, y),
        lambda x, y: np.less_equal(x, y),
        lambda x, y: np.subtract(x, y),
    ],
)
def test_ufunc_cudf_series(np_ar_tup, func):
    x, y = np_ar_tup[0], np_ar_tup[1]
    s_1, s_2 = cudf.Series(x), cudf.Series(y)
    expect = func(x, y)
    got = func(s_1, s_2)
    if np.isscalar(expect):
        assert_eq(expect, got)
    else:
        assert_eq(expect, got.to_array())


@pytest.mark.parametrize(
    "np_ar_tup", [(np.random.random(100), np.random.random(100))]
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: np.greater(x, y),
        lambda x, y: np.less(x, y),
        lambda x, y: np.less_equal(x, y),
    ],
)
def test_ufunc_cudf_series_cupy_array(np_ar_tup, func):
    x, y = np_ar_tup[0], np_ar_tup[1]
    expect = func(x, y)

    cudf_s = cudf.Series(x)
    cupy_ar = cp.array(y)
    got = func(cudf_s, cupy_ar)
    if np.isscalar(expect):
        assert_eq(expect, got)
    else:
        assert_eq(expect, got.to_array())


@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: np.greater(x, y),
        lambda x, y: np.less(x, y),
        lambda x, y: np.less_equal(x, y),
    ],
)
def test_error_with_null_cudf_series(func):
    s_1 = cudf.Series([1, 2])
    s_2 = cudf.Series([1, None])

    # this thows a value error
    # because of nulls in cudf.Series
    with pytest.raises(ValueError):
        func(s_1, s_2)

    s_1 = cudf.Series([1, 2])
    s_2 = cudf.Series([1, 2, None])

    # this throws a type-error because
    # indexes are not aligned
    with pytest.raises(TypeError):
        func(s_1, s_2)


@pytest.mark.parametrize(
    "func", [np.absolute,  np.sign],
)
def test_ufunc_cudf_series_with_index(func):
    data = [-1, 2, 3, 0]
    index = [2, 3, 1, 0]
    cudf_s = cudf.Series(data=data, index=index)
    pd_s = pd.Series(data=data, index=index)

    expect = func(pd_s)
    got = func(cudf_s)

    assert_eq(got, expect)


@pytest.mark.parametrize(
    "func", [lambda x, y: np.greater(x, y), lambda x, y: np.logaddexp(x, y)],
)
def test_ufunc_cudf_series_with_nonaligned_index(func):
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0], index=[2, 3, 1, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0], index=[3, 1, 0, 2])

    with pytest.raises(TypeError):
        func(cudf_s1, cudf_s2)

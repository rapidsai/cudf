import cudf
import numpy as np
import cupy as cp
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
    s_2 = cudf.Series([1, 2, None])
    with pytest.raises(ValueError):
        func(s_1, s_2)

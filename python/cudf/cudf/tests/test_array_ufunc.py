import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


@pytest.fixture
def np_ar_tup():
    np.random.seed(0)
    return (np.random.random(100), np.random.random(100))


comparison_ops_ls = [
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
]


@pytest.mark.parametrize(
    "func", comparison_ops_ls + [np.subtract, np.fmod, np.power]
)
def test_ufunc_cudf_non_nullseries(np_ar_tup, func):
    x, y = np_ar_tup[0], np_ar_tup[1]
    s_1, s_2 = cudf.Series(x), cudf.Series(y)
    expect = func(x, y)
    got = func(s_1, s_2)
    assert_eq(expect, got.to_array())


@pytest.mark.parametrize(
    "func", [np.bitwise_and, np.bitwise_or, np.bitwise_xor],
)
def test_ufunc_cudf_series_bitwise(func):
    np.random.seed(0)
    x = np.random.randint(size=100, low=0, high=100)
    y = np.random.randint(size=100, low=0, high=100)

    s_1, s_2 = cudf.Series(x), cudf.Series(y)
    expect = func(x, y)
    got = func(s_1, s_2)
    assert_eq(expect, got.to_array())


@pytest.mark.parametrize(
    "func",
    [
        np.subtract,
        np.multiply,
        np.floor_divide,
        np.true_divide,
        np.power,
        np.remainder,
        np.divide,
    ],
)
def test_ufunc_cudf_null_series(np_ar_tup, func):
    x, y = np_ar_tup[0].astype(np.float32), np_ar_tup[1].astype(np.float32)
    x[0] = np.nan
    y[1] = np.nan
    s_1, s_2 = cudf.Series(x), cudf.Series(y)
    expect = func(x, y)
    got = func(s_1, s_2)
    assert_eq(expect, got.fillna(np.nan).to_array())

    scalar = 0.5
    expect = func(x, scalar)
    got = func(s_1, scalar)
    assert_eq(expect, got.fillna(np.nan).to_array())

    expect = func(scalar, x)
    got = func(scalar, s_1)
    assert_eq(expect, got.fillna(np.nan).to_array())


@pytest.mark.xfail(
    reason="""cuDF comparison operations with <NA> incorrectly
    returns False rather than <NA>"""
)
@pytest.mark.parametrize(
    "func", comparison_ops_ls,
)
def test_ufunc_cudf_null_series_comparison_ops(np_ar_tup, func):
    x, y = np_ar_tup[0].astype(np.float32), np_ar_tup[1].astype(np.float32)
    x[0] = np.nan
    y[1] = np.nan
    s_1, s_2 = cudf.Series(x), cudf.Series(y)
    expect = func(x, y)
    got = func(s_1, s_2)
    assert_eq(expect, got.fillna(np.nan).to_array())

    scalar = 0.5
    expect = func(x, scalar)
    got = func(s_1, scalar)
    assert_eq(expect, got.fillna(np.nan).to_array())

    expect = func(scalar, x)
    got = func(scalar, s_1)
    assert_eq(expect, got.fillna(np.nan).to_array())


@pytest.mark.parametrize(
    "func", [np.logaddexp, np.fmax, np.fmod],
)
def test_ufunc_cudf_series_cupy_array(np_ar_tup, func):
    x, y = np_ar_tup[0], np_ar_tup[1]
    expect = func(x, y)

    cudf_s = cudf.Series(x)
    cupy_ar = cp.array(y)
    got = func(cudf_s, cupy_ar)
    assert_eq(expect, got.to_array())


@pytest.mark.parametrize(
    "func",
    [np.fmod, np.logaddexp, np.bitwise_and, np.bitwise_or, np.bitwise_xor],
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

    # this throws a value-error if indexes are not aligned
    # following pandas behavior for ufunc numpy dispatching
    with pytest.raises(
        ValueError, match="Can only compare identically-labeled Series objects"
    ):
        func(s_1, s_2)


@pytest.mark.parametrize(
    "func", [np.absolute, np.sign, np.exp2, np.tanh],
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
    "func", [np.logaddexp2],
)
def test_ufunc_cudf_series_with_nonaligned_index(func):
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0], index=[2, 3, 1, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0], index=[3, 1, 0, 2])

    # this throws a value-error if indexes are not aligned
    # following pandas behavior for ufunc numpy dispatching
    with pytest.raises(
        ValueError, match="Can only compare identically-labeled Series objects"
    ):
        func(cudf_s1, cudf_s2)


@pytest.mark.parametrize(
    "func", [np.add],
)
def test_ufunc_cudf_series_error_with_out_kwarg(func):
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s3 = cudf.Series(data=[0, 0, 0, 0])
    # this throws a value-error because of presence of out kwarg
    with pytest.raises(TypeError):
        func(x1=cudf_s1, x2=cudf_s2, out=cudf_s3)

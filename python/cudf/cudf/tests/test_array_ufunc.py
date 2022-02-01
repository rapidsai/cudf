import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq

_UFUNCS = [
    obj
    for obj in (getattr(np, name) for name in dir(np))
    if isinstance(obj, np.ufunc)
]


@pytest.mark.parametrize("ufunc", _UFUNCS)
def test_ufunc_series(ufunc):
    is_binary = ufunc.nin == 2
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc
    args = [cudf.Series(cp.random.randint(size=100, low=1, high=10))]
    if is_binary:
        args.append(cudf.Series(cp.random.randint(size=100, low=1, high=10)))

    try:
        got = ufunc(*args)
    except AttributeError as e:
        # We xfail if we don't have an explicit dispatch and cupy doesn't have
        # the method. xfail is preferable to a silent pass so that we can
        # identify these if we decide to implement them in the future. As of
        # this writing, the only missing methods are isnat and heaviside.
        if "module 'cupy' has no attribute" in str(e):
            pytest.xfail(reason="Operation not supported by cupy")
        raise

    expect = ufunc(*(arg.to_pandas() for arg in args))

    try:
        assert_eq(got, expect)
    except AssertionError:
        if ufunc.__name__ in ("power", "float_power"):
            equivalence = cudf.from_pandas(expect) != got
            diffs = got[equivalence] - expect[equivalence.to_pandas()]
            if np.max(np.abs(diffs)) == 1:
                pytest.xfail("https://github.com/rapidsai/cudf/issues/10178")
        raise


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
    assert_eq(expect, got.to_numpy())


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
    assert_eq(expect, got.to_numpy())


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
    assert_eq(expect, got.fillna(np.nan).to_numpy())

    scalar = 0.5
    expect = func(x, scalar)
    got = func(s_1, scalar)
    assert_eq(expect, got.fillna(np.nan).to_numpy())

    expect = func(scalar, x)
    got = func(scalar, s_1)
    assert_eq(expect, got.fillna(np.nan).to_numpy())


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
    assert_eq(expect, got.fillna(np.nan).to_numpy())

    scalar = 0.5
    expect = func(x, scalar)
    got = func(s_1, scalar)
    assert_eq(expect, got.fillna(np.nan).to_numpy())

    expect = func(scalar, x)
    got = func(scalar, s_1)
    assert_eq(expect, got.fillna(np.nan).to_numpy())


@pytest.mark.parametrize(
    "func", [np.logaddexp, np.fmax, np.fmod],
)
def test_ufunc_cudf_series_cupy_array(np_ar_tup, func):
    x, y = np_ar_tup[0], np_ar_tup[1]
    expect = func(x, y)

    cudf_s = cudf.Series(x)
    cupy_ar = cp.array(y)
    got = func(cudf_s, cupy_ar)
    assert_eq(expect, got.to_numpy())


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
    ps1 = cudf_s1.to_pandas()
    ps2 = cudf_s2.to_pandas()

    expect = func(ps1, ps2)
    got = func(cudf_s1, cudf_s2)

    assert_eq(got, expect)


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

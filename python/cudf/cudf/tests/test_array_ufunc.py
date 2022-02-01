import operator
from functools import reduce

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq, set_random_null_mask_inplace

_UFUNCS = [
    obj
    for obj in (getattr(np, name) for name in dir(np))
    if isinstance(obj, np.ufunc)
]


@pytest.mark.parametrize("ufunc", _UFUNCS)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_series(ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__
    if indexed and fname in (
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "not_equal",
        "equal",
    ):
        pytest.skip("Comparison operators do not support misaligned indexes.")

    if (indexed or has_nulls) and fname == "matmul":
        pytest.xfail("Frame.dot currently does not support indexes or nulls")

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    pandas_args = args = [
        cudf.Series(
            cp.random.randint(low=1, high=10, size=N),
            index=cp.random.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(ufunc.nin)
    ]

    if has_nulls:
        # Converting nullable integer cudf.Series to pandas will produce a
        # float pd.Series, so instead we replace nulls with an arbitrary
        # integer value, precompute the mask, and then reapply it afterwards.
        for arg in args:
            set_random_null_mask_inplace(arg)
        pandas_args = [arg.fillna(0) for arg in args]

        # Note: Different indexes must be aligned before the mask is computed.
        # This requires using an internal function (_align_indices), and that
        # is unlikely to change for the foreseeable future.
        aligned = (
            cudf.core.series._align_indices(args, allow_non_unique=True)
            if indexed and ufunc.nin == 2
            else args
        )
        mask = reduce(operator.or_, (a.isna() for a in aligned)).to_pandas()

    try:
        got = ufunc(*args)
    except AttributeError as e:
        # We xfail if we don't have an explicit dispatch and cupy doesn't have
        # the method so that we can easily identify these methods. As of this
        # writing, the only missing methods are isnat and heaviside.
        if "module 'cupy' has no attribute" in str(e):
            pytest.xfail(reason="Operation not supported by cupy")
        raise

    expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    try:
        if ufunc.nout > 1:
            for g, e in zip(got, expect):
                if has_nulls:
                    e[mask] = np.nan
                assert_eq(g, e)
        else:
            if has_nulls:
                expect[mask] = np.nan
            assert_eq(got, expect)
    except AssertionError:
        if fname in ("power", "float_power"):
            not_equal = cudf.from_pandas(expect) != got
            not_equal[got.isna()] = False
            diffs = got[not_equal] - expect[not_equal.to_pandas()]
            if diffs.abs().max() == 1:
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

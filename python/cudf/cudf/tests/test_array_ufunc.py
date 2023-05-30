# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import operator
import warnings
from contextlib import contextmanager
from functools import reduce

import cupy as cp
import numpy as np
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_150
from cudf.testing._utils import assert_eq, set_random_null_mask_inplace

_UFUNCS = [
    obj
    for obj in (getattr(np, name) for name in dir(np))
    if isinstance(obj, np.ufunc)
]


@contextmanager
def _hide_ufunc_warnings(ufunc):
    # pandas raises warnings for some inputs to the following ufuncs:
    name = ufunc.__name__
    if name in {
        "arccos",
        "arccosh",
        "arcsin",
        "arctanh",
        "fmod",
        "log",
        "log10",
        "log2",
        "reciprocal",
    }:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                f"invalid value encountered in {name}",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                f"divide by zero encountered in {name}",
                category=RuntimeWarning,
            )
            yield
    else:
        yield


@pytest.mark.parametrize("ufunc", _UFUNCS)
def test_ufunc_index(request, ufunc):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                fname in {"bitwise_and", "bitwise_or", "bitwise_xor"}
                and not PANDAS_GE_150
            ),
            reason="https://github.com/pandas-dev/pandas/issues/46769",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=not hasattr(cp, fname),
            reason=f"cupy has no support for '{fname}'",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    pandas_args = args = [
        cudf.Index(
            cp.random.randint(low=1, high=10, size=N),
        )
        for _ in range(ufunc.nin)
    ]

    got = ufunc(*args)

    with _hide_ufunc_warnings(ufunc):
        expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            assert_eq(g, e, check_exact=False)
    else:
        assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_index_array(ufunc, reflect):
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    args = [cudf.Index(cp.random.rand(N)) for _ in range(ufunc.nin)]

    arg1 = args[1].to_cupy()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            if reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize("ufunc", _UFUNCS)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_series(request, ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and fname
                in {
                    "greater",
                    "greater_equal",
                    "less",
                    "less_equal",
                    "not_equal",
                    "equal",
                }
            ),
            reason="Comparison operators do not support misaligned indexes.",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=ufunc == np.matmul and has_nulls,
            reason="Can't call cupy on column with nulls",
        )
    )
    # If we don't have explicit dispatch and cupy doesn't support the operator,
    # we expect a failure
    request.applymarker(
        pytest.mark.xfail(
            condition=not hasattr(cp, fname),
            reason=f"cupy has no support for '{fname}'",
        )
    )

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

    got = ufunc(*args)

    with _hide_ufunc_warnings(ufunc):
        expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            if has_nulls:
                e[mask] = np.nan
            assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            expect[mask] = np.nan
            assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_series_array(
    request, ufunc, has_nulls, indexed, reflect
):
    fname = ufunc.__name__
    request.applymarker(
        pytest.mark.xfail(
            condition=reflect and has_nulls,
            reason=(
                "When cupy is the left operand there is no way for us to "
                "avoid calling its binary operators, which cannot handle "
                "cudf objects that contain nulls."
            ),
        )
    )
    # The way cudf casts nans in arrays to nulls during binops with cudf
    # objects is currently incompatible with pandas.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                fname in {"greater", "greater_equal", "logical_and"}
                and has_nulls
            ),
            reason=(
                "cudf and pandas incompatible casting nans "
                "to nulls in binops"
            ),
        )
    )
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    args = [
        cudf.Series(
            cp.random.rand(N),
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

        # Cupy doesn't support nulls, so we fill with nans before converting.
        args[1] = args[1].fillna(cp.nan)
        mask = args[0].isna().to_pandas()

    arg1 = args[1].to_cupy()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            if has_nulls:
                e[mask] = np.nan
            if reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            expect[mask] = np.nan
        if reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "func",
    [np.add],
)
def test_ufunc_cudf_series_error_with_out_kwarg(func):
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s3 = cudf.Series(data=[0, 0, 0, 0])
    # this throws a value-error because of presence of out kwarg
    with pytest.raises(TypeError):
        func(x1=cudf_s1, x2=cudf_s2, out=cudf_s3)


# Skip matmul since it requires aligned shapes.
@pytest.mark.parametrize("ufunc", (uf for uf in _UFUNCS if uf != np.matmul))
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_dataframe(request, ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and fname
                in {
                    "greater",
                    "greater_equal",
                    "less",
                    "less_equal",
                    "not_equal",
                    "equal",
                }
            ),
            reason="Comparison operators do not support misaligned indexes.",
        )
    )
    # If we don't have explicit dispatch and cupy doesn't support the operator,
    # we expect a failure
    request.applymarker(
        pytest.mark.xfail(
            condition=not hasattr(cp, fname),
            reason=f"cupy has no support for '{fname}'",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and fname
                in {
                    "add",
                    "arctan2",
                    "bitwise_and",
                    "bitwise_or",
                    "bitwise_xor",
                    "copysign",
                    "divide",
                    "divmod",
                    "float_power",
                    "floor_divide",
                    "fmax",
                    "fmin",
                    "fmod",
                    "heaviside",
                    "gcd",
                    "hypot",
                    "lcm",
                    "ldexp",
                    "left_shift",
                    "logaddexp",
                    "logaddexp2",
                    "logical_and",
                    "logical_or",
                    "logical_xor",
                    "maximum",
                    "minimum",
                    "multiply",
                    "nextafter",
                    "power",
                    "remainder",
                    "right_shift",
                    "subtract",
                }
            ),
            reason=(
                "pandas does not currently support misaligned "
                "indexes in DataFrames"
            ),
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    # TODO: Add tests of mismatched columns etc.
    pandas_args = args = [
        cudf.DataFrame(
            {"foo": cp.random.randint(low=1, high=10, size=N)},
            index=cp.random.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(ufunc.nin)
    ]

    if has_nulls:
        # Converting nullable integer cudf.Series to pandas will produce a
        # float pd.Series, so instead we replace nulls with an arbitrary
        # integer value, precompute the mask, and then reapply it afterwards.
        for arg in args:
            set_random_null_mask_inplace(arg["foo"])
        pandas_args = [arg.copy() for arg in args]
        for arg in pandas_args:
            arg["foo"] = arg["foo"].fillna(0)

        # Note: Different indexes must be aligned before the mask is computed.
        # This requires using an internal function (_align_indices), and that
        # is unlikely to change for the foreseeable future.
        aligned = (
            cudf.core.dataframe._align_indices(*args)
            if indexed and ufunc.nin == 2
            else args
        )
        mask = reduce(
            operator.or_, (a["foo"].isna() for a in aligned)
        ).to_pandas()

    got = ufunc(*args)

    with _hide_ufunc_warnings(ufunc):
        expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            if has_nulls:
                e[mask] = np.nan
            assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            expect[mask] = np.nan
        assert_eq(got, expect, check_exact=False)

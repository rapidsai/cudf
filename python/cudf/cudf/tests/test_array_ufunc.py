# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import operator
import warnings
from contextlib import contextmanager
from functools import reduce

import cupy as cp
import numpy as np
import pytest

import cudf
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
def test_ufunc_index(ufunc):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__

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
                assert_eq(g, e, check_exact=False)
        else:
            assert_eq(got, expect, check_exact=False)
    except AssertionError:
        # TODO: This branch can be removed when
        # https://github.com/rapidsai/cudf/issues/10178 is resolved
        if fname in ("power", "float_power"):
            if (got - expect).abs().max() == 1:
                pytest.xfail("https://github.com/rapidsai/cudf/issues/10178")
        elif fname in ("bitwise_and", "bitwise_or", "bitwise_xor"):
            pytest.xfail("https://github.com/pandas-dev/pandas/issues/46769")
        raise


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("type_", ["cupy", "numpy", "list"])
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_index_array(ufunc, type_, reflect):
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    args = [cudf.Index(cp.random.rand(N)) for _ in range(ufunc.nin)]

    arg1 = args[1].to_cupy() if type_ == "cupy" else args[1].to_numpy()
    if type_ == "list":
        arg1 = arg1.tolist()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect):
            if type_ == "cupy" and reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if type_ == "cupy" and reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)


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

    with _hide_ufunc_warnings(ufunc):
        expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    try:
        if ufunc.nout > 1:
            for g, e in zip(got, expect):
                if has_nulls:
                    e[mask] = np.nan
                assert_eq(g, e, check_exact=False)
        else:
            if has_nulls:
                expect[mask] = np.nan
            assert_eq(got, expect, check_exact=False)
    except AssertionError:
        # TODO: This branch can be removed when
        # https://github.com/rapidsai/cudf/issues/10178 is resolved
        if fname in ("power", "float_power"):
            not_equal = cudf.from_pandas(expect) != got
            not_equal[got.isna()] = False
            diffs = got[not_equal] - expect[not_equal.to_pandas()]
            if diffs.abs().max() == 1:
                pytest.xfail("https://github.com/rapidsai/cudf/issues/10178")
        raise


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("type_", ["cupy", "numpy", "list"])
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_series_array(ufunc, has_nulls, indexed, type_, reflect):
    fname = ufunc.__name__
    if fname in ("greater", "greater_equal", "logical_and") and has_nulls:
        pytest.xfail(
            "The way cudf casts nans in arrays to nulls during binops with "
            "cudf objects is currently incompatible with pandas."
        )
    if reflect and has_nulls and type_ == "cupy":
        pytest.skip(
            "When cupy is the left operand there is no way for us to avoid "
            "calling its binary operators, which cannot handle cudf objects "
            "that contain nulls."
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

    arg1 = args[1].to_cupy() if type_ == "cupy" else args[1].to_numpy()
    if type_ == "list":
        arg1 = arg1.tolist()

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
            if type_ == "cupy" and reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            expect[mask] = np.nan
        if type_ == "cupy" and reflect:
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
def test_ufunc_dataframe(ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    fname = ufunc.__name__
    # TODO: When pandas starts supporting misaligned indexes properly, remove
    # this check but enable the one below.
    if indexed:
        pytest.xfail(
            "pandas does not currently support misaligned indexes in "
            "DataFrames, but we do. Until this is fixed we will skip these "
            "tests. See the error here: "
            "https://github.com/pandas-dev/pandas/blob/main/pandas/core/arraylike.py#L212, "  # noqa: E501
            "called from https://github.com/pandas-dev/pandas/blob/main/pandas/core/arraylike.py#L258"  # noqa: E501
        )
    # TODO: Enable the check below when we remove the check above.
    # if indexed and fname in (
    #     "greater",
    #     "greater_equal",
    #     "less",
    #     "less_equal",
    #     "not_equal",
    #     "equal",
    # ):
    #     pytest.skip("Comparison operators do not support misaligned indexes.")  # noqa: E501

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

    try:
        got = ufunc(*args)
    except AttributeError as e:
        # We xfail if we don't have an explicit dispatch and cupy doesn't have
        # the method so that we can easily identify these methods. As of this
        # writing, the only missing methods are isnat and heaviside.
        if "module 'cupy' has no attribute" in str(e):
            pytest.xfail(reason="Operation not supported by cupy")
        raise

    with _hide_ufunc_warnings(ufunc):
        expect = ufunc(*(arg.to_pandas() for arg in pandas_args))

    try:
        if ufunc.nout > 1:
            for g, e in zip(got, expect):
                if has_nulls:
                    e[mask] = np.nan
                assert_eq(g, e, check_exact=False)
        else:
            if has_nulls:
                expect[mask] = np.nan
            assert_eq(got, expect, check_exact=False)
    except AssertionError:
        # TODO: This branch can be removed when
        # https://github.com/rapidsai/cudf/issues/10178 is resolved
        if fname in ("power", "float_power"):
            not_equal = cudf.from_pandas(expect) != got
            not_equal[got.isna()] = False
            diffs = got[not_equal] - cudf.from_pandas(
                expect[not_equal.to_pandas()]
            )
            if diffs["foo"].abs().max() == 1:
                pytest.xfail("https://github.com/rapidsai/cudf/issues/10178")
        raise

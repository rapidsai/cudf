# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime
import operator
from functools import reduce

import cupy as cp
import numpy as np
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if, set_random_null_mask_inplace


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_series(request, numpy_ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and numpy_ufunc
                in {
                    np.greater,
                    np.greater_equal,
                    np.less,
                    np.less_equal,
                    np.not_equal,
                    np.equal,
                }
            ),
            reason="Comparison operators do not support misaligned indexes.",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc == np.matmul and has_nulls,
            reason="Can't call cupy on column with nulls",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc.__name__.startswith("bitwise")
            and indexed
            and has_nulls,
            reason="https://github.com/pandas-dev/pandas/issues/52500",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    pandas_args = args = [
        cudf.Series(
            rng.integers(low=1, high=10, size=N),
            index=rng.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(numpy_ufunc.nin)
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
            if indexed and numpy_ufunc.nin == 2
            else args
        )
        mask = reduce(operator.or_, (a.isna() for a in aligned)).to_pandas()

    got = numpy_ufunc(*args)

    expect = numpy_ufunc(*(arg.to_pandas() for arg in pandas_args))

    if numpy_ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if has_nulls:
                e[mask] = np.nan
            assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            with expect_warning_if(
                numpy_ufunc
                in (
                    np.isfinite,
                    np.isinf,
                    np.isnan,
                    np.logical_and,
                    np.logical_not,
                    np.logical_or,
                    np.logical_xor,
                    np.signbit,
                    np.equal,
                    np.greater,
                    np.greater_equal,
                    np.less,
                    np.less_equal,
                    np.not_equal,
                )
            ):
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
                "cudf and pandas incompatible casting nans to nulls in binops"
            ),
        )
    )
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    args = [
        cudf.Series(
            rng.random(N),
            index=rng.choice(range(N), N, False) if indexed else None,
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
        for g, e in zip(got, expect, strict=True):
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


def test_ufunc_cudf_series_error_with_out_kwarg():
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s3 = cudf.Series(data=[0, 0, 0, 0])
    with pytest.raises(TypeError):
        np.add(x1=cudf_s1, x2=cudf_s2, out=cudf_s3)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("op", [np.add, np.subtract])
def test_datetime_series_ops_with_scalars_misc(
    data, scalar, datetime_types_as_str, op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expect = op(psr, scalar)
    got = op(gsr, scalar)

    assert_eq(expect, got)

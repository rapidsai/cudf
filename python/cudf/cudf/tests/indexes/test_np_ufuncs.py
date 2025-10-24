# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cupy as cp
import numpy as np
import pytest
from packaging.version import parse

import cudf
from cudf.core._compat import (
    PANDAS_LT_300,
)
from cudf.testing import assert_eq


def test_ufunc_index(request, numpy_ufunc):
    # Note: This test assumes that all ufuncs are unary or binary.
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc == np.matmul and PANDAS_LT_300,
            reason="Fixed by https://github.com/pandas-dev/pandas/pull/57079",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc in {np.ceil, np.floor, np.trunc}
            and parse(np.__version__) >= parse("2.1")
            and parse(cp.__version__) < parse("14"),
            reason="https://github.com/cupy/cupy/issues/9018",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    pandas_args = args = [
        cudf.Index(
            rng.integers(low=1, high=10, size=N),
        )
        for _ in range(numpy_ufunc.nin)
    ]

    got = numpy_ufunc(*args)

    expect = numpy_ufunc(*(arg.to_pandas() for arg in pandas_args))

    if numpy_ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
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
    rng = np.random.default_rng(0)
    args = [cudf.Index(rng.random(N)) for _ in range(ufunc.nin)]

    arg1 = args[1].to_cupy()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq


@pytest.fixture(params=[0, 5])
def nelem(request):
    return request.param


@pytest.fixture(params=["cumsum", "cummin", "cummax", "cumprod"])
def cumulative_methods(request):
    return request.param


def test_cumulative_methods(numeric_types_as_str, nelem, cumulative_methods):
    dtype = np.dtype(numeric_types_as_str)
    rng = np.random.default_rng(0)
    if dtype == np.int8:
        data = rng.integers(-2, 2, size=nelem).astype(np.int8)
    elif dtype.kind == "f":
        data = rng.random(nelem).astype(dtype) * 2 - 1
    elif dtype.kind in ("i", "u"):
        if dtype == np.int16:
            low, high = -32, 32
        elif dtype.kind == "i":
            low, high = -10000, 10000
        elif dtype in (np.uint8, np.uint16):
            low, high = 0, 32
        else:
            low, high = 0, 128
        data = rng.integers(low=low, high=high, size=nelem).astype(dtype)

    decimal = 4 if dtype == np.float32 else 6

    gs = cudf.Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        getattr(gs, cumulative_methods)().to_numpy(),
        getattr(ps, cumulative_methods)(),
        decimal=decimal,
    )


def test_cumulative_methods_masked(numeric_types_as_str, cumulative_methods):
    data = [1, 2, None, 4, 5]
    gs = cudf.Series(data).astype(numeric_types_as_str)
    # float64 since pandas usses NaN as missing value
    ps = pd.Series(data).astype("float64")
    assert_eq(
        getattr(gs, cumulative_methods)(),
        getattr(ps, cumulative_methods)(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(8, 4),
        Decimal64Dtype(10, 5),
        Decimal64Dtype(12, 7),
        Decimal32Dtype(8, 5),
        Decimal128Dtype(13, 6),
    ],
)
def test_cumsum_decimal(dtype):
    data = ["243.32", "48.245", "-7234.298", np.nan, "-467.2"]
    gser = cudf.Series(data).astype(dtype)
    pser = pd.Series(data, dtype="float64")

    got = gser.cumsum()
    expected = cudf.Series(pser.cumsum()).astype(dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(8, 4),
        Decimal64Dtype(11, 6),
        Decimal64Dtype(14, 7),
        Decimal32Dtype(8, 4),
        Decimal128Dtype(11, 6),
    ],
)
def test_cummin_decimal(dtype):
    data = ["8394.294", np.nan, "-9940.444", np.nan, "-23.928"]
    gser = cudf.Series(data).astype(dtype)
    pser = pd.Series(data, dtype="float64")

    got = gser.cummin()
    expected = cudf.Series(pser.cummin()).astype(dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(8, 4),
        Decimal64Dtype(11, 6),
        Decimal64Dtype(14, 7),
        Decimal32Dtype(8, 4),
        Decimal128Dtype(11, 6),
    ],
)
def test_cummax_decimal(dtype):
    data = [np.nan, "54.203", "8.222", "644.32", "-562.272"]
    gser = cudf.Series(data).astype(dtype)
    pser = pd.Series(data, dtype="float64")

    got = gser.cummax()
    expected = cudf.Series(pser.cummax()).astype(dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("method", ["cumsum", "cumprod"])
def test_scan_boolean(method):
    s = cudf.Series([True, False, True, False])

    got = getattr(s, method)()
    expect = getattr(s.to_pandas(), method)()

    assert_eq(expect, got)


def test_cummin_cummax_strings():
    data = ["dog", "cat", "zebra", "ant", "bat"]
    gser = cudf.Series(data)
    pser = pd.Series(data)

    got_min = gser.cummin()
    expected_min = pser.cummin()
    assert_eq(got_min, expected_min)

    got_max = gser.cummax()
    expected_max = pser.cummax()
    assert_eq(got_max, expected_max)

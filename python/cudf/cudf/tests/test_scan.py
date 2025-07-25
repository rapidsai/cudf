# Copyright (c) 2021-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import INTEGER_TYPES, NUMERIC_TYPES, gen_rand


@pytest.fixture(params=NUMERIC_TYPES)
def dtype(request):
    return request.param


@pytest.fixture(params=[0, 1, 5])
def nelem(request):
    return request.param


def test_cumsum(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = cudf.Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cumsum().to_numpy(), ps.cumsum(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cumsum().to_numpy(), pdf.a.cumsum(), decimal=decimal
    )


def test_cumsum_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = cudf.Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumsum(), ps.cumsum())

    for type_ in INTEGER_TYPES:
        gs = cudf.Series(data).astype(type_)
        got = gs.cumsum()
        expected = pd.Series([1, 3, np.nan, 7, 12], dtype="float64")
        assert_eq(got, expected)


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
    expected = cudf.Series.from_pandas(pser.cumsum()).astype(dtype)

    assert_eq(got, expected)


def test_cummin(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = cudf.Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cummin().to_numpy(), ps.cummin(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cummin().to_numpy(), pdf.a.cummin(), decimal=decimal
    )


def test_cummin_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = cudf.Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummin(), ps.cummin())

    for type_ in INTEGER_TYPES:
        gs = cudf.Series(data).astype(type_)
        expected = pd.Series([1, 1, np.nan, 1, 1]).astype("float64")
        assert_eq(gs.cummin(), expected)


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
    expected = cudf.Series.from_pandas(pser.cummin()).astype(dtype)

    assert_eq(got, expected)


def test_cummax(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = cudf.Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cummax().to_numpy(), ps.cummax(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cummax().to_numpy(), pdf.a.cummax(), decimal=decimal
    )


def test_cummax_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = cudf.Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummax(), ps.cummax())

    for type_ in INTEGER_TYPES:
        gs = cudf.Series(data).astype(type_)
        expected = pd.Series([1, 2, np.nan, 4, 5]).astype("float64")
        assert_eq(gs.cummax(), expected)


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
    expected = cudf.Series.from_pandas(pser.cummax()).astype(dtype)

    assert_eq(got, expected)


def test_cumprod(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = cudf.Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cumprod().to_numpy(), ps.cumprod(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cumprod().to_numpy(), pdf.a.cumprod(), decimal=decimal
    )


def test_cumprod_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = cudf.Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumprod(), ps.cumprod())

    for type_ in INTEGER_TYPES:
        gs = cudf.Series(data).astype(type_)
        got = gs.cumprod()
        expected = pd.Series([1, 2, np.nan, 8, 40], dtype="float64")
        assert_eq(got, expected)


def test_scan_boolean_cumsum():
    s = cudf.Series([0, -1, -300, 23, 4, -3, 0, 0, 100])

    # cumsum test
    got = (s > 0).cumsum()
    expect = (s > 0).to_pandas().cumsum()

    assert_eq(expect, got)


def test_scan_boolean_cumprod():
    s = cudf.Series([0, -1, -300, 23, 4, -3, 0, 0, 100])

    # cumprod test
    got = (s > 0).cumprod()
    expect = (s > 0).to_pandas().cumprod()

    assert_eq(expect, got)

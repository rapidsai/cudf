from itertools import product

import numpy as np
import pandas as pd
import pytest

from cudf.core.dataframe import DataFrame, Series
from cudf.tests.utils import INTEGER_TYPES, NUMERIC_TYPES, assert_eq, gen_rand

params_sizes = [0, 1, 2, 5]


def _gen_params():
    for t, n in product(NUMERIC_TYPES, params_sizes):
        if (t == np.int8 or t == np.int16) and n > 20:
            # to keep data in range
            continue
        yield t, n


@pytest.mark.parametrize("dtype,nelem", list(_gen_params()))
def test_cumsum(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cumsum().to_array(), ps.cumsum(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = DataFrame()
    gdf["a"] = Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cumsum().to_array(), pdf.a.cumsum(), decimal=decimal
    )


def test_cumsum_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumsum(), ps.cumsum())

    for type_ in INTEGER_TYPES:
        gs = Series(data).astype(type_)
        got = gs.cumsum()
        expected = pd.Series(
            [1, 3, pd.NA, 7, 12], dtype=pd.Int64Dtype()
        )
        assert_eq(got, expected)


@pytest.mark.parametrize("dtype,nelem", list(_gen_params()))
def test_cummin(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cummin().to_array(), ps.cummin(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = DataFrame()
    gdf["a"] = Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cummin().to_array(), pdf.a.cummin(), decimal=decimal
    )


def test_cummin_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummin(), ps.cummin())

    for type_ in INTEGER_TYPES:
        gs = Series(data).astype(type_)
        expected = pd.Series(
            [1, 1, pd.NA, 1, 1]
        ).astype(gs.to_pandas().dtype)
        assert_eq(gs.cummin(), expected)


@pytest.mark.parametrize("dtype,nelem", list(_gen_params()))
def test_cummax(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cummax().to_array(), ps.cummax(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = DataFrame()
    gdf["a"] = Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cummax().to_array(), pdf.a.cummax(), decimal=decimal
    )


def test_cummax_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummax(), ps.cummax())

    for type_ in INTEGER_TYPES:
        gs = Series(data).astype(type_)
        expected = pd.Series(
            [1, 2, pd.NA, 4, 5]
        ).astype(gs.to_pandas().dtype)
        assert_eq(gs.cummax(), expected)


@pytest.mark.parametrize("dtype,nelem", list(_gen_params()))
def test_cumprod(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    decimal = 4 if dtype == np.float32 else 6

    # series
    gs = Series(data)
    ps = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gs.cumprod().to_array(), ps.cumprod(), decimal=decimal
    )

    # dataframe series (named series)
    gdf = DataFrame()
    gdf["a"] = Series(data)
    pdf = pd.DataFrame()
    pdf["a"] = pd.Series(data)
    np.testing.assert_array_almost_equal(
        gdf.a.cumprod().to_array(), pdf.a.cumprod(), decimal=decimal
    )


def test_cumprod_masked():
    data = [1, 2, None, 4, 5]
    float_types = ["float32", "float64"]

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumprod(), ps.cumprod())

    for type_ in INTEGER_TYPES:
        gs = Series(data).astype(type_)
        got = gs.cumprod()
        expected = pd.Series(
            [1, 2, pd.NA, 8, 40], dtype=pd.Int64Dtype()
        )
        assert_eq(got, expected)


def test_scan_boolean_cumsum():
    data = [0, -1, -300, 23, 4, -3, 0, 0, 100]

    p_s = Series(data)
    g_s = pd.Series(data)
    # cumsum test
    got = (g_s > 0).cumsum()
    expect = (p_s > 0).cumsum()

    # cumsum and cumprod don't work with BooleanArray
    assert_eq(expect, got, downcast=True)


def test_scan_boolean_cumprod():
    data = [0, -1, -300, 23, 4, -3, 0, 0, 100]
    g_s = Series(data)
    p_s = pd.Series(data)

    # cumprod test
    got = (g_s > 0).cumprod()
    expect = (p_s > 0).cumprod()

    assert_eq(expect, got, downcast=True)

from itertools import product

import pytest
import numpy as np
import pandas as pd

from cudf.dataframe.dataframe import Series, DataFrame
from cudf.tests.utils import gen_rand, assert_eq


params_dtype = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]

params_sizes = [1, 2, 13, 64, 100, 1000]


def _gen_params():
    for t, n in product(params_dtype, params_sizes):
        if (t == np.int8 or t == np.int16) and n > 20:
            # to keep data in range
            continue
        yield t, n


@pytest.mark.parametrize('dtype,nelem', list(_gen_params()))
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
    np.testing.assert_array_almost_equal(gs.cumsum(), ps.cumsum(),
                                         decimal=decimal)

    # dataframe series (named series)
    gdf = DataFrame()
    gdf['a'] = Series(data)
    pdf = pd.DataFrame()
    pdf['a'] = pd.Series(data)
    np.testing.assert_array_almost_equal(gdf.a.cumsum(), pdf.a.cumsum(),
                                         decimal=decimal)


def test_cumsum_masked():
    data = [1, 2, None, 4, 5]
    float_types = ['float32', 'float64']
    int_types = ['int8', 'int16', 'int32', 'int64']

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumsum(), ps.cumsum())

    for type_ in int_types:
        expected = pd.Series([1, 3, -1, 7, 12]).astype('int64')
        gs = Series(data).astype(type_)
        assert_eq(gs.cumsum(), expected)


@pytest.mark.parametrize('dtype,nelem', list(_gen_params()))
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
    np.testing.assert_array_almost_equal(gs.cummin(), ps.cummin(),
                                         decimal=decimal)

    # dataframe series (named series)
    gdf = DataFrame()
    gdf['a'] = Series(data)
    pdf = pd.DataFrame()
    pdf['a'] = pd.Series(data)
    np.testing.assert_array_almost_equal(gdf.a.cummin(), pdf.a.cummin(),
                                         decimal=decimal)


def test_cummin_masked():
    data = [1, 2, None, 4, 5]
    float_types = ['float32', 'float64']
    int_types = ['int8', 'int16', 'int32', 'int64']

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummin(), ps.cummin())

    for type_ in int_types:
        expected = pd.Series([1, 1, -1, 1, 1]).astype(type_)
        gs = Series(data).astype(type_)
        assert_eq(gs.cummin(), expected)


@pytest.mark.parametrize('dtype,nelem', list(_gen_params()))
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
    np.testing.assert_array_almost_equal(gs.cummax(), ps.cummax(),
                                         decimal=decimal)

    # dataframe series (named series)
    gdf = DataFrame()
    gdf['a'] = Series(data)
    pdf = pd.DataFrame()
    pdf['a'] = pd.Series(data)
    np.testing.assert_array_almost_equal(gdf.a.cummax(), pdf.a.cummax(),
                                         decimal=decimal)


def test_cummax_masked():
    data = [1, 2, None, 4, 5]
    float_types = ['float32', 'float64']
    int_types = ['int8', 'int16', 'int32', 'int64']

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cummax(), ps.cummax())

    for type_ in int_types:
        expected = pd.Series([1, 2, -1, 4, 5]).astype(type_)
        gs = Series(data).astype(type_)
        assert_eq(gs.cummax(), expected)


@pytest.mark.parametrize('dtype,nelem', list(_gen_params()))
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
    np.testing.assert_array_almost_equal(gs.cumprod(), ps.cumprod(),
                                         decimal=decimal)

    # dataframe series (named series)
    gdf = DataFrame()
    gdf['a'] = Series(data)
    pdf = pd.DataFrame()
    pdf['a'] = pd.Series(data)
    np.testing.assert_array_almost_equal(gdf.a.cumprod(), pdf.a.cumprod(),
                                         decimal=decimal)


def test_cumprod_masked():
    data = [1, 2, None, 4, 5]
    float_types = ['float32', 'float64']
    int_types = ['int8', 'int16', 'int32', 'int64']

    for type_ in float_types:
        gs = Series(data).astype(type_)
        ps = pd.Series(data).astype(type_)
        assert_eq(gs.cumprod(), ps.cumprod())

    for type_ in int_types:
        expected = pd.Series([1, 2, -1, 8, 40]).astype('int64')
        gs = Series(data).astype(type_)
        assert_eq(gs.cumprod(), expected)


def test_scan_boolean():
    s = Series([0, -1, -300, 23, 4, -3, 0, 0, 100])

    got = (s > 0).cumsum()
    expect = pd.Series([False, False, False, True, True,
                        True, True, True, True])

    assert_eq(expect, got)

    got = (s > 0).astype(np.int32).cumsum()
    expect = pd.Series([0, 0, 0, 1, 2, 2, 2, 2, 3])
    assert_eq(expect, got)

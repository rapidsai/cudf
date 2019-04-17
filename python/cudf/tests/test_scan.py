from __future__ import division

import pytest
import numpy as np
import pandas as pd
import cudf.bindings.reduce as cpp_reduce

from itertools import product
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.dataframe import Series, DataFrame
from cudf.tests import utils
from cudf.tests.utils import gen_rand, assert_eq

from librmm_cffi import librmm as rmm


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
def test_prefixsum(dtype, nelem):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    d_data = rmm.to_device(data)

    # Allocate output
    d_result = rmm.device_array(shape=data.size, dtype=dtype)

    # construct numerical columns
    in_col = NumericalColumn(data=Buffer(d_data), mask=None,
                             null_count=0, dtype=dtype)
    out_col = NumericalColumn(data=Buffer(d_result), mask=None,
                              null_count=0, dtype=dtype)

    # compute scan
    inclusive = True
    cpp_reduce.apply_scan(in_col, out_col, 'sum', inclusive=inclusive)

    expect = np.cumsum(d_data.copy_to_host())

    got = d_result.copy_to_host()
    if not inclusive:
        expect = expect[:-1]
        assert got[0] == 0
        got = got[1:]

    decimal = 4 if dtype == np.float32 else 6
    np.testing.assert_array_almost_equal(expect, got, decimal=decimal)


@pytest.mark.parametrize('dtype,nelem', list(_gen_params()))
def test_prefixsum_masked(dtype, nelem):
    if dtype == np.int8:
        data = gen_rand(dtype, nelem, low=-2, high=2)
    else:
        data = gen_rand(dtype, nelem)

    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)

    result_mask = utils.random_bitmask(nelem)

    d_data = rmm.to_device(data)
    d_mask = rmm.to_device(mask)

    d_result = rmm.device_array(d_data.size, dtype=d_data.dtype)
    d_result_mask = rmm.to_device(result_mask)

    # construct numerical columns
    in_col = NumericalColumn(data=Buffer(d_data), mask=Buffer(d_mask),
                             null_count=null_count, dtype=dtype)
    out_col = NumericalColumn(data=Buffer(d_result),
                              mask=Buffer(d_result_mask),
                              null_count=null_count, dtype=dtype)

    # compute scan
    inclusive = True
    cpp_reduce.apply_scan(in_col, out_col, 'sum', inclusive=inclusive)

    res_mask = np.asarray(bitmask, dtype=np.bool_)[:data.size]

    expect = np.cumsum(data[res_mask])
    got = d_result.copy_to_host()[res_mask]

    if not inclusive:
        expect = expect[:-1]
        assert got[0] == 0
        got = got[1:]

    decimal = 4 if dtype == np.float32 else 6
    np.testing.assert_array_almost_equal(expect, got, decimal=decimal)


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


def test_cummin():
    pass


def test_cummin_masked():
    pass


def test_cummax():
    pass


def test_cummax_masked():
    pass


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

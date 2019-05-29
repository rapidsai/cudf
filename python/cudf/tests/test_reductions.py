from __future__ import division, print_function

import pytest
import random
import numpy as np

from itertools import product
from cudf.dataframe import Series
from cudf.tests import utils
from cudf.tests.utils import gen_rand


params_dtype = [
    np.float64,
    np.float32,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
]

params_sizes = [1, 2, 3, 127, 128, 129, 200, 10000]

params = list(product(params_dtype, params_sizes))


@pytest.mark.parametrize('dtype,nelem', params)
def test_sum(dtype, nelem):
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.sum()
    expect = dtype(data.sum())

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


@pytest.mark.parametrize('dtype,nelem', params)
def test_product(dtype, nelem):
    if np.dtype(dtype).kind == 'i':
        data = np.ones(nelem, dtype=dtype)
        # Set at most 30 items to [0..2) to keep the value within 2^32
        for _ in range(30):
            data[random.randrange(nelem)] = random.random() * 2
    else:
        data = gen_rand(dtype, nelem)

    sr = Series(data)

    got = sr.product()
    expect = np.product(data)

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


accuracy_for_dtype = {
    np.float64: 6,
    np.float32: 5
}


@pytest.mark.parametrize('dtype,nelem', params)
def test_sum_of_squares(dtype, nelem):
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.sum_of_squares()
    expect = (data ** 2).sum()

    if np.dtype(dtype).kind == 'i':
        if 0 <= expect <= np.iinfo(dtype).max:
            np.testing.assert_array_almost_equal(expect, got)
        else:
            print('overflow, passing')
    else:
        np.testing.assert_approx_equal(expect, got,
                                       significant=accuracy_for_dtype[dtype])


@pytest.mark.parametrize('dtype,nelem', params)
def test_min(dtype, nelem):
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.min()
    expect = dtype(data.min())

    assert expect == got


@pytest.mark.parametrize('dtype,nelem', params)
def test_max(dtype, nelem):
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.max()
    expect = dtype(data.max())

    assert expect == got


@pytest.mark.parametrize('nelem', params_sizes)
def test_sum_masked(nelem):
    dtype = np.float64
    data = gen_rand(dtype, nelem)

    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)

    sr = Series.from_masked_array(data, mask, null_count)

    got = sr.sum()
    res_mask = np.asarray(bitmask, dtype=np.bool_)[:data.size]
    expect = data[res_mask].sum()

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


def test_sum_boolean():
    s = Series(np.arange(100000))
    got = (s > 1).sum(dtype=np.int32)
    expect = 99998

    assert expect == got

    got = (s > 1).sum(dtype=np.bool_)
    expect = True

    assert expect == got


def test_date_minmax():
    np_data = np.random.normal(size=10**3)
    gdf_data = Series(np_data)

    np_casted = np_data.astype('datetime64[ms]')
    gdf_casted = gdf_data.astype('datetime64[ms]')

    np_min = np_casted.min()
    gdf_min = gdf_casted.min()
    assert np_min == gdf_min

    np_max = np_casted.max()
    gdf_max = gdf_casted.max()
    assert np_max == gdf_max

from __future__ import division, print_function
import pytest
from itertools import product

import numpy as np

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

from libgdf_cffi.tests.utils import (new_column, unwrap_devary, get_dtype, gen_rand)


params_dtype = [
    np.int8,
    np.int32,
    np.int64,
]

params_sizes = [1, 2, 13, 64, 100, 1000]


def _gen_params():
    for t, n in product(params_dtype, params_sizes):
        if t == np.int8 and n > 20:
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
    d_result = rmm.device_array(d_data.size, dtype=d_data.dtype)

    col_data = new_column()
    gdf_dtype = get_dtype(dtype)
    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           gdf_dtype)

    col_result = new_column()
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem, gdf_dtype)

    inclusive = True
    libgdf.gdf_prefixsum_generic(col_data, col_result, inclusive)

    expect = np.cumsum(d_data.copy_to_host())
    got = d_result.copy_to_host()
    if not inclusive:
        expect = expect[:-1]
        assert got[0] == 0
        got = got[1:]

    np.testing.assert_array_equal(expect, got)

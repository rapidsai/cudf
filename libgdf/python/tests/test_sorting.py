from itertools import product

import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

from .utils import new_column, unwrap_devary, get_dtype, gen_rand


radixsort_args = list(product([2, 3, 10, 11, 100, 1000, 5000],
                              [True, False],
                              [np.int8, np.int32, np.int64,
                               np.float32, np.float64]))


@pytest.mark.parametrize('nelem,descending,dtype', radixsort_args)
def test_radixsort(nelem, descending, dtype):
    def expected_fn(key):
        # Use mergesort for stable sort
        # Negate the key for descending
        if issubclass(dtype, np.integer):
            def negate_values(v):
                return ~key
        else:
            # Note: this doesn't work on the smallest value of integer
            #       i.e. -((int8)-128) -> -128
            def negate_values(v):
                return -key

        sorted_idx = np.argsort(negate_values(key) if descending else key,
                                kind='mergesort')
        sorted_keys = key[sorted_idx]
        # Returns key, vals
        return sorted_keys, sorted_idx

    # Make data
    key = gen_rand(dtype, nelem)
    d_key = rmm.to_device(key)
    col_key = new_column()
    libgdf.gdf_column_view(col_key, unwrap_devary(d_key), ffi.NULL, nelem,
                           get_dtype(d_key.dtype))

    val = np.arange(nelem, dtype=np.int64)
    d_val = rmm.to_device(val)
    col_val = new_column()
    libgdf.gdf_column_view(col_val, unwrap_devary(d_val), ffi.NULL, nelem,
                           get_dtype(d_val.dtype))

    sizeof_key = d_key.dtype.itemsize
    sizeof_val = d_val.dtype.itemsize
    begin_bit = 0
    end_bit = sizeof_key * 8

    # Setup plan
    plan = libgdf.gdf_radixsort_plan(nelem, descending, begin_bit, end_bit)
    libgdf.gdf_radixsort_plan_setup(plan, sizeof_key, sizeof_val)
    # Sort
    libgdf.gdf_radixsort_generic(plan, col_key, col_val)
    # Cleanup
    libgdf.gdf_radixsort_plan_free(plan)

    # Check
    got_keys = d_key.copy_to_host()
    got_vals = d_val.copy_to_host()
    sorted_keys, sorted_vals = expected_fn(key)

    np.testing.assert_array_equal(sorted_keys, got_keys)
    np.testing.assert_array_equal(sorted_vals, got_vals)

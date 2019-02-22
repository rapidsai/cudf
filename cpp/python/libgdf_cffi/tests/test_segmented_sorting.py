from itertools import product
import random

import pytest

import numpy as np

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

from libgdf_cffi.tests.utils import new_column, unwrap_devary, get_dtype, gen_rand


def segsort_args():
    nelems = [2, 3, 10, 100, 1000]
    descendings = [True, False]
    dtypes = [np.int8, np.int32, np.int64, np.float32, np.float64]
    for nelem, descending, dtype in product(nelems, descendings, dtypes):
        for numseg in range(1, 4):
            if nelem // numseg > 0:
                yield nelem, numseg, descending, dtype


@pytest.mark.parametrize('nelem,num_segments,descending,dtype',
                         list(segsort_args()))
def test_segradixsort(nelem, num_segments, descending, dtype):
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

    def make_segments(n, k):
        sampled = random.sample(list(range(n)), k)
        return list(sorted(sampled))

    begin_offsets = np.asarray(make_segments(nelem, num_segments),
                               dtype=np.uint32)
    end_offsets = np.asarray(begin_offsets.tolist()[1:] + [nelem],
                             dtype=begin_offsets.dtype)

    # Make data
    key = gen_rand(dtype, nelem)
    d_key = rmm.to_device(key)
    col_key = new_column()
    libgdf.gdf_column_view(col_key, unwrap_devary(d_key), ffi.NULL,
                           nelem, get_dtype(d_key.dtype))

    val = np.arange(nelem, dtype=np.int64)
    d_val = rmm.to_device(val)
    col_val = new_column()
    libgdf.gdf_column_view(col_val, unwrap_devary(d_val), ffi.NULL, nelem,
                           get_dtype(d_val.dtype))

    d_begin_offsets = rmm.to_device(begin_offsets)
    d_end_offsets = rmm.to_device(end_offsets)

    sizeof_key = d_key.dtype.itemsize
    sizeof_val = d_val.dtype.itemsize
    begin_bit = 0
    end_bit = sizeof_key * 8

    # Setup plan
    plan = libgdf.gdf_segmented_radixsort_plan(nelem, descending,
                                               begin_bit, end_bit)
    libgdf.gdf_segmented_radixsort_plan_setup(plan, sizeof_key, sizeof_val)

    # Sort
    libgdf.gdf_segmented_radixsort_generic(plan, col_key, col_val,
                                           num_segments,
                                           unwrap_devary(d_begin_offsets),
                                           unwrap_devary(d_end_offsets))

    # Cleanup
    libgdf.gdf_segmented_radixsort_plan_free(plan)

    # Check
    got_keys = d_key.copy_to_host()
    got_vals = d_val.copy_to_host()

    # Check a segment at a time
    for s, e in zip(begin_offsets, end_offsets):
        segment = key[s:e]
        exp_keys, exp_vals = expected_fn(segment)
        exp_vals += s

        np.testing.assert_array_equal(exp_keys, got_keys[s:e])
        np.testing.assert_array_equal(exp_vals, got_vals[s:e])

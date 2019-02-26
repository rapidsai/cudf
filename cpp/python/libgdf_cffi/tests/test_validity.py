import pytest
from itertools import product

import numpy as np

from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm

from libgdf_cffi.tests.utils import new_column, unwrap_devary, get_dtype, gen_rand
from libgdf_cffi.tests.utils import buffer_as_bits


_dtypes = [np.int32]
_nelems = [1, 2, 7, 8, 9, 32, 128]


@pytest.mark.parametrize('dtype,nelem', list(product(_dtypes, _nelems)))
def test_validity_add(dtype, nelem):
    expect_fn = np.add
    test_fn = libgdf.gdf_add_generic

    # data
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)
    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array_like(d_lhs)

    # valids
    h_lhs_valids = gen_rand(np.int8, (nelem + 8 - 1) // 8)
    h_rhs_valids = gen_rand(np.int8, (nelem + 8 - 1) // 8)

    d_lhs_valids = rmm.to_device(h_lhs_valids)
    d_rhs_valids = rmm.to_device(h_rhs_valids)
    d_result_valids = rmm.device_array_like(d_lhs_valids)

    # columns
    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    libgdf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           unwrap_devary(d_lhs_valids), nelem, gdf_dtype)
    libgdf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           unwrap_devary(d_rhs_valids), nelem, gdf_dtype)
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result),
                           unwrap_devary(d_result_valids), nelem, gdf_dtype)

    libgdf.gdf_validity_and(col_lhs, col_rhs, col_result)

    expect = expect_fn(h_lhs, h_rhs)
    test_fn(col_lhs, col_rhs, col_result)
    got = d_result.copy_to_host()

    # Ensure validity mask is matching
    expect_valids = h_lhs_valids & h_rhs_valids
    got_valids = d_result_valids.copy_to_host()

    np.testing.assert_array_equal(expect_valids, got_valids)

    # Masked data
    mask = buffer_as_bits(expect_valids.data)[:expect.size]
    expect_masked = expect[mask]
    got_masked = got[mask]

    print('expect')
    print(expect_masked)
    print('got')
    print(got_masked)

    np.testing.assert_array_equal(expect_masked, got_masked)

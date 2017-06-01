
import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf

from .utils import new_column, unwrap_devary, get_dtype, gen_rand, fix_zeros


def arith_op_test(dtype, ulp, expect_fn, test_fn, nelem=128,
                  non_zero_rhs=False):
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)
    if non_zero_rhs:
        fix_zeros(h_rhs)
    d_lhs = cuda.to_device(h_lhs)
    d_rhs = cuda.to_device(h_rhs)
    d_result = cuda.device_array_like(d_lhs)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    libgdf.gdf_column_view(col_lhs, unwrap_devary(d_lhs), ffi.NULL, nelem,
                           gdf_dtype)
    libgdf.gdf_column_view(col_rhs, unwrap_devary(d_rhs), ffi.NULL, nelem,
                           gdf_dtype)
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL, nelem,
                           gdf_dtype)

    expect = expect_fn(h_lhs, h_rhs)
    test_fn(col_lhs, col_rhs, col_result)
    got = d_result.copy_to_host()
    print('got')
    print(got)
    print('expect')
    print(expect)
    np.testing.assert_array_max_ulp(expect, got, maxulp=ulp)


def logical_op_test(dtype, expect_fn, test_fn, nelem=128):
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)
    d_lhs = cuda.to_device(h_lhs)
    d_rhs = cuda.to_device(h_rhs)
    d_result = cuda.device_array(d_lhs.size, dtype=np.bool)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    libgdf.gdf_column_view(col_lhs, unwrap_devary(d_lhs), ffi.NULL, nelem,
                           gdf_dtype)
    libgdf.gdf_column_view(col_rhs, unwrap_devary(d_rhs), ffi.NULL, nelem,
                           gdf_dtype)
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL, nelem,
                           libgdf.GDF_INT8)

    expect = expect_fn(h_lhs, h_rhs)
    test_fn(col_lhs, col_rhs, col_result)

    got = d_result.copy_to_host()
    print(expect, got)
    np.testing.assert_equal(expect, got)


# arith

params_arith_types = [
    (np.float64, 0),
    (np.float32, 0),
    (np.int32, 0),
    (np.int64, 0),
]

@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_add(dtype, ulp):
    arith_op_test(dtype, ulp, np.add, libgdf.gdf_add_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_sub(dtype, ulp):
    arith_op_test(dtype, ulp, np.subtract, libgdf.gdf_sub_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_mul(dtype, ulp):
    arith_op_test(dtype, ulp, np.multiply, libgdf.gdf_mul_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_floordiv(dtype, ulp):
    arith_op_test(dtype, ulp, np.floor_divide, libgdf.gdf_floordiv_generic,
                   non_zero_rhs=True)


@pytest.mark.parametrize('dtype,ulp', [
    (np.float64, 0),
    (np.float32, 0),
])
def test_div(dtype, ulp):
    arith_op_test(dtype, ulp, np.divide, libgdf.gdf_div_generic,
                  non_zero_rhs=True)

# logical

params_logical_types = [
    np.float64,
    np.float32,
    np.int32,
    np.int64,
]

@pytest.mark.parametrize('dtype', params_logical_types)
def test_gt(dtype):
    logical_op_test(dtype, np.greater, libgdf.gdf_gt_generic)

@pytest.mark.parametrize('dtype', params_logical_types)
def test_ge(dtype):
    logical_op_test(dtype, np.greater_equal, libgdf.gdf_ge_generic)

@pytest.mark.parametrize('dtype', params_logical_types)
def test_lt(dtype):
    logical_op_test(dtype, np.less, libgdf.gdf_lt_generic)

@pytest.mark.parametrize('dtype', params_logical_types)
def test_le(dtype):
    logical_op_test(dtype, np.less_equal, libgdf.gdf_le_generic)


@pytest.mark.parametrize('dtype', params_logical_types)
def test_eq(dtype):
    logical_op_test(dtype, np.equal, libgdf.gdf_eq_generic)

@pytest.mark.parametrize('dtype', params_logical_types)
def test_ne(dtype):
    logical_op_test(dtype, np.not_equal, libgdf.gdf_ne_generic)


if __name__ == '__main__':
    test_add()


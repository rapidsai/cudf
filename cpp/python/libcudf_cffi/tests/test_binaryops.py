import pytest

import numpy as np

from libcudf_cffi import ffi, libcudf, GDFError
from librmm_cffi import librmm as rmm

from libcudf_cffi.tests.utils import new_column, unwrap_devary, get_dtype, gen_rand, fix_zeros


def arith_op_test(dtype, ulp, expect_fn, test_fn, nelem=128,
                  non_zero_rhs=False):
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)
    if non_zero_rhs:
        fix_zeros(h_rhs)
    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array_like(d_lhs)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    libcudf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_result, unwrap_devary(d_result),
                           ffi.NULL, nelem, gdf_dtype)

    expect = expect_fn(h_lhs, h_rhs)
    test_fn(col_lhs, col_rhs, col_result)
    got = d_result.copy_to_host()
    print('got')
    print(got)
    print('expect')
    print(expect)
    np.testing.assert_array_max_ulp(expect, got, maxulp=ulp)


def logical_op_test(dtype, expect_fn, test_fn, nelem=128, gdf_dtype=None):
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)
    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array(d_lhs.size, dtype=np.bool)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype) if gdf_dtype is None else gdf_dtype

    libcudf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_result, unwrap_devary(d_result),
                           ffi.NULL, nelem, libcudf.GDF_INT8)

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
    arith_op_test(dtype, ulp, np.add, libcudf.gdf_add_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_sub(dtype, ulp):
    arith_op_test(dtype, ulp, np.subtract, libcudf.gdf_sub_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_mul(dtype, ulp):
    arith_op_test(dtype, ulp, np.multiply, libcudf.gdf_mul_generic)


@pytest.mark.parametrize('dtype,ulp', params_arith_types)
def test_floordiv(dtype, ulp):
    arith_op_test(dtype, ulp, np.floor_divide, libcudf.gdf_floordiv_generic,
                  non_zero_rhs=True)


@pytest.mark.parametrize('dtype,ulp', [
    (np.float64, 0),
    (np.float32, 0),
])
def test_div(dtype, ulp):
    arith_op_test(dtype, ulp, np.divide, libcudf.gdf_div_generic,
                  non_zero_rhs=True)


# logical

params_logical_types = [
    (np.float64, None),
    (np.float32, None),
    (np.int32, None),
    (np.int64, None),
    (np.bool, None),
    (np.int32, libcudf.GDF_DATE32),
    (np.int64, libcudf.GDF_DATE64),
    (np.int64, libcudf.GDF_TIMESTAMP),
]


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_gt(dtype, gdf_dtype):
    logical_op_test(dtype, np.greater, libcudf.gdf_gt_generic,
                    gdf_dtype=gdf_dtype)


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_ge(dtype, gdf_dtype):
    logical_op_test(dtype, np.greater_equal, libcudf.gdf_ge_generic,
                    gdf_dtype=gdf_dtype)


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_lt(dtype, gdf_dtype):
    logical_op_test(dtype, np.less, libcudf.gdf_lt_generic,
                    gdf_dtype=gdf_dtype)


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_le(dtype, gdf_dtype):
    logical_op_test(dtype, np.less_equal, libcudf.gdf_le_generic,
                    gdf_dtype=gdf_dtype)


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_eq(dtype, gdf_dtype):
    logical_op_test(dtype, np.equal, libcudf.gdf_eq_generic,
                    gdf_dtype=gdf_dtype)


@pytest.mark.parametrize('dtype, gdf_dtype', params_logical_types)
def test_ne(dtype, gdf_dtype):
    logical_op_test(dtype, np.not_equal, libcudf.gdf_ne_generic,
                    gdf_dtype=gdf_dtype)


# bitwise

params_bitwise_types = [
    np.int32,
    np.int64,
    np.int8,
]


def bitwise_op_test(dtype, expect_fn, test_fn, nelem=128):
    h_lhs = gen_rand(dtype, nelem)
    h_rhs = gen_rand(dtype, nelem)

    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array_like(d_lhs)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    libcudf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           ffi.NULL, nelem, gdf_dtype)
    libcudf.gdf_column_view(col_result, unwrap_devary(d_result),
                           ffi.NULL, nelem, gdf_dtype)

    expect = expect_fn(h_lhs, h_rhs)
    test_fn(col_lhs, col_rhs, col_result)
    got = d_result.copy_to_host()
    print('got')
    print(got)
    print('expect')
    print(expect)
    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize('dtype', params_bitwise_types)
def test_bitwise_and(dtype):
    bitwise_op_test(dtype, np.bitwise_and, libcudf.gdf_bitwise_and_generic)


@pytest.mark.parametrize('dtype', params_bitwise_types)
def test_bitwise_or(dtype):
    bitwise_op_test(dtype, np.bitwise_or, libcudf.gdf_bitwise_or_generic)


@pytest.mark.parametrize('dtype', params_bitwise_types)
def test_bitwise_xor(dtype):
    bitwise_op_test(dtype, np.bitwise_xor, libcudf.gdf_bitwise_xor_generic)


def test_lhs_rhs_dtype_mismatch():
    lhs_dtype = np.int32
    rhs_dtype = np.float32
    nelem = 5
    h_lhs = np.arange(nelem, dtype=lhs_dtype)
    h_rhs = np.arange(nelem, dtype=rhs_dtype)

    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array_like(d_lhs)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()

    libcudf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           ffi.NULL, nelem, get_dtype(lhs_dtype))
    libcudf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           ffi.NULL, nelem, get_dtype(rhs_dtype))
    libcudf.gdf_column_view(col_result, unwrap_devary(d_result),
                           ffi.NULL, nelem, get_dtype(lhs_dtype))

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_add_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_eq_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_bitwise_and_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")


def test_output_dtype_mismatch():
    lhs_dtype = np.int32
    rhs_dtype = np.int32
    nelem = 5
    h_lhs = np.arange(nelem, dtype=lhs_dtype)
    h_rhs = np.arange(nelem, dtype=rhs_dtype)

    d_lhs = rmm.to_device(h_lhs)
    d_rhs = rmm.to_device(h_rhs)
    d_result = rmm.device_array(d_lhs.size, dtype=np.float32)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()

    libcudf.gdf_column_view(col_lhs, unwrap_devary(d_lhs),
                           ffi.NULL, nelem, get_dtype(lhs_dtype))
    libcudf.gdf_column_view(col_rhs, unwrap_devary(d_rhs),
                           ffi.NULL, nelem, get_dtype(rhs_dtype))
    libcudf.gdf_column_view(col_result, unwrap_devary(d_result),
                           ffi.NULL, nelem, get_dtype(d_result.dtype))

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_add_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_eq_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")

    with pytest.raises(GDFError) as raises:
        libcudf.gdf_bitwise_and_generic(col_lhs, col_rhs, col_result)
    raises.match("GDF_UNSUPPORTED_DTYPE")


if __name__ == '__main__':
    test_add()

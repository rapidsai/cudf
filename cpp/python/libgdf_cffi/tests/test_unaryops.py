
import pytest

import itertools

import numpy as np

from libgdf_cffi import ffi, libgdf, GDFError
from librmm_cffi import librmm as rmm

from libgdf_cffi.tests.utils import new_column, unwrap_devary, get_dtype, gen_rand


def math_op_test(dtype, ulp, expect_fn, test_fn, nelem=128, scale=1,
                 positive_only=False):
    randvals = gen_rand(dtype, nelem, positive_only=positive_only)
    h_data = (randvals * scale).astype(dtype)
    d_data = rmm.to_device(h_data)
    d_result = rmm.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()
    gdf_dtype = get_dtype(dtype)

    # data column
    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           gdf_dtype)
    # result column
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem, gdf_dtype)

    expect = expect_fn(h_data)
    libgdf.gdf_unary_math(col_data, col_result, test_fn)

    got = d_result.copy_to_host()

    print('got')
    print(got)
    print('expect')
    print(expect)
    np.testing.assert_array_max_ulp(expect, got, maxulp=ulp)


def cast_op_test(dtype, to_dtype, nelem=128):
    h_data = gen_rand(dtype, nelem).astype(dtype)
    d_data = rmm.to_device(h_data)
    d_result = rmm.device_array(d_data.size, dtype=to_dtype)

    assert d_data.dtype == dtype
    assert d_result.dtype == to_dtype

    col_data = new_column()
    col_result = new_column()

    # data column
    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           get_dtype(dtype))
    # result column
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem, get_dtype(to_dtype))

    expect = h_data.astype(to_dtype)
    libgdf.gdf_cast(col_data, col_result)

    got = d_result.copy_to_host()

    print('got')
    print(got)
    print('expect')
    print(expect)
    np.testing.assert_equal(expect, got)


def test_col_mismatch_error():
    nelem = 128
    h_data = np.random.random(nelem).astype(np.float32)
    d_data = rmm.to_device(h_data)
    d_result = rmm.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)

    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem + 10, libgdf.GDF_FLOAT32)

    with pytest.raises(GDFError) as excinfo:
        libgdf.gdf_unary_math(col_data, col_result, libgdf.GDF_SIN)

    assert 'GDF_COLUMN_SIZE_MISMATCH' == str(excinfo.value)


def test_unsupported_dtype_error():
    nelem = 128
    h_data = np.random.random(nelem).astype(np.float32)
    d_data = rmm.to_device(h_data)
    d_result = rmm.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_DATE32)

    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem + 10, libgdf.GDF_FLOAT32)

    with pytest.raises(GDFError) as excinfo:
        libgdf.gdf_unary_math(col_data, col_result, libgdf.GDF_SIN)

    assert 'GDF_UNSUPPORTED_DTYPE' == str(excinfo.value)


params_real_types = [
    (np.float64, 3),
    (np.float32, 4),
]


# trig

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_sin(dtype, ulp):
    math_op_test(dtype, ulp, np.sin, libgdf.GDF_SIN)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_cos(dtype, ulp):
    math_op_test(dtype, ulp, np.cos, libgdf.GDF_COS)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_tan(dtype, ulp):
    math_op_test(dtype, ulp, np.tan, libgdf.GDF_TAN)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_asin(dtype, ulp):
    math_op_test(dtype, ulp, np.arcsin, libgdf.GDF_ARCSIN)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_acos(dtype, ulp):
    math_op_test(dtype, ulp, np.arccos, libgdf.GDF_ARCCOS)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_atan(dtype, ulp):
    math_op_test(dtype, ulp, np.arctan, libgdf.GDF_ARCTAN)


# exponential

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_exp(dtype, ulp):
    math_op_test(dtype, ulp, np.exp, libgdf.GDF_EXP)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_log(dtype, ulp):
    math_op_test(dtype, ulp, np.log, libgdf.GDF_LOG,
                 positive_only=True)


# power

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_sqrt(dtype, ulp):
    math_op_test(dtype, ulp, np.sqrt, libgdf.GDF_SQRT,
                 positive_only=True)


# rounding

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_ceil(dtype, ulp):
    math_op_test(dtype, ulp, np.ceil, libgdf.GDF_CEIL,
                 scale=100)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_floor(dtype, ulp):
    math_op_test(dtype, ulp, np.floor, libgdf.GDF_FLOOR,
                 scale=100)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_abs(dtype, ulp):
    math_op_test(dtype, ulp, np.abs, libgdf.GDF_ABS,
                 scale=100)


# casting

_cast_dtypes = [
    np.float64,
    np.float32,
    np.int64,
    np.int32,
    np.int8,
]

_param_cast_pairs = list(itertools.product(_cast_dtypes, _cast_dtypes))


@pytest.mark.parametrize('dtype,to_dtype', _param_cast_pairs)
def test_cast(dtype, to_dtype):
    cast_op_test(dtype, to_dtype)


if __name__ == '__main__':
    test_col_mismatch_error()

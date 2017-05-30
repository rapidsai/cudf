
import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf, GDFError

from .utils import new_column, unwrap_devary, get_dtype


def unary_op_test(dtype, ulp, expect_fn, test_fn, nelem=128, scale=1):
    h_data = (np.random.random(nelem) * scale).astype(dtype)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(d_data)

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
    test_fn(col_data, col_result)

    got = d_result.copy_to_host()
    np.testing.assert_array_max_ulp(expect, got, maxulp=ulp)


def test_col_mismatch_error():
    nelem = 128
    h_data = np.random.random(nelem).astype(np.float32)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)

    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem + 10, libgdf.GDF_FLOAT32)

    with pytest.raises(GDFError) as excinfo:
        libgdf.gdf_sin_generic(col_data, col_result)

    assert 'GDF_COLUMN_SIZE_MISMATCH' == str(excinfo.value)


def test_unsupported_dtype_error():
    nelem = 128
    h_data = np.random.random(nelem).astype(np.float32)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_INT32)

    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem + 10, libgdf.GDF_FLOAT32)

    with pytest.raises(GDFError) as excinfo:
        libgdf.gdf_sin_generic(col_data, col_result)

    assert 'GDF_UNSUPPORTED_DTYPE' == str(excinfo.value)


params_real_types = [
    (np.float64, 2),
    (np.float32, 3),
]


# trig

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_sin(dtype, ulp):
    unary_op_test(dtype, ulp, np.sin, libgdf.gdf_sin_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_cos(dtype, ulp):
    unary_op_test(dtype, ulp, np.cos, libgdf.gdf_cos_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_tan(dtype, ulp):
    unary_op_test(dtype, ulp, np.tan, libgdf.gdf_tan_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_asin(dtype, ulp):
    unary_op_test(dtype, ulp, np.arcsin, libgdf.gdf_asin_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_acos(dtype, ulp):
    unary_op_test(dtype, ulp, np.arccos, libgdf.gdf_acos_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_atan(dtype, ulp):
    unary_op_test(dtype, ulp, np.arctan, libgdf.gdf_atan_generic)


# exponential

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_exp(dtype, ulp):
    unary_op_test(dtype, ulp, np.exp, libgdf.gdf_exp_generic)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_log(dtype, ulp):
    unary_op_test(dtype, ulp, np.log, libgdf.gdf_log_generic)


# power

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_sqrt(dtype, ulp):
    unary_op_test(dtype, ulp, np.sqrt, libgdf.gdf_sqrt_generic)


# misc

@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_ceil(dtype, ulp):
    unary_op_test(dtype, ulp, np.ceil, libgdf.gdf_ceil_generic,
                  scale=100)


@pytest.mark.parametrize('dtype,ulp', params_real_types)
def test_floor(dtype, ulp):
    unary_op_test(dtype, ulp, np.floor, libgdf.gdf_floor_generic,
                  scale=100)


if __name__ == '__main__':
    test_col_mismatch_error()


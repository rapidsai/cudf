
import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf, GDFError

from .utils import new_column, unwrap_devary


def test_col_mismatch_error():
    np.random.seed(0)
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


def test_sin():
    np.random.seed(0)
    nelem = 128
    h_data = np.random.random(nelem).astype(np.float32)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)

    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL,
                           nelem, libgdf.GDF_FLOAT32)

    libgdf.gdf_sin_generic(col_data, col_result)

    h_result = d_result.copy_to_host()

    np.testing.assert_allclose(np.sin(h_data), h_result, rtol=1e-5)


if __name__ == '__main__':
    test_col_mismatch_error()


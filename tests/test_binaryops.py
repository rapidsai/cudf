
import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf

from .utils import new_column, unwrap_devary


def test_add():
    np.random.seed(0)
    nelem = 128
    h_lhs = np.random.random(nelem).astype(np.float32)
    h_rhs = np.random.random(nelem).astype(np.float32)
    d_lhs = cuda.to_device(h_lhs)
    d_rhs = cuda.to_device(h_rhs)
    d_result = cuda.device_array_like(d_lhs)

    col_lhs = new_column()
    col_rhs = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_lhs, unwrap_devary(d_lhs), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)
    libgdf.gdf_column_view(col_rhs, unwrap_devary(d_rhs), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)

    libgdf.gdf_add_f32(col_lhs, col_rhs, col_result)

    h_result = d_result.copy_to_host()

    np.testing.assert_allclose(h_lhs + h_rhs, h_result, rtol=1e-5)


if __name__ == '__main__':
    test_add()



import numpy as np
from numba import cuda

from libgdf_cffi import ffi

libgdf = ffi.dlopen('libgdf.so')


def new_column():
    return ffi.new('gdf_column*')


def unwrap_devary(devary):
    return ffi.cast('void*', devary.device_ctypes_pointer.value)


def test_sin():
    np.random.seed(0)
    nelem = 128
    d_data = cuda.to_device(np.random.random(nelem).astype(np.float32))
    d_result = cuda.device_array_like(d_data)

    col_data = new_column()
    col_result = new_column()

    libgdf.gdf_column_view(col_data, unwrap_devary(d_data), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)
    libgdf.gdf_column_view(col_result, unwrap_devary(d_result), ffi.NULL, nelem,
                           libgdf.GDF_FLOAT32)

    libgdf.gdf_sin_generic(col_data, col_result)

    np.testing.assert_allclose(np.sin(d_data.copy_to_host()),
                               d_result.copy_to_host(),
                               rtol=1e-5)


if __name__ == '__main__':
    test_sin()


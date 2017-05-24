
import numpy as np

from libgdf_cffi import ffi, libgdf


def new_column():
    return ffi.new('gdf_column*')


def unwrap_devary(devary):
    return ffi.cast('void*', devary.device_ctypes_pointer.value)


def get_dtype(dtype):
    return {
        np.float32: libgdf.GDF_FLOAT32,
        np.float64: libgdf.GDF_FLOAT64,
    }[np.dtype(dtype).type]

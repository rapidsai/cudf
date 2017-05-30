
import numpy as np

from libgdf_cffi import ffi, libgdf


def new_column():
    return ffi.new('gdf_column*')


def unwrap_devary(devary):
    return ffi.cast('void*', devary.device_ctypes_pointer.value)


def get_dtype(dtype):
    return {
        np.float64: libgdf.GDF_FLOAT64,
        np.float32: libgdf.GDF_FLOAT32,
        np.int64:   libgdf.GDF_INT64,
        np.int32:   libgdf.GDF_INT32,
        np.int8:    libgdf.GDF_INT8,
    }[np.dtype(dtype).type]


def gen_rand(dtype, size):
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        return np.random.random(size).astype(dtype)
    elif dtype.kind == 'i':
        return np.random.random_integers(low=-10000, high=10000, size=size).astype(dtype)
    raise NotImplementedError('dtype.kind={}'.format(dtype.kind))


def fix_zeros(arr, val=1):
    arr[arr == 0] = val

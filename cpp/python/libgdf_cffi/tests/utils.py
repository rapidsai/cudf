
import numpy as np

from libgdf_cffi import ffi, libgdf


def new_column():
    return ffi.new('gdf_column*')


def new_context():
    return ffi.new('gdf_context*')


def unwrap_devary(devary):
    return ffi.cast('void*', devary.device_ctypes_pointer.value)


def get_dtype(dtype):
    return {
        np.float64: libgdf.GDF_FLOAT64,
        np.float32: libgdf.GDF_FLOAT32,
        np.int64:   libgdf.GDF_INT64,
        np.int32:   libgdf.GDF_INT32,
        np.int16:   libgdf.GDF_INT16,
        np.int8:    libgdf.GDF_INT8,
        np.bool_:   libgdf.GDF_INT8,
    }[np.dtype(dtype).type]


def seed_rand():
    # A constant seed for deterministic testing
    np.random.seed(0xabcdef)


def gen_rand(dtype, size, **kwargs):
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        res = np.random.random(size=size).astype(dtype)
        if kwargs.get('positive_only', False):
            return res
        else:
            return (res * 2 - 1)
    elif dtype == np.int8:
        low = kwargs.get('low', -32)
        high = kwargs.get('high', 32)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == 'i':
        low = kwargs.get('low', -10000)
        high = kwargs.get('high', 10000)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == 'b':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 1)
        return np.random.randint(low=low, high=high, size=size).astype(np.bool)
    raise NotImplementedError('dtype.kind={}'.format(dtype.kind))


def fix_zeros(arr, val=1):
    arr[arr == 0] = val


def buffer_as_bits(data):
    def fix_binary(x):
        x = x[2:]
        diff = 8 - len(x)
        return ('0' * diff + x)[::-1]

    binaries = ''.join(fix_binary(bin(x)) for x in bytearray(data))
    return list(map(lambda x: x == '1', binaries))

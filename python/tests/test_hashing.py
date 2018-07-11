from __future__ import print_function
import ctypes
from contextlib import contextmanager

import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf

from .utils import new_column, unwrap_devary, get_dtype


@contextmanager
def _make_hash_input(hash_input, ncols):
    ci = []

    for i in range(ncols):
        d_input = cuda.to_device(hash_input[i])
        col_input = new_column()
        libgdf.gdf_column_view(col_input, unwrap_devary(d_input), ffi.NULL,
                            hash_input[i].size, get_dtype(d_input.dtype))
        ci.append(col_input)

    yield ci

def _call_hash_multi(api, ncols, col_input, magic):
    hash_result_ptr = ffi.new("gdf_join_result_type**", None)

    api(ncols, col_input, magic, hash_result_ptr)
    hash_result = hash_result_ptr[0]
    print('hash_result', hash_result)

    dataptr = libgdf.gdf_join_result_data(hash_result)
    print(dataptr)
    datasize = libgdf.gdf_join_result_size(hash_result)
    print(datasize)

    addr = ctypes.c_uint64(int(ffi.cast("uintptr_t", dataptr)))
    print(hex(addr.value))
    memptr = cuda.driver.MemoryPointer(context=cuda.current_context(),
                                       pointer=addr, size=4 * datasize)
    print(memptr)
    ary = cuda.devicearray.DeviceNDArray(shape=(datasize,), strides=(4,),
                                         dtype=np.dtype(np.int32),
                                         gpu_data=memptr)

    hashed_idx = ary.copy_to_host()
    print(hashed_idx)

    libgdf.gdf_join_result_free(hash_result)
    return hashed_idx


multi_params_dtypes = [np.int32, np.int64]


@pytest.mark.parametrize('dtype', multi_params_dtypes)
def test_hashing(dtype):
    # Make data
    nrows = 8
    hash_input = []
    hash_input1 = np.array(np.random.randint(0, 28, nrows), dtype=np.int32)
    hash_input1[-1] = hash_input1[0]
    hash_input.append(hash_input1)
    hash_input2 = np.array(np.random.randint(0, 28, nrows), dtype=np.int64)
    hash_input2[-1] = hash_input2[0]
    hash_input.append(hash_input2)
    hash_input3 = np.array(np.random.randint(0, 28, nrows), dtype=np.float32)
    hash_input3[-1] = hash_input3[0]
    hash_input.append(hash_input3)
    hash_input4 = np.array(np.random.randint(0, 28, nrows), dtype=np.float64)
    hash_input4[-1] = hash_input4[0]
    hash_input.append(hash_input4) 

    ncols = len(hash_input)
    magic = 0
    
    with _make_hash_input(hash_input, ncols) as (col_input):
        # Join
        hashed_column = _call_hash_multi(libgdf.gdf_hash, ncols, col_input, magic)

    # Check if first and last row are equal
    assert tuple([hashed_column[0]]) == tuple([hashed_column[-1]])
            
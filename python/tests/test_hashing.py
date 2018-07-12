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

def _call_hash_multi(api, ncols, col_input, magic, nrows):
    out_ary = np.zeros(nrows,dtype=np.int32)
    d_out = cuda.to_device(out_ary)
    col_out = new_column()
    libgdf.gdf_column_view(col_out, unwrap_devary(d_out), ffi.NULL,
                           out_ary.size, get_dtype(d_out.dtype))

    api(ncols, col_input, magic, col_out)

    dataptr = col_out.data
    print(dataptr)
    datasize = col_out.size
    print(datasize)

    addr = ctypes.c_uint64(int(ffi.cast("uintptr_t", dataptr)))
    print(hex(addr.value))
    memptr = cuda.driver.MemoryPointer(context=cuda.current_context(),
                                       pointer=addr, size=4 * datasize)
    print(memptr)
    ary = cuda.devicearray.DeviceNDArray(shape=(datasize,), strides=(4,),
                                         dtype=np.dtype(np.int32),
                                         gpu_data=memptr)

    hashed_result = ary.copy_to_host()
    print(hashed_result)

    return hashed_result


def test_hashing():
    # Make data
    nrows = 8
    hash_input = []
    hash_input1 = np.array(np.random.randint(0, 28, nrows), dtype=np.int8)
    hash_input1[-1] = hash_input1[0]
    hash_input.append(hash_input1)

    hash_input2 = np.array(np.random.randint(0, 28, nrows), dtype=np.int16)
    hash_input2[-1] = hash_input2[0]
    hash_input.append(hash_input2)

    hash_input3 = np.array(np.random.randint(0, 28, nrows), dtype=np.int32)
    hash_input3[-1] = hash_input3[0]
    hash_input.append(hash_input3)

    hash_input4 = np.array(np.random.randint(0, 28, nrows), dtype=np.int64)
    hash_input4[-1] = hash_input4[0]
    hash_input.append(hash_input4)

    hash_input5 = np.array(np.random.randint(0, 28, nrows), dtype=np.float32)
    hash_input5[-1] = hash_input5[0]
    hash_input.append(hash_input5)
    
    # hash_input6 = np.array(np.random.randint(0, 28, nrows), dtype=np.float64)
    # hash_input6[-1] = hash_input6[0]
    # hash_input.append(hash_input6)

    # pytest on all test files fails if columns>5, but works well if called seperately like
    # pytest --cache-clear -vvs ~/libgdf/python/tests/test_hashing.py

    ncols = len(hash_input)
    magic = 0
    
    with _make_hash_input(hash_input, ncols) as (col_input):
        # Hash
        hashed_column = _call_hash_multi(libgdf.gdf_hash, ncols, col_input, magic, nrows)

    # Check if first and last row are equal
    assert tuple([hashed_column[0]]) == tuple([hashed_column[-1]])
    
import pytest
import functools
from itertools import product

import numpy as np
from numba import cuda

from libgdf_cffi import libgdf
from librmm_cffi import ffi, librmm

from .utils import new_column, unwrap_devary, get_dtype, gen_rand, fix_zeros
from .utils import buffer_as_bits


_dtypes = [np.int32]
_nelems = [128]

@pytest.fixture(scope="module")
def rmm():
    print("initialize librmm")
    assert librmm.initialize() == librmm.RMM_SUCCESS
    yield librmm
    print("finalize librmm")
    assert librmm.finalize() == librmm.RMM_SUCCESS


@pytest.mark.parametrize('dtype,nelem', list(product(_dtypes, _nelems)))
def test_rmm_alloc(dtype, nelem, rmm):

    expect_fn = np.add
    test_fn = libgdf.gdf_add_generic

    #import cffi
    #ffi = cffi.FFI()

    # data
    h_in = gen_rand(dtype, nelem)
    h_result = gen_rand(dtype, nelem)
    
    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    print('expect')
    print(h_in)
    print('got')
    print(h_result)

    np.testing.assert_array_equal(h_result, h_in)

    assert rmm.free_device_array_memory(d_in) == rmm.RMM_SUCCESS
    assert rmm.free_device_array_memory(d_result) == rmm.RMM_SUCCESS


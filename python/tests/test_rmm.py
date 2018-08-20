import pytest
import functools
from itertools import product

import numpy as np
from numba import cuda

from librmm_cffi import librmm as rmm

from .utils import gen_rand

_dtypes = [np.int32]
_nelems = [1, 2, 7, 8, 9, 32, 128]

@pytest.mark.parametrize('dtype,nelem', list(product(_dtypes, _nelems)))
def test_rmm_alloc(dtype, nelem):
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

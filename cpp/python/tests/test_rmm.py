import pytest
from itertools import product

import numpy as np

from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg

from .utils import gen_rand

def array_tester(dtype, nelem):
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

_dtypes = [np.int32]
_nelems = [1, 2, 7, 8, 9, 32, 128]


@pytest.mark.parametrize('dtype,nelem', list(product(_dtypes, _nelems)))
def test_rmm_alloc(dtype, nelem):
    array_tester(dtype, nelem)


@pytest.mark.parametrize('managed, pool', 
                         list(product([False, True], [False, True])))
def test_rmm_modes(managed, pool):
    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    array_tester(np.int32, 128)


def test_rmm_csv_log():
    dtype = np.int32
    nelem = 1024

    # data
    h_in = gen_rand(dtype, nelem)
    gen_rand(dtype, nelem)

    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    d_result.copy_to_host()

    csv = rmm.csv_log()

    print(csv[:1000])

    assert(csv.find("Event Type,Device ID,Address,Stream,Size (bytes),"
                    "Free Memory,Total Memory,Current Allocs,Start,End,"
                    "Elapsed,Location") >= 0)

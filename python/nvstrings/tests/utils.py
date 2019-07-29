# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np

import nvstrings


def methodcaller(name, *args, **kwargs):
    def caller(obj):
        return getattr(obj, name)(*args, **kwargs)

    return caller


def assert_eq(arr1, arr2):
    if isinstance(arr1, nvstrings.nvstrings):
        arr1 = arr1.to_host()

    if isinstance(arr2, nvstrings.nvstrings):
        arr2 = arr2.to_host()

    assert np.array_equiv(arr1, arr2)


def initialize_rmm_pool():
    from librmm_cffi import librmm as rmm
    from librmm_cffi import librmm_config as rmm_cfg

    rmm_cfg.use_pool_allocator = True
    rmm_cfg.initial_pool_size = 2 << 30  # set to 2GiB. Default is 1/2 total GPU memory
    rmm_cfg.use_managed_memory = False  # default is false
    rmm_cfg.enable_logging = True
    return rmm.initialize()


def finalize_rmm():
    from librmm_cffi import librmm as rmm
    return rmm.finalize()

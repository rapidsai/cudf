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
    import rmm
    from rmm import rmm_config

    rmm_config.use_pool_allocator = True
    # set to 2GiB. Default is 1/2 total GPU memory
    rmm_config.initial_pool_size = 2 << 30
    # default is false
    rmm_config.use_managed_memory = False
    rmm_config.enable_logging = True
    return rmm.initialize()


def finalize_rmm():
    import rmm

    return rmm.finalize()

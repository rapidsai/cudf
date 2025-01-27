# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import pylibcudf
import rmm.mr

from .fast_slow_proxy import is_proxy_object
from .magics import load_ipython_extension
from .profiler import Profiler

__all__ = ["Profiler", "install", "is_proxy_object", "load_ipython_extension"]


LOADED = False

_SUPPORTED_PREFETCHES = {
    "column_view::get_data",
    "mutable_column_view::get_data",
    "gather",
    "hash_join",
}


def _enable_managed_prefetching(rmm_mode, managed_memory_is_supported):
    if managed_memory_is_supported and "managed" in rmm_mode:
        for key in _SUPPORTED_PREFETCHES:
            pylibcudf.experimental.enable_prefetching(key)


def install():
    """Enable Pandas Accelerator Mode."""
    from .module_accelerator import ModuleAccelerator

    loader = ModuleAccelerator.install("pandas", "cudf", "pandas")
    global LOADED
    LOADED = loader is not None

    # The default mode is "managed_pool" if UVM is supported, otherwise "pool"
    managed_memory_is_supported = (
        pylibcudf.utils._is_concurrent_managed_access_supported()
    )
    default_rmm_mode = (
        "managed_pool" if managed_memory_is_supported else "pool"
    )
    rmm_mode = os.getenv("CUDF_PANDAS_RMM_MODE", default_rmm_mode)

    if "managed" in rmm_mode and not managed_memory_is_supported:
        raise ValueError(
            f"Managed memory is not supported on this system, so the requested {rmm_mode=} is invalid."
        )

    # Check if a non-default memory resource is set
    current_mr = rmm.mr.get_current_device_resource()
    if not isinstance(current_mr, rmm.mr.CudaMemoryResource):
        warnings.warn(
            f"cudf.pandas detected an already configured memory resource, ignoring 'CUDF_PANDAS_RMM_MODE'={rmm_mode!s}",
            UserWarning,
        )
        return

    free_memory, _ = rmm.mr.available_device_memory()
    free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)
    new_mr = current_mr

    if rmm_mode == "pool":
        new_mr = rmm.mr.PoolMemoryResource(
            current_mr,
            initial_pool_size=free_memory,
        )
    elif rmm_mode == "async":
        new_mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=free_memory)
    elif rmm_mode == "managed":
        new_mr = rmm.mr.PrefetchResourceAdaptor(rmm.mr.ManagedMemoryResource())
    elif rmm_mode == "managed_pool":
        new_mr = rmm.mr.PrefetchResourceAdaptor(
            rmm.mr.PoolMemoryResource(
                rmm.mr.ManagedMemoryResource(),
                initial_pool_size=free_memory,
            )
        )
    elif rmm_mode != "cuda":
        raise ValueError(f"Unsupported {rmm_mode=}")

    rmm.mr.set_current_device_resource(new_mr)

    _enable_managed_prefetching(rmm_mode, managed_memory_is_supported)


def pytest_load_initial_conftests(early_config, parser, args):
    # We need to install ourselves before conftest.py import (which
    # might import pandas) This hook is guaranteed to run before that
    # happens see
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded pandas. Interposing failed."
        )

# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import rmm.mr

from .fast_slow_proxy import is_proxy_object
from .magics import load_ipython_extension
from .profiler import Profiler

__all__ = ["Profiler", "load_ipython_extension", "install", "is_proxy_object"]


LOADED = False


def install():
    """Enable Pandas Accelerator Mode."""
    from .module_accelerator import ModuleAccelerator

    loader = ModuleAccelerator.install("pandas", "cudf", "pandas")
    global LOADED
    LOADED = loader is not None

    if (rmm_mode := os.getenv("CUDF_PANDAS_RMM_MODE", None)) is not None:
        # Check if a non-default memory resource is set
        current_mr = rmm.mr.get_current_device_resource()
        if not isinstance(current_mr, rmm.mr.CudaMemoryResource):
            warnings.warn(
                f"cudf.pandas detected an already configured memory resource, ignoring 'CUDF_PANDAS_RMM_MODE'={str(rmm_mode)}",
                UserWarning,
            )
        free_memory, _ = rmm.mr.available_device_memory()
        free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)

        if rmm_mode == "cuda":
            mr = rmm.mr.CudaMemoryResource()
            rmm.mr.set_current_device_resource(mr)
        elif rmm_mode == "pool":
            rmm.mr.set_current_device_resource(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.get_current_device_resource(),
                    initial_pool_size=free_memory,
                )
            )
        elif rmm_mode == "async":
            mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=free_memory)
            rmm.mr.set_current_device_resource(mr)
        elif rmm_mode == "managed":
            mr = rmm.mr.ManagedMemoryResource()
            rmm.mr.set_current_device_resource(mr)
        elif rmm_mode == "managed_pool":
            mr = rmm.mr.PoolMemoryResource(
                rmm.mr.ManagedMemoryResource(),
                initial_pool_size=free_memory,
            )
            rmm.mr.set_current_device_resource(mr)
        else:
            raise ValueError(f"Unsupported rmm mode: {rmm_mode}")


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

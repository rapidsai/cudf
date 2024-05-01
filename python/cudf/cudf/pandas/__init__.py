# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .magics import load_ipython_extension
from .profiler import Profiler

__all__ = ["Profiler", "load_ipython_extension", "install"]


LOADED = False


def install():
    """Enable Pandas Accelerator Mode."""
    from .module_accelerator import ModuleAccelerator

    loader = ModuleAccelerator.install("pandas", "cudf", "pandas")
    global LOADED
    LOADED = loader is not None
    import os

    if cudf_pandas_mr := os.getenv("CUDF_PANDAS_MEMORY_RESOURCE", None) is not None:
        import rmm.mr

        if cudf_pandas_mr := getattr(rmm.mr, cudf_pandas_mr, None) is not None:
            from rmm.mr import PoolMemoryResource

            mr = PoolMemoryResource(
                cudf_pandas_mr(),
                initial_pool_size=os.getenv(
                    "CUDF_PANDAS_INITIAL_POOL_SIZE", None
                ),
                maximum_pool_size=os.getenv("CUDF_PANDAS_MAX_POOL_SIZE", None),
            )
            rmm.mr.set_current_device_resource(mr)


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

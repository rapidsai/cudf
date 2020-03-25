# Copyright (c) 2018-2019, NVIDIA CORPORATION.
""" __init__.py

   isort:skip_file
"""
import cupy

from cudf.errors import UnSupportedGPUError, UnSupportedCUDAError

gpus_count = cupy.cuda.runtime.getDeviceCount()


if gpus_count > 0:
    for device in range(0, gpus_count):
        # cudaDevAttrComputeCapabilityMajor - 75
        major_version = cupy.cuda.runtime.deviceGetAttribute(75, device)

        if major_version >= 6:
            # You have a GPU with NVIDIA Pascal™ architecture or better
            pass
        else:
            raise UnSupportedGPUError(
                "You will need a GPU with NVIDIA Pascal™ architecture or better"
            )

    cuda_runtime_version = cupy.cuda.runtime.runtimeGetVersion()

    if cuda_runtime_version > 10000:
        # CUDA Runtime Version Check: Runtime version is greater than 10000
        pass
    else:
        raise UnSupportedCUDAError(
            "Please update your CUDA Runtime to 10.0 or above"
        )

    cuda_driver_version = cupy.cuda.runtime.driverGetVersion()

    if cuda_driver_version == 0:
        raise UnSupportedCUDAError("Please install CUDA Driver")
    elif cuda_driver_version >= cuda_runtime_version:
        # CUDA Driver Version Check: Driver Runtime version is >= Runtime version
        pass
    else:
        raise UnSupportedCUDAError(
            "The detected driver version does not support the detected CUDA Runtime version. Please update your NVIDIA GPU Driver.\n"
            "Detected CUDA Runtime version : "
            + str(cuda_runtime_version)
            + "\n"
            "Latest version of CUDA supported by current NVIDIA GPU Driver : "
            + str(cuda_driver_version)
        )

else:
    import warnings

    warnings.warn(
        "You donot have an NVIDIA GPU, please install one and try again"
    )

import rmm
from cudf import core, datasets
from cudf._version import get_versions
from cudf.core import DataFrame, Index, MultiIndex, Series, from_pandas, merge
from cudf.core.dtypes import CategoricalDtype
from cudf.core.groupby import Grouper
from cudf.core.ops import (
    arccos,
    arcsin,
    arctan,
    cos,
    exp,
    log,
    logical_and,
    logical_not,
    logical_or,
    sin,
    sqrt,
    tan,
)
from cudf.core.reshape import concat, get_dummies, melt, merge_sorted
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
)
from cudf.utils.utils import set_allocator

cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

__version__ = get_versions()["version"]
del get_versions

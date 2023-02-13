# Copyright (c) 2022-2023, NVIDIA CORPORATION.
import os

from cuda import cudart
from ptxcompiler.patch import NO_DRIVER, safe_get_versions

from cudf.core.udf.utils import _get_cuda_version_from_ptx_file, _get_ptx_file

from . import _version

__version__ = _version.get_versions()["version"]


path = os.path.dirname(__file__)


# Maximum size of a string column is 2 GiB
_STRINGS_UDF_DEFAULT_HEAP_SIZE = os.environ.get(
    "STRINGS_UDF_HEAP_SIZE", 2**31
)
heap_size = 0


def set_malloc_heap_size(size=None):
    """
    Heap size control for strings_udf, size in bytes.
    """
    global heap_size
    if size is None:
        size = _STRINGS_UDF_DEFAULT_HEAP_SIZE
    if size != heap_size:
        (ret,) = cudart.cudaDeviceSetLimit(
            cudart.cudaLimit.cudaLimitMallocHeapSize, size
        )
        if ret.value != 0:
            raise RuntimeError("Unable to set cudaMalloc heap size")

        heap_size = size


ptxpath = None
versions = safe_get_versions()
if versions != NO_DRIVER:
    ptxpath = _get_ptx_file(path, "shim_")

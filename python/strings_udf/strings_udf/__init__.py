# Copyright (c) 2022, NVIDIA CORPORATION.
import glob
import os

from cubinlinker.patch import _numba_version_ok, get_logger, new_patched_linker
from cuda import cudart
from numba import cuda
from numba.cuda.cudadrv.driver import Linker
from ptxcompiler.patch import NO_DRIVER, safe_get_versions

from cudf.core.udf.utils import _get_ptx_file

from . import _version

__version__ = _version.get_versions()["version"]

logger = get_logger()

# tracks the version of CUDA used to build the c++ and PTX components
STRINGS_UDF_PTX_VERSION = (11, 5)

path = os.path.dirname(__file__)


def maybe_patch_numba_linker(driver_version):
    # Numba thinks cubinlinker is only needed if the driver is older than the ctk
    # but when strings_udf is present, it might also need to patch because the PTX
    # file strings_udf relies on may be newer than the driver as well
    if driver_version < STRINGS_UDF_PTX_VERSION:
        logger.debug(
            "Driver version %s.%s needs patching due to strings_udf"
            % driver_version
        )
        if _numba_version_ok:
            logger.debug("Patching Numba Linker")
            Linker.new = new_patched_linker
        else:
            logger.debug("Cannot patch Numba Linker - unsupported version")


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
    driver_version, runtime_version = versions
    maybe_patch_numba_linker(driver_version)
    ptxpath = _get_ptx_file(path, "shim_")

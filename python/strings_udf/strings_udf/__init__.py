# Copyright (c) 2022-2023, NVIDIA CORPORATION.
import glob
import os

from cubinlinker.patch import _numba_version_ok, get_logger, new_patched_linker
from cuda import cudart
from numba import cuda
from numba.cuda.cudadrv.driver import Linker
from ptxcompiler.patch import NO_DRIVER, safe_get_versions

from . import _version

__version__ = _version.get_versions()["version"]

logger = get_logger()

# tracks the version of CUDA used to build the c++ and PTX components
STRINGS_UDF_PTX_VERSION = (11, 8)


def _get_appropriate_file(sms, cc):
    filtered_sms = list(filter(lambda x: x[0] <= cc, sms))
    if filtered_sms:
        return max(filtered_sms, key=lambda y: y[0])
    else:
        return None


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


def _get_ptx_file():
    if "RAPIDS_NO_INITIALIZE" in os.environ:
        # shim_60.ptx is always built
        cc = int(os.environ.get("STRINGS_UDF_CC", "60"))
    else:
        dev = cuda.get_current_device()

        # Load the highest compute capability file available that is less than
        # the current device's.
        cc = int("".join(str(x) for x in dev.compute_capability))
    files = glob.glob(os.path.join(os.path.dirname(__file__), "shim_*.ptx"))
    if len(files) == 0:
        raise RuntimeError(
            "This strings_udf installation is missing the necessary PTX "
            f"files for compute capability {cc}. "
            "Please file an issue reporting this error and how you "
            "installed cudf and strings_udf."
            "https://github.com/rapidsai/cudf/issues"
        )

    regular_sms = []

    for f in files:
        file_name = os.path.basename(f)
        sm_number = file_name.rstrip(".ptx").lstrip("shim_")
        if sm_number.endswith("a"):
            processed_sm_number = int(sm_number.rstrip("a"))
            if processed_sm_number == cc:
                return f
        else:
            regular_sms.append((int(sm_number), f))

    regular_result = None

    if regular_sms:
        regular_result = _get_appropriate_file(regular_sms, cc)

    if regular_result is None:
        raise RuntimeError(
            "This strings_udf installation is missing the necessary PTX "
            f"files that are <={cc}."
        )
    else:
        return regular_result[1]


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
    ptxpath = _get_ptx_file()

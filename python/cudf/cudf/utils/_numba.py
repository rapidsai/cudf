# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import glob
import os
import sys
from functools import lru_cache

from numba import config as numba_config


# Use an lru_cache with a single value to allow a delayed import of
# strings_udf. This is the easiest way to break an otherwise circular import
# loop of _lib.*->cudautils->_numba->_lib.strings_udf
@lru_cache
def _get_cuda_build_version():
    from cudf._lib import strings_udf

    # The version is an integer, parsed as 1000 * major + 10 * minor
    cuda_build_version = strings_udf.get_cuda_build_version()
    cuda_major_version = cuda_build_version // 1000
    cuda_minor_version = (cuda_build_version % 1000) // 10
    return (cuda_major_version, cuda_minor_version)


def _get_best_ptx_file(archs, max_compute_capability):
    """
    Determine of the available PTX files which one is
    the most recent up to and including the device compute capability.
    """
    filtered_archs = [x for x in archs if x[0] <= max_compute_capability]
    if filtered_archs:
        return max(filtered_archs, key=lambda x: x[0])
    else:
        return None


def _get_ptx_file(path, prefix):
    if "RAPIDS_NO_INITIALIZE" in os.environ:
        # cc=70 ptx is always built
        cc = int(os.environ.get("STRINGS_UDF_CC", "70"))
    else:
        from numba import cuda

        dev = cuda.get_current_device()

        # Load the highest compute capability file available that is less than
        # the current device's.
        cc = int("".join(str(x) for x in dev.compute_capability))
    files = glob.glob(os.path.join(path, f"{prefix}*.ptx"))
    if len(files) == 0:
        raise RuntimeError(f"Missing PTX files for cc={cc}")
    regular_sms = []

    for f in files:
        file_name = os.path.basename(f)
        sm_number = file_name.rstrip(".ptx").lstrip(prefix)
        if sm_number.endswith("a"):
            processed_sm_number = int(sm_number.rstrip("a"))
            if processed_sm_number == cc:
                return f
        else:
            regular_sms.append((int(sm_number), f))

    regular_result = None

    if regular_sms:
        regular_result = _get_best_ptx_file(regular_sms, cc)

    if regular_result is None:
        raise RuntimeError(
            "This cuDF installation is missing the necessary PTX "
            f"files that are <={cc}."
        )
    else:
        return regular_result[1]


def patch_numba_linker_cuda_11():
    # Enable the config option for minor version compatibility
    numba_config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1

    if "numba.cuda" in sys.modules:
        # Patch numba for version 0.57.0 MVC support, which must know the
        # config value at import time. We cannot guarantee the order of imports
        # between cudf and numba.cuda so we patch numba to ensure it has these
        # names available.
        # See https://github.com/numba/numba/issues/8977 for details.
        import numba.cuda
        from cubinlinker import CubinLinker, CubinLinkerError
        from ptxcompiler import compile_ptx

        numba.cuda.cudadrv.driver.compile_ptx = compile_ptx
        numba.cuda.cudadrv.driver.CubinLinker = CubinLinker
        numba.cuda.cudadrv.driver.CubinLinkerError = CubinLinkerError


def _setup_numba():
    """
    Configure the numba linker for use with cuDF. This consists of
    potentially putting numba into enhanced compatibility mode
    based on the user driver and runtime versions as well as the
    version of the CUDA Toolkit used to build the PTX files shipped
    with the user cuDF package.
    """

    # Either ptxcompiler, or our vendored version (_ptxcompiler.py)
    # is needed to determine the driver and runtime CUDA versions in
    # the environment. In a CUDA 11.x environment, ptxcompiler is used
    # to provide MVC directly, whereas for CUDA 12.x this is provided
    # through pynvjitlink. The presence of either package does not
    # perturb cuDF's operation in situations where they aren't used.
    try:
        from ptxcompiler.patch import NO_DRIVER, safe_get_versions
    except ModuleNotFoundError:
        # use vendored version
        from cudf.utils._ptxcompiler import NO_DRIVER, safe_get_versions

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        shim_ptx_cuda_version = _get_cuda_build_version()

        # MVC is required whenever any PTX is newer than the driver
        # This could be the shipped shim PTX file (determined by the CUDA
        # version used at build time) or the PTX emitted by the version of NVVM
        # on the user system (determined by the user's CUDA runtime version)
        if (driver_version < shim_ptx_cuda_version) or (
            driver_version < runtime_version
        ):
            if driver_version < (12, 0):
                patch_numba_linker_cuda_11()
            else:
                from pynvjitlink.patch import patch_numba_linker

                patch_numba_linker()


class _CUDFNumbaConfig:
    def __enter__(self):
        self.CUDA_LOW_OCCUPANCY_WARNINGS = (
            numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self.CAPTURED_ERRORS = numba_config.CAPTURED_ERRORS
        numba_config.CAPTURED_ERRORS = "new_style"

    def __exit__(self, exc_type, exc_value, traceback):
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
            self.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CAPTURED_ERRORS = self.CAPTURED_ERRORS

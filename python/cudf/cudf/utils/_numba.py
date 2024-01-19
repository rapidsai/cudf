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
def _get_cc_60_ptx_file():
    from cudf._lib import strings_udf

    return os.path.join(
        os.path.dirname(strings_udf.__file__),
        "..",
        "core",
        "udf",
        "shim_60.ptx",
    )


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
        # cc=60 ptx is always built
        cc = int(os.environ.get("STRINGS_UDF_CC", "60"))
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
        ptx_toolkit_version = _get_cuda_version_from_ptx_file(
            _get_cc_60_ptx_file()
        )

        # MVC is required whenever any PTX is newer than the driver
        # This could be the shipped PTX file or the PTX emitted by
        # the version of NVVM on the user system, the latter aligning
        # with the runtime version
        if (driver_version < ptx_toolkit_version) or (
            driver_version < runtime_version
        ):
            if driver_version < (12, 0):
                patch_numba_linker_cuda_11()
            else:
                from pynvjitlink.patch import patch_numba_linker

                patch_numba_linker()


def _get_cuda_version_from_ptx_file(path):
    """
    https://docs.nvidia.com/cuda/parallel-thread-execution/
    Each PTX module must begin with a .version
    directive specifying the PTX language version

    example header:
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-31057947
    // Cuda compilation tools, release 11.6, V11.6.124
    // Based on NVVM 7.0.1
    //

    .version 7.6
    .target sm_52
    .address_size 64

    """
    with open(path) as ptx_file:
        for line in ptx_file:
            if line.startswith(".version"):
                ver_line = line
                break
        else:
            raise ValueError("Could not read CUDA version from ptx file.")
    version = ver_line.strip("\n").split(" ")[1]
    # This dictionary maps from supported versions of NVVM to the
    # PTX version it produces. The lowest value should be the minimum
    # CUDA version required to compile the library. Currently CUDA 11.5
    # or higher is required to build cudf. New CUDA versions should
    # be added to this dictionary when officially supported.
    ver_map = {
        "7.5": (11, 5),
        "7.6": (11, 6),
        "7.7": (11, 7),
        "7.8": (11, 8),
        "8.0": (12, 0),
        "8.1": (12, 1),
        "8.2": (12, 2),
        "8.3": (12, 3),
    }

    cuda_ver = ver_map.get(version)
    if cuda_ver is None:
        raise ValueError(
            f"Could not map PTX version {version} to a CUDA version"
        )

    return cuda_ver


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

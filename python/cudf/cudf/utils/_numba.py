# Copyright (c) 2023, NVIDIA CORPORATION.

import glob
import os
import sys
import warnings

from numba import config as numba_config

CC_60_PTX_FILE = os.path.join(
    os.path.dirname(__file__), "../core/udf/shim_60.ptx"
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


def _patch_numba_mvc():
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
    # ptxcompiler is a requirement for cuda 11.x packages but not
    # cuda 12.x packages. However its version checking machinery
    # is still necessary. If a user happens to have ptxcompiler
    # in a cuda 12 environment, it's use for the purposes of
    # checking the driver and runtime versions is harmless
    try:
        from ptxcompiler.patch import NO_DRIVER, safe_get_versions
    except ModuleNotFoundError:
        # use vendored version
        from cudf.utils._ptxcompiler import NO_DRIVER, safe_get_versions

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        if driver_version >= (12, 0) and runtime_version > driver_version:
            warnings.warn(
                f"Using CUDA toolkit version {runtime_version} with CUDA "
                f"driver version {driver_version} requires minor version "
                "compatibility, which is not yet supported for CUDA "
                "driver versions 12.0 and above. It is likely that many "
                "cuDF operations will not work in this state. Please "
                f"install CUDA toolkit version {driver_version} to "
                "continue using cuDF."
            )
        else:
            # Support MVC for all CUDA versions in the 11.x range
            ptx_toolkit_version = _get_cuda_version_from_ptx_file(
                CC_60_PTX_FILE
            )
            # Numba thinks cubinlinker is only needed if the driver is older
            # than the CUDA runtime, but when PTX files are present, it might
            # also need to patch because those PTX files may be compiled by
            # a CUDA version that is newer than the driver as well
            if (driver_version < ptx_toolkit_version) or (
                driver_version < runtime_version
            ):
                _patch_numba_mvc()


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
    }

    cuda_ver = ver_map.get(version)
    if cuda_ver is None:
        raise ValueError(
            f"Could not map PTX version {version} to a CUDA version"
        )

    return cuda_ver


class _CUDFNumbaConfig:
    def __enter__(self):
        self.enter_val = numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    def __exit__(self, exc_type, exc_value, traceback):
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = self.enter_val

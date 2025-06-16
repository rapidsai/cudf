# Copyright (c) 2023-2025, NVIDIA CORPORATION.
from __future__ import annotations

import glob
import os
from importlib.util import find_spec

import numba
from numba import config as numba_config
from packaging import version


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


def _setup_numba():
    """
    Configure the numba linker for use with cuDF. This consists of
    potentially putting numba into enhanced compatibility mode
    based on the user driver and runtime versions as well as the
    version of the CUDA Toolkit used to build the PTX files shipped
    with the user cuDF package.
    """

    # In CUDA 12+, pynvjitlink is used to provide minor version compatibility.
    if find_spec("pynvjitlink") is not None:
        # Assume CUDA 12+ if pynvjitlink is available and unconditionally
        # enable pynvjitlink.
        #
        # If somehow pynvjitlink is available on a CUDA 11 installation, numba
        # will fail with an error: "Enabling pynvjitlink requires CUDA 12."
        # This is not a supported use case.
        numba_config.CUDA_ENABLE_PYNVJITLINK = True


# Avoids using contextlib.contextmanager due to additional overhead
class _CUDFNumbaConfig:
    def __enter__(self) -> None:
        self.CUDA_LOW_OCCUPANCY_WARNINGS = (
            numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self.is_numba_lt_061 = version.parse(
            numba.__version__
        ) < version.parse("0.61")

        if self.is_numba_lt_061:
            self.CAPTURED_ERRORS = numba_config.CAPTURED_ERRORS
            numba_config.CAPTURED_ERRORS = "new_style"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
            self.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        if self.is_numba_lt_061:
            numba_config.CAPTURED_ERRORS = self.CAPTURED_ERRORS

# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import subprocess
import sys
import warnings

NO_DRIVER = (math.inf, math.inf)

NUMBA_CHECK_VERSION_CMD = """\
from ctypes import c_int, byref
from numba import cuda
dv = c_int(0)
cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
drv_major = dv.value // 1000
drv_minor = (dv.value - (drv_major * 1000)) // 10
run_major, run_minor = cuda.runtime.get_version()
print(f'{drv_major} {drv_minor} {run_major} {run_minor}')
"""


def check_disabled_in_env():
    # We should avoid checking whether the patch is
    # needed if the user requested that we don't check
    # (e.g. in a non-fork-safe environment)
    check = os.getenv("PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED")
    if check is not None:
        try:
            check = int(check)
        except ValueError:
            check = False
    else:
        check = True

    return not check


def get_versions():
    cp = subprocess.run(
        [sys.executable, "-c", NUMBA_CHECK_VERSION_CMD], capture_output=True
    )
    if cp.returncode:
        msg = (
            f"Error getting driver and runtime versions:\n\nstdout:\n\n"
            f"{cp.stdout.decode()}\n\nstderr:\n\n{cp.stderr.decode()}\n\n"
            "Not patching Numba"
        )
        warnings.warn(msg, UserWarning)
        return NO_DRIVER

    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    runtime_version = tuple(versions[2:])

    return driver_version, runtime_version


def safe_get_versions():
    """
    Return a 2-tuple of deduced driver and runtime versions.

    To ensure that this function does not initialize a CUDA context,
    calls to the runtime and driver are made in a subprocess.

    If PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED is set
    in the environment, then this subprocess call is not launched.
    To specify the driver and runtime versions of the environment
    in this case, set PTXCOMPILER_KNOWN_DRIVER_VERSION and
    PTXCOMPILER_KNOWN_RUNTIME_VERSION appropriately.
    """
    if check_disabled_in_env():
        try:
            # allow user to specify driver/runtime
            # versions manually, if necessary
            driver_version = os.environ[
                "PTXCOMPILER_KNOWN_DRIVER_VERSION"
            ].split(".")
            runtime_version = os.environ[
                "PTXCOMPILER_KNOWN_RUNTIME_VERSION"
            ].split(".")
            driver_version, runtime_version = (
                tuple(map(int, driver_version)),
                tuple(map(int, runtime_version)),
            )
        except (KeyError, ValueError):
            warnings.warn(
                "No way to determine driver and runtime versions for "
                "patching, set PTXCOMPILER_KNOWN_DRIVER_VERSION and "
                "PTXCOMPILER_KNOWN_RUNTIME_VERSION"
            )
            return NO_DRIVER
    else:
        driver_version, runtime_version = get_versions()
    return driver_version, runtime_version

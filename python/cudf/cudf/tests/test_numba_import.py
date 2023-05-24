# Copyright (c) 2023, NVIDIA CORPORATION.
import subprocess
import sys

import pytest

skip = True
try:
    from ptxcompiler.patch import NO_DRIVER, safe_get_versions

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        if driver_version < (12, 0):
            skip = False
except ModuleNotFoundError:
    pass

test = """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig, _patch_numba_mvc

from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1

_patch_numba_mvc()

@numba.cuda.jit
def test_kernel(x):
    id = numba.cuda.grid(1)
    if id < len(x):
        x[id] += 1

s = cudf.Series([1,2,3])
with _CUDFNumbaConfig():
    test_kernel.forall(len(s))(s)
"""


@pytest.mark.skipif(False, reason="MVC Not Required")
def test_numba_mvc_enabled():
    cp = subprocess.run([sys.executable, "-c", test], capture_output=True)
    assert cp.returncode == 0

# Copyright (c) 2023, NVIDIA CORPORATION.
import subprocess
import sys

import pytest

IS_CUDA_11 = False
try:
    from ptxcompiler.patch import NO_DRIVER, safe_get_versions

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        if driver_version < (12, 0):
            IS_CUDA_11 = True
except ModuleNotFoundError:
    pass

TEST_NUMBA_MVC_ENABLED = """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig, _patch_numba_mvc


_patch_numba_mvc()

@numba.cuda.jit
def test_kernel(x):
    id = numba.cuda.grid(1)
    if id < len(x):
        x[id] += 1

s = cudf.Series([1, 2, 3])
with _CUDFNumbaConfig():
    test_kernel.forall(len(s))(s)
"""


@pytest.mark.skipif(
    not IS_CUDA_11, reason="Minor Version Compatibility test for CUDA 11"
)
def test_numba_mvc_enabled_cuda_11():
    cp = subprocess.run(
        [sys.executable, "-c", TEST_NUMBA_MVC_ENABLED],
        capture_output=True,
        cwd="/",
    )
    assert cp.returncode == 0

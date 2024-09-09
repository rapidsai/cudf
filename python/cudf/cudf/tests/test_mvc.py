# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import subprocess
import sys

import pytest
from packaging import version

IS_CUDA_11 = False
IS_CUDA_12 = False
try:
    from ptxcompiler.patch import safe_get_versions
except ModuleNotFoundError:
    from cudf.utils._ptxcompiler import safe_get_versions

# do not test cuda 12 if pynvjitlink isn't present
HAVE_PYNVJITLINK = False
try:
    import numba
    import pynvjitlink  # noqa: F401

    HAVE_PYNVJITLINK = version.parse(numba.__version__) >= version.parse(
        "0.58"
    )
except ModuleNotFoundError:
    pass


versions = safe_get_versions()
driver_version, runtime_version = versions

if (11, 0) <= driver_version < (12, 0):
    IS_CUDA_11 = True
if (12, 0) <= driver_version < (13, 0):
    IS_CUDA_12 = True


TEST_BODY = """
@numba.cuda.jit
def test_kernel(x):
    id = numba.cuda.grid(1)
    if id < len(x):
        x[id] += 1

s = cudf.Series([1, 2, 3])
with _CUDFNumbaConfig():
    test_kernel.forall(len(s))(s)
"""

CUDA_11_TEST = (
    """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig, patch_numba_linker_cuda_11


patch_numba_linker_cuda_11()
"""
    + TEST_BODY
)


CUDA_12_TEST = (
    """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig
from pynvjitlink.patch import (
    patch_numba_linker as patch_numba_linker_pynvjitlink,
)

patch_numba_linker_pynvjitlink()
"""
    + TEST_BODY
)


@pytest.mark.parametrize(
    "test",
    [
        pytest.param(
            CUDA_11_TEST,
            marks=pytest.mark.skipif(
                not IS_CUDA_11,
                reason="Minor Version Compatibility test for CUDA 11",
            ),
        ),
        pytest.param(
            CUDA_12_TEST,
            marks=pytest.mark.skipif(
                not IS_CUDA_12 or not HAVE_PYNVJITLINK,
                reason="Minor Version Compatibility test for CUDA 12",
            ),
        ),
    ],
)
def test_numba_mvc(test):
    cp = subprocess.run(
        [sys.executable, "-c", test],
        capture_output=True,
        cwd="/",
    )

    assert cp.returncode == 0

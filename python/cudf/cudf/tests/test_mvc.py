# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import subprocess
import sys
from importlib.util import find_spec

import pytest

IS_CUDA_12_PLUS = find_spec("pynvjitlink") is not None
IS_CUDA_11 = not IS_CUDA_12_PLUS


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


CUDA_12_PLUS_TEST = (
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
            CUDA_12_PLUS_TEST,
            marks=pytest.mark.skipif(
                not IS_CUDA_12_PLUS,
                reason="Minor Version Compatibility test for CUDA 12+",
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

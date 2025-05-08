# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import subprocess
import sys
from importlib.util import find_spec

IS_CUDA_12_PLUS = find_spec("pynvjitlink") is not None
IS_CUDA_11 = not IS_CUDA_12_PLUS


TEST_SCRIPT = """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig, _setup_numba

_setup_numba()

@numba.cuda.jit
def test_kernel(x):
    id = numba.cuda.grid(1)
    if id < len(x):
        x[id] += 1

s = cudf.Series([1, 2, 3])
with _CUDFNumbaConfig():
    test_kernel.forall(len(s))(s)
"""


def test_numba_mvc():
    cp = subprocess.run(
        [sys.executable, "-c", TEST_SCRIPT],
        capture_output=True,
        cwd="/",
    )

    assert cp.returncode == 0

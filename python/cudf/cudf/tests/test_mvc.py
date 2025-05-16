# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import subprocess
import sys

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

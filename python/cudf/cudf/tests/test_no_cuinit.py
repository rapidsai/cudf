# Copyright (c) 2023, NVIDIA CORPORATION.

import os
import platform
import subprocess
import sys
from shutil import which

import pytest

gdb = which("cuda-gdb")
machine_arch = platform.uname().machine


GDB_COMMANDS = b"""
set confirm off
set breakpoint pending on
break cuInit
run
exit
"""


pytestmark = [
    pytest.mark.skipif(
        machine_arch != "x86_64",
        reason=(
            "cuda-gdb install is broken on nvidia/cuda aarch64 images "
            "(libexpat is missing)"
        ),
    ),
    pytest.mark.skipif(
        gdb is None, reason="cuda-gdb not found, can't detect cuInit"
    ),
]


def test_cudf_import_no_cuinit():
    # When RAPIDS_NO_INITIALIZE is set, importing cudf should _not_
    # create a CUDA context (i.e. cuInit should not be called).
    # Intercepting the call to cuInit programmatically is tricky since
    # the way it is resolved from dynamic libraries by
    # cuda-python/numba/cupy is multitudinous (see discussion at
    # https://github.com/rapidsai/cudf/pull/12361 which does this, but
    # needs provide hooks that override dlsym, cuGetProcAddress, and
    # cuInit.
    # Instead, we just run under GDB and see if we hit a breakpoint
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    output: str = subprocess.check_output(
        [
            gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf",
        ],
        input=GDB_COMMANDS,
        env=env,
        stderr=subprocess.DEVNULL,
    ).decode()

    cuInit_called = output.find("in cuInit ()")
    assert cuInit_called < 0


def test_cudf_create_series_cuinit():
    # This tests that our gdb scripting correctly identifies cuInit
    # when it definitely should have been called.
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    output: str = subprocess.check_output(
        [
            gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf; cudf.Series([1])",
        ],
        input=GDB_COMMANDS,
        env=env,
        stderr=subprocess.DEVNULL,
    ).decode()

    cuInit_called = output.find("in cuInit ()")
    assert cuInit_called >= 0

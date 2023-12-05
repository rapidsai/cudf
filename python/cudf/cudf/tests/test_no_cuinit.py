# Copyright (c) 2023, NVIDIA CORPORATION.

import os
import subprocess
import sys
from shutil import which

import pytest

GDB_COMMANDS = """
set confirm off
set breakpoint pending on
break cuInit
run
exit
"""


@pytest.fixture(scope="module")
def cuda_gdb(request):
    gdb = which("cuda-gdb")
    if gdb is None:
        request.applymarker(
            pytest.mark.xfail(reason="No cuda-gdb found, can't detect cuInit"),
        )
        return gdb
    else:
        output = subprocess.run(
            [gdb, "--version"], capture_output=True, text=True, cwd="/"
        )
        if output.returncode != 0:
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "cuda-gdb not working on this platform, "
                        f"can't detect cuInit: {output.stderr}"
                    )
                ),
            )
        return gdb


def test_cudf_import_no_cuinit(cuda_gdb):
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
    output = subprocess.run(
        [
            cuda_gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf",
        ],
        input=GDB_COMMANDS,
        env=env,
        capture_output=True,
        text=True,
        cwd="/",
    )

    cuInit_called = output.stdout.find("in cuInit ()")
    print("Command output:\n")
    print("*** STDOUT ***")
    print(output.stdout)
    print("*** STDERR ***")
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called < 0


def test_cudf_create_series_cuinit(cuda_gdb):
    # This tests that our gdb scripting correctly identifies cuInit
    # when it definitely should have been called.
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    output = subprocess.run(
        [
            cuda_gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf; cudf.Series([1])",
        ],
        input=GDB_COMMANDS,
        env=env,
        capture_output=True,
        text=True,
        cwd="/",
    )

    cuInit_called = output.stdout.find("in cuInit ()")
    print("Command output:\n")
    print("*** STDOUT ***")
    print(output.stdout)
    print("*** STDERR ***")
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called >= 0

# Copyright (c) 2022, NVIDIA CORPORATION.

import os
import subprocess
import sys
from pathlib import Path

import pytest

location = Path(__file__)
cpp_build_dir = location / ".." / ".." / ".." / ".." / ".." / "cpp" / "build"
libintercept = (cpp_build_dir / "libcudfcuinit_intercept.so").resolve()


@pytest.mark.skipif(
    not libintercept.exists(),
    reason="libcudfcuinit_intercept.so not built, can't check for cuInit",
)
def test_import_no_cuinit():
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    env["LD_PRELOAD"] = str(libintercept)
    output = subprocess.check_output(
        [sys.executable, "-c", "import cudf"],
        env=env,
        stderr=subprocess.STDOUT,
    )
    assert "cuInit has been called" not in output.decode()

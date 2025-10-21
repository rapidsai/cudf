# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import sys

import pytest


# Proxy check for whether the proxy could be referencing GPU objects without
# trying to import cuDF, which could poison the global environment
def _gpu_available():
    try:
        import rmm

        return rmm._cuda.gpu.getDeviceCount() >= 1
    except ImportError:
        return False


LOCATION = pathlib.Path(__file__).absolute().parent


@pytest.mark.skipif(
    not _gpu_available(), reason="Skipping test if a GPU isn't available."
)
def test_magics_gpu():
    sp_completed = subprocess.run(
        [sys.executable, LOCATION / "_magics_gpu_test.py"], capture_output=True
    )
    assert sp_completed.stderr.decode() == ""


@pytest.mark.skip(
    "This test was viable when cudf.pandas was separate from cudf, but now "
    "that it is a subpackage we always require a GPU to be present and cannot "
    "run this test."
)
def test_magics_cpu():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    sp_completed = subprocess.run(
        [sys.executable, LOCATION / "_magics_cpu_test.py"],
        capture_output=True,
        env=env,
    )
    assert sp_completed.stderr.decode() == ""

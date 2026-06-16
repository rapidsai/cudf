# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import os
import pathlib
import subprocess

import pytest

import rmm.mr
import rmm.statistics

LOCATION = pathlib.Path(__file__).absolute().parent


@pytest.fixture
def rmm_reset():
    """Fixture to reset the RMM resource before and after the test"""
    mr = rmm.mr.get_current_device_resource()
    try:
        rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
        yield
    finally:
        rmm.mr.set_current_device_resource(mr)


def test_memory_profiling(rmm_reset):
    env = os.environ.copy()
    env["CUDF_MEMORY_PROFILING"] = "true"

    # We need to run this test in a subprocess because we
    # need to set the env variable `CUDF_MEMORY_PROFILING=1` prior to
    # the launch of the Python interpreter if `memory_profiling` is needed.
    result = subprocess.run(
        ["python", LOCATION / "_rmm_stats_script.py"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Test failed: {result.stderr}"

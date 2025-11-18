# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest

import rmm.mr
import rmm.statistics


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
    import subprocess
    import sys

    test_code = """
import rmm.mr
import rmm.statistics
import cudf
from cudf.utils.performance_tracking import get_memory_records, print_memory_report
from io import StringIO

# Reset RMM
rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

df1 = cudf.DataFrame({"a": [1, 2, 3]})
assert len(get_memory_records()) == 0

rmm.statistics.enable_statistics()
cudf.set_option("memory_profiling", True)

df1.merge(df1)

assert len(get_memory_records()) > 0

out = StringIO()
print_memory_report(file=out)
assert "DataFrame.merge" in out.getvalue()
"""

    # We need to run this test in a subprocess because we
    # need to set the env variable `CUDF_MEMORY_PROFILING=1` prior to
    # the launch of the Python interpreter if `memory_profiling` is needed.
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        env={**os.environ, "CUDF_MEMORY_PROFILING": "true"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Test failed: {result.stderr}"

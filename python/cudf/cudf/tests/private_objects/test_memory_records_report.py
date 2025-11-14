# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest

import rmm.mr
import rmm.statistics

import cudf
from cudf.utils.performance_tracking import (
    get_memory_records,
    print_memory_report,
)


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
    df1 = cudf.DataFrame({"a": [1, 2, 3]})
    assert len(get_memory_records()) == 0

    rmm.statistics.enable_statistics()
    cudf.set_option("memory_profiling", True)

    df1.merge(df1)

    assert len(get_memory_records()) > 0

    out = StringIO()
    print_memory_report(file=out)
    assert "DataFrame.merge" in out.getvalue()

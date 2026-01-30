# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.experimental.base import PartitionInfo, RuntimeProfiler
from cudf_polars.experimental.explain import _repr_profile_tree, write_profile_output
from cudf_polars.testing.asserts import DEFAULT_CLUSTER, DEFAULT_RUNTIME


def test_runtime_profiler_and_output(tmp_path):
    """Test RuntimeProfiler, merge, and output formatting."""

    class MockIR:
        children = ()

        def __init__(self):
            self.schema = {"x": pl.Int64}

    ir1, ir2 = MockIR(), MockIR()

    # Test accumulation and merge
    profiler1 = RuntimeProfiler()
    profiler1.row_count[ir1] = 100
    profiler1.chunk_count[ir1] = 5
    profiler1.decisions[ir1] = "shuffle"

    profiler2 = RuntimeProfiler()
    profiler2.row_count[ir1] = 150
    profiler2.row_count[ir2] = 200
    profiler2.chunk_count[ir1] = 3
    profiler2.chunk_count[ir2] = 4

    profiler1.merge(profiler2)
    assert profiler1.row_count[ir1] == 250
    assert profiler1.chunk_count[ir1] == 8

    # Test _repr_profile_tree output format
    partition_info = {ir1: PartitionInfo(count=1)}
    output = _repr_profile_tree(ir1, partition_info, profiler1)
    assert "rows=250" in output
    assert "chunks=8" in output
    assert "decision=shuffle" in output

    # Test write_profile_output
    output_path = tmp_path / "profile.txt"
    write_profile_output(output_path, ir1, partition_info, profiler1)
    assert output_path.exists()
    content = output_path.read_text()
    assert "rows=250" in content


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_profiling_with_real_query(tmp_path):
    """Test profiling output with a real query execution."""
    output_path = tmp_path / "profile.txt"
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "profiling": {"output_path": str(output_path)},
        },
    )

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).select(["x", "y"])
    q.collect(engine=engine)

    # Verify profile was written
    assert output_path.exists()
    content = output_path.read_text()
    assert "rows=" in content
    assert "chunks=" in content

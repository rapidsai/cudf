# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RuntimeQueryProfiler and related output functions."""

from __future__ import annotations

import polars as pl

from cudf_polars.experimental.base import PartitionInfo, RuntimeQueryProfiler
from cudf_polars.experimental.explain import _repr_profile_tree, write_profile_output


def test_runtime_profiler_and_output(tmp_path):
    """Test RuntimeQueryProfiler, merge, and output formatting."""

    class MockIR:
        children = ()

        def __init__(self):
            self.schema = {"x": pl.Int64}

        def get_hashable(self):
            return (type(self), tuple(self.schema.items()))

    ir1, ir2 = MockIR(), MockIR()

    # Test node profiler accumulation
    profiler1 = RuntimeQueryProfiler()
    np1 = profiler1.get_or_create(ir1)
    np1.row_count = 100
    np1.chunk_count = 5
    np1.decision = "shuffle"

    profiler2 = RuntimeQueryProfiler()
    profiler2.get_or_create(ir1).row_count = 150
    profiler2.get_or_create(ir1).chunk_count = 3
    profiler2.get_or_create(ir1).decision = "shuffle"
    profiler2.get_or_create(ir2).row_count = 200
    profiler2.get_or_create(ir2).chunk_count = 3
    profiler2.get_or_create(ir2).add_chunk()

    # Test merge
    profiler1.merge(profiler2)
    assert profiler1.node_profilers[ir1].row_count == 250
    assert profiler1.node_profilers[ir1].chunk_count == 8

    # Test _repr_profile_tree output format
    partition_info = {ir1: PartitionInfo(count=4)}
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

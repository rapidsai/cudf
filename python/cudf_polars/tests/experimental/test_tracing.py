# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StreamingQueryTracer and related output functions."""

from __future__ import annotations

import polars as pl

from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.explain import _repr_trace_tree, write_query_trace
from cudf_polars.experimental.rapidsmpf.tracing import StreamingQueryTracer


def test_streaming_query_tracer_and_output(tmp_path):
    """Test StreamingQueryTracer, merge, and output formatting."""

    class MockIR:
        children = ()

        def __init__(self):
            self.schema = {"x": pl.Int64}

        def get_hashable(self):
            return (type(self), tuple(self.schema.items()))

    ir1, ir2 = MockIR(), MockIR()

    # Test node tracer accumulation
    tracer1 = StreamingQueryTracer()
    nt1 = tracer1.get_or_create(ir1)
    nt1.row_count = 100
    nt1.chunk_count = 5
    nt1.decision = "shuffle"

    tracer2 = StreamingQueryTracer()
    tracer2.get_or_create(ir1).row_count = 150
    tracer2.get_or_create(ir1).chunk_count = 3
    tracer2.get_or_create(ir1).decision = "shuffle"
    tracer2.get_or_create(ir2).row_count = 200
    tracer2.get_or_create(ir2).chunk_count = 3
    tracer2.get_or_create(ir2).add_chunk()

    # Test merge
    tracer1.merge(tracer2)
    assert tracer1.node_tracers[ir1].row_count == 250
    assert tracer1.node_tracers[ir1].chunk_count == 8

    # Test _repr_trace_tree output format
    partition_info = {ir1: PartitionInfo(count=4)}
    output = _repr_trace_tree(ir1, partition_info, tracer1)
    assert "rows=250" in output
    assert "chunks=8" in output
    assert "decision=shuffle" in output

    # Test write_query_trace
    output_path = tmp_path / "trace.txt"
    write_query_trace(output_path, ir1, partition_info, tracer1)
    assert output_path.exists()
    content = output_path.read_text()
    assert "rows=250" in content

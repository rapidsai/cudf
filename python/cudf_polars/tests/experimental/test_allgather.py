# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF AllGather functionality."""

from __future__ import annotations

import asyncio

import numpy as np
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.utils import allgather_reduce


async def _test_allgather(engine) -> None:
    """Very simple test that AllGatherManager can concatenate tables."""
    context = engine.context
    comm = engine.comm
    stream = context.get_stream_from_pool()

    # Create simple test tables with different sizes
    tables = [
        plc.Table([plc.Column.from_array(np.full(num_elements, i).astype(np.int32))])
        for i, num_elements in enumerate([100, 200, 300])
    ]

    # Insert tables into AllGatherManager
    allgather = AllGatherManager(context, comm, 0)
    with allgather.inserting() as inserter:
        for i, table in enumerate(tables):
            inserter.insert(
                i,
                TableChunk.from_pylibcudf_table(
                    table, stream, exclusive_view=True, br=context.br()
                ),
            )

    # Extract concatenated result
    result = await allgather.extract_concatenated(stream, ordered=True)

    # Verify the concatenated table has the expected shape
    assert result.num_rows() == 600  # 100 + 200 + 300
    assert result.num_columns() == 1

    # Verify the column type is correct
    col = result.columns()[0]
    assert col.size() == 600
    assert col.type().id().value == plc.types.TypeId.INT32.value


def test_allgather(streaming_engine) -> None:
    asyncio.run(_test_allgather(streaming_engine))


async def _test_allgather_reduce(engine) -> None:
    """Test allgather_reduce with single and multiple values."""
    context = engine.context
    comm = engine.comm

    # Test with a single value
    (result,) = await allgather_reduce(context, comm, 0, 42)
    assert result == 42  # Single rank, so sum is just the local value

    # Test with multiple values
    results = await allgather_reduce(context, comm, 1, 10, 20, 30)
    assert results == (10, 20, 30)  # Single rank, so sums are just the local values


def test_allgather_reduce(streaming_engine) -> None:
    asyncio.run(_test_allgather_reduce(streaming_engine))

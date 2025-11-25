# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF AllGather functionality."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
import rmm.mr

from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager


@pytest.fixture
def local_context() -> Context:
    """Fixture to create a single-GPU streaming context for testing."""
    options = Options(get_environment_variables())
    comm = single_process_comm(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)
    return Context(comm, br, options)


async def _test_allgather(context: Context) -> None:
    """Very simple test that AllGatherManager can concatenate tables."""
    stream = context.get_stream_from_pool()

    # Create simple test tables with different sizes
    tables = [
        plc.Table([plc.Column.from_array(np.full(num_elements, i).astype(np.int32))])  # type: ignore[call-arg]
        for i, num_elements in enumerate([100, 200, 300])
    ]

    # Insert tables into AllGatherManager
    allgather = AllGatherManager(context, 0)
    for i, table in enumerate(tables):
        allgather.insert(
            i, TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)
        )
    allgather.insert_finished()

    # Extract concatenated result
    result = await allgather.extract_concatenated(stream, ordered=True)

    # Verify the concatenated table has the expected shape
    assert result.num_rows() == 600  # 100 + 200 + 300
    assert result.num_columns() == 1

    # Verify the column type is correct
    col = result.columns()[0]
    assert col.size() == 600
    assert col.type().id().value == plc.types.TypeId.INT32.value


def test_allgather(local_context: Context) -> None:
    asyncio.run(_test_allgather(local_context))

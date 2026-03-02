# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming-engine using the SPMD Cluster style."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.streaming.core.actor import (
    run_actor_network,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import cudf_polars.experimental.rapidsmpf.collectives.shuffle
import cudf_polars.experimental.rapidsmpf.groupby
import cudf_polars.experimental.rapidsmpf.io
import cudf_polars.experimental.rapidsmpf.join
import cudf_polars.experimental.rapidsmpf.repartition
import cudf_polars.experimental.rapidsmpf.union  # noqa: F401
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    import polars as pl

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_pipeline_spmd_style(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline in SPMD mode.

    In SPMD mode every rank executes the same Python/Polars script
    independently.  Each rank owns its local DataFrames, which are
    treated as rank-local fragments of a larger distributed dataset and
    fed directly into the pipeline.  Collective operations (shuffles,
    all-gathers, etc.) coordinate across ranks to produce a globally
    consistent result.

    Parameters
    ----------
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        Executor configuration, including the rapidsmpf context and the
        Python thread-pool executor used to drive the actor network.
    stats
        The statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation
        IDs.
    collect_metadata
        Whether to collect runtime metadata.

    Returns
    -------
    The concatenated output DataFrame and, if ``collect_metadata`` is
    True, the list of channel metadata objects; otherwise ``None``.
    """
    assert config_options.executor.runtime == "rapidsmpf", "Runtime must be rapidsmpf"
    context: Context = config_options.executor.spmd["context"]
    py_executor: ThreadPoolExecutor = config_options.executor.spmd["py_executor"]

    # Create the IR execution context
    ir_context = IRExecutionContext(get_cuda_stream=context.get_stream_from_pool)

    # Generate network nodes
    metadata_collector: list[ChannelMetadata] | None = [] if collect_metadata else None

    nodes, output = generate_network(
        context,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
    )

    # Run the network
    run_actor_network(actors=nodes, py_executor=py_executor)

    # Extract/return the concatenated result.
    # Keep chunks alive until after concatenation to prevent
    # use-after-free with stream-ordered allocations
    messages = output.release()
    chunks = [
        TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        for msg in messages
    ]
    dfs: list[DataFrame] = []
    if chunks:
        dfs = [
            DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                chunk.stream,
            )
            for chunk in chunks
        ]
        df = _concat(*dfs, context=ir_context)
    else:
        # No chunks received - create an empty DataFrame with correct schema
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, context, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )

    result = df.to_polars()
    df.stream.synchronize()
    return result, metadata_collector

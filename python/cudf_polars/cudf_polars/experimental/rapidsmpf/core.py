# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation with the RAPIDS-MPF streaming engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import (
    run_streaming_pipeline,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import rmm

import cudf_polars.experimental.rapidsmpf.io
import cudf_polars.experimental.rapidsmpf.lower
import cudf_polars.experimental.rapidsmpf.nodes  # noqa: F401
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.traversal import CachingVisitor
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
    lower_ir_node,
)
from cudf_polars.experimental.rapidsmpf.rechunk import Rechunk
from cudf_polars.experimental.statistics import collect_statistics

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.leaf_node import DeferredMessages

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.experimental.rapidsmpf.dispatch import (
        GenState,
        LowerIRTransformer,
        LowerState,
        SubNetGenerator,
    )


def evaluate_logical_plan(ir: IR, config_options: ConfigOptions) -> DataFrame:
    """Evaluate a logical plan with RAPIDS-MPF."""
    assert config_options.executor.name == "streaming", "Executor must be streaming"
    assert config_options.executor.engine == "rapidsmpf", "Engine must be rapidsmpf"

    if config_options.executor.scheduler == "distributed":
        # TODO: Add distributed-execution support
        raise NotImplementedError(
            "The rapidsmpf engine does not support distributed execution yet."
        )

    # Collect statistics up-front on the client process (for now).
    stats = collect_statistics(ir, config_options)

    # Configure the context.
    # TODO: Multi-GPU version will be different. The rest of this function
    #       will be executed on each rank independently.
    # TODO: Need a way to configure options specific to the rapidmspf engine.
    options = Options(get_environment_variables())
    comm = new_communicator(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    br = BufferResource(mr)
    rmm.mr.set_current_device_resource(mr)
    ctx = Context(comm, br, options)
    executor = ThreadPoolExecutor(max_workers=1)

    # Lower the IR graph
    ir, partition_info = lower_ir_graph(ctx, ir, config_options, stats)

    # Generate network nodes
    nodes, output = generate_network(ctx, ir, partition_info, config_options)

    # Run the network
    run_streaming_pipeline(nodes=nodes, py_executor=executor)

    # Extract/return the result
    msgs = output.release()
    assert len(msgs) == 1, "Expected exactly one message"
    return DataFrame.from_table(
        TableChunk.from_message(msgs[0]).table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
    )


def lower_ir_graph(
    ctx: Context, ir: IR, config_options: ConfigOptions, stats: StatsCollector
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ctx
        The context.
    ir
        Root of the graph to rewrite.
    config_options
        GPUEngine configuration options.
    stats
        Statistics collector.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    state: LowerState = {
        "ctx": ctx,
        "config_options": config_options,
        "stats": stats,
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node, state=state)
    ir, partition_info = mapper(ir)

    # Ensure the output is always a single chunk
    ir = Rechunk(
        ir.schema,
        "single",
        ir,
    )
    partition_info[ir] = PartitionInfo(count=1)

    return ir, partition_info


def generate_network(
    ctx: Context,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[list[Any], DeferredMessages]:
    """
    Generate network nodes for a logical plan.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    """
    # Generate the network
    state: GenState = {
        "ctx": ctx,
        "config_options": config_options,
        "partition_info": partition_info,
    }
    mapper: SubNetGenerator = CachingVisitor(generate_ir_sub_network, state=state)
    node_mapping, channels = mapper(ir)
    nodes = [node for sublist in node_mapping.values() for node in sublist]
    ch_out = channels[ir]

    # Add final node to pull from the output channel
    output_node, output = pull_from_channel(ctx, ch_in=ch_out)
    nodes.append(output_node)

    # Return network and output hook
    return nodes, output

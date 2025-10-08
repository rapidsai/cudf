# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core RapidsMPF streaming-engine API."""

from __future__ import annotations

import asyncio
from collections import defaultdict
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
import cudf_polars.experimental.rapidsmpf.join
import cudf_polars.experimental.rapidsmpf.lower
import cudf_polars.experimental.rapidsmpf.shuffle
import cudf_polars.experimental.rapidsmpf.union  # noqa: F401
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.rapidsmpf.dispatch import lower_ir_node
from cudf_polars.experimental.rapidsmpf.nodes import generate_ir_sub_network_wrapper
from cudf_polars.experimental.rapidsmpf.repartition import Repartition
from cudf_polars.experimental.statistics import collect_statistics

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.leaf_node import DeferredMessages

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.experimental.rapidsmpf.dispatch import (
        GenState,
        LowerIRTransformer,
        LowerState,
        SubNetGenerator,
    )


def evaluate_logical_plan(ir: IR, config_options: ConfigOptions) -> DataFrame:
    """
    Evaluate a logical plan with the RapidsMPF streaming engine.

    Parameters
    ----------
    ir
        The IR node.
    config_options
        The configuration options.

    Returns
    -------
    The output DataFrame.
    """
    assert config_options.executor.name == "streaming", "Executor must be streaming"
    assert config_options.executor.engine == "rapidsmpf", "Engine must be rapidsmpf"

    if (
        config_options.executor.scheduler == "distributed"
    ):  # pragma: no cover; Requires distributed
        # TODO: Add distributed-execution support
        raise NotImplementedError(
            "The rapidsmpf engine does not support distributed execution yet."
        )

    # Lower the IR graph on the client process (for now).
    # NOTE: The `PartitionInfo.count` attribute is only used
    # for "guidance", because the number of chunks produced
    # by a streaming node may be dynamic. For now, we populate
    # and use the `count` attribute to trigger fallback
    # warnings in the lower_ir_graph call below.
    ir, partition_info = lower_ir_graph(ir, config_options)

    # Configure the context.
    # TODO: Multi-GPU version will be different. The rest of this function
    #       will be executed on each rank independently.
    # TODO: Need a way to configure options specific to the rapidmspf engine.
    options = Options(get_environment_variables())
    comm = new_communicator(options)
    # NOTE: Maybe use rmm.mr.CudaAsyncMemoryResource() by default
    # in callback.py instead of UVM (by default)?
    # mr = RmmResourceAdaptor(rmm.mr.get_current_device_resource())
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    br = BufferResource(mr)
    rmm.mr.set_current_device_resource(mr)
    ctx = Context(comm, br, options)
    executor = ThreadPoolExecutor(max_workers=1)

    # Generate network nodes
    nodes, output = generate_network(ctx, ir, partition_info, config_options)

    # Run the network
    run_streaming_pipeline(nodes=nodes, py_executor=executor)

    # Extract/return the result
    msgs = output.release()
    assert len(msgs) == 1, f"Expected exactly one output message, got {len(msgs)}"
    return DataFrame.from_table(
        TableChunk.from_message(msgs[0]).table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
    )


def lower_ir_graph(
    ir: IR,
    config_options: ConfigOptions,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function is nearly identical to the `lower_ir_graph` function
    in the `parallel` module, but with some differences:
    - A distinct `lower_ir_node` function is used.
    - A `Repartition` node is added to ensure a single chunk is produced.

    See Also
    --------
    lower_ir_node
    """
    state: LowerState = {
        "config_options": config_options,
        "stats": collect_statistics(ir, config_options),
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node, state=state)
    ir, partition_info = mapper(ir)

    # Ensure the output is always a single chunk
    bcasted = partition_info[ir].bcasted
    ir = Repartition(ir.schema, ir)
    partition_info[ir] = PartitionInfo(
        count=1,
        bcasted=bcasted,
    )

    return ir, partition_info


def generate_network(
    ctx: Context,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[list[Any], DeferredMessages]:
    """
    Translate the IR graph to a RapidsMPF streaming network.

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

    Returns
    -------
    The network nodes and output hook.
    """
    # Find IR nodes with multiple references.
    # We will need to multiply the output channel
    # for these nodes.
    output_ch_count: defaultdict[IR, int] = defaultdict(int)
    for node in traversal([ir]):
        for child in node.children:
            output_ch_count[child] += 1

    # IO Throttling
    max_io_threads = 3  # TODO: Make this configurable
    # TODO: Debug hang with value <3
    io_throttle = asyncio.Semaphore(max_io_threads)

    # Generate the network
    state: GenState = {
        "ctx": ctx,
        "config_options": config_options,
        "partition_info": partition_info,
        "output_ch_count": output_ch_count,
        "io_throttle": io_throttle,
    }
    mapper: SubNetGenerator = CachingVisitor(
        generate_ir_sub_network_wrapper, state=state
    )
    node_mapping, channels = mapper(ir)
    nodes = [node for sublist in node_mapping.values() for node in sublist]
    ch_out = channels[ir].pop()

    # TODO: If `ir` corresponds to broadcasted data, we can
    # inject a node to drain the channel and return an empty
    # DataFrame on ranks > 0. Eventually, we can push the
    # drain all the way down to the last AllGather node.

    # Add final node to pull from the output channel
    output_node, output = pull_from_channel(ctx, ch_in=ch_out)
    nodes.append(output_node)

    # Return network and output hook
    return nodes, output

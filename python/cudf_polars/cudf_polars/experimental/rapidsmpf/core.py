# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core RapidsMPF streaming-engine API."""

from __future__ import annotations

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
import cudf_polars.experimental.rapidsmpf.repartition
import cudf_polars.experimental.rapidsmpf.shuffle
import cudf_polars.experimental.rapidsmpf.union  # noqa: F401
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Join, Union
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.rapidsmpf.dispatch import FanoutInfo, lower_ir_node
from cudf_polars.experimental.rapidsmpf.nodes import generate_ir_sub_network_wrapper
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.leaf_node import DeferredMessages

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
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
    assert config_options.executor.runtime == "rapidsmpf", "Runtime must be rapidsmpf"

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
    mr = RmmResourceAdaptor(rmm.mr.get_current_device_resource())
    br = BufferResource(mr)
    rmm.mr.set_current_device_resource(mr)
    ctx = Context(comm, br, options)
    # TODO: Make this configurable.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpse")

    # Generate network nodes
    nodes, output = generate_network(ctx, ir, partition_info, config_options)

    # Run the network
    run_streaming_pipeline(nodes=nodes, py_executor=executor)

    # Extract/return the result
    return combine_output_chunks(
        ir,
        *(TableChunk.from_message(msg) for msg in output.release()),
    )


def combine_output_chunks(ir: IR, *chunks: TableChunk) -> DataFrame:
    """
    Combine the output chunks into a single DataFrame.

    Parameters
    ----------
    ir
        The IR node.
    chunks
        The output chunks.

    Returns
    -------
    The combined DataFrame, ordered by sequence number.
    """
    return _concat(
        *(
            DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                chunk.stream,
            )
            for chunk in sorted(chunks, key=lambda c: c.sequence_number)
        )
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
    return mapper(ir)


def determine_fanout_nodes(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> dict[IR, FanoutInfo]:
    """
    Determine which IR nodes need fanout and what type.

    Parameters
    ----------
    ir
        The root IR node.
    partition_info
        Partition information for each IR node.

    Returns
    -------
    Dictionary mapping IR nodes to FanoutInfo tuples where:
    - num_consumers: number of consumers
    - unbounded: whether the node needs unbounded fanout
    Only includes nodes that need fanout (i.e., have multiple consumers).
    """
    # Calculate output channel counts (number of consumers per node)
    output_ch_count: defaultdict[IR, int] = defaultdict(int)
    for node in traversal([ir]):
        for child in node.children:
            output_ch_count[child] += 1

    # Determine which nodes need unbounded fanout
    unbounded: set[IR] = set()

    def _mark_children_unbounded(node: IR) -> None:
        for child in node.children:
            unbounded.add(child)

    # Traverse the graph and identify nodes that need unbounded fanout
    for node in traversal([ir]):
        if node in unbounded:
            _mark_children_unbounded(node)
        elif isinstance(node, Union):
            # Union processes children sequentially, so all children
            # with multiple consumers need unbounded fanout
            _mark_children_unbounded(node)
        elif isinstance(node, Join):
            # This may be a broadcast join
            _mark_children_unbounded(node)
        elif len(node.children) > 1:
            # Check if this node is doing any broadcasting.
            # When we move to dynamic partitioning, we will need a
            # new way to indicate that a node is broadcasting 1+ children.
            counts = [partition_info[c].count for c in node.children]
            has_broadcast = any(c == 1 for c in counts) and not all(
                c == 1 for c in counts
            )
            if has_broadcast:
                # Broadcasting operation - children need unbounded fanout
                _mark_children_unbounded(node)

    # Build result dictionary: only include nodes with multiple consumers
    fanout_nodes: dict[IR, FanoutInfo] = {}
    for node, count in output_ch_count.items():
        if count > 1:
            fanout_nodes[node] = FanoutInfo(
                num_consumers=count,
                unbounded=node in unbounded,
            )

    return fanout_nodes


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
    # Determine which nodes need fanout
    fanout_nodes = determine_fanout_nodes(ir, partition_info)

    # Generate the network
    state: GenState = {
        "ctx": ctx,
        "config_options": config_options,
        "partition_info": partition_info,
        "fanout_nodes": fanout_nodes,
    }
    mapper: SubNetGenerator = CachingVisitor(
        generate_ir_sub_network_wrapper, state=state
    )
    nodes, channels = mapper(ir)
    ch_out = channels[ir].reserve_output_slot()

    # TODO: We will need an additional node here to drain
    # the metadata channel once we start plumbing metadata
    # through the network. This node could also drop
    # "duplicated" data on all but rank 0.

    # Add final node to pull from the output data channel
    # (metadata channel is unused)
    output_node, output = pull_from_channel(ctx, ch_in=ch_out.data)
    nodes.append(output_node)

    # Return network and output hook
    return nodes, output

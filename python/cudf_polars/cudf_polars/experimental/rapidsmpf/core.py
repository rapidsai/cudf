# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core RapidsMPF streaming-engine API."""

from __future__ import annotations

import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.actor import (
    run_actor_network,
)
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

import cudf_polars.experimental.rapidsmpf.collectives.shuffle
import cudf_polars.experimental.rapidsmpf.collectives.sort
import cudf_polars.experimental.rapidsmpf.groupby
import cudf_polars.experimental.rapidsmpf.io
import cudf_polars.experimental.rapidsmpf.join
import cudf_polars.experimental.rapidsmpf.repartition
import cudf_polars.experimental.rapidsmpf.union
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import (
    DataFrameScan,
    IRExecutionContext,
    Join,
    Scan,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.rapidsmpf.collectives import ReserveOpIDs
from cudf_polars.experimental.rapidsmpf.dispatch import FanoutInfo
from cudf_polars.experimental.rapidsmpf.nodes import (
    generate_ir_sub_network_wrapper,
    metadata_drain_node,
)
from cudf_polars.experimental.rapidsmpf.tracing import log_query_plan
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.statistics import collect_statistics

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.leaf_actor import DeferredMessages
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    import polars as pl

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.experimental.rapidsmpf.dispatch import (
        GenState,
        SubNetGenerator,
    )
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_logical_plan(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a logical plan with the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    config_options
        The configuration options.
    collect_metadata
        Whether to collect runtime metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    query_id = uuid.uuid4()

    with cudf_polars.dsl.tracing.bound_contextvars(
        cudf_polars_query_id=str(query_id),
    ):
        match config_options.executor.cluster:
            case "spmd":
                from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
                    evaluate_pipeline_spmd_mode,
                )

                result, metadata_collector = evaluate_pipeline_spmd_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "ray":
                from cudf_polars.experimental.rapidsmpf.frontend.ray import (
                    evaluate_pipeline_ray_mode,
                )

                result, metadata_collector = evaluate_pipeline_ray_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "dask":
                from cudf_polars.experimental.rapidsmpf.frontend.dask import (
                    evaluate_pipeline_dask_mode,
                )

                result, metadata_collector = evaluate_pipeline_dask_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "default_singleton":
                # Single-process execution: lower and run locally. Reuse the
                # process-wide ``DefaultSingletonEngine`` so the rapidsmpf
                # Context, RMM adaptor, and py-executor persist across
                # ``.collect()`` calls instead of being rebuilt per query.
                from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
                    DefaultSingletonEngine,
                )

                engine = DefaultSingletonEngine.create_or_get()
                stats = collect_statistics(ir, config_options)
                ir, partition_info = lower_ir_graph(ir, config_options, stats)
                with ReserveOpIDs(ir, config_options) as collective_id_map:
                    log_query_plan(ir, config_options)
                    result, metadata_collector = evaluate_pipeline(
                        ir,
                        partition_info,
                        config_options,
                        stats,
                        collective_id_map,
                        engine.comm,
                        engine.context,
                        collect_metadata=collect_metadata,
                        query_id=query_id,
                    )
            case other:
                raise ValueError(f"Unknown cluster mode: {other}")

    return result, metadata_collector


def evaluate_pipeline(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    comm: Communicator,
    rmpf_context: Context,
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline.

    Parameters
    ----------
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        The statistics collector.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    comm
        The communicator describing the participating processes.
    rmpf_context
        The RapidsMPF context.
    collect_metadata
        Whether to collect runtime metadata.
    query_id
        A unique identifier for the query.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    br = rmpf_context.br()

    # Create the IR execution context (always uses the rapidsmpf stream pool).
    ir_context = IRExecutionContext(
        get_cuda_stream=rmpf_context.get_stream_from_pool, query_id=query_id
    )

    # Generate actor graph nodes.
    metadata_collector: list[ChannelMetadata] | None = [] if collect_metadata else None
    nodes, output = generate_network(
        rmpf_context,
        comm,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
    )

    # Run the network.
    with ThreadPoolExecutor(
        max_workers=config_options.executor.num_py_executors,
        thread_name_prefix="cpse",
    ) as executor:
        run_actor_network(actors=nodes, py_executor=executor)

    # Extract/return the concatenated result. Keep chunks alive until after
    # concatenation to prevent use-after-free with stream-ordered allocations.
    messages = output.release()
    chunks = [
        TableChunk.from_message(msg, br=br).make_available_and_spill(
            br, allow_overbooking=True
        )
        for msg in messages
    ]
    if chunks:
        col_names = list(ir.schema.keys())
        col_dtypes = list(ir.schema.values())
        dfs = [
            DataFrame.from_table(
                chunk.table_view(), col_names, col_dtypes, chunk.stream
            )
            for chunk in chunks
        ]
        if len(dfs) == 1:
            df = dfs[0]
        else:
            with ir_context.stream_ordered_after(*dfs) as stream:
                df = DataFrame.from_table(
                    plc.concatenate.concatenate([d.table for d in dfs], stream=stream),
                    col_names,
                    col_dtypes,
                    stream,
                )
    else:
        # No chunks received - create an empty DataFrame with correct schema.
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, rmpf_context, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )

    result = df.to_polars()

    return result, metadata_collector


def determine_fanout_nodes(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir_dep_count: defaultdict[IR, int],
) -> dict[IR, FanoutInfo]:
    """
    Determine which IR nodes need fanout and what type.

    Parameters
    ----------
    ir
        The root IR node.
    partition_info
        Partition information for each IR node.
    ir_dep_count
        The number of IR dependencies for each IR node.

    Returns
    -------
    Dictionary mapping IR nodes to FanoutInfo tuples where:
    - num_consumers: number of consumers
    - unbounded: whether the node needs unbounded fanout
    Only includes nodes that need fanout (i.e., have multiple consumers).
    """
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
    for node, count in ir_dep_count.items():
        if count > 1:
            fanout_nodes[node] = FanoutInfo(
                num_consumers=count,
                unbounded=node in unbounded,
            )

    return fanout_nodes


def generate_network(
    context: Context,
    comm: Communicator,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    *,
    ir_context: IRExecutionContext,
    collective_id_map: dict[IR, list[int]],
    metadata_collector: list[ChannelMetadata] | None,
) -> tuple[list[Any], DeferredMessages]:
    """
    Translate the IR graph to a RapidsMPF streaming network.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator the network generation is collective over.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        Statistics collector.
    ir_context
        The execution context for the IR node.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.

    Returns
    -------
    The network nodes and output hook.
    """
    # Count the number of IO nodes and the number of IR dependencies
    num_io_nodes: int = 0
    ir_dep_count: defaultdict[IR, int] = defaultdict(int)
    for node in traversal([ir]):
        if isinstance(node, (DataFrameScan, Scan)):
            num_io_nodes += 1
        for child in node.children:
            ir_dep_count[child] += 1

    # Determine which nodes need fanout
    fanout_nodes = determine_fanout_nodes(ir, partition_info, ir_dep_count)

    # Get max_io_threads from config (default: 2)
    max_io_threads_global = config_options.executor.max_io_threads
    max_io_threads_local = max(1, max_io_threads_global // max(1, num_io_nodes))

    # Generate the network
    state: GenState = {
        "context": context,
        "comm": comm,
        "config_options": config_options,
        "partition_info": partition_info,
        "fanout_nodes": fanout_nodes,
        "ir_context": ir_context,
        "max_io_threads": max_io_threads_local,
        "stats": stats,
        "collective_id_map": collective_id_map,
    }
    mapper: SubNetGenerator = CachingVisitor(
        generate_ir_sub_network_wrapper, state=state
    )
    nodes_dict, channels = mapper(ir)
    ch_out = channels[ir].reserve_output_slot()

    # Add node to drain metadata before pull_from_channel
    # (since pull_from_channel doesn't handle metadata messages)
    ch_final_data: Channel[TableChunk] = context.create_channel()
    drain_node = metadata_drain_node(
        context,
        comm,
        ir,
        ir_context,
        ch_out,
        ch_final_data,
        metadata_collector,
    )

    # Add final node to pull from the output data channel
    output_node, output = pull_from_channel(context, ch_in=ch_final_data)

    # Flatten the nodes dictionary into a list for run_actor_network
    nodes: list[Any] = [node for node_list in nodes_dict.values() for node in node_list]
    nodes.extend([drain_node, output_node])

    # Return network and output hook
    return nodes, output

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RAPIDS-MPF streaming-network evaluation."""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from cuda.bindings import runtime as cudart
from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import (
    define_py_node,
    run_streaming_pipeline,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
import rmm

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
)
from cudf_polars.dsl.traversal import CachingVisitor
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import (
    generate_ir_sub_network,
    lower_ir_node,
    lower_ir_node_rapidsmpf,
)
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, MutableMapping

    from rapidsmpf.streaming.core.leaf_node import DeferredMessages

    from cudf_polars.experimental.dispatch import LowerIRTransformer, State
    from cudf_polars.experimental.parallel import ConfigOptions


class GenState(TypedDict):
    """
    State used for generating a streaming sub-network.

    Parameters
    ----------
    ctx
        The rapidsmpf context.
    config_options
        GPUEngine configuration options.
    partition_info
        Partition information.
    """

    ctx: Context
    config_options: ConfigOptions
    partition_info: MutableMapping[IR, PartitionInfo]


GenerateNetworkTransformer: TypeAlias = GenericTransformer[
    "IR", "tuple[dict[IR, list[Any]], dict[IR, Any]]", GenState
]
"""Protocol for Generating a sub-network."""


def evaluate_logical_plan(ir: IR, config_options: ConfigOptions) -> DataFrame:
    """Evaluate a logical plan with RAPIDS-MPF."""
    # Configure the context
    # TODO: Mulit-GPU version may be very different.
    options = Options(get_environment_variables())
    comm = new_communicator(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    br = BufferResource(mr)
    rmm.mr.set_current_device_resource(mr)
    ctx = Context(comm, br, options)
    executor = ThreadPoolExecutor(max_workers=1)

    # Lower the IR graph
    ir, partition_info = lower_ir_graph_rapidsmpf(ir, config_options)

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


@lower_ir_node_rapidsmpf.register(IR)
def _(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    # Default logic - Use ``lower_ir_node``, but
    # many IR types will need different logic.
    return lower_ir_node(ir, rec)


@lower_ir_node_rapidsmpf.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node_rapidsmpf'"
    )

    # TODO: Handle multiple workers.
    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)
    return ir, {ir: PartitionInfo(count=count)}


def lower_ir_graph_rapidsmpf(
    ir: IR, config_options: ConfigOptions
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
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    state: State = {
        "config_options": config_options,
        "stats": collect_statistics(ir, config_options),
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node_rapidsmpf, state=state)
    return mapper(ir)


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
    mapper: GenerateNetworkTransformer = CachingVisitor(
        generate_ir_sub_network, state=state
    )
    node_mapping, channels = mapper(ir)
    nodes = list(node_mapping.values())
    ch_out = channels[ir]

    # Add node to concatenate final chunks
    choncat_ch_out = Channel()
    nodes.append(concatenate(ctx, ch_in=ch_out, ch_out=choncat_ch_out))
    ch_out = choncat_ch_out

    # Add final node to pull from the output channel
    output_node, output = pull_from_channel(ctx, ch_in=ch_out)
    nodes.append(output_node)

    # Return network and output hook
    return nodes, output


@asynccontextmanager
async def shutdown_on_error(
    ctx: Context, *channels: Channel[Any]
) -> AsyncIterator[None]:
    """
    Shutdown on error for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    channels
        The channels to shutdown.
    """
    try:
        yield
    except Exception:
        await asyncio.gather(*(ch.shutdown(ctx) for ch in channels))
        raise


@define_py_node()
async def pointwise_single_channel_node(
    ctx: Context,
    ir: IR,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    partition_info: PartitionInfo,
) -> None:
    """
    Pointwise single node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_in
        The input channel.
    ch_out
        The output channel.
    partition_info
        The partition information.
    """
    async with shutdown_on_error(ctx, ch_in, ch_out):
        while (msg := await ch_in.recv(ctx)) is not None:
            # Receive an input chunk
            chunk: TableChunk = TableChunk.from_message(msg).to_device(ctx.br())

            # Evaluate the IR node
            df: DataFrame = ir.do_evaluate(
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(ir.schema.keys()),
                    list(ir.schema.values()),
                ),
            )

            # Return the output chunk
            chunk = TableChunk.from_pylibcudf_table(
                chunk.sequence_number,
                df.table,
                chunk.stream,
            )
            await ch_out.send(ctx, Message(chunk))
        await ch_out.drain(ctx)


@define_py_node()
async def concatenate(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    """
    Concatenate node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_in
        The input channel.
    ch_out
        The output channel.
    """
    async with shutdown_on_error(ctx, ch_in, ch_out):
        chunks = []
        build_stream = ctx.get_stream()
        err, event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(err)
        chunk_streams = set()
        while (msg := await ch_in.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg).to_device(ctx.br())
            chunks.append(chunk)
            (err,) = cudart.cudaEventRecord(event, chunk.stream.value())
            chunk_streams.add(chunk.stream)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(err)
            (err,) = cudart.cudaStreamWaitEvent(
                build_stream.value(), event, cudart.cudaEventWaitDefault
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(err)
        table = plc.concatenate.concatenate(
            [chunk.table_view() for chunk in chunks], build_stream
        )
        (err,) = cudart.cudaEventRecord(event, build_stream.value())
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(err)
        for s in chunk_streams:
            (err,) = cudart.cudaStreamWaitEvent(
                s.value(), event, cudart.cudaEventWaitDefault
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(err)
        await ch_out.send(
            ctx, Message(TableChunk.from_pylibcudf_table(0, table, build_stream))
        )
        await ch_out.drain(ctx)


@define_py_node()
async def dataframe_scan_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    ir: DataFrameScan,
    rows_per_partition: int,
) -> None:
    """
    DataFrame scan node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_out
        The output channel.
    ir
        The DataFrameScan node.
    rows_per_partition
        The number of rows per partition.
    """
    # TODO: Use (throttled) thread pool
    # TODO: Use multiple streams
    nrows = max(ir.df.shape()[0], 1)
    stream = ctx.get_stream()
    async with shutdown_on_error(ctx, ch_out):
        for seq_num, offset in enumerate(range(0, nrows, rows_per_partition)):
            ir_slice = DataFrameScan(
                ir.schema,
                ir.df.slice(offset, rows_per_partition),
                ir.projection,
            )

            # Evaluate the IR node
            df: DataFrame = ir_slice.do_evaluate(*ir_slice._non_child_args)

            # Return the output chunk
            chunk = TableChunk.from_pylibcudf_table(seq_num, df.table, stream)
            await ch_out.send(ctx, Message(chunk))
        await ch_out.drain(ctx)


@generate_ir_sub_network.register(IR)
def _(
    ir: IR, rec: GenerateNetworkTransformer
) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Process children
    if len(ir.children) != 1:
        raise NotImplementedError(f"Unsupported IR node for rapidsmpf: {type(ir)}.")
    nodes, channels = rec(ir.children[0])

    # Create output channel
    channels[ir] = Channel()

    # Add simple python node
    nodes[ir] = [
        pointwise_single_channel_node(
            rec.state["ctx"],
            ir,
            channels[ir.children[0]],
            channels[ir],
            rec.state["partition_info"],
        )
    ]

    return nodes, channels


@generate_ir_sub_network.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: GenerateNetworkTransformer
) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    rows_per_partition = config_options.executor.max_rows_per_partition

    ctx = rec.state["ctx"]
    ch_out = Channel()
    nodes: dict[IR, list[Any]] = {
        ir: [dataframe_scan_node(ctx, ch_out, ir, rows_per_partition)]
    }
    channels: dict[IR, Any] = {ir: ch_out}
    return nodes, channels

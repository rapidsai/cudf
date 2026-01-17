# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Cache, Empty, Filter, Projection
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    empty_table_chunk,
    opaque_reservation,
    process_children,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@define_py_node()
async def default_node_single(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    *,
    preserve_partitioning: bool = False,
) -> None:
    """
    Single-channel default node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_in
        The input ChannelPair.
    preserve_partitioning
        Whether to preserve the partitioning metadata of the input chunks.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Recv/send metadata.
        metadata_in = await ch_in.recv_metadata(context)
        metadata_out = Metadata(
            metadata_in.count,
            partitioned_on=metadata_in.partitioned_on if preserve_partitioning else (),
            duplicated=metadata_in.duplicated,
        )
        await ch_out.send_metadata(context, metadata_out)

        # Recv/send data.
        seq_num = 0
        receiving = True
        received_any = False
        while receiving:
            msg = await ch_in.data.recv(context)
            if msg is None:
                receiving = False
                if received_any:
                    break
                else:
                    # Make sure we have an empty chunk in case do_evaluate
                    # always produces rows (e.g. aggregation)
                    stream = ir_context.get_cuda_stream()
                    chunk = empty_table_chunk(ir.children[0], context, stream)
            else:
                received_any = True
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                seq_num = msg.sequence_number
            del msg

            input_bytes = chunk.data_alloc_size(MemoryType.DEVICE)
            with opaque_reservation(context, input_bytes):
                df = await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    DataFrame.from_table(
                        chunk.table_view(),
                        list(ir.children[0].schema.keys()),
                        list(ir.children[0].schema.values()),
                        chunk.stream,
                    ),
                    context=ir_context,
                )
                await ch_out.data.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            df.table, chunk.stream, exclusive_view=True
                        ),
                    ),
                )
                del df, chunk

        await ch_out.data.drain(context)


@define_py_node()
async def default_node_multi(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    chs_in: tuple[ChannelPair, ...],
    *,
    partitioning_index: int | None = None,
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    chs_in
        Tuple of input ChannelPairs.
    partitioning_index
        Index of the input channel to preserve partitioning information for.
        If None, no partitioning information is preserved.
    """
    async with shutdown_on_error(
        context,
        *[ch.metadata for ch in chs_in],
        ch_out.metadata,
        *[ch.data for ch in chs_in],
        ch_out.data,
    ):
        # Merge and forward basic metadata.
        metadata = Metadata(1)
        for idx, ch_in in enumerate(chs_in):
            md_child = await ch_in.recv_metadata(context)
            metadata.count = max(md_child.count, metadata.count)
            metadata.duplicated = metadata.duplicated and md_child.duplicated
            if idx == partitioning_index:
                metadata.partitioned_on = md_child.partitioned_on
        await ch_out.send_metadata(context, metadata)

        seq_num = 0
        n_children = len(chs_in)
        finished_channels: set[int] = set()
        # Store TableChunk objects to keep data alive and prevent use-after-free
        # with stream-ordered allocations
        ready_chunks: list[TableChunk | None] = [None] * n_children
        chunk_count: list[int] = [0] * n_children

        # Recv/send data.
        while True:
            # Receive from all non-finished channels
            for ch_idx, ch_in in enumerate(chs_in):
                if ch_idx in finished_channels:
                    continue  # This channel already finished, reuse its data

                msg = await ch_in.data.recv(context)
                if msg is None:
                    # Channel finished - keep its last chunk for reuse
                    finished_channels.add(ch_idx)
                else:
                    # Store the new chunk (replacing previous if any)
                    ready_chunks[ch_idx] = TableChunk.from_message(msg)
                    chunk_count[ch_idx] += 1
                del msg

            # If all channels finished, we're done
            if len(finished_channels) == n_children:
                break

            # Check if any channel drained without providing data.
            # If so, create an empty chunk for that channel.
            for ch_idx, child in enumerate(ir.children):
                if ready_chunks[ch_idx] is None:
                    # Channel drained without data - create empty chunk
                    stream = ir_context.get_cuda_stream()
                    ready_chunks[ch_idx] = empty_table_chunk(child, context, stream)

            # Ensure all table chunks are unspilled and available.
            ready_chunks = [
                chunk.make_available_and_spill(context.br(), allow_overbooking=True)
                for chunk in cast(list[TableChunk], ready_chunks)
            ]
            dfs = [
                DataFrame.from_table(
                    chunk.table_view(),  # type: ignore[union-attr]
                    list(child.schema.keys()),
                    list(child.schema.values()),
                    chunk.stream,  # type: ignore[union-attr]
                )
                for chunk, child in zip(ready_chunks, ir.children, strict=True)
            ]

            input_bytes = sum(
                chunk.data_alloc_size(MemoryType.DEVICE)
                for chunk in cast(list[TableChunk], ready_chunks)
            )
            with opaque_reservation(context, input_bytes):
                df = await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *dfs,
                    context=ir_context,
                )
                await ch_out.data.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            df.table,
                            df.stream,
                            exclusive_view=True,
                        ),
                    ),
                )
                seq_num += 1
                del df, dfs

        # Drain the output channel
        del ready_chunks
        await ch_out.data.drain(context)


@define_py_node()
async def fanout_metadata_node(
    context: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Metadata-only fanout node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.

    Notes
    -----
    This node only handles metadata forwarding - it receives metadata
    from the input channel and broadcasts it to all output channels.
    The actual data fanout is handled by the native rapidsmpf fanout node.
    """
    async with shutdown_on_error(
        context,
        ch_in.metadata,
        *[ch.metadata for ch in chs_out],
    ):
        # Forward metadata to all outputs.
        metadata = await ch_in.recv_metadata(context)
        await asyncio.gather(*(ch.send_metadata(context, metadata) for ch in chs_out))


@generate_ir_sub_network.register(IR)
def _(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if len(ir.children) == 1:
        # Single-channel default node
        preserve_partitioning = isinstance(
            # TODO: We don't need to worry about
            # non-pointwise Filter operations here,
            # because the lowering stage would have
            # collapsed to one partition anyway.
            ir,
            (Cache, Projection, Filter),
        )
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
                preserve_partitioning=preserve_partitioning,
            )
        ]
    else:
        # Multi-channel default node
        nodes[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
            )
        ]

    return nodes, channels


@define_py_node()
async def empty_node(
    context: Context,
    ir: Empty,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
) -> None:
    """
    Empty node for rapidsmpf - produces a single empty chunk.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Empty node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    """
    async with shutdown_on_error(context, ch_out.metadata, ch_out.data):
        # Send metadata indicating a single empty chunk
        await ch_out.send_metadata(context, Metadata(1, duplicated=True))

        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args, context=ir_context)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True
        )
        await ch_out.data.send(context, Message(0, chunk))

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Empty)
def _(
    ir: Empty, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: dict[IR, list[Any]] = {
        ir: [empty_node(context, ir, ir_context, channels[ir].reserve_input_slot())]
    }
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output ChannelManager.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)

    # Check if this node needs fanout
    if (fanout_info := rec.state["fanout_nodes"].get(ir)) is not None:
        count = fanout_info.num_consumers
        manager = ChannelManager(rec.state["context"], count=count)

        # Get input/output channel pairs
        ch_in = channels[ir].reserve_output_slot()
        chs_out = [manager.reserve_input_slot() for _ in range(count)]

        # Use native rapidsmpf fanout for data channels
        data_fanout_node = fanout(
            rec.state["context"],
            ch_in.data,
            [ch.data for ch in chs_out],
            FanoutPolicy.UNBOUNDED if fanout_info.unbounded else FanoutPolicy.BOUNDED,
        )

        # Use metadata-only fanout node for metadata channels
        metadata_node = fanout_metadata_node(
            rec.state["context"],
            ch_in,
            *chs_out,
        )

        nodes[ir].extend([data_fanout_node, metadata_node])
        channels[ir] = manager
    return nodes, channels


@define_py_node()
async def metadata_feeder_node(
    context: Context,
    channel: ChannelPair,
    metadata: Metadata,
) -> None:
    """
    Feed metadata to a channel pair.

    Parameters
    ----------
    context
        The rapidsmpf context.
    channel
        The channel pair.
    metadata
        The metadata to feed.
    """
    async with shutdown_on_error(context, channel.metadata, channel.data):
        await channel.send_metadata(context, metadata)


@define_py_node()
async def metadata_drain_node(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_in: ChannelPair,
    ch_out: Any,
    metadata_collector: list[Metadata] | None,
) -> None:
    """
    Drain metadata and forward data to a single channel.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        The input ChannelPair (with metadata and data channels).
    ch_out
        The output data channel.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.
    """
    async with shutdown_on_error(context, ch_in.metadata, ch_in.data, ch_out):
        # Drain metadata channel (we don't need it after this point)
        metadata = await ch_in.recv_metadata(context)
        send_empty = metadata.duplicated and context.comm().rank != 0
        if metadata_collector is not None:
            metadata_collector.append(metadata)

        # Forward non-duplicated data messages
        while (msg := await ch_in.data.recv(context)) is not None:
            if not send_empty:
                await ch_out.send(context, msg)

        # Send empty data if needed
        if send_empty:
            stream = ir_context.get_cuda_stream()
            await ch_out.send(
                context, Message(0, empty_table_chunk(ir, context, stream))
            )

        await ch_out.drain(context)

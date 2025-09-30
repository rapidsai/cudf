# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Helper utilities for the RAPIDS-MPF streaming engine."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo


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
            chunk: TableChunk = TableChunk.from_message(msg)

            # Evaluate the IR node
            df: DataFrame = ir.do_evaluate(
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(ir.children[0].schema.keys()),
                    list(ir.children[0].schema.values()),
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
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, ch_in, ch_out):
        chunks = []
        build_stream = DEFAULT_STREAM
        while (msg := await ch_in.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg)
            chunks.append(chunk)

        table = (
            chunks[0].table_view()
            if len(chunks) == 1
            else plc.concatenate.concatenate(
                [chunk.table_view() for chunk in chunks], build_stream
            )
        )
        await ch_out.send(
            ctx, Message(TableChunk.from_pylibcudf_table(0, table, build_stream))
        )
        await ch_out.drain(ctx)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Window over() actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.ir import IR, HStack, Select
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    NormalizedPartitioning,
    _make_hash_shuffle_metadata,
    chunkwise_evaluate,
    evaluate_chunk,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.utils import _extract_over_shuffle_indices

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


@define_actor()
async def over_window_actor(
    context: Context,
    comm: Communicator,
    ir: Select | HStack,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> None:
    """
    Streaming actor for window over() expressions.

    Strategy selection based on observed data:
    - Chunk-wise: Data already partitioned on the over() keys
    - Shuffle: Hash-shuffle by partition-by keys, then local evaluation

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Select or HStack IR node containing GroupedWindow expressions.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    collective_id
        The collective ID for the shuffle operation.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        exprs = [e.value for e in (ir.exprs if isinstance(ir, Select) else ir.columns)]
        key_indices = _extract_over_shuffle_indices(exprs, ir.children[0].schema)
        assert key_indices is not None
        assert len(key_indices) > 0

        partitioning = NormalizedPartitioning.from_indices(
            metadata_in.partitioning,
            comm.nranks,
            indices=key_indices,
            allow_subset=False,
        )
        if partitioning:
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context, ir, ir_context, ch_out, ch_in, metadata_out, tracer=tracer
            )
            return

        modulus = max(comm.nranks, metadata_in.local_count)
        metadata_out = _make_hash_shuffle_metadata(
            comm, key_indices, modulus, metadata_in
        )
        await send_metadata(ch_out, context, metadata_out)

        shuffle = ShuffleManager(context, comm, modulus, collective_id)

        # TODO: The hash shuffle does not preserve the original input row order.
        while (msg := await ch_in.recv(context)) is not None:
            shuffle.insert_hash(
                TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
                    context.br(), allow_overbooking=True
                ),
                key_indices,
            )

        await shuffle.insert_finished()

        for partition_id in shuffle.local_partitions():
            stream = ir_context.get_cuda_stream()
            partition_chunk = TableChunk.from_pylibcudf_table(
                shuffle.extract_chunk(partition_id, stream),
                stream,
                exclusive_view=True,
                br=context.br(),
            )
            result = await evaluate_chunk(
                context, partition_chunk, ir, ir_context=ir_context
            )
            if tracer is not None:
                tracer.add_chunk(table=result.table_view())
            await ch_out.send(context, Message(partition_id, result))

        await ch_out.drain(context)


@generate_ir_sub_network.register(Select)
@generate_ir_sub_network.register(HStack)
def _(
    ir: Select | HStack, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]

    if config_options.executor.dynamic_planning is None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    exprs = [e.value for e in (ir.exprs if isinstance(ir, Select) else ir.columns)]
    indices = _extract_over_shuffle_indices(exprs, ir.children[0].schema)

    if not (indices is not None and len(indices) > 0):
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    actors[ir] = [
        over_window_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_ids.pop(),
        )
    ]
    return actors, channels

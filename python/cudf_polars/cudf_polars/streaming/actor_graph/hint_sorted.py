# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming actor for ``MapFunction("hint_sorted")``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import IR, MapFunction
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.sort import (
    extract_orderscheme_partitioning,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.utils import sorting
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.utils import ChunkStore


HintSortedOptions: TypeAlias = tuple[
    tuple[str, ...], tuple[bool, ...], tuple[bool, ...]
]


def _hint_sorted_options(
    ir: MapFunction,
) -> HintSortedOptions:
    """Return normalized ``hint_sorted`` options."""
    assert ir.name == "hint_sorted"
    return cast("HintSortedOptions", ir.options)


def _hint_sorted_order_keys(ir: MapFunction) -> list[OrderKey]:
    """Convert ``MapFunction("hint_sorted")`` options to ordering keys."""
    column_names, descending, nulls_last = _hint_sorted_options(ir)
    orders, null_orders = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(column_names)
    )
    return [
        OrderKey(index, order, null_order)
        for index, order, null_order in zip(
            names_to_indices(column_names, ir.schema),
            orders,
            null_orders,
            strict=True,
        )
    ]


def _order_scheme_has_keys(scheme: OrderScheme, keys: list[OrderKey]) -> bool:
    """Check for an exact ordering match."""
    return any(list(ordering.keys) == keys for ordering in scheme.orderings)


def _metadata_satisfies_hint(metadata: ChannelMetadata, keys: list[OrderKey]) -> bool:
    """Check whether existing metadata already advertises the requested ordering."""
    if metadata.partitioning is None:
        return False
    scheme = metadata.partitioning.inter_rank
    return isinstance(scheme, OrderScheme) and _order_scheme_has_keys(scheme, keys)


def _with_extracted_ordering(
    metadata: ChannelMetadata, partitioning: Partitioning | None
) -> ChannelMetadata:
    """Return metadata with extracted ordering when extraction succeeded."""
    if partitioning is None:
        return metadata

    orderings: list[Ordering] = []
    local = partitioning.local
    if metadata.partitioning is not None:
        local = metadata.partitioning.local
        if isinstance(metadata.partitioning.inter_rank, OrderScheme):
            orderings.extend(metadata.partitioning.inter_rank.orderings)

    scheme = partitioning.inter_rank
    if isinstance(scheme, OrderScheme):
        orderings.extend(scheme.orderings)

    return ChannelMetadata(
        local_count=metadata.local_count,
        partitioning=Partitioning(OrderScheme(orderings), local),
        duplicated=metadata.duplicated,
    )


def _trivial_ordering_metadata(
    context: Context,
    comm: Communicator,
    ir: MapFunction,
    metadata: ChannelMetadata,
    keys: list[OrderKey],
) -> ChannelMetadata | None:
    """Attach ordering without scanning data when the boundaries are trivial."""
    if comm.nranks != 1 or metadata.local_count > 1:
        return None

    partitioning = metadata.partitioning
    existing_orderings: list[Ordering] = []
    local = "inherit"
    if partitioning is not None:
        local = partitioning.local
        if isinstance(partitioning.inter_rank, OrderScheme):
            existing_orderings = list(partitioning.inter_rank.orderings)

    column_names = _hint_sorted_options(ir)[0]
    stream = context.br().stream_pool.get_stream()
    boundaries = TableChunk.from_pylibcudf_table(
        plc.Table(
            [make_empty_column(ir.schema[name], stream) for name in column_names]
        ),
        stream,
        exclusive_view=False,
        br=context.br(),
    )
    ordering = Ordering(keys, boundaries, strict_boundaries=True)
    return ChannelMetadata(
        local_count=metadata.local_count,
        partitioning=Partitioning(
            OrderScheme([*existing_orderings, ordering]),
            local,
        ),
        duplicated=metadata.duplicated,
    )


async def extract_hint_sorted_metadata(
    context: Context,
    comm: Communicator,
    ir: MapFunction,
    ir_context: IRExecutionContext,
    metadata: ChannelMetadata,
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> tuple[ChannelMetadata, ChunkStore | None]:
    """Resolve output metadata and any chunks consumed during extraction."""
    keys = _hint_sorted_order_keys(ir)
    if not keys or _metadata_satisfies_hint(metadata, keys):
        return metadata, None

    trivial_metadata = _trivial_ordering_metadata(context, comm, ir, metadata, keys)
    if trivial_metadata is not None:
        return trivial_metadata, None

    result = await extract_orderscheme_partitioning(
        context,
        comm,
        ir,
        ir_context,
        ch_in,
        keys,
        collective_id,
    )
    return _with_extracted_ordering(metadata, result.partitioning), result.chunks


@define_actor()
async def hint_sorted_actor(
    context: Context,
    comm: Communicator,
    ir: MapFunction,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> None:
    """Forward data and attach safe ordering metadata for ``hint_sorted``."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        metadata = await recv_metadata(ch_in, context)
        metadata, buffered_chunks = await extract_hint_sorted_metadata(
            context,
            comm,
            ir,
            ir_context,
            metadata,
            ch_in,
            collective_id,
        )
        await send_metadata(ch_out, context, metadata)
        if buffered_chunks is None:
            while (msg := await ch_in.recv(context)) is not None:
                await ch_out.send(context, msg)
        else:
            for msg in buffered_chunks:
                await ch_out.send(context, msg)
        await ch_out.drain(context)


@generate_ir_sub_network.register(MapFunction)
def _(
    ir: MapFunction, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    if ir.name != "hint_sorted":
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    nodes[ir] = [
        hint_sorted_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id,
        )
    ]
    return nodes, channels

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming actor for ``MapFunction("hint_sorted")``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

from cudf_streaming.channel_metadata import OrderKey
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import IR, MapFunction
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.sort import (
    ensure_ordering_metadata,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    process_children,
    shutdown_on_error,
)
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


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
        await ensure_ordering_metadata(
            context,
            comm,
            ir,
            ir_context,
            ch_in,
            ch_out,
            _hint_sorted_order_keys(ir),
            collective_id,
        )


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

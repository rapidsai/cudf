# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.ir import Union
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.nodes import define_actor, shutdown_on_error
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkStore,
    allgather_reduce,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.utils import partition_owner, partition_range

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.core import SubNetGenerator


@dataclass(frozen=True)
class OrderedUnionPlan:
    """
    Positional routing plan for an order-preserving multi-rank ``Union``.

    Every chunk of the union output is given a position in the global
    output order: all chunks of child 0 (rank 0 first, then rank 1, ...),
    followed by all chunks of child 1, and so on. Positions are split into
    contiguous ranges with :func:`partition_range`, and each chunk is sent to
    the rank that owns its position. Each rank then emits the chunks it owns
    in increasing position order, so reading the ranks in order reproduces
    the order required by ``pl.concat``.

    Chunk counts come from ``ChannelMetadata.local_count``, which is an
    estimate. If a child produces more chunks on a rank than estimated, the
    surplus is sent to the rank that owns the last estimated chunk of that
    child on that rank. This can unbalance the output, but destinations
    never decrease along the global order, so row order is preserved.

    Parameters
    ----------
    nranks
        Number of ranks.
    counts
        ``counts[i][j]`` is the estimated number of chunks child ``i``
        produces on rank ``j``.
    """

    nranks: int
    counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validate the plan."""
        if self.nranks < 1:
            raise ValueError(f"nranks must be positive, got {self.nranks}")
        for child_counts in self.counts:
            if len(child_counts) != self.nranks:
                raise ValueError(
                    "Each child needs one chunk count per rank, got "
                    f"{len(child_counts)} for {self.nranks} ranks"
                )
            if any(c < 0 for c in child_counts):
                raise ValueError(f"Chunk counts must be non-negative: {child_counts}")

    @property
    def total(self) -> int:
        """Total estimated number of chunks across all children and ranks."""
        return sum(sum(child_counts) for child_counts in self.counts)

    def offset(self, child: int, rank: int) -> int:
        """Global position of the first chunk of ``child`` on ``rank``."""
        flat = itertools.chain.from_iterable(self.counts)
        return sum(itertools.islice(flat, child * self.nranks + rank))

    def _owner(self, position: int) -> int:
        """Rank owning a global chunk position."""
        return partition_owner(position, self.nranks, self.total)

    def overflow_destination(self, child: int, rank: int) -> int:
        """
        Destination for chunks beyond the estimated count.

        This is the owner of the last estimated chunk of ``child`` on
        ``rank``. When that block is empty, it is the owner of the next
        position, which keeps destinations non-decreasing.
        """
        total = self.total
        if total == 0:
            return 0
        start = self.offset(child, rank)
        last = start + max(self.counts[child][rank] - 1, 0)
        return self._owner(min(last, total - 1))

    def destination(self, child: int, rank: int, index: int) -> int:
        """
        Destination of the ``index``-th local chunk of ``child`` on ``rank``.

        Parameters
        ----------
        child
            Child index.
        rank
            Source rank.
        index
            Zero-based position of the chunk in the child's local stream.

        Returns
        -------
        The rank that must emit this chunk.
        """
        if index < self.counts[child][rank]:
            return self._owner(self.offset(child, rank) + index)
        return self.overflow_destination(child, rank)

    def destinations(self, child: int, rank: int) -> frozenset[int]:
        """All ranks that may receive chunks of ``child`` from ``rank``."""
        start = self.offset(child, rank)
        stop = start + self.counts[child][rank]
        owners = {
            dst
            for dst in range(self.nranks)
            if _ranges_overlap(
                (start, stop), partition_range(dst, self.nranks, self.total)
            )
        }
        owners.add(self.overflow_destination(child, rank))
        return frozenset(owners)

    def remote_destinations(self, child: int, rank: int) -> list[int]:
        """Ranks other than ``rank`` that may receive chunks of ``child``."""
        return sorted(self.destinations(child, rank) - {rank})

    def remote_sources(self, child: int, rank: int) -> list[int]:
        """Ranks other than ``rank`` that may send chunks of ``child`` to it."""
        return [
            src
            for src in range(self.nranks)
            if src != rank and rank in self.destinations(child, src)
        ]

    def local_count(self, rank: int) -> int:
        """Estimated number of output chunks emitted by ``rank``."""
        start, stop = partition_range(rank, self.nranks, self.total)
        return stop - start


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two half-open ranges overlap."""
    return a[0] < b[1] and b[0] < a[1]


def needs_ordered_exchange(
    ir: Union, nranks: int, metadata: Sequence[ChannelMetadata]
) -> bool:
    """
    Return whether a ``Union`` must exchange chunks to preserve row order.

    Parameters
    ----------
    ir
        The Union IR node.
    nranks
        Number of ranks.
    metadata
        Channel metadata of the union's children.

    Returns
    -------
    ``True`` when the union must preserve order, has more than one child,
    runs on more than one rank, and at least one child is distributed
    across ranks.
    """
    return (
        ir.maintain_order
        and len(metadata) > 1
        and nranks > 1
        and not all(meta.duplicated for meta in metadata)
    )


async def _send_chunk(
    context: Context,
    ch_out: Channel[TableChunk],
    chunk: TableChunk,
    sequence_number: int,
) -> int:
    """Make ``chunk`` available, send it, and return the next sequence number."""
    chunk, _ = await make_table_chunks_available_or_wait(
        context, chunk, reserve_extra=0, net_memory_delta=0
    )
    await ch_out.send(context, Message(sequence_number, chunk))
    return sequence_number + 1


async def _passthrough_union(
    context: Context,
    comm: Communicator,
    ir: Union,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chs_in: Sequence[Channel[TableChunk]],
    metadata: Sequence[ChannelMetadata],
) -> None:
    """Forward each child's local chunks in channel order (no exchange)."""
    # Chunk counts on the wire are uniform across ranks, so report the
    # full sum.
    total_local_count = sum(meta.local_count for meta in metadata)
    duplicated = all(meta.duplicated for meta in metadata)
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(
            local_count=total_local_count,
            duplicated=duplicated,
        ),
    )

    # When a child has duplicated=True, every rank has produced the same
    # rows, so we drop them everywhere except rank 0 to avoid N counting.
    suppress = tuple(meta.duplicated and comm.rank != 0 for meta in metadata)

    seq_num_offset = 0
    for ch_in, drop in zip(chs_in, suppress, strict=True):
        num_ch_chunks = 0
        while (msg := await ch_in.recv(context)) is not None:
            if drop:
                stream = ir_context.get_cuda_stream()
                out_chunk = empty_table_chunk(ir, context, stream)
            else:
                # Relay: the unspilled chunk is forwarded as-is, nothing
                # is duplicated or freed.
                out_chunk, _ = await make_table_chunks_available_or_wait(
                    context,
                    TableChunk.from_message(msg, br=context.br()),
                    reserve_extra=0,
                    net_memory_delta=0,
                )
            await ch_out.send(
                context,
                Message(
                    msg.sequence_number + seq_num_offset,
                    out_chunk,
                ),
            )
            num_ch_chunks += 1
        seq_num_offset += num_ch_chunks


async def _ordered_union_child(
    context: Context,
    comm: Communicator,
    plan: OrderedUnionPlan,
    child: int,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    collective_id: int,
    sequence_number: int,
    *,
    drop: bool,
) -> int:
    """
    Route one child's chunks to their owning ranks and emit this rank's share.

    Returns the next output sequence number.
    """
    rank = comm.rank
    srcs = plan.remote_sources(child, rank)
    dsts = plan.remote_destinations(child, rank)
    exchange = (
        SparseAlltoall(context, comm, collective_id, srcs=srcs, dsts=dsts)
        if srcs or dsts
        else None
    )
    # Local chunks can be emitted as they arrive unless a lower rank sends
    # us chunks that come before them in the global order.
    emit_local_now = all(src > rank for src in srcs)
    local = ChunkStore(context)

    index = 0
    try:
        while (msg := await ch_in.recv(context)) is not None:
            if drop:
                # Duplicated child: only rank 0's copy takes part.
                continue
            dst = plan.destination(child, rank, index)
            chunk = TableChunk.from_message(msg, br=context.br())
            if dst == rank:
                if emit_local_now:
                    sequence_number = await _send_chunk(
                        context, ch_out, chunk, sequence_number
                    )
                else:
                    local.insert(Message(index, chunk))
            else:
                assert exchange is not None, "chunk routed to a rank with no exchange"
                chunk, extra = await make_table_chunks_available_or_wait(
                    context,
                    chunk,
                    reserve_extra=chunk.into_packed_data_cost(),
                    net_memory_delta=0,
                )
                exchange.insert(dst, chunk.into_packed_data(extra))
            index += 1
    finally:
        if exchange is not None:
            await exchange.insert_finished(context)

    # Emit in global order: chunks from lower ranks, then our own, then
    # chunks from higher ranks.
    remote = set(srcs)
    for src in range(comm.nranks):
        if src == rank:
            for local_msg in local:
                sequence_number = await _send_chunk(
                    context,
                    ch_out,
                    TableChunk.from_message(local_msg, br=context.br()),
                    sequence_number,
                )
        elif src in remote:
            assert exchange is not None
            for packed in exchange.extract(src):
                sequence_number = await _send_chunk(
                    context,
                    ch_out,
                    TableChunk.from_packed_data(packed, context.br()),
                    sequence_number,
                )
    return sequence_number


async def _ordered_union(
    context: Context,
    comm: Communicator,
    ch_out: Channel[TableChunk],
    chs_in: Sequence[Channel[TableChunk]],
    metadata: Sequence[ChannelMetadata],
    collective_ids: Sequence[int],
) -> None:
    """Concatenate children across ranks while preserving global row order."""
    nchildren = len(chs_in)
    if len(collective_ids) != nchildren + 1:
        raise RuntimeError(
            f"Ordered multi-rank Union needs {nchildren + 1} collective IDs, "
            f"got {len(collective_ids)}."
        )

    # One AllGather of every (child, rank) chunk count. Each rank fills its
    # own slots and leaves the rest zero, so the summed result is the full
    # count matrix. Duplicated children only count on rank 0.
    local_values = [0] * (nchildren * comm.nranks)
    for child, meta in enumerate(metadata):
        if not meta.duplicated or comm.rank == 0:
            local_values[child * comm.nranks + comm.rank] = meta.local_count
    gathered = await allgather_reduce(context, comm, collective_ids[0], *local_values)
    plan = OrderedUnionPlan(
        nranks=comm.nranks,
        counts=tuple(
            tuple(gathered[child * comm.nranks : (child + 1) * comm.nranks])
            for child in range(nchildren)
        ),
    )

    await send_metadata(
        ch_out, context, ChannelMetadata(local_count=plan.local_count(comm.rank))
    )

    sequence_number = 0
    for child, (ch_in, meta) in enumerate(zip(chs_in, metadata, strict=True)):
        sequence_number = await _ordered_union_child(
            context,
            comm,
            plan,
            child,
            ch_in,
            ch_out,
            collective_ids[child + 1],
            sequence_number,
            drop=meta.duplicated and comm.rank != 0,
        )


@define_actor()
async def union_node(
    context: Context,
    comm: Communicator,
    ir: Union,
    ir_context: IRExecutionContext,
    collective_ids: Sequence[int],
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Union IR node.
    ir_context
        The execution context for the IR node.
    collective_ids
        Collective IDs for an order-preserving multi-rank union: one for
        the chunk-count AllGather, then one ``SparseAlltoall`` per child.
        Empty (and unused) when the union does not need to exchange chunks.
    ch_out
        The output Channel[TableChunk].
    chs_in
        The input Channel[TableChunk]s.
    """
    async with shutdown_on_error(
        context,
        chs_in=chs_in,
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        metadata = await gather_in_task_group(
            *(recv_metadata(ch, context) for ch in chs_in)
        )
        if needs_ordered_exchange(ir, comm.nranks, metadata):
            await _ordered_union(
                context, comm, ch_out, chs_in, metadata, collective_ids
            )
        else:
            # Unordered union, single rank, or every child duplicated: each
            # rank's local concatenation is already a valid result.
            await _passthrough_union(
                context, comm, ir, ir_context, ch_out, chs_in, metadata
            )
        await ch_out.drain(context)


@generate_ir_sub_network.register(Union)
def _(
    ir: Union, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Union operation.
    # Unordered (or single-rank) unions pass child chunks through in channel
    # order; ordered multi-rank unions route chunks so the global order is
    # preserved (see ``OrderedUnionPlan``).

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])
    ir_context = ir_context_for_node(rec, ir)

    nodes[ir] = [
        union_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            list(rec.state["collective_id_map"].get(ir, [])),
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
        )
    ]
    return nodes, channels

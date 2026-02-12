# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy and Distinct logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, Distinct, GroupBy, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.groupby import combine, decompose
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunkwise_evaluate,
    concat_batch,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    is_partitioned_on_keys,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.repartition import Repartition

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer
    from cudf_polars.typing import Schema


@dataclass
class DecomposedGroupBy:
    """Holds decomposed GroupBy or Distinct operations for multi-phase aggregation."""

    ir: GroupBy | Distinct
    """The original IR node."""
    piecewise_ir: GroupBy | Distinct
    """The decomposed piecewise IR node."""
    reduction_ir: GroupBy | Distinct
    """The decomposed reduction IR node. Same as piecewise_ir for Distinct."""
    select_ir: Select | None
    """The decomposed select IR node. Always None for Distinct."""
    need_preshuffle: bool
    """Whether the operation requires preshuffling."""

    @classmethod
    def from_ir(cls, ir: GroupBy | Distinct) -> DecomposedGroupBy:
        """Decompose a GroupBy IR node into multi-phase operations."""
        piecewise_ir: GroupBy | Distinct
        reduction_ir: GroupBy | Distinct
        select_ir: Select | None
        need_preshuffle: bool

        if isinstance(ir, Distinct):
            piecewise_ir = reduction_ir = ir
            select_ir = None
            need_preshuffle = (
                ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE
            )
        elif isinstance(ir, GroupBy):
            name_generator = unique_names(ir.schema.keys())
            selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = (
                combine(
                    *(
                        decompose(agg.name, agg.value, names=name_generator)
                        for agg in ir.agg_requests
                    )
                )
            )

            # Piecewise groupby schema and IR
            pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
                k.name: k.value.dtype for k in piecewise_exprs
            }
            piecewise_ir = GroupBy(
                pwise_schema,
                ir.keys,
                piecewise_exprs,
                ir.maintain_order,
                None,
                ir.children[0],
            )

            # Grouped keys for reduction and selection
            groupby_keys = tuple(
                NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys
            )

            # Reduction groupby schema and IR (must match pwise_schema for tree reduction)
            reduction_schema = {k.name: k.value.dtype for k in groupby_keys} | {
                k.name: k.value.dtype for k in reduction_exprs
            }
            assert pwise_schema == reduction_schema, (
                "piecewise and reduction schemas must match for tree reduction"
            )
            reduction_ir = GroupBy(
                reduction_schema,
                groupby_keys,
                reduction_exprs,
                ir.maintain_order,
                None,
                piecewise_ir,
            )

            # Selection IR (child is reduction_ir, not piecewise_ir)
            select_ir = Select(
                ir.schema,
                [
                    *(
                        NamedExpr(k.name, Col(k.value.dtype, k.name))
                        for k in groupby_keys
                    ),
                    *selection_exprs,
                ],
                False,  # noqa: FBT003
                reduction_ir,
            )
        else:  # pragma: no cover
            raise TypeError(f"Unsupported IR type: {type(ir)}")

        return cls(
            ir=ir,
            piecewise_ir=piecewise_ir,
            reduction_ir=reduction_ir,
            select_ir=select_ir,
            need_preshuffle=need_preshuffle,
        )


async def _tree_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    target_partition_size: int,
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    collective_id: int | None = None,
    reduction_ran: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby or distinct using tree reduction to a single output.

    Reads chunks, applies piecewise aggregation, and reduces incrementally.
    When collective_id is provided and data is not duplicated, uses allgather
    to collect partial results from all ranks before final reduction.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    decomposed
        The decomposed groupby containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    target_partition_size
        Target size in bytes for output partitions.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    collective_id
        Optional collective ID for allgather. If None, no allgather is performed.
    reduction_ran
        Whether evaluated_chunks have already been through reduction_ir.
    tracer
        Optional tracer for runtime metrics.
    """
    tree_reduction_ran = reduction_ran
    nranks = context.comm().nranks
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )
    is_distinct = isinstance(decomposed.ir, Distinct)

    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    evaluated_chunks = evaluated_chunks or []
    total_size = sum(c.data_alloc_size() for c in evaluated_chunks)

    receiving = True
    while receiving or len(evaluated_chunks) > 1:
        if receiving:
            msg = await ch_in.recv(context)
            if msg is None:
                receiving = False
            else:
                chunk = await evaluate_chunk(
                    context,
                    TableChunk.from_message(msg),
                    decomposed.piecewise_ir,
                    ir_context,
                )
                del msg
                evaluated_chunks.append(chunk)
                total_size += chunk.data_alloc_size()

        if len(evaluated_chunks) > 1 and (
            not receiving or total_size > target_partition_size
        ):
            merged = await evaluate_batch(
                evaluated_chunks, context, decomposed.reduction_ir, ir_context
            )
            evaluated_chunks = [merged]
            total_size = merged.data_alloc_size()
            tree_reduction_ran = True

    chunk_schema = (
        decomposed.reduction_ir.schema
        if tree_reduction_ran
        else decomposed.piecewise_ir.schema
    )

    # Allgather partial results from all ranks if needed
    if need_allgather:
        assert collective_id is not None

        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        if evaluated_chunks:
            if tree_reduction_ran or is_distinct:
                reduced_chunk = await concat_batch(
                    evaluated_chunks, context, chunk_schema, ir_context
                )
            else:
                reduced_chunk = await evaluate_batch(
                    evaluated_chunks, context, decomposed.reduction_ir, ir_context
                )
            del evaluated_chunks
            allgather.insert(0, reduced_chunk)

        allgather.insert_finished()

        gathered_table = await allgather.extract_concatenated(stream)
        gathered_chunk = TableChunk.from_pylibcudf_table(
            gathered_table, stream, exclusive_view=True
        )
        evaluated_chunks = [
            await evaluate_chunk(
                context, gathered_chunk, decomposed.reduction_ir, ir_context
            )
        ]
        del gathered_chunk

    if evaluated_chunks:
        # Final result
        if decomposed.select_ir is not None:
            chunk = await evaluate_chunk(
                context, evaluated_chunks.pop(0), decomposed.select_ir, ir_context
            )
        else:
            chunk = evaluated_chunks.pop(0)
        if tracer is not None:
            tracer.add_chunk(table=chunk.table_view())
        await ch_out.send(context, Message(0, chunk))
        del chunk
    else:
        # No data - send empty chunk
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(decomposed.ir, context, stream)
        if tracer is not None:
            tracer.add_chunk()
        await ch_out.send(context, Message(0, chunk))

    await ch_out.drain(context)


async def _shuffle_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    output_count: int,
    collective_id: int,
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    shuffle_context: Context | None = None,
    reduction_ran: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Shuffle-based groupby or distinct.

    Shuffles data by keys, then applies local groupby or distinct to
    each partition. Use when the expected output is large.

    Parameters
    ----------
    context
        The rapidsmpf streaming context for channel operations.
    decomposed
        The decomposed groupby or distinct containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    output_count
        The number of output partitions per rank.
    collective_id
        The collective ID for the shuffle operation.
    key_indices
        The column indices of the distinct keys.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    shuffle_context
        Optional context for shuffle operations.
        Defaults to a temporary local context.
    reduction_ran
        Whether evaluated_chunks have already been through reduction_ir.
    tracer
        Optional tracer for runtime metrics.
    """
    # Define shuffle context
    if shuffle_context is None:
        # Create a temporary local context
        options = Options(get_environment_variables())
        local_comm = single_comm(options)
        shuffle_context = Context(local_comm, context.br(), options)
    shuf_nranks = shuffle_context.comm().nranks
    shuf_rank = shuffle_context.comm().rank
    modulus = shuf_nranks * output_count

    is_local_shuffle = shuf_nranks == 1 and metadata_in.partitioning is not None
    output_key_indices = _key_indices(decomposed.ir, decomposed.ir.schema)
    if isinstance(decomposed.ir, Distinct):
        shuffle_key_indices = output_key_indices
    else:
        shuffle_key_indices = _key_indices(
            decomposed.piecewise_ir, decomposed.piecewise_ir.schema
        )

    if is_local_shuffle:
        inter_rank_scheme = metadata_in.partitioning.inter_rank
        local_scheme = HashScheme(column_indices=output_key_indices, modulus=modulus)
    else:
        inter_rank_scheme = HashScheme(
            column_indices=output_key_indices, modulus=modulus
        )
        local_scheme = "inherit"

    metadata_out = ChannelMetadata(
        local_count=max(1, output_count // shuf_nranks),
        partitioning=Partitioning(inter_rank_scheme, local_scheme),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    shuffle = ShuffleManager(context, output_count, shuffle_key_indices, collective_id)

    for chunk in evaluated_chunks or []:
        shuffle.insert_chunk(chunk)

    pwise_ir: list[IR] = [decomposed.piecewise_ir]
    if reduction_ran:
        pwise_ir.append(decomposed.reduction_ir)

    while (msg := await ch_in.recv(context)) is not None:
        shuffle.insert_chunk(
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg),
                pwise_ir,
                ir_context,
            )
        )
        del msg

    await shuffle.insert_finished()

    stream = ir_context.get_cuda_stream()
    extract_ir: list[IR] = [decomposed.reduction_ir]
    if decomposed.select_ir is not None:
        extract_ir.append(decomposed.select_ir)
    for partition_id in range(shuf_rank, output_count, shuf_nranks):
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )
        output_chunk = await evaluate_chunk(
            context,
            partition_chunk,
            extract_ir,
            ir_context,
        )
        del partition_chunk
        if tracer is not None:
            tracer.add_chunk(table=output_chunk.table_view())
        await ch_out.send(context, Message(partition_id, output_chunk))

    await ch_out.drain(context)


def _key_indices(ir: GroupBy | Distinct, schema: Schema) -> tuple[int, ...]:
    schema_keys = list(schema.keys())
    if isinstance(ir, GroupBy):
        groupby_key_names = tuple(ne.name for ne in ir.keys)
        return tuple(
            schema_keys.index(k) for k in groupby_key_names if k in schema_keys
        )
    else:
        subset = ir.subset or frozenset(ir.schema)
        return tuple(schema_keys.index(k) for k in subset if k in schema_keys)


def _require_tree(ir: GroupBy | Distinct) -> bool:
    if isinstance(ir, GroupBy):
        return ir.maintain_order
    else:
        return ir.stable or ir.keep in (
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        )


@define_py_node()
async def keyed_reduction_node(
    context: Context,
    ir: GroupBy | Distinct,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    sample_chunk_count: int,
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic GroupBy or Distinct node that selects the best strategy at runtime.

    Strategy selection based on sampled data:
    - Chunk-wise: Data already partitioned on the necessary keys
    - Tree reduction: Small estimated output (< target_partition_size)
    - Shuffle: Large estimated output requiring redistribution

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    sample_chunk_count
        The number of chunks to sample.
    target_partition_size
        The target partition size.
    collective_ids
        The collective IDs.
    """
    async with shutdown_on_error(context, ch_in, ch_out, trace_ir=ir) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        # Check if already partitioned on keys
        nranks = context.comm().nranks
        key_indices = _key_indices(ir, ir.children[0].schema)
        require_tree = _require_tree(ir)
        partitioned_inter_rank, partitioned_local = is_partitioned_on_keys(
            metadata_in,
            key_indices,
            nranks,
        )
        fully_partitioned = partitioned_inter_rank and partitioned_local
        can_skip_global_comm = metadata_in.duplicated or partitioned_inter_rank
        fallback_case = (
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        # If already partitioned or concatenated, just do a chunk-wise groupby
        if fully_partitioned or fallback_case:
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                tracer=tracer,
            )
            return

        # Decompose for multi-phase execution
        # Note: Lowering guarantees decomposition succeeds and preshuffle is done
        decomposed = DecomposedGroupBy.from_ir(ir)
        assert not decomposed.need_preshuffle, "Should already be shuffled."

        evaluated_chunks: list[TableChunk] = []
        total_size = 0
        merge_count = 0
        chunks_sampled = 0

        for _ in range(sample_chunk_count):
            msg = await ch_in.recv(context)
            if msg is None:
                break
            chunks_sampled += 1
            chunk = await evaluate_chunk(
                context,
                TableChunk.from_message(msg),
                decomposed.piecewise_ir,
                ir_context,
            )
            del msg
            total_size += chunk.data_alloc_size(MemoryType.DEVICE)
            evaluated_chunks.append(chunk)

            if total_size > target_partition_size and len(evaluated_chunks) > 1:
                merged = await evaluate_batch(
                    evaluated_chunks, context, decomposed.reduction_ir, ir_context
                )
                total_size = merged.data_alloc_size(MemoryType.DEVICE)
                evaluated_chunks = [merged]
                merge_count += 1
                if total_size > target_partition_size:
                    break

        local_count = metadata_in.local_count
        if can_skip_global_comm:
            global_chunk_count = local_count
            global_chunks_sampled = chunks_sampled
        else:
            (
                total_size,
                global_chunk_count,
                global_chunks_sampled,
            ) = await allgather_reduce(
                context, collective_ids.pop(), total_size, local_count, chunks_sampled
            )
        reduction_ran = merge_count > 0

        if global_chunks_sampled > 0:
            global_size = (total_size // global_chunks_sampled) * global_chunk_count
        else:
            global_size = 0

        # TODO: Fast return if we already have the slice ready
        if ir.zlice is not None and ir.zlice[1] is not None and evaluated_chunks:
            total_rows = sum(c.table_view().num_rows() for c in evaluated_chunks)
            if total_rows > 0:
                avg_row_size = total_size / total_rows
                global_size = min(global_size, int(avg_row_size * ir.zlice[1]))

        use_tree = global_size < target_partition_size or require_tree

        if can_skip_global_comm:
            if use_tree:
                if tracer is not None:
                    tracer.decision = "tree_local"
                await _tree_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    target_partition_size,
                    evaluated_chunks=evaluated_chunks,
                    reduction_ran=reduction_ran,
                    tracer=tracer,
                )
            else:
                if tracer is not None:
                    tracer.decision = "shuffle_local"
                ideal_count = max(1, global_size // target_partition_size)
                output_count = max(1, min(ideal_count, local_count))
                await _shuffle_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    output_count,
                    collective_ids.pop(),
                    evaluated_chunks=evaluated_chunks,
                    reduction_ran=reduction_ran,
                    shuffle_context=context if nranks == 1 else None,
                    tracer=tracer,
                )
        elif use_tree:
            if tracer is not None:
                tracer.decision = "tree_allgather"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                target_partition_size,
                evaluated_chunks=evaluated_chunks,
                collective_id=collective_ids.pop(),
                reduction_ran=reduction_ran,
                tracer=tracer,
            )
        else:
            if tracer is not None:
                tracer.decision = "shuffle"
            ideal_count = max(1, global_size // target_partition_size)
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            await _shuffle_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                evaluated_chunks=evaluated_chunks,
                reduction_ran=reduction_ran,
                tracer=tracer,
            )


@generate_ir_sub_network.register(GroupBy)
@generate_ir_sub_network.register(Distinct)
def _(
    ir: GroupBy | Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy or Distinct operation."""
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming"

    # Only use the dynamic reduction node when dynamic planning is enabled
    if config_options.executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass GroupBy/Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    nodes[ir] = [
        keyed_reduction_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor.dynamic_planning.sample_chunk_count_reduce,
            config_options.executor.target_partition_size,
            collective_ids,
        )
    ]

    return nodes, channels

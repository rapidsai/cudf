# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy and Distinct logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataType
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


async def _local_aggregation(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    *,
    allow_early_exit: bool = True,
) -> tuple[TableChunk, bool, int]:
    """
    Local groupby or distinct aggregation.

    Parameters
    ----------
    context
        The rapidsmpf context.
    decomposed
        The decomposed groupby or distinct.
    ir_context
        The IR execution context.
    ch_in
        The input channel.
    target_partition_size
        The target partition size.
    allow_early_exit
        Whether to allow early exit from the loop when
        the total size exceeds the target partition size.

    Returns
    -------
    aggregated
        The aggregated result.
    input_drained
        Whether the input channel is drained.
    chunks_received
        The number of chunks received from the input channel.
    """
    total_size = 0
    chunks_received = 0
    input_drained = False
    evaluated_chunks: list[TableChunk] = []
    while True:
        msg = await ch_in.recv(context)
        if msg is None:
            input_drained = True
            break

        chunks_received += 1
        chunk = await evaluate_chunk(
            context,
            TableChunk.from_message(msg),
            decomposed.piecewise_ir,
            ir_context=ir_context,
        )
        del msg
        total_size += chunk.data_alloc_size(MemoryType.DEVICE)
        evaluated_chunks.append(chunk)
        if total_size > target_partition_size and len(evaluated_chunks) > 1:
            evaluated_chunks = [
                await evaluate_batch(
                    evaluated_chunks,
                    context,
                    decomposed.reduction_ir,
                    ir_context=ir_context,
                )
            ]
            total_size = evaluated_chunks[0].data_alloc_size(MemoryType.DEVICE)
        if total_size > target_partition_size and allow_early_exit:
            break

    aggregated: TableChunk
    if len(evaluated_chunks) > 1:
        aggregated = await evaluate_batch(
            evaluated_chunks,
            context,
            decomposed.reduction_ir,
            ir_context=ir_context,
        )
    elif evaluated_chunks:
        aggregated = evaluated_chunks[0]
    else:
        aggregated = empty_table_chunk(
            decomposed.reduction_ir,
            context,
            ir_context.get_cuda_stream(),
        )
    del evaluated_chunks

    return aggregated, input_drained, chunks_received


async def _tree_reduce(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_id: int,
    allgather_context: Context | None,
    *,
    aggregated: TableChunk,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby or distinct using tree reduction to a single output.

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
    metadata_in
        The input channel metadata.
    aggregated
        The aggregated result for already-evaluated chunks.
    collective_id
        Collective ID for allgather (used when allgather_context is not None).
    allgather_context
        Context for allgather; if None, no allgather is performed.
    tracer
        Optional tracer for runtime metrics.
    """
    need_allgather = (
        allgather_context is not None
        and not metadata_in.duplicated
        and context.comm().nranks > 1
    )

    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    if need_allgather:
        allgather = AllGatherManager(allgather_context, collective_id)

        allgather.insert(
            0,
            _enforce_schema(aggregated, decomposed.reduction_ir.schema),
        )

        allgather.insert_finished()

        stream = ir_context.get_cuda_stream()
        aggregated = await evaluate_chunk(
            allgather_context,
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(stream),
                stream,
                exclusive_view=True,
            ),
            decomposed.reduction_ir,
            ir_context=ir_context,
        )

    if decomposed.select_ir is not None:
        aggregated = await evaluate_chunk(
            context, aggregated, decomposed.select_ir, ir_context=ir_context
        )
    if tracer is not None:
        tracer.add_chunk(table=aggregated.table_view())
    await ch_out.send(context, Message(0, aggregated))
    del aggregated

    await ch_out.drain(context)


async def _shuffle_reduce(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    modulus: int,
    collective_id: int,
    shuffle_context: Context | None,
    target_partition_size: int,
    *,
    aggregated: TableChunk,
    input_drained: bool = False,
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
    modulus
        The modulus for the shuffle operation.
    collective_id
        The collective ID for the shuffle operation.
    shuffle_context
        Context to use for shuffling.
    target_partition_size
        Target partition size for local aggregation batches.
    aggregated
        The aggregated result for already-evaluated chunks.
    input_drained
        Whether the input channel is drained.
    tracer
        Optional tracer for runtime metrics.
    """
    if shuffle_context is None:
        if context.comm().nranks == 1:
            shuffle_context = context
        else:
            options = Options(get_environment_variables())
            local_comm = single_comm(options)
            shuffle_context = Context(local_comm, context.br(), options)
    assert shuffle_context is not None, "Shuffle context required."
    shuf_nranks = shuffle_context.comm().nranks
    shuf_rank = shuffle_context.comm().rank
    modulus = max(shuf_nranks, modulus)

    output_key_indices = _key_indices(decomposed.ir, decomposed.ir.schema)
    if isinstance(decomposed.ir, Distinct):
        shuffle_key_indices = output_key_indices
    else:
        shuffle_key_indices = _key_indices(
            decomposed.piecewise_ir, decomposed.piecewise_ir.schema
        )

    if shuf_nranks == 1:
        inter_rank_scheme = (
            None
            if metadata_in.partitioning is None
            else metadata_in.partitioning.inter_rank
        )
        local_scheme = HashScheme(column_indices=output_key_indices, modulus=modulus)
        local_output_count = modulus
    else:
        inter_rank_scheme = HashScheme(
            column_indices=output_key_indices, modulus=modulus
        )
        local_scheme = "inherit"
        local_output_count = (modulus - shuf_rank + shuf_nranks - 1) // shuf_nranks

    metadata_out = ChannelMetadata(
        local_count=local_output_count,
        partitioning=Partitioning(inter_rank_scheme, local_scheme),
        duplicated=metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    shuffle = ShuffleManager(
        shuffle_context, modulus, shuffle_key_indices, collective_id
    )

    shuffle.insert_chunk(
        _enforce_schema(
            aggregated,
            decomposed.reduction_ir.schema,
        )
    )
    del aggregated

    while not input_drained:
        aggregated, input_drained, _ = await _local_aggregation(
            context,
            decomposed,
            ir_context,
            ch_in,
            target_partition_size,
        )
        shuffle.insert_chunk(
            _enforce_schema(aggregated, decomposed.reduction_ir.schema)
        )
        del aggregated

    await shuffle.insert_finished()
    extract_irs = [decomposed.reduction_ir] + (
        [decomposed.select_ir] if decomposed.select_ir else []
    )
    stream = ir_context.get_cuda_stream()
    for partition_id in shuffle.local_partitions():
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )
        output_chunk = await evaluate_chunk(
            context,
            partition_chunk,
            *extract_irs,
            ir_context=ir_context,
        )
        del partition_chunk
        if tracer is not None:
            tracer.add_chunk(table=output_chunk.table_view())
        await ch_out.send(context, Message(partition_id, output_chunk))

    await ch_out.drain(context)


def _enforce_schema(
    chunk: TableChunk,
    canonical_schema: dict[str, Any],
) -> TableChunk:
    """Enforce the canonical schema of a TableChunk."""
    tbl = chunk.table_view()
    cols = tbl.columns()
    names = list(canonical_schema.keys())
    if len(cols) != len(names):  # pragma: no cover
        raise ValueError(
            f"Column count ({len(cols)}) does not match schema ({len(names)})"
        )

    target_plcs = []
    needs_cast = False
    for col, name in zip(cols, names, strict=True):
        dt = canonical_schema[name]
        target_plc = (dt if isinstance(dt, DataType) else DataType(dt)).plc_type
        target_plcs.append(target_plc)
        if col.type().id() != target_plc.id():
            needs_cast = True
    if not needs_cast:
        return chunk
    new_columns = [
        plc.unary.cast(col, target_plc, stream=chunk.stream)
        if col.type().id() != target_plc.id()
        else col
        for col, target_plc in zip(cols, target_plcs, strict=True)
    ]
    return TableChunk.from_pylibcudf_table(
        plc.Table(new_columns), chunk.stream, exclusive_view=True
    )


def _key_indices(ir: GroupBy | Distinct, schema: Schema) -> tuple[int, ...]:
    schema_keys = {n: i for i, n in enumerate(schema.keys())}
    if isinstance(ir, GroupBy):
        groupby_key_names = tuple(ne.name for ne in ir.keys)
        if not all(k in schema_keys for k in groupby_key_names):
            return ()
        return tuple(schema_keys[k] for k in groupby_key_names)
    else:
        subset = ir.subset or frozenset(ir.schema)
        return tuple(schema_keys[k] for k in subset)


def _require_tree(ir: GroupBy | Distinct) -> bool:
    if isinstance(ir, GroupBy):
        return ir.maintain_order
    else:
        return ir.stable or ir.keep in (
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        )


async def _select_algorithm(
    context: Context,
    local_count: int,
    aggregated: TableChunk,
    chunks_received: int,
    input_drained: bool,  # noqa: FBT001
    collective_ids: list[int],
    target_partition_size: int,
    skip_global_comm: bool,  # noqa: FBT001
    tracer: ActorTracer | None,
) -> tuple[int, Context | None]:
    """
    Select the best algorithm for the given context and metadata.

    Parameters
    ----------
    context
        The rapidsmpf context.
    local_count
        The local count of the input channel.
    aggregated
        The aggregated result for already-evaluated chunks.
    chunks_received
        The number of chunks received from the input channel.
    input_drained
        Whether the input channel is drained.
    collective_ids
        The collective IDs.
    target_partition_size
        The target partition size.
    skip_global_comm
        Whether to skip the global communication.
    tracer
        Optional tracer for runtime metrics.

    Returns
    -------
    output_count
        The output count.
    collective_context
        The context to use for shuffle or allgather.
    """
    if skip_global_comm:
        total_size = aggregated.data_alloc_size(MemoryType.DEVICE)
        total_chunk_count = local_count
        total_chunks_received = chunks_received
        total_need_shuffle = int(not input_drained)
    else:
        (
            total_size,
            total_chunk_count,
            total_chunks_received,
            total_need_shuffle,
        ) = await allgather_reduce(
            context,
            collective_ids.pop(),
            aggregated.data_alloc_size(MemoryType.DEVICE),
            local_count,
            chunks_received,
            int(not input_drained),
        )

    ideal_count = 1
    use_tree = total_need_shuffle == 0
    if not use_tree:
        if total_chunks_received > 0:
            total_size = (total_size // total_chunks_received) * total_chunk_count
        ideal_count = max(2, total_size // target_partition_size)

    output_count_limit = local_count if skip_global_comm else total_chunk_count
    output_count = min(ideal_count, output_count_limit)
    collective_context = None if skip_global_comm else context
    if tracer is not None:
        tracer.decision = (
            "tree_local"
            if skip_global_comm and use_tree
            else "shuffle_local"
            if skip_global_comm
            else "tree_allgather"
            if use_tree
            else "shuffle"
        )

    return output_count, collective_context


@define_actor()
async def keyed_reduction_actor(
    context: Context,
    ir: GroupBy | Distinct,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic GroupBy or Distinct actor that selects the best strategy at runtime.

    Strategy selection based on observed data:
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
    target_partition_size
        The target partition size.
    collective_ids
        The collective IDs.
    """
    async with shutdown_on_error(context, ch_in, ch_out, trace_ir=ir) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        nranks = context.comm().nranks
        key_indices = _key_indices(ir, ir.children[0].schema)
        require_tree = _require_tree(ir)
        partitioned_inter_rank, partitioned_local = is_partitioned_on_keys(
            metadata_in,
            key_indices,
            nranks,
        )
        fully_partitioned = partitioned_inter_rank and partitioned_local
        fallback_case = (
            # NOTE: This criteria means that we fell back
            # to one partition at lowering time.
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        if fully_partitioned or fallback_case:
            if tracer is not None:
                tracer.decision = "chunkwise"
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

        decomposed = DecomposedGroupBy.from_ir(ir)
        assert not decomposed.need_preshuffle, "Should already be shuffled."

        aggregated, input_drained, chunks_received = await _local_aggregation(
            context,
            decomposed,
            ir_context,
            ch_in,
            target_partition_size,
            allow_early_exit=not require_tree,
        )

        skip_global_comm = metadata_in.duplicated or partitioned_inter_rank
        output_count, collective_context = await _select_algorithm(
            context,
            metadata_in.local_count,
            aggregated,
            chunks_received,
            input_drained,
            collective_ids,
            target_partition_size,
            skip_global_comm,
            tracer,
        )

        if output_count == 1:
            await _tree_reduce(
                context,
                decomposed,
                ir_context,
                ch_out,
                metadata_in,
                collective_ids.pop(),
                collective_context,
                aggregated=aggregated,
                tracer=tracer,
            )
        else:
            await _shuffle_reduce(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                collective_context,
                target_partition_size,
                aggregated=aggregated,
                input_drained=input_drained,
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

    if config_options.executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass GroupBy/Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    actors[ir] = [
        keyed_reduction_actor(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor.target_partition_size,
            collective_ids,
        )
    ]

    return actors, channels

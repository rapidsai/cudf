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
    tracer
        Optional tracer for runtime metrics.
    """
    nranks = context.comm().nranks
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )

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
                    ir_context=ir_context,
                )
                del msg
                evaluated_chunks.append(chunk)
                total_size += chunk.data_alloc_size()

        if len(evaluated_chunks) > 1 and (
            not receiving or total_size > target_partition_size
        ):
            merged = await evaluate_batch(
                evaluated_chunks,
                context,
                decomposed.reduction_ir,
                ir_context=ir_context,
            )
            evaluated_chunks = [merged]
            total_size = merged.data_alloc_size()

    evaluated_chunk: TableChunk | None = None
    if evaluated_chunks:
        evaluated_chunk = evaluated_chunks.pop(0)
    assert len(evaluated_chunks) == 0, "Expected no chunks left"

    # Allgather partial results from all ranks if needed
    if need_allgather:
        assert collective_id is not None

        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        if evaluated_chunk is not None:
            allgather.insert(0, evaluated_chunk)

        allgather.insert_finished()

        evaluated_chunk = await evaluate_chunk(
            context,
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(stream),
                stream,
                exclusive_view=True,
            ),
            decomposed.reduction_ir,
            ir_context=ir_context,
        )

    if evaluated_chunk is not None:
        if decomposed.select_ir is not None:
            evaluated_chunk = await evaluate_chunk(
                context, evaluated_chunk, decomposed.select_ir, ir_context=ir_context
            )
        if tracer is not None:
            tracer.add_chunk(table=evaluated_chunk.table_view())
        await ch_out.send(context, Message(0, evaluated_chunk))
        del evaluated_chunk
    else:
        # No data - send empty chunk
        stream = ir_context.get_cuda_stream()
        if tracer is not None:
            tracer.add_chunk()
        await ch_out.send(
            context, Message(0, empty_table_chunk(decomposed.ir, context, stream))
        )

    await ch_out.drain(context)


async def _shuffle_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    modulus: int,
    collective_id: int,
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    shuffle_context: Context | None = None,
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
    key_indices
        The column indices of the distinct keys.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    shuffle_context
        Optional context for shuffle operations.
        Defaults to a temporary local context.
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

    output_key_indices = _key_indices(decomposed.ir, decomposed.ir.schema)
    if isinstance(decomposed.ir, Distinct):
        shuffle_key_indices = output_key_indices
    else:
        shuffle_key_indices = _key_indices(
            decomposed.piecewise_ir, decomposed.piecewise_ir.schema
        )

    if shuf_nranks == 1:
        # Local shuffle
        inter_rank_scheme = metadata_in.partitioning.inter_rank
        local_scheme = HashScheme(column_indices=output_key_indices, modulus=modulus)
        local_output_count = modulus
    else:
        # Global shuffle
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

    evaluated_chunks = evaluated_chunks or []
    while len(evaluated_chunks) > 0:
        shuffle.insert_chunk(
            _enforce_schema(
                evaluated_chunks.pop(0),
                decomposed.reduction_ir.schema,
            )
        )

    while (msg := await ch_in.recv(context)) is not None:
        shuffle.insert_chunk(
            _enforce_schema(
                await evaluate_chunk(
                    context,
                    TableChunk.from_message(msg),
                    decomposed.piecewise_ir,
                    ir_context=ir_context,
                ),
                decomposed.reduction_ir.schema,
            )
        )
        del msg

    await shuffle.insert_finished()
    extract_irs: list[IR] = [decomposed.reduction_ir]
    if decomposed.select_ir is not None:
        extract_irs.append(decomposed.select_ir)
    stream = ir_context.get_cuda_stream()
    for partition_id in range(shuf_rank, modulus, shuf_nranks):
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
        target_dtype = canonical_schema[name]
        if not isinstance(target_dtype, DataType):
            target_dtype = DataType(target_dtype)
        target_plc = target_dtype.plc_type
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
                ir_context=ir_context,
            )
            del msg
            total_size += chunk.data_alloc_size(MemoryType.DEVICE)
            evaluated_chunks.append(chunk)

            if total_size > target_partition_size and len(evaluated_chunks) > 1:
                merged = await evaluate_batch(
                    evaluated_chunks,
                    context,
                    decomposed.reduction_ir,
                    ir_context=ir_context,
                )
                total_size = merged.data_alloc_size(MemoryType.DEVICE)
                evaluated_chunks = [merged]
                merge_count += 1
                if total_size > target_partition_size:
                    break

        local_count = metadata_in.local_count
        if can_skip_global_comm:
            total_chunk_count = local_count
            total_chunks_sampled = chunks_sampled
        else:
            (
                total_size,
                total_chunk_count,
                total_chunks_sampled,
            ) = await allgather_reduce(
                context, collective_ids.pop(), total_size, local_count, chunks_sampled
            )

        if total_chunks_sampled > 0:
            total_size = (total_size // total_chunks_sampled) * total_chunk_count
        use_tree = total_size < target_partition_size or require_tree
        ideal_count = max(1, total_size // target_partition_size)

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
                    tracer=tracer,
                )
            else:
                if tracer is not None:
                    tracer.decision = "shuffle_local"
                await _shuffle_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    max(1, min(ideal_count, local_count)),
                    collective_ids.pop(),
                    evaluated_chunks=evaluated_chunks,
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
                tracer=tracer,
            )
        else:
            if tracer is not None:
                tracer.decision = "shuffle"
            await _shuffle_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                max(nranks, min(ideal_count, total_chunk_count)),
                collective_ids.pop(),
                evaluated_chunks=evaluated_chunks,
                shuffle_context=context,
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

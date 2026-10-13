# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the RapidsMPF streaming sort."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import OrderKey, OrderScheme, Ordering
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Empty
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.utils import (
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
)
from cudf_polars.streaming.sort import _get_final_sort_boundaries

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext, Sort
    from cudf_polars.typing import Schema


def boundary_schema(by: list[str], by_dtypes: list[DataType]) -> Schema:
    """
    Return the schema of a sort boundaries table.

    Parameters
    ----------
    by
        Names of the sort key columns.
    by_dtypes
        Data types of the sort key columns, in the order of ``by``.

    Returns
    -------
    The sort key columns followed by two ``UInt32`` tie-breaker columns, the
    source partition (chunk sequence number) and the local row number of each
    split candidate, under names that do not clash with ``by``.
    """
    name_gen = unique_names(by)
    part_id_dtype = DataType(pl.UInt32())
    return dict(
        zip(
            [*by, next(name_gen), next(name_gen)],
            [*by_dtypes, part_id_dtype, part_id_dtype],
            strict=True,
        )
    )


async def compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[TableChunk],
    ir: Sort,
    by: list[str],
    num_partitions: int,
    allgather_id: int | None,
) -> DataFrame:
    """
    Compute the range-partition boundaries for a global sort.

    Parameters
    ----------
    context
        The RapidsMPF context.
    comm
        The RapidsMPF communicator.
    ir_context
        The IR execution context.
    local_candidates_list
        This rank's split candidates, one chunk per input chunk, each with the
        schema of :func:`boundary_schema`.
    ir
        The ``Sort`` node providing the key orders and dtypes.
    by
        Names of the sort key columns.
    num_partitions
        Number of output partitions.
    allgather_id
        Collective ID for allgathering the local boundaries across ranks, or
        ``None`` to use this rank's boundaries only (single rank or duplicated
        input).

    Returns
    -------
    The boundaries, with the schema of :func:`boundary_schema` and
    ``num_partitions - 1`` rows in global sort order (empty when
    ``num_partitions <= 1`` or there are no candidates). Every rank returns the
    same boundaries when ``allgather_id`` is given.
    """
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_dtypes = [ir.schema[b] for b in by]
    boundary_ir = Empty(boundary_schema(by, by_dtypes))
    local_boundaries_df = _get_final_sort_boundaries(
        chunk_to_frame(
            await concat_batch(
                local_candidates_list,
                context,
                boundary_ir.schema,
                ir_context,
            )
            if local_candidates_list
            else empty_table_chunk(
                boundary_ir,
                context,
                ir_context.get_cuda_stream(),
            ),
            boundary_ir,
        ),
        column_order,
        null_order,
        num_partitions,
    )
    stream = local_boundaries_df.stream

    if allgather_id is not None:
        chunk = TableChunk.from_pylibcudf_table(
            local_boundaries_df.table,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        allgather = AllGatherManager(context, comm, allgather_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, chunk)
        concat_table = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )
        return _get_final_sort_boundaries(
            DataFrame.from_table(
                concat_table,
                list(boundary_ir.schema.keys()),
                list(boundary_ir.schema.values()),
                stream=stream,
            ),
            column_order,
            null_order,
            num_partitions,
        )
    else:
        return local_boundaries_df


def sort_by_column_names(ir: Sort) -> list[str]:
    """
    Resolve the underlying column names for a ``Sort`` node's keys.

    A sort key's ``NamedExpr.name`` reflects its output alias, which may differ from
    the referenced column after upstream renames. E.g., a join dedup may leave
    ``ORDER BY df2.text`` represented as ``Col('text:df2').alias('text')``. Resolve
    through the underlying ``Col`` so schema lookups use the actual column name.

    Parameters
    ----------
    ir
        The ``Sort`` node.

    Returns
    -------
    The referenced column names, in sort key order.

    Raises
    ------
    NotImplementedError
        If any sort key is not a bare column reference.
    """
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")
    return by


def sort_to_order_keys(ir: Sort) -> list[OrderKey]:
    """
    Convert a ``Sort`` node's keys to order keys.

    Parameters
    ----------
    ir
        The ``Sort`` node.

    Returns
    -------
    One ``OrderKey`` per sort key: the key's column index in ``ir.schema`` with
    its sort order and null order.
    """
    return [
        OrderKey(index, order, null_order)
        for index, order, null_order in zip(
            names_to_indices(tuple(sort_by_column_names(ir)), ir.schema),
            ir.order,
            ir.null_order,
            strict=False,
        )
    ]


def build_order_scheme(
    context: Context,
    order_keys: list[OrderKey],
    sort_boundaries_df: DataFrame,
) -> OrderScheme:
    """
    Build the ``OrderScheme`` metadata describing a range-partitioned sort output.

    Parameters
    ----------
    context
        The RapidsMPF context.
    order_keys
        The output's order keys, as returned by :func:`sort_to_order_keys`.
    sort_boundaries_df
        The boundaries from :func:`compute_sort_boundaries`. Only the leading
        ``len(order_keys)`` key columns are used; the tie-breaker columns are
        dropped.

    Returns
    -------
    A single-``Ordering`` scheme over the boundary keys, with strict boundaries
    when there are no boundaries or no two boundaries share the same key.
    """
    n_keys = len(order_keys)
    stream = sort_boundaries_df.stream
    # sort_boundaries_df will contain a tie-breaker column
    by_table = plc.Table(sort_boundaries_df.table.columns()[:n_keys])
    n_rows = by_table.num_rows()

    strict_boundaries = (
        n_rows == 0
        # TODO: Use unique_count_table
        # Requires https://github.com/NVIDIA/cudf/pull/22487
        or plc.stream_compaction.unique(
            by_table,
            list(range(n_keys)),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            stream=stream,
        ).num_rows()
        == n_rows
    )

    boundaries_chunk = TableChunk.from_pylibcudf_table(
        by_table, stream, exclusive_view=False, br=context.br()
    )
    return OrderScheme(
        [Ordering(order_keys, boundaries_chunk, strict_boundaries=strict_boundaries)]
    )

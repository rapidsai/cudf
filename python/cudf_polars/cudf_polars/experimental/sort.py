# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sorting Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Slice, Sort
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.utils import (
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer


def find_sort_splits(
    tbl: plc.Table,
    sort_boundaries: plc.Table,
    my_part_id: int,
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    stream: Stream,
    *,
    chunk_relative: bool = False,
) -> list[int]:
    """
    Find local sort splits given all (global) split candidates.

    When multiple rows share a boundary value, they go to the later partition.
    The ``part_id`` and ``local_row`` columns in ``sort_boundaries`` are used
    for tiebreaking when ``chunk_relative=False``.

    Parameters
    ----------
    tbl
        Locally sorted table only containing sort columns.
    sort_boundaries
        Sorted table containing the global sort/split boundaries.  Compared to `tbl`
        must contain additional partition_id and local_row_number columns.
    my_part_id
        The partition id of the local node (as the `split_candidates` column).
    column_order
        The order in which tbl is sorted.
    null_order
        The null order in which tbl is sorted.
    stream
        CUDA stream used for device memory operations and kernel launches.
        The values in both ``tbl`` and ``sort_boundaries`` must be valid on
        ``stream``.
    chunk_relative
        If True, when the boundary belongs to this partition (part_id == my_part_id)
        use the position of the boundary value in ``tbl`` (``first``) instead of
        ``local_row``. Use True when ``tbl`` is a chunk of the partition (e.g.
        multiple chunks per rank); False when ``tbl`` is the full partition.

    Returns
    -------
    The split points for the local partition.
    """
    column_order = list(column_order)
    null_order = list(null_order)

    *boundary_cols, split_part_id, split_local_row = sort_boundaries.columns()
    sort_boundaries = plc.Table(boundary_cols)
    split_first_col = plc.search.lower_bound(
        tbl,
        sort_boundaries,
        column_order,
        null_order,
        stream=stream,
    )
    # Use DataFrame.to_polars() for a stream-aware D→H transfer.
    # lower_bound returns size_type (INT32); part_id/local_row are UInt32.
    _u32 = DataType(pl.UInt32())
    df = DataFrame.from_table(
        plc.Table([split_first_col, split_part_id, split_local_row]),
        ["first", "part_id", "local_row"],
        [DataType(pl.Int32()), _u32, _u32],
        stream=stream,
    ).to_polars()
    out = (
        pl.col("first")
        if chunk_relative
        else pl.when(pl.col("part_id") == my_part_id)
        .then(pl.col("local_row"))
        .otherwise(pl.col("first"))
    )
    cap = tbl.num_rows()
    return df.select(out.clip(0, cap).sort()).to_series().to_list()


def _select_local_split_candidates(
    df: DataFrame,
    by: Sequence[str],
    num_partitions: int,
    my_part_id: int,
) -> DataFrame:
    """
    Create a graph that selects the local sort boundaries for a partition.

    Returns a pylibcudf table with the local sort boundaries (including part and
    row id columns).  The columns are already in the order of `by`.
    """
    df = df.select(by)
    name_gen = unique_names(df.column_names)
    part_id_dtype = DataType(pl.UInt32())
    if df.num_rows == 0:
        # Return empty DataFrame with the correct column names and dtypes
        return DataFrame(
            [
                *df.columns,
                Column(
                    plc.column_factories.make_empty_column(
                        part_id_dtype.plc_type, stream=df.stream
                    ),
                    dtype=part_id_dtype,
                    name=next(name_gen),
                ),
                Column(
                    plc.column_factories.make_empty_column(
                        part_id_dtype.plc_type, stream=df.stream
                    ),
                    dtype=part_id_dtype,
                    name=next(name_gen),
                ),
            ],
            stream=df.stream,
        )

    candidates = [i * df.num_rows // num_partitions for i in range(num_partitions)]
    row_id = plc.Column.from_iterable_of_py(
        candidates, part_id_dtype.plc_type, stream=df.stream
    )

    res = plc.copying.gather(
        df.table, row_id, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=df.stream
    )
    part_id = plc.Column.from_scalar(
        plc.Scalar.from_py(my_part_id, part_id_dtype.plc_type, stream=df.stream),
        len(candidates),
        stream=df.stream,
    )

    return DataFrame.from_table(
        plc.Table([*res.columns(), part_id, row_id]),
        [*df.column_names, next(name_gen), next(name_gen)],
        [*df.dtypes, part_id_dtype, part_id_dtype],
        stream=df.stream,
    )


def _get_final_sort_boundaries(
    sort_boundaries_candidates: DataFrame,
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    num_partitions: int,
) -> DataFrame:
    """
    Find the global sort split boundaries from all gathered split candidates.

    Parameters
    ----------
    sort_boundaries_candidates
        All gathered split candidates.
    column_order
        The order in which the split candidates are sorted.
    null_order
        The null order in which the split candidates are sorted.
    num_partitions
        The number of partitions to split the data into.

    Returns
    -------
    sort_boundaries
        Same schema as input (sort keys plus ``partition_id``,
        ``local_row_number``). Empty when ``num_partitions <= 1``
        or the candidate list is empty. Otherwise, the DataFrame
        will contain ``num_partitions - 1`` rows of split
        boundaries in global sort order.
    """
    column_order = list(column_order)
    null_order = list(null_order)

    if num_partitions <= 1 or sort_boundaries_candidates.table.num_rows() == 0:
        return sort_boundaries_candidates.slice((0, 0))  # pragma: no cover

    # The global split candidates need to be stable sorted to find the correct
    # final split points.
    # NOTE: This could be a merge if done earlier (but it should be small data).
    sorted_candidates = plc.sorting.stable_sort(
        sort_boundaries_candidates.table,
        # split candidates has the additional partition_id and row_number columns
        column_order + [plc.types.Order.ASCENDING] * 2,
        null_order + [plc.types.NullOrder.AFTER] * 2,
        stream=sort_boundaries_candidates.stream,
    )
    selected_candidates = plc.Column.from_iterable_of_py(
        [
            i * sorted_candidates.num_rows() // num_partitions
            for i in range(1, num_partitions)
        ],
        stream=sort_boundaries_candidates.stream,
    )
    # Get the actual values at which we will split the data
    sort_boundaries = plc.copying.gather(
        sorted_candidates,
        selected_candidates,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=sort_boundaries_candidates.stream,
    )

    return DataFrame.from_table(
        sort_boundaries,
        sort_boundaries_candidates.column_names,
        sort_boundaries_candidates.dtypes,
        stream=sort_boundaries_candidates.stream,
    )


def _has_simple_zlice(zlice: tuple[int, int | None] | None) -> bool:
    """Check if a zlice is a simple top-k/bottom-k operation."""
    if zlice is None:
        return False
    has_offset = zlice[0] > 0 or (
        zlice[0] < 0 and zlice[1] is not None and zlice[0] + zlice[1] < 0
    )
    return not has_offset


@lower_ir_node.register(Sort)
def _(
    ir: Sort, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check sort keys
    if not all(
        isinstance(expr, Col) for expr in traversal([e.value for e in ir.by])
    ):  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="sort currently only supports column names as `by` keys.",
        )

    if ir.zlice is not None and not _has_simple_zlice(ir.zlice):
        # Pull "complex" slices out of the Sort node altogether.
        return rec(
            Slice(
                ir.schema,
                *ir.zlice,
                Sort(
                    ir.schema,
                    ir.by,
                    ir.order,
                    ir.null_order,
                    ir.stable,
                    None,
                    ir.children[0],
                ),
            )
        )

    # Extract child partitioning
    child, partition_info = rec(ir.children[0])

    sort_node = ir.reconstruct([child])
    partition_info[sort_node] = partition_info[child]
    return sort_node, partition_info

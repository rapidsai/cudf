# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sorting Logic."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import polars as pl

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import IR, Sort
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import _simple_shuffle_graph
from cudf_polars.experimental.utils import _concat, _fallback_inform, _lower_ir_fallback
from cudf_polars.utils.config import ShuffleMethod
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream, get_joined_cuda_stream

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema


def find_sort_splits(
    tbl: plc.Table,
    sort_boundaries: plc.Table,
    my_part_id: int,
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    stream: Stream,
) -> list[int]:
    """
    Find local sort splits given all (global) split candidates.

    The reason for much of the complexity is to get the result sizes as
    precise as possible even when e.g. all values are equal.
    In other words, this goes through extra effort to split the data at the
    precise boundaries (which includes part_id and local_row_number).

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

    Returns
    -------
    The split points for the local partition.
    """
    column_order = list(column_order)
    null_order = list(null_order)

    # We now need to find the local split points.  To do this, first split out
    # the partition id and the local row number of the final split values
    *boundary_cols, split_part_id, split_local_row = sort_boundaries.columns()
    sort_boundaries = plc.Table(boundary_cols)
    # Now we find the first and last row in the local table corresponding to the split value
    # (first and last, because there may be multiple rows with the same split value)
    split_first_col = plc.search.lower_bound(
        tbl,
        sort_boundaries,
        column_order,
        null_order,
        stream=stream,
    )
    split_last_col = plc.search.upper_bound(
        tbl,
        sort_boundaries,
        column_order,
        null_order,
        stream=stream,
    )
    # And convert to list for final processing
    # The type ignores are for cross-library boundaries: plc.Column -> pl.Series
    # These work at runtime via the Arrow C Data Interface protocol
    # TODO: Find a way for pylibcudf types to show they export the Arrow protocol
    # (mypy wasn't happy with a custom protocol)
    split_first_list = pl.Series(split_first_col).to_list()
    split_last_list = pl.Series(split_last_col).to_list()
    split_part_id_list = pl.Series(split_part_id).to_list()
    split_local_row_list = pl.Series(split_local_row).to_list()

    # Find the final split points.  This is slightly tricky because of the possibility
    # of equal values, which is why we need the part_id and local_row.
    # Consider for example the case when all data is equal.
    split_points = []
    for first, last, part_id, local_row in zip(
        split_first_list,
        split_last_list,
        split_part_id_list,
        split_local_row_list,
        strict=False,
    ):
        if part_id < my_part_id:
            # Local data is globally later so split at first valid row.
            split_points.append(first)
        elif part_id > my_part_id:
            # Local data is globally earlier so split after last valid row.
            split_points.append(last)
        else:
            # The split point is within our chunk, so use original local row
            split_points.append(local_row)

    return split_points


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

    """
    column_order = list(column_order)
    null_order = list(null_order)

    # The global split candidates need to be stable sorted to find the correct
    # final split points.
    # NOTE: This could be a merge if done earlier (but it should be small data).
    sorted_candidates = plc.sorting.sort(
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


def _sort_boundaries_graph(
    name_in: str,
    by: Sequence[str],
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    count: int,
    context: IRExecutionContext,
) -> tuple[str, MutableMapping[Any, Any]]:
    """Graph to get the boundaries from all partitions."""
    local_boundaries_name = f"sort-boundaries_local-{name_in}"
    concat_boundaries_name = f"sort-boundaries-concat-{name_in}"
    global_boundaries_name = f"sort-boundaries-{name_in}"
    graph: MutableMapping[Any, Any] = {}

    _concat_list = []
    for part_id in range(count):
        graph[(local_boundaries_name, part_id)] = (
            _select_local_split_candidates,
            (name_in, part_id),
            by,
            count,
            part_id,
        )
        _concat_list.append((local_boundaries_name, part_id))

    graph[concat_boundaries_name] = (partial(_concat, context=context), *_concat_list)
    graph[global_boundaries_name] = (
        _get_final_sort_boundaries,
        concat_boundaries_name,
        column_order,
        null_order,
        count,
    )
    return global_boundaries_name, graph


class SortedShuffleOptions(TypedDict):
    """RapidsMPF shuffling options."""

    by: Sequence[str]
    order: Sequence[plc.types.Order]
    null_order: Sequence[plc.types.NullOrder]
    column_names: Sequence[str]
    column_dtypes: Sequence[DataType]
    cluster_kind: Literal["dask", "single"]


# Experimental rapidsmpf shuffler integration
class RMPFIntegrationSortedShuffle:  # pragma: no cover
    """cuDF-Polars protocol for rapidsmpf shuffler."""

    @staticmethod
    def insert_partition(
        df: DataFrame,
        partition_id: int,
        partition_count: int,
        shuffler: Any,
        options: SortedShuffleOptions,
        sort_boundaries: DataFrame,
    ) -> None:
        """Add cudf-polars DataFrame chunks to an RMP shuffler."""
        from rapidsmpf.integrations.cudf.partition import split_and_pack

        if options["cluster_kind"] == "dask":
            from rapidsmpf.integrations.dask import get_worker_context

        else:
            from rapidsmpf.integrations.single import get_worker_context

        context = get_worker_context()

        by = options["by"]

        stream = get_joined_cuda_stream(
            get_dask_cuda_stream, upstreams=(df.stream, sort_boundaries.stream)
        )

        splits = find_sort_splits(
            df.select(by).table,
            sort_boundaries.table,
            partition_id,
            options["order"],
            options["null_order"],
            stream=stream,
        )
        packed_inputs = split_and_pack(
            df.table,
            splits=splits,
            br=context.br,
            stream=stream,
        )
        # TODO: figure out handoff with rapidsmpf
        # https://github.com/rapidsai/cudf/issues/20337
        shuffler.insert_chunks(packed_inputs)

    @staticmethod
    def extract_partition(
        partition_id: int,
        shuffler: Any,
        options: SortedShuffleOptions,
    ) -> DataFrame:
        """Extract a finished partition from the RMP shuffler."""
        from rapidsmpf.integrations.cudf.partition import (
            unpack_and_concat,
            unspill_partitions,
        )

        if options["cluster_kind"] == "dask":
            from rapidsmpf.integrations.dask import get_worker_context

        else:
            from rapidsmpf.integrations.single import get_worker_context

        context = get_worker_context()

        shuffler.wait_on(partition_id)
        column_names = options["column_names"]
        column_dtypes = options["column_dtypes"]

        stream = DEFAULT_STREAM

        # TODO: When sorting, this step should finalize with a merge (unless we
        # require stability, as cudf merge is not stable).
        # TODO: figure out handoff with rapidsmpf
        # https://github.com/rapidsai/cudf/issues/20337
        return DataFrame.from_table(
            unpack_and_concat(
                unspill_partitions(
                    shuffler.extract(partition_id),
                    br=context.br,
                    allow_overbooking=True,
                    statistics=context.statistics,
                ),
                br=context.br,
                stream=stream,
            ),
            column_names,
            column_dtypes,
            stream=stream,
        )


def _sort_partition_dataframe(
    df: DataFrame,
    partition_id: int,  # Not currently used
    partition_count: int,
    options: MutableMapping[str, Any],
    sort_boundaries: DataFrame,
) -> MutableMapping[int, DataFrame]:
    """
    Partition a sorted DataFrame for shuffling.

    Parameters
    ----------
    df
        The DataFrame to partition.
    partition_id
        The partition id of the current partition.
    partition_count
        The total number of partitions.
    options
        The sort options ``(by, order, null_order)``.
    sort_boundaries
        The global sort boundary candidates used to decide where to split.
    """
    if df.num_rows == 0:  # pragma: no cover
        # Fast path for empty DataFrame
        return dict.fromkeys(range(partition_count), df)

    stream = get_joined_cuda_stream(
        get_dask_cuda_stream, upstreams=(df.stream, sort_boundaries.stream)
    )

    splits = find_sort_splits(
        df.select(options["by"]).table,
        sort_boundaries.table,
        partition_id,
        options["order"],
        options["null_order"],
        stream=stream,
    )

    # Split and return the partitioned result
    return {
        i: DataFrame.from_table(
            split,
            df.column_names,
            df.dtypes,
            stream=df.stream,
        )
        for i, split in enumerate(plc.copying.split(df.table, splits, stream=stream))
    }


class ShuffleSorted(IR):
    """
    Shuffle already locally sorted multi-partition data.

    Shuffling is performed by extracting sort boundary candidates from all partitions,
    sharing them all-to-all and then exchanging data accordingly.
    The sorting information is required to be passed in identically to the already
    performed local sort and as of now the final result needs to be sorted again to
    merge the partitions.
    """

    __slots__ = ("by", "null_order", "order", "shuffle_method")
    _non_child = ("schema", "by", "order", "null_order", "shuffle_method")
    by: tuple[NamedExpr, ...]
    """Keys by which the data was sorted."""
    order: tuple[plc.types.Order, ...]
    """Sort order if sorted."""
    null_order: tuple[plc.types.NullOrder, ...]
    """Null precedence if sorted."""
    shuffle_method: ShuffleMethod
    """Shuffle method to use."""

    def __init__(
        self,
        schema: Schema,
        by: tuple[NamedExpr, ...],
        order: tuple[plc.types.Order, ...],
        null_order: tuple[plc.types.NullOrder, ...],
        shuffle_method: ShuffleMethod,
        df: IR,
    ):
        self.schema = schema
        self.by = by
        self.order = order
        self.null_order = null_order
        self.shuffle_method = shuffle_method
        self._non_child_args = (schema, by, order, null_order, shuffle_method)
        self.children = (df,)

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        by: tuple[NamedExpr, ...],
        order: tuple[plc.types.Order, ...],
        null_order: tuple[plc.types.NullOrder, ...],
        shuffle_method: ShuffleMethod,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition ShuffleSorted evaluation is a no-op
        return df


@lower_ir_node.register(Sort)
def _(
    ir: Sort, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Special handling for slicing
    # (May be a top- or bottom-k operation)

    if ir.zlice is not None:
        # TODO: Handle large slices (e.g. 1m+ rows), this should go into the branch
        # below, but will require additional logic there.

        # Check if zlice has an offset, i.e. includes the start or reaches the end.
        # If an offset exists it would be incorrect to apply in the first pwise sort.
        has_offset = ir.zlice[0] > 0 or (
            ir.zlice[0] < 0
            and ir.zlice[1] is not None
            and ir.zlice[0] + ir.zlice[1] < 0
        )
        if has_offset:
            return _lower_ir_fallback(
                ir,
                rec,
                msg="Sort does not support a multi-partition slice with an offset.",
            )

        from cudf_polars.experimental.parallel import _lower_ir_pwise

        # Sort input partitions
        new_node, partition_info = _lower_ir_pwise(ir, rec)
        if partition_info[new_node].count > 1:
            # Collapse down to single partition
            inter = Repartition(new_node.schema, new_node)
            partition_info[inter] = PartitionInfo(count=1)
            # Sort reduced partition
            new_node = ir.reconstruct([inter])
            partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    # Check sort keys
    if not all(
        isinstance(expr, Col) for expr in traversal([e.value for e in ir.by])
    ):  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="sort currently only supports column names as `by` keys.",
        )

    # Extract child partitioning
    child, partition_info = rec(ir.children[0])

    # Extract shuffle method
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )
    # Avoid rapidsmpf shuffle with maintain_order=True (for now)
    shuffle_method = (
        ShuffleMethod("tasks") if ir.stable else config_options.executor.shuffle_method
    )
    if (
        shuffle_method != config_options.executor.shuffle_method
    ):  # pragma: no cover; Requires rapidsmpf
        _fallback_inform(
            f"shuffle_method={shuffle_method} does not support maintain_order=True. "
            "Falling back to shuffle_method='tasks'.",
            config_options,
        )

    # Handle single-partition case
    if partition_info[child].count == 1:
        single_part_node = ir.reconstruct([child])
        partition_info[single_part_node] = partition_info[child]
        return single_part_node, partition_info

    local_sort_node = ir.reconstruct([child])
    partition_info[local_sort_node] = partition_info[child]

    shuffle = ShuffleSorted(
        ir.schema,
        ir.by,
        ir.order,
        ir.null_order,
        shuffle_method,
        local_sort_node,
    )
    partition_info[shuffle] = partition_info[child]

    # We sort again locally.
    assert ir.zlice is None  # zlice handling would be incorrect without adjustment
    final_sort_node = ir.reconstruct([shuffle])
    partition_info[final_sort_node] = partition_info[shuffle]

    return final_sort_node, partition_info


@generate_ir_tasks.register(ShuffleSorted)
def _(
    ir: ShuffleSorted,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):  # pragma: no cover
        # We should not reach here as this is checked in the lower_ir_node
        raise NotImplementedError("Sorting columns must be column names.")

    (child,) = ir.children

    sort_boundaries_name, graph = _sort_boundaries_graph(
        get_key_name(child),
        by,
        ir.order,
        ir.null_order,
        partition_info[child].count,
        context,
    )

    options = {
        "by": by,
        "order": ir.order,
        "null_order": ir.null_order,
        "column_names": list(ir.schema.keys()),
        "column_dtypes": list(ir.schema.values()),
    }

    # Try using rapidsmpf shuffler if we have "simple" shuffle
    # keys, and the "shuffle_method" config is set to "rapidsmpf"
    shuffle_method = ir.shuffle_method
    if shuffle_method in ("rapidsmpf", "rapidsmpf-single"):  # pragma: no cover
        try:
            if shuffle_method == "rapidsmpf-single":
                from rapidsmpf.integrations.single import rapidsmpf_shuffle_graph

                options["cluster_kind"] = "single"
            else:
                from rapidsmpf.integrations.dask import rapidsmpf_shuffle_graph

                options["cluster_kind"] = "dask"
            graph.update(
                rapidsmpf_shuffle_graph(
                    get_key_name(child),
                    get_key_name(ir),
                    partition_info[child].count,
                    partition_info[ir].count,
                    RMPFIntegrationSortedShuffle,
                    options,
                    sort_boundaries_name,
                )
            )
        except (ImportError, ValueError) as err:
            # ImportError: rapidsmpf is not installed
            # ValueError: rapidsmpf couldn't find a distributed client
            if shuffle_method == "rapidsmpf":  # pragma: no cover
                # Only raise an error if the user specifically
                # set the shuffle method to "rapidsmpf"
                raise ValueError(
                    "Rapidsmpf is not installed correctly or the current "
                    "Dask cluster does not support rapidsmpf shuffling."
                ) from err
        else:
            return graph

    # Simple task-based fall-back
    graph.update(
        partial(_simple_shuffle_graph, context=context)(
            get_key_name(child),
            get_key_name(ir),
            partition_info[child].count,
            partition_info[ir].count,
            _sort_partition_dataframe,
            options,
            sort_boundaries_name,
        )
    )
    return graph

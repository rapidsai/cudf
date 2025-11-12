# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeVar, TypedDict

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import IR
from cudf_polars.dsl.tracing import log_do_evaluate, nvtx_annotate_cudf_polars
from cudf_polars.experimental.base import get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.utils import _concat
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ShuffleMethod


# Supported shuffle methods
_SHUFFLE_METHODS = ("rapidsmpf", "tasks")


class ShuffleOptions(TypedDict):
    """RapidsMPF shuffling options."""

    on: Sequence[str]
    column_names: Sequence[str]
    dtypes: Sequence[DataType]
    cluster_kind: Literal["dask", "single"]


# Experimental rapidsmpf shuffler integration
class RMPFIntegration:  # pragma: no cover
    """cuDF-Polars protocol for rapidsmpf shuffler."""

    @staticmethod
    @nvtx_annotate_cudf_polars(message="RMPFIntegration.insert_partition")
    def insert_partition(
        df: DataFrame,
        partition_id: int,  # Not currently used
        partition_count: int,
        shuffler: Any,
        options: ShuffleOptions,
        *other: Any,
    ) -> None:
        """Add cudf-polars DataFrame chunks to an RMP shuffler."""
        from rapidsmpf.integrations.cudf.partition import partition_and_pack

        if options["cluster_kind"] == "dask":
            from rapidsmpf.integrations.dask import get_worker_context

        else:
            from rapidsmpf.integrations.single import get_worker_context

        context = get_worker_context()

        on = options["on"]
        assert not other, f"Unexpected arguments: {other}"
        columns_to_hash = tuple(df.column_names.index(val) for val in on)
        packed_inputs = partition_and_pack(
            df.table,
            columns_to_hash=columns_to_hash,
            num_partitions=partition_count,
            br=context.br,
            stream=DEFAULT_STREAM,
        )
        shuffler.insert_chunks(packed_inputs)

    @staticmethod
    @nvtx_annotate_cudf_polars(message="RMPFIntegration.extract_partition")
    def extract_partition(
        partition_id: int,
        shuffler: Any,
        options: ShuffleOptions,
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
        dtypes = options["dtypes"]
        return DataFrame.from_table(
            unpack_and_concat(
                unspill_partitions(
                    shuffler.extract(partition_id),
                    br=context.br,
                    allow_overbooking=True,
                    statistics=context.statistics,
                ),
                br=context.br,
                stream=DEFAULT_STREAM,
            ),
            column_names,
            dtypes,
            get_dask_cuda_stream(),
        )


class Shuffle(IR):
    """
    Shuffle multi-partition data.

    Notes
    -----
    Only hash-based partitioning is supported (for now).  See
    `ShuffleSorted` for sorting-based shuffling.
    """

    __slots__ = ("keys", "shuffle_method")
    _non_child = ("schema", "keys", "shuffle_method")
    keys: tuple[NamedExpr, ...]
    """Keys to shuffle on."""
    shuffle_method: ShuffleMethod
    """Shuffle method to use."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        shuffle_method: ShuffleMethod,
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self.shuffle_method = shuffle_method
        self._non_child_args = (schema, keys, shuffle_method)
        self.children = (df,)

    # the type-ignore is for
    # Argument 1 to "log_do_evaluate" has incompatible type "Callable[[type[Shuffle], <snip>]"
    #    expected Callable[[type[IR], <snip>]
    # But Shuffle is a subclass of IR, so this is fine.
    @classmethod  # type: ignore[arg-type]
    @log_do_evaluate
    def do_evaluate(
        cls,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        shuffle_method: ShuffleMethod,
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition Shuffle evaluation is a no-op
        return df


@nvtx_annotate_cudf_polars(message="Shuffle")
def _hash_partition_dataframe(
    df: DataFrame,
    partition_id: int,  # Used only by sorted shuffling
    partition_count: int,
    options: MutableMapping[str, Any] | None,  # No options required
    on: tuple[NamedExpr, ...],
) -> dict[int, DataFrame]:
    """
    Partition an input DataFrame for hash-based shuffling.

    Parameters
    ----------
    df
        DataFrame to partition.
    partition_id
        Partition index (unused for hash partitioning).
    partition_count
        Total number of output partitions.
    options
        Options (unused for hash partitioning).
    on
        Expressions used for the hash partitioning.

    Returns
    -------
    A dictionary mapping between int partition indices and
    DataFrame fragments.
    """
    assert not options, f"Expected no options, got: {options}"

    if df.num_rows == 0:
        # Fast path for empty DataFrame
        return dict.fromkeys(range(partition_count), df)

    # Hash the specified keys to calculate the output
    # partition for each row
    partition_map = plc.binaryop.binary_operation(
        plc.hashing.murmurhash3_x86_32(
            DataFrame([expr.evaluate(df) for expr in on], stream=df.stream).table,
            stream=df.stream,
        ),
        plc.Scalar.from_py(
            partition_count, plc.DataType(plc.TypeId.UINT32), stream=df.stream
        ),
        plc.binaryop.BinaryOperator.PYMOD,
        plc.types.DataType(plc.types.TypeId.UINT32),
        stream=df.stream,
    )

    # Apply partitioning
    t, offsets = plc.partitioning.partition(
        df.table,
        partition_map,
        partition_count,
        stream=df.stream,
    )
    splits = offsets[1:-1]

    # Split and return the partitioned result
    return {
        i: DataFrame.from_table(
            split,
            df.column_names,
            df.dtypes,
            df.stream,
        )
        for i, split in enumerate(plc.copying.split(t, splits, stream=df.stream))
    }


# When dropping Python 3.10, can use _simple_shuffle_graph[OPT_T](...)
OPT_T = TypeVar("OPT_T")


def _simple_shuffle_graph(
    name_in: str,
    name_out: str,
    count_in: int,
    count_out: int,
    _partition_dataframe_func: Callable[
        Concatenate[DataFrame, int, int, OPT_T, ...],
        MutableMapping[int, DataFrame],
    ],
    options: OPT_T,
    *other: Any,
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    """Make a simple all-to-all shuffle graph."""
    split_name = f"split-{name_out}"
    inter_name = f"inter-{name_out}"

    graph: MutableMapping[Any, Any] = {}
    for part_out in range(count_out):
        _concat_list = []
        for part_in in range(count_in):
            graph[(split_name, part_in)] = (
                _partition_dataframe_func,
                (name_in, part_in),
                part_in,
                count_out,
                options,
                *other,
            )
            _concat_list.append((inter_name, part_out, part_in))
            graph[_concat_list[-1]] = (
                operator.getitem,
                (split_name, part_in),
                part_out,
            )
        graph[(name_out, part_out)] = (partial(_concat, context=context), *_concat_list)
    return graph


@lower_ir_node.register(Shuffle)
def _(
    ir: Shuffle, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple lower_ir_node handling for the default hash-based shuffle.
    # More-complex logic (e.g. joining and sorting) should
    # be handled separately.
    from cudf_polars.experimental.parallel import PartitionInfo

    (child,) = ir.children

    new_child, pi = rec(child)
    if pi[new_child].count == 1 or ir.keys == pi[new_child].partitioned_on:
        # Already shuffled
        return new_child, pi
    new_node = ir.reconstruct([new_child])
    pi[new_node] = PartitionInfo(
        # Default shuffle preserves partition count
        count=pi[new_child].count,
        # Add partitioned_on info
        partitioned_on=ir.keys,
    )
    return new_node, pi


@generate_ir_tasks.register(Shuffle)
def _(
    ir: Shuffle,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    # Extract "shuffle_method" configuration
    shuffle_method = ir.shuffle_method

    # Try using rapidsmpf shuffler if we have "simple" shuffle
    # keys, and the "shuffle_method" config is set to "rapidsmpf"
    _keys: list[Col]
    if shuffle_method in ("rapidsmpf", "rapidsmpf-single") and len(
        _keys := [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    ) == len(ir.keys):  # pragma: no cover
        cluster_kind: Literal["dask", "single"]
        if shuffle_method == "rapidsmpf-single":
            from rapidsmpf.integrations.single import rapidsmpf_shuffle_graph

            cluster_kind = "single"
        else:
            from rapidsmpf.integrations.dask import rapidsmpf_shuffle_graph

            cluster_kind = "dask"

        shuffle_on = [k.name for k in _keys]

        try:
            return rapidsmpf_shuffle_graph(
                get_key_name(ir.children[0]),
                get_key_name(ir),
                partition_info[ir.children[0]].count,
                partition_info[ir].count,
                RMPFIntegration,
                {
                    "on": shuffle_on,
                    "column_names": list(ir.schema.keys()),
                    "dtypes": list(ir.schema.values()),
                    "cluster_kind": cluster_kind,
                },
            )
        except ValueError as err:
            # ValueError: rapidsmpf couldn't find a distributed client
            if shuffle_method == "rapidsmpf":
                # Only raise an error if the user specifically
                # set the shuffle method to "rapidsmpf"
                raise ValueError(
                    "The current Dask cluster does not support rapidsmpf shuffling."
                ) from err

    # Simple task-based fall-back
    return partial(_simple_shuffle_graph, context=context)(
        get_key_name(ir.children[0]),
        get_key_name(ir),
        partition_info[ir.children[0]].count,
        partition_info[ir].count,
        _hash_partition_dataframe,
        None,
        ir.keys,
    )

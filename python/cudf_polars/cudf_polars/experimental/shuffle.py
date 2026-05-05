# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR
from cudf_polars.dsl.tracing import log_do_evaluate, nvtx_annotate_cudf_polars
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.utils import _dynamic_planning_on
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class ShuffleOptions(TypedDict):
    """RapidsMPF shuffling options."""

    on: Sequence[str]
    column_names: Sequence[str]
    dtypes: Sequence[DataType]


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
        from rapidsmpf.integrations.single import get_worker_context

        context = get_worker_context()

        shuffler.wait()
        column_names = options["column_names"]
        dtypes = options["dtypes"]
        return DataFrame.from_table(
            unpack_and_concat(
                unspill_partitions(
                    shuffler.extract(partition_id),
                    br=context.br,
                    allow_overbooking=True,
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
    Only hash-based partitioning is supported (for now).
    """

    __slots__ = ("keys",)
    _non_child = (
        "schema",
        "keys",
    )
    _n_non_child_args = 2
    keys: tuple[NamedExpr, ...]
    """Keys to shuffle on."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self._non_child_args = (schema, keys)
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
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition Shuffle evaluation is a no-op
        return df


@lower_ir_node.register(Shuffle)
def _(
    ir: Shuffle, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple lower_ir_node handling for the default hash-based shuffle.
    # More-complex logic (e.g. joining and sorting) should
    # be handled separately.
    from cudf_polars.experimental.parallel import PartitionInfo

    (child,) = ir.children

    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    new_child, pi = rec(child)
    already_partitioned = ir.keys == pi[new_child].partitioned_on
    single_partition = pi[new_child].count == 1 and not _dynamic_planning_on(
        config_options
    )
    if single_partition or already_partitioned:
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

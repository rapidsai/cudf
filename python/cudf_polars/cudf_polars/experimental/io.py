# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel IO Logic."""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


##
## DataFrameScan
##


class ParDataFrameScan(DataFrameScan):
    """Parallel DataFrameScan."""

    @property
    def _max_n_rows(self) -> int:
        """Row-count threshold for splitting a DataFrame."""
        parallel_options = self.config_options.get("parallel_options", {})
        return parallel_options.get("num_rows_threshold", 1_000_000)

    @cached_property
    def _count(self) -> int:
        """Partition count."""
        total_rows = max(self.df.shape()[0], 1)
        return math.ceil(total_rows / self._max_n_rows)

    def _tasks(self) -> MutableMapping[Any, Any]:
        """Task graph."""
        total_rows = max(self.df.shape()[0], 1)
        stride = math.ceil(total_rows / self._count)
        key_name = get_key_name(self)
        return {
            (key_name, i): (
                self.do_evaluate,
                self.schema,
                self.df.slice(offset, stride),
                self.projection,
                self.predicate,
            )
            for i, offset in enumerate(range(0, total_rows, stride))
        }


def lower_dataframescan_node(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a Scan node with proper partitioning."""
    new_node = ParDataFrameScan(
        ir.schema,
        ir.df,
        ir.projection,
        ir.predicate,
        ir.config_options,
    )
    return new_node, {new_node: PartitionInfo(count=new_node._count)}


@generate_ir_tasks.register(ParDataFrameScan)
def _(
    ir: ParDataFrameScan, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    assert (
        partition_info[ir].count == ir._count
    ), "Inconsistent ParDataFrameScan partitioning."
    return ir._tasks()

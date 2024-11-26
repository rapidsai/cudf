# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel IO Logic."""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import polars as pl

from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.experimental.parallel import (
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import PartitionInfo


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

    def _tasks(
        self, partition_info: MutableMapping[IR, PartitionInfo]
    ) -> MutableMapping[Any, Any]:
        """Task graph."""
        assert (
            partition_info[self].count == self._count
        ), "Inconsistent ParDataFrameScan partitioning."
        total_rows = max(self.df.shape()[0], 1)
        stride = math.ceil(total_rows / self._count)
        key_name = get_key_name(self)
        return {
            (key_name, i): (
                self.do_evaluate,
                self.schema,
                pl.DataFrame._from_pydf(self.df.slice(offset, stride)),
                self.projection,
                self.predicate,
            )
            for i, offset in enumerate(range(0, total_rows, stride))
        }


@generate_ir_tasks.register(ParDataFrameScan)
def _(
    ir: ParDataFrameScan, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return ir._tasks(partition_info)

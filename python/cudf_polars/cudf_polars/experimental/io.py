# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import DataFrameScan, Union
from cudf_polars.experimental.parallel import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer, PartitionInfo


##
## DataFrameScan
##


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    rows_per_partition = ir.config_options.get("executor_options", {}).get(
        "num_rows_threshold", 1_000_000
    )

    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    if count > 1:
        length = math.ceil(nrows / count)
        slices = [
            DataFrameScan(
                ir.schema,
                ir.df.slice(offset, length),
                ir.projection,
                ir.predicate,
                ir.config_options,
            )
            for offset in range(0, nrows, length)
        ]
        return rec(Union(ir.schema, None, *slices))

    return rec.state["default_mapper"](ir)

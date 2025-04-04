# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.base import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer


def _concat(*dfs: DataFrame) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)


def _lower_ir_fallback(
    ir: IR,
    rec: LowerIRTransformer,
    *,
    msg: str | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Catch-all single-partition lowering logic.
    # If any children contain multiple partitions,
    # those children will be collapsed with `Repartition`.
    from cudf_polars.experimental.repartition import Repartition

    # Lower children
    _children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    fallback = False
    for c in _children:
        child = c
        if partition_info[c].count > 1:
            # Fall-back logic
            fallback = True
            child = Repartition(child.schema, child)
            partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if fallback and msg:
        # Warn/raise the user if any children were collapsed
        # and the "fallback_mode" configuration is not "silent"
        fallback_mode = rec.state["config_options"].get(
            "executor_options.fallback_mode", default="warn"
        )
        if fallback_mode == "warn":
            warnings.warn(msg, stacklevel=2)
        elif fallback_mode == "raise":
            raise NotImplementedError(msg)
        elif fallback_mode != "silent":
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' option. "
                "Please use 'warn', 'raise', or 'silent'."
            )

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info

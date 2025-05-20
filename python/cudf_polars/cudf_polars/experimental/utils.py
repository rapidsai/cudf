# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.base import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def _concat(*dfs: DataFrame) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)


def _fallback_inform(msg: str, config_options: ConfigOptions) -> None:
    """Inform the user of single-partition fallback."""
    # Satisfy type checking
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in '_fallback_inform'"
    )

    match fallback_mode := config_options.executor.fallback_mode:
        case "warn":
            warnings.warn(msg, stacklevel=2)
        case "raise":
            raise NotImplementedError(msg)
        case "silent":
            pass
        case _:  # pragma: no cover; Should never get here.
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' "
                "option. Please use 'warn', 'raise', or 'silent'."
            )


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
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    fallback = False
    for c in lowered_children:
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
        _fallback_inform(msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info


def _leaf_column_names(expr: Expr) -> tuple[str, ...]:
    """Find the leaf column names of an expression."""
    if expr.children:
        return tuple(
            chain.from_iterable(_leaf_column_names(child) for child in expr.children)
        )
    elif isinstance(expr, Col):
        return (expr.name,)
    else:
        return ()

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.dsl.expr import Col, Len
from cudf_polars.dsl.ir import Empty, HConcat, Scan, Select, Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.expressions import decompose_expr_graph
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def decompose_select(
    select_ir: Select,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a multi-partition Select operation.

    Parameters
    ----------
    select_ir
        The original Select operation to decompose.
        This object has not been reconstructed with
        ``input_ir`` as its child yet.
    input_ir
        The lowered child of ``select_ir``. This object
        will be decomposed into a "partial" selection
        for each element of  ``select_ir.exprs``.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten Select node, and a mapping from
        unique nodes in the new graph to associated
        partitioning information.

    Notes
    -----
    This function uses ``decompose_expr_graph`` to further
    decompose each element of  ``select_ir.exprs``.

    See Also
    --------
    decompose_expr_graph
    """
    # Collect partial selections
    selections = []
    for ne in select_ir.exprs:
        # Decompose this partial expression
        new_ne, partial_input_ir, _partition_info = decompose_expr_graph(
            ne, input_ir, partition_info, config_options
        )
        pi = _partition_info[partial_input_ir]
        partial_input_ir = Select(
            {ne.name: ne.value.dtype},
            [new_ne],
            True,  # noqa: FBT003
            partial_input_ir,
        )
        _partition_info[partial_input_ir] = pi
        partition_info.update(_partition_info)
        selections.append(partial_input_ir)

    # Concatenate partial selections
    new_ir: HConcat | Select
    if len(selections) > 1:
        new_ir = HConcat(
            select_ir.schema,
            True,  # noqa: FBT003
            *selections,
        )
        partition_info[new_ir] = PartitionInfo(
            count=max(partition_info[c].count for c in selections)
        )
    else:
        new_ir = selections[0]

    return new_ir, partition_info


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]
    if (
        pi.count == 1
        and Select._is_len_expr(ir.exprs)
        and isinstance(child, Union)
        and len(child.children) == 1
        and isinstance(child.children[0], Scan)
        and child.children[0].predicate is None
    ):
        # Special Case: Fast count.
        scan = child.children[0]
        count = scan.fast_count()
        dtype = ir.exprs[0].value.dtype

        lit_expr = expr.LiteralColumn(
            dtype, pl.Series(values=[count], dtype=dtype.polars)
        )
        named_expr = expr.NamedExpr(ir.exprs[0].name or "len", lit_expr)

        new_node = Select(
            {named_expr.name: named_expr.value.dtype},
            [named_expr],
            should_broadcast=True,
            df=child,
        )
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    if not any(
        isinstance(expr, (Col, Len)) for expr in traversal([e.value for e in ir.exprs])
    ):
        # Special Case: Selection does not depend on any columns.
        new_node = ir.reconstruct([input_ir := Empty()])
        partition_info[input_ir] = partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    if pi.count > 1 and not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    ):
        # Special Case: Multiple partitions with 1+ non-pointwise expressions.
        try:
            # Try decomposing the underlying expressions
            return decompose_select(
                ir, child, partition_info, rec.state["config_options"]
            )
        except NotImplementedError:
            return _lower_ir_fallback(
                ir, rec, msg="This selection is not supported for multiple partitions."
            )

    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info

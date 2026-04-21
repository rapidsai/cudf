# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Over IR node for streaming window expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from cudf_polars.dsl.expr import Col, GroupedWindow, NamedExpr
from cudf_polars.dsl.ir import IR, GroupBy, HStack, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.groupby import combine, decompose
from cudf_polars.experimental.utils import _all_over_scalar_and_top_level

if TYPE_CHECKING:
    from cudf_polars.typing import Schema
    from cudf_polars.dsl.ir import HStack


def _build_over_groupby_irs(
    gw_nodes: tuple[GroupedWindow, ...],
    child_ir: IR,
) -> tuple[GroupBy, GroupBy, Select | None]:
    """
    Build piecewise, reduction, and (optionally) selection GroupBy IRs.

    Parameters
    ----------
    gw_nodes
        Top-level GroupedWindow nodes sharing the same partition-by keys;
        all must be scalar (Agg/Len only in named_aggs).
    child_ir
        Child IR of the enclosing Select/HStack; defines the input schema.

    Returns
    -------
    piecewise_ir
        GroupBy IR that computes partial aggregates per chunk.
    reduction_ir
        GroupBy IR that reduces partial aggregates to a single result.
    agg_select_ir
        Select IR for post-aggregation expressions (e.g. division for mean);
        None when all aggregations are pass-through.
    """
    gw = gw_nodes[0]
    by_exprs = cast("list[Col]", list(gw.children[: gw.by_count]))
    key_named_exprs = [NamedExpr(e.name, e) for e in by_exprs]
    key_schema = {e.name: child_ir.schema[e.name] for e in by_exprs}

    seen: set[str] = set()
    all_scalar_named: list[NamedExpr] = []
    for gw_node in gw_nodes:
        reductions, _ = gw_node._split_named_expr()
        for ne in reductions:
            if ne.name not in seen:
                all_scalar_named.append(ne)
                seen.add(ne.name)

    name_gen = unique_names(child_ir.schema.keys())
    decompositions = [
        decompose(ne.name, ne.value, names=name_gen) for ne in all_scalar_named
    ]
    selection_exprs, piecewise_exprs, reduction_exprs, _ = combine(*decompositions)

    pwise_schema = dict(key_schema) | {
        ne.name: ne.value.dtype for ne in piecewise_exprs
    }
    piecewise_ir = GroupBy(
        pwise_schema,
        key_named_exprs,
        piecewise_exprs,
        False,  # noqa: FBT003
        None,
        child_ir,
    )

    reduction_key_exprs = [
        NamedExpr(ne.name, Col(pwise_schema[ne.name], ne.name))
        for ne in key_named_exprs
    ]
    reduction_schema = {ne.name: pwise_schema[ne.name] for ne in key_named_exprs} | {
        ne.name: ne.value.dtype for ne in reduction_exprs
    }
    reduction_ir = GroupBy(
        reduction_schema,
        reduction_key_exprs,
        reduction_exprs,
        False,  # noqa: FBT003
        None,
        piecewise_ir,
    )

    agg_select_ir: Select | None
    if any(
        ne.name not in reduction_schema or reduction_schema[ne.name] != ne.value.dtype
        for ne in selection_exprs
    ):
        select_key_exprs = [
            NamedExpr(ne.name, Col(reduction_schema[ne.name], ne.name))
            for ne in key_named_exprs
        ]
        select_schema = {
            ne.name: reduction_schema[ne.name] for ne in key_named_exprs
        } | {ne.name: ne.value.dtype for ne in selection_exprs}
        agg_select_ir = Select(
            select_schema,
            [*select_key_exprs, *selection_exprs],
            False,  # noqa: FBT003
            reduction_ir,
        )
    else:
        agg_select_ir = None

    return piecewise_ir, reduction_ir, agg_select_ir


class Over(IR):
    """
    Window over() IR node for the streaming runtime.

    Wraps a Select or HStack containing GroupedWindow expressions and carries
    pre-computed metadata for the over_actor: group key indices, scalar/non-scalar
    classification, and (for scalar) decomposed aggregation IRs.
    """

    __slots__ = (
        "agg_select_ir",
        "gw_nodes",
        "is_scalar",
        "key_indices",
        "key_names",
        "piecewise_ir",
        "reduction_ir",
        "row_idx_col",
    )
    _non_child: ClassVar[tuple[str, ...]] = (
        "schema",
        "key_indices",
        "is_scalar",
        "gw_nodes",
        "key_names",
        "piecewise_ir",
        "reduction_ir",
        "agg_select_ir",
        "row_idx_col",
    )
    _n_non_child_args: ClassVar[int] = 0

    key_indices: tuple[int, ...]
    is_scalar: bool
    gw_nodes: tuple[GroupedWindow, ...] | None
    key_names: tuple[str, ...] | None
    piecewise_ir: GroupBy | None
    reduction_ir: GroupBy | None
    agg_select_ir: Select | None
    row_idx_col: str | None

    def __init__(
        self,
        schema: Schema,
        key_indices: tuple[int, ...],
        is_scalar: bool,  # noqa: FBT001
        gw_nodes: tuple[GroupedWindow, ...] | None,
        key_names: tuple[str, ...] | None,
        piecewise_ir: GroupBy | None,
        reduction_ir: GroupBy | None,
        agg_select_ir: Select | None,
        row_idx_col: str | None,
        ir: Select | HStack,
    ):
        self.schema = schema
        self.key_indices = key_indices
        self.is_scalar = is_scalar
        self.gw_nodes = gw_nodes
        self.key_names = key_names
        self.piecewise_ir = piecewise_ir
        self.reduction_ir = reduction_ir
        self.agg_select_ir = agg_select_ir
        self.row_idx_col = row_idx_col
        self._non_child_args = ()
        self.children = (ir,)


def make_over_node(
    ir: Select | HStack,
    child: IR,
    key_indices: tuple[int, ...],
) -> Over:
    """
    Construct an Over node wrapping a Select or HStack with GroupedWindow expressions.

    Parameters
    ----------
    ir
        The original Select or HStack IR node (before child replacement).
    child
        The lowered child IR to attach.
    key_indices
        Column indices of the group-by keys in the child schema.

    Returns
    -------
    Over
        A new Over node with pre-computed aggregation IRs (scalar path) or
        a unique row-index column name (non-scalar path).
    """
    exprs = [e.value for e in (ir.exprs if isinstance(ir, Select) else ir.columns)]
    is_scalar = _all_over_scalar_and_top_level(exprs)
    gw_nodes: tuple[GroupedWindow, ...] | None
    key_names: tuple[str, ...] | None
    piecewise_ir: GroupBy | None
    reduction_ir: GroupBy | None
    agg_select_ir: Select | None
    row_idx_col: str | None
    if is_scalar:
        gw_nodes = tuple(e for e in exprs if isinstance(e, GroupedWindow))
        key_names = tuple(
            e.name
            for e in gw_nodes[0].children[: gw_nodes[0].by_count]
            if isinstance(e, Col)
        )
        piecewise_ir, reduction_ir, agg_select_ir = _build_over_groupby_irs(
            gw_nodes, child
        )
        row_idx_col = None
    else:
        gw_nodes = None
        key_names = None
        piecewise_ir = None
        reduction_ir = None
        agg_select_ir = None
        row_idx_col = next(unique_names(child.schema.keys()))
    wrapped_ir = ir.reconstruct([child])
    return Over(
        wrapped_ir.schema,
        key_indices,
        is_scalar,
        gw_nodes,
        key_names,
        piecewise_ir,
        reduction_ir,
        agg_select_ir,
        row_idx_col,
        wrapped_ir,
    )

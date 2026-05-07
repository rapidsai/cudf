# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Over IR node for streaming window expressions."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, cast

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Agg, Col, Len, NamedExpr
from cudf_polars.dsl.ir import IR, GroupBy, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.experimental.groupby import combine, decompose
from cudf_polars.experimental.rapidsmpf.utils import names_to_indices

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

    from cudf_polars.dsl.expr import GroupedWindow
    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


# Aggregations whose partial results can be combined across partitions
# without a pre-shuffle. n_unique is omitted because its decomposition
# sets need_preshuffle=True (see groupby.decompose).
_DECOMPOSABLE_AGG_NAMES: frozenset[str] = frozenset(
    ("sum", "count", "mean", "min", "max", "std", "var")
)


def _build_over_groupby_irs(
    gw_nodes: tuple[GroupedWindow, ...],
    child_ir: IR,
) -> tuple[GroupBy, GroupBy, Select | None, dict[GroupedWindow, dict[str, str]]]:
    """
    Build piecewise, reduction, and (optionally) selection GroupBy IRs.

    Parameters
    ----------
    gw_nodes
        Top-level GroupedWindow nodes sharing the same partition-by keys;
        all must be scalar (Agg/Len only in named_aggs).
    child_ir
        Input IR feeding the Over node; defines the schema seen by the
        per-chunk piecewise GroupBy.

    Returns
    -------
    piecewise_ir
        GroupBy IR that computes partial aggregates per chunk.
    reduction_ir
        GroupBy IR that reduces partial aggregates to a single result.
    agg_select_ir
        Select IR for post-aggregation expressions (e.g. division for mean);
        None when all aggregations are pass-through.
    name_remaps
        Per-gw mapping from original named-agg name to its globally-unique
        name in the aggregate IRs. Polars assigns each ``GroupedWindow`` its
        own local internal name (often the same string across gws), so we
        rename here to avoid collisions in the shared piecewise schema.
    """
    gw = gw_nodes[0]
    by_exprs = cast("list[Col]", list(gw.children[: gw.by_count]))
    key_named_exprs = [NamedExpr(e.name, e) for e in by_exprs]
    key_schema = {e.name: child_ir.schema[e.name] for e in by_exprs}

    name_gen = unique_names(child_ir.schema.keys())
    all_scalar_named: list[NamedExpr] = []
    name_remaps: dict[GroupedWindow, dict[str, str]] = {}
    for gw_node in gw_nodes:
        reductions, _ = gw_node._split_named_expr()
        remap: dict[str, str] = {}
        for ne in reductions:
            if ne.name in remap:
                continue
            renamed = next(name_gen)
            all_scalar_named.append(NamedExpr(renamed, ne.value))
            remap[ne.name] = renamed
        name_remaps[gw_node] = remap

    decompositions = [
        decompose(ne.name, ne.value, names=name_gen) for ne in all_scalar_named
    ]
    selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = combine(
        *decompositions
    )
    assert not need_preshuffle, (
        "Scalar AllGather path does not support aggregations requiring pre-shuffle"
    )

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

    return piecewise_ir, reduction_ir, agg_select_ir, name_remaps


class Over(IR):
    """Window over() IR node for the streaming runtime."""

    __slots__ = ("exprs", "is_scalar", "key_indices")
    _non_child: ClassVar[tuple[str, ...]] = (
        "schema",
        "key_indices",
        "is_scalar",
        "exprs",
    )
    _n_non_child_args: ClassVar[int] = 3

    key_indices: tuple[int, ...]
    is_scalar: bool
    exprs: tuple[NamedExpr, ...]

    def __init__(
        self,
        schema: Schema,
        key_indices: tuple[int, ...],
        is_scalar: bool,  # noqa: FBT001
        exprs: tuple[NamedExpr, ...],
        input_ir: IR,
    ):
        assert len(key_indices) > 0, "Over node requires at least one partition-by key"
        self.schema = schema
        self.key_indices = key_indices
        self.is_scalar = is_scalar
        self.exprs = exprs
        self._non_child_args = (key_indices, is_scalar, exprs)
        self.children = (input_ir,)

    @classmethod
    def do_evaluate(
        cls,
        key_indices: tuple[int, ...],
        is_scalar: bool,  # noqa: FBT001
        exprs: tuple[NamedExpr, ...],
        df: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate window expressions against df."""
        columns = [ne.evaluate(df) for ne in exprs]
        columns = broadcast(*columns, stream=df.stream)
        return DataFrame(columns, stream=df.stream)


def _is_scalar_grouped_window(expr: GroupedWindow) -> bool:
    """Return True if this GroupedWindow can use the scalar broadcast path."""
    reductions, unary_ops = expr._split_named_expr()
    if any(ops for ops in unary_ops.values()):
        return False
    if not all(isinstance(c, Col) for c in expr.children[: expr.by_count]):
        return False
    return all(
        isinstance(ne.value, Len)
        or (isinstance(ne.value, Agg) and ne.value.name in _DECOMPOSABLE_AGG_NAMES)
        for ne in reductions
    )


def _extract_over_shuffle_indices(
    expr: GroupedWindow, child_schema: Schema
) -> tuple[int, ...] | None:
    """
    Return partition-by column indices in ``child_schema``, or None.

    Returns None when any partition-by expression is not a plain column
    reference (the multi-partition path only supports Col keys today).
    """
    by_children = expr.children[: expr.by_count]
    if not all(isinstance(c, Col) for c in by_children):
        return None
    return names_to_indices(
        tuple(cast("Col", c).name for c in by_children), child_schema
    )


def _decompose_grouped_window_node(
    expr: GroupedWindow,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Build an Over IR node wrapping a single GroupedWindow expression.

    Every GroupedWindow becomes its own Over here; co-keyed Overs are
    fused together later by select fusion so the actor evaluates all
    window expressions in one pass.

    Returns
    -------
    Expr
        A ``Col`` referencing the Over node's output column, suitable
        for substitution into the enclosing expression.
    IR
        The new ``Over`` IR node.
    MutableMapping[IR, PartitionInfo]
        ``partition_info`` augmented with an entry for the new node.
    """
    indices = _extract_over_shuffle_indices(expr, input_ir.schema)
    if indices is None:
        raise NotImplementedError(
            "GroupedWindow with non-Col partition-by keys "
            "is not supported for multiple partitions."
        )
    if len(indices) == 0:
        raise NotImplementedError(
            "GroupedWindow with empty partition-by keys "
            "is not supported for multiple partitions."
        )
    is_scalar = _is_scalar_grouped_window(expr)
    col_name = next(names)
    over_node = Over(
        {col_name: expr.dtype},
        indices,
        is_scalar,
        (NamedExpr(col_name, expr),),
        input_ir,
    )
    partition_info[over_node] = partition_info[input_ir]
    return Col(expr.dtype, col_name), over_node, partition_info


def _fuse_over_nodes(
    selections: list[Select],
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[list[Select], MutableMapping[IR, PartitionInfo]]:
    """
    Fuse per-expression Over nodes that share the same grouping key.

    Each GroupedWindow starts as its own Over node. This pass merges
    co-keyed Overs across the input Selects, and absorbs passthrough
    selections that share an input IR into the merged Over so the
    actor produces the full output schema in one shuffle pass.

    The grouping key is ``(key_indices, is_scalar, input_ir)``.
    """
    over_groups: defaultdict[tuple[tuple[int, ...], bool, IR], list[Select]] = (
        defaultdict(list)
    )
    passthrough: list[Select] = []

    for sel in selections:
        child = sel.children[0]
        if isinstance(child, Over):
            input_ir = child.children[0]
            over_groups[(child.key_indices, child.is_scalar, input_ir)].append(sel)
        else:
            passthrough.append(sel)

    if not over_groups:
        return selections, partition_info

    result: list[Select] = []
    for (key_indices, is_scalar, input_ir), group in over_groups.items():
        first_over = cast("Over", group[0].children[0])
        pi = partition_info[first_over]

        combined_window_exprs: list[NamedExpr] = []
        for sel in group:
            over = cast("Over", sel.children[0])
            combined_window_exprs.extend(over.exprs)

        # Absorb passthrough selections that share the same input_ir into
        # the Over node so it outputs all columns in one pass.
        absorbed: list[Select] = []
        remaining: list[Select] = []
        for p_sel in passthrough:
            if p_sel.children[0] == input_ir:
                absorbed.append(p_sel)
            else:
                remaining.append(p_sel)
        passthrough = remaining

        passthrough_exprs: list[NamedExpr] = []
        for p_sel in absorbed:
            passthrough_exprs.extend(p_sel.exprs)

        all_over_exprs = passthrough_exprs + combined_window_exprs
        combined_schema = {ne.name: ne.value.dtype for ne in all_over_exprs}
        merged_over = Over(
            combined_schema, key_indices, is_scalar, tuple(all_over_exprs), input_ir
        )
        partition_info[merged_over] = pi

        # Build outer_exprs in original selection order to preserve column ordering.
        all_this_group = set(itertools.chain(absorbed, group))
        outer_exprs: list[NamedExpr] = []
        outer_schema: Schema = {}
        for sel in selections:
            if sel in all_this_group:
                outer_exprs.extend(sel.exprs)
                outer_schema |= sel.schema

        merged_sel = Select(outer_schema, outer_exprs, True, merged_over)  # noqa: FBT003
        partition_info[merged_sel] = pi
        result.append(merged_sel)

    result.extend(passthrough)
    return result, partition_info

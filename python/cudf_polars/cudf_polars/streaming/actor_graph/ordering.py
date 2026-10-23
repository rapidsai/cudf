# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-ordering metadata utilities for the streaming actor graph."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    HashScheme,
    OrderKey,
    RowOrdering,
    RowOrderingScope,
)

from cudf_polars.dsl.expr import Cast, Col, TemporalFunction
from cudf_polars.dsl.ir import HStack, Select
from cudf_polars.dsl.utils.column_domain import column_domain_bindings
from cudf_polars.dsl.utils.naming import indices_to_names, names_to_indices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.expr import Expr, NamedExpr
    from cudf_polars.dsl.ir import IR


def _is_truncate_transparent_cast(expr: Cast) -> bool:
    src_id = expr.children[0].dtype.id()
    dst_id = expr.dtype.id()
    if src_id == dst_id:
        return True
    return (
        src_id == plc.TypeId.INT64 and dst_id == plc.TypeId.TIMESTAMP_NANOSECONDS
    ) or (src_id == plc.TypeId.TIMESTAMP_NANOSECONDS and dst_id == plc.TypeId.INT64)


def _unwrap_truncate_transparent_casts(expr: Expr) -> Expr:
    while isinstance(expr, Cast) and _is_truncate_transparent_cast(expr):
        (expr,) = expr.children
    return expr


def ordering_derivation(ne: NamedExpr) -> tuple[str, bool] | None:
    """Return the source column and boundary strictness for a derived ordering."""
    expr = _unwrap_truncate_transparent_casts(ne.value)
    if not (
        isinstance(expr, TemporalFunction)
        and expr.name is TemporalFunction.Name.Truncate
    ):
        return None
    source = _unwrap_truncate_transparent_casts(expr.children[0])
    if not isinstance(source, Col):
        return None
    return source.name, False


def _ordering_implies(actual: RowOrdering, required: RowOrdering) -> bool:
    if actual.scope == RowOrderingScope.RANK_LOCAL and (
        required.scope == RowOrderingScope.GLOBAL
    ):
        return False

    required_groups = set(required.group_keys)
    if not set(actual.group_keys).issubset(required_groups):
        return False

    order_keys = iter(actual.order_keys)
    for key in order_keys:
        if key.column_index not in required_groups:
            break
    else:
        return False

    remaining = (key, *order_keys)
    return remaining[: len(required.order_keys)] == required.order_keys


def satisfies_row_ordering(
    metadata: ChannelMetadata,
    required: RowOrdering,
    *,
    nranks: int,
) -> bool:
    """Return whether metadata proves the requested row ordering."""
    for actual in metadata.row_orderings:
        if _ordering_implies(actual, required):
            return True
        if required.scope != RowOrderingScope.GLOBAL:
            continue
        if actual.scope != RowOrderingScope.RANK_LOCAL:
            continue
        if nranks == 1 or metadata.duplicated:
            local_required = RowOrdering(
                required.order_keys,
                RowOrderingScope.RANK_LOCAL,
                required.group_keys,
            )
            if _ordering_implies(actual, local_required):
                return True
            continue
        partitioning = metadata.partitioning
        if partitioning is None or not isinstance(partitioning.inter_rank, HashScheme):
            continue
        if not set(partitioning.inter_rank.column_indices).issubset(
            set(required.group_keys)
        ):
            continue
        local_required = RowOrdering(
            required.order_keys,
            RowOrderingScope.RANK_LOCAL,
            required.group_keys,
        )
        if _ordering_implies(actual, local_required):
            return True
    return False


def matching_row_ordering(
    metadata: ChannelMetadata,
    order_keys: Sequence[OrderKey],
    *,
    nranks: int,
    group_keys: Sequence[int] = (),
) -> RowOrdering | None:
    """Return the strongest row-ordering guarantee for the requested keys."""
    for scope in (RowOrderingScope.GLOBAL, RowOrderingScope.RANK_LOCAL):
        required = RowOrdering(order_keys, scope, group_keys)
        if satisfies_row_ordering(metadata, required, nranks=nranks):
            return required
    return None


def global_row_orderings(
    metadata: ChannelMetadata, *, nranks: int
) -> tuple[RowOrdering, ...]:
    """Return row-ordering guarantees that hold across ranks."""
    result: list[RowOrdering] = []
    for ordering in metadata.row_orderings:
        global_ordering = RowOrdering(
            ordering.order_keys,
            RowOrderingScope.GLOBAL,
            ordering.group_keys,
        )
        if global_ordering not in result and satisfies_row_ordering(
            metadata, global_ordering, nranks=nranks
        ):
            result.append(global_ordering)
    return tuple(result)


def remap_row_orderings(
    ir: IR,
    row_orderings: Sequence[RowOrdering],
    *,
    child_index: int = 0,
    preserves_order: bool | None = None,
) -> tuple[RowOrdering, ...]:
    """Remap row-ordering keys through an order-preserving IR node."""
    if not (ir.preserves_output_order if preserves_order is None else preserves_order):
        return ()

    child = ir.children[child_index]
    old_to_new: defaultdict[str, list[str]] = defaultdict(list)
    for output_name, source in column_domain_bindings(ir).items():
        if source.child_index == child_index:
            old_to_new[source.name].append(output_name)

    def remap_indices(indices: Sequence[int]) -> tuple[int, ...] | None:
        names = indices_to_names(indices, child.schema)
        if not all(name in old_to_new for name in names):
            return None
        targets = tuple(
            name if name in old_to_new[name] else old_to_new[name][0] for name in names
        )
        return names_to_indices(targets, ir.schema)

    expressions = ir.exprs if isinstance(ir, Select) else ()
    if isinstance(ir, HStack):
        expressions = ir.columns

    result: list[RowOrdering] = []
    for ordering in row_orderings:
        group_keys = remap_indices(ordering.group_keys)
        if group_keys is None:
            continue
        old_order_names = indices_to_names(
            tuple(key.column_index for key in ordering.order_keys), child.schema
        )
        preserved_names: list[str] = []
        for name in old_order_names:
            if name not in old_to_new:
                break
            preserved_names.append(
                name if name in old_to_new[name] else old_to_new[name][0]
            )
        if preserved_names:
            order_indices = names_to_indices(tuple(preserved_names), ir.schema)
            order_keys = tuple(
                OrderKey(index, key.order, key.null_order)
                for index, key in zip(
                    order_indices,
                    ordering.order_keys[: len(order_indices)],
                    strict=True,
                )
            )
            result.append(RowOrdering(order_keys, ordering.scope, group_keys))

        for ne in expressions:
            derivation = ordering_derivation(ne)
            if derivation is None:
                continue
            source_name, _ = derivation
            try:
                source_position = old_order_names.index(source_name)
            except ValueError:
                continue
            prefix_names = old_order_names[:source_position]
            if not all(name in old_to_new for name in prefix_names):
                continue
            target_names = (
                *(
                    name if name in old_to_new[name] else old_to_new[name][0]
                    for name in prefix_names
                ),
                ne.name,
            )
            target_indices = names_to_indices(target_names, ir.schema)
            derived_keys = tuple(
                OrderKey(index, key.order, key.null_order)
                for index, key in zip(
                    target_indices,
                    ordering.order_keys[: source_position + 1],
                    strict=True,
                )
            )
            result.append(RowOrdering(derived_keys, ordering.scope, group_keys))
    return tuple(result)


def leading_order_keys(
    row_orderings: Sequence[RowOrdering],
) -> dict[int, OrderKey]:
    """Return unambiguous leading keys from ungrouped row orderings."""
    candidates: dict[int, OrderKey | None] = {}
    for ordering in row_orderings:
        if ordering.group_keys:
            continue
        key = ordering.order_keys[0]
        current = candidates.get(key.column_index, key)
        candidates[key.column_index] = key if current == key else None
    return {index: key for index, key in candidates.items() if key is not None}

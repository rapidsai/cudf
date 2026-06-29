# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic derived key-domain prefilters for streaming joins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    Cache,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Scan,
    Select,
)
from cudf_polars.dsl.tracing import Scope, log
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@dataclass(frozen=True)
class _Producer:
    """A subtree that can provide a key domain."""

    node: IR
    column: str
    rows: int


@dataclass(frozen=True)
class _Candidate:
    """A derived key-domain prefilter candidate."""

    mode: Literal["simple", "composite"]
    target_side: Literal["left", "right"]
    target: IR
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col
    constraint_domain: _Producer | None = None
    domain_constraint_key: expr.Col | None = None
    target_constraint_key: expr.Col | None = None

    @property
    def domain_rows(self) -> int:
        """Estimated rows in the domain input."""
        return self.domain.rows

    @property
    def target_rows(self) -> int:
        """Estimated rows in the target input."""
        return _estimate_rows(self.target) or 0

    @property
    def score(self) -> tuple[int, int, int]:
        """Prefer composite filters, then smaller constraint/domain inputs."""
        constraint_rows = (
            self.constraint_domain.rows
            if self.constraint_domain is not None
            else self.domain.rows
        )
        return (
            0 if self.mode == "composite" else 1,
            constraint_rows,
            self.domain.rows,
        )


_ROW_ESTIMATES: dict[IR, int | None] = {}
_SELECTIVE: dict[IR, bool] = {}
_STATS: StatsCollector | None = None


def optimize_join_domain_prefilters(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions[StreamingExecutor],
) -> IR:
    """
    Insert generic semi-join key-domain prefilters before streaming lowering.

    The rewrite is intentionally conservative: only inner joins with simple
    column equality keys are considered, and the original full join remains
    after every inserted row-reduction semi join.
    """
    dynamic_options = config_options.executor.dynamic_planning
    if dynamic_options is None or not dynamic_options.join_domain_prefilter_enabled:
        return ir
    threshold = dynamic_options.join_domain_prefilter_threshold
    trace = dynamic_options.join_domain_prefilter_trace
    if threshold is None or threshold == 0 or trace is None:
        return ir

    global _ROW_ESTIMATES, _SELECTIVE, _STATS
    old_estimates, old_selective, old_stats = _ROW_ESTIMATES, _SELECTIVE, _STATS
    _ROW_ESTIMATES, _SELECTIVE, _STATS = {}, {}, stats
    try:
        return _rewrite_node(
            ir,
            threshold=threshold,
            trace=trace,
        )
    finally:
        _ROW_ESTIMATES, _SELECTIVE, _STATS = (
            old_estimates,
            old_selective,
            old_stats,
        )


def _rewrite_node(ir: IR, *, threshold: float, trace: bool) -> IR:
    children = tuple(
        _rewrite_node(child, threshold=threshold, trace=trace) for child in ir.children
    )
    node = ir if children == ir.children else ir.reconstruct(children)

    if not isinstance(node, Join):
        return node

    candidate, reason = _select_candidate(node, threshold)
    if trace:
        _trace_decision(node, threshold, candidate, reason)
    if candidate is None:
        return node

    left, right = node.children
    target_filter = _make_target_filter(node, candidate)
    if candidate.target_side == "left":
        left = _replace_identity(left, candidate.target, target_filter)
    else:
        right = _replace_identity(right, candidate.target, target_filter)
    return node.reconstruct((left, right))


def _select_candidate(ir: Join, threshold: float) -> tuple[_Candidate | None, str]:
    if ir.options[0] != "Inner":
        return None, "not_inner_join"
    if ir.options[2] is not None:
        return None, "sliced_join"
    if ir.options[5] != "none":
        return None, "maintain_order"

    left_keys = _simple_keys(ir.left_on)
    right_keys = _simple_keys(ir.right_on)
    if len(left_keys) != len(ir.left_on) or len(right_keys) != len(ir.right_on):
        return None, "non_column_join_key"
    if len(left_keys) != len(right_keys):
        return None, "key_count_mismatch"

    candidates: list[_Candidate] = []
    for target_side in ("left", "right"):
        target_child, domain_child = (
            (ir.children[0], ir.children[1])
            if target_side == "left"
            else (ir.children[1], ir.children[0])
        )
        target_keys, domain_keys = (
            (left_keys, right_keys)
            if target_side == "left"
            else (right_keys, left_keys)
        )
        candidates.extend(
            _composite_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                threshold,
            )
        )
        candidates.extend(
            _simple_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                threshold,
            )
        )

    if not candidates:
        return None, "no_selective_domain"
    return min(candidates, key=lambda c: c.score), "applied"


def _simple_keys(keys: Sequence[expr.NamedExpr]) -> tuple[expr.Col, ...]:
    return tuple(key.value for key in keys if isinstance(key.value, expr.Col))


def _simple_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
) -> Iterable[_Candidate]:
    for target_key, domain_key in zip(target_keys, domain_keys, strict=True):
        target = _largest_key_source(target_child, target_key.name)
        if target is None:
            continue
        target_rows = _estimate_rows(target)
        if target_rows is None or target_rows <= 0:
            continue
        domain = _smallest_key_producer(
            domain_child, domain_key.name, require_selective=True
        )
        if domain is None:
            continue
        if _contains_identity(target, domain.node):
            continue
        if domain.rows / target_rows > threshold:
            continue
        yield _Candidate(
            mode="simple",
            target_side=target_side,
            target=target,
            target_key=target_key,
            domain=domain,
            domain_key=domain_key,
        )


def _composite_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
) -> Iterable[_Candidate]:
    if len(target_keys) < 2:
        return

    for filter_index, (target_key, domain_key) in enumerate(
        zip(target_keys, domain_keys, strict=True)
    ):
        target = _largest_key_source(target_child, target_key.name)
        if target is None:
            continue
        target_rows = _estimate_rows(target)
        if target_rows is None or target_rows <= 0:
            continue

        for constraint_index, (
            target_constraint_key,
            domain_constraint_key,
        ) in enumerate(zip(target_keys, domain_keys, strict=True)):
            if constraint_index == filter_index:
                continue
            domain = _smallest_node_containing_all(
                domain_child, (domain_key.name, domain_constraint_key.name)
            )
            if domain is None:
                continue
            constraint_domain = _smallest_key_producer(
                target_child,
                target_constraint_key.name,
                require_selective=True,
                exclude=target,
            )
            if constraint_domain is None:
                continue
            if _contains_identity(target, domain.node) or _contains_identity(
                target, constraint_domain.node
            ):
                continue
            if domain.rows / target_rows > threshold:
                continue
            if constraint_domain.rows / domain.rows > threshold:
                continue
            yield _Candidate(
                mode="composite",
                target_side=target_side,
                target=target,
                target_key=target_key,
                domain=domain,
                domain_key=domain_key,
                constraint_domain=constraint_domain,
                domain_constraint_key=domain_constraint_key,
                target_constraint_key=target_constraint_key,
            )


def _make_target_filter(ir: Join, candidate: _Candidate) -> Join:
    domain = _make_domain(candidate, ir)
    return _make_semi_join(
        candidate.target,
        candidate.target_key,
        domain,
        expr.Col(domain.schema[candidate.domain_key.name], candidate.domain_key.name),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )


def _make_domain(candidate: _Candidate, ir: Join) -> IR:
    if candidate.mode == "simple":
        return _select_key(
            candidate.domain.node,
            candidate.domain.column,
            candidate.domain_key.name,
        )

    assert candidate.constraint_domain is not None
    assert candidate.domain_constraint_key is not None
    assert candidate.target_constraint_key is not None

    constraint_domain = _select_key(
        candidate.constraint_domain.node,
        candidate.constraint_domain.column,
        candidate.target_constraint_key.name,
    )
    constrained = _make_semi_join(
        candidate.domain.node,
        expr.Col(
            candidate.domain.node.schema[candidate.domain_constraint_key.name],
            candidate.domain_constraint_key.name,
        ),
        constraint_domain,
        expr.Col(
            constraint_domain.schema[candidate.target_constraint_key.name],
            candidate.target_constraint_key.name,
        ),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )
    return _select_key(constrained, candidate.domain.column, candidate.domain_key.name)


def _select_key(source: IR, source_column: str, output_column: str) -> Select:
    dtype = source.schema[source_column]
    return Select(
        {output_column: dtype},
        (expr.NamedExpr(output_column, expr.Col(dtype, source_column)),),
        True,  # noqa: FBT003
        source,
    )


def _make_semi_join(
    target: IR,
    target_key: expr.Col,
    domain: IR,
    domain_key: expr.Col,
    *,
    nulls_equal: bool,
    suffix: str,
) -> Join:
    return Join(
        target.schema,
        (expr.NamedExpr(target_key.name, target_key),),
        (expr.NamedExpr(domain_key.name, domain_key),),
        ("Semi", nulls_equal, None, suffix, False, "none"),
        target,
        domain,
    )


def _smallest_key_producer(
    root: IR, column: str, *, require_selective: bool, exclude: IR | None = None
) -> _Producer | None:
    candidates = []
    for node in traversal([root]):
        if node is exclude or column not in node.schema:
            continue
        rows = _estimate_rows(node)
        if rows is None or rows <= 0:
            continue
        if require_selective and not _is_selective(node):
            continue
        candidates.append((rows, len(node.schema), _Producer(node, column, rows)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _smallest_node_containing_all(root: IR, columns: Sequence[str]) -> _Producer | None:
    candidates = []
    needed = set(columns)
    for node in traversal([root]):
        if not needed.issubset(node.schema):
            continue
        rows = _estimate_rows(node)
        if rows is None or rows <= 0:
            continue
        candidates.append((rows, len(node.schema), _Producer(node, columns[0], rows)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _largest_key_source(root: IR, column: str) -> IR | None:
    source_candidates = []
    fallback_candidates = []
    for node in traversal([root]):
        if column not in node.schema:
            continue
        rows = _estimate_rows(node)
        if rows is None or rows <= 0:
            continue
        item = (rows, len(node.schema), node)
        if isinstance(node, (Scan, DataFrameScan)):
            source_candidates.append(item)
        else:
            fallback_candidates.append(item)
    candidates = source_candidates or fallback_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]


def _estimate_rows(ir: IR) -> int | None:
    try:
        return _ROW_ESTIMATES[ir]
    except KeyError:
        pass

    rows: int | None
    if isinstance(ir, (Scan, DataFrameScan)):
        source = None if _STATS is None else _STATS.scan_stats.get(ir)
        rows = None if source is None else source.row_count
        if rows is None and isinstance(ir, DataFrameScan):
            rows = ir.df.shape()[0]
    elif isinstance(ir, (Select, Projection, HStack, Cache, Filter, Distinct, GroupBy)):
        rows = _estimate_rows(ir.children[0])
    elif isinstance(ir, Join):
        left_rows = _estimate_rows(ir.children[0])
        right_rows = _estimate_rows(ir.children[1])
        rows = _estimate_join_rows(ir.options[0], left_rows, right_rows)
    else:
        estimates = [
            estimate for child in ir.children if (estimate := _estimate_rows(child))
        ]
        rows = max(estimates) if estimates else None

    _ROW_ESTIMATES[ir] = rows
    return rows


def _estimate_join_rows(
    how: str, left_rows: int | None, right_rows: int | None
) -> int | None:
    if left_rows is None:
        return right_rows
    if right_rows is None:
        return left_rows
    if how in ("Inner", "Semi", "Anti"):
        return min(left_rows, right_rows)
    if how == "Left":
        return left_rows
    if how == "Right":
        return right_rows
    if how == "Full":
        return max(left_rows, right_rows)
    return None


def _is_selective(ir: IR) -> bool:
    try:
        return _SELECTIVE[ir]
    except KeyError:
        pass

    if isinstance(ir, Scan):
        selective = ir.predicate is not None
    elif isinstance(ir, Filter):
        selective = True
    else:
        selective = any(_is_selective(child) for child in ir.children)

    _SELECTIVE[ir] = selective
    return selective


def _contains_identity(root: IR, needle: IR) -> bool:
    return any(node is needle for node in traversal([root]))


def _replace_identity(root: IR, old: IR, new: IR) -> IR:
    if root is old:
        return new
    if not root.children:
        return root
    children = tuple(_replace_identity(child, old, new) for child in root.children)
    if children == root.children:
        return root
    return root.reconstruct(children)


def _trace_decision(
    ir: Join, threshold: float, candidate: _Candidate | None, reason: str
) -> None:
    join_domain_prefilter: dict[str, Any] = {
        "considered": True,
        "threshold": threshold,
        "reason": reason,
    }
    record = {
        "scope": Scope.PLAN.value,
        "join_domain_prefilter": join_domain_prefilter,
        "actor_ir_id": ir.get_stable_id(),
        "actor_ir_type": type(ir).__name__,
    }
    if candidate is not None:
        join_domain_prefilter.update(
            {
                "mode": candidate.mode,
                "target_side": candidate.target_side,
                "target_key": candidate.target_key.name,
                "domain_key": candidate.domain_key.name,
                "estimated_target_rows": candidate.target_rows,
                "estimated_domain_rows": candidate.domain_rows,
                "target_node_type": type(candidate.target).__name__,
                "domain_node_type": type(candidate.domain.node).__name__,
            }
        )
        if candidate.constraint_domain is not None:
            join_domain_prefilter.update(
                {
                    "constraint_key": candidate.target_constraint_key.name
                    if candidate.target_constraint_key is not None
                    else None,
                    "estimated_constraint_rows": candidate.constraint_domain.rows,
                }
            )
    log("Join Domain Prefilter", **record)

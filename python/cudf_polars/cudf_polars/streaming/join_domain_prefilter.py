# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic derived key-domain prefilters for streaming joins."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
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
    Sort,
)
from cudf_polars.dsl.tracing import Scope, log
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    post_traversal,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.dsl.utils.replace import replace

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.typing import GenericTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@dataclass(frozen=True)
class _Producer:
    """A subtree and its bound column names at an insertion point."""

    node: IR
    columns: tuple[str, ...]
    rows: int
    cost: int

    @property
    def column(self) -> str:
        """First bound column in the producer."""
        return self.columns[0]


@dataclass(frozen=True)
class _Candidate:
    """A derived key-domain prefilter candidate."""

    mode: Literal["simple", "composite", "reused"]
    target_side: Literal["left", "right"]
    target: _Producer
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col
    target_rows: int
    constraint_domain: _Producer | None = None
    domain_constraint_key: expr.Col | None = None
    target_constraint_key: expr.Col | None = None

    @property
    def domain_rows(self) -> int:
        """Estimated rows in the domain input."""
        return self.domain.rows

    @property
    def domain_cost(self) -> int:
        """Estimated scan-row cost to build the domain input."""
        return self.domain.cost

    @property
    def score(self) -> tuple[int, int, int]:
        """Prefer composite filters, then cheaper constraint/domain inputs."""
        mode_score = {"composite": 0, "simple": 1, "reused": 2}[self.mode]
        constraint_cost = (
            self.constraint_domain.cost
            if self.constraint_domain is not None
            else self.domain.cost
        )
        return (
            mode_score,
            constraint_cost,
            self.domain.cost,
        )


class _RewriteState(TypedDict):
    """State shared by the join-domain prefilter DAG rewrite."""

    threshold: float
    trace: bool
    stats: StatsCollector
    row_estimates: dict[IR, int | None]
    source_costs: dict[IR, int | None]
    source_counts: dict[IR, int]
    selective_nodes: set[IR]


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
    options = config_options.executor.join_domain_prefilter
    if options is None:
        return ir
    threshold = options.threshold
    trace = options.trace
    if threshold == 0:
        return ir

    row_estimates = _estimate_row_counts(ir, stats)
    source_costs, source_counts = _estimate_source_stats(ir, row_estimates)
    state = _RewriteState(
        threshold=threshold,
        trace=trace,
        stats=stats,
        row_estimates=row_estimates,
        source_costs=source_costs,
        source_counts=source_counts,
        selective_nodes=_collect_selective_nodes(ir),
    )
    mapper: GenericTransformer[IR, IR, _RewriteState] = CachingVisitor(
        _rewrite, state=state
    )
    return mapper(ir)


@singledispatch
def _rewrite(node: IR, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    raise AssertionError


@_rewrite.register(IR)
def _(node: IR, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    return reuse_if_unchanged(node, rec)


@_rewrite.register(Join)
def _(node: Join, rec: GenericTransformer[IR, IR, _RewriteState]) -> IR:
    original = node
    rewritten = reuse_if_unchanged(node, rec)
    assert isinstance(rewritten, Join)
    node = rewritten
    if node is original:
        row_estimates = rec.state["row_estimates"]
        source_costs = rec.state["source_costs"]
        source_counts = rec.state["source_counts"]
        selective_nodes = rec.state["selective_nodes"]
    else:
        # Child rewrites introduce new semi joins and reconstructed ancestors.
        # Re-analyze that current subtree so parent joins can use the derived
        # selectivity and cardinality when ranking their own candidates.
        row_estimates = _estimate_row_counts(node, rec.state["stats"])
        source_costs, source_counts = _estimate_source_stats(node, row_estimates)
        selective_nodes = _collect_selective_nodes(node)
    candidate, reason = _select_candidate(
        node,
        rec.state["threshold"],
        row_estimates,
        source_costs,
        source_counts,
        selective_nodes,
    )
    if rec.state["trace"]:
        _trace_decision(node, rec.state["threshold"], candidate, reason, source_costs)
    if candidate is None:
        return node

    left, right = node.children
    domain = _make_domain(candidate, node)
    target = candidate.target
    target_filter = _make_semi_join(
        target.node,
        expr.Col(target.node.schema[target.column], target.column),
        domain,
        expr.Col(domain.schema[candidate.domain_key.name], candidate.domain_key.name),
        nulls_equal=node.options[1],
        suffix=node.options[3],
    )
    # A DAG may share the target with the domain side, so only rewrite the
    # side for which this candidate was selected.
    if candidate.target_side == "left":
        (left,) = replace([left], {candidate.target.node: target_filter})
    else:
        (right,) = replace([right], {candidate.target.node: target_filter})
    return node.reconstruct((left, right))


def _select_candidate(
    ir: Join,
    threshold: float,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    source_counts: dict[IR, int],
    selective_nodes: set[IR],
) -> tuple[_Candidate | None, str]:
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

    candidates: list[_Candidate] = []
    left: tuple[Literal["left", "right"], IR, tuple[expr.Col, ...]] = (
        "left",
        ir.children[0],
        left_keys,
    )
    right: tuple[Literal["left", "right"], IR, tuple[expr.Col, ...]] = (
        "right",
        ir.children[1],
        right_keys,
    )
    for (target_side, target_child, target_keys), (
        _,
        domain_child,
        domain_keys,
    ) in ((left, right), (right, left)):
        candidates.extend(
            _composite_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                threshold,
                row_estimates,
                source_costs,
                selective_nodes,
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
                row_estimates,
                source_costs,
                source_counts,
                selective_nodes,
            )
        )
        candidates.extend(
            _reused_semi_domain_candidates(
                target_side,
                target_child,
                domain_child,
                target_keys,
                domain_keys,
                row_estimates,
                source_costs,
                source_counts,
                selective_nodes,
            )
        )

    if not candidates:
        return None, "no_profitable_domain"
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
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    source_counts: dict[IR, int],
    selective_nodes: set[IR],
) -> Iterable[_Candidate]:
    for target_key, domain_key in zip(target_keys, domain_keys, strict=True):
        target = _largest_key_source(
            target_child, target_key.name, row_estimates, source_costs
        )
        if target is None:
            continue
        domain = _smallest_key_producer(
            domain_child,
            domain_key.name,
            row_estimates,
            source_costs,
            selective_nodes,
            require_selective=True,
        )
        if domain is None:
            continue
        if domain.rows / target.rows > threshold:
            continue
        if _contains_identity(target.node, domain.node):
            continue
        if source_counts.get(domain.node) == 1 and _has_filtering_semi_ancestor(
            target_child, target.node
        ):
            continue
        if not _domain_cost_is_small(
            domain.node, target.node, threshold, row_estimates, source_costs
        ):
            continue
        yield _Candidate(
            mode="simple",
            target_side=target_side,
            target=target,
            target_key=target_key,
            domain=domain,
            domain_key=domain_key,
            target_rows=target.rows,
        )


def _reused_semi_domain_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    source_counts: dict[IR, int],
    selective_nodes: set[IR],
) -> Iterable[_Candidate]:
    for target_key, domain_key in zip(target_keys, domain_keys, strict=True):
        for domain, semi_domain_key in _filtering_semi_domains(
            domain_child,
            domain_key,
            row_estimates,
            source_costs,
            selective_nodes,
        ):
            target = _largest_key_reusable_target(
                target_child, target_key.name, domain.node, row_estimates, source_costs
            )
            if target is None:
                continue
            if domain.rows > target.rows:
                continue
            if _contains_identity(target.node, domain.node):
                continue
            if source_counts.get(domain.node) == 1 and _has_filtering_semi_ancestor(
                target_child, target.node
            ):
                continue
            yield _Candidate(
                mode="reused",
                target_side=target_side,
                target=target,
                target_key=target_key,
                domain=domain,
                domain_key=semi_domain_key,
                target_rows=target.rows,
            )


def _composite_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    selective_nodes: set[IR],
) -> Iterable[_Candidate]:
    if len(target_keys) < 2:
        return

    for filter_index, (target_key, domain_key) in enumerate(
        zip(target_keys, domain_keys, strict=True)
    ):
        target = _largest_key_source(
            target_child, target_key.name, row_estimates, source_costs
        )
        if target is None:
            continue

        for constraint_index, (
            target_constraint_key,
            domain_constraint_key,
        ) in enumerate(zip(target_keys, domain_keys, strict=True)):
            if constraint_index == filter_index:
                continue
            domain = _smallest_node_containing_all(
                domain_child,
                (domain_key.name, domain_constraint_key.name),
                row_estimates,
                source_costs,
            )
            if domain is None:
                continue
            if domain.rows / target.rows > threshold:
                continue
            constraint_domain = _smallest_key_producer(
                target_child,
                target_constraint_key.name,
                row_estimates,
                source_costs,
                selective_nodes,
                require_selective=True,
                exclude=target.node,
            )
            if constraint_domain is None:
                continue
            if constraint_domain.rows / domain.rows > threshold:
                continue
            if _contains_identity(target.node, domain.node) or _contains_identity(
                target.node, constraint_domain.node
            ):
                continue
            if not _domain_cost_is_small(
                domain.node, target.node, threshold, row_estimates, source_costs
            ):
                continue
            if not _domain_cost_is_small(
                constraint_domain.node,
                domain.node,
                threshold,
                row_estimates,
                source_costs,
            ):
                continue
            yield _Candidate(
                mode="composite",
                target_side=target_side,
                target=target,
                target_key=target_key,
                domain=domain,
                domain_key=domain_key,
                target_rows=target.rows,
                constraint_domain=constraint_domain,
                domain_constraint_key=domain_constraint_key,
                target_constraint_key=target_constraint_key,
            )


def _make_domain(candidate: _Candidate, ir: Join) -> IR:
    if candidate.mode in ("simple", "reused"):
        return _project_bound_key(
            candidate.domain.node,
            candidate.domain.column,
            candidate.domain_key,
        )

    assert candidate.constraint_domain is not None
    assert candidate.domain_constraint_key is not None
    assert candidate.target_constraint_key is not None

    constraint_domain = _project_bound_key(
        candidate.constraint_domain.node,
        candidate.constraint_domain.column,
        candidate.target_constraint_key,
    )
    constrained = _make_semi_join(
        candidate.domain.node,
        expr.Col(
            candidate.domain.node.schema[candidate.domain.columns[1]],
            candidate.domain.columns[1],
        ),
        constraint_domain,
        expr.Col(
            constraint_domain.schema[candidate.target_constraint_key.name],
            candidate.target_constraint_key.name,
        ),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )
    return _project_bound_key(
        constrained, candidate.domain.column, candidate.domain_key
    )


def _project_bound_key(source: IR, bound_column: str, output_key: expr.Col) -> Select:
    """Project a bound source column under its join-visible key name."""
    dtype = source.schema[bound_column]
    assert dtype == output_key.dtype
    return Select(
        {output_key.name: dtype},
        (expr.NamedExpr(output_key.name, expr.Col(dtype, bound_column)),),
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
    root: IR,
    column: str,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    selective_nodes: set[IR],
    *,
    require_selective: bool,
    exclude: IR | None = None,
) -> _Producer | None:
    candidates = []
    for node, bound_column in _column_bindings(root, column):
        if node is exclude:
            continue
        rows = row_estimates.get(node)
        if rows is None or rows <= 0:
            continue
        if require_selective and node not in selective_nodes:
            continue
        cost = source_costs.get(node)
        if cost is None:
            continue
        producer = _Producer(node, (bound_column,), rows, cost)
        candidates.append((cost, rows, len(node.schema), producer))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _filtering_semi_domains(
    root: IR,
    filtered_key: expr.Col,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
    selective_nodes: set[IR],
) -> Iterable[tuple[_Producer, expr.Col]]:
    """Yield domains from semi joins on the exact filtered-key lineage."""
    for node, bound_key in _column_bindings(root, filtered_key.name):
        if not isinstance(node, Join) or node.options[0] != "Semi":
            continue
        left_keys = _simple_keys(node.left_on)
        right_keys = _simple_keys(node.right_on)
        if len(left_keys) != len(node.left_on) or len(right_keys) != len(node.right_on):
            continue
        assert len(left_keys) == len(right_keys)
        for left_key, right_key in zip(left_keys, right_keys, strict=True):
            if left_key.name != bound_key:
                continue
            domain = _smallest_key_producer(
                node.children[1],
                right_key.name,
                row_estimates,
                source_costs,
                selective_nodes,
                require_selective=True,
            )
            if domain is not None:
                yield domain, right_key


def _smallest_node_containing_all(
    root: IR,
    columns: Sequence[str],
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
) -> _Producer | None:
    candidates = []
    lineages = [tuple(_column_bindings(root, column)) for column in columns]
    if not lineages or any(not lineage for lineage in lineages):
        return None
    for node, first_column in lineages[0]:
        bound_columns = [first_column]
        for lineage in lineages[1:]:
            match = next(
                (
                    bound_column
                    for candidate, bound_column in lineage
                    if candidate is node
                ),
                None,
            )
            if match is None:
                break
            bound_columns.append(match)
        else:
            rows = row_estimates.get(node)
            if rows is None or rows <= 0:
                continue
            cost = source_costs.get(node)
            if cost is None:
                continue
            producer = _Producer(node, tuple(bound_columns), rows, cost)
            candidates.append((cost, rows, len(node.schema), producer))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _largest_key_source(
    root: IR,
    column: str,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
) -> _Producer | None:
    source_candidates = []
    fallback_candidates = []
    for node, bound_column in _column_bindings(root, column):
        rows = row_estimates.get(node)
        if rows is None or rows <= 0:
            continue
        cost = source_costs.get(node)
        if cost is None:
            continue
        item = (rows, len(node.schema), _Producer(node, (bound_column,), rows, cost))
        if isinstance(node, (Scan, DataFrameScan)):
            source_candidates.append(item)
        else:
            fallback_candidates.append(item)
    candidates = source_candidates or fallback_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]


def _largest_key_reusable_target(
    root: IR,
    column: str,
    domain: IR,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
) -> _Producer | None:
    """Select a target without duplicating a source used by the domain."""
    shared_cache_candidates = []
    source_candidates = []
    fallback_candidates = []
    domain_sources = _scan_sources(domain)
    for order, (node, bound_column) in enumerate(_column_bindings(root, column)):
        rows = row_estimates.get(node)
        if rows is None or rows <= 0:
            continue
        cost = source_costs.get(node)
        if cost is None:
            continue
        producer = _Producer(node, (bound_column,), rows, cost)
        item = (rows, len(node.schema), -order, producer)
        if _contains_identity(domain, node):
            if isinstance(node, Cache):
                shared_cache_candidates.append(item)
            continue
        if not _scan_sources(node).isdisjoint(domain_sources):
            continue
        if isinstance(node, (Scan, DataFrameScan)):
            source_candidates.append(item)
        else:
            fallback_candidates.append(item)
    candidates = shared_cache_candidates or source_candidates or fallback_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1], item[2]))[3]


def _scan_sources(root: IR) -> frozenset[IR]:
    """Return physical scan nodes reachable from a subtree."""
    return frozenset(
        node for node in traversal([root]) if isinstance(node, (Scan, DataFrameScan))
    )


def _column_bindings(root: IR, column: str) -> Iterable[tuple[IR, str]]:
    """Yield exact output-to-input bindings for a column through a subplan."""
    node = root
    while column in node.schema:
        yield node, column
        binding = _input_binding(node, column)
        if binding is None:
            return
        node, column = binding


def _input_binding(node: IR, column: str) -> tuple[IR, str] | None:
    """Return a proven direct input binding, stopping at ambiguous operations."""
    child = node.children[0] if len(node.children) == 1 else None
    if isinstance(node, Select):
        selected = next((item for item in node.exprs if item.name == column), None)
        return _column_expression_binding(child, selected)
    if isinstance(node, HStack):
        stacked = next((item for item in node.columns if item.name == column), None)
        if stacked is not None:
            return _column_expression_binding(child, stacked)
        return _passthrough_binding(child, column)
    if isinstance(node, GroupBy):
        if node.zlice is not None:
            return None
        key = next((item for item in node.keys if item.name == column), None)
        return _column_expression_binding(child, key)
    if isinstance(node, Join):
        return _join_input_binding(node, column)
    if isinstance(node, Distinct):
        return None if node.zlice is not None else _passthrough_binding(child, column)
    if isinstance(node, Sort):
        return None if node.zlice is not None else _passthrough_binding(child, column)
    if isinstance(node, (Cache, Filter, Projection)):
        return _passthrough_binding(child, column)
    return None


def _column_expression_binding(
    child: IR | None, expression: expr.NamedExpr | None
) -> tuple[IR, str] | None:
    if (
        child is not None
        and expression is not None
        and isinstance(expression.value, expr.Col)
        and expression.value.name in child.schema
    ):
        return child, expression.value.name
    return None


def _passthrough_binding(child: IR | None, column: str) -> tuple[IR, str] | None:
    if child is not None and column in child.schema:
        return child, column
    return None


def _join_input_binding(node: Join, column: str) -> tuple[IR, str] | None:
    if node.options[2] is not None:
        return None
    left, right = node.children
    if node.options[0] in ("Semi", "Anti"):
        return _passthrough_binding(left, column)
    if node.options[0] != "Inner":
        return None
    bindings = []
    if column in left.schema:
        bindings.append((left, column))
    suffix = node.options[3]
    for right_column in right.schema:
        output_column = (
            f"{right_column}{suffix}" if right_column in left.schema else right_column
        )
        if output_column == column and output_column in node.schema:
            bindings.append((right, right_column))
    if len(bindings) == 1:
        return bindings[0]
    return None


def _estimate_row_counts(ir: IR, stats: StatsCollector) -> dict[IR, int | None]:
    estimates: dict[IR, int | None] = {}
    for node in post_traversal([ir]):
        if isinstance(node, (Scan, DataFrameScan)):
            source = stats.scan_stats.get(node)
            rows = None if source is None else source.row_count
            if rows is None and isinstance(node, DataFrameScan):
                rows = node.df.shape()[0]
        elif isinstance(
            node, (Select, Projection, HStack, Cache, Filter, Distinct, GroupBy)
        ):
            rows = estimates[node.children[0]]
        elif isinstance(node, Join):
            rows = _estimate_join_rows(
                node.options[0],
                estimates[node.children[0]],
                estimates[node.children[1]],
            )
        else:
            child_estimates = [
                estimate
                for child in node.children
                if (estimate := estimates[child]) is not None
            ]
            rows = max(child_estimates, default=None)
        estimates[node] = rows
    return estimates


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


def _estimate_source_stats(
    ir: IR, row_estimates: dict[IR, int | None]
) -> tuple[dict[IR, int | None], dict[IR, int]]:
    source_costs: dict[IR, int | None] = {}
    source_counts: dict[IR, int] = {}
    source_nodes: dict[IR, set[IR]] = {}
    for node in post_traversal([ir]):
        sources: set[IR]
        if isinstance(node, (Scan, DataFrameScan)):
            sources = {node}
        else:
            sources = set()
            for child in node.children:
                sources.update(source_nodes[child])
        source_nodes[node] = sources
        source_counts[node] = len(sources)
        rows = [
            rows
            for source in sources
            if (rows := row_estimates.get(source)) is not None and rows > 0
        ]
        source_costs[node] = sum(rows) if rows else row_estimates.get(node)
    return source_costs, source_counts


def _domain_cost_is_small(
    domain: IR,
    target: IR,
    threshold: float,
    row_estimates: dict[IR, int | None],
    source_costs: dict[IR, int | None],
) -> bool:
    """Return whether building a domain is cheap enough for the target it reduces."""
    domain_cost = source_costs.get(domain)
    target_rows = row_estimates.get(target)
    if domain_cost is None or target_rows is None or target_rows <= 0:
        return False
    return domain_cost / target_rows <= threshold


def _has_filtering_semi_ancestor(root: IR, target: IR) -> bool:
    """Return whether target is already below a semi join's filtered side."""
    if root is target:
        return False

    for index, child in enumerate(root.children):
        if not _contains_identity(child, target):
            continue
        if isinstance(root, Join) and root.options[0] == "Semi" and index == 0:
            return True
        if _has_filtering_semi_ancestor(child, target):
            return True
    return False


def _collect_selective_nodes(ir: IR) -> set[IR]:
    selective: set[IR] = set()
    for node in post_traversal([ir]):
        if (
            (isinstance(node, Scan) and node.predicate is not None)
            or isinstance(node, Filter)
            or any(child in selective for child in node.children)
        ):
            selective.add(node)
    return selective


def _contains_identity(root: IR, needle: IR) -> bool:
    return any(node is needle for node in traversal([root]))


def _trace_decision(
    ir: Join,
    threshold: float,
    candidate: _Candidate | None,
    reason: str,
    source_costs: dict[IR, int | None],
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
                "estimated_target_cost": source_costs.get(candidate.target.node),
                "estimated_domain_cost": candidate.domain_cost,
                "target_node_type": type(candidate.target.node).__name__,
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
                    "estimated_constraint_cost": candidate.constraint_domain.cost,
                }
            )
    log("Join Domain Prefilter", **record)

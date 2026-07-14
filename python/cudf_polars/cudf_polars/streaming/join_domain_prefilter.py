# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic derived key-domain prefilters for streaming joins."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    ConditionalJoin,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Rolling,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.tracing import Scope, log
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    post_traversal,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.dsl.utils.column_domain import (
    ColumnLineage,
    ColumnRef,
    column_domain_bindings,
)
from cudf_polars.dsl.utils.replace import replace

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.typing import GenericTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@dataclass(frozen=True)
class _Producer:
    """A subtree and its bound column names at an insertion point."""

    node: IR
    columns: tuple[str, ...]
    rows: int

    @property
    def column(self) -> str:
        """First bound column in the producer."""
        return self.columns[0]


@dataclass(frozen=True)
class SimpleCandidate:
    """A direct key-domain prefilter candidate."""

    mode = "simple"
    target_side: Literal["left", "right"]
    target: _Producer
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col

    @property
    def score(self) -> tuple[int, int, int]:
        """Rank after composite candidates, then by domain size."""
        return (1, self.domain.rows, self.domain.rows)


@dataclass(frozen=True)
class CompositeCandidate:
    """A key-domain prefilter constrained by another join key."""

    mode = "composite"
    target_side: Literal["left", "right"]
    target: _Producer
    target_key: expr.Col
    domain: _Producer
    domain_key: expr.Col
    constraint_domain: _Producer
    domain_constraint_key: expr.Col
    target_constraint_key: expr.Col

    @property
    def score(self) -> tuple[int, int, int]:
        """Prefer smaller constraint and domain inputs."""
        return (0, self.constraint_domain.rows, self.domain.rows)


Candidate: TypeAlias = SimpleCandidate | CompositeCandidate
DecisionReason: TypeAlias = Literal[
    "applied",
    "maintain_order",
    "no_selective_domain",
    "non_column_join_key",
    "not_inner_join",
    "sliced_join",
]


@dataclass(frozen=True)
class Decision:
    """Result of considering a join for a domain prefilter."""

    reason: DecisionReason
    candidate: Candidate | None = None


@dataclass(frozen=True)
class PlanFacts:
    """Facts derived in one bottom-up traversal of an IR DAG."""

    row_estimates: Mapping[IR, int | None]
    selective_nodes: frozenset[IR]
    column_lineages: Mapping[ColumnRef, ColumnLineage]


class _RewriteState(TypedDict):
    """State shared by the join-domain prefilter DAG rewrite."""

    threshold: float
    trace: bool
    stats: StatsCollector
    facts: PlanFacts


def analyze_plan(ir: IR, stats: StatsCollector) -> PlanFacts:
    """
    Derive row, selectivity, and column-domain facts for an IR DAG.

    Parameters
    ----------
    ir
        Root node to gather facts for.
    stats
        Pre-populated statistics

    Returns
    -------
    Gather facts about the plan.
    """
    row_estimates: dict[IR, int | None] = {}
    selective_nodes: set[IR] = set()
    column_lineages: dict[ColumnRef, ColumnLineage] = {}

    for node in post_traversal([ir]):
        if isinstance(node, (Scan, DataFrameScan)):
            source_info = stats.scan_stats.get(node)
            rows = None if source_info is None else source_info.row_count
            if rows is None and isinstance(node, DataFrameScan):
                rows = node.df.shape()[0]
        elif isinstance(
            node, (Select, Projection, HStack, Cache, Filter, Distinct, GroupBy)
        ):
            rows = row_estimates[node.children[0]]
        elif isinstance(node, Join):
            rows = _estimate_join_rows(
                node.options[0],
                row_estimates[node.children[0]],
                row_estimates[node.children[1]],
            )
        else:
            child_estimates = [
                estimate
                for child in node.children
                if (estimate := row_estimates[child]) is not None
            ]
            rows = max(child_estimates, default=None)
        row_estimates[node] = rows

        if (
            (isinstance(node, Scan) and node.predicate is not None)
            or isinstance(node, Filter)
            or any(child in selective_nodes for child in node.children)
        ):
            selective_nodes.add(node)

        bindings = column_domain_bindings(node)
        for name in node.schema:
            column = ColumnRef(node, name)
            source = bindings.get(name)
            if source is None:
                source_lineage = None
                source_child_index = None
            else:
                source_lineage = column_lineages[source]
                source_children = tuple(
                    index
                    for index, child in enumerate(node.children)
                    if child == source.node
                )
                source_child_index = (
                    source_children[0] if len(source_children) == 1 else None
                )
            column_lineages[column] = ColumnLineage(
                column, source_lineage, source_child_index
            )

    return PlanFacts(
        row_estimates=row_estimates,
        selective_nodes=frozenset(selective_nodes),
        column_lineages=column_lineages,
    )


def blocks_pushdown(node: IR) -> bool:
    """
    Return whether a node blocks filter pushdown.

    Parameters
    ----------
    node
        Node to check

    Returns
    -------
    bool
        True if a semijoin cannot be pushed past this node, otherwise False.
    """
    return (
        # TODO: Distinct and Rolling only block pushdown in some
        # circumstances, but we'd need to make the logic more complicated:
        # - We can push through distinct if the filter applies to the columns
        #   that are being used to determine distinct rows
        # - We can push through rolling if the filter applies to the
        #   groupby keys.
        # TODO: We can push through an unsliced Union, but need to
        # distribute the filter onto every child.
        isinstance(node, (Distinct, Rolling, Slice, Union))
        # Can't push through anything that is sliced.
        or (isinstance(node, (GroupBy, Sort)) and node.zlice is not None)
        or (isinstance(node, (ConditionalJoin, Join)) and node.options[2] is not None)
    )


def semijoin_pushdown_candidates(
    facts: PlanFacts, root: IR, column: str
) -> Iterator[ColumnRef]:
    """
    Yield column domain lineage providing valid locations for semijoin pushdown.

    Parameters
    ----------
    facts
        Gathered facts about the plan
    root
        Root node to search from
    column
        Name of column we're finding the lineage of.

    Returns
    -------
    Iterator
        Of valid insertion points for a semijoin filter on the given column name.
    """
    try:
        lineage = facts.column_lineages[ColumnRef(root, column)]
    except KeyError:
        return
    while True:
        yield lineage.column
        if (
            blocks_pushdown(lineage.column.node)
            or lineage.source is None
            or lineage.source_child_index is None
        ):
            return
        lineage = lineage.source


def optimize_join_domain_prefilters(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions[StreamingExecutor],
) -> IR:
    """
    Insert generic semi-join key-domain prefilters before streaming lowering.

    The rewrite is intentionally conservative: only inner joins with simple
    column equality keys are considered.
    """
    options = config_options.executor.join_domain_prefilter
    if options is None:
        return ir
    threshold = options.threshold
    trace = options.trace
    if threshold == 0:
        return ir

    state = _RewriteState(
        threshold=threshold,
        trace=trace,
        stats=stats,
        facts=analyze_plan(ir, stats),
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
        facts = rec.state["facts"]
    else:
        # Child rewrites introduce new semi joins and reconstructed ancestors.
        # Re-analyze that current subtree so parent joins can use the derived
        # selectivity and cardinality when ranking their own candidates.
        facts = analyze_plan(node, rec.state["stats"])
    decision = _select_candidate(
        node,
        rec.state["threshold"],
        facts,
    )
    if rec.state["trace"]:
        _trace_decision(node, rec.state["threshold"], decision)
    if decision.candidate is None:
        return node
    return apply_candidate(node, decision.candidate)


def apply_candidate(ir: Join, candidate: Candidate) -> IR:
    """Apply a selected join-domain prefilter candidate to a join."""
    left, right = ir.children
    domain = _make_domain(candidate, ir)
    target = candidate.target
    target_filter = _make_semi_join(
        target.node,
        expr.Col(target.node.schema[target.column], target.column),
        domain,
        expr.Col(domain.schema[candidate.domain_key.name], candidate.domain_key.name),
        nulls_equal=ir.options[1],
        suffix=ir.options[3],
    )
    # A DAG may share the target with the domain side, so only rewrite the
    # side for which this candidate was selected.
    if candidate.target_side == "left":
        (left,) = replace([left], {candidate.target.node: target_filter})
    else:
        (right,) = replace([right], {candidate.target.node: target_filter})
    return ir.reconstruct((left, right))


def _select_candidate(
    ir: Join,
    threshold: float,
    facts: PlanFacts,
) -> Decision:
    if ir.options[0] != "Inner":
        return Decision(reason="not_inner_join")
    if ir.options[2] is not None:
        return Decision(reason="sliced_join")
    if ir.options[5] != "none":
        return Decision(reason="maintain_order")

    left_keys = _simple_keys(ir.left_on)
    right_keys = _simple_keys(ir.right_on)
    if len(left_keys) != len(ir.left_on) or len(right_keys) != len(ir.right_on):
        return Decision(reason="non_column_join_key")

    candidates: list[Candidate] = []
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
                facts,
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
                facts,
            )
        )

    if not candidates:
        return Decision(reason="no_selective_domain")
    return Decision(reason="applied", candidate=min(candidates, key=lambda c: c.score))


def _simple_keys(keys: Sequence[expr.NamedExpr]) -> tuple[expr.Col, ...]:
    return tuple(key.value for key in keys if isinstance(key.value, expr.Col))


def _simple_candidates(
    target_side: Literal["left", "right"],
    target_child: IR,
    domain_child: IR,
    target_keys: tuple[expr.Col, ...],
    domain_keys: tuple[expr.Col, ...],
    threshold: float,
    facts: PlanFacts,
) -> Iterable[SimpleCandidate]:
    for target_key, domain_key in zip(target_keys, domain_keys, strict=True):
        target = _largest_key_source(target_child, target_key.name, facts)
        if target is None:
            continue
        domain = _smallest_key_producer(
            domain_child,
            domain_key.name,
            facts,
            require_selective=True,
        )
        if domain is None:
            continue
        if domain.rows / target.rows > threshold:
            continue
        if contains_node(target.node, domain.node):
            continue
        yield SimpleCandidate(
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
    facts: PlanFacts,
) -> Iterable[CompositeCandidate]:
    if len(target_keys) < 2:
        return

    for filter_index, (target_key, domain_key) in enumerate(
        zip(target_keys, domain_keys, strict=True)
    ):
        target = _largest_key_source(target_child, target_key.name, facts)
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
                facts,
            )
            if domain is None:
                continue
            if domain.rows / target.rows > threshold:
                continue
            constraint_domain = _smallest_key_producer(
                target_child,
                target_constraint_key.name,
                facts,
                require_selective=True,
                exclude=target.node,
            )
            if constraint_domain is None:
                continue
            if constraint_domain.rows / domain.rows > threshold:
                continue
            if contains_node(target.node, domain.node) or contains_node(
                target.node, constraint_domain.node
            ):
                continue
            yield CompositeCandidate(
                target_side=target_side,
                target=target,
                target_key=target_key,
                domain=domain,
                domain_key=domain_key,
                constraint_domain=constraint_domain,
                domain_constraint_key=domain_constraint_key,
                target_constraint_key=target_constraint_key,
            )


def _make_domain(candidate: Candidate, ir: Join) -> IR:
    if isinstance(candidate, SimpleCandidate):
        return _project_bound_key(
            candidate.domain.node,
            candidate.domain.column,
            candidate.domain_key,
        )

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
    facts: PlanFacts,
    *,
    require_selective: bool,
    exclude: IR | None = None,
) -> _Producer | None:
    candidates = []
    for reference in semijoin_pushdown_candidates(facts, root, column):
        node, bound_column = reference.node, reference.name
        if node is exclude:
            continue
        rows = facts.row_estimates.get(node)
        if rows is None or rows <= 0:
            continue
        if require_selective and node not in facts.selective_nodes:
            continue
        candidates.append(
            (rows, len(node.schema), _Producer(node, (bound_column,), rows))
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _smallest_node_containing_all(
    root: IR, columns: Sequence[str], facts: PlanFacts
) -> _Producer | None:
    candidates = []
    lineages: list[ColumnLineage] = []
    for column in columns:
        lineage = facts.column_lineages.get(ColumnRef(root, column))
        if lineage is None:
            return None
        lineages.append(lineage)
    if not lineages:
        return None
    while True:
        node = lineages[0].column.node
        if any(lineage.column.node != node for lineage in lineages[1:]):
            break
        bound_columns = tuple(lineage.column.name for lineage in lineages)
        rows = facts.row_estimates.get(node)
        if rows is not None and rows > 0:
            candidates.append(
                (
                    rows,
                    len(node.schema),
                    _Producer(node, bound_columns, rows),
                )
            )
        if blocks_pushdown(node):
            break
        source_child_indices = {lineage.source_child_index for lineage in lineages}
        if len(source_child_indices) != 1 or None in source_child_indices:
            break
        sources = [lineage.source for lineage in lineages]
        if any(source is None for source in sources):
            break
        lineages = [source for source in sources if source is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _largest_key_source(root: IR, column: str, facts: PlanFacts) -> _Producer | None:
    source_candidates = []
    fallback_candidates = []
    for reference in semijoin_pushdown_candidates(facts, root, column):
        node, bound_column = reference.node, reference.name
        rows = facts.row_estimates.get(node)
        if rows is None or rows <= 0:
            continue
        item = (rows, len(node.schema), _Producer(node, (bound_column,), rows))
        if isinstance(node, (Scan, DataFrameScan)):
            source_candidates.append(item)
        else:
            fallback_candidates.append(item)
    candidates = source_candidates or fallback_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]


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


def contains_node(root: IR, needle: IR) -> bool:
    """Return whether an equal node occurs in a DAG rooted at ``root``."""
    return needle in traversal([root])


def _trace_decision(ir: Join, threshold: float, decision: Decision) -> None:
    join_domain_prefilter: dict[str, Any] = {
        "considered": True,
        "threshold": threshold,
        "reason": decision.reason,
    }
    record = {
        "scope": Scope.PLAN.value,
        "join_domain_prefilter": join_domain_prefilter,
        "actor_ir_id": ir.get_stable_id(),
        "actor_ir_type": type(ir).__name__,
    }
    if (candidate := decision.candidate) is not None:
        join_domain_prefilter.update(
            {
                "mode": candidate.mode,
                "target_side": candidate.target_side,
                "target_key": candidate.target_key.name,
                "domain_key": candidate.domain_key.name,
                "estimated_target_rows": candidate.target.rows,
                "estimated_domain_rows": candidate.domain.rows,
                "target_node_type": type(candidate.target.node).__name__,
                "domain_node_type": type(candidate.domain.node).__name__,
            }
        )
        if isinstance(candidate, CompositeCandidate):
            join_domain_prefilter.update(
                {
                    "constraint_key": candidate.target_constraint_key.name,
                    "estimated_constraint_rows": candidate.constraint_domain.rows,
                }
            )
    log("Join Domain Prefilter", **record)

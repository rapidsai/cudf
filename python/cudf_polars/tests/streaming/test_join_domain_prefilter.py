# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import Cache, Distinct, Join, Scan, Select, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.column_domain import ColumnRef
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.join_domain_prefilter import (
    CompositeCandidate,
    Decision,
    PlanFacts,
    SimpleCandidate,
    _select_candidate,
    _smallest_node_containing_all,
    analyze_plan,
    apply_candidate,
    contains_node,
    optimize_join_domain_prefilters,
    semijoin_pushdown_candidates,
)
from cudf_polars.streaming.parallel import optimize_with_stats, remove_cache_nodes
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions, ParquetOptions

if TYPE_CHECKING:
    import concurrent.futures

    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.streaming.base import SerializedDataSourceInfo

I64 = DataType(pl.Int64())
BOOL = DataType(pl.Boolean())


@pytest.fixture
def engine(spmd_engine_factory) -> SPMDEngine:
    """Return an SPMD engine configured for join-domain prefilter tests."""
    return spmd_engine_factory(
        StreamingOptions(
            join_domain_prefilter={"threshold": 0.5},
            raise_on_fail=True,
        )
    )


class _SourceInfo:
    type: Literal["parquet"] = "parquet"

    def __init__(self, row_count: int | None) -> None:
        self.row_count = row_count

    def column_storage_size(self, column: str) -> int | None:
        del column
        return None

    def serialize(self) -> SerializedDataSourceInfo:
        return {"type": self.type, "row_count": self.row_count, "per_file_means": {}}

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> _SourceInfo:
        return cls(data["row_count"])


def _scan(name: str, columns: tuple[str, ...], *, predicate: bool = False) -> Scan:
    schema = dict.fromkeys(columns, I64)
    mask = (
        expr.NamedExpr("__predicate", expr.Literal(BOOL, True))  # noqa: FBT003
        if predicate
        else None
    )
    return Scan(
        schema,
        "parquet",
        {},
        None,
        [f"/tmp/{name}.parquet"],
        list(columns),
        0,
        -1,
        None,
        None,
        mask,
        ParquetOptions(),
    )


def _key(node: IR, name: str) -> expr.NamedExpr:
    return expr.NamedExpr(name, expr.Col(node.schema[name], name))


def _select(node: IR, **columns: str) -> Select:
    schema = {output: node.schema[source] for output, source in columns.items()}
    return Select(
        schema,
        tuple(
            expr.NamedExpr(output, expr.Col(schema[output], source))
            for output, source in columns.items()
        ),
        True,  # noqa: FBT003
        node,
    )


def _join(
    left: IR,
    right: IR,
    left_on: tuple[str, ...],
    right_on: tuple[str, ...],
    *,
    how: str = "Inner",
    maintain_order: str = "none",
) -> Join:
    schema = dict(left.schema)
    schema.update(right.schema)
    return Join(
        schema,
        tuple(_key(left, name) for name in left_on),
        tuple(_key(right, name) for name in right_on),
        (how, False, None, "_right", False, maintain_order),
        left,
        right,
    )


def _stats(**row_counts: tuple[Scan, int]) -> StatsCollector:
    stats = StatsCollector()
    for scan, rows in row_counts.values():
        stats.scan_stats[scan] = _SourceInfo(rows)
    return stats


def _config(
    *, dynamic_planning: bool = True, join_domain_prefilter: bool = True
) -> ConfigOptions:
    executor_options: dict[str, object] = {
        "join_domain_prefilter": {"trace": False} if join_domain_prefilter else None
    }
    if not dynamic_planning:
        executor_options["dynamic_planning"] = None
    return ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options=executor_options,
        )
    )


def _joins(ir: IR, how: str | None = None) -> list[Join]:
    return [
        node
        for node in traversal([ir])
        if isinstance(node, Join) and (how is None or node.options[0] == how)
    ]


def test_simple_domain_prefilter_filters_large_side() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey", "l_suppkey"))
    root = _join(part, lineitem, ("p_partkey",), ("l_partkey",))

    facts = analyze_plan(root, _stats(part=(part, 6), lineitem=(lineitem, 1_800)))
    decision = _select_candidate(root, 0.5, facts)

    assert decision.reason == "applied"
    assert isinstance(decision.candidate, SimpleCandidate)
    optimized = apply_candidate(root, decision.candidate)

    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    assert isinstance(optimized.children[1], Join)
    assert optimized.children[1].options[0] == "Semi"
    assert optimized.children[1].children[0] is lineitem
    assert optimized.children[0] is part


def test_domain_prefilter_is_independent_of_dynamic_planning() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey",))
    root = _join(part, lineitem, ("p_partkey",), ("l_partkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(part=(part, 6), lineitem=(lineitem, 1_800)),
        _config(dynamic_planning=False),
    )

    assert _joins(optimized, "Semi")


def test_domain_prefilter_can_be_disabled() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey",))
    root = _join(part, lineitem, ("p_partkey",), ("l_partkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(part=(part, 6), lineitem=(lineitem, 1_800)),
        _config(join_domain_prefilter=False),
    )

    assert optimized is root


@pytest.mark.parametrize(
    "nulls_equal", [False, True], ids=["nulls_not_equal", "nulls_equal"]
)
def test_nullable_join_keys_preserve_results(
    nulls_equal: bool,  # noqa: FBT001
    engine: SPMDEngine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    domain = pl.LazyFrame(
        {
            "key": [None, 1, 2, 9],
            "active": [True, True, True, False],
        }
    ).filter("active")
    target = pl.LazyFrame(
        {
            "key": [None, 1, 2, 3] * 10,
            "value": range(40),
        }
    )
    query = domain.join(target, on="key", nulls_equal=nulls_equal)

    ir = Translator(query._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    optimized = optimize_join_domain_prefilters(
        ir,
        collect_statistics(ir, config, parquet_stats_executor),
        config,
    )

    semi_joins = _joins(optimized, "Semi")
    assert semi_joins
    assert all(join.options[1] is nulls_equal for join in semi_joins)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_prefilter_does_not_move_below_distinct_on_non_subset_column(
    engine: SPMDEngine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    target = pl.LazyFrame(
        {
            "group": [1, 1] * 100,
            "key": [0, 1] * 100,
        }
    ).unique(subset="group", keep="first", maintain_order=True)
    domain = pl.LazyFrame(
        {
            "key": [1, 2],
            "active": [True, False],
        }
    ).filter("active")
    query = target.join(domain, on="key")

    ir = Translator(query._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    optimized = optimize_join_domain_prefilters(
        ir,
        collect_statistics(ir, config, parquet_stats_executor),
        config,
    )

    semis = _joins(optimized, "Semi")
    assert any(isinstance(semi.children[0], Distinct) for semi in semis)
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_no_simple_domain_prefilter_when_domain_is_not_selective() -> None:
    supplier = _scan("supplier", ("s_suppkey",))
    lineitem = _scan("lineitem", ("l_suppkey",))
    root = _join(supplier, lineitem, ("s_suppkey",), ("l_suppkey",))
    stats = _stats(supplier=(supplier, 30), lineitem=(lineitem, 1_800))

    decision = _select_candidate(root, 0.5, analyze_plan(root, stats))

    optimized = optimize_join_domain_prefilters(
        root,
        stats,
        _config(),
    )

    assert decision == Decision(reason="no_selective_domain")
    assert optimized is root
    assert not _joins(optimized, "Semi")


def test_composite_domain_prefilter_constrains_domain_first() -> None:
    nation = _scan("nation", ("n_nationkey",), predicate=True)
    orders = _scan("orders", ("o_orderkey", "n_nationkey"))
    lineitem = _scan("lineitem", ("l_orderkey", "l_suppkey"))
    supplier = _scan("supplier", ("s_suppkey", "s_nationkey"))

    nation_orders = _join(nation, orders, ("n_nationkey",), ("n_nationkey",))
    order_lineitem = _join(
        nation_orders,
        lineitem,
        ("o_orderkey",),
        ("l_orderkey",),
        maintain_order="left",
    )
    root = _join(
        order_lineitem,
        supplier,
        ("l_suppkey", "n_nationkey"),
        ("s_suppkey", "s_nationkey"),
    )

    stats = _stats(
        nation=(nation, 5),
        orders=(orders, 900),
        lineitem=(lineitem, 1_800),
        supplier=(supplier, 30),
    )
    decision = _select_candidate(root, 0.5, analyze_plan(root, stats))
    optimized = optimize_join_domain_prefilters(
        root,
        stats,
        _config(),
    )

    assert decision.reason == "applied"
    assert isinstance(decision.candidate, CompositeCandidate)
    semis = _joins(optimized, "Semi")
    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    assert optimized.children[1] is supplier
    assert any(semi.children[0] is supplier for semi in semis)
    assert any(semi.children[0] is lineitem for semi in semis)


def test_derived_selectivity_propagates_through_rewritten_children() -> None:
    region = _scan("region", ("r_regionkey",), predicate=True)
    nation = _scan("nation", ("n_nationkey", "n_regionkey"))
    customer = _scan("customer", ("c_custkey", "c_nationkey"))
    orders = _scan("orders", ("o_orderkey", "o_custkey"))

    region_nation = _join(region, nation, ("r_regionkey",), ("n_regionkey",))
    nation_customer = _join(region_nation, customer, ("n_nationkey",), ("c_nationkey",))
    root = _join(nation_customer, orders, ("c_custkey",), ("o_custkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            region=(region, 1),
            nation=(nation, 25),
            customer=(customer, 150),
            orders=(orders, 1_500),
        ),
        _config(),
    )

    filtered = {semi.children[0] for semi in _joins(optimized, "Semi")}
    assert {nation, customer, orders} <= filtered


def test_rewritten_domain_filters_other_side_instead_of_stacking() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_orderkey", "l_partkey", "l_suppkey"))
    supplier = _scan("supplier", ("s_suppkey",))
    orders = _scan("orders", ("o_orderkey",), predicate=True)

    part_lineitem = _join(part, lineitem, ("p_partkey",), ("l_partkey",))
    line_supplier = _join(part_lineitem, supplier, ("l_suppkey",), ("s_suppkey",))
    root = _join(line_supplier, orders, ("l_orderkey",), ("o_orderkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            part=(part, 60),
            lineitem=(lineitem, 1_800),
            supplier=(supplier, 30),
            orders=(orders, 150),
        ),
        _config(),
    )

    semis = _joins(optimized, "Semi")
    assert sum(semi.children[0] is lineitem for semi in semis) == 1
    assert any(semi.children[0] is orders for semi in semis)
    assert not any(
        isinstance(semi.children[0], Join) and semi.children[0].options[0] == "Semi"
        for semi in semis
    )


def test_target_source_follows_join_key_through_rename() -> None:
    big = _scan("big", ("left_key", "other"))
    renamed_big = _select(big, foo="left_key", other="other")
    small = _scan("small", ("left_key", "other2"))
    joined = _join(
        renamed_big,
        small,
        ("other",),
        ("other2",),
        maintain_order="left",
    )
    domain = _scan("domain", ("domain_key",), predicate=True)
    root = _join(joined, domain, ("left_key",), ("domain_key",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(big=(big, 1_000), small=(small, 100), domain=(domain, 5)),
        _config(),
    )

    semis = _joins(optimized, "Semi")
    assert any(semi.children[0] is small for semi in semis)
    assert not any(semi.children[0] is big for semi in semis)


def test_domain_source_follows_join_key_through_rename() -> None:
    target = _scan("target", ("target_key",))
    unrelated = _scan("unrelated", ("domain_key", "other"), predicate=True)
    renamed_unrelated = _select(unrelated, foo="domain_key", other="other")
    domain_source = _scan("domain_source", ("domain_key", "other2"), predicate=True)
    domain = _join(
        renamed_unrelated,
        domain_source,
        ("other",),
        ("other2",),
        maintain_order="left",
    )
    root = _join(target, domain, ("target_key",), ("domain_key",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            target=(target, 1_000),
            unrelated=(unrelated, 1),
            domain_source=(domain_source, 5),
        ),
        _config(),
    )

    semi = next(
        semi for semi in _joins(optimized, "Semi") if semi.children[0] is target
    )
    selected_domain = semi.children[1]
    assert isinstance(selected_domain, Select)
    assert selected_domain.children[0] is domain


def test_composite_domain_columns_follow_renames() -> None:
    source = _scan("source", ("raw_key", "raw_constraint"))
    renamed = _select(
        source,
        domain_key="raw_key",
        domain_constraint="raw_constraint",
    )

    analyzed = analyze_plan(renamed, _stats(source=(source, 10)))
    facts = PlanFacts(
        row_estimates={renamed: 20, source: 10},
        selective_nodes=analyzed.selective_nodes,
        column_lineages=analyzed.column_lineages,
    )
    producer = _smallest_node_containing_all(
        renamed, ("domain_key", "domain_constraint"), facts
    )

    assert producer is not None
    assert producer.node is source
    assert producer.columns == ("raw_key", "raw_constraint")


def test_composite_domain_columns_do_not_reconverge_after_join(
    engine: SPMDEngine,
) -> None:
    source = pl.LazyFrame({"key": [1, 1, 2], "value": [10, 20, 30]})
    query = source.join(source, on="key", suffix="_right")
    joined = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(joined, Join)
    assert isinstance(joined.children[0], Cache)
    assert joined.children[0] is joined.children[1]
    joined = remove_cache_nodes(joined)
    assert isinstance(joined, Join)
    assert joined.children[0] is joined.children[1]

    facts = analyze_plan(joined, StatsCollector())
    producer = _smallest_node_containing_all(joined, ("value", "value_right"), facts)

    assert tuple(semijoin_pushdown_candidates(facts, joined, "value")) == (
        (ColumnRef(joined, "value"), ()),
        (ColumnRef(joined.children[0], "value"), (0,)),
    )
    assert producer is not None
    assert producer.node is joined
    assert producer.columns == ("value", "value_right")


def test_contains_node_uses_dag_equality() -> None:
    source = _scan("source", ("key",))
    equal_source = _scan("source", ("key",))
    root = _select(source, key="key")

    assert source is not equal_source
    assert source == equal_source
    assert contains_node(root, equal_source)


def test_plan_facts_share_lineage_suffixes_across_shared_dag() -> None:
    source = _scan("source", ("raw_key",))
    left = _select(source, left_key="raw_key")
    right = _select(source, right_key="raw_key")
    root = _join(left, right, ("left_key",), ("right_key",))

    facts = analyze_plan(root, _stats(source=(source, 10)))
    left_lineage = facts.column_lineages[ColumnRef(left, "left_key")]
    right_lineage = facts.column_lineages[ColumnRef(right, "right_key")]
    source_lineage = facts.column_lineages[ColumnRef(source, "raw_key")]

    assert left_lineage.column == ColumnRef(left, "left_key")
    assert right_lineage.column == ColumnRef(right, "right_key")
    assert left_lineage.source is source_lineage
    assert right_lineage.source is source_lineage
    assert source_lineage.source is None


def test_target_prefilter_does_not_move_below_slice() -> None:
    target = _scan("target", ("target_key",))
    sliced = Slice(target.schema, 0, 100, target)
    domain = _scan("domain", ("domain_key",), predicate=True)
    root = _join(sliced, domain, ("target_key",), ("domain_key",))

    stats = _stats(target=(target, 1_000), domain=(domain, 5))
    facts = analyze_plan(root, stats)
    lineage = facts.column_lineages[ColumnRef(sliced, "target_key")]
    assert lineage.column == ColumnRef(sliced, "target_key")
    assert lineage.source is facts.column_lineages[ColumnRef(target, "target_key")]
    assert lineage.source.source is None
    assert tuple(semijoin_pushdown_candidates(facts, sliced, "target_key")) == (
        (ColumnRef(sliced, "target_key"), ()),
    )

    optimized = optimize_join_domain_prefilters(
        root,
        stats,
        _config(),
    )

    semis = _joins(optimized, "Semi")
    assert any(semi.children[0] is sliced for semi in semis)
    assert not any(semi.children[0] is target for semi in semis)


def test_target_replacement_does_not_rewrite_shared_domain_side() -> None:
    shared = _scan("shared", ("target_key", "other"))
    domain_source = _scan("domain_source", ("domain_key", "other2"), predicate=True)
    domain = _join(
        shared,
        domain_source,
        ("other",),
        ("other2",),
        maintain_order="left",
    )
    root = _join(shared, domain, ("target_key",), ("domain_key",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(shared=(shared, 1_000), domain_source=(domain_source, 5)),
        _config(),
    )

    assert isinstance(optimized, Join)
    assert isinstance(optimized.children[0], Join)
    assert optimized.children[0].options[0] == "Semi"
    assert optimized.children[0].children[0] is shared
    assert optimized.children[1] is domain
    assert domain.children[0] is shared


def test_target_prefilter_rewrites_only_selected_self_join_edge(
    engine: SPMDEngine,
) -> None:
    source = pl.LazyFrame(
        {
            "key": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )
    domain = (
        pl.LazyFrame(
            {
                "domain_value": [10, 999],
                "active": [True, False],
            }
        )
        .filter("active")
        .select("domain_value")
    )
    query = source.join(source, on="key", suffix="_right").join(
        domain,
        left_on="value",
        right_on="domain_value",
    )
    translated = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(translated, Join)
    translated_self_join = translated.children[0]
    assert isinstance(translated_self_join, Join)
    shared_cache = translated_self_join.children[0]
    assert isinstance(shared_cache, Cache)
    assert translated_self_join.children[1] is shared_cache
    source_ir = shared_cache.children[0]

    optimized = optimize_with_stats(
        translated,
        ConfigOptions.from_polars_engine(engine),
        StatsCollector(),
    )

    assert isinstance(optimized, Join)
    rewritten_self_join = optimized.children[0]
    assert isinstance(rewritten_self_join, Join)
    filtered, unfiltered = rewritten_self_join.children
    assert isinstance(filtered, Join)
    assert filtered.options[0] == "Semi"
    assert filtered.children[0] is source_ir
    assert unfiltered is source_ir

    expected = query.collect()
    assert sorted(expected.select("value", "value_right").rows()) == [
        (10, 10),
        (10, 20),
    ]
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


def test_no_domain_prefilter_for_outer_join() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey",))
    root = _join(part, lineitem, ("p_partkey",), ("l_partkey",), how="Left")

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(part=(part, 6), lineitem=(lineitem, 1_800)),
        _config(),
    )

    assert optimized is root
    assert not _joins(optimized, "Semi")

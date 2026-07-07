# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import Cache, Filter, GroupBy, Join, Scan, Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.default_singleton_engine import DefaultSingletonEngine
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.join_domain_prefilter import (
    _smallest_node_containing_all,
    optimize_join_domain_prefilters,
)
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions, ParquetOptions

if TYPE_CHECKING:
    import concurrent.futures

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import SerializedDataSourceInfo

I64 = DataType(pl.Int64())
F64 = DataType(pl.Float64())
BOOL = DataType(pl.Boolean())


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


def _typed_scan(
    name: str, schema: dict[str, DataType], *, predicate: bool = False
) -> Scan:
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
        list(schema),
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


def _contains_node(ir: IR, needle: IR) -> bool:
    return any(node is needle for node in traversal([ir]))


def _join_key_names(keys: tuple[expr.NamedExpr, ...]) -> tuple[str, ...]:
    names = []
    for key in keys:
        assert isinstance(key.value, expr.Col)
        names.append(key.value.name)
    return tuple(names)


def _filtered_groupby_domain(source: IR, key: str) -> Select:
    grouped = GroupBy(
        {key: source.schema[key]},
        (_key(source, key),),
        (),
        False,  # noqa: FBT003
        None,
        source,
    )
    filtered = Filter(
        grouped.schema,
        expr.NamedExpr("__predicate", expr.Literal(BOOL, True)),  # noqa: FBT003
        grouped,
    )
    return Select(
        {key: filtered.schema[key]},
        (_key(filtered, key),),
        True,  # noqa: FBT003
        filtered,
    )


def _filtered_sum_domain(
    source: IR, key: str, value: str, aggregate: str
) -> tuple[Filter, Select]:
    grouped = GroupBy(
        {key: source.schema[key], aggregate: source.schema[value]},
        (_key(source, key),),
        (
            expr.NamedExpr(
                aggregate,
                expr.Agg(
                    source.schema[value],
                    "sum",
                    None,
                    ExecutionContext.GROUPBY,
                    expr.Col(source.schema[value], value),
                ),
            ),
        ),
        False,  # noqa: FBT003
        None,
        source,
    )
    filtered = Filter(
        grouped.schema,
        expr.NamedExpr("__predicate", expr.Literal(BOOL, True)),  # noqa: FBT003
        grouped,
    )
    keys = Select(
        {key: filtered.schema[key]},
        (_key(filtered, key),),
        True,  # noqa: FBT003
        filtered,
    )
    return filtered, keys


def test_simple_domain_prefilter_filters_large_side() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey", "l_suppkey"))
    root = _join(part, lineitem, ("p_partkey",), ("l_partkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(part=(part, 6), lineitem=(lineitem, 1_800)),
        _config(),
    )

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
    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"join_domain_prefilter": {"threshold": 0.5}},
    )

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
    try:
        assert_gpu_result_equal(query, engine=engine, check_row_order=False)
    finally:
        DefaultSingletonEngine.shutdown()


def test_no_simple_domain_prefilter_when_domain_is_not_selective() -> None:
    supplier = _scan("supplier", ("s_suppkey",))
    lineitem = _scan("lineitem", ("l_suppkey",))
    root = _join(supplier, lineitem, ("s_suppkey",), ("l_suppkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(supplier=(supplier, 30), lineitem=(lineitem, 1_800)),
        _config(),
    )

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

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            nation=(nation, 5),
            orders=(orders, 900),
            lineitem=(lineitem, 1_800),
            supplier=(supplier, 30),
        ),
        _config(),
    )

    semis = _joins(optimized, "Semi")
    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    assert optimized.children[1] is supplier
    assert any(semi.children[0] is supplier for semi in semis)
    assert any(semi.children[0] is lineitem for semi in semis)


def test_prefilter_uses_cheaper_source_domain_and_skips_expensive_domain() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    partsupp = _scan("partsupp", ("ps_partkey", "ps_suppkey"))
    supplier = _scan("supplier", ("s_suppkey",))
    lineitem = _scan("lineitem", ("l_partkey", "l_suppkey", "l_orderkey"))
    orders = _scan("orders", ("o_orderkey",))

    part_partsupp = _join(part, partsupp, ("p_partkey",), ("ps_partkey",))
    part_partsupp_supplier = _join(
        part_partsupp, supplier, ("ps_suppkey",), ("s_suppkey",)
    )
    q9_left = _join(
        part_partsupp_supplier,
        lineitem,
        ("p_partkey", "ps_suppkey"),
        ("l_partkey", "l_suppkey"),
    )
    root = _join(q9_left, orders, ("l_orderkey",), ("o_orderkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            part=(part, 60),
            partsupp=(partsupp, 120),
            supplier=(supplier, 30),
            lineitem=(lineitem, 1_800),
            orders=(orders, 900),
        ),
        _config(),
    )

    semis = _joins(optimized, "Semi")
    lineitem_semis = [semi for semi in semis if semi.children[0] is lineitem]
    assert lineitem_semis
    assert not any(semi.children[0] is orders for semi in semis)
    assert _contains_node(lineitem_semis[0].children[1], part)
    assert not _contains_node(lineitem_semis[0].children[1], supplier)


def test_source_only_domain_does_not_stack_on_prefiltered_source() -> None:
    part = _scan("part", ("p_partkey",), predicate=True)
    lineitem = _scan("lineitem", ("l_partkey", "l_orderkey"))
    orders = _scan("orders", ("o_orderkey",), predicate=True)

    part_lineitem = _join(part, lineitem, ("p_partkey",), ("l_partkey",))
    root = _join(part_lineitem, orders, ("l_orderkey",), ("o_orderkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(part=(part, 60), lineitem=(lineitem, 1_800), orders=(orders, 150)),
        _config(),
    )

    lineitem_semis = [
        semi for semi in _joins(optimized, "Semi") if semi.children[0] is lineitem
    ]
    assert any(
        _join_key_names(semi.left_on) == ("l_partkey",) for semi in lineitem_semis
    )
    assert not any(
        _join_key_names(semi.left_on) == ("l_orderkey",) for semi in lineitem_semis
    )


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


def test_rewritten_analysis_respects_stack_and_cost_guards() -> None:
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
    assert not any(semi.children[0] is orders for semi in semis)
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
    assert selected_domain.children[0] is domain_source


def test_composite_domain_columns_follow_renames() -> None:
    source = _scan("source", ("raw_key", "raw_constraint"))
    renamed = _select(
        source,
        domain_key="raw_key",
        domain_constraint="raw_constraint",
    )

    producer = _smallest_node_containing_all(
        renamed,
        ("domain_key", "domain_constraint"),
        {renamed: 20, source: 10},
        {renamed: 10, source: 10},
    )

    assert producer is not None
    assert producer.node is source
    assert producer.columns == ("raw_key", "raw_constraint")


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


def test_reuses_existing_semi_domain_to_filter_shared_target() -> None:
    lineitem = _scan("lineitem", ("l_orderkey", "l_quantity"))
    cached_lineitem = Cache(lineitem.schema, 0, 2, lineitem)
    selected_orders = _filtered_groupby_domain(cached_lineitem, "l_orderkey")
    orders = _scan("orders", ("o_orderkey", "o_custkey"))

    filtered_orders = _join(
        orders, selected_orders, ("o_orderkey",), ("l_orderkey",), how="Semi"
    )
    root = _join(filtered_orders, cached_lineitem, ("o_orderkey",), ("l_orderkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(lineitem=(lineitem, 1_800), orders=(orders, 450)),
        _config(),
    )

    assert isinstance(optimized, Join)
    assert optimized.options[0] == "Inner"
    assert optimized.children[0] is filtered_orders
    assert isinstance(optimized.children[1], Join)
    assert optimized.children[1].options[0] == "Semi"
    assert optimized.children[1].children[0] is cached_lineitem
    assert _contains_node(optimized.children[1].children[1], selected_orders)


def test_reused_semi_domain_follows_filtered_key_rename() -> None:
    lineitem = _scan("lineitem", ("l_orderkey", "l_quantity"))
    cached_lineitem = Cache(lineitem.schema, 0, 2, lineitem)
    selected_orders = _filtered_groupby_domain(cached_lineitem, "l_orderkey")
    orders = _scan("orders", ("o_orderkey", "o_custkey"))

    filtered_orders = _join(
        orders, selected_orders, ("o_orderkey",), ("l_orderkey",), how="Semi"
    )
    renamed_orders = _select(
        filtered_orders,
        joined_orderkey="o_orderkey",
        o_custkey="o_custkey",
    )
    root = _join(
        renamed_orders,
        cached_lineitem,
        ("joined_orderkey",),
        ("l_orderkey",),
    )

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(lineitem=(lineitem, 1_800), orders=(orders, 450)),
        _config(),
    )

    assert isinstance(optimized, Join)
    assert isinstance(optimized.children[1], Join)
    assert optimized.children[1].options[0] == "Semi"
    assert optimized.children[1].children[0] is cached_lineitem
    assert _contains_node(optimized.children[1].children[1], selected_orders)


def test_reused_semi_domain_does_not_duplicate_uncached_source() -> None:
    lineitem = _scan("lineitem", ("l_orderkey", "l_quantity"))
    selected_orders = _filtered_groupby_domain(lineitem, "l_orderkey")
    orders = _scan("orders", ("o_orderkey", "o_custkey"))

    filtered_orders = _join(
        orders, selected_orders, ("o_orderkey",), ("l_orderkey",), how="Semi"
    )
    root = _join(filtered_orders, lineitem, ("o_orderkey",), ("l_orderkey",))

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(lineitem=(lineitem, 1_800), orders=(orders, 450)),
        _config(),
    )

    lineitem_semis = [
        semi for semi in _joins(optimized, "Semi") if semi.children[0] is lineitem
    ]
    assert not lineitem_semis


@pytest.mark.parametrize(
    "sum_fill, expect_reuse",
    [(None, True), (0.0, True), (1.0, False)],
    ids=["direct", "zero_fill", "nonzero_fill"],
)
def test_reuses_existing_aggregate_domain_to_replace_detail_join(
    sum_fill: float | None,
    expect_reuse: bool,  # noqa: FBT001
) -> None:
    lineitem = _typed_scan("lineitem", {"l_orderkey": I64, "l_quantity": F64})
    aggregate_cache = Cache(lineitem.schema, 0, 2, lineitem)
    detail_cache = Cache(lineitem.schema, 0, 2, lineitem)
    selected_sums, _ = _filtered_sum_domain(
        aggregate_cache, "l_orderkey", "l_quantity", "sum_quantity"
    )
    if sum_fill is not None:
        grouped = selected_sums.children[0]
        normalized_sums = Select(
            grouped.schema,
            (
                _key(grouped, "l_orderkey"),
                expr.NamedExpr(
                    "sum_quantity",
                    expr.UnaryFunction(
                        F64,
                        "fill_null",
                        (),
                        expr.Col(F64, "sum_quantity"),
                        expr.Literal(F64, sum_fill),
                    ),
                ),
            ),
            True,  # noqa: FBT003
            grouped,
        )
        selected_sums = Filter(
            normalized_sums.schema,
            selected_sums.mask,
            normalized_sums,
        )
    renamed_sums = _select(
        selected_sums,
        l_orderkey="l_orderkey",
        sum_for_reuse="sum_quantity",
    )
    selected_keys = _select(renamed_sums, l_orderkey="l_orderkey")
    orders = _typed_scan(
        "orders",
        {
            "o_orderkey": I64,
            "o_custkey": I64,
            "o_totalprice": F64,
        },
    )
    customer = _typed_scan("customer", {"c_custkey": I64, "c_name": I64})

    filtered_orders = _join(
        orders, selected_keys, ("o_orderkey",), ("l_orderkey",), how="Semi"
    )
    filtered_lineitem = _join(
        detail_cache,
        selected_keys,
        ("l_orderkey",),
        ("l_orderkey",),
        how="Semi",
    )
    order_lineitem = _join(
        filtered_orders, filtered_lineitem, ("o_orderkey",), ("l_orderkey",)
    )
    joined = _join(order_lineitem, customer, ("o_custkey",), ("c_custkey",))
    projected = Select(
        {
            "c_name": I64,
            "o_custkey": I64,
            "o_orderkey": I64,
            "o_totalprice": F64,
            "l_quantity": F64,
        },
        tuple(
            _key(joined, name)
            for name in (
                "c_name",
                "o_custkey",
                "o_orderkey",
                "o_totalprice",
                "l_quantity",
            )
        ),
        True,  # noqa: FBT003
        joined,
    )
    root = GroupBy(
        {
            "c_name": I64,
            "o_custkey": I64,
            "o_orderkey": I64,
            "o_totalprice": F64,
            "sum(l_quantity)": F64,
        },
        tuple(
            _key(projected, name)
            for name in (
                "c_name",
                "o_custkey",
                "o_orderkey",
                "o_totalprice",
            )
        ),
        (
            expr.NamedExpr(
                "sum(l_quantity)",
                expr.Agg(
                    F64,
                    "sum",
                    None,
                    ExecutionContext.GROUPBY,
                    expr.Col(F64, "l_quantity"),
                ),
            ),
        ),
        False,  # noqa: FBT003
        None,
        projected,
    )

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            lineitem=(lineitem, 1_800), orders=(orders, 450), customer=(customer, 150)
        ),
        _config(),
    )

    detail_semis = [
        semi for semi in _joins(optimized, "Semi") if semi.children[0] is detail_cache
    ]
    if not expect_reuse:
        assert detail_semis
        return
    assert not detail_semis

    replacement_details = [
        join.children[1]
        for join in _joins(optimized, "Inner")
        if _join_key_names(join.left_on) == ("o_orderkey",)
        and _join_key_names(join.right_on) == ("l_orderkey",)
    ]
    assert len(replacement_details) == 1
    replacement_detail = replacement_details[0]
    assert isinstance(replacement_detail, Filter)
    aggregate_values = replacement_detail.children[0]
    assert isinstance(aggregate_values, Select)
    assert aggregate_values.schema == {"l_orderkey": I64, "l_quantity": F64}
    assert _contains_node(aggregate_values, renamed_sums)


@pytest.mark.parametrize(
    "nulls_equal", [False, True], ids=["nulls_not_equal", "nulls_equal"]
)
def test_aggregate_domain_reuse_preserves_results(
    nulls_equal: bool,  # noqa: FBT001
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    lineitem = pl.LazyFrame(
        {
            "l_orderkey": [None, None, 1, 1, 2, 3],
            "l_quantity": [4, 5, 2, 3, 7, 1],
        }
    )
    selected_keys = (
        lineitem.group_by("l_orderkey")
        .agg(pl.col("l_quantity").sum().alias("sum_quantity"))
        .filter(pl.col("sum_quantity") > 4)
        .select("l_orderkey")
    )
    orders = pl.LazyFrame({"o_orderkey": [None, 1, 2, 3]})
    query = (
        orders.join(
            selected_keys,
            left_on="o_orderkey",
            right_on="l_orderkey",
            nulls_equal=nulls_equal,
        )
        .join(
            lineitem,
            left_on="o_orderkey",
            right_on="l_orderkey",
            nulls_equal=nulls_equal,
        )
        .group_by("o_orderkey")
        .agg(pl.col("l_quantity").sum())
    )
    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"join_domain_prefilter": {"threshold": 0.5}},
    )

    ir = Translator(query._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    optimized = optimize_join_domain_prefilters(
        ir,
        collect_statistics(ir, config, parquet_stats_executor),
        config,
    )

    detail_semis = [
        join
        for join in _joins(optimized, "Semi")
        if isinstance(join.children[0], Cache)
        and "l_quantity" in join.children[0].schema
    ]
    assert not detail_semis
    try:
        assert_gpu_result_equal(query, engine=engine, check_row_order=False)
    finally:
        DefaultSingletonEngine.shutdown()


def test_aggregate_domain_reuse_skips_filter_between_join_and_groupby() -> None:
    lineitem = _typed_scan("lineitem", {"l_orderkey": I64, "l_quantity": F64})
    selected_sums, selected_keys = _filtered_sum_domain(
        lineitem, "l_orderkey", "l_quantity", "sum_quantity"
    )
    orders = _typed_scan("orders", {"o_orderkey": I64})
    filtered_lineitem = _join(
        lineitem,
        selected_keys,
        ("l_orderkey",),
        ("l_orderkey",),
        how="Semi",
    )
    joined = _join(orders, filtered_lineitem, ("o_orderkey",), ("l_orderkey",))
    filtered_join = Filter(
        joined.schema,
        expr.NamedExpr("__predicate", expr.Literal(BOOL, True)),  # noqa: FBT003
        joined,
    )
    root = GroupBy(
        {"o_orderkey": I64, "sum(l_quantity)": F64},
        (_key(filtered_join, "o_orderkey"),),
        (
            expr.NamedExpr(
                "sum(l_quantity)",
                expr.Agg(
                    F64,
                    "sum",
                    None,
                    ExecutionContext.GROUPBY,
                    expr.Col(F64, "l_quantity"),
                ),
            ),
        ),
        False,  # noqa: FBT003
        None,
        filtered_join,
    )

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(lineitem=(lineitem, 1_800), orders=(orders, 450)),
        _config(),
    )

    assert _contains_node(optimized, filtered_lineitem)
    assert _contains_node(optimized, selected_sums)


def test_aggregate_domain_reuse_requires_same_detail_source() -> None:
    lineitem = _typed_scan("lineitem", {"l_orderkey": I64, "l_quantity": F64})
    other_lineitem = _typed_scan(
        "other_lineitem", {"l_orderkey": I64, "l_quantity": F64}
    )
    _, selected_keys = _filtered_sum_domain(
        other_lineitem, "l_orderkey", "l_quantity", "sum_quantity"
    )
    orders = _typed_scan("orders", {"o_orderkey": I64})
    filtered_lineitem = _join(
        lineitem,
        selected_keys,
        ("l_orderkey",),
        ("l_orderkey",),
        how="Semi",
    )
    joined = _join(orders, filtered_lineitem, ("o_orderkey",), ("l_orderkey",))
    root = GroupBy(
        {"o_orderkey": I64, "sum(l_quantity)": F64},
        (_key(joined, "o_orderkey"),),
        (
            expr.NamedExpr(
                "sum(l_quantity)",
                expr.Agg(
                    F64,
                    "sum",
                    None,
                    ExecutionContext.GROUPBY,
                    expr.Col(F64, "l_quantity"),
                ),
            ),
        ),
        False,  # noqa: FBT003
        None,
        joined,
    )

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(
            lineitem=(lineitem, 1_800),
            other_lineitem=(other_lineitem, 1_800),
            orders=(orders, 450),
        ),
        _config(),
    )

    assert _contains_node(optimized, filtered_lineitem)


def test_reused_semi_domain_does_not_duplicate_wrapped_uncached_source() -> None:
    lineitem = _scan("lineitem", ("l_orderkey", "l_quantity"))
    projected_lineitem = _select(
        lineitem,
        l_orderkey="l_orderkey",
        l_quantity="l_quantity",
    )
    selected_orders = _filtered_groupby_domain(projected_lineitem, "l_orderkey")
    orders = _scan("orders", ("o_orderkey", "o_custkey"))

    filtered_orders = _join(
        orders, selected_orders, ("o_orderkey",), ("l_orderkey",), how="Semi"
    )
    root = _join(
        filtered_orders,
        projected_lineitem,
        ("o_orderkey",),
        ("l_orderkey",),
    )

    optimized = optimize_join_domain_prefilters(
        root,
        _stats(lineitem=(lineitem, 1_800), orders=(orders, 450)),
        _config(),
    )

    lineitem_semis = [
        semi
        for semi in _joins(optimized, "Semi")
        if semi.children[0] is projected_lineitem
    ]
    assert not lineitem_semis


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

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import Join, Scan
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.join_domain_prefilter import (
    optimize_join_domain_prefilters,
)
from cudf_polars.utils.config import ConfigOptions, ParquetOptions

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import SerializedDataSourceInfo

I64 = DataType(pl.Int64())
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


def _key(node: IR, name: str) -> expr.NamedExpr:
    return expr.NamedExpr(name, expr.Col(node.schema[name], name))


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


def _config() -> ConfigOptions:
    return ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={
                "dynamic_planning": {
                    "join_domain_prefilter_enabled": True,
                    "join_domain_prefilter_trace": False,
                }
            },
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

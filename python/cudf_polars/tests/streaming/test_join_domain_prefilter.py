# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import Join, Scan, Select
from cudf_polars.dsl.traversal import traversal
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
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


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

    producer = _smallest_node_containing_all(
        renamed,
        ("domain_key", "domain_constraint"),
        {renamed: 20, source: 10},
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

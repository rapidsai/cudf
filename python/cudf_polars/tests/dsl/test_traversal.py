# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import singledispatch

import pylibcudf as plc

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars import translate_ir
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    make_recursive,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.typing import ExprTransformer, IRTransformer


def make_expr(dt, n1, n2):
    a1 = expr.Col(dt, n1)
    a2 = expr.Col(dt, n2)

    return expr.BinOp(dt, plc.binaryop.BinaryOperator.MUL, a1, a2)


def test_traversal_unique():
    dt = plc.DataType(plc.TypeId.INT8)

    e1 = make_expr(dt, "a", "a")
    unique_exprs = list(traversal(e1))

    assert len(unique_exprs) == 2
    assert set(unique_exprs) == {expr.Col(dt, "a"), e1}
    assert unique_exprs == [e1, expr.Col(dt, "a")]

    e2 = make_expr(dt, "a", "b")
    unique_exprs = list(traversal(e2))

    assert len(unique_exprs) == 3
    assert set(unique_exprs) == {expr.Col(dt, "a"), expr.Col(dt, "b"), e2}
    assert unique_exprs == [e2, expr.Col(dt, "a"), expr.Col(dt, "b")]

    e3 = make_expr(dt, "b", "a")
    unique_exprs = list(traversal(e3))

    assert len(unique_exprs) == 3
    assert set(unique_exprs) == {expr.Col(dt, "a"), expr.Col(dt, "b"), e3}
    assert unique_exprs == [e3, expr.Col(dt, "b"), expr.Col(dt, "a")]


def rename(e, rec):
    mapping = rec.state["mapping"]
    if isinstance(e, expr.Col) and e.name in mapping:
        return type(e)(e.dtype, mapping[e.name])
    return reuse_if_unchanged(e, rec)


def test_caching_visitor():
    dt = plc.DataType(plc.TypeId.INT8)

    e1 = make_expr(dt, "a", "b")

    mapper = CachingVisitor(rename, state={"mapping": {"b": "c"}})

    renamed = mapper(e1)
    assert renamed == make_expr(dt, "a", "c")
    assert len(mapper.cache) == 3

    e2 = make_expr(dt, "a", "a")
    mapper = CachingVisitor(rename, state={"mapping": {"b": "c"}})

    renamed = mapper(e2)
    assert renamed == make_expr(dt, "a", "a")
    assert len(mapper.cache) == 2
    mapper = CachingVisitor(rename, state={"mapping": {"a": "c"}})

    renamed = mapper(e2)
    assert renamed == make_expr(dt, "c", "c")
    assert len(mapper.cache) == 2


def test_noop_visitor():
    dt = plc.DataType(plc.TypeId.INT8)

    e1 = make_expr(dt, "a", "b")

    mapper = make_recursive(rename, state={"mapping": {"b": "c"}})

    renamed = mapper(e1)
    assert renamed == make_expr(dt, "a", "c")

    e2 = make_expr(dt, "a", "a")
    mapper = make_recursive(rename, state={"mapping": {"b": "c"}})

    renamed = mapper(e2)
    assert renamed == make_expr(dt, "a", "a")
    mapper = make_recursive(rename, state={"mapping": {"a": "c"}})

    renamed = mapper(e2)
    assert renamed == make_expr(dt, "c", "c")


def test_rewrite_ir_node():
    df = pl.LazyFrame({"a": [1, 2, 1], "b": [1, 3, 4]})
    q = df.group_by("a").agg(pl.col("b").sum()).sort("b")

    orig = translate_ir(q._ldf.visit())

    new_df = pl.DataFrame({"a": [1, 1, 2], "b": [-1, -2, -4]})

    def replace_df(node, rec):
        if isinstance(node, ir.DataFrameScan):
            return ir.DataFrameScan(
                node.schema, new_df._df, node.projection, node.predicate
            )
        return reuse_if_unchanged(node, rec)

    mapper = CachingVisitor(replace_df)

    new = mapper(orig)

    result = new.evaluate(cache={}).to_polars()

    expect = pl.DataFrame({"a": [2, 1], "b": [-4, -3]})

    assert_frame_equal(result, expect)


def test_rewrite_scan_node(tmp_path):
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 3, 4]})
    right = pl.DataFrame({"a": [1, 4, 2], "c": [1, 2, 3]})

    right.write_parquet(tmp_path / "right.pq")

    right_s = pl.scan_parquet(tmp_path / "right.pq")

    q = left.join(right_s, on="a", how="inner")

    def replace_scan(node, rec):
        if isinstance(node, ir.Scan):
            return ir.DataFrameScan(
                node.schema, right._df, node.with_columns, node.predicate
            )
        return reuse_if_unchanged(node, rec)

    mapper = CachingVisitor(replace_scan)

    orig = translate_ir(q._ldf.visit())
    new = mapper(orig)

    result = new.evaluate(cache={}).to_polars()

    expect = q.collect()

    assert_frame_equal(result, expect, check_row_order=False)


def test_rewrite_names_and_ops():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})

    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    # We will replace a -> d, c -> d, and addition with multiplication
    expect = (
        df.select(
            (pl.col("d") - (pl.col("b") * pl.col("d") * 2)).alias("a"), pl.col("d")
        )
        .sort("d")
        .collect()
    )

    qir = translate_ir(q._ldf.visit())

    @singledispatch
    def _transform(e: expr.Expr, fn: ExprTransformer) -> expr.Expr:
        raise NotImplementedError("Unhandled")

    @_transform.register
    def _(e: expr.Col, fn: ExprTransformer):
        mapping = fn.state["mapping"]
        if e.name in mapping:
            return type(e)(e.dtype, mapping[e.name])
        return e

    @_transform.register
    def _(e: expr.BinOp, fn: ExprTransformer):
        if e.op == plc.binaryop.BinaryOperator.ADD:
            return type(e)(
                e.dtype, plc.binaryop.BinaryOperator.MUL, *map(fn, e.children)
            )
        return reuse_if_unchanged(e, fn)

    _transform.register(expr.Expr)(reuse_if_unchanged)

    @singledispatch
    def _rewrite(node: ir.IR, fn: IRTransformer) -> ir.IR:
        raise NotImplementedError("Unhandled")

    @_rewrite.register
    def _(node: ir.Select, fn: IRTransformer):
        expr_mapper = fn.state["expr_mapper"]
        return type(node)(
            node.schema,
            [expr.NamedExpr(e.name, expr_mapper(e.value)) for e in node.exprs],
            node.should_broadcast,
            fn(node.children[0]),
        )

    _rewrite.register(ir.IR)(reuse_if_unchanged)

    rewriter = CachingVisitor(
        _rewrite,
        state={
            "expr_mapper": CachingVisitor(
                _transform, state={"mapping": {"a": "d", "c": "d"}}
            )
        },
    )

    new_ir = rewriter(qir)

    got = new_ir.evaluate(cache={}).to_polars()

    assert_frame_equal(expect, got)

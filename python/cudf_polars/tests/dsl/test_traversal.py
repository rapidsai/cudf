# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf_polars.dsl import expr
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    make_recursive,
    reuse_if_unchanged,
    traversal,
)


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

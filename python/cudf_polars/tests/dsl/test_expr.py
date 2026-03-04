# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr


def test_expression_equality_not_expression():
    col = expr.Col(DataType(pl.datatypes.Int8()), "a")
    assert not (col == "a")  # noqa: SIM201
    assert col != "a"


@pytest.mark.parametrize("dtype", [pl.datatypes.Int8(), pl.datatypes.Int16()])
def test_column_ne_dtypes_differ(dtype):
    a = expr.Col(DataType(dtype), "a")
    b = expr.Col(DataType(pl.datatypes.Float32()), "a")
    assert a != b


@pytest.mark.parametrize("dtype", [pl.datatypes.Int8(), pl.datatypes.Int16()])
def test_column_ne_names_differ(dtype):
    a = expr.Col(DataType(dtype), "a")
    b = expr.Col(DataType(dtype), "b")
    assert a != b


@pytest.mark.parametrize("dtype", [pl.datatypes.Int8(), pl.datatypes.Int16()])
def test_column_eq_names_eq(dtype):
    a = expr.Col(DataType(dtype), "a")
    b = expr.Col(DataType(dtype), "a")
    assert a == b


def test_expr_hashable():
    a = expr.Col(DataType(pl.datatypes.Int8()), "a")
    b = expr.Col(DataType(pl.datatypes.Int8()), "b")
    c = expr.Col(DataType(pl.datatypes.Float32()), "c")

    collection = {a, b, c}
    assert len(collection) == 3
    assert a in collection
    assert b in collection
    assert c in collection


def test_namedexpr_hashable():
    b = expr.NamedExpr("b", expr.Col(DataType(pl.datatypes.Int8()), "a"))
    c = expr.NamedExpr("c", expr.Col(DataType(pl.datatypes.Int8()), "a"))

    collection = {b, c}

    assert len(collection) == 2

    assert b in collection
    assert c in collection


def test_namedexpr_ne_values():
    b1 = expr.NamedExpr("b1", expr.Col(DataType(pl.datatypes.Int8()), "a"))
    b2 = expr.NamedExpr("b2", expr.Col(DataType(pl.datatypes.Int16()), "a"))

    assert b1 != b2


def test_namedexpr_repr_stable():
    b1 = expr.NamedExpr("b1", expr.Col(DataType(pl.datatypes.Int8()), "a"))
    b2 = expr.NamedExpr("b1", expr.Col(DataType(pl.datatypes.Int8()), "a"))

    assert repr(b1) == repr(b2)


def test_equality_cse():
    dt = DataType(pl.datatypes.Int8())

    def make_expr(n1, n2):
        a = expr.Col(DataType(pl.datatypes.Int8()), n1)
        b = expr.Col(DataType(pl.datatypes.Int8()), n2)

        return expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)

    e1 = make_expr("a", "b")
    e2 = make_expr("a", "b")
    e3 = make_expr("a", "c")

    assert e1.children is not e2.children
    assert e1 == e2
    assert e1.children is e2.children
    assert e1 == e2
    assert e1 != e3
    assert e2 != e3


def test_reconstruct_named_expr():
    ne1 = expr.NamedExpr("a", expr.Col(DataType(pl.datatypes.Int8()), "a"))
    new_value = expr.Col(DataType(pl.datatypes.Int16()), "a")
    ne2 = ne1.reconstruct(new_value)
    assert ne1.name == ne2.name
    assert ne1 != ne2
    assert ne2.value == new_value

    ne3 = ne2.reconstruct(ne2.value)
    assert ne3 is ne2

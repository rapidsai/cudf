# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc
import pytest

from cudf_polars.dsl import expr


def test_expression_equality_not_expression():
    col = expr.Col(plc.DataType(plc.TypeId.INT8), "a")
    assert not (col == "a")  # noqa: SIM201
    assert col != "a"


@pytest.mark.parametrize("dtype", [plc.TypeId.INT8, plc.TypeId.INT16])
def test_column_ne_dtypes_differ(dtype):
    a = expr.Col(plc.DataType(dtype), "a")
    b = expr.Col(plc.DataType(plc.TypeId.FLOAT32), "a")
    assert a != b


@pytest.mark.parametrize("dtype", [plc.TypeId.INT8, plc.TypeId.INT16])
def test_column_ne_names_differ(dtype):
    a = expr.Col(plc.DataType(dtype), "a")
    b = expr.Col(plc.DataType(dtype), "b")
    assert a != b


@pytest.mark.parametrize("dtype", [plc.TypeId.INT8, plc.TypeId.INT16])
def test_column_eq_names_eq(dtype):
    a = expr.Col(plc.DataType(dtype), "a")
    b = expr.Col(plc.DataType(dtype), "a")
    assert a == b


def test_expr_hashable():
    a = expr.Col(plc.DataType(plc.TypeId.INT8), "a")
    b = expr.Col(plc.DataType(plc.TypeId.INT8), "b")
    c = expr.Col(plc.DataType(plc.TypeId.FLOAT32), "c")

    collection = {a, b, c}
    assert len(collection) == 3
    assert a in collection
    assert b in collection
    assert c in collection


def test_namedexpr_hashable():
    b = expr.NamedExpr("b", expr.Col(plc.DataType(plc.TypeId.INT8), "a"))
    c = expr.NamedExpr("c", expr.Col(plc.DataType(plc.TypeId.INT8), "a"))

    collection = {b, c}

    assert len(collection) == 2

    assert b in collection
    assert c in collection


def test_namedexpr_ne_values():
    b1 = expr.NamedExpr("b1", expr.Col(plc.DataType(plc.TypeId.INT8), "a"))
    b2 = expr.NamedExpr("b2", expr.Col(plc.DataType(plc.TypeId.INT16), "a"))

    assert b1 != b2


@pytest.mark.xfail(reason="pylibcudf datatype repr not stable")
def test_namedexpr_repr_stable():
    b1 = expr.NamedExpr("b1", expr.Col(plc.DataType(plc.TypeId.INT8), "a"))
    b2 = expr.NamedExpr("b1", expr.Col(plc.DataType(plc.TypeId.INT8), "a"))

    assert repr(b1) == repr(b2)

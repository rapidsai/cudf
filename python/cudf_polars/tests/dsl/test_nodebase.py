# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext

if TYPE_CHECKING:
    from cudf_polars.dsl.nodebase import Node


@pytest.mark.parametrize(
    "node",
    [
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 42),
        expr.Cast(
            DataType(pl.Float64()),
            expr.Col(DataType(pl.Int64), "foo"),
        ),
        expr.BinOp(
            DataType(pl.Int64()),
            plc.binaryop.BinaryOperator.ADD,
            expr.Col(DataType(pl.Int64()), "foo"),
            expr.Literal(DataType(pl.Int64()), 1),
        ),
        expr.GroupedRollingWindow(
            DataType(pl.Float64),
            ("groups_to_rows", True, False, False),
            [
                expr.NamedExpr(
                    "foo",
                    expr.Agg(
                        DataType(pl.Float64),
                        "sum",
                        None,
                        ExecutionContext.WINDOW,
                        expr.Col(DataType(pl.Int64()), "bar"),
                    ),
                )
            ],
            expr.NamedExpr("foo", expr.Col(DataType(pl.Float64()), "foo")),
            1,
            expr.Col(DataType(pl.Int64()), "foo"),
            expr.Col(DataType(pl.Int64()), "bar"),
            expr.Col(DataType(pl.Int64()), "baz"),
        ),
    ],
)
def test_reconstruct(node: Node):
    assert node.reconstruct(node.children) == node

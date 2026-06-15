# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl
from polars import polars as plrs  # type: ignore[attr-defined]

from cudf_polars import Translator
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.traversal import traversal
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("expr", [pl.col("c"), pl.col("b") < 1, pl.lit(value=True)])
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_filter(engine: pl.GPUEngine, expr, predicate_pushdown):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [True, False, False, True, True, True, None],
        }
    ).lazy()

    query = ldf.filter(expr)
    assert_gpu_result_equal(
        query,
        engine=engine,
        collect_kwargs={
            "optimizations": pl.QueryOptFlags(predicate_pushdown=predicate_pushdown)
        },
    )


def test_filter_drops_dynamic_predicate_hint():
    class DynamicPredHint(Exception):
        pass

    class DynamicPredVisitor:
        def __init__(self, inner, op):
            self.inner = inner
            self.op = op

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def view_expression(self, n):
            node = self.inner.view_expression(n)
            if isinstance(node, plrs._expr_nodes.BinaryExpr) and node.op == self.op:
                raise DynamicPredHint("dynamic_pred")
            return node

    ldf = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [1, 1, 3, 3, 5]}
    )
    query = ldf.filter(
        ((pl.col("b") < 5) & (pl.col("a") > 1))
        & ((pl.col("c") == 3) & (pl.col("b") < 3))
    )

    visitor = DynamicPredVisitor(query._ldf.visit(), plrs._expr_nodes.Operator.Lt)
    translator = Translator(visitor, pl.GPUEngine())
    node = translator.translate_ir()

    assert translator.errors == []
    assert isinstance(node, ir.Filter)
    columns = {n.name for n in traversal([node.mask.value]) if isinstance(n, expr.Col)}
    assert columns == {"a", "c"}

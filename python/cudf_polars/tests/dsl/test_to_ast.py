# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

import cudf_polars.dsl.expr as expr_nodes
import cudf_polars.dsl.ir as ir_nodes
from cudf_polars import Translator
from cudf_polars.containers.dataframe import DataFrame, NamedColumn
from cudf_polars.dsl.to_ast import insert_colrefs, to_ast, to_parquet_filter


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "c": ["a", "b", "c", "d", "e", "f"],
            "a": [1, 2, 3, None, 4, 5],
            "b": pl.Series([None, None, 3, float("inf"), 4, 0], dtype=pl.Float64),
            "d": [False, True, True, None, False, False],
        }
    )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_in([0, 1]),
        pl.col("a").is_between(0, 2),
        (pl.col("a") < pl.col("b")).not_(),
        pl.lit(2) > pl.col("a"),
        pl.lit(2) >= pl.col("a"),
        pl.lit(2) < pl.col("a"),
        pl.lit(2) <= pl.col("a"),
        pl.lit(0) == pl.col("a"),
        pl.lit(1) != pl.col("a"),
        (pl.col("b") < pl.lit(2, dtype=pl.Float64).sqrt()),
        (pl.col("a") >= pl.lit(2)) & (pl.col("b") > 0),
        pl.col("a").is_null(),
        pl.col("a").is_not_null(),
        pl.col("b").is_finite(),
        pytest.param(
            pl.col("a").sin(),
            marks=pytest.mark.xfail(reason="Need to insert explicit casts"),
        ),
        pl.col("b").cos(),
        pl.col("a").abs().is_between(0, 2),
        pl.col("a").ne_missing(pl.lit(None, dtype=pl.Int64)),
        [pl.col("a") * 2, pl.col("b") + pl.col("a")],
        pl.col("d").not_(),
    ],
)
def test_compute_column(expr, df):
    q = df.select(expr)
    ir = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()

    assert isinstance(ir, ir_nodes.Select)
    table = ir.children[0].evaluate(cache={})
    name_to_index = {c.name: i for i, c in enumerate(table.columns)}

    def compute_column(e):
        e_with_colrefs = insert_colrefs(
            e.value,
            table_ref=plc.expressions.TableReference.LEFT,
            name_to_index=name_to_index,
        )
        with pytest.raises(NotImplementedError):
            e_with_colrefs.evaluate(table)
        ast = to_ast(e_with_colrefs)
        if ast is not None:
            return NamedColumn(
                plc.transform.compute_column(table.table, ast), name=e.name
            )
        return e.evaluate(table)

    got = DataFrame(map(compute_column, ir.exprs)).to_polars()

    expect = q.collect()

    assert_frame_equal(expect, got)


def test_invalid_colref_construction_raises():
    literal = expr_nodes.Literal(
        plc.DataType(plc.TypeId.INT8), pa.scalar(1, type=pa.int8())
    )
    with pytest.raises(TypeError):
        expr_nodes.ColRef(
            literal.dtype, 0, plc.expressions.TableReference.LEFT, literal
        )


def test_to_ast_without_colref_raises():
    col = expr_nodes.Col(plc.DataType(plc.TypeId.INT8), "a")

    with pytest.raises(TypeError):
        to_ast(col)


def test_to_parquet_filter_with_colref_raises():
    col = expr_nodes.Col(plc.DataType(plc.TypeId.INT8), "a")
    colref = expr_nodes.ColRef(col.dtype, 0, plc.expressions.TableReference.LEFT, col)

    with pytest.raises(TypeError):
        to_parquet_filter(colref)

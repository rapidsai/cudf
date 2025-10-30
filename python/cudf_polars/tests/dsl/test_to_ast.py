# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

import cudf_polars.dsl.expr as expr_nodes
import cudf_polars.dsl.ir as ir_nodes
from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.containers.dataframe import DataFrame, NamedColumn
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.dsl.to_ast import insert_colrefs, to_ast, to_parquet_filter
from cudf_polars.utils.cuda_stream import get_cuda_stream


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
    stream = get_cuda_stream()

    q = df.select(expr)
    ir = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()

    assert isinstance(ir, ir_nodes.Select)
    table = ir.children[0].evaluate(
        cache={}, timer=None, context=IRExecutionContext(get_cuda_stream=lambda: stream)
    )
    name_to_index = {c.name: i for i, c in enumerate(table.columns)}

    def compute_column(e):
        e_with_colrefs = insert_colrefs(
            e.value,
            table_ref=plc.expressions.TableReference.LEFT,
            name_to_index=name_to_index,
        )
        with pytest.raises(NotImplementedError):
            e_with_colrefs.evaluate(table)
        try:
            ast = to_ast(e_with_colrefs, stream=stream)
        except (KeyError, NotImplementedError):
            return e.evaluate(table)
        else:
            return NamedColumn(
                plc.transform.compute_column(table.table, ast, stream=stream),
                name=e.name,
                dtype=e.value.dtype,
            )

    got = DataFrame(map(compute_column, ir.exprs), stream=stream).to_polars()

    expect = q.collect()

    assert_frame_equal(expect, got)


def test_invalid_colref_construction_raises():
    literal = expr_nodes.Literal(DataType(pl.datatypes.Int8()), 1)
    with pytest.raises(TypeError):
        expr_nodes.ColRef(
            literal.dtype, 0, plc.expressions.TableReference.LEFT, literal
        )


def test_to_ast_without_colref_raises():
    stream = get_cuda_stream()
    col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")

    with pytest.raises(TypeError, match="Should always be wrapped"):
        to_ast(col, stream=stream)


def test_to_parquet_filter_with_colref_raises():
    col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")
    colref = expr_nodes.ColRef(col.dtype, 0, plc.expressions.TableReference.LEFT, col)

    with pytest.raises(TypeError):
        to_parquet_filter(colref, stream=get_cuda_stream())


def test_validate_to_ast_null_not_equals():
    from cudf_polars.dsl.to_ast import validate_to_ast

    operand = expr_nodes.Literal(DataType(pl.datatypes.Boolean()), True)

    binop = expr_nodes.BinOp(
        DataType(pl.datatypes.Boolean()),
        plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
        operand,
        operand,
    )

    validate_to_ast(binop)


def test_validate_to_ast_boolean_function_isin():
    from cudf_polars.dsl.to_ast import validate_to_ast

    col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")
    haystack = expr_nodes.LiteralColumn(
        DataType(pl.datatypes.List(pl.datatypes.Int8())),
        [1, 2, 3, 4, 5],
    )

    isin_func = expr_nodes.BooleanFunction(
        DataType(pl.datatypes.Boolean()),
        expr_nodes.BooleanFunction.Name.IsIn,
        col,
        haystack,
    )

    # Should validate successfully for small haystack
    with pytest.raises(TypeError, match="Should always be wrapped"):
        # Will raise because Col is not wrapped, but validates the IsIn logic first
        validate_to_ast(isin_func)


# def test_validate_to_ast_boolean_function_parquet_col():
#     """Test parquet filter validation with BooleanFunction on column (lines 209-212)."""
#     from cudf_polars.dsl.to_ast import _validate_to_ast
#     from cudf_polars.dsl.traversal import CachingVisitor

#     col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")

#     is_null_func = expr_nodes.BooleanFunction(
#         DataType(pl.datatypes.Boolean()),
#         expr_nodes.BooleanFunction.Name.IsNull,
#         col,
#     )

#     validator = CachingVisitor(_validate_to_ast, state={"for_parquet": True})
#     with pytest.raises(NotImplementedError, match="Parquet filters don't support"):
#         validator(is_null_func)


# def test_validate_to_ast_boolean_function_isnull_isnotnull_not():
#     """Test validate_to_ast with IsNull, IsNotNull, Not (lines 213-219)."""
#     from cudf_polars.dsl.to_ast import validate_to_ast

#     lit = expr_nodes.Literal(DataType(pl.datatypes.Int8()), 5)

#     # Test IsNull
#     is_null_func = expr_nodes.BooleanFunction(
#         DataType(pl.datatypes.Boolean()),
#         expr_nodes.BooleanFunction.Name.IsNull,
#         lit,
#     )
#     validate_to_ast(is_null_func)  # Should not raise

#     # Test IsNotNull
#     is_not_null_func = expr_nodes.BooleanFunction(
#         DataType(pl.datatypes.Boolean()),
#         expr_nodes.BooleanFunction.Name.IsNotNull,
#         lit,
#     )
#     validate_to_ast(is_not_null_func)  # Should not raise

#     # Test Not
#     not_func = expr_nodes.BooleanFunction(
#         DataType(pl.datatypes.Boolean()),
#         expr_nodes.BooleanFunction.Name.Not,
#         lit,
#     )
#     validate_to_ast(not_func)  # Should not raise


# def test_validate_to_ast_boolean_function_unsupported():
#     """Test validate_to_ast with unsupported BooleanFunction (line 220)."""
#     from cudf_polars.dsl.to_ast import validate_to_ast

#     lit = expr_nodes.Literal(DataType(pl.datatypes.Int8()), 5)

#     # IsFinite is not supported in AST conversion
#     is_finite_func = expr_nodes.BooleanFunction(
#         DataType(pl.datatypes.Boolean()),
#         expr_nodes.BooleanFunction.Name.IsFinite,
#         lit,
#     )

#     with pytest.raises(NotImplementedError, match="AST conversion does not support"):
#         validate_to_ast(is_finite_func)


# def test_validate_to_ast_unary_function_parquet():
#     """Test parquet filter validation with UnaryFunction on column (lines 225-228)."""
#     from cudf_polars.dsl.to_ast import _validate_to_ast
#     from cudf_polars.dsl.traversal import CachingVisitor

#     col = expr_nodes.Col(DataType(pl.datatypes.Float64()), "a")

#     unary_func = expr_nodes.UnaryFunction(
#         DataType(pl.datatypes.Float64()),
#         "sin",
#         {},
#         col,
#     )

#     validator = CachingVisitor(_validate_to_ast, state={"for_parquet": True})
#     with pytest.raises(NotImplementedError, match="Parquet filters don't support"):
#         validator(unary_func)


# def test_validate_to_ast_unary_function():
#     """Test validate_to_ast with UnaryFunction (lines 229-230)."""
#     from cudf_polars.dsl.to_ast import validate_to_ast

#     lit = expr_nodes.Literal(DataType(pl.datatypes.Float64()), 5.0)

#     unary_func = expr_nodes.UnaryFunction(
#         DataType(pl.datatypes.Float64()),
#         "sin",
#         {},
#         lit,
#     )

#     validate_to_ast(unary_func)  # Should not raise

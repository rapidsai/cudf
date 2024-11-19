# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Conversion of expression nodes to libcudf AST nodes."""

from __future__ import annotations

from functools import partial, reduce, singledispatch
from typing import TYPE_CHECKING, TypeAlias

from polars.polars import _expr_nodes as pl_expr

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

from cudf_polars.dsl import expr
from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.typing import ExprTransformer

# Can't merge these op-mapping dictionaries because scoped enum values
# are exposed by cython with equality/hash based one their underlying
# representation type. So in a dict they are just treated as integers.
BINOP_TO_ASTOP = {
    plc.binaryop.BinaryOperator.EQUAL: plc_expr.ASTOperator.EQUAL,
    plc.binaryop.BinaryOperator.NULL_EQUALS: plc_expr.ASTOperator.NULL_EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL: plc_expr.ASTOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS: plc_expr.ASTOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL: plc_expr.ASTOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER: plc_expr.ASTOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: plc_expr.ASTOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.ADD: plc_expr.ASTOperator.ADD,
    plc.binaryop.BinaryOperator.SUB: plc_expr.ASTOperator.SUB,
    plc.binaryop.BinaryOperator.MUL: plc_expr.ASTOperator.MUL,
    plc.binaryop.BinaryOperator.DIV: plc_expr.ASTOperator.DIV,
    plc.binaryop.BinaryOperator.TRUE_DIV: plc_expr.ASTOperator.TRUE_DIV,
    plc.binaryop.BinaryOperator.FLOOR_DIV: plc_expr.ASTOperator.FLOOR_DIV,
    plc.binaryop.BinaryOperator.PYMOD: plc_expr.ASTOperator.PYMOD,
    plc.binaryop.BinaryOperator.BITWISE_AND: plc_expr.ASTOperator.BITWISE_AND,
    plc.binaryop.BinaryOperator.BITWISE_OR: plc_expr.ASTOperator.BITWISE_OR,
    plc.binaryop.BinaryOperator.BITWISE_XOR: plc_expr.ASTOperator.BITWISE_XOR,
    plc.binaryop.BinaryOperator.LOGICAL_AND: plc_expr.ASTOperator.LOGICAL_AND,
    plc.binaryop.BinaryOperator.LOGICAL_OR: plc_expr.ASTOperator.LOGICAL_OR,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_AND: plc_expr.ASTOperator.NULL_LOGICAL_AND,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_OR: plc_expr.ASTOperator.NULL_LOGICAL_OR,
}

UOP_TO_ASTOP = {
    plc.unary.UnaryOperator.SIN: plc_expr.ASTOperator.SIN,
    plc.unary.UnaryOperator.COS: plc_expr.ASTOperator.COS,
    plc.unary.UnaryOperator.TAN: plc_expr.ASTOperator.TAN,
    plc.unary.UnaryOperator.ARCSIN: plc_expr.ASTOperator.ARCSIN,
    plc.unary.UnaryOperator.ARCCOS: plc_expr.ASTOperator.ARCCOS,
    plc.unary.UnaryOperator.ARCTAN: plc_expr.ASTOperator.ARCTAN,
    plc.unary.UnaryOperator.SINH: plc_expr.ASTOperator.SINH,
    plc.unary.UnaryOperator.COSH: plc_expr.ASTOperator.COSH,
    plc.unary.UnaryOperator.TANH: plc_expr.ASTOperator.TANH,
    plc.unary.UnaryOperator.ARCSINH: plc_expr.ASTOperator.ARCSINH,
    plc.unary.UnaryOperator.ARCCOSH: plc_expr.ASTOperator.ARCCOSH,
    plc.unary.UnaryOperator.ARCTANH: plc_expr.ASTOperator.ARCTANH,
    plc.unary.UnaryOperator.EXP: plc_expr.ASTOperator.EXP,
    plc.unary.UnaryOperator.LOG: plc_expr.ASTOperator.LOG,
    plc.unary.UnaryOperator.SQRT: plc_expr.ASTOperator.SQRT,
    plc.unary.UnaryOperator.CBRT: plc_expr.ASTOperator.CBRT,
    plc.unary.UnaryOperator.CEIL: plc_expr.ASTOperator.CEIL,
    plc.unary.UnaryOperator.FLOOR: plc_expr.ASTOperator.FLOOR,
    plc.unary.UnaryOperator.ABS: plc_expr.ASTOperator.ABS,
    plc.unary.UnaryOperator.RINT: plc_expr.ASTOperator.RINT,
    plc.unary.UnaryOperator.BIT_INVERT: plc_expr.ASTOperator.BIT_INVERT,
    plc.unary.UnaryOperator.NOT: plc_expr.ASTOperator.NOT,
}

SUPPORTED_STATISTICS_BINOPS = {
    plc.binaryop.BinaryOperator.EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL,
}

REVERSED_COMPARISON = {
    plc.binaryop.BinaryOperator.EQUAL: plc.binaryop.BinaryOperator.EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL: plc.binaryop.BinaryOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS: plc.binaryop.BinaryOperator.GREATER,
    plc.binaryop.BinaryOperator.LESS_EQUAL: plc.binaryop.BinaryOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.GREATER: plc.binaryop.BinaryOperator.LESS,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: plc.binaryop.BinaryOperator.LESS_EQUAL,
}


Transformer: TypeAlias = GenericTransformer[expr.Expr, plc_expr.Expression]


@singledispatch
def _to_ast(node: expr.Expr, self: Transformer) -> plc_expr.Expression:
    """
    Translate an expression to a pylibcudf Expression.

    Parameters
    ----------
    node
        Expression to translate.
    self
        Recursive transformer. The state dictionary should contain a
       `for_parquet` key indicating if this transformation should
        provide an expression suitable for use in parquet filters.

        If `for_parquet` is `False`, the dictionary should contain a
        `name_to_index` mapping that maps column names to their
        integer index in the table that will be used for evaluation of
        the expression.

    Returns
    -------
    pylibcudf Expression.

    Raises
    ------
    NotImplementedError or KeyError if the expression cannot be translated.
    """
    raise NotImplementedError(f"Unhandled expression type {type(node)}")


@_to_ast.register
def _(node: expr.Col, self: Transformer) -> plc_expr.Expression:
    if self.state["for_parquet"]:
        return plc_expr.ColumnNameReference(node.name)
    raise TypeError("Should always be wrapped in a ColRef node before translation")


@_to_ast.register
def _(node: expr.ColRef, self: Transformer) -> plc_expr.Expression:
    if self.state["for_parquet"]:
        raise TypeError("Not expecting ColRef node in parquet filter")
    return plc_expr.ColumnReference(node.index, node.table_ref)


@_to_ast.register
def _(node: expr.Literal, self: Transformer) -> plc_expr.Expression:
    return plc_expr.Literal(plc.interop.from_arrow(node.value))


@_to_ast.register
def _(node: expr.BinOp, self: Transformer) -> plc_expr.Expression:
    if node.op == plc.binaryop.BinaryOperator.NULL_NOT_EQUALS:
        return plc_expr.Operation(
            plc_expr.ASTOperator.NOT,
            self(
                # Reconstruct and apply, rather than directly
                # constructing the right expression so we get the
                # handling of parquet special cases for free.
                expr.BinOp(
                    node.dtype, plc.binaryop.BinaryOperator.NULL_EQUALS, *node.children
                )
            ),
        )
    if self.state["for_parquet"]:
        op1_col, op2_col = (isinstance(op, expr.Col) for op in node.children)
        if op1_col ^ op2_col:
            op = node.op
            if op not in SUPPORTED_STATISTICS_BINOPS:
                raise NotImplementedError(
                    f"Parquet filter binop with column doesn't support {node.op!r}"
                )
            op1, op2 = node.children
            if op2_col:
                (op1, op2) = (op2, op1)
                op = REVERSED_COMPARISON[op]
            if not isinstance(op2, expr.Literal):
                raise NotImplementedError(
                    "Parquet filter binops must have form 'col binop literal'"
                )
            return plc_expr.Operation(BINOP_TO_ASTOP[op], self(op1), self(op2))
        elif op1_col and op2_col:
            raise NotImplementedError(
                "Parquet filter binops must have one column reference not two"
            )
    return plc_expr.Operation(BINOP_TO_ASTOP[node.op], *map(self, node.children))


@_to_ast.register
def _(node: expr.BooleanFunction, self: Transformer) -> plc_expr.Expression:
    if node.name == pl_expr.BooleanFunction.IsIn:
        needles, haystack = node.children
        if isinstance(haystack, expr.LiteralColumn) and len(haystack.value) < 16:
            # 16 is an arbitrary limit
            needle_ref = self(needles)
            values = [
                plc_expr.Literal(plc.interop.from_arrow(v)) for v in haystack.value
            ]
            return reduce(
                partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_OR),
                (
                    plc_expr.Operation(plc_expr.ASTOperator.EQUAL, needle_ref, value)
                    for value in values
                ),
            )
    if self.state["for_parquet"] and isinstance(node.children[0], expr.Col):
        raise NotImplementedError(
            f"Parquet filters don't support {node.name} on columns"
        )
    if node.name == pl_expr.BooleanFunction.IsNull:
        return plc_expr.Operation(plc_expr.ASTOperator.IS_NULL, self(node.children[0]))
    elif node.name == pl_expr.BooleanFunction.IsNotNull:
        return plc_expr.Operation(
            plc_expr.ASTOperator.NOT,
            plc_expr.Operation(plc_expr.ASTOperator.IS_NULL, self(node.children[0])),
        )
    elif node.name == pl_expr.BooleanFunction.Not:
        return plc_expr.Operation(plc_expr.ASTOperator.NOT, self(node.children[0]))
    raise NotImplementedError(f"AST conversion does not support {node.name}")


@_to_ast.register
def _(node: expr.UnaryFunction, self: Transformer) -> plc_expr.Expression:
    if isinstance(node.children[0], expr.Col) and self.state["for_parquet"]:
        raise NotImplementedError(
            "Parquet filters don't support {node.name} on columns"
        )
    return plc_expr.Operation(
        UOP_TO_ASTOP[node._OP_MAPPING[node.name]], self(node.children[0])
    )


def to_parquet_filter(node: expr.Expr) -> plc_expr.Expression | None:
    """
    Convert an expression to libcudf AST nodes suitable for parquet filtering.

    Parameters
    ----------
    node
        Expression to convert.

    Returns
    -------
    pylibcudf Expression if conversion is possible, otherwise None.
    """
    mapper = CachingVisitor(_to_ast, state={"for_parquet": True})
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None


def to_ast(node: expr.Expr) -> plc_expr.Expression | None:
    """
    Convert an expression to libcudf AST nodes suitable for compute_column.

    Parameters
    ----------
    node
        Expression to convert.

    Notes
    -----
    `Col` nodes must always be wrapped in `TableRef` nodes when
    converting to an ast expression so that their table reference and
    index are provided.

    Returns
    -------
    pylibcudf Expression if conversion is possible, otherwise None.
    """
    mapper = CachingVisitor(_to_ast, state={"for_parquet": False})
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None


def _insert_colrefs(node: expr.Expr, rec: ExprTransformer) -> expr.Expr:
    if isinstance(node, expr.Col):
        return expr.ColRef(
            node.dtype,
            rec.state["name_to_index"][node.name],
            rec.state["table_ref"],
            node,
        )
    return reuse_if_unchanged(node, rec)


def insert_colrefs(
    node: expr.Expr,
    *,
    table_ref: plc.expressions.TableReference,
    name_to_index: Mapping[str, int],
) -> expr.Expr:
    """
    Insert column references into an expression before conversion to libcudf AST.

    Parameters
    ----------
    node
        Expression to insert references into.
    table_ref
        pylibcudf `TableReference` indicating whether column
        references are coming from the left or right table.
    name_to_index:
        Mapping from column names to column indices in the table
        eventually used for evaluation.

    Notes
    -----
    All column references are wrapped in the same, singular, table
    reference, so this function relies on the expression only
    containing column references from a single table.

    Returns
    -------
    New expression with column references inserted.
    """
    mapper = CachingVisitor(
        _insert_colrefs, state={"table_ref": table_ref, "name_to_index": name_to_index}
    )
    return mapper(node)

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Conversion of expression nodes to libcudf AST nodes."""

from __future__ import annotations

from functools import partial, reduce, singledispatch
from typing import TYPE_CHECKING, TypeAlias

import pylibcudf as plc
from pylibcudf import expressions as pexpr

from polars.polars import _expr_nodes as pl_expr

from cudf_polars.dsl import expr
from cudf_polars.dsl.traversal import make_recursive
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import Mapping

# Can't merge these op-mapping dictionaries because scoped enum values
# are exposed by cython with equality/hash based one their underlying
# representation type. So in a dict they are just treated as integers.
BINOP_TO_ASTOP = {
    plc.binaryop.BinaryOperator.EQUAL: pexpr.ASTOperator.EQUAL,
    plc.binaryop.BinaryOperator.NULL_EQUALS: pexpr.ASTOperator.NULL_EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL: pexpr.ASTOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS: pexpr.ASTOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL: pexpr.ASTOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER: pexpr.ASTOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: pexpr.ASTOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.ADD: pexpr.ASTOperator.ADD,
    plc.binaryop.BinaryOperator.SUB: pexpr.ASTOperator.SUB,
    plc.binaryop.BinaryOperator.MUL: pexpr.ASTOperator.MUL,
    plc.binaryop.BinaryOperator.DIV: pexpr.ASTOperator.DIV,
    plc.binaryop.BinaryOperator.TRUE_DIV: pexpr.ASTOperator.TRUE_DIV,
    plc.binaryop.BinaryOperator.FLOOR_DIV: pexpr.ASTOperator.FLOOR_DIV,
    plc.binaryop.BinaryOperator.PYMOD: pexpr.ASTOperator.PYMOD,
    plc.binaryop.BinaryOperator.BITWISE_AND: pexpr.ASTOperator.BITWISE_AND,
    plc.binaryop.BinaryOperator.BITWISE_OR: pexpr.ASTOperator.BITWISE_OR,
    plc.binaryop.BinaryOperator.BITWISE_XOR: pexpr.ASTOperator.BITWISE_XOR,
    plc.binaryop.BinaryOperator.LOGICAL_AND: pexpr.ASTOperator.LOGICAL_AND,
    plc.binaryop.BinaryOperator.LOGICAL_OR: pexpr.ASTOperator.LOGICAL_OR,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_AND: pexpr.ASTOperator.NULL_LOGICAL_AND,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_OR: pexpr.ASTOperator.NULL_LOGICAL_OR,
}

UOP_TO_ASTOP = {
    plc.unary.UnaryOperator.SIN: pexpr.ASTOperator.SIN,
    plc.unary.UnaryOperator.COS: pexpr.ASTOperator.COS,
    plc.unary.UnaryOperator.TAN: pexpr.ASTOperator.TAN,
    plc.unary.UnaryOperator.ARCSIN: pexpr.ASTOperator.ARCSIN,
    plc.unary.UnaryOperator.ARCCOS: pexpr.ASTOperator.ARCCOS,
    plc.unary.UnaryOperator.ARCTAN: pexpr.ASTOperator.ARCTAN,
    plc.unary.UnaryOperator.SINH: pexpr.ASTOperator.SINH,
    plc.unary.UnaryOperator.COSH: pexpr.ASTOperator.COSH,
    plc.unary.UnaryOperator.TANH: pexpr.ASTOperator.TANH,
    plc.unary.UnaryOperator.ARCSINH: pexpr.ASTOperator.ARCSINH,
    plc.unary.UnaryOperator.ARCCOSH: pexpr.ASTOperator.ARCCOSH,
    plc.unary.UnaryOperator.ARCTANH: pexpr.ASTOperator.ARCTANH,
    plc.unary.UnaryOperator.EXP: pexpr.ASTOperator.EXP,
    plc.unary.UnaryOperator.LOG: pexpr.ASTOperator.LOG,
    plc.unary.UnaryOperator.SQRT: pexpr.ASTOperator.SQRT,
    plc.unary.UnaryOperator.CBRT: pexpr.ASTOperator.CBRT,
    plc.unary.UnaryOperator.CEIL: pexpr.ASTOperator.CEIL,
    plc.unary.UnaryOperator.FLOOR: pexpr.ASTOperator.FLOOR,
    plc.unary.UnaryOperator.ABS: pexpr.ASTOperator.ABS,
    plc.unary.UnaryOperator.RINT: pexpr.ASTOperator.RINT,
    plc.unary.UnaryOperator.BIT_INVERT: pexpr.ASTOperator.BIT_INVERT,
    plc.unary.UnaryOperator.NOT: pexpr.ASTOperator.NOT,
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


Transformer: TypeAlias = GenericTransformer[expr.Expr, pexpr.Expression]


@singledispatch
def _to_ast(node: expr.Expr, self: Transformer) -> pexpr.Expression:
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
def _(node: expr.Col, self: Transformer) -> pexpr.Expression:
    if self.state["for_parquet"]:
        return pexpr.ColumnNameReference(node.name)
    return pexpr.ColumnReference(self.state["name_to_index"][node.name])


@_to_ast.register
def _(node: expr.Literal, self: Transformer) -> pexpr.Expression:
    return pexpr.Literal(plc.interop.from_arrow(node.value))


@_to_ast.register
def _(node: expr.BinOp, self: Transformer) -> pexpr.Expression:
    if node.op == plc.binaryop.BinaryOperator.NULL_NOT_EQUALS:
        return pexpr.Operation(
            pexpr.ASTOperator.NOT,
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
            return pexpr.Operation(BINOP_TO_ASTOP[op], self(op1), self(op2))
        elif op1_col and op2_col:
            raise NotImplementedError(
                "Parquet filter binops must have one column reference not two"
            )
    return pexpr.Operation(BINOP_TO_ASTOP[node.op], *map(self, node.children))


@_to_ast.register
def _(node: expr.BooleanFunction, self: Transformer) -> pexpr.Expression:
    if node.name == pl_expr.BooleanFunction.IsIn:
        needles, haystack = node.children
        if isinstance(haystack, expr.LiteralColumn) and len(haystack.value) < 16:
            # 16 is an arbitrary limit
            needle_ref = self(needles)
            values = [pexpr.Literal(plc.interop.from_arrow(v)) for v in haystack.value]
            return reduce(
                partial(pexpr.Operation, pexpr.ASTOperator.LOGICAL_OR),
                (
                    pexpr.Operation(pexpr.ASTOperator.EQUAL, needle_ref, value)
                    for value in values
                ),
            )
    if self.state["for_parquet"] and isinstance(node.children[0], expr.Col):
        raise NotImplementedError(
            f"Parquet filters don't support {node.name} on columns"
        )
    if node.name == pl_expr.BooleanFunction.IsNull:
        return pexpr.Operation(pexpr.ASTOperator.IS_NULL, self(node.children[0]))
    elif node.name == pl_expr.BooleanFunction.IsNotNull:
        return pexpr.Operation(
            pexpr.ASTOperator.NOT,
            pexpr.Operation(pexpr.ASTOperator.IS_NULL, self(node.children[0])),
        )
    elif node.name == pl_expr.BooleanFunction.Not:
        return pexpr.Operation(pexpr.ASTOperator.NOT, self(node.children[0]))
    raise NotImplementedError(f"AST conversion does not support {node.name}")


@_to_ast.register
def _(node: expr.UnaryFunction, self: Transformer) -> pexpr.Expression:
    if isinstance(node.children[0], expr.Col) and self.state["for_parquet"]:
        raise NotImplementedError(
            "Parquet filters don't support {node.name} on columns"
        )
    return pexpr.Operation(
        UOP_TO_ASTOP[node._OP_MAPPING[node.name]], self(node.children[0])
    )


def to_parquet_filter(node: expr.Expr) -> pexpr.Expression | None:
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
    mapper: Transformer = make_recursive(_to_ast, state={"for_parquet": True})
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None


def to_ast(
    node: expr.Expr, *, name_to_index: Mapping[str, int]
) -> pexpr.Expression | None:
    """
    Convert an expression to libcudf AST nodes suitable for compute_column.

    Parameters
    ----------
    node
        Expression to convert.
    name_to_index
        Mapping from column names to their index in the table that
        will be used for expression evaluation.

    Returns
    -------
    pylibcudf Expressoin if conversion is possible, otherwise None.
    """
    mapper: Transformer = make_recursive(
        _to_ast, state={"for_parquet": False, "name_to_index": name_to_index}
    )
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None

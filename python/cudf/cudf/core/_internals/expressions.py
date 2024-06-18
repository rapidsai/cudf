# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from __future__ import annotations

import ast
import functools

from cudf._lib.pylibcudf.expressions import (
    ASTOperator,
    ColumnReference,
    Expression,
    Literal,
    Operation,
)

# This dictionary encodes the mapping from Python AST operators to their cudf
# counterparts.
python_cudf_operator_map = {
    # Binary operators
    ast.Add: ASTOperator.ADD,
    ast.Sub: ASTOperator.SUB,
    ast.Mult: ASTOperator.MUL,
    ast.Div: ASTOperator.DIV,
    ast.FloorDiv: ASTOperator.FLOOR_DIV,
    ast.Mod: ASTOperator.PYMOD,
    ast.Pow: ASTOperator.POW,
    ast.Eq: ASTOperator.EQUAL,
    ast.NotEq: ASTOperator.NOT_EQUAL,
    ast.Lt: ASTOperator.LESS,
    ast.Gt: ASTOperator.GREATER,
    ast.LtE: ASTOperator.LESS_EQUAL,
    ast.GtE: ASTOperator.GREATER_EQUAL,
    ast.BitXor: ASTOperator.BITWISE_XOR,
    # TODO: The mapping of logical/bitwise operators here is inconsistent with
    # pandas. In pandas, Both `BitAnd` and `And` map to
    # `ASTOperator.LOGICAL_AND` for booleans, while they map to
    # `ASTOperator.BITWISE_AND` for integers. However, there is no good way to
    # encode this at present because expressions can be arbitrarily nested so
    # we won't know the dtype of the input without inserting a much more
    # complex traversal of the expression tree to determine the output types at
    # each node. For now, we'll rely on users to use the appropriate operator.
    ast.BitAnd: ASTOperator.BITWISE_AND,
    ast.BitOr: ASTOperator.BITWISE_OR,
    ast.And: ASTOperator.LOGICAL_AND,
    ast.Or: ASTOperator.LOGICAL_OR,
    # Unary operators
    ast.Invert: ASTOperator.BIT_INVERT,
    ast.Not: ASTOperator.NOT,
    # TODO: Missing USub, possibility other unary ops?
}


# Mapping between Python function names encode in an ast.Call node and the
# corresponding libcudf C++ AST operators.
python_cudf_function_map = {
    # TODO: Operators listed on
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#expression-evaluation-via-eval  # noqa: E501
    # that we don't support yet:
    # expm1, log1p, arctan2 and log10.
    "isnull": ASTOperator.IS_NULL,
    "isna": ASTOperator.IS_NULL,
    "sin": ASTOperator.SIN,
    "cos": ASTOperator.COS,
    "tan": ASTOperator.TAN,
    "arcsin": ASTOperator.ARCSIN,
    "arccos": ASTOperator.ARCCOS,
    "arctan": ASTOperator.ARCTAN,
    "sinh": ASTOperator.SINH,
    "cosh": ASTOperator.COSH,
    "tanh": ASTOperator.TANH,
    "arcsinh": ASTOperator.ARCSINH,
    "arccosh": ASTOperator.ARCCOSH,
    "arctanh": ASTOperator.ARCTANH,
    "exp": ASTOperator.EXP,
    "log": ASTOperator.LOG,
    "sqrt": ASTOperator.SQRT,
    "abs": ASTOperator.ABS,
    "ceil": ASTOperator.CEIL,
    "floor": ASTOperator.FLOOR,
    # TODO: Operators supported by libcudf with no Python function analog.
    # ast.rint: ASTOperator.RINT,
    # ast.cbrt: ASTOperator.CBRT,
}


class libcudfASTVisitor(ast.NodeVisitor):
    """A NodeVisitor specialized for constructing a libcudf expression tree.

    This visitor is designed to handle AST nodes that have libcudf equivalents.
    It constructs column references from names and literals from constants,
    then builds up operations. The final result can be accessed using the
    `expression` property. The visitor must be kept in scope for as long as the
    expression is needed because all of the underlying libcudf expressions will
    be destroyed when the libcudfASTVisitor is.

    Parameters
    ----------
    col_names : Tuple[str]
        The column names used to map the names in an expression.
    """

    def __init__(self, col_names: tuple[str]):
        self.stack: list[Expression] = []
        self.nodes: list[Expression] = []
        self.col_names = col_names

    @property
    def expression(self):
        """Expression: The result of parsing an AST."""
        assert len(self.stack) == 1
        return self.stack[-1]

    def visit_Name(self, node):
        try:
            col_id = self.col_names.index(node.id)
        except ValueError:
            raise ValueError(f"Unknown column name {node.id}")
        self.stack.append(ColumnReference(col_id))

    def visit_Constant(self, node):
        if not isinstance(node, (ast.Num, ast.Str)):
            raise ValueError(
                f"Unsupported literal {repr(node.value)} of type "
                "{type(node.value).__name__}"
            )
        self.stack.append(Literal(node.value))

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self.nodes.append(self.stack.pop())
        if isinstance(node.op, ast.USub):
            # TODO: Except for leaf nodes, we won't know the type of the
            # operand, so there's no way to know whether this should be a float
            # or an int. We should maybe see what Spark does, and this will
            # probably require casting.
            self.nodes.append(Literal(-1))
            op = ASTOperator.MUL
            self.stack.append(Operation(op, self.nodes[-1], self.nodes[-2]))
        elif isinstance(node.op, ast.UAdd):
            self.stack.append(self.nodes[-1])
        else:
            op = python_cudf_operator_map[type(node.op)]
            self.stack.append(Operation(op, self.nodes[-1]))

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.nodes.append(self.stack.pop())
        self.nodes.append(self.stack.pop())

        op = python_cudf_operator_map[type(node.op)]
        self.stack.append(Operation(op, self.nodes[-1], self.nodes[-2]))

    def _visit_BoolOp_Compare(self, operators, operands, has_multiple_ops):
        # Helper function handling the common components of parsing BoolOp and
        # Compare AST nodes. These two types of nodes both support chaining
        # (e.g. `a > b > c` is equivalent to `a > b and b > c`, so this
        # function helps standardize that.

        # TODO: Whether And/Or and BitAnd/BitOr actually correspond to
        # logical or bitwise operators depends on the data types that they
        # are applied to. We'll need to add logic to map to that.
        inner_ops = []
        for op, (left, right) in zip(operators, operands):
            # Note that this will lead to duplicate nodes, e.g. if
            # the comparison is `a < b < c` that will be encoded as
            # `a < b and b < c`. We could potentially optimize by caching
            # expressions by name so that we only construct them once.
            self.visit(left)
            self.visit(right)

            self.nodes.append(self.stack.pop())
            self.nodes.append(self.stack.pop())

            op = python_cudf_operator_map[type(op)]
            inner_ops.append(Operation(op, self.nodes[-1], self.nodes[-2]))

        self.nodes.extend(inner_ops)

        # If we have more than one comparator, we need to link them
        # together with LOGICAL_AND operators.
        if has_multiple_ops:
            op = ASTOperator.LOGICAL_AND

            def _combine_compare_ops(left, right):
                self.nodes.append(Operation(op, left, right))
                return self.nodes[-1]

            functools.reduce(_combine_compare_ops, inner_ops)

        self.stack.append(self.nodes[-1])

    def visit_BoolOp(self, node):
        operators = [node.op] * (len(node.values) - 1)
        operands = zip(node.values[:-1], node.values[1:])
        self._visit_BoolOp_Compare(operators, operands, len(node.values) > 2)

    def visit_Compare(self, node):
        operands = (node.left, *node.comparators)
        has_multiple_ops = len(operands) > 2
        operands = zip(operands[:-1], operands[1:])
        self._visit_BoolOp_Compare(node.ops, operands, has_multiple_ops)

    def visit_Call(self, node):
        try:
            op = python_cudf_function_map[node.func.id]
        except KeyError:
            raise ValueError(f"Unsupported function {node.func}.")
        # Assuming only unary functions are supported, which is checked above.
        if len(node.args) != 1 or node.keywords:
            raise ValueError(
                f"Function {node.func} only accepts one positional "
                "argument."
            )
        self.visit(node.args[0])

        self.nodes.append(self.stack.pop())
        self.stack.append(Operation(op, self.nodes[-1]))


@functools.lru_cache(256)
def parse_expression(expr: str, col_names: tuple[str]):
    visitor = libcudfASTVisitor(col_names)
    visitor.visit(ast.parse(expr))
    return visitor

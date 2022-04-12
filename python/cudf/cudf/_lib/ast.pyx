# Copyright (c) 2022, NVIDIA CORPORATION.

import ast
import functools
from enum import Enum
from typing import List

from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from cudf._lib.ast cimport underlying_type_ast_operator
from cudf._lib.column cimport Column
from cudf._lib.cpp cimport ast as libcudf_ast, transform as libcudf_transform
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport table_view_from_table

# Aliases for simplicity
ctypedef unique_ptr[libcudf_ast.expression] expression_ptr


class ASTOperator(Enum):
    ADD = libcudf_ast.ast_operator.ADD
    SUB = libcudf_ast.ast_operator.SUB
    MUL = libcudf_ast.ast_operator.MUL
    DIV = libcudf_ast.ast_operator.DIV
    TRUE_DIV = libcudf_ast.ast_operator.TRUE_DIV
    FLOOR_DIV = libcudf_ast.ast_operator.FLOOR_DIV
    MOD = libcudf_ast.ast_operator.MOD
    PYMOD = libcudf_ast.ast_operator.PYMOD
    POW = libcudf_ast.ast_operator.POW
    EQUAL = libcudf_ast.ast_operator.EQUAL
    NULL_EQUAL = libcudf_ast.ast_operator.NULL_EQUAL
    NOT_EQUAL = libcudf_ast.ast_operator.NOT_EQUAL
    LESS = libcudf_ast.ast_operator.LESS
    GREATER = libcudf_ast.ast_operator.GREATER
    LESS_EQUAL = libcudf_ast.ast_operator.LESS_EQUAL
    GREATER_EQUAL = libcudf_ast.ast_operator.GREATER_EQUAL
    BITWISE_AND = libcudf_ast.ast_operator.BITWISE_AND
    BITWISE_OR = libcudf_ast.ast_operator.BITWISE_OR
    BITWISE_XOR = libcudf_ast.ast_operator.BITWISE_XOR
    LOGICAL_AND = libcudf_ast.ast_operator.LOGICAL_AND
    NULL_LOGICAL_AND = libcudf_ast.ast_operator.NULL_LOGICAL_AND
    LOGICAL_OR = libcudf_ast.ast_operator.LOGICAL_OR
    NULL_LOGICAL_OR = libcudf_ast.ast_operator.NULL_LOGICAL_OR
    # Unary operators
    IDENTITY = libcudf_ast.ast_operator.IDENTITY
    SIN = libcudf_ast.ast_operator.SIN
    COS = libcudf_ast.ast_operator.COS
    TAN = libcudf_ast.ast_operator.TAN
    ARCSIN = libcudf_ast.ast_operator.ARCSIN
    ARCCOS = libcudf_ast.ast_operator.ARCCOS
    ARCTAN = libcudf_ast.ast_operator.ARCTAN
    SINH = libcudf_ast.ast_operator.SINH
    COSH = libcudf_ast.ast_operator.COSH
    TANH = libcudf_ast.ast_operator.TANH
    ARCSINH = libcudf_ast.ast_operator.ARCSINH
    ARCCOSH = libcudf_ast.ast_operator.ARCCOSH
    ARCTANH = libcudf_ast.ast_operator.ARCTANH
    EXP = libcudf_ast.ast_operator.EXP
    LOG = libcudf_ast.ast_operator.LOG
    SQRT = libcudf_ast.ast_operator.SQRT
    CBRT = libcudf_ast.ast_operator.CBRT
    CEIL = libcudf_ast.ast_operator.CEIL
    FLOOR = libcudf_ast.ast_operator.FLOOR
    ABS = libcudf_ast.ast_operator.ABS
    RINT = libcudf_ast.ast_operator.RINT
    BIT_INVERT = libcudf_ast.ast_operator.BIT_INVERT
    NOT = libcudf_ast.ast_operator.NOT


class TableReference(Enum):
    LEFT = libcudf_ast.table_reference.LEFT
    RIGHT = libcudf_ast.table_reference.RIGHT


# Note that this function only currently supports numeric literals. libcudf
# expressions don't really support other types yet though, so this isn't
# restrictive at the moment.
cdef class Literal(Expression):
    def __cinit__(self, value):
        cdef int val = value
        self.c_scalar = make_unique[numeric_scalar[int64_t]](val, True)
        self.c_obj = <expression_ptr> make_unique[libcudf_ast.literal](
            <numeric_scalar[int64_t] &>dereference(self.c_scalar)
        )


cdef class ColumnReference(Expression):
    def __cinit__(self, size_type index):
        self.c_obj = <expression_ptr>make_unique[libcudf_ast.column_reference](
            index
        )


cdef class Operation(Expression):
    def __cinit__(self, op, Expression left, Expression right=None):
        # This awkward double casting is the only way to get Cython to generate
        # valid C++ that doesn't try to apply the shift operator directly to
        # values of the enum (which is invalid).
        cdef libcudf_ast.ast_operator op_value = <libcudf_ast.ast_operator>(
            <underlying_type_ast_operator> op.value
        )

        if right is None:
            self.c_obj = <expression_ptr> make_unique[libcudf_ast.operation](
                op_value, dereference(left.c_obj)
            )
        else:
            self.c_obj = <expression_ptr> make_unique[libcudf_ast.operation](
                op_value, dereference(left.c_obj), dereference(right.c_obj)
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
}


# Mapping between Python function names encode in an ast.Call node and the
# corresponding libcudf C++ AST operators.
python_cudf_function_map = {
    # TODO: Operators listed on
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#expression-evaluation-via-eval
    # that we don't support yet:
    # expm1, log1p, arctan2 and log10.
    'sin': ASTOperator.SIN,
    'cos': ASTOperator.COS,
    'tan': ASTOperator.TAN,
    'arcsin': ASTOperator.ARCSIN,
    'arccos': ASTOperator.ARCCOS,
    'arctan': ASTOperator.ARCTAN,
    'sinh': ASTOperator.SINH,
    'cosh': ASTOperator.COSH,
    'tanh': ASTOperator.TANH,
    'arcsinh': ASTOperator.ARCSINH,
    'arccosh': ASTOperator.ARCCOSH,
    'arctanh': ASTOperator.ARCTANH,
    'exp': ASTOperator.EXP,
    'log': ASTOperator.LOG,
    'sqrt': ASTOperator.SQRT,
    'abs': ASTOperator.ABS,

    # TODO: Operators supported by libcudf with no Python function analog.
    # ast.rint: ASTOperator.RINT,
    # ast.cbrt: ASTOperator.CBRT,
    # ast.ceil: ASTOperator.CEIL,
    # ast.floor: ASTOperator.FLOOR,
}


cdef parse_expression(root, tuple col_names, list stack, list nodes):
    """Construct an evaluable libcudf expression by traversing Python AST.

    This function performs a recursive traversal of the provided root
    node, constructing column references from names and literal values
    from constants, then building up expressions. The final result is
    the expression contained in the ``stack`` list once the function has
    terminated: this list will always have length one at the end of parsing
    a valid expression.

    Parameters
    ----------
    root : ast.AST
        An ast node generated by :py:func:`ast.parse`.
    col_names : tuple
        The column names in the data frame, which are used to generate indices
        for column references to named columns in the expression.
    stack : list
        The current set of nodes to process. This list is empty on the initial
        call to this function. New elements are added whenever new nodes are
        created. When parsing the current root requires creating an Operation
        node, a suitable number of elements (corresponding to the arity of the
        operator) are popped from the stack as the operands for the operation.
        When the recursive traversal is complete, the stack will contain
        exactly one element, the expression to evaluate.
    nodes : list
        The set of all nodes created while parsing the expression. This
        argument is necessary because all C++ node types are non-owning
        objects, so if the Python Expressions corresponding to nodes in the
        expression go out of scope and are garbage-collected the final
        expression will contain references to invalid data and seg fault upon
        evaluation. This list must remain in scope until the expression has
        been evaluated.
    """
    # TODO: We'll eventually need to find a way to support operations on mixed
    # but compatible dtypes (e.g. adding int to float).
    # Base cases: Name
    if isinstance(root, ast.Name):
        try:
            col_id = col_names.index(root.id) + 1
        except ValueError:
            raise ValueError(f"Unknown column name {root.id}")
        stack.append(ColumnReference(col_id))
    # Note: in Python > 3.7 ast.Num is a subclass of ast.Constant. We may need
    # to generalize this code eventually if that inheritance is removed.
    if isinstance(root, ast.Constant):
        if isinstance(root, ast.Num):
            stack.append(Literal(root.n))
        else:
            raise ValueError(
                f"Unsupported literal {repr(root.value)} of type "
                "{type(root.value)}"
            )
    elif isinstance(root, ast.UnaryOp):
        parse_expression(root.operand, col_names, stack, nodes)
        nodes.append(stack.pop())
        op = python_cudf_operator_map[type(root.op)]
        stack.append(Operation(op, nodes[-1]))
    elif isinstance(root, ast.BinOp):
        parse_expression(root.left, col_names, stack, nodes)
        parse_expression(root.right, col_names, stack, nodes)
        nodes.append(stack.pop())
        nodes.append(stack.pop())

        op = python_cudf_operator_map[type(root.op)]
        stack.append(Operation(op, nodes[-1], nodes[-2]))

    # TODO: Whether And/Or and BitAnd/BitOr actually correspond to
    # logical or bitwise operators depends on the data types that they
    # are applied to. We'll need to add logic to map to that.
    elif isinstance(root, (ast.BoolOp, ast.Compare)):
        if isinstance(root, ast.BoolOp):
            operators = [root.op] * (len(root.values) - 1)
            operands = zip(root.values[:-1], root.values[1:])
            multiple_ops = len(root.values) > 2
        else:
            operators = root.ops
            operands = (root.left, *root.comparators)
            multiple_ops = len(operands) > 2
            operands = zip(operands[:-1], operands[1:])

        inner_ops = []
        for op, (left, right) in zip(operators, operands):
            # Note that this will lead to duplicate nodes, e.g. if
            # the comparison is `a < b < c` that will be encoded as
            # `a < b and b < c`.
            parse_expression(left, col_names, stack, nodes)
            parse_expression(right, col_names, stack, nodes)

            nodes.append(stack.pop())
            nodes.append(stack.pop())

            op = python_cudf_operator_map[type(op)]
            inner_ops.append(Operation(op, nodes[-1], nodes[-2]))

        nodes.extend(inner_ops)

        # If we have more than one comparator, we need to link them
        # together with LOGICAL_AND operators.
        if multiple_ops:
            op = ASTOperator.LOGICAL_AND

            def _combine_compare_ops(left, right):
                nodes.append(Operation(op, left, right))
                return nodes[-1]

            functools.reduce(_combine_compare_ops, inner_ops)

        stack.append(nodes[-1])
    elif isinstance(root, ast.Call):
        try:
            op = python_cudf_function_map[root.func.id]
        except KeyError:
            raise ValueError(f"Unsupported function {root.func}.")
        # Assuming only unary functions are supported, which is checked above.
        if len(root.args) != 1 or root.keywords:
            raise ValueError(
                f"Function {root.func} only accepts one positional "
                "argument."
            )
        parse_expression(root.args[0], col_names, stack, nodes)

        nodes.append(stack.pop())
        stack.append(Operation(op, nodes[-1]))
    elif isinstance(root, list):
        for item in root:
            parse_expression(item, col_names, stack, nodes)


# TODO: It would be nice to use a dataclass for this, but Cython won't support
# it until we upgrade to 3.0.
class _OwningExpression:
    """A container for an Expression that owns the Expression's subnodes."""
    def __init__(self, expression: Expression, nodes: List[Expression]):
        self.expression = expression
        self.nodes = nodes


@functools.lru_cache(256)
def parse_expression_cached(str expr, tuple col_names):
    """A caching wrapper for parse_expression.

    The signature is chosen so as to appropriately determine the cache key.
    """
    stack = []
    nodes = []
    parse_expression(
        ast.parse(expr).body[0].value, col_names, stack, nodes
    )
    return _OwningExpression(stack[-1], nodes)


def evaluate_expression(df: "cudf.DataFrame", expr: str):
    """Create a cudf evaluable expression from a string and evaluate it."""
    expr_container = parse_expression_cached(expr, df._column_names)

    # At the end, all the stack contains is the expression to evaluate.
    cdef Expression cudf_expr = expr_container.expression
    cdef table_view tbl = table_view_from_table(df)
    cdef unique_ptr[column] col
    with nogil:
        col = move(
            libcudf_transform.compute_column(
                tbl,
                <libcudf_ast.expression &> dereference(cudf_expr.c_obj.get())
            )
        )
    return {None: Column.from_unique_ptr(move(col))}

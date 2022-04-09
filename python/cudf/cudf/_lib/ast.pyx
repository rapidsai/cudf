# Copyright (c) 2022, NVIDIA CORPORATION.

import ast
import functools
from enum import Enum

from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from cudf._lib.ast cimport underlying_type_ast_operator
from cudf._lib.column cimport Column
from cudf._lib.cpp cimport ast as libcudf_ast
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport table_view_from_table


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
    # NULL_EQUAL = libcudf_ast.ast_operator.NULL_EQUAL
    NOT_EQUAL = libcudf_ast.ast_operator.NOT_EQUAL
    LESS = libcudf_ast.ast_operator.LESS
    GREATER = libcudf_ast.ast_operator.GREATER
    LESS_EQUAL = libcudf_ast.ast_operator.LESS_EQUAL
    GREATER_EQUAL = libcudf_ast.ast_operator.GREATER_EQUAL
    BITWISE_AND = libcudf_ast.ast_operator.BITWISE_AND
    BITWISE_OR = libcudf_ast.ast_operator.BITWISE_OR
    BITWISE_XOR = libcudf_ast.ast_operator.BITWISE_XOR
    LOGICAL_AND = libcudf_ast.ast_operator.LOGICAL_AND
    # NULL_LOGICAL_AND = libcudf_ast.ast_operator.NULL_LOGICAL_AND
    LOGICAL_OR = libcudf_ast.ast_operator.LOGICAL_OR
    # NULL_LOGICAL_OR = libcudf_ast.ast_operator.NULL_LOGICAL_OR
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


cdef class Literal(Expression):
    def __cinit__(self, value):
        # TODO: Generalize this to other types of literals.
        cdef int val = value
        self.c_scalar = make_unique[numeric_scalar[int64_t]](val, True)
        self.c_obj = <unique_ptr[libcudf_ast.expression]> make_unique[
            libcudf_ast.literal](
                <numeric_scalar[int64_t] &>dereference(self.c_scalar))


cdef class ColumnReference(Expression):
    def __cinit__(self, size_type index):
        self.c_obj = <unique_ptr[libcudf_ast.expression]> make_unique[
            libcudf_ast.column_reference](index)


cdef class Operation(Expression):
    def __cinit__(self, op, Expression left, Expression right=None):
        # This awkward double casting is the only way to get Cython to generate
        # valid C++ that doesn't try to apply the shift operator directly to
        # values of the enum (which is invalid).
        cdef libcudf_ast.ast_operator op_value = <libcudf_ast.ast_operator> (
            <underlying_type_ast_operator> op.value)

        if right is None:
            self.c_obj = <unique_ptr[libcudf_ast.expression]> make_unique[
                libcudf_ast.operation](op_value, dereference(left.c_obj))
        else:
            self.c_obj = <unique_ptr[libcudf_ast.expression]> make_unique[
                libcudf_ast.operation](
                    op_value, dereference(left.c_obj), dereference(right.c_obj)
            )


# This dictionary encodes the mapping from Python AST operators to their cudf
# counterparts.
python_cudf_ast_map = {
    # TODO: Mapping TBD for commented out operators.
    # Binary operators
    ast.Add: ASTOperator.ADD,
    ast.Sub: ASTOperator.SUB,
    ast.Mult: ASTOperator.MUL,
    ast.Div: ASTOperator.DIV,
    # ast.True: ASTOperator.TRUE_DIV,
    ast.FloorDiv: ASTOperator.FLOOR_DIV,
    ast.Mod: ASTOperator.PYMOD,
    # ast.Pymod: ASTOperator.PYMOD,
    ast.Pow: ASTOperator.POW,
    ast.Eq: ASTOperator.EQUAL,
    ast.NotEq: ASTOperator.NOT_EQUAL,
    ast.Lt: ASTOperator.LESS,
    ast.Gt: ASTOperator.GREATER,
    ast.LtE: ASTOperator.LESS_EQUAL,
    ast.GtE: ASTOperator.GREATER_EQUAL,
    ast.BitAnd: ASTOperator.BITWISE_AND,
    ast.BitOr: ASTOperator.BITWISE_OR,
    ast.BitXor: ASTOperator.BITWISE_XOR,
    # TODO: These maps are wrong, but for pandas compatibility they actually
    # need to be dtype-specific so this is just for testing the AST parsing
    # logic. Eventually the lookup for these will need to be updated.
    ast.And: ASTOperator.BITWISE_AND,
    ast.Or: ASTOperator.BITWISE_OR,
    # ast.And: ASTOperator.LOGICAL_AND,
    # ast.Or: ASTOperator.LOGICAL_OR,

    # Unary operators
    # ast.Identity: ASTOperator.IDENTITY,
    # ast.Sin: ASTOperator.SIN,
    # ast.Cos: ASTOperator.COS,
    # ast.Tan: ASTOperator.TAN,
    # ast.Arcsin: ASTOperator.ARCSIN,
    # ast.Arccos: ASTOperator.ARCCOS,
    # ast.Arctan: ASTOperator.ARCTAN,
    # ast.Sinh: ASTOperator.SINH,
    # ast.Cosh: ASTOperator.COSH,
    # ast.Tanh: ASTOperator.TANH,
    # ast.Arcsinh: ASTOperator.ARCSINH,
    # ast.Arccosh: ASTOperator.ARCCOSH,
    # ast.Arctanh: ASTOperator.ARCTANH,
    # ast.Exp: ASTOperator.EXP,
    # ast.Log: ASTOperator.LOG,
    # ast.Sqrt: ASTOperator.SQRT,
    # ast.Cbrt: ASTOperator.CBRT,
    # ast.Ceil: ASTOperator.CEIL,
    # ast.Floor: ASTOperator.FLOOR,
    # ast.Abs: ASTOperator.ABS,
    # ast.Rint: ASTOperator.RINT,
    # ast.Bit: ASTOperator.BIT_INVERT,
    # ast.Not: ASTOperator.NOT,
}


cdef ast_traverse(root, tuple col_names, list stack, list nodes):
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
        When the recursive traversal is complete, the stack will have length
        exactly one and contain the expression to evaluate.
    nodes : list
        The set of all nodes created while parsing the expression. This
        argument is necessary because all C++ node types are non-owning
        objects, so if the Python Expressions corresponding to nodes in the
        expression go out of scope and are garbage-collected the final
        expression will contain references to invalid data and seg fault upon
        evaluation.  This list must remain in scope until the expression has
        been evaluated.
    """
    # TODO: We'll eventually need to find a way to support operations on mixed
    # but compatible dtypes (e.g. adding int to float).
    # Base cases: Name
    if isinstance(root, ast.Name):
        stack.append(ColumnReference(col_names.index(root.id) + 1))
    # Note: in Python > 3.7 ast.Num is a subclass of ast.Constant. We may need
    # to generalize this code eventually if that inheritance is removed.
    elif isinstance(root, ast.Num):
        stack.append(Literal(root.n))
    else:
        # Iterating over `_fields` is _much_ faster than calling
        # `iter_child_nodes`. That may not matter for practical data sizes
        # though, so we'll need to benchmark.
        # for value in ast.iter_child_nodes(root):
        for field in root._fields:
            # Note that since the fields of `value` are ordered, in some cases
            # (e.g. `ast.BinOp`) a single call to `ast_traverse(value, ...)`
            # would have the same effect as explicitly invoking it on the
            # fields (e.g. `left` and `right` for `ast.BinOp`). However, this
            # relies on the fields and their ordering not changing in future
            # Python versions, and that we won't change the parsing logic for
            # e.g. the operators, so it's best to be explicit in all the
            # branches below.
            value = getattr(root, field)
            if isinstance(value, ast.UnaryOp):
                # Faster to directly parse the operand and skip the op.
                ast_traverse(value.operand, col_names, stack, nodes)
                op = python_cudf_ast_map[type(value.op)]
                nodes.append(stack.pop())
                stack.append(Operation(op, nodes[-1]))
            elif isinstance(value, ast.BinOp):
                op = python_cudf_ast_map[type(value.op)]

                ast_traverse(value.left, col_names, stack, nodes)
                ast_traverse(value.right, col_names, stack, nodes)

                nodes.append(stack.pop())
                nodes.append(stack.pop())
                stack.append(Operation(op, nodes[-1], nodes[-2]))

            # TODO: Whether And/Or and BitAnd/BitOr actually correspond to
            # logical or bitwise operators depends on the data types that they
            # are applied to. We'll need to add logic to map to that.
            elif isinstance(value, ast.BoolOp):
                # Chained comparators should be split into multiple sets of
                # comparators. Parsing occurs left to right.
                inner_ops = []
                for left, right in zip(value.values[:-1], value.values[1:]):
                    # Note that this will lead to duplicate nodes, e.g. if
                    # the comparison is `a < b < c` that will be encoded as
                    # `a < b and b < c`.
                    op = python_cudf_ast_map[type(value.op)]

                    ast_traverse(left, col_names, stack, nodes)
                    ast_traverse(right, col_names, stack, nodes)

                    nodes.append(stack.pop())
                    nodes.append(stack.pop())
                    inner_ops.append(Operation(op, nodes[-1], nodes[-2]))
                    nodes.append(inner_ops[-1])

                # If we have more than one comparator, we need to link them
                # together with a bunch of LOGICAL_AND operators.
                if len(value.values) > 2:
                    op = ASTOperator.LOGICAL_AND

                    def _combine_compare_ops(left, right):
                        nodes.append(Operation(op, left, right))
                        return nodes[-1]

                    functools.reduce(_combine_compare_ops, inner_ops)

                stack.append(nodes[-1])
            elif isinstance(value, ast.Compare):
                # Chained comparators should be split into multiple sets of
                # comparators. Parsing occurs left to right.
                operands = (value.left, *value.comparators)
                inner_ops = []
                for op, (left, right) in zip(
                    value.ops, zip(operands[:-1], operands[1:])
                ):
                    # Note that this will lead to duplicate nodes, e.g. if
                    # the comparison is `a < b < c` that will be encoded as
                    # `a < b and b < c`.
                    op = python_cudf_ast_map[type(op)]

                    ast_traverse(left, col_names, stack, nodes)
                    ast_traverse(right, col_names, stack, nodes)

                    nodes.append(stack.pop())
                    nodes.append(stack.pop())
                    inner_ops.append(Operation(op, nodes[-1], nodes[-2]))
                    nodes.append(inner_ops[-1])

                # If we have more than one comparator, we need to link them
                # together with a bunch of LOGICAL_AND operators.
                if len(operands) > 2:
                    op = ASTOperator.LOGICAL_AND

                    def _combine_compare_ops(left, right):
                        nodes.append(Operation(op, left, right))
                        return nodes[-1]

                    functools.reduce(_combine_compare_ops, inner_ops)

                stack.append(nodes[-1])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ast_traverse(item, col_names, stack, nodes)
            elif isinstance(value, ast.AST):
                ast_traverse(value, col_names, stack, nodes)


def evaluate_expression(object df, Expression expr):
    """Evaluate an Expression on a DataFrame."""
    cdef unique_ptr[column] col = libcudf_ast.compute_column(
        table_view_from_table(df),
        <libcudf_ast.expression &> dereference(expr.c_obj.get())
    )
    return {'None': Column.from_unique_ptr(move(col))}


def make_and_evaluate_expression(df, expr):
    """Create a cudf evaluable expression from a string and evaluate it."""
    # Important: both make and evaluate must be coupled to guarantee that the
    # nodes created (the owning ColumnReferences and Literals) remain in scope.
    stack = []
    nodes = []
    ast_traverse(ast.parse(expr), df._column_names, stack, nodes)
    # At the end, all the stack contains is the expression to evaluate.
    return evaluate_expression(df, stack[-1])

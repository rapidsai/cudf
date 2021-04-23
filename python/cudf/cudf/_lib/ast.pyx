# Copyright (c) 2021, NVIDIA CORPORATION.

from enum import Enum
import ast

from cudf.core.column_accessor import ColumnAccessor
from cudf.core.dataframe import DataFrame

from cython.operator cimport dereference
from cudf._lib.cpp.types cimport size_type
from libc.stdint cimport int64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from cudf._lib.ast cimport underlying_type_ast_operator
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.table cimport Table

cimport cudf._lib.cpp.ast as libcudf_ast


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
    NOT_EQUAL = libcudf_ast.ast_operator.NOT_EQUAL
    LESS = libcudf_ast.ast_operator.LESS
    GREATER = libcudf_ast.ast_operator.GREATER
    LESS_EQUAL = libcudf_ast.ast_operator.LESS_EQUAL
    GREATER_EQUAL = libcudf_ast.ast_operator.GREATER_EQUAL
    BITWISE_AND = libcudf_ast.ast_operator.BITWISE_AND
    BITWISE_OR = libcudf_ast.ast_operator.BITWISE_OR
    BITWISE_XOR = libcudf_ast.ast_operator.BITWISE_XOR
    LOGICAL_AND = libcudf_ast.ast_operator.LOGICAL_AND
    LOGICAL_OR = libcudf_ast.ast_operator.LOGICAL_OR
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
    OUTPUT = libcudf_ast.table_reference.OUTPUT


cdef class Literal(Node):
    def __cinit__(self, value):
        # TODO: Generalize this to other types of literals.
        cdef int val = value
        self.c_scalar = make_unique[numeric_scalar[int64_t]](val, True)
        self.c_obj = <unique_ptr[libcudf_ast.node]> make_unique[
            libcudf_ast.literal](
                <numeric_scalar[int64_t] &>dereference(self.c_scalar))


cdef class ColumnReference(Node):
    def __cinit__(self, size_type index):
        self.c_obj = <unique_ptr[libcudf_ast.node]> make_unique[
            libcudf_ast.column_reference](index)


cdef class Expression(Node):
    def __cinit__(self, op, Node left, Node right=None):
        # This awkward double casting appears to be the only way to get Cython
        # to generate valid C++ that doesn't try to apply the shift operator
        # directly to values of the enum (which is invalid).
        cdef libcudf_ast.ast_operator op_value = <libcudf_ast.ast_operator> (
            <underlying_type_ast_operator> op.value)

        if right is None:
            self.c_obj = <unique_ptr[libcudf_ast.node]> make_unique[
                libcudf_ast.expression](op_value, dereference(left.c_obj))
        else:
            self.c_obj = <unique_ptr[libcudf_ast.node]> make_unique[
                libcudf_ast.expression](
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
    ast.And: ASTOperator.LOGICAL_AND,
    ast.Or: ASTOperator.LOGICAL_OR,
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
        created. When parsing the current root requires creating an Expression
        node, a suitable number of elements (corresponding to the arity of the
        operator) are popped from the stack as the operands for the operation.
        When the recursive traversal is complete, the stack will have length
        exactly one and contain the expression to evaluate.
    nodes : list
        The set of all nodes created while parsing the expression. This
        argument is necessary because all C++ node types are non-owning
        objects, so if the Python Nodes corresponding to nodes in the
        expression go out of scope and are garbage-collected the final
        expression will contain references to invalid data and seg fault upon
        evaluation.  This list must remain in scope until the expression has
        been evaluated.
    """
    # Base cases: Name
    if isinstance(root, ast.Name):
        stack.append(ColumnReference(col_names.index(root.id) + 1))
    # Note: in Python > 3.7 ast.Num is a subclass of ast.Constant. We may need
    # to generalize this code eventually if that inheritance is removed.
    elif isinstance(root, ast.Num):
        stack.append(Literal(root.n))
    else:
        # for value in ast.iter_child_nodes(root):
        for field in root._fields:
            value = getattr(root, field)
            if isinstance(value, ast.UnaryOp):
                # TODO: I think here we can optimize by just calling on
                # value.operand, need to verify.
                ast_traverse(value.operand, col_names, stack, nodes)
                op = python_cudf_ast_map[type(value.op)]
                nodes.append(stack.pop())
                stack.append(Expression(op, nodes[-1]))
            elif isinstance(value, ast.BinOp):
                ast_traverse(value, col_names, stack, nodes)
                op = python_cudf_ast_map[type(value.op)]
                # TODO: This assumes that left is parsed before right, should
                # maybe handle this more explicitly.
                nodes.append(stack.pop())
                nodes.append(stack.pop())
                stack.append(Expression(op, nodes[-1], nodes[-2]))
            elif isinstance(value, ast.Compare):
                if len(value.comparators) != 1:
                    # TODO: Can relax this comparison by unpacking the
                    # comparison into multiple.
                    raise ValueError("Only binary comparisons are supported.")
                ast_traverse(value, col_names, stack, nodes)
                op = python_cudf_ast_map[type(value.ops[0])]
                # TODO: This assumes that left is parsed before comparators,
                # should maybe handle this more explicitly.
                nodes.append(stack.pop())
                nodes.append(stack.pop())
                stack.append(Expression(op, nodes[-1], nodes[-2]))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ast_traverse(item, col_names, stack, nodes)
            elif isinstance(value, ast.AST):
                ast_traverse(value, col_names, stack, nodes)


def evaluate_expression(Table df, Expression expr):
    result_data = ColumnAccessor()
    cdef unique_ptr[column] col = libcudf_ast.compute_column(
        df.view(),
        <libcudf_ast.expression &> dereference(expr.c_obj.get())
    )
    result_data['result'] = Column.from_unique_ptr(move(col))
    result_table = Table(data=result_data)
    return DataFrame._from_table(result_table)


def make_and_evaluate_expression(expr, df):
    """Create a cudf evaluable expression from a string and evaluate it."""
    # Important: both make and evaluate must be coupled to guarantee that the
    # nodes created (the owning ColumnReferences and Literals) remain in scope.
    stack = []
    nodes = []
    ast_traverse(ast.parse(expr), df._column_names, stack, nodes)
    # At the end, all the stack contains is the expression to evaluate.
    return evaluate_expression(df, stack[-1])

# Copyright (c) 2021, NVIDIA CORPORATION.

from enum import Enum

from cython.operator cimport dereference
from cudf._lib.cpp.types cimport size_type
from libcpp.memory cimport make_shared, shared_ptr
from cudf._lib.ast cimport underlying_type_ast_operator

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


cdef class Node:
    cdef libcudf_ast.node * _get_ptr(self):
        return self.c_node.get()


cdef class Literal(Node):
    def __cinit__(self, value):
        # TODO: Generalize this to other types of literals.
        cdef float val = value
        self.c_scalar = make_shared[numeric_scalar[float]](val, True)
        self.c_obj = make_shared[libcudf_ast.literal](
            <numeric_scalar[float] &>dereference(self.c_scalar))
        self.c_node = <shared_ptr[libcudf_ast.node]> self.c_obj


cdef class ColumnReference(Node):
    def __cinit__(self, index):
        cdef size_type idx = index
        self.c_obj = make_shared[libcudf_ast.column_reference](idx)
        self.c_node = <shared_ptr[libcudf_ast.node]> self.c_obj


cdef class Expression(Node):
    def __cinit__(self, op, Node left, Node right=None):
        # This awkward double casting appears to be the only way to get Cython
        # to generate valid C++ that doesn't try to apply the shift operator
        # directly to values of the enum (which is invalid).
        cdef libcudf_ast.ast_operator op_value = <libcudf_ast.ast_operator> (
            <underlying_type_ast_operator> op.value)

        if right is None:
            self.c_obj = make_shared[libcudf_ast.expression](
                op_value,
                <const libcudf_ast.node &>dereference(left._get_ptr())
            )
        else:
            self.c_obj = make_shared[libcudf_ast.expression](
                op_value,
                <const libcudf_ast.node &>dereference(left._get_ptr()),
                <const libcudf_ast.node &>dereference(right._get_ptr())
            )

        self.c_node = <shared_ptr[libcudf_ast.node]> self.c_obj

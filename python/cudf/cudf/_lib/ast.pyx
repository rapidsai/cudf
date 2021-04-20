# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np
from enum import Enum

from cython.operator cimport dereference
from cudf._lib.cpp.types cimport size_type

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


cdef class Literal:
    def __cinit__(self, value):
        self.c_scalar = new numeric_scalar[float](value, True)
        self.c_literal = new libcudf_ast.literal(
            <numeric_scalar[float] &>dereference(self.c_scalar))

    def __dealloc__(self):
        del self.c_literal


cdef class ColumnReference:
    def __cinit__(self, index):
        cdef size_type idx = index
        self.c_column_reference = new libcudf_ast.column_reference(idx)

    def __dealloc__(self):
        del self.c_column_reference

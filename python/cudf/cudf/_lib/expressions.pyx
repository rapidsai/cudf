# Copyright (c) 2022, NVIDIA CORPORATION.

from enum import Enum

from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp.memory cimport make_unique, unique_ptr

from cudf._lib.cpp cimport expressions as libcudf_exp
from cudf._lib.cpp.types cimport size_type

# Necessary for proper casting, see below.
ctypedef int32_t underlying_type_ast_operator


# Aliases for simplicity
ctypedef unique_ptr[libcudf_exp.expression] expression_ptr


class ASTOperator(Enum):
    ADD = libcudf_exp.ast_operator.ADD
    SUB = libcudf_exp.ast_operator.SUB
    MUL = libcudf_exp.ast_operator.MUL
    DIV = libcudf_exp.ast_operator.DIV
    TRUE_DIV = libcudf_exp.ast_operator.TRUE_DIV
    FLOOR_DIV = libcudf_exp.ast_operator.FLOOR_DIV
    MOD = libcudf_exp.ast_operator.MOD
    PYMOD = libcudf_exp.ast_operator.PYMOD
    POW = libcudf_exp.ast_operator.POW
    EQUAL = libcudf_exp.ast_operator.EQUAL
    NULL_EQUAL = libcudf_exp.ast_operator.NULL_EQUAL
    NOT_EQUAL = libcudf_exp.ast_operator.NOT_EQUAL
    LESS = libcudf_exp.ast_operator.LESS
    GREATER = libcudf_exp.ast_operator.GREATER
    LESS_EQUAL = libcudf_exp.ast_operator.LESS_EQUAL
    GREATER_EQUAL = libcudf_exp.ast_operator.GREATER_EQUAL
    BITWISE_AND = libcudf_exp.ast_operator.BITWISE_AND
    BITWISE_OR = libcudf_exp.ast_operator.BITWISE_OR
    BITWISE_XOR = libcudf_exp.ast_operator.BITWISE_XOR
    LOGICAL_AND = libcudf_exp.ast_operator.LOGICAL_AND
    NULL_LOGICAL_AND = libcudf_exp.ast_operator.NULL_LOGICAL_AND
    LOGICAL_OR = libcudf_exp.ast_operator.LOGICAL_OR
    NULL_LOGICAL_OR = libcudf_exp.ast_operator.NULL_LOGICAL_OR
    # Unary operators
    IDENTITY = libcudf_exp.ast_operator.IDENTITY
    SIN = libcudf_exp.ast_operator.SIN
    COS = libcudf_exp.ast_operator.COS
    TAN = libcudf_exp.ast_operator.TAN
    ARCSIN = libcudf_exp.ast_operator.ARCSIN
    ARCCOS = libcudf_exp.ast_operator.ARCCOS
    ARCTAN = libcudf_exp.ast_operator.ARCTAN
    SINH = libcudf_exp.ast_operator.SINH
    COSH = libcudf_exp.ast_operator.COSH
    TANH = libcudf_exp.ast_operator.TANH
    ARCSINH = libcudf_exp.ast_operator.ARCSINH
    ARCCOSH = libcudf_exp.ast_operator.ARCCOSH
    ARCTANH = libcudf_exp.ast_operator.ARCTANH
    EXP = libcudf_exp.ast_operator.EXP
    LOG = libcudf_exp.ast_operator.LOG
    SQRT = libcudf_exp.ast_operator.SQRT
    CBRT = libcudf_exp.ast_operator.CBRT
    CEIL = libcudf_exp.ast_operator.CEIL
    FLOOR = libcudf_exp.ast_operator.FLOOR
    ABS = libcudf_exp.ast_operator.ABS
    RINT = libcudf_exp.ast_operator.RINT
    BIT_INVERT = libcudf_exp.ast_operator.BIT_INVERT
    NOT = libcudf_exp.ast_operator.NOT


class TableReference(Enum):
    LEFT = libcudf_exp.table_reference.LEFT
    RIGHT = libcudf_exp.table_reference.RIGHT


# Note that this function only currently supports numeric literals. libcudf
# expressions don't really support other types yet though, so this isn't
# restrictive at the moment.
cdef class Literal(Expression):
    def __cinit__(self, value):
        # TODO: Would love to find a better solution than unions for literals.
        cdef int intval
        cdef double doubleval

        if isinstance(value, int):
            self.c_scalar_type = scalar_type_t.INT
            intval = value
            self.c_scalar.int_ptr = make_unique[numeric_scalar[int64_t]](
                intval, True
            )
            self.c_obj = <expression_ptr> make_unique[libcudf_exp.literal](
                <numeric_scalar[int64_t] &>dereference(self.c_scalar.int_ptr)
            )
        elif isinstance(value, float):
            self.c_scalar_type = scalar_type_t.DOUBLE
            doubleval = value
            self.c_scalar.double_ptr = make_unique[numeric_scalar[double]](
                doubleval, True
            )
            self.c_obj = <expression_ptr> make_unique[libcudf_exp.literal](
                <numeric_scalar[double] &>dereference(self.c_scalar.double_ptr)
            )


cdef class ColumnReference(Expression):
    def __cinit__(self, size_type index):
        self.c_obj = <expression_ptr>make_unique[libcudf_exp.column_reference](
            index
        )


cdef class Operation(Expression):
    def __cinit__(self, op, Expression left, Expression right=None):
        # This awkward double casting is the only way to get Cython to generate
        # valid C++. Cython doesn't support scoped enumerations, so it assumes
        # that enums correspond to their underlying value types and will thus
        # attempt operations that are invalid without first explicitly casting
        # to the underlying before casting to the desired type.
        cdef libcudf_exp.ast_operator op_value = <libcudf_exp.ast_operator>(
            <underlying_type_ast_operator> op.value
        )

        if right is None:
            self.c_obj = <expression_ptr> make_unique[libcudf_exp.operation](
                op_value, dereference(left.c_obj)
            )
        else:
            self.c_obj = <expression_ptr> make_unique[libcudf_exp.operation](
                op_value, dereference(left.c_obj), dereference(right.c_obj)
            )

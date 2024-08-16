# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport (
    duration_scalar,
    numeric_scalar,
    timestamp_scalar,
)
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/ast/expressions.hpp" namespace "cudf::ast" nogil:
    cpdef enum class ast_operator(int32_t):
        # Binary operators
        ADD
        SUB
        MUL
        DIV
        TRUE_DIV
        FLOOR_DIV
        MOD
        PYMOD
        POW
        EQUAL
        NULL_EQUAL
        NOT_EQUAL
        LESS
        GREATER
        LESS_EQUAL
        GREATER_EQUAL
        BITWISE_AND
        BITWISE_OR
        BITWISE_XOR
        NULL_LOGICAL_AND
        LOGICAL_AND
        NULL_LOGICAL_OR
        LOGICAL_OR
        # Unary operators
        IDENTITY
        IS_NULL
        SIN
        COS
        TAN
        ARCSIN
        ARCCOS
        ARCTAN
        SINH
        COSH
        TANH
        ARCSINH
        ARCCOSH
        ARCTANH
        EXP
        LOG
        SQRT
        CBRT
        CEIL
        FLOOR
        ABS
        RINT
        BIT_INVERT
        NOT

    cdef cppclass expression:
        pass

    cpdef enum class table_reference(int32_t):
        LEFT
        RIGHT

    cdef cppclass literal(expression):
        # Due to https://github.com/cython/cython/issues/3198, we need to
        # specify a return type for templated constructors.
        literal literal[T](numeric_scalar[T] &) except +
        literal literal[T](timestamp_scalar[T] &) except +
        literal literal[T](duration_scalar[T] &) except +

    cdef cppclass column_reference(expression):
        # Allow for default C++ parameters by declaring multiple constructors
        # with the default parameters optionally omitted.
        column_reference(size_type) except +
        column_reference(size_type, table_reference) except +

    cdef cppclass operation(expression):
        operation(ast_operator, const expression &)
        operation(ast_operator, const expression &, const expression&)

    cdef cppclass column_name_reference(expression):
        # column_name_reference is only meant for use in file I/O such as the
        # Parquet reader.
        column_name_reference(string) except +

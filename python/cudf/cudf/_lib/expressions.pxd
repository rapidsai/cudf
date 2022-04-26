# Copyright (c) 2022, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.expressions cimport (
    column_reference,
    expression,
    literal,
    operation,
)
from cudf._lib.cpp.scalar.scalar cimport numeric_scalar

ctypedef enum scalar_type_t:
    INT
    DOUBLE


ctypedef union int_or_double_scalar_ptr:
    unique_ptr[numeric_scalar[int64_t]] int_ptr
    unique_ptr[numeric_scalar[double]] double_ptr


cdef class Expression:
    cdef unique_ptr[expression] c_obj


cdef class Literal(Expression):
    cdef scalar_type_t c_scalar_type
    cdef int_or_double_scalar_ptr c_scalar


cdef class ColumnReference(Expression):
    pass


cdef class Operation(Expression):
    pass

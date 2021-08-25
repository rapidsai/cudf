# Copyright (c) 2021, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.ast cimport column_reference, expression, literal, operation
from cudf._lib.cpp.scalar.scalar cimport numeric_scalar

# Since Cython <3 doesn't support scoped enumerations but attempts to work with
# the underlying value of an enum, typedefing this to cast seems unavoidable.
ctypedef int32_t underlying_type_ast_operator


cdef class Expression:
    cdef unique_ptr[expression] c_obj


cdef class Literal(Expression):
    cdef unique_ptr[numeric_scalar[int64_t]] c_scalar


cdef class ColumnReference(Expression):
    pass


cdef class Operation(Expression):
    pass

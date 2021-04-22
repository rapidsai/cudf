# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr

from libc.stdint cimport int32_t
from cudf._lib.cpp.scalar.scalar cimport numeric_scalar
from cudf._lib.cpp.ast cimport literal, column_reference, expression, node


# Since Cython <3 doesn't support scoped enumerations but attempts to work with
# the underlying value of an enum, typedefing this to cast seems unavoidable.
ctypedef int32_t underlying_type_ast_operator


cdef class Node:
    cdef shared_ptr[node] c_obj


cdef class Literal(Node):
    cdef shared_ptr[numeric_scalar[float]] c_scalar


cdef class ColumnReference(Node):
    pass


cdef class Expression(Node):
    pass

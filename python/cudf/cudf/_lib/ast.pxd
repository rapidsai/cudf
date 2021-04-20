# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr

from libc.stdint cimport int32_t
from cudf._lib.cpp.scalar.scalar cimport numeric_scalar
from cudf._lib.cpp.ast cimport literal, column_reference, expression, node


# Since Cython <3 doesn't support scoped enumerations but attempts to work with
# the underlying value of an enum, typedefing this to cast seems unavoidable.
ctypedef int32_t underlying_type_ast_operator


cdef class Node:
    # Using a shared pointer here to standardize getting the underlying raw
    # pointer for passing to C++ across all subclasses.
    # TODO: Consider making this a weak pointer. Probably not worth the extra
    # effort though since these classes are part of a hierarchy anyway.
    cdef shared_ptr[node] c_node
    cdef node * _get_ptr(self)


cdef class Literal(Node):
    cdef shared_ptr[numeric_scalar[float]] c_scalar
    cdef shared_ptr[literal] c_obj


cdef class ColumnReference(Node):
    cdef shared_ptr[column_reference] c_obj


cdef class Expression(Node):
    cdef shared_ptr[expression] c_obj

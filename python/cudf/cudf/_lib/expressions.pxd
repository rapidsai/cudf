# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.expressions cimport (
    column_reference,
    expression,
    literal,
    operation,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport (
    numeric_scalar,
    scalar,
    string_scalar,
    timestamp_scalar,
)


cdef class Expression:
    cdef unique_ptr[expression] c_obj


cdef class Literal(Expression):
    cdef unique_ptr[scalar] c_scalar


cdef class ColumnReference(Expression):
    pass


cdef class Operation(Expression):
    pass

cdef class ColumnNameReference(Expression):
    pass

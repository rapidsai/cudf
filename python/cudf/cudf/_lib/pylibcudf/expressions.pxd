# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.expressions cimport (
    ast_operator,
    expression,
    table_reference,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar


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

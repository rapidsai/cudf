# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.expressions cimport (
    ast_operator,
    expression,
    table_reference,
)
from pylibcudf.libcudf.types cimport size_type

from .scalar cimport Scalar


cdef class Expression:
    cdef unique_ptr[expression] c_obj

cdef class Literal(Expression):
    # Hold on to input scalar so it doesn't get gc'ed
    cdef public Scalar scalar

cdef class ColumnReference(Expression):
    cdef public size_type index
    cdef public table_reference table_source

cdef class Operation(Expression):
    # Hold on to the input expressions so
    # they don't get gc'ed
    cdef public ast_operator op
    cdef public Expression right
    cdef public Expression left

cdef class ColumnNameReference(Expression):
    cdef public str name

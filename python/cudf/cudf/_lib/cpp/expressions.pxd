# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport (
    duration_scalar,
    numeric_scalar,
    timestamp_scalar,
)
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/ast/expressions.hpp" namespace "cudf::ast" nogil:
    ctypedef enum ast_operator:
        # Binary operators
        ADD "cudf::ast::ast_operator::ADD"
        SUB "cudf::ast::ast_operator::SUB"
        MUL "cudf::ast::ast_operator::MUL"
        DIV "cudf::ast::ast_operator::DIV"
        TRUE_DIV "cudf::ast::ast_operator::TRUE_DIV"
        FLOOR_DIV "cudf::ast::ast_operator::FLOOR_DIV"
        MOD "cudf::ast::ast_operator::MOD"
        PYMOD "cudf::ast::ast_operator::PYMOD"
        POW "cudf::ast::ast_operator::POW"
        EQUAL "cudf::ast::ast_operator::EQUAL"
        NULL_EQUAL "cudf::ast::ast_operator::NULL_EQUAL"
        NOT_EQUAL "cudf::ast::ast_operator::NOT_EQUAL"
        LESS "cudf::ast::ast_operator::LESS"
        GREATER "cudf::ast::ast_operator::GREATER"
        LESS_EQUAL "cudf::ast::ast_operator::LESS_EQUAL"
        GREATER_EQUAL "cudf::ast::ast_operator::GREATER_EQUAL"
        BITWISE_AND "cudf::ast::ast_operator::BITWISE_AND"
        BITWISE_OR "cudf::ast::ast_operator::BITWISE_OR"
        BITWISE_XOR "cudf::ast::ast_operator::BITWISE_XOR"
        NULL_LOGICAL_AND "cudf::ast::ast_operator::NULL_LOGICAL_AND"
        LOGICAL_AND "cudf::ast::ast_operator::LOGICAL_AND"
        NULL_LOGICAL_OR "cudf::ast::ast_operator::NULL_LOGICAL_OR"
        LOGICAL_OR "cudf::ast::ast_operator::LOGICAL_OR"
        # Unary operators
        IDENTITY "cudf::ast::ast_operator::IDENTITY"
        SIN "cudf::ast::ast_operator::SIN"
        COS "cudf::ast::ast_operator::COS"
        TAN "cudf::ast::ast_operator::TAN"
        ARCSIN "cudf::ast::ast_operator::ARCSIN"
        ARCCOS "cudf::ast::ast_operator::ARCCOS"
        ARCTAN "cudf::ast::ast_operator::ARCTAN"
        SINH "cudf::ast::ast_operator::SINH"
        COSH "cudf::ast::ast_operator::COSH"
        TANH "cudf::ast::ast_operator::TANH"
        ARCSINH "cudf::ast::ast_operator::ARCSINH"
        ARCCOSH "cudf::ast::ast_operator::ARCCOSH"
        ARCTANH "cudf::ast::ast_operator::ARCTANH"
        EXP "cudf::ast::ast_operator::EXP"
        LOG "cudf::ast::ast_operator::LOG"
        SQRT "cudf::ast::ast_operator::SQRT"
        CBRT "cudf::ast::ast_operator::CBRT"
        CEIL "cudf::ast::ast_operator::CEIL"
        FLOOR "cudf::ast::ast_operator::FLOOR"
        ABS "cudf::ast::ast_operator::ABS"
        RINT "cudf::ast::ast_operator::RINT"
        BIT_INVERT "cudf::ast::ast_operator::BIT_INVERT"
        NOT "cudf::ast::ast_operator::NOT"

    cdef cppclass expression:
        pass

    ctypedef enum table_reference:
        LEFT "cudf::ast::table_reference::LEFT"
        RIGHT "cudf::ast::table_reference::RIGHT"

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

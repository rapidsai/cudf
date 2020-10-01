# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
import numpy as np

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view, mutable_column_view
)
from cudf._lib.types import np_to_cudf_types
from cudf._lib.cpp.types cimport (
    size_type,
    data_type,
    type_id,
)
from cudf._lib.column import np_to_cudf_types, cudf_to_np_types
from cudf._lib.cpp.unary cimport (
    underlying_type_t_unary_op,
    unary_op,
)

from cudf._lib.types cimport underlying_type_t_type_id

cimport cudf._lib.cpp.unary as libcudf_unary


class UnaryOp(IntEnum):
    SIN = <underlying_type_t_unary_op> unary_op.SIN
    COS = <underlying_type_t_unary_op> unary_op.COS
    TAN = <underlying_type_t_unary_op> unary_op.TAN
    ASIN = <underlying_type_t_unary_op> unary_op.ARCSIN
    ACOS = <underlying_type_t_unary_op> unary_op.ARCCOS
    ATAN = <underlying_type_t_unary_op> unary_op.ARCTAN
    SINH = <underlying_type_t_unary_op> unary_op.SINH
    COSH = <underlying_type_t_unary_op> unary_op.COSH
    TANH = <underlying_type_t_unary_op> unary_op.TANH
    ARCSINH = <underlying_type_t_unary_op> unary_op.ARCSINH
    ARCCOSH = <underlying_type_t_unary_op> unary_op.ARCCOSH
    ARCTANH = <underlying_type_t_unary_op> unary_op.ARCTANH
    EXP = <underlying_type_t_unary_op> unary_op.EXP
    LOG = <underlying_type_t_unary_op> unary_op.LOG
    SQRT = <underlying_type_t_unary_op> unary_op.SQRT
    CBRT = <underlying_type_t_unary_op> unary_op.CBRT
    CEIL = <underlying_type_t_unary_op> unary_op.CEIL
    FLOOR = <underlying_type_t_unary_op> unary_op.FLOOR
    ABS = <underlying_type_t_unary_op> unary_op.ABS
    RINT = <underlying_type_t_unary_op> unary_op.RINT
    INVERT = <underlying_type_t_unary_op> unary_op.BIT_INVERT
    NOT = <underlying_type_t_unary_op> unary_op.NOT


def unary_operation(Column input, object op):
    cdef column_view c_input = input.view()
    cdef unary_op c_op = <unary_op>(<underlying_type_t_unary_op> op)
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            libcudf_unary.unary_operation(
                c_input,
                c_op
            )
        )

    return Column.from_unique_ptr(move(c_result))


def is_null(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_null(c_input))

    return Column.from_unique_ptr(move(c_result))


def is_valid(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_valid(c_input))

    return Column.from_unique_ptr(move(c_result))


def cast(Column input, object dtype=np.float64):
    cdef column_view c_input = input.view()
    cdef type_id tid = (
        <type_id> (
            <underlying_type_t_type_id> (
                np_to_cudf_types[np.dtype(dtype)]
            )
        )
    )
    cdef data_type c_dtype = data_type(tid)
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.cast(c_input, c_dtype))

    return Column.from_unique_ptr(move(c_result))


def is_nan(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_nan(c_input))

    return Column.from_unique_ptr(move(c_result))


def is_non_nan(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_not_nan(c_input))

    return Column.from_unique_ptr(move(c_result))

# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.api.types import is_decimal_dtype
from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import numpy as np

cimport cudf._lib.cpp.unary as libcudf_unary
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type
from cudf._lib.cpp.unary cimport unary_operator
from cudf._lib.types cimport dtype_to_data_type

from cudf._lib.cpp.unary import unary_operator as UnaryOp  # no-cython-lint


@acquire_spill_lock()
def unary_operation(Column input, object op):
    cdef column_view c_input = input.view()
    cdef unary_operator c_op = op
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            libcudf_unary.unary_operation(
                c_input,
                c_op
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def is_null(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_null(c_input))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def is_valid(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_valid(c_input))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def cast(Column input, object dtype=np.float64):
    cdef column_view c_input = input.view()
    cdef data_type c_dtype = dtype_to_data_type(dtype)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.cast(c_input, c_dtype))

    result = Column.from_unique_ptr(move(c_result))
    if is_decimal_dtype(result.dtype):
        result.dtype.precision = dtype.precision
    return result


@acquire_spill_lock()
def is_nan(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_nan(c_input))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def is_non_nan(Column input):
    cdef column_view c_input = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(libcudf_unary.is_not_nan(c_input))

    return Column.from_unique_ptr(move(c_result))

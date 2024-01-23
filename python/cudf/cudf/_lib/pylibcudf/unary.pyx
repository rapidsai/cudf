# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp cimport unary as cpp_unary
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.unary cimport unary_operator

from cudf._lib.cpp.unaryop import \
    unary_operator as UnaryOperator  # no-cython-lint

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(Column input, unary_operator op):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.unary_operation(input.view(), op))

    return Column.from_libcudf(move(result))


cpdef Column is_null(Column input):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_null(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column is_valid(Column input):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_valid(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column cast(Column input, DataType data_type):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.cast(input.view(), data_type.c_obj))

    return Column.from_libcudf(move(result))


cpdef Column is_nan(Column input):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_nan(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column is_non_nan(Column input):
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_not_nan(input.view()))

    return Column.from_libcudf(move(result))

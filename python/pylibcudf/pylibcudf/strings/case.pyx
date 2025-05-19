# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport case as cpp_case

__all__ = ["swapcase", "to_lower", "to_upper"]

cpdef Column to_lower(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_lower(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column to_upper(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_upper(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column swapcase(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.swapcase(input.view())

    return Column.from_libcudf(move(c_result))
